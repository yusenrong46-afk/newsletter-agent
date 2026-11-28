"""Hybrid semantic search over indexed briefing bullets, top stories, and subjects.

Combines SQLite FTS5 (BM25 lexical ranking) with dense vector cosine similarity
using Reciprocal Rank Fusion (RRF).

Privacy invariant: raw email bodies are never written to disk or indexed here.
Only fact-checked briefing bullets, top stories, and subscription subject/header
metadata enter the search index.

Live runs can embed with OpenAI's `text-embedding-3-small`; `--demo`, offline CLI
searches, and tests use `OfflineEmbedder`, a deterministic zero-dependency vector
stand-in based on hashed word tokens, character n-grams, and topic clusters.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import sys
from array import array
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from .briefing import publication_names
from .classify import SUBSCRIPTION_CATEGORIES
from .store import InventoryStore, list_briefing_paths, load_latest_report

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 512
EMBEDDING_BATCH_SIZE = 64

OFFLINE_EMBEDDING_MODEL = "offline-ngram-v1"
OFFLINE_EMBEDDING_DIMS = 384

RRF_K = 60
MIN_VECTOR_SIMILARITY = 0.05
DAY_MS = 86_400_000

SearchMode = Literal["hybrid", "lexical", "vector"]
DocType = Literal["briefing_bullet", "top_story", "issue_subject"]

STOP_WORDS = frozenset({
    "a", "about", "an", "and", "any", "are", "as", "at", "be", "been", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "in",
    "into", "is", "it", "its", "me", "my", "of", "on", "or", "our", "say",
    "said", "show", "some", "tell", "that", "the", "their", "them", "there",
    "these", "they", "this", "those", "to", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
})

# Lightweight concept anchors for the deterministic offline embedder.
# Live runs use OpenAI text-embedding-3-small; these clusters give `--demo` and
# offline CI meaningful conceptual recall beyond exact lexical overlap.
OFFLINE_CONCEPT_CLUSTERS: dict[str, frozenset[str]] = {
    "monetary_policy": frozenset({
        "rate", "rates", "interest", "inflation", "inflationary", "fed", "central",
        "bank", "banks", "yield", "yields", "bond", "bonds", "treasury", "monetary",
        "economy", "economic", "cpi", "deflation", "hike", "cut", "cuts",
    }),
    "personal_finance": frozenset({
        "budget", "budgets", "household", "savers", "saving", "savings", "spending",
        "debt", "mortgage", "retirement", "income", "ladders", "ladder", "portfolio",
        "equities", "stocks", "small", "caps", "jobs", "employment", "payrolls",
    }),
    "software_dev": frozenset({
        "open", "source", "opensource", "editor", "ide", "code", "coding", "rust",
        "embedded", "firmware", "microcontroller", "microcontrollers", "compiler",
        "release", "notes", "version", "export", "preview", "previews", "bug",
        "bugs", "update", "software", "developer", "tools", "programming",
    }),
    "ai_hardware": frozenset({
        "gpu", "gpus", "chip", "chips", "silicon", "ai", "model", "models", "llm",
        "inference", "training", "hobby", "local", "compute", "hardware", "cuda",
        "neural", "machine", "learning",
    }),
    "privacy_web": frozenset({
        "browser", "browsers", "privacy", "tracking", "cookies", "api", "web",
        "standards", "vendors", "security", "encryption", "consent", "surveillance",
    }),
    "ecology_wildlife": frozenset({
        "wetland", "wetlands", "river", "rivers", "otter", "otters", "wildlife",
        "habitat", "restoration", "ecosystem", "species", "birds", "migrating",
        "migration", "nature", "trail", "conservation", "volunteers", "ecology",
    }),
    "climate_agriculture": frozenset({
        "climate", "heat", "urban", "warming", "drought", "dry", "spring", "farm",
        "farms", "farming", "agriculture", "crops", "weather", "temperature",
        "rainfall", "citizen", "scientists", "map",
    }),
    "space_astronomy": frozenset({
        "space", "spacecraft", "orbit", "orbital", "lander", "lunar", "moon",
        "water", "ice", "launch", "rocket", "rockets", "telescope", "deep",
        "field", "galaxy", "solar", "sail", "sails", "propulsion", "mission",
        "astronomy", "satellite", "satellites",
    }),
    "retail_commerce": frozenset({
        "sale", "sales", "discount", "off", "deal", "deals", "shipping", "coffee",
        "beans", "roast", "gear", "shells", "rain", "runners", "packs", "shop",
        "arrivals", "weekend", "promo", "promotion", "order", "receipt",
    }),
}

_WORD_TO_CONCEPTS: dict[str, tuple[str, ...]] = {}
for _concept, _words in OFFLINE_CONCEPT_CLUSTERS.items():
    for _word in _words:
        _WORD_TO_CONCEPTS[_word] = (*_WORD_TO_CONCEPTS.get(_word, ()), _concept)


# --- Data structures ----------------------------------------------------------

@dataclass(frozen=True)
class SearchDocument:
    """One privacy-safe searchable unit (never a raw email body)."""

    doc_id: str
    doc_type: DocType
    message_id: str
    publication: str
    subject: str
    text: str
    received: str
    internal_date: int
    briefing_date: str = ""

    @property
    def embedding_text(self) -> str:
        parts = [self.publication, self.subject, self.text]
        return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class SearchHit:
    """One ranked result returned by HybridSearchIndex.search()."""

    doc_id: str
    doc_type: str
    message_id: str
    publication: str
    subject: str
    text: str
    received: str
    internal_date: int
    briefing_date: str
    score: float
    rrf_score: float
    lexical_rank: int | None
    vector_rank: int | None
    vector_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "message_id": self.message_id,
            "publication": self.publication,
            "subject": self.subject,
            "text": self.text,
            "received": self.received,
            "internal_date": self.internal_date,
            "briefing_date": self.briefing_date,
            "score": round(self.score, 6),
            "rrf_score": round(self.rrf_score, 6),
            "lexical_rank": self.lexical_rank,
            "vector_rank": self.vector_rank,
            "vector_score": round(self.vector_score, 4) if self.vector_score is not None else None,
        }


# --- Vector helpers -----------------------------------------------------------

def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return [0.0] * len(vector)
    return [value / norm for value in vector]


def pack_vector(vector: list[float]) -> bytes:
    buf = array("f", vector)
    if sys.byteorder != "little":
        buf.byteswap()
    return buf.tobytes()


def unpack_vector(blob: bytes) -> list[float]:
    buf = array("f")
    buf.frombytes(blob)
    if sys.byteorder != "little":
        buf.byteswap()
    return list(buf)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Dot product for L2-normalized vectors of equal dimension."""
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))


# --- Embedders ----------------------------------------------------------------

class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OfflineEmbedder:
    """Zero-dependency deterministic embedder for `--demo`, offline CLI, and tests.

    Projects word unigrams, character 3/4-grams, and lightweight concept anchors
    into a fixed L2-normalized vector via signed feature hashing (blake2b).
    """

    def __init__(self, dimensions: int = OFFLINE_EMBEDDING_DIMS):
        self.model_name = OFFLINE_EMBEDDING_MODEL
        self.dimensions = dimensions

    def _hash_feature(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % self.dimensions
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        return index, sign

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) >= 2]
        if not tokens:
            return vec

        content = [tok for tok in tokens if tok not in STOP_WORDS] or tokens
        concepts_seen: dict[str, int] = {}

        for token in content:
            idx, sign = self._hash_feature(f"w:{token}")
            vec[idx] += sign * 1.4

            # Character n-grams capture morphology, compounds, and minor typos.
            padded = f"^{token}$"
            for n in (3, 4):
                if len(padded) >= n:
                    for start in range(len(padded) - n + 1):
                        gram = padded[start:start + n]
                        g_idx, g_sign = self._hash_feature(f"c{n}:{gram}")
                        vec[g_idx] += g_sign * 0.45

            for concept in _WORD_TO_CONCEPTS.get(token, ()):
                concepts_seen[concept] = concepts_seen.get(concept, 0) + 1

        # Add concept anchors with diminishing weight per repeat so domain themes cluster cleanly.
        for concept, count in concepts_seen.items():
            weight = 2.6 * (1.0 + math.log1p(count))
            for slot in range(3):
                c_idx, c_sign = self._hash_feature(f"concept:{concept}:{slot}")
                vec[c_idx] += c_sign * weight

        return normalize(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]


class OpenAIEmbedder:
    """OpenAI `text-embedding-3-small` embedder with batching and token tracking."""

    def __init__(self, client: Any, model_name: str = EMBEDDING_MODEL, dimensions: int = EMBEDDING_DIMS):
        self._client = client
        self.model_name = model_name
        self.dimensions = dimensions
        self.usage: dict[str, int] = {"calls": 0, "total_tokens": 0}

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = [text.strip() or " " for text in texts[start:start + EMBEDDING_BATCH_SIZE]]
            response = self._client.embeddings.create(
                model=self.model_name,
                input=batch,
                dimensions=self.dimensions,
            )
            self.usage["calls"] += 1
            actual = getattr(response, "usage", None)
            if actual is not None:
                self.usage["total_tokens"] += int(getattr(actual, "total_tokens", 0) or 0)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(normalize(list(item.embedding)) for item in ordered)
        return vectors


# --- Document extraction (privacy-safe) ---------------------------------------

def extract_briefing_documents(briefing: dict) -> list[SearchDocument]:
    """Extract fact-checked bullets and top stories from a saved briefing dict."""
    docs: list[SearchDocument] = []
    briefing_date = str(briefing.get("generated_at", ""))[:10]
    window_end_ms = int(briefing.get("window_end_ms", 0) or 0)

    for section in briefing.get("sections", []):
        publication = str(section.get("publication", ""))
        for issue in section.get("issues", []):
            message_id = str(issue.get("message_id", ""))
            subject = str(issue.get("subject", ""))
            received = str(issue.get("received", ""))
            internal_date = int(issue.get("internal_date", 0) or window_end_ms)
            for idx, bullet in enumerate(issue.get("bullets", [])):
                text = str(bullet).strip()
                if not text:
                    continue
                docs.append(SearchDocument(
                    doc_id=f"bullet:{message_id}:{idx}",
                    doc_type="briefing_bullet",
                    message_id=message_id,
                    publication=publication,
                    subject=subject,
                    text=text,
                    received=received,
                    internal_date=internal_date,
                    briefing_date=briefing_date,
                ))

    for idx, story in enumerate(briefing.get("top_stories", [])):
        headline = str(story.get("headline", "")).strip()
        if not headline:
            continue
        pubs = [str(p) for p in story.get("publications", []) if p]
        bullets = story.get("bullets", [])
        bullet_texts = [str(b.get("text", "")).strip() for b in bullets if b.get("text")]
        msg_ids = sorted({str(b["source_message_id"]) for b in bullets if b.get("source_message_id")})
        combined = headline
        if bullet_texts:
            combined = f"{headline} — {' '.join(bullet_texts)}"
        docs.append(SearchDocument(
            doc_id=f"story:{briefing_date}:{idx}",
            doc_type="top_story",
            message_id=",".join(msg_ids),
            publication=", ".join(pubs),
            subject=headline,
            text=combined,
            received=briefing_date,
            internal_date=window_end_ms,
            briefing_date=briefing_date,
        ))
    return docs


def extract_inventory_documents(report: dict) -> list[SearchDocument]:
    """Extract subscription issue subjects/headers from an inventory report (no email bodies)."""
    names = publication_names(report)
    docs: list[SearchDocument] = []
    for message in report.get("messages", []):
        if message.get("classification") not in SUBSCRIPTION_CATEGORIES:
            continue
        message_id = str(message.get("message_id", ""))
        subject = str(message.get("subject", "")).strip()
        if not message_id or not subject:
            continue
        publication = str(
            names.get(message.get("group_key", ""))
            or message.get("publication_name")
            or message.get("from", "")
        )
        docs.append(SearchDocument(
            doc_id=f"subject:{message_id}",
            doc_type="issue_subject",
            message_id=message_id,
            publication=publication,
            subject=subject,
            text=f"{publication}: {subject}" if publication else subject,
            received=str(message.get("received", "")),
            internal_date=int(message.get("internal_date", 0) or 0),
            briefing_date="",
        ))
    return docs


# --- FTS5 query sanitization --------------------------------------------------

def sanitize_fts_query(query: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression.

    Every token is double-quoted so punctuation, hyphens, or FTS5 operators
    (AND/OR/NOT/NEAR) can never cause a syntax error or query injection.
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    if not tokens:
        return ""
    meaningful = [tok for tok in tokens if tok not in STOP_WORDS and len(tok) >= 2]
    chosen = meaningful or tokens
    deduped = list(dict.fromkeys(chosen))
    return " OR ".join(f'"{token}"' for token in deduped)


# --- Hybrid search index ------------------------------------------------------

class HybridSearchIndex:
    """SQLite FTS5 + dense vector index fused with Reciprocal Rank Fusion."""

    def __init__(self, store_or_connection: InventoryStore | sqlite3.Connection):
        if isinstance(store_or_connection, InventoryStore):
            self._conn = store_or_connection.connection
        else:
            self._conn = store_or_connection

    def count_documents(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM search_docs").fetchone()[0])

    def stored_embedding_models(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT model, COUNT(*) FROM search_embeddings GROUP BY model").fetchall()
        return {str(model): int(count) for model, count in rows}

    def upsert_documents(self, docs: list[SearchDocument], embedder: Embedder | None = None) -> int:
        """Insert or update documents in search_docs, search_fts, and optionally search_embeddings."""
        if not docs:
            return 0
        vectors = embedder.embed([doc.embedding_text for doc in docs]) if embedder is not None else None
        for idx, doc in enumerate(docs):
            self._conn.execute(
                """
                INSERT OR REPLACE INTO search_docs
                    (doc_id, doc_type, message_id, publication, subject, text, received, internal_date, briefing_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.doc_id, doc.doc_type, doc.message_id, doc.publication, doc.subject,
                    doc.text, doc.received, doc.internal_date, doc.briefing_date,
                ),
            )
            self._conn.execute("DELETE FROM search_fts WHERE doc_id = ?", (doc.doc_id,))
            self._conn.execute(
                "INSERT INTO search_fts (doc_id, publication, subject, text) VALUES (?, ?, ?, ?)",
                (doc.doc_id, doc.publication, doc.subject, doc.text),
            )
            if vectors is not None and embedder is not None:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO search_embeddings (doc_id, model, dims, vector)
                    VALUES (?, ?, ?, ?)
                    """,
                    (doc.doc_id, embedder.model_name, embedder.dimensions, pack_vector(vectors[idx])),
                )
        self._conn.commit()
        return len(docs)

    def ensure_embeddings(self, embedder: Embedder) -> int:
        """Embed any indexed documents that do not yet have a vector for `embedder.model_name`."""
        rows = self._conn.execute(
            """
            SELECT d.doc_id, d.publication, d.subject, d.text
            FROM search_docs d
            LEFT JOIN search_embeddings e
                ON e.doc_id = d.doc_id AND e.model = ?
            WHERE e.doc_id IS NULL
            """,
            (embedder.model_name,),
        ).fetchall()
        if not rows:
            return 0
        texts = ["\n".join(part for part in (pub, subj, text) if part) for _, pub, subj, text in rows]
        vectors = embedder.embed(texts)
        self._conn.executemany(
            "INSERT OR REPLACE INTO search_embeddings (doc_id, model, dims, vector) VALUES (?, ?, ?, ?)",
            [
                (doc_id, embedder.model_name, embedder.dimensions, pack_vector(vec))
                for (doc_id, _, _, _), vec in zip(rows, vectors, strict=True)
            ],
        )
        self._conn.commit()
        return len(rows)

    def index_briefing(self, briefing: dict, embedder: Embedder | None = None) -> int:
        docs = extract_briefing_documents(briefing)
        return self.upsert_documents(docs, embedder=embedder or OfflineEmbedder())

    def index_inventory(self, report: dict, embedder: Embedder | None = None) -> int:
        docs = extract_inventory_documents(report)
        return self.upsert_documents(docs, embedder=embedder or OfflineEmbedder())

    def _filter_clauses(
        self,
        *,
        alias: str,
        publication: str | None,
        since_ms: int | None,
        doc_types: tuple[str, ...] | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if publication:
            clauses.append(f"LOWER({alias}.publication) LIKE ?")
            params.append(f"%{publication.strip().lower()}%")
        if since_ms is not None:
            clauses.append(f"{alias}.internal_date >= ?")
            params.append(since_ms)
        if doc_types:
            placeholders = ",".join("?" for _ in doc_types)
            clauses.append(f"{alias}.doc_type IN ({placeholders})")
            params.extend(doc_types)
        return (" AND ".join(clauses), params)

    def _lexical_candidates(
        self,
        query: str,
        *,
        publication: str | None,
        since_ms: int | None,
        doc_types: tuple[str, ...] | None,
        pool_size: int,
    ) -> tuple[dict[str, int], dict[str, tuple[Any, ...]]]:
        fts_query = sanitize_fts_query(query)
        if not fts_query:
            return {}, {}
        where_extra, extra_params = self._filter_clauses(
            alias="d", publication=publication, since_ms=since_ms, doc_types=doc_types,
        )
        where_sql = f"search_fts MATCH ?{f' AND {where_extra}' if where_extra else ''}"
        rows = self._conn.execute(
            f"""
            SELECT d.doc_id, d.doc_type, d.message_id, d.publication, d.subject, d.text,
                   d.received, d.internal_date, d.briefing_date
            FROM search_fts
            JOIN search_docs d ON d.doc_id = search_fts.doc_id
            WHERE {where_sql}
            ORDER BY bm25(search_fts, 2.0, 1.5, 1.0) ASC, d.internal_date DESC
            LIMIT ?
            """,
            [fts_query, *extra_params, pool_size],
        ).fetchall()
        ranks: dict[str, int] = {}
        docs_by_id: dict[str, tuple[Any, ...]] = {}
        for rank, row in enumerate(rows, start=1):
            doc_id = str(row[0])
            ranks[doc_id] = rank
            docs_by_id[doc_id] = row
        return ranks, docs_by_id

    def _vector_candidates(
        self,
        query: str,
        *,
        embedder: Embedder,
        publication: str | None,
        since_ms: int | None,
        doc_types: tuple[str, ...] | None,
        pool_size: int,
    ) -> tuple[dict[str, int], dict[str, float], dict[str, tuple[Any, ...]]]:
        if isinstance(embedder, OfflineEmbedder):
            self.ensure_embeddings(embedder)
        query_vec = embedder.embed([query])[0]
        where_extra, extra_params = self._filter_clauses(
            alias="d", publication=publication, since_ms=since_ms, doc_types=doc_types,
        )
        where_sql = f"e.model = ?{f' AND {where_extra}' if where_extra else ''}"
        rows = self._conn.execute(
            f"""
            SELECT d.doc_id, d.doc_type, d.message_id, d.publication, d.subject, d.text,
                   d.received, d.internal_date, d.briefing_date, e.vector
            FROM search_embeddings e
            JOIN search_docs d ON d.doc_id = e.doc_id
            WHERE {where_sql}
            """,
            [embedder.model_name, *extra_params],
        ).fetchall()
        scored: list[tuple[float, tuple[Any, ...]]] = []
        for row in rows:
            sim = cosine_similarity(query_vec, unpack_vector(row[9]))
            if sim >= MIN_VECTOR_SIMILARITY:
                scored.append((sim, row[:9]))
        scored.sort(key=lambda item: (-item[0], -int(item[1][7]), str(item[1][0])))
        ranks: dict[str, int] = {}
        scores: dict[str, float] = {}
        docs_by_id: dict[str, tuple[Any, ...]] = {}
        for rank, (sim, doc_row) in enumerate(scored[:pool_size], start=1):
            doc_id = str(doc_row[0])
            ranks[doc_id] = rank
            scores[doc_id] = sim
            docs_by_id[doc_id] = doc_row
        return ranks, scores, docs_by_id

    def search(
        self,
        query: str,
        *,
        mode: SearchMode = "hybrid",
        embedder: Embedder | None = None,
        publication: str | None = None,
        days: int | None = None,
        doc_types: tuple[str, ...] | None = None,
        limit: int = 8,
        now_ms: int | None = None,
    ) -> list[SearchHit]:
        """Search indexed documents using lexical BM25, dense vectors, or hybrid RRF."""
        cleaned = query.strip()
        if not cleaned or limit < 1:
            return []
        if mode not in ("hybrid", "lexical", "vector"):
            raise ValueError(f"Unknown search mode: {mode!r}. Choose hybrid, lexical, or vector.")

        ref_ms = now_ms if now_ms is not None else int(datetime.now(UTC).timestamp() * 1000)
        since_ms = ref_ms - days * DAY_MS if days is not None else None
        pool_size = max(limit * 5, 40)

        lex_ranks: dict[str, int] = {}
        vec_ranks: dict[str, int] = {}
        vec_scores: dict[str, float] = {}
        docs_by_id: dict[str, tuple[Any, ...]] = {}

        if mode in ("hybrid", "lexical"):
            lex_ranks, lex_docs = self._lexical_candidates(
                cleaned, publication=publication, since_ms=since_ms, doc_types=doc_types, pool_size=pool_size,
            )
            docs_by_id.update(lex_docs)

        if mode in ("hybrid", "vector"):
            active_embedder = embedder or OfflineEmbedder()
            vec_ranks, vec_scores, vec_docs = self._vector_candidates(
                cleaned, embedder=active_embedder, publication=publication,
                since_ms=since_ms, doc_types=doc_types, pool_size=pool_size,
            )
            docs_by_id.update(vec_docs)

        hits: list[SearchHit] = []
        for doc_id, row in docs_by_id.items():
            l_rank = lex_ranks.get(doc_id)
            v_rank = vec_ranks.get(doc_id)
            v_score = vec_scores.get(doc_id)
            rrf = (1.0 / (RRF_K + l_rank) if l_rank is not None else 0.0) + (
                1.0 / (RRF_K + v_rank) if v_rank is not None else 0.0
            )
            if mode == "lexical":
                primary = 1.0 / (RRF_K + l_rank) if l_rank is not None else 0.0
            elif mode == "vector":
                primary = v_score if v_score is not None else 0.0
            else:
                primary = rrf

            hits.append(SearchHit(
                doc_id=doc_id,
                doc_type=str(row[1]),
                message_id=str(row[2]),
                publication=str(row[3]),
                subject=str(row[4]),
                text=str(row[5]),
                received=str(row[6]),
                internal_date=int(row[7]),
                briefing_date=str(row[8]),
                score=primary,
                rrf_score=rrf,
                lexical_rank=l_rank,
                vector_rank=v_rank,
                vector_score=v_score,
            ))

        # Slight deterministic tie-break preferring richer briefing bullets over raw subjects, then newer docs.
        type_priority = {"briefing_bullet": 2, "top_story": 1, "issue_subject": 0}
        hits.sort(
            key=lambda hit: (
                -hit.score,
                -type_priority.get(hit.doc_type, 0),
                -hit.internal_date,
                hit.doc_id,
            )
        )
        return hits[:limit]


def sync_saved_artifacts_to_index(store: InventoryStore, embedder: Embedder | None = None) -> int:
    """Populate the search index from any saved latest report and dated briefings if needed."""
    index = HybridSearchIndex(store)
    active = embedder or OfflineEmbedder()
    indexed = 0
    report = load_latest_report()
    if report:
        indexed += index.index_inventory(report, embedder=active)
    for path in list_briefing_paths():
        try:
            briefing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        indexed += index.index_briefing(briefing, embedder=active)
    return indexed


def run_search_command(
    query: str,
    publication: str | None = None,
    days: int | None = None,
    mode: SearchMode = "hybrid",
    limit: int = 8,
    as_json: bool = False,
) -> int:
    """CLI handler for `newsletter-agent search`."""
    store = InventoryStore()
    try:
        index = HybridSearchIndex(store)
        if index.count_documents() == 0:
            sync_saved_artifacts_to_index(store)
        if index.count_documents() == 0:
            print("No indexed briefings or inventory yet. Run `newsletter-agent inventory`, `brief`, or `demo`.")
            return 1
        hits = index.search(query, mode=mode, publication=publication, days=days, limit=limit)
    finally:
        store.close()

    if as_json:
        print(json.dumps({"query": query, "mode": mode, "results": [h.to_dict() for h in hits]},
                         ensure_ascii=False, indent=2))
        return 0

    if not hits:
        print(f"No matches for {query!r} (mode={mode}).")
        return 0

    print(f"Search results for {query!r} (mode={mode}, {len(hits)} shown):")
    for idx, hit in enumerate(hits, start=1):
        meta = [hit.publication or "Unknown", hit.doc_type.replace("_", " ")]
        if hit.received:
            meta.append(hit.received)
        scores = [f"rrf={hit.rrf_score:.4f}"]
        if hit.lexical_rank is not None:
            scores.append(f"bm25=#{hit.lexical_rank}")
        if hit.vector_score is not None:
            scores.append(f"cos={hit.vector_score:.3f}")
        print(f"\n{idx}. {hit.text}")
        print(f"   {' · '.join(meta)}  [{', '.join(scores)}]")
        if hit.subject and hit.subject not in hit.text:
            print(f"   Subject: {hit.subject}")
    return 0
