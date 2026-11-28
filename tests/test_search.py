"""Hybrid semantic search: SQLite FTS5 + dense vectors + Reciprocal Rank Fusion."""

import json
import math
import sqlite3
from datetime import UTC, datetime

import httpx
import openai
import pytest

from newsletter_agent import cli
from newsletter_agent.search import (
    DAY_MS,
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    HybridSearchIndex,
    OfflineEmbedder,
    OpenAIEmbedder,
    SearchDocument,
    cosine_similarity,
    extract_briefing_documents,
    extract_inventory_documents,
    pack_vector,
    sanitize_fts_query,
    unpack_vector,
)
from newsletter_agent.store import DATABASE_NAME, InventoryStore

NOW_MS = int(datetime(2026, 6, 1, 12, 0, tzinfo=UTC).timestamp() * 1000)


def sample_docs() -> list[SearchDocument]:
    return [
        SearchDocument(
            doc_id="bullet:m1:0",
            doc_type="briefing_bullet",
            message_id="m1",
            publication="Field Notes Daily",
            subject="Field Notes Daily: River otters return",
            text="Volunteers counted river otters returning across three rivers after cleanup.",
            received="2026-06-01 10:00 UTC",
            internal_date=NOW_MS - DAY_MS,
            briefing_date="2026-06-01",
        ),
        SearchDocument(
            doc_id="bullet:m2:0",
            doc_type="briefing_bullet",
            message_id="m2",
            publication="The Quiet Ledger",
            subject="The Quiet Ledger: Rates hold steady as inflation cools",
            text="Rates hold steady as inflation cools, easing pressure on mortgages.",
            received="2026-05-30 08:00 UTC",
            internal_date=NOW_MS - 3 * DAY_MS,
            briefing_date="2026-05-30",
        ),
        SearchDocument(
            doc_id="bullet:m3:0",
            doc_type="briefing_bullet",
            message_id="m3",
            publication="Byte Garden",
            subject="Byte Garden #120: Open-source editor 1.0",
            text="A new open-source editor reaches 1.0 with plugin sandboxing.",
            received="2026-05-01 09:00 UTC",
            internal_date=NOW_MS - 31 * DAY_MS,
            briefing_date="2026-05-01",
        ),
    ]


def test_vector_pack_unpack_roundtrip_and_cosine():
    raw = [0.6, 0.8, 0.0, -0.5]
    blob = pack_vector(raw)
    restored = unpack_vector(blob)
    assert len(restored) == len(raw)
    for expected, actual in zip(raw, restored, strict=True):
        assert math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)


@pytest.mark.parametrize("nasty", [
    'what\'s "new" in C++ (4.0)?',
    "AND OR NOT NEAR/3 -hyphen:colon*star",
    '"""unbalanced quotes',
    "   ",
    "!!! ??? ---",
])
def test_fts_query_sanitization_never_crashes_sqlite(tmp_path, nasty):
    store = InventoryStore(tmp_path / "inventory")
    try:
        index = HybridSearchIndex(store)
        index.upsert_documents(sample_docs(), embedder=OfflineEmbedder())
        hits = index.search(nasty, mode="hybrid", now_ms=NOW_MS)
        assert isinstance(hits, list)
    finally:
        store.close()


def test_sanitize_fts_query_quotes_tokens_and_drops_stop_words():
    assert sanitize_fts_query("What did Byte Garden say about Rust?") == '"byte" OR "garden" OR "rust"'
    assert sanitize_fts_query("AND OR NOT") == '"not"'
    assert sanitize_fts_query("!!!") == ""


def test_upsert_is_idempotent_and_does_not_duplicate_fts_rows(tmp_path):
    store = InventoryStore(tmp_path / "inventory")
    try:
        index = HybridSearchIndex(store)
        docs = sample_docs()
        index.upsert_documents(docs, embedder=OfflineEmbedder())
        index.upsert_documents(docs, embedder=OfflineEmbedder())
        assert index.count_documents() == len(docs)
        fts_count = store.connection.execute("SELECT COUNT(*) FROM search_fts").fetchone()[0]
        assert fts_count == len(docs)
    finally:
        store.close()


def test_hybrid_search_supports_modes_and_filters(tmp_path):
    store = InventoryStore(tmp_path / "inventory")
    try:
        index = HybridSearchIndex(store)
        index.upsert_documents(sample_docs(), embedder=OfflineEmbedder())

        # Lexical exact match
        lex = index.search("river otters", mode="lexical", now_ms=NOW_MS)
        assert lex[0].doc_id == "bullet:m1:0"
        assert lex[0].lexical_rank == 1 and lex[0].vector_rank is None

        # Semantic query with zero exact keyword overlap
        lex_zero = index.search("central bank monetary policy cpi", mode="lexical", now_ms=NOW_MS)
        assert lex_zero == []
        hybrid = index.search("central bank monetary policy cpi", mode="hybrid", now_ms=NOW_MS)
        assert hybrid and hybrid[0].doc_id == "bullet:m2:0"
        assert hybrid[0].vector_score is not None and hybrid[0].vector_score > 0.1

        # Publication filter
        filtered_pub = index.search("editor", publication="Field Notes", mode="hybrid", now_ms=NOW_MS)
        assert all("Field Notes" in hit.publication for hit in filtered_pub)

        # Recency filter (m3 is 31 days old)
        recent = index.search("open-source editor", days=14, mode="hybrid", now_ms=NOW_MS)
        assert all(hit.doc_id != "bullet:m3:0" for hit in recent)
        older_ok = index.search("open-source editor", days=60, mode="hybrid", now_ms=NOW_MS)
        assert any(hit.doc_id == "bullet:m3:0" for hit in older_ok)
    finally:
        store.close()


def test_openai_embedder_batches_and_normalizes_over_real_sdk():
    requests = []

    def fake_api(request):
        body = json.loads(request.content)
        requests.append(body)
        data = [
            {"object": "embedding", "index": idx, "embedding": [3.0, 4.0] + [0.0] * (body["dimensions"] - 2)}
            for idx in range(len(body["input"]))
        ]
        return httpx.Response(200, json={
            "object": "list",
            "data": data,
            "model": body["model"],
            "usage": {"prompt_tokens": 12, "total_tokens": 12},
        })

    client = openai.OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(fake_api)))
    embedder = OpenAIEmbedder(client, model_name=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMS)
    vectors = embedder.embed(["hello world", "second doc"])
    assert len(vectors) == 2 and len(vectors[0]) == EMBEDDING_DIMS
    assert math.isclose(vectors[0][0], 0.6, rel_tol=1e-5)
    assert math.isclose(vectors[0][1], 0.8, rel_tol=1e-5)
    assert embedder.usage == {"calls": 1, "total_tokens": 12}
    assert requests[0]["model"] == EMBEDDING_MODEL and requests[0]["dimensions"] == EMBEDDING_DIMS


def test_schema_v1_migrates_cleanly_to_v2_without_losing_data(tmp_path):
    directory = tmp_path / "inventory"
    directory.mkdir()
    db_path = directory / DATABASE_NAME
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE metadata (id TEXT PRIMARY KEY, internal_date INTEGER NOT NULL, record TEXT NOT NULL);
        CREATE TABLE classifications (key TEXT PRIMARY KEY, message_id TEXT NOT NULL, result TEXT NOT NULL);
        CREATE TABLE kv (name TEXT PRIMARY KEY, value TEXT NOT NULL);
        PRAGMA user_version = 1;
    """)
    conn.execute("INSERT INTO metadata VALUES ('m1', 1000, '{\"id\": \"m1\", \"internal_date\": 1000}')")
    conn.commit()
    conn.close()

    store = InventoryStore(directory)
    try:
        assert "m1" in store.metadata
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == 2
        index = HybridSearchIndex(store)
        assert index.upsert_documents(sample_docs()[:1], embedder=OfflineEmbedder()) == 1
        assert index.count_documents() == 1
    finally:
        store.close()


def test_store_prune_removes_expired_search_documents(tmp_path):
    store = InventoryStore(tmp_path / "inventory")
    try:
        index = HybridSearchIndex(store)
        old = SearchDocument(
            doc_id="bullet:old:0", doc_type="briefing_bullet", message_id="old",
            publication="Old News", subject="Ancient", text="Ancient archived bullet",
            received="2024-01-01", internal_date=NOW_MS - 400 * DAY_MS, briefing_date="2024-01-01",
        )
        fresh = sample_docs()[0]
        index.upsert_documents([old, fresh], embedder=OfflineEmbedder())
        assert index.count_documents() == 2

        store.prune(NOW_MS, retention_days=180)
        assert index.count_documents() == 1
        assert index.search("Ancient archived", mode="hybrid", now_ms=NOW_MS) == []
        assert index.search("river otters", mode="hybrid", now_ms=NOW_MS)[0].doc_id == fresh.doc_id
    finally:
        store.close()


def test_extractors_only_index_privacy_safe_fields():
    report = {
        "publications_observed": [{"group_key": "list:a", "name": "Byte Garden"}],
        "publications_provisional": [],
        "messages": [
            {
                "message_id": "m1", "group_key": "list:a", "classification": "editorial_newsletter",
                "publication_name": "Byte Garden", "subject": "Issue #1", "received": "2026-06-01",
                "internal_date": NOW_MS, "body": "SECRET BODY MUST NOT BE INDEXED",
            },
            {
                "message_id": "p1", "group_key": "sender:friend", "classification": "personal_or_other",
                "publication_name": None, "subject": "Private dinner", "received": "2026-06-01",
                "internal_date": NOW_MS,
            },
        ],
    }
    docs = extract_inventory_documents(report)
    assert len(docs) == 1
    assert docs[0].message_id == "m1"
    assert "SECRET BODY" not in docs[0].text

    briefing = {
        "generated_at": "2026-06-01T08:00:00+00:00",
        "window_end_ms": NOW_MS,
        "sections": [
            {
                "publication": "Byte Garden",
                "issues": [{"message_id": "m1", "subject": "Issue #1", "received": "2026-06-01",
                            "internal_date": NOW_MS, "bullets": ["Fact-checked summary bullet."]}],
            }
        ],
        "top_stories": [
            {
                "headline": "Big open-source release",
                "publications": ["Byte Garden"],
                "bullets": [{"source_message_id": "m1", "text": "Fact-checked summary bullet."}],
            }
        ],
    }
    b_docs = extract_briefing_documents(briefing)
    assert {d.doc_type for d in b_docs} == {"briefing_bullet", "top_story"}


def test_cli_search_command_json_and_empty_index(tmp_path, capsys):
    # Empty home returns exit code 1 with guidance
    assert cli.main(["search", "otters"]) == 1
    assert "No indexed briefings" in capsys.readouterr().out

    # Index sample docs and query via CLI --json
    store = InventoryStore()
    try:
        HybridSearchIndex(store).upsert_documents(sample_docs(), embedder=OfflineEmbedder())
    finally:
        store.close()

    assert cli.main(["search", "river otters", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "hybrid"
    assert payload["results"][0]["doc_id"] == "bullet:m1:0"

    assert cli.main(["search", "river otters", "--days", "0"]) == 1
    assert "--days must be at least 1" in capsys.readouterr().err
