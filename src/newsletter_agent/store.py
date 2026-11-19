"""Private storage: caches, checkpoints, and reports, all owner-only.

Caches and checkpoints live in one SQLite database, inventory/newsletter.db:

    metadata         message ID -> header record (never bodies)
    classifications  cache key  -> model label, name, confidence, rationale
    kv               checkpoint name -> saved run state

`store.metadata` and `store.classifications` behave like dicts. Writes become
durable when the matching save_*() method commits, which the scan and the
classifier do after every page and every batch, so an interruption loses at
most one page or batch. Reports stay as separate JSON/Markdown/HTML files.

Version 0.1 kept the caches in JSON files. They are imported once, then
renamed to *.migrated and kept as a backup.
"""

import fcntl
import json
import os
import sqlite3
import sys
from collections.abc import Iterator, MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from . import config

DATABASE_NAME = "newsletter.db"
SCHEMA_VERSION = 1
LEGACY_JSON_FILES = ("metadata_cache.json", "classification_cache.json", "checkpoint.json",
                     "briefing_checkpoint.json")
# A checkpoint's listing query reaches one day past its window, so keep two days of margin.
RETENTION_MARGIN_MS = 2 * 86_400_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    id TEXT PRIMARY KEY,
    internal_date INTEGER NOT NULL,
    record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS metadata_by_date ON metadata (internal_date);
CREATE TABLE IF NOT EXISTS classifications (
    key TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    result TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS classifications_by_message ON classifications (message_id);
CREATE TABLE IF NOT EXISTS kv (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SqliteCache(MutableMapping[str, dict]):
    """A dict-like view of one table. Values are JSON objects."""

    def __init__(self, connection: sqlite3.Connection, table: str, key_column: str, value_column: str,
                 extra_column: str, extra_value):
        self._connection = connection
        self._table, self._key, self._value = table, key_column, value_column
        self._extra_column, self._extra_value = extra_column, extra_value

    def __getitem__(self, key: str) -> dict:
        row = self._connection.execute(
            f"SELECT {self._value} FROM {self._table} WHERE {self._key} = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def __setitem__(self, key: str, value: dict):
        self._connection.execute(
            f"INSERT OR REPLACE INTO {self._table} ({self._key}, {self._extra_column}, {self._value}) "
            "VALUES (?, ?, ?)",
            (key, self._extra_value(key, value), json.dumps(value, ensure_ascii=False, separators=(",", ":"))),
        )

    def __delitem__(self, key: str):
        cursor = self._connection.execute(f"DELETE FROM {self._table} WHERE {self._key} = ?", (key,))
        if cursor.rowcount == 0:
            raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return self._connection.execute(
            f"SELECT 1 FROM {self._table} WHERE {self._key} = ?", (key,)).fetchone() is not None

    def __iter__(self) -> Iterator[str]:
        keys = [row[0] for row in self._connection.execute(f"SELECT {self._key} FROM {self._table}")]
        return iter(keys)

    def __len__(self) -> int:
        return self._connection.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0]


def _create_private_file(path: Path):
    """Create an empty owner-only file if it does not exist yet."""
    if path.is_symlink():
        raise ValueError(f"Refusing to use a symlink for private data: {path}")
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT, 0o600))
    os.chmod(path, 0o600)


class InventoryStore:
    """Private data for one folder. Normal runs use ~/.newsletter-agent/inventory."""

    def __init__(self, directory: Path | None = None, checkpoint_name: str = "checkpoint.json"):
        self.directory = directory or config.inventory_dir()
        self.reports_directory = self.directory / "reports"
        self.checkpoint_name = checkpoint_name
        for folder in (self.directory, self.reports_directory):
            if folder.is_symlink():
                raise ValueError(f"Refusing to use a symlink for private data: {folder}")
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(folder, 0o700)
        self._lock_file: TextIO | None = None
        self.database_path = self.directory / DATABASE_NAME
        self._connection = self._open_database()
        self._migrate_legacy_json()
        self.metadata = SqliteCache(self._connection, "metadata", "id", "record", "internal_date",
                                    lambda key, record: int(record.get("internal_date", 0)))
        self.classifications = SqliteCache(self._connection, "classifications", "key", "result", "message_id",
                                           lambda key, result: result.get("message_id") or key.split("|")[0])

    # --- Opening, corruption, and migration -----------------------------------------

    def _open_database(self) -> sqlite3.Connection:
        try:
            return self._connect()
        except sqlite3.DatabaseError:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            aside = self.database_path.with_name(f"{DATABASE_NAME}.corrupt-{stamp}")
            for suffix in ("", "-wal", "-shm"):
                path = Path(f"{self.database_path}{suffix}")
                if path.exists():
                    os.replace(path, Path(f"{aside}{suffix}"))
            print(f"Warning: {DATABASE_NAME} was unreadable and was moved to {aside.name}; "
                  "caches will be rebuilt.", file=sys.stderr)
            return self._connect()

    def _connect(self) -> sqlite3.Connection:
        _create_private_file(self.database_path)
        connection = sqlite3.connect(self.database_path, timeout=30)
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(SCHEMA)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
        except sqlite3.DatabaseError:
            connection.close()
            raise
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self.database_path}{suffix}")
            if sidecar.exists():
                os.chmod(sidecar, 0o600)
        return connection

    def _is_empty(self) -> bool:
        return not any(self._connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
                       for table in ("metadata", "classifications", "kv"))

    def _migrate_legacy_json(self):
        """Import v0.1 JSON caches once, so an upgrade never re-reads Gmail or re-bills the model."""
        legacy = [self.directory / name for name in LEGACY_JSON_FILES if (self.directory / name).exists()]
        if not legacy:
            return
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            if self._is_empty():
                for path in legacy:
                    data = self._read_legacy_json(path)
                    if path.name == "metadata_cache.json":
                        self._connection.executemany(
                            "INSERT OR REPLACE INTO metadata (id, internal_date, record) VALUES (?, ?, ?)",
                            [(key, int(value.get("internal_date", 0)), json.dumps(value, ensure_ascii=False))
                             for key, value in (data or {}).items()])
                    elif path.name == "classification_cache.json":
                        self._connection.executemany(
                            "INSERT OR REPLACE INTO classifications (key, message_id, result) VALUES (?, ?, ?)",
                            [(key, value.get("message_id") or key.split("|")[0], json.dumps(value, ensure_ascii=False))
                             for key, value in (data or {}).items()])
                    elif data is not None:
                        self._connection.execute("INSERT OR REPLACE INTO kv (name, value) VALUES (?, ?)",
                                                 (path.name, json.dumps(data, ensure_ascii=False)))
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        for path in legacy:
            if path.exists():
                os.replace(path, path.with_name(path.name + ".migrated"))

    @staticmethod
    def _read_legacy_json(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"Warning: ignoring unreadable file {path.name}; it will be rebuilt.", file=sys.stderr)
            return None

    # --- Locking ---------------------------------------------------------------------

    def lock(self):
        """Stop two runs (say, a scheduled briefing and a manual inventory) sharing the caches."""
        # Kept open on purpose: the lock lasts until unlock() closes the file.
        lock_file = open(self.directory / ".lock", "w")  # noqa: SIM115
        self._lock_file = lock_file
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another newsletter-agent run is using this data. Try again when it finishes.") from None

    def unlock(self):
        if self._lock_file:
            self._lock_file.close()
            self._lock_file = None

    def close(self):
        """Commit nothing further and release the database."""
        self._connection.close()

    # --- Saving ----------------------------------------------------------------------

    def save_metadata(self):
        self._connection.commit()

    def save_classifications(self):
        self._connection.commit()

    def load_checkpoint(self) -> dict | None:
        row = self._connection.execute("SELECT value FROM kv WHERE name = ?", (self.checkpoint_name,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_checkpoint(self, state_dict: dict):
        self._connection.execute("INSERT OR REPLACE INTO kv (name, value) VALUES (?, ?)",
                                 (self.checkpoint_name, json.dumps(state_dict, ensure_ascii=False)))
        self._connection.commit()

    # --- Retention -------------------------------------------------------------------

    def retention_cutoff_ms(self, now_ms: int, retention_days: int) -> int:
        """Rows older than this may go. Never inside any saved checkpoint's window."""
        cutoff = now_ms - retention_days * 86_400_000
        for (value,) in self._connection.execute("SELECT value FROM kv"):
            try:
                window_start = int(json.loads(value).get("window_start_ms"))
            except (TypeError, ValueError, AttributeError):
                continue
            cutoff = min(cutoff, window_start - RETENTION_MARGIN_MS)
        return cutoff

    def prune(self, now_ms: int, retention_days: int) -> int:
        """Delete header records (and their labels) older than the retention window. Returns rows removed."""
        cutoff = self.retention_cutoff_ms(now_ms, retention_days)
        self._connection.execute(
            "DELETE FROM classifications WHERE message_id IN (SELECT id FROM metadata WHERE internal_date < ?)",
            (cutoff,))
        removed = self._connection.execute("DELETE FROM metadata WHERE internal_date < ?", (cutoff,)).rowcount
        self._connection.commit()
        return removed

    # --- Reports ---------------------------------------------------------------------

    def save_report(self, report: dict, text: str, html: str) -> list[Path]:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        paths = []
        for suffix, content in (
            ("json", json.dumps(report, ensure_ascii=False, indent=1)),
            ("md", text),
            ("html", html),
        ):
            path = self.reports_directory / f"inventory-{stamp}.{suffix}"
            config.write_private_file(path, content)
            paths.append(path)
        return paths


def latest_report_path(suffix: str = "json") -> Path | None:
    """The newest saved inventory report, or None if there is none yet."""
    reports = sorted((config.inventory_dir() / "reports").glob(f"inventory-*.{suffix}"))
    return reports[-1] if reports else None


def load_latest_report() -> dict | None:
    path = latest_report_path()
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def database_rows(directory: Path) -> dict[str, Any]:
    """Row counts per table, for `newsletter-agent setup` and tests."""
    connection = sqlite3.connect(directory / DATABASE_NAME)
    try:
        return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("metadata", "classifications", "kv")}
    finally:
        connection.close()
