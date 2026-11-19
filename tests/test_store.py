"""The SQLite store: migration from v0.1, durability, corruption, permissions, and retention.

Red tests (fail on v0.1): the JSON-cache migration and "a scan never rewrites
whole cache files". Failure-mode tests: rollback of uncommitted writes, a hard
kill mid-scan, an unreadable database, unreadable legacy JSON, file modes,
and retention that never deletes what a saved window still needs.
"""

import json
import os
import signal
import sqlite3
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from helpers import DAY_MS, NOW_MS, FakeOpenAI, fake_gmail, many_newsletters, newsletter, run_workflow
from newsletter_agent import config
from newsletter_agent.demo import make_message
from newsletter_agent.store import DATABASE_NAME, InventoryStore, database_rows

TESTS_DIR = Path(__file__).parent
POSIX = os.name == "posix"


def export_as_v01_json(directory: Path):
    """Write the caches the way v0.1 stored them, then remove the database."""
    store = InventoryStore(directory)
    (directory / "metadata_cache.json").write_text(json.dumps(dict(store.metadata.items())))
    (directory / "classification_cache.json").write_text(json.dumps(dict(store.classifications.items())))
    (directory / "checkpoint.json").write_text(json.dumps(store.load_checkpoint()))
    store.close()
    for suffix in ("", "-wal", "-shm"):
        Path(f"{directory / DATABASE_NAME}{suffix}").unlink(missing_ok=True)


# --- Red tests --------------------------------------------------------------------

def test_upgrading_from_json_caches_rereads_nothing_and_rebills_nothing(tmp_path):
    corpus = many_newsletters(30)
    first, _ = run_workflow(tmp_path, fake_gmail(corpus))
    export_as_v01_json(tmp_path / "inventory")

    google = fake_gmail(corpus)
    second, fake = run_workflow(tmp_path, google)

    assert sum(1 for call in google.get_calls if call[1] == "metadata") == 0  # No Gmail re-reads.
    assert fake.calls == []  # Nothing sent to the model again, so nothing billed again.
    assert second["counts"]["classified"] == first["counts"]["classified"] == 30
    assert second["reconciliation"]["ok"]
    for name in ("metadata_cache.json", "classification_cache.json", "checkpoint.json"):
        assert not (tmp_path / "inventory" / name).exists()
        assert (tmp_path / "inventory" / f"{name}.migrated").exists()  # Kept as a backup.


def test_a_scan_never_rewrites_whole_cache_files(tmp_path, monkeypatch):
    rewrites = []
    original = config.write_private_file

    def counting(path, text):
        rewrites.append(path.name)
        return original(path, text)

    monkeypatch.setattr(config, "write_private_file", counting)
    report, _ = run_workflow(tmp_path, fake_gmail(many_newsletters(200)), preflight=True, quiet=True)
    assert report["counts"]["discovered"] == 200
    assert rewrites == []  # v0.1 rewrote the whole metadata cache and checkpoint after every page.


# --- Failure modes ------------------------------------------------------------------

def test_uncommitted_writes_are_rolled_back(tmp_path):
    store = InventoryStore(tmp_path / "inventory")
    store.metadata["kept"] = {"id": "kept", "internal_date": NOW_MS}
    store.save_metadata()
    store.metadata["lost"] = {"id": "lost", "internal_date": NOW_MS}
    store.close()  # Like a crash before the next commit.

    reopened = InventoryStore(tmp_path / "inventory")
    assert "kept" in reopened.metadata and "lost" not in reopened.metadata


HARD_KILL_SCRIPT = textwrap.dedent("""
    import asyncio, sys
    from pathlib import Path
    sys.path.insert(0, {tests!r})
    from helpers import SlowGmail, many_newsletters, NOW
    from mcp.shared.memory import create_connected_server_and_client_session
    from newsletter_agent.gmail_client import GmailMcpClient
    from newsletter_agent.gmail_server import build_mcp_server
    from newsletter_agent.inventory import run_inventory
    from newsletter_agent.scan import InventoryOptions
    from newsletter_agent.store import InventoryStore

    google = SlowGmail(many_newsletters(400), latency=0.01, page_size=20)

    async def go():
        async with create_connected_server_and_client_session(build_mcp_server(lambda: google)) as session:
            await run_inventory(GmailMcpClient(session), InventoryOptions(preflight=True, quiet=True),
                                InventoryStore(Path(sys.argv[1])), now=NOW)

    asyncio.run(go())
""")


@pytest.mark.skipif(not POSIX, reason="SIGKILL is POSIX-only")
def test_a_hard_kill_mid_scan_leaves_a_sound_database_and_resume_completes(tmp_path):
    directory = tmp_path / "inventory"
    script = tmp_path / "scan.py"
    script.write_text(HARD_KILL_SCRIPT.format(tests=str(TESTS_DIR)))
    process = subprocess.Popen([sys.executable, str(script), str(directory)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (directory / DATABASE_NAME).exists() and database_rows(directory)["metadata"] >= 60:
                break
            time.sleep(0.05)
        else:
            pytest.fail("The scan never saved 60 records.")
        os.kill(process.pid, signal.SIGKILL)  # No cleanup, no commit, no finally blocks.
    finally:
        process.wait(timeout=10)

    connection = sqlite3.connect(directory / DATABASE_NAME)
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    saved = connection.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    connection.close()
    assert 60 <= saved < 400

    google = fake_gmail(many_newsletters(400))
    report, _ = run_workflow(tmp_path, google, preflight=True, resume=True)
    assert report["counts"]["discovered"] == 400
    assert report["reconciliation"]["ok"]
    assert sum(1 for call in google.get_calls if call[1] == "metadata") == 400 - saved  # Saved work was reused.


def test_an_unreadable_database_is_moved_aside_and_rebuilt(tmp_path, capsys):
    directory = tmp_path / "inventory"
    directory.mkdir()
    (directory / DATABASE_NAME).write_bytes(b"this is not a database" * 100)

    report, _ = run_workflow(tmp_path, fake_gmail([newsletter("a")]))
    assert report["run"]["status"] == "complete"
    assert "was unreadable and was moved" in capsys.readouterr().err
    assert len(list(directory.glob(f"{DATABASE_NAME}.corrupt-*"))) == 1


def test_unreadable_legacy_json_is_skipped_with_a_warning(tmp_path, capsys):
    directory = tmp_path / "inventory"
    directory.mkdir()
    (directory / "metadata_cache.json").write_text("{not json")

    report, _ = run_workflow(tmp_path, fake_gmail([newsletter("a")]))
    assert report["run"]["status"] == "complete"
    assert "ignoring unreadable file metadata_cache.json" in capsys.readouterr().err
    assert (directory / "metadata_cache.json.migrated").exists()


@pytest.mark.skipif(not POSIX, reason="POSIX permission bits")
def test_database_files_are_owner_only(tmp_path):
    store = InventoryStore(tmp_path / "inventory")
    store.metadata["a"] = {"id": "a", "internal_date": NOW_MS}
    store.save_metadata()  # The -wal and -shm files exist while the connection is open.
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{tmp_path / 'inventory' / DATABASE_NAME}{suffix}")
        assert path.exists(), suffix
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, suffix


def test_a_symlinked_database_is_refused(tmp_path):
    directory = tmp_path / "inventory"
    directory.mkdir()
    (tmp_path / "elsewhere.db").write_bytes(b"")
    (directory / DATABASE_NAME).symlink_to(tmp_path / "elsewhere.db")
    with pytest.raises(ValueError, match="symlink"):
        InventoryStore(directory)


# --- Retention ----------------------------------------------------------------------

def add_old_record(directory: Path, message_id: str, days_ago: int):
    store = InventoryStore(directory)
    store.metadata[message_id] = {"id": message_id, "internal_date": NOW_MS - days_ago * DAY_MS}
    store.classifications[f"{message_id}|excerpt|x"] = {"message_id": message_id, "classification": "uncertain"}
    store.save_metadata()
    store.close()


def test_a_complete_run_prunes_only_records_older_than_retention(tmp_path):
    directory = tmp_path / "inventory"
    add_old_record(directory, "ancient", days_ago=400)
    add_old_record(directory, "recent-past", days_ago=120)  # Outside the 90-day window, inside retention.

    report, _ = run_workflow(tmp_path, fake_gmail([newsletter("a")]))
    store = InventoryStore(directory)
    assert "ancient" not in store.metadata
    assert not any(key.startswith("ancient|") for key in store.classifications)
    assert "recent-past" in store.metadata
    assert "a" in store.metadata and report["reconciliation"]["ok"]


def test_retention_never_prunes_inside_a_saved_window(tmp_path):
    directory = tmp_path / "inventory"
    add_old_record(directory, "in-long-window", days_ago=300)
    # A 365-day inventory's checkpoint keeps its whole window, even past the 180-day retention.
    run_workflow(tmp_path, fake_gmail([newsletter("a")]), days=365)
    assert "in-long-window" in InventoryStore(directory).metadata


def test_a_partial_listing_never_prunes(tmp_path):
    directory = tmp_path / "inventory"
    add_old_record(directory, "ancient", days_ago=400)
    report, _ = run_workflow(tmp_path, fake_gmail(many_newsletters(5)), max_messages=2)
    assert report["run"]["status"] == "partial_message_cap"
    assert "ancient" in InventoryStore(directory).metadata


def test_a_saved_config_keeps_retention_days(tmp_path):
    settings = config.load_config()
    settings["retention_days"] = 30
    config.save_config(settings)
    assert config.load_config()["retention_days"] == 30


def test_cached_reruns_still_call_the_model_for_new_mail_only(tmp_path):
    corpus = many_newsletters(4)
    run_workflow(tmp_path, fake_gmail(corpus))
    newer = corpus + [make_message("new-1", when_ms=NOW_MS - DAY_MS, headers={"List-Id": "<weekly.example.com>"})]
    _, fake = run_workflow(tmp_path, fake_gmail(newer), fake_openai=FakeOpenAI())
    assert [item["id"] for call in fake.calls for item in json.loads(call["input"])["messages"]] == ["new-1"]
