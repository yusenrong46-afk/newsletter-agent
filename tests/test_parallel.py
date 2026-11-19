"""Parallel Gmail reads: faster, but with exactly the same answers.

Red tests (fail on v0.1, which read one message at a time): reads overlap,
capped at 8, and a slow mailbox scans in well under half the sequential time.
Failure-mode tests: a deleted message inside a parallel chunk, rate-limit
bursts (slow mode) and persistent rate limits, Ctrl-C with reads in flight,
identical reports under random timing, the exact --max-messages cap, and
one Gmail client per server thread.
"""

import asyncio
import json
import random
import threading
import time

from mcp.shared.memory import create_connected_server_and_client_session

from helpers import NOW, FakeOpenAI, SlowGmail, many_newsletters, run_workflow
from newsletter_agent import gmail_server
from newsletter_agent.demo import SyntheticGmail, http_error
from newsletter_agent.gmail_client import READ_CONCURRENCY, GmailMcpClient, gather_in_order
from newsletter_agent.gmail_server import build_mcp_server
from newsletter_agent.inventory import run_inventory
from newsletter_agent.scan import InventoryOptions
from newsletter_agent.store import InventoryStore


class FlakyGmail(SyntheticGmail):
    """Answers HTTP 429 for the first N reads of chosen IDs."""

    def __init__(self, messages, failures: dict[str, int], page_size=100):
        super().__init__(messages, page_size=page_size)
        self.failures = dict(failures)
        self._lock = threading.Lock()

    def get(self, userId, id, format, metadataHeaders=None):
        request = super().get(userId, id, format, metadataHeaders)
        inner = request.action

        def action():
            with self._lock:
                remaining = self.failures.get(id, 0)
                if remaining:
                    self.failures[id] = remaining - 1
            if remaining:
                raise http_error(429)
            return inner()

        request.action = action
        return request


def metadata_reads(google) -> int:
    return sum(1 for call in google.get_calls if call[1] == "metadata")


# --- Red tests ----------------------------------------------------------------------

def test_reads_overlap_and_never_exceed_the_limit(tmp_path):
    google = SlowGmail(many_newsletters(60), latency=0.02, page_size=100)
    report, _ = run_workflow(tmp_path, google, preflight=True)
    assert report["counts"]["discovered"] == 60
    assert 1 < google.peak_in_flight <= READ_CONCURRENCY


def test_a_slow_mailbox_scans_in_well_under_half_the_sequential_time(tmp_path):
    messages, latency = 120, 0.025
    google = SlowGmail(many_newsletters(messages), latency=latency, page_size=500)
    start = time.perf_counter()
    report, _ = run_workflow(tmp_path, google, preflight=True)
    elapsed = time.perf_counter() - start
    assert report["counts"]["discovered"] == messages
    assert elapsed < 0.4 * messages * latency, f"{elapsed:.2f}s"


def test_body_reads_for_classification_also_run_in_parallel(tmp_path):
    google = SlowGmail(many_newsletters(25), latency=0.02, page_size=100)
    report, fake = run_workflow(tmp_path, google)
    assert report["counts"]["classified"] == 25
    assert google.peak_in_flight > 1
    assert [item["id"] for item in json.loads(fake.calls[0]["input"])["messages"]] == \
        [f"m{i:03}" for i in range(25)]  # Still sent in the original order.


# --- Failure modes -------------------------------------------------------------------

def test_a_deleted_message_inside_a_parallel_chunk(tmp_path):
    google = SyntheticGmail(many_newsletters(20), page_size=100)
    google.listed_but_deleted = ["gone"]
    report, _ = run_workflow(tmp_path, google)
    assert report["counts"]["unavailable"] == 1
    assert report["counts"]["classified"] == 20
    assert report["reconciliation"]["ok"]


def test_a_burst_of_rate_limits_switches_to_slow_mode_and_completes(tmp_path, monkeypatch):
    monkeypatch.setattr(GmailMcpClient, "RATE_LIMIT_BACKOFF_SECONDS", 0)
    google = FlakyGmail(many_newsletters(20), failures={"m003": 2, "m011": 1})
    report, _ = run_workflow(tmp_path, google)
    assert report["run"]["status"] == "complete"
    assert report["counts"]["classified"] == 20
    assert any("one message at a time" in note for note in report["notes"])


def test_persistent_rate_limits_fail_safely_and_resume_without_double_counting(tmp_path, monkeypatch):
    monkeypatch.setattr(GmailMcpClient, "RATE_LIMIT_BACKOFF_SECONDS", 0)
    corpus = many_newsletters(30)
    google = FlakyGmail(corpus, failures={"m010": 99})
    report, fake = run_workflow(tmp_path, google)
    assert report["run"]["status"] == "partial_failure"
    assert report["counts"]["discovered"] == 10  # Every ID before the failure, in listing order.
    assert report["reconciliation"]["ok"]
    assert fake.calls == []

    report, _ = run_workflow(tmp_path, SyntheticGmail(corpus, page_size=100), resume=True)
    assert report["run"]["status"] == "complete"
    assert report["counts"]["discovered"] == 30
    assert report["counts"]["duplicate_listings_ignored"] == 0


def test_ctrl_c_with_reads_in_flight_saves_a_consistent_partial_report(tmp_path):
    corpus = many_newsletters(80)
    google = SlowGmail(corpus, latency=0.03, page_size=100)
    store = InventoryStore(tmp_path / "inventory")

    async def go():
        async with create_connected_server_and_client_session(build_mcp_server(lambda: google)) as session:
            task = asyncio.create_task(run_inventory(
                GmailMcpClient(session), InventoryOptions(preflight=True), store, now=NOW))
            # Interrupt in the second chunk, while several reads are still running.
            while not (len(google.get_calls) >= 16 and google.in_flight > 1):
                await asyncio.sleep(0.002)
            task.cancel()
            return await task

    report = asyncio.run(go())
    assert report["run"]["status"] == "interrupted"
    assert report["reconciliation"]["ok"]
    discovered = report["counts"]["discovered"]
    assert 0 < discovered < 80
    checker = InventoryStore(tmp_path / "inventory")
    assert all(f"m{i:03}" in checker.metadata for i in range(discovered))  # No discovered ID without a record.

    resumed, _ = run_workflow(tmp_path, SyntheticGmail(corpus, page_size=100), resume=True)
    fresh, _ = run_workflow(tmp_path / "fresh", SyntheticGmail(corpus, page_size=100))
    for key in ("discovered", "in_scope", "classified", "subscription_messages"):
        assert resumed["counts"][key] == fresh["counts"][key]


class JitterGmail(SlowGmail):
    """Random read times, so parallel reads finish in a different order every run."""

    def __init__(self, messages, seed):
        super().__init__(messages, latency=0, page_size=100)
        self.random = random.Random(seed)

    def get(self, userId, id, format, metadataHeaders=None):
        self.latency = self.random.uniform(0, 0.01)
        return super().get(userId, id, format, metadataHeaders)


def stable(report: dict) -> dict:
    report = json.loads(json.dumps(report))
    for key in ("report_generated_at", "unread_observed_at"):
        report["run"].pop(key, None)
    report.pop("report_paths", None)
    return report


def test_reports_are_identical_whatever_order_reads_finish_in(tmp_path):
    corpus = many_newsletters(50)
    first, _ = run_workflow(tmp_path / "a", JitterGmail(corpus, seed=1), fake_openai=FakeOpenAI())
    second, _ = run_workflow(tmp_path / "b", JitterGmail(corpus, seed=2), fake_openai=FakeOpenAI())
    assert stable(first) == stable(second)


def test_the_message_cap_is_exact_with_parallel_reads(tmp_path):
    google = SyntheticGmail(many_newsletters(40), page_size=100)
    report, _ = run_workflow(tmp_path, google, max_messages=3)
    assert report["counts"]["in_scope"] == 3
    assert report["counts"]["discovered"] == 3
    assert report["run"]["status"] == "partial_message_cap"
    assert metadata_reads(google) <= 3 + 16  # At most one chunk of extra reads, which are only cached.


def test_each_server_thread_builds_its_own_gmail_client(monkeypatch, tmp_path):
    builds = []
    shared = SlowGmail(many_newsletters(40), latency=0.02, page_size=100)
    monkeypatch.setattr(gmail_server, "_thread_state", threading.local())
    monkeypatch.setattr(gmail_server, "load_credentials", lambda: "credentials")

    def build(credentials):
        builds.append(threading.get_ident())
        return shared

    monkeypatch.setattr(gmail_server, "build_gmail_client", build)

    async def go():
        async with create_connected_server_and_client_session(build_mcp_server()) as session:
            return await run_inventory(GmailMcpClient(session), InventoryOptions(preflight=True),
                                       InventoryStore(tmp_path / "inventory"), now=NOW)

    report = asyncio.run(go())
    assert report["counts"]["discovered"] == 40
    assert len(builds) == len(set(builds)) > 1  # One client per thread, never shared.
    assert len(builds) <= gmail_server.SERVER_THREADS


def test_gather_in_order_keeps_order_and_raises_the_first_error():
    async def worker(value):
        await asyncio.sleep(random.uniform(0, 0.01))
        if value == 7:
            raise KeyError("seven")
        return value * 2

    assert asyncio.run(gather_in_order(range(6), worker)) == [0, 2, 4, 6, 8, 10]
    try:
        asyncio.run(gather_in_order(range(10), worker))
    except KeyError as error:
        assert error.args == ("seven",)  # The original error, not an exception group.
    else:
        raise AssertionError("expected KeyError")
