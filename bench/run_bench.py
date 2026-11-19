"""Benchmarks for the before/after upgrade report. Synthetic email only.

This script only uses APIs that exist in every version being compared
(run_inventory, InventoryOptions, InventoryStore, build_mcp_server,
GmailMcpClient, SyntheticGmail, make_message, OfflineModel), so the same file
can measure the baseline and the upgraded code:

    # measure the current tree
    uv run python bench/run_bench.py --label after

    # measure another checkout (for example a git worktree of the baseline)
    cd ../baseline && uv sync && uv run python ../newsletter-agent/bench/run_bench.py --label before --tree .

Nothing here touches Gmail or OpenAI. Gmail is SyntheticGmail with an
artificial delay per read; the model is the offline demo stand-in.
"""

import argparse
import asyncio
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from mcp.shared.memory import create_connected_server_and_client_session

from newsletter_agent import config
from newsletter_agent.demo import OfflineModel, SyntheticGmail, make_message
from newsletter_agent.gmail_client import GmailMcpClient
from newsletter_agent.gmail_server import build_mcp_server
from newsletter_agent.inventory import run_inventory
from newsletter_agent.scan import InventoryOptions
from newsletter_agent.store import InventoryStore

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
NOW_MS = int(NOW.timestamp() * 1000)
DAY_MS = 86_400_000

# Real Gmail: the README measures about 3 metadata reads per second, one at a time.
REAL_GMAIL_SECONDS_PER_READ = 0.33
PROJECTION_MESSAGES = 5000

RUFF_ARGS = ["--line-length", "120", "--select", "E,F,W,B,UP,SIM,I", "--target-version", "py311"]
MYPY_ARGS = ["--python-version", "3.11", "--check-untyped-defs", "--ignore-missing-imports"]
RUFF_VERSION = "0.17.6"
MYPY_VERSION = "2.4.1"


class SlowGmail(SyntheticGmail):
    """SyntheticGmail with a fixed delay per message read, and a count of parallel reads."""

    def __init__(self, messages, latency_seconds: float, page_size: int = 500):
        super().__init__(messages, page_size=page_size)
        self.latency = latency_seconds
        self.in_flight = 0
        self.peak_in_flight = 0
        self._lock = threading.Lock()

    def get(self, userId, id, format, metadataHeaders=None):
        request = super().get(userId, id, format, metadataHeaders)
        inner = request.action

        def action():
            with self._lock:
                self.in_flight += 1
                self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            try:
                time.sleep(self.latency)
                return inner()
            finally:
                with self._lock:
                    self.in_flight -= 1

        request.action = action
        return request


def synthetic_newsletters(count: int) -> list[dict]:
    """Newsletters from 20 publications, spread over the last 89 days."""
    messages = []
    for index in range(count):
        publication = index % 20
        when = NOW_MS - int((index / max(count, 1)) * 89 * DAY_MS) - 60_000
        headers = {"List-Id": f"<pub{publication}.bench.example>",
                   "List-Unsubscribe": "<mailto:leave@bench.example>"}
        messages.append(make_message(
            f"b{index:05d}", f"Publication {publication} <news@pub{publication}.bench.example>",
            f"Issue {index}: notes from publication {publication}", when,
            ["INBOX", "UNREAD"] if index % 3 else ["INBOX"], headers,
            f"Issue {index}. A short synthetic article body for benchmarking.",
        ))
    return messages


async def run_once(google, store_dir: Path, options: InventoryOptions, model=None) -> dict:
    store = InventoryStore(store_dir)
    model = model or OfflineModel()
    async with create_connected_server_and_client_session(build_mcp_server(lambda: google)) as session:
        gmail = GmailMcpClient(session)
        await gmail.verify_tools()
        return await run_inventory(gmail, options, store, openai_factory=lambda: model,
                                   confirm=lambda preflight: True, now=NOW)


def metadata_reads(google) -> int:
    return sum(1 for call in google.get_calls if call[1] == "metadata")


def bytes_written() -> int | None:
    """Bytes this process passed to write() so far (Linux only)."""
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            if line.startswith("wchar:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


class WriteCounter:
    """Counts whole-file rewrites through config.write_private_file."""

    def __init__(self):
        self.calls = 0
        self.bytes = 0
        self._original = config.write_private_file

    def __enter__(self):
        def counting(path, text):
            self.calls += 1
            self.bytes += len(text.encode("utf-8"))
            return self._original(path, text)

        config.write_private_file = counting
        return self

    def __exit__(self, *exc):
        config.write_private_file = self._original


def bench_scan_speed(messages: int, latency_ms: int) -> dict:
    """Scan only (preflight: list + read headers + triage), with a delay per Gmail read."""
    google = SlowGmail(synthetic_newsletters(messages), latency_ms / 1000)
    with tempfile.TemporaryDirectory() as scratch:
        start = time.perf_counter()
        report = asyncio.run(run_once(google, Path(scratch) / "inventory",
                                      InventoryOptions(days=90, preflight=True, quiet=True)))
        seconds = time.perf_counter() - start
    assert report["counts"]["discovered"] == messages, report["counts"]
    sequential_seconds = messages * latency_ms / 1000
    speedup = sequential_seconds / seconds
    projection_before_minutes = PROJECTION_MESSAGES * REAL_GMAIL_SECONDS_PER_READ / 60
    return {
        "messages": messages, "latency_ms": latency_ms, "seconds": round(seconds, 2),
        "messages_per_second": round(messages / seconds, 1),
        "peak_parallel_reads": google.peak_in_flight,
        "metadata_reads": metadata_reads(google),
        "speedup_vs_sequential": round(speedup, 2),
        "projection": {
            "messages": PROJECTION_MESSAGES,
            "seconds_per_read": REAL_GMAIL_SECONDS_PER_READ,
            "sequential_minutes": round(projection_before_minutes, 1),
            "minutes": round(projection_before_minutes / max(speedup, 1.0), 1),
            "assumption": "Real Gmail at about 3 reads per second one at a time, scaled by the speedup measured here.",
        },
    }


def bench_full_run(messages: int, latency_ms: int) -> dict:
    """Scan, read body excerpts and classify (offline model), with a delay per Gmail read."""
    google = SlowGmail(synthetic_newsletters(messages), latency_ms / 1000)
    with tempfile.TemporaryDirectory() as scratch:
        start = time.perf_counter()
        report = asyncio.run(run_once(google, Path(scratch) / "inventory",
                                      InventoryOptions(days=90, approved=True, quiet=True)))
        seconds = time.perf_counter() - start
    assert report["counts"]["classified"] == messages, report["counts"]
    return {
        "messages": messages, "latency_ms": latency_ms, "seconds": round(seconds, 2),
        "gmail_reads": len(google.get_calls), "peak_parallel_reads": google.peak_in_flight,
        "status": report["run"]["status"],
    }


def bench_disk_and_rerun(messages: int) -> tuple[dict, dict]:
    """Bytes written during a large scan, then a fully cached rerun."""
    corpus = synthetic_newsletters(messages)
    with tempfile.TemporaryDirectory() as scratch:
        store_dir = Path(scratch) / "inventory"
        google = SyntheticGmail(corpus, page_size=500)
        before = bytes_written()
        with WriteCounter() as counter:
            start = time.perf_counter()
            report = asyncio.run(run_once(google, store_dir, InventoryOptions(days=90, preflight=True, quiet=True)))
            seconds = time.perf_counter() - start
        after = bytes_written()
        assert report["counts"]["discovered"] == messages
        on_disk = sum(path.stat().st_size for path in store_dir.rglob("*") if path.is_file())
        disk = {
            "messages": messages, "pages": (messages + 499) // 500, "seconds": round(seconds, 2),
            "bytes_written": (after - before) if before is not None and after is not None else None,
            "whole_file_rewrites": counter.calls,
            "whole_file_rewrite_bytes": counter.bytes,
            "bytes_on_disk": on_disk,
        }

        google_again = SyntheticGmail(corpus, page_size=500)
        start = time.perf_counter()
        rerun_report = asyncio.run(run_once(google_again, store_dir,
                                            InventoryOptions(days=90, preflight=True, quiet=True)))
        rerun = {
            "messages": messages, "seconds": round(time.perf_counter() - start, 2),
            "metadata_reads": metadata_reads(google_again),
            "discovered": rerun_report["counts"]["discovered"],
        }
    return disk, rerun


def bench_disk_scaling(sizes: tuple[int, ...]) -> list[dict]:
    """Bytes written by a first scan at several mailbox sizes (shows how the cost grows)."""
    rows = []
    for size in sizes:
        disk, _ = bench_disk_and_rerun(size)
        rows.append({"messages": size, "bytes_written": disk["bytes_written"],
                     "whole_file_rewrites": disk["whole_file_rewrites"], "seconds": disk["seconds"]})
    return rows


def demo_answers() -> dict:
    """The numbers the upgrade must not change: the demo pipeline with a fixed date."""
    from newsletter_agent.demo import run_demo_async

    model = OfflineModel()
    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch) / "site"
        output.mkdir()
        report, briefing = asyncio.run(run_demo_async(output, lambda: model, True, NOW))
    groups = report["publications_observed"] + report["publications_provisional"]
    return {
        "counts": report["counts"],
        "category_breakdown": report["category_breakdown"],
        "groups": sorted(
            ({"group_key": g["group_key"], "name": g["name"], "category": g["category"], "status": g["status"],
              "received": g["received_count"], "unread": g["unread_count"]} for g in groups),
            key=lambda g: g["group_key"],
        ),
        "reconciliation_ok": report["reconciliation"]["ok"],
        "status": report["run"]["status"],
        "briefing_counts": briefing["counts"],
    }


def run_command(command: list[str], cwd: Path, stdout_only: bool = False) -> tuple[int, str]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return result.returncode, result.stdout if stdout_only else result.stdout + result.stderr


def code_health(tree: Path) -> dict:
    """Lint findings, type errors and tests, with the same settings for every version."""
    _, ruff_output = run_command(["uvx", f"ruff@{RUFF_VERSION}", "check", *RUFF_ARGS, "--isolated",
                                  "--output-format", "json", "src", "tests"], tree, stdout_only=True)
    try:
        ruff_findings = len(json.loads(ruff_output))
    except ValueError:
        ruff_findings = None

    _, mypy_output = run_command(["uv", "run", "--with", f"mypy=={MYPY_VERSION}", "mypy", *MYPY_ARGS,
                                  "--config-file", os.devnull, "src"], tree)
    match = re.search(r"Found (\d+) errors?", mypy_output)
    mypy_errors = int(match.group(1)) if match else (0 if "Success" in mypy_output else None)

    start = time.perf_counter()
    _, pytest_output = run_command(["uv", "run", "pytest", "-q", "-p", "no:cacheprovider"], tree)
    tests_seconds = time.perf_counter() - start
    passed = re.search(r"(\d+) passed", pytest_output)
    failed = re.search(r"(\d+) failed", pytest_output)
    return {
        "ruff_findings": ruff_findings, "mypy_errors": mypy_errors,
        "tests_passed": int(passed.group(1)) if passed else 0,
        "tests_failed": int(failed.group(1)) if failed else 0,
        "tests_seconds": round(tests_seconds, 1),
        "settings": {"ruff": f"ruff {RUFF_VERSION} " + " ".join(RUFF_ARGS),
                     "mypy": f"mypy {MYPY_VERSION} " + " ".join(MYPY_ARGS)},
    }


def git_commit(tree: Path) -> str | None:
    code, output = run_command(["git", "rev-parse", "--short", "HEAD"], tree)
    return output.strip() if code == 0 else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="before or after")
    parser.add_argument("--tree", default=str(Path(__file__).resolve().parent.parent),
                        help="Checkout to run lint, type and test checks in")
    parser.add_argument("--output", help="Where to write the JSON (default bench/results/<label>.json)")
    parser.add_argument("--quick", action="store_true", help="Smaller sizes, for a smoke test")
    parser.add_argument("--skip-code-health", action="store_true")
    args = parser.parse_args(argv)

    os.environ.setdefault("NEWSLETTER_AGENT_HOME", tempfile.mkdtemp(prefix="bench-home-"))
    tree = Path(args.tree).resolve()
    scan_messages, full_messages, disk_messages = (120, 60, 600) if args.quick else (1000, 300, 5000)

    def step(name):
        print(f"[bench] {name}", file=sys.stderr, flush=True)

    results: dict = {
        "label": args.label,
        "commit": git_commit(tree),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
    }
    step("scan speed")
    results["scan_speed"] = bench_scan_speed(scan_messages, latency_ms=30)
    step("full run")
    results["full_run"] = bench_full_run(full_messages, latency_ms=30)
    step("disk writes and cached rerun")
    results["disk"], results["rerun"] = bench_disk_and_rerun(disk_messages)
    step("disk scaling")
    results["disk_scaling"] = bench_disk_scaling((300, 600) if args.quick else (1000, 5000, 20000))
    step("demo answers")
    results["answers"] = demo_answers()
    if not args.skip_code_health:
        step("code health")
        results["code_health"] = code_health(tree)

    output = Path(args.output) if args.output else Path(__file__).resolve().parent / "results" / f"{args.label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "answers"}, indent=1))
    print(f"Saved {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
