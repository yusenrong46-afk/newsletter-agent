"""Scanning Gmail: freeze a window, list every ID, read metadata, and triage.

Coverage rules: the whole window is listed (archived mail included), IDs are
deduplicated, and Python enforces exact time bounds with internalDate.
"""

import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

from .gmail_client import GmailMcpClient, GmailToolError, make_message_record, sender_address
from .store import InventoryStore

SECONDS_PER_DAY = 86400
EXCLUDED_LABELS = ("SENT", "DRAFT", "SPAM", "TRASH", "CHAT")
RECURRING_SENDER_MINIMUM = 3
DEFAULT_MAX_MODEL_CALLS = 400
DEFAULT_MAX_TOKENS = 4_500_000  # Covers a mailbox of about 5,000 candidates.


@dataclass
class InventoryOptions:
    days: int = 90
    start: datetime | None = None  # If set, scan from here to now instead of `days`.
    preflight: bool = False
    max_messages: int | None = None
    resume: bool = False
    approved: bool = False
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS
    max_tokens: int = DEFAULT_MAX_TOKENS
    quiet: bool = False  # Briefings: no printed preflight or report, no saved report.


@dataclass
class InventoryState:
    """What a checkpoint saves, so --resume can continue the same frozen window."""

    days: int
    window_start_ms: int
    window_end_ms: int
    query: str
    started_at: str
    discovered: list[str] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)
    unresolved: dict[str, str] = field(default_factory=dict)
    disappeared: list[str] = field(default_factory=list)
    usage_total: dict[str, int] = field(default_factory=lambda: {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0})


@dataclass
class InventoryRun:
    """Everything about one command invocation."""

    options: InventoryOptions
    store: InventoryStore
    state: InventoryState
    resumed: bool = False
    mode: str = "classify"  # classify, preflight, or awaiting_approval
    listing_complete: bool = False
    message_cap_reached: bool = False
    budget_exhausted: bool = False
    interrupted: bool = False
    failures: list[dict] = field(default_factory=list)
    duplicate_listings: int = 0
    pages_listed: int = 0
    metadata_fetched: int = 0
    metadata_from_cache: int = 0
    classified_now: int = 0
    classified_from_cache: int = 0
    fabricated_ids: int = 0
    calls_without_usage: int = 0
    model_calls_this_run: int = 0
    tokens_this_run: int = 0
    unread_ids: set | None = None
    unread_observed_at: str | None = None
    preflight: dict | None = None
    report_paths: list[str] = field(default_factory=list)

    def save_checkpoint(self):
        self.store.save_checkpoint(asdict(self.state))


def utc_iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def progress(message: str):
    print(message, file=sys.stderr, flush=True)


def build_scan_query(start_ms: int, end_ms: int) -> str:
    """Search slightly wider than needed; Python enforces exact bounds with internalDate."""
    after = start_ms // 1000 - SECONDS_PER_DAY
    before = end_ms // 1000 + SECONDS_PER_DAY
    return f"after:{after} before:{before} -in:sent -in:drafts -in:chats"


def start_or_resume_state(options: InventoryOptions, store: InventoryStore, now: datetime) -> tuple[InventoryState, bool]:
    if options.resume:
        saved = store.load_checkpoint()
        if saved is None:
            progress("No checkpoint found; starting a new scan.")
        elif saved["days"] != options.days:
            raise ValueError(f"The checkpoint covers {saved['days']} days, not {options.days}. "
                             f"Use --days {saved['days']} or drop --resume.")
        else:
            return InventoryState(**saved), True
    start = options.start or now - timedelta(days=options.days)
    end_ms = int(now.timestamp() * 1000)
    start_ms = int(start.timestamp() * 1000)
    return InventoryState(
        days=options.days, window_start_ms=start_ms, window_end_ms=end_ms,
        query=build_scan_query(start_ms, end_ms), started_at=utc_iso(now),
    ), False


def exclusion_reason(record: dict, state: InventoryState) -> str | None:
    """Why a listed message is outside this inventory, or None if it is in scope."""
    if not state.window_start_ms <= record["internal_date"] < state.window_end_ms:
        return "outside_window"
    for label in EXCLUDED_LABELS:
        if label in record["labels"]:
            return f"label_{label.lower()}"
    return None


async def scan_mailbox(run: InventoryRun, gmail: GmailMcpClient):
    """List every page of IDs, and fetch metadata for IDs not already cached.

    The listing always restarts from page one, because an old page token may
    no longer be valid. Deduplicating by ID prevents double counting.
    """
    state, store = run.state, run.store
    previously_discovered = set(state.discovered) if run.resumed else set()
    seen, seen_set, in_scope_count = [], set(), 0
    page_token, stop = None, False
    try:
        while not stop:
            page = await gmail.search(state.query, page_token)
            run.pages_listed += 1
            for item in page.get("messages", []):
                message_id = item["id"]
                if message_id in seen_set:
                    run.duplicate_listings += 1
                    continue
                if run.options.max_messages and in_scope_count >= run.options.max_messages:
                    run.message_cap_reached = True
                    stop = True
                    break
                state.unavailable.pop(message_id, None)
                record = await fetch_metadata(run, gmail, message_id)
                # Count an ID only after its read finished, so an interrupted or
                # failed read can never leave a discovered ID with no outcome.
                seen.append(message_id)
                seen_set.add(message_id)
                if record and exclusion_reason(record, state) is None:
                    in_scope_count += 1
            store.save_metadata()
            save_discovered(run, seen, previously_discovered)
            progress(f"Scan: page {run.pages_listed}, {len(seen)} unique IDs, "
                     f"{run.metadata_fetched} fetched, {run.metadata_from_cache} cached.")
            next_token = page.get("next_page_token")
            if not stop and (not next_token or next_token == page_token):
                run.listing_complete = not next_token
                if next_token:
                    run.failures.append({"stage": "listing", "error": "Gmail repeated a page token."})
                break
            page_token = next_token
    except GmailToolError as error:
        run.failures.append({"stage": "scan", "error": error.code, "detail": str(error)})
    finally:
        store.save_metadata()
        save_discovered(run, seen, previously_discovered)


def save_discovered(run: InventoryRun, seen: list[str], previously_discovered: set):
    state = run.state
    if run.listing_complete:
        state.disappeared = sorted(previously_discovered.difference(seen))
        for message_id in state.disappeared:
            state.unavailable.pop(message_id, None)
        state.discovered = list(seen)
    else:
        # A partial listing cannot prove anything vanished, so keep earlier IDs.
        state.discovered = list(seen) + sorted(previously_discovered.difference(seen))
    run.save_checkpoint()


async def fetch_metadata(run: InventoryRun, gmail: GmailMcpClient, message_id: str) -> dict | None:
    cached = run.store.metadata.get(message_id)
    if cached:
        run.metadata_from_cache += 1
        return cached
    try:
        raw = await gmail.read_metadata(message_id)
    except GmailToolError as error:
        if error.code != "not_found":
            raise
        run.state.unavailable[message_id] = "not_found"
        return None
    record = make_message_record(raw)
    run.store.metadata[message_id] = record
    run.metadata_fetched += 1
    return record


async def refresh_unread_flags(run: InventoryRun, gmail: GmailMcpClient):
    """Unread can change at any time, so ask Gmail again instead of trusting the cache."""
    unread, page_token = set(), None
    try:
        while True:
            page = await gmail.search(f"{run.state.query} is:unread", page_token)
            unread.update(item["id"] for item in page.get("messages", []))
            page_token = page.get("next_page_token")
            if not page_token:
                break
    except GmailToolError as error:
        progress(f"Could not refresh unread flags ({error.code}); using flags from when metadata was read.")
        return
    run.unread_ids = unread
    run.unread_observed_at = utc_iso(datetime.now(timezone.utc))


def in_scope_records(run: InventoryRun) -> list[dict]:
    records = []
    for message_id in run.state.discovered:
        record = run.store.metadata.get(message_id)
        if record and message_id not in run.state.unavailable and exclusion_reason(record, run.state) is None:
            records.append(record)
    return records


def triage_messages(records: list[dict]) -> dict[str, str]:
    """Decide what may be sent to OpenAI, using only metadata.

    bulk           mailing-list evidence exists: send headers and a body excerpt
    recurring      repeat sender with no bulk evidence (maybe a friend): send headers only
    not_candidate  send nothing, and never count it as "not a newsletter"
    """
    per_sender = Counter(sender_address(record) for record in records)
    decisions = {}
    for record in records:
        if record["bulk_signals"]:
            decisions[record["id"]] = "bulk"
        elif per_sender[sender_address(record)] >= RECURRING_SENDER_MINIMUM:
            decisions[record["id"]] = "recurring"
        else:
            decisions[record["id"]] = "not_candidate"
    return decisions


def is_unread(run: InventoryRun, record: dict) -> bool:
    if run.unread_ids is not None:
        return record["id"] in run.unread_ids
    return "UNREAD" in record["labels"]
