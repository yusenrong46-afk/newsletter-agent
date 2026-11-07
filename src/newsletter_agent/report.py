"""Grouping, counting, and the text report.

Every number here is calculated by Python from stored records. The model
supplies labels and names only.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parseaddr

from . import config
from .classify import ALL_CATEGORIES, MODEL, SUBSCRIPTION_CATEGORIES, body_policy, classification_cache_key
from .gmail_client import EXCERPT_CHARS, sender_address
from .scan import InventoryRun, exclusion_reason, in_scope_records, is_unread, triage_messages, utc_iso


def display_time(milliseconds: int, zone=None) -> str:
    moment = datetime.fromtimestamp(milliseconds / 1000, tz=zone or config.display_zone())
    return moment.strftime("%Y-%m-%d %H:%M %Z")


def group_key(record: dict) -> str:
    """Group by List-ID, otherwise by exact sender address. Never by domain."""
    return f"list:{record['list_id']}" if record["list_id"] else f"sender:{sender_address(record)}"


def most_common(values) -> str | None:
    counts = Counter(value for value in values if value)
    # Ties resolve alphabetically so a report does not change between runs.
    return min(counts, key=lambda value: (-counts[value], value)) if counts else None


def build_group(key: str, messages: list[dict], other_messages: list[dict], zone) -> dict:
    """Summarize one publication group using exact counts from stored records."""
    newest = max(messages, key=lambda message: message["record"]["internal_date"])
    names = Counter(m["result"]["publication_name"] for m in messages if m["result"]["publication_name"])
    fallback_name = parseaddr(newest["record"]["from"])[0] or sender_address(newest["record"])
    is_list_group = key.startswith("list:")
    strong_messages = sum(
        1 for m in messages if m["record"]["has_list_unsubscribe"] and m["result"]["confidence"] != "low"
    )
    observed = is_list_group or strong_messages >= 2
    return {
        "group_key": key,
        "name": most_common(names.elements()) or fallback_name,
        "name_source": "model" if names else "sender display name (no name from model)",
        "name_variants": sorted(names),
        "status": "observed" if observed else "provisional",
        "provisional_reason": None if observed else (
            "No List-ID, and fewer than 2 messages with List-Unsubscribe at medium or high confidence."
        ),
        "senders": sorted({sender_address(m["record"]) for m in messages}),
        "category": most_common(m["result"]["classification"] for m in messages),
        "category_counts": dict(Counter(m["result"]["classification"] for m in messages)),
        "received_count": len(messages),
        "unread_count": sum(1 for m in messages if m["unread"]),
        "confidence": most_common(m["result"]["confidence"] for m in messages),
        "low_confidence_count": sum(1 for m in messages if m["result"]["confidence"] == "low"),
        "grouping_basis": f"List-ID {key.removeprefix('list:')}" if is_list_group else "exact sender address, no List-ID",
        "other_messages_same_identity": dict(Counter(m["result"]["classification"] for m in other_messages)),
        "example": {
            "subject": newest["record"]["subject"],
            "message_id": newest["record"]["id"],
            "received": display_time(newest["record"]["internal_date"], zone),
        },
    }


def build_groups(classified: list[dict], zone=None) -> list[dict]:
    """Only subscription categories form publications; other mail stays visible beside them."""
    subscription, other = defaultdict(list), defaultdict(list)
    for message in classified:
        key = group_key(message["record"])
        if message["result"]["classification"] in SUBSCRIPTION_CATEGORIES:
            subscription[key].append(message)
        else:
            other[key].append(message)
    groups = [build_group(key, messages, other.get(key, []), zone) for key, messages in subscription.items()]
    return sorted(groups, key=lambda group: (-group["received_count"], group["name"].lower(), group["group_key"]))


def find_possible_duplicates(groups: list[dict]) -> list[dict]:
    """Point out groups with the same name. These are hints; they are never merged."""
    by_name = defaultdict(list)
    for group in groups:
        by_name[re.sub(r"[^a-z0-9]", "", group["name"].casefold())].append(group)
    return [
        {"name": matches[0]["name"], "group_keys": sorted(group["group_key"] for group in matches)}
        for normalized, matches in sorted(by_name.items())
        if normalized and len(matches) > 1
    ]


def check_reconciliation(counts: dict, categories: dict, groups: list[dict]) -> list[str]:
    """Recalculate each total a second way. Any mismatch is reported, not hidden."""
    problems = []
    if counts["discovered"] != counts["in_scope"] + counts["excluded"] + counts["unavailable"]:
        problems.append("discovered != in_scope + excluded + unavailable")
    if counts["in_scope"] != counts["candidates"] + counts["not_candidates"]:
        problems.append("in_scope != candidates + not_candidates")
    if counts["candidates"] != counts["classified"] + counts["unresolved"] + counts["not_yet_classified"]:
        problems.append("candidates != classified + unresolved + not_yet_classified")
    if sum(categories.values()) != counts["classified"]:
        problems.append("sum of categories != classified")
    subscription_total = sum(categories[name] for name in SUBSCRIPTION_CATEGORIES)
    if sum(group["received_count"] for group in groups) != subscription_total:
        problems.append("sum of group counts != subscription messages")
    return problems


def decide_status(run: InventoryRun, counts: dict) -> str:
    if run.interrupted:
        return "interrupted"
    if run.failures or (not run.listing_complete and not run.message_cap_reached):
        return "partial_failure"
    if run.mode in ("preflight", "awaiting_approval"):
        return run.mode
    if run.budget_exhausted:
        return "partial_budget"
    if counts["unresolved"] or counts["not_yet_classified"]:
        return "partial_failure"
    if run.message_cap_reached:
        return "partial_message_cap"
    return "complete"


def build_notes(run: InventoryRun, counts: dict, body_statuses: Counter) -> list[str]:
    notes = [
        "This inventories observed email. It does not prove current or paid subscription status.",
        "Gmail is a changing mailbox, not a snapshot: messages and flags can change during a scan.",
        "Unread means the Gmail UNREAD flag when observed, not that you never read the message.",
        "Publication counts are grouped by Python rules; categories and names are model judgments.",
    ]
    if counts["not_candidates"]:
        notes.append(f"{counts['not_candidates']} in-scope messages had no bulk-mail signal and were not sent to OpenAI. "
                     "They are unclassified, not proven non-newsletters.")
    if body_statuses["truncated"]:
        notes.append(f"{body_statuses['truncated']} messages were classified from a body excerpt ({EXCERPT_CHARS} characters at most).")
    if body_statuses["withheld"]:
        notes.append(f"{body_statuses['withheld']} messages were classified from headers only, because no bulk signal justified sending the body.")
    if body_statuses["unavailable"]:
        notes.append(f"{body_statuses['unavailable']} messages had no readable body and were classified from headers only.")
    if run.message_cap_reached:
        notes.append(f"Stopped at the --max-messages limit of {run.options.max_messages}. Coverage is partial.")
    if not run.listing_complete and not run.message_cap_reached:
        notes.append("The Gmail listing did not finish. Coverage is partial.")
    if run.budget_exhausted:
        notes.append("The model budget ran out. Remaining candidates are not yet classified.")
    if run.interrupted:
        notes.append("The run was interrupted. Use --resume to continue.")
    if run.state.disappeared:
        notes.append(f"{len(run.state.disappeared)} messages seen earlier are no longer listed and were removed from all counts.")
    if run.state.unavailable:
        notes.append(f"{len(run.state.unavailable)} listed messages could not be read, probably because they were deleted during the scan.")
    if run.fabricated_ids:
        notes.append(f"The model returned {run.fabricated_ids} IDs that were not in its input; they were discarded.")
    if run.calls_without_usage:
        notes.append(f"{run.calls_without_usage} model calls returned no usable usage data, so actual token totals are understated.")
    if run.unread_ids is None:
        notes.append("Unread flags could not be refreshed; they reflect when each message's metadata was read.")
    return notes


@dataclass
class Outcomes:
    """Where every in-scope message ended up."""

    records: list[dict]
    excluded: Counter
    classified: list[dict]
    unresolved: list[dict]
    not_yet: list[str]
    not_candidate_senders: Counter


def collect_outcomes(run: InventoryRun) -> Outcomes:
    state, classifications = run.state, run.store.classifications
    records = in_scope_records(run)
    triage = triage_messages(records)

    excluded = Counter()
    for message_id in state.discovered:
        record = run.store.metadata.get(message_id)
        if record and message_id not in state.unavailable:
            reason = exclusion_reason(record, state)
            if reason:
                excluded[reason] += 1

    classified, unresolved, not_yet, not_candidate_senders = [], [], [], Counter()
    for record in records:
        decision = triage[record["id"]]
        if decision == "not_candidate":
            not_candidate_senders[sender_address(record)] += 1
            continue
        result = classifications.get(classification_cache_key(record["id"], body_policy(decision)))
        if result:
            classified.append({"record": record, "result": result, "unread": is_unread(run, record)})
        elif record["id"] in state.unresolved:
            unresolved.append({"message_id": record["id"], "subject": record["subject"], "reason": state.unresolved[record["id"]]})
        else:
            not_yet.append(record["id"])
    return Outcomes(records, excluded, classified, unresolved, not_yet, not_candidate_senders)


def build_report(run: InventoryRun) -> dict:
    zone = config.display_zone()
    state = run.state
    outcomes = collect_outcomes(run)
    classified = outcomes.classified

    categories = {name: 0 for name in ALL_CATEGORIES}
    categories.update(Counter(m["result"]["classification"] for m in classified))
    groups = build_groups(classified, zone)
    observed = [group for group in groups if group["status"] == "observed"]
    provisional = [group for group in groups if group["status"] == "provisional"]
    body_statuses = Counter(m["result"]["body_status"] for m in classified)
    subscription_messages = [m for m in classified if m["result"]["classification"] in SUBSCRIPTION_CATEGORIES]
    not_candidates = sum(outcomes.not_candidate_senders.values())

    counts = {
        "discovered": len(state.discovered),
        "in_scope": len(outcomes.records),
        "excluded": sum(outcomes.excluded.values()),
        "excluded_by_reason": dict(outcomes.excluded),
        "unavailable": len(state.unavailable),
        "duplicate_listings_ignored": run.duplicate_listings,
        "metadata_fetched_this_run": run.metadata_fetched,
        "metadata_from_cache_this_run": run.metadata_from_cache,
        "candidates": len(outcomes.records) - not_candidates,
        "not_candidates": not_candidates,
        "classified": len(classified),
        "classified_this_run": run.classified_now,
        "classified_from_cache": run.classified_from_cache,
        "unresolved": len(outcomes.unresolved),
        "not_yet_classified": len(outcomes.not_yet),
        "subscription_messages": len(subscription_messages),
        "subscription_unread": sum(1 for m in subscription_messages if m["unread"]),
        "in_scope_unread": sum(1 for record in outcomes.records if is_unread(run, record)),
        "publications_observed": len(observed),
        "publications_provisional": len(provisional),
    }
    problems = check_reconciliation(counts, categories, groups)
    return {
        "run": {
            "status": decide_status(run, counts),
            "resumed": run.resumed,
            "run_started_at": state.started_at,
            "report_generated_at": utc_iso(datetime.now(timezone.utc)),
            "unread_observed_at": run.unread_observed_at,
            "window_start": display_time(state.window_start_ms, zone),
            "window_end": display_time(state.window_end_ms, zone),
            "window_start_ms": state.window_start_ms,
            "window_end_ms": state.window_end_ms,
            "days": state.days,
            "exclusions": "Spam, trash, drafts, sent mail, and chats. Archived mail is included.",
            "queries": [state.query, f"{state.query} is:unread"],
            "model": MODEL,
            "failures": run.failures,
        },
        "counts": counts,
        "denominators": {
            "category_breakdown": "classified messages",
            "publication_groups": "classified messages in editorial_newsletter, promotional_subscription, or product_update",
            "in_scope_unread": "all in-scope messages, classified or not",
        },
        "category_breakdown": categories,
        "publications_observed": observed,
        "publications_provisional": provisional,
        "possible_duplicates": find_possible_duplicates(groups),
        "unresolved": outcomes.unresolved,
        "unavailable": [{"message_id": key, "reason": value} for key, value in sorted(state.unavailable.items())],
        "not_candidate_senders": [{"sender": s, "messages": n} for s, n in outcomes.not_candidate_senders.most_common()],
        "messages": [
            {
                "message_id": m["record"]["id"], "thread_id": m["record"]["thread_id"],
                "group_key": group_key(m["record"]), "subject": m["record"]["subject"],
                "from": m["record"]["from"], "internal_date": m["record"]["internal_date"],
                "received": display_time(m["record"]["internal_date"], zone),
                "unread": m["unread"], "classification": m["result"]["classification"],
                "publication_name": m["result"]["publication_name"], "confidence": m["result"]["confidence"],
                "rationale": m["result"]["rationale"], "body_status": m["result"]["body_status"],
            }
            for m in classified
        ],
        "usage": {
            "model_calls_this_run": run.model_calls_this_run,
            "tokens_this_run": run.tokens_this_run,
            "total_for_this_window": state.usage_total,
            "calls_without_usage_data": run.calls_without_usage,
            "fabricated_ids_discarded": run.fabricated_ids,
        },
        "preflight": run.preflight,
        "notes": build_notes(run, counts, body_statuses),
        "reconciliation": {"ok": not problems, "problems": problems},
    }


# --- Text rendering -----------------------------------------------------------

def render_preflight(preflight: dict) -> str:
    """Show only aggregate numbers: no senders or subjects."""
    budget = preflight["budget"]
    fits = "fits" if preflight["fits_budget"] else "MAY NOT FIT; the run would stop with a partial report"
    return "\n".join([
        "Preflight (no email has been sent to OpenAI)",
        f"  Unique IDs listed:              {preflight['unique_ids_listed']}",
        f"  In-scope messages:              {preflight['in_scope_messages']}",
        f"  With bulk-mail signals:         {preflight['bulk_signal_messages']}",
        f"  Repeat senders, no bulk signal: {preflight['recurring_sender_only_messages']} (headers only)",
        f"  Will not be sent to OpenAI:     {preflight['not_sent_to_openai']}",
        f"  Already in cache:               {preflight['already_classified_in_cache']}",
        f"  To classify now:                {preflight['messages_to_classify']} in {preflight['batches']} batches",
        f"  Estimated input tokens:         {preflight['estimated_input_tokens']:,}",
        f"  Maximum output tokens:          {preflight['maximum_output_tokens']:,}",
        f"  Budget:                         {budget['max_model_calls']} calls, {budget['max_tokens']:,} tokens ({fits})",
        f"  Assumptions: {preflight['assumptions']}",
    ])


def render_group(group: dict) -> list[str]:
    lines = [
        f"- {group['name']}: {group['received_count']} received, {group['unread_count']} unread",
        f"    Category: {group['category']} | Confidence: {group['confidence']} ({group['low_confidence_count']} low)",
        f"    Senders: {', '.join(group['senders'])}",
        f"    Grouped by: {group['grouping_basis']}",
        f"    Example: \"{group['example']['subject']}\" ({group['example']['received']}, message {group['example']['message_id']})",
    ]
    if group["provisional_reason"]:
        lines.append(f"    Provisional because: {group['provisional_reason']}")
    if group["other_messages_same_identity"]:
        other = ", ".join(f"{name} {count}" for name, count in sorted(group["other_messages_same_identity"].items()))
        lines.append(f"    Other mail from this identity (not counted above): {other}")
    if len(group["name_variants"]) > 1:
        lines.append(f"    Name variants from the model: {', '.join(group['name_variants'])}")
    return lines


def render_report(report: dict) -> str:
    run, counts = report["run"], report["counts"]
    total_usage = report["usage"]["total_for_this_window"]
    lines = [
        "# Gmail Newsletter Inventory",
        "",
        f"Status: {run['status']}",
        f"Period: {run['window_start']} to {run['window_end']} ({run['days']} days, exact internalDate bounds)",
        f"Exclusions: {run['exclusions']}",
        f"Queries: {' || '.join(run['queries'])}",
        f"Run started: {run['run_started_at']} | Report: {run['report_generated_at']}",
        f"Unread observed: {run['unread_observed_at'] or 'when metadata was read'}",
        "",
        "## Answer",
        f"Publications: {counts['publications_observed']} observed + {counts['publications_provisional']} provisional",
        f"Subscription messages: {counts['subscription_messages']} ({counts['subscription_unread']} unread)",
        "",
        "## Coverage (exact counts)",
        f"Unique IDs discovered: {counts['discovered']} "
        f"(in scope {counts['in_scope']}, excluded {counts['excluded']}, unavailable {counts['unavailable']})",
        f"Metadata this run: {counts['metadata_fetched_this_run']} fetched, {counts['metadata_from_cache_this_run']} from cache",
        f"Candidates: {counts['candidates']} | Not sent to OpenAI: {counts['not_candidates']}",
        f"Classified: {counts['classified']} ({counts['classified_this_run']} new, {counts['classified_from_cache']} from cache)",
        f"Unresolved: {counts['unresolved']} | Not yet classified: {counts['not_yet_classified']}",
        f"All in-scope unread: {counts['in_scope_unread']}",
        f"Failures: {len(run['failures'])}",
        "",
        "## Categories (share of classified messages; model judgment, Python counts)",
    ]
    lines.extend(f"- {name}: {count}" for name, count in report["category_breakdown"].items())
    lines.extend(["", f"## Observed publication groups ({counts['publications_observed']})"])
    for group in report["publications_observed"]:
        lines.extend(render_group(group))
    lines.extend(["", f"## Provisional groups ({counts['publications_provisional']})"])
    for group in report["publications_provisional"]:
        lines.extend(render_group(group))
    if report["possible_duplicates"]:
        lines.extend(["", "## Possibly the same publication (not merged)"])
        lines.extend(f"- {item['name']}: {', '.join(item['group_keys'])}" for item in report["possible_duplicates"])
    if report["unresolved"]:
        lines.extend(["", "## Unresolved messages (not counted as non-newsletters)"])
        lines.extend(f"- {item['message_id']}: {item['reason']}" for item in report["unresolved"])
    if run["failures"]:
        lines.extend(["", "## Failures"])
        lines.extend(f"- {item['stage']}: {item.get('detail') or item['error']}" for item in run["failures"])
    lines.extend([
        "", "## OpenAI usage (actual, from the API)",
        f"Model calls this run: {report['usage']['model_calls_this_run']}",
        f"Total for this window: {total_usage['model_calls']} calls, {total_usage['input_tokens']:,} input tokens, "
        f"{total_usage['output_tokens']:,} output tokens, {total_usage['total_tokens']:,} total tokens",
        "", "## Notes",
    ])
    lines.extend(f"- {note}" for note in report["notes"])
    reconciliation = report["reconciliation"]
    lines.extend([
        "", "## Reconciliation",
        "All totals match." if reconciliation["ok"] else "PROBLEM: " + "; ".join(reconciliation["problems"]),
    ])
    return "\n".join(lines) + "\n"
