"""Measure classification quality on a labelled, synthetic dataset.

    newsletter-agent eval                 offline stand-in model (free, no key)
    newsletter-agent eval --real-model    the real model, on the bundled synthetic emails only

The eval runs the real pipeline (MCP server, scan, triage, classification,
validation) on invented emails with known answers, then scores it:

- subscription detection: is this email a newsletter/promotion/product update
  or not? Scored end to end, so mail the privacy triage never sends to the
  model counts as "not a subscription".
- categories: per-category precision, recall and F1, and a confusion matrix,
  over the emails that reached the model.
- triage: how many real subscriptions reached the model, and how many personal
  emails were kept from it (or sent as headers only).
- publication names, unresolved answers, and tokens per 1,000 emails.

No real email is involved: the mailbox is SyntheticGmail, and --real-model
refuses any dataset except the bundled one.
"""

import asyncio
import json
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from html import escape
from importlib import resources
from pathlib import Path

from . import config
from .classify import ALL_CATEGORIES, MODEL, SUBSCRIPTION_CATEGORIES, ModelClient
from .demo import DAY_MS, OfflineModel, SyntheticGmail, demo_gmail_client, encode, make_message
from .html_report import CATEGORY_LABELS, card, page
from .inventory import run_inventory
from .scan import InventoryOptions, triage_messages
from .store import InventoryStore

EVAL_NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
EVAL_MAX_MODEL_CALLS = 20
EVAL_MAX_TOKENS = 400_000
EXTRA_OUTCOMES = ("unresolved", "pending", "not_sent")
SHIP_BAR_SUBSCRIPTION_F1 = 0.90
REQUIRED_KEYS = ("id", "from", "subject", "expected", "days_ago")


class DatasetError(ValueError):
    pass


def bundled_dataset_text() -> str:
    return (resources.files("newsletter_agent") / "eval_data" / "dataset.jsonl").read_text(encoding="utf-8")


def load_dataset(path: Path | None = None) -> list[dict]:
    """Read and validate a JSONL dataset. Without a path, the bundled synthetic one."""
    text = path.read_text(encoding="utf-8") if path else bundled_dataset_text()
    cases, seen = [], set()
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            raise DatasetError(f"Line {number} is not valid JSON.") from None
        missing = [key for key in REQUIRED_KEYS if key not in item]
        if missing:
            raise DatasetError(f"Line {number} is missing {', '.join(missing)}.")
        if item["expected"] not in ALL_CATEGORIES:
            raise DatasetError(f"Line {number}: unknown label {item['expected']!r}. "
                               f"Use one of {', '.join(ALL_CATEGORIES)}.")
        if item["id"] in seen:
            raise DatasetError(f"Line {number}: duplicate id {item['id']!r}.")
        if not 0 <= float(item["days_ago"]) < 89:
            raise DatasetError(f"Line {number}: days_ago must be between 0 and 89, so the email is in the window.")
        seen.add(item["id"])
        cases.append(item)
    if not cases:
        raise DatasetError("The dataset is empty.")
    return cases


def build_mailbox(cases: list[dict]) -> SyntheticGmail:
    now_ms = int(EVAL_NOW.timestamp() * 1000)
    messages = []
    for item in cases:
        payload = None
        if item.get("html"):
            payload = {"mimeType": "multipart/alternative",
                       "parts": [{"mimeType": "text/html", "body": {"data": encode(item["html"])}}]}
        messages.append(make_message(
            item["id"], item["from"], item["subject"], now_ms - int(float(item["days_ago"]) * DAY_MS),
            item.get("labels", ["INBOX"]), item.get("headers") or {}, item.get("body", ""), payload,
        ))
    return SyntheticGmail(messages, page_size=100)


async def run_cases(cases: list[dict], model_factory: Callable[[], ModelClient], approved: bool,
                    confirm, max_model_calls: int, max_tokens: int) -> tuple[dict, dict[str, str]]:
    """Run the real inventory on the dataset; return the report and each email's triage tier."""
    mailbox = build_mailbox(cases)
    with tempfile.TemporaryDirectory() as scratch:
        store = InventoryStore(Path(scratch) / "inventory")
        try:
            async with demo_gmail_client(mailbox) as gmail:
                options = InventoryOptions(days=90, approved=approved, quiet=True,
                                           max_model_calls=max_model_calls, max_tokens=max_tokens)
                report = await run_inventory(gmail, options, store, openai_factory=model_factory,
                                             confirm=confirm, now=EVAL_NOW)
            records = [store.metadata[item["id"]] for item in cases if item["id"] in store.metadata]
            triage = triage_messages(records)
        finally:
            store.close()
    return report, triage


def predictions(cases: list[dict], report: dict, triage: dict[str, str]) -> dict[str, str]:
    """Each email's outcome: a category, or unresolved, pending (budget) or not_sent (privacy triage)."""
    classified = {message["message_id"]: message["classification"] for message in report["messages"]}
    unresolved = {item["message_id"] for item in report["unresolved"]}
    outcome = {}
    for item in cases:
        message_id = item["id"]
        if message_id in classified:
            outcome[message_id] = classified[message_id]
        elif triage.get(message_id) == "not_candidate":
            outcome[message_id] = "not_sent"
        elif message_id in unresolved:
            outcome[message_id] = "unresolved"
        else:
            outcome[message_id] = "pending"
    return outcome


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def f1(precision: float | None, recall: float | None) -> float | None:
    if not precision or not recall:
        return 0.0 if precision is not None and recall is not None else None
    return round(2 * precision * recall / (precision + recall), 4)


def score(cases: list[dict], outcome: dict[str, str], triage: dict[str, str],
          names: dict[str, str | None] | None = None) -> dict:
    """Every number here is plain arithmetic over expected vs. predicted labels."""
    columns = ALL_CATEGORIES + EXTRA_OUTCOMES
    confusion = {expected: dict.fromkeys(columns, 0) for expected in ALL_CATEGORIES}
    for item in cases:
        confusion[item["expected"]][outcome[item["id"]]] += 1

    per_category = {}
    for category in ALL_CATEGORIES:
        true_positive = confusion[category][category]
        predicted = sum(confusion[expected][category] for expected in ALL_CATEGORIES)
        reached = sum(count for column, count in confusion[category].items() if column != "not_sent")
        precision, recall = ratio(true_positive, predicted), ratio(true_positive, reached)
        # A category with examples that the model never predicts has failed completely: its F1 is 0,
        # not "n/a", or macro F1 would leave it out and look better than it is.
        score_f1 = 0.0 if reached and not predicted else f1(precision, recall)
        per_category[category] = {"support": reached, "predicted": predicted, "precision": precision,
                                  "recall": recall, "f1": score_f1}
    scored = [values["f1"] for values in per_category.values() if values["support"] and values["f1"] is not None]

    reached_model = [item for item in cases if outcome[item["id"]] != "not_sent"]
    correct = sum(1 for item in reached_model if outcome[item["id"]] == item["expected"])

    counts: Counter[tuple[bool, bool]] = Counter()
    for item in cases:
        actual = item["expected"] in SUBSCRIPTION_CATEGORIES
        predicted_subscription = outcome[item["id"]] in SUBSCRIPTION_CATEGORIES
        counts[(actual, predicted_subscription)] += 1
    tp, fp, fn, tn = counts[(True, True)], counts[(False, True)], counts[(True, False)], counts[(False, False)]
    sub_precision, sub_recall = ratio(tp, tp + fp), ratio(tp, tp + fn)

    subscriptions = [item for item in cases if item["expected"] in SUBSCRIPTION_CATEGORIES]
    personal = [item for item in cases if item["expected"] == "personal_or_other"]
    names = names or {}
    named = [item for item in cases if item.get("publication") and item["id"] in names]
    name_matches = sum(1 for item in named if (names[item["id"]] or "").casefold() == item["publication"].casefold())

    mistakes = [
        {"id": item["id"], "subject": item["subject"], "expected": item["expected"],
         "predicted": outcome[item["id"]], "note": item.get("note", "")}
        for item in cases if outcome[item["id"]] != item["expected"]
        and not (outcome[item["id"]] == "not_sent" and item["expected"] == "personal_or_other")
    ]
    return {
        "cases": len(cases),
        "subscription_detection": {
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "precision": sub_precision, "recall": sub_recall, "f1": f1(sub_precision, sub_recall),
            "accuracy": ratio(tp + tn, len(cases)),
        },
        "category_accuracy": ratio(correct, len(reached_model)),
        "macro_f1": round(sum(scored) / len(scored), 4) if scored else None,  # type: ignore[arg-type]
        "per_category": per_category,
        "confusion": confusion,
        "triage": {
            "tiers": dict(Counter(triage.get(item["id"], "missing") for item in cases)),
            "subscriptions_reaching_model": ratio(
                sum(1 for item in subscriptions if triage.get(item["id"]) != "not_candidate"), len(subscriptions)),
            "personal_kept_from_model_or_headers_only": ratio(
                sum(1 for item in personal if triage.get(item["id"]) in ("not_candidate", "recurring")), len(personal)),
        },
        "unresolved_rate": ratio(sum(1 for item in reached_model if outcome[item["id"]] == "unresolved"),
                                 len(reached_model)),
        "publication_name_accuracy": ratio(name_matches, len(named)),
        "mistakes": mistakes,
    }


def run_eval(cases: list[dict], model_factory: Callable[[], ModelClient], label: str, approved: bool = True,
             confirm=lambda preflight: False, max_model_calls: int = EVAL_MAX_MODEL_CALLS,
             max_tokens: int = EVAL_MAX_TOKENS) -> dict:
    report, triage = asyncio.run(run_cases(cases, model_factory, approved, confirm, max_model_calls, max_tokens))
    outcome = predictions(cases, report, triage)
    names = {message["message_id"]: message["publication_name"] for message in report["messages"]}
    usage = report["usage"]["total_for_this_window"]
    status = report["run"]["status"]
    result = {
        "label": label,
        "model": MODEL if label != "offline" else "offline stand-in (keyword rules)",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {"cases": len(cases), "by_label": dict(Counter(item["expected"] for item in cases)),
                    "by_source": dict(Counter(item.get("source", "custom") for item in cases))},
        "status": status,
        "complete": status == "complete",
        **score(cases, outcome, triage, names),
        "usage": {**usage, "tokens_per_1000_emails": round(1000 * usage["total_tokens"] / len(cases))},
        "ship_bar": {"metric": "subscription_detection.f1", "minimum": SHIP_BAR_SUBSCRIPTION_F1},
    }
    if not result["complete"]:
        result["incomplete_reason"] = f"The run finished with status {status}; some emails were not classified."
    f1_value = result["subscription_detection"]["f1"]
    result["ship_bar"]["met"] = bool(f1_value is not None and f1_value >= SHIP_BAR_SUBSCRIPTION_F1)
    return result


# --- Output -----------------------------------------------------------------------

def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def render_eval_html(result: dict) -> str:
    columns = ALL_CATEGORIES + EXTRA_OUTCOMES
    short = {**CATEGORY_LABELS, "unresolved": "Unresolved", "pending": "Pending", "not_sent": "Not sent"}
    header = "".join(f'<th class="num">{escape(short[column])}</th>' for column in columns)
    rows = "".join(
        f"<tr><td>{escape(short[expected])}</td>"
        + "".join(f'<td class="num">{result["confusion"][expected][column] or ""}</td>' for column in columns)
        + "</tr>"
        for expected in ALL_CATEGORIES if sum(result["confusion"][expected].values())
    )
    category_rows = "".join(
        f'<tr><td>{escape(short[name])}</td><td class="num">{values["support"]}</td>'
        f'<td class="num">{percent(values["precision"])}</td><td class="num">{percent(values["recall"])}</td>'
        f'<td class="num">{percent(values["f1"])}</td></tr>'
        for name, values in result["per_category"].items() if values["support"] or values["predicted"]
    )
    mistakes = "".join(
        f'<li>{escape(item["subject"])}: expected {escape(short[item["expected"]])}, '
        f'got {escape(short[item["predicted"]])}{" · " + escape(item["note"]) if item["note"] else ""}</li>'
        for item in result["mistakes"]
    ) or "<li>None.</li>"
    detection = result["subscription_detection"]
    bar = result["ship_bar"]
    body = f"""
<h1>Classification quality</h1>
<p class="muted">{escape(result['model'])} · {result['cases']} synthetic emails · status {escape(result['status'])}
· ship bar: subscription F1 ≥ {percent(bar['minimum'])} ({'met' if bar['met'] else 'not met'})</p>
<div class="cards">
  {card(percent(detection['f1']), 'subscription detection F1')}
  {card(percent(result['category_accuracy']), 'category accuracy')}
  {card(percent(result['macro_f1']), 'macro F1 over categories')}
  {card(percent(result['triage']['subscriptions_reaching_model']), 'subscriptions reaching the model')}
  {card(percent(result['unresolved_rate']), 'unresolved answers')}
  {card(f"{result['usage']['tokens_per_1000_emails']:,}", 'tokens per 1,000 emails')}
</div>
<h2>Per category (emails that reached the model)</h2>
<div class="table-wrap"><table><thead><tr><th>Category</th><th class="num">Emails</th><th class="num">Precision</th>
<th class="num">Recall</th><th class="num">F1</th></tr></thead><tbody>{category_rows}</tbody></table></div>
<h2>Confusion matrix (rows: true label, columns: outcome)</h2>
<div class="table-wrap"><table><thead><tr><th>True label</th>{header}</tr></thead><tbody>{rows}</tbody></table></div>
<h2>Mistakes</h2>
<ul class="notes">{mistakes}</ul>
<p class="muted small">"Not sent" means the privacy triage kept the email from the model. For personal email that is
the right outcome and is not counted as a mistake.</p>
"""
    return page("Classification quality", body)


def summary_lines(result: dict) -> list[str]:
    detection = result["subscription_detection"]
    lines = [
        f"Model: {result['model']} · {result['cases']} synthetic emails · status {result['status']}",
        f"Subscription detection: F1 {percent(detection['f1'])} "
        f"(precision {percent(detection['precision'])}, recall {percent(detection['recall'])})",
        f"Category accuracy: {percent(result['category_accuracy'])} · macro F1 {percent(result['macro_f1'])}",
        f"Subscriptions reaching the model: {percent(result['triage']['subscriptions_reaching_model'])} · "
        f"unresolved: {percent(result['unresolved_rate'])}",
        f"Tokens: {result['usage']['total_tokens']:,} ({result['usage']['tokens_per_1000_emails']:,} per 1,000 emails)",
        f"Ship bar (subscription F1 ≥ {percent(result['ship_bar']['minimum'])}): "
        f"{'met' if result['ship_bar']['met'] else 'NOT met'}",
    ]
    if not result["complete"]:
        lines.append(f"Incomplete: {result['incomplete_reason']}")
    return lines


def save_result(result: dict, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{result['label']}-{result['generated_at'][:10]}"
    json_path, html_path = directory / f"{stem}.json", directory / f"{stem}.html"
    config.write_private_file(json_path, json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    config.write_private_file(html_path, render_eval_html(result))
    return [json_path, html_path]


def run_eval_command(real_model: bool = False, dataset: str | None = None, output: str | None = None,
                     approved: bool = False) -> int:
    if real_model and dataset:
        raise ValueError("The real-model eval only runs on the bundled synthetic dataset, "
                         "so no other email can be sent to OpenAI. Drop --dataset or --real-model.")
    cases = load_dataset(Path(dataset) if dataset else None)
    directory = Path(output) if output else config.private_dir("evals")
    if real_model:
        from .briefing import ask_yes_no
        from .classify import make_openai_client

        def confirm(preflight: dict) -> bool:
            return ask_yes_no(
                f"Send {preflight['messages_to_classify']} invented emails to OpenAI ({MODEL}) for the eval? "
                f"At most {preflight['estimated_maximum_total_tokens']:,} tokens; "
                f"hard limit {EVAL_MAX_MODEL_CALLS} calls / {EVAL_MAX_TOKENS:,} tokens.")

        result = run_eval(cases, make_openai_client, MODEL, approved=approved, confirm=confirm)
        if result["status"] == "awaiting_approval":
            # Nothing was sent, so there is nothing to score or save (and no earlier result to overwrite).
            if sys.stdin.isatty():
                print("Cancelled. Nothing was sent to OpenAI and no result was saved.")
                return 0
            print("Not interactive: re-run with --yes to approve sending the synthetic emails to OpenAI.")
            return 2
    else:
        model = OfflineModel()
        result = run_eval(cases, lambda: model, "offline")
    for line in summary_lines(result):
        print(line)
    for path in save_result(result, directory):
        print(f"Saved: {path}", file=sys.stderr)
    return 0 if result["complete"] else 1
