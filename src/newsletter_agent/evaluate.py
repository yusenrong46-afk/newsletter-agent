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
- the briefing team: how many planted errors the fact-checker catches and how
  many true bullets it wrongly removes, and how well the editor groups bullets
  that report the same story.

Two bundled sets are scored: "standard" (clear-cut emails) and "hard" (mixed
purposes, misleading subjects, missing headers, borderline cases). A hard case
may list other labels a careful person could give; the eval then reports
strict accuracy (the expected label only) and lenient accuracy (either).

--runs N repeats everything N times, because a model does not give the same
answers every time. The result then shows the mean and the range, and names
the emails whose answer changed between runs.

No real email is involved: the mailbox is SyntheticGmail, and --real-model
refuses any dataset except the bundled one.
"""

import asyncio
import json
import statistics
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from html import escape
from importlib import resources
from pathlib import Path

from . import config
from .briefing import Budget
from .briefing_team import BriefingState, Bullet, TeamContext, edit, fact_check
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
TEAM_MAX_MODEL_CALLS = 10
TEAM_MAX_TOKENS = 150_000
MAX_RUNS = 10
BUNDLED_SETS = {"standard": "dataset.jsonl", "hard": "dataset_hard.jsonl"}


class DatasetError(ValueError):
    pass


def bundled_dataset_text(name: str = "standard") -> str:
    return (resources.files("newsletter_agent") / "eval_data" / BUNDLED_SETS[name]).read_text(encoding="utf-8")


def load_dataset(path: Path | None = None, name: str = "standard") -> list[dict]:
    """Read and validate a JSONL dataset. Without a path, one of the bundled synthetic sets."""
    text = path.read_text(encoding="utf-8") if path else bundled_dataset_text(name)
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
        also = item.get("also_acceptable", [])
        if not isinstance(also, list) or any(label not in ALL_CATEGORIES for label in also):
            raise DatasetError(f"Line {number}: also_acceptable must be a list of labels from "
                               f"{', '.join(ALL_CATEGORIES)}.")
        if item["expected"] in also:
            raise DatasetError(f"Line {number}: also_acceptable repeats the expected label {item['expected']!r}.")
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


def is_acceptable(item: dict, answer: str) -> bool:
    """The expected label, or another label the dataset says a careful person could give."""
    return answer == item["expected"] or answer in item.get("also_acceptable", [])


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
    lenient = sum(1 for item in reached_model if is_acceptable(item, outcome[item["id"]]))

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
         "predicted": outcome[item["id"]], "note": item.get("note", ""),
         # True when the answer is not the expected label but is one a careful person could give.
         "acceptable": is_acceptable(item, outcome[item["id"]])}
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
        "lenient_category_accuracy": ratio(lenient, len(reached_model)),
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
        "clear_mistakes": sum(1 for mistake in mistakes if not mistake["acceptable"]),
        "answers": {item["id"]: outcome[item["id"]] for item in cases},
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


# --- Repeated runs --------------------------------------------------------------------------

def spread(values: list) -> dict | None:
    """Mean, lowest and highest of the values that exist (None means a run had nothing to score)."""
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return {"mean": round(statistics.fmean(numbers), 4), "min": min(numbers), "max": max(numbers),
            "values": numbers}


def aggregate_runs(cases: list[dict], results: list[dict]) -> dict:
    """Summarize several runs on the same emails: ranges, and which emails changed answer."""
    flaky, always_wrong, always_right = [], [], 0
    for item in cases:
        answers = Counter(result["answers"][item["id"]] for result in results)
        if set(answers) == {"not_sent"} and item["expected"] == "personal_or_other":
            always_right += 1  # Kept from the model, which is the right outcome for personal mail.
            continue
        right = sum(count for answer, count in answers.items() if is_acceptable(item, answer))
        entry = {"id": item["id"], "subject": item["subject"], "expected": item["expected"],
                 "also_acceptable": item.get("also_acceptable", []), "answers": dict(answers),
                 "acceptable_runs": right, "note": item.get("note", "")}
        if len(answers) > 1:
            flaky.append(entry)  # The model gave different answers on different runs.
        elif right == 0:
            always_wrong.append(entry)
        else:
            always_right += 1
    return {
        "runs": len(results), "complete_runs": sum(1 for result in results if result["complete"]),
        "cases": len(cases),
        "category_accuracy": spread([result["category_accuracy"] for result in results]),
        "lenient_category_accuracy": spread([result["lenient_category_accuracy"] for result in results]),
        "subscription_f1": spread([result["subscription_detection"]["f1"] for result in results]),
        "macro_f1": spread([result["macro_f1"] for result in results]),
        "unresolved_rate": spread([result["unresolved_rate"] for result in results]),
        "subscriptions_reaching_model": results[0]["triage"]["subscriptions_reaching_model"],
        "total_tokens": sum(result["usage"]["total_tokens"] for result in results),
        "stable_right": always_right, "always_wrong": always_wrong, "changed_between_runs": flaky,
    }


def aggregate_team(teams: list[dict]) -> dict:
    return {
        "runs": len(teams),
        "complete_runs": sum(1 for team in teams
                             if team["fact_check"]["status"] == team["editor"]["status"] == "done"),
        "catch_rate": spread([team["fact_check"]["catch_rate"] for team in teams]),
        "wrongly_removed_rate": spread([team["fact_check"]["wrongly_removed_rate"] for team in teams]),
        "grouping_f1": spread([team["editor"]["f1"] for team in teams]),
        "total_tokens": sum(team["usage"]["total_tokens"] for team in teams),
    }


def estimate_run_tokens(cases: list[dict]) -> int:
    """A rough upper estimate of one classification run, for the approval question."""
    text = sum(len((item.get("body") or item.get("html") or "")[:2000]) + len(item["subject"]) + 200 for item in cases)
    return text // 4 + 80 * len(cases) + 600 * (len(cases) // 25 + 1)


# --- The briefing team ------------------------------------------------------------------

def load_team_cases() -> dict:
    text = (resources.files("newsletter_agent") / "eval_data" / "briefing_team.json").read_text(encoding="utf-8")
    return json.loads(text)


def score_fact_check(expected: dict[str, bool], verdicts: dict[str, str]) -> dict:
    """expected: bullet ID -> is it supported by its text. verdicts: bullet ID -> the checker's verdict."""
    planted = [bullet_id for bullet_id, supported in expected.items() if not supported]
    true = [bullet_id for bullet_id, supported in expected.items() if supported]
    caught = sum(1 for bullet_id in planted if verdicts[bullet_id] == "unsupported")
    wrongly_removed = sum(1 for bullet_id in true if verdicts[bullet_id] == "unsupported")
    return {
        "bullets": len(expected), "planted_errors": len(planted), "errors_caught": caught,
        "errors_missed": len(planted) - caught, "catch_rate": ratio(caught, len(planted)),
        "true_bullets": len(true), "true_bullets_wrongly_removed": wrongly_removed,
        "wrongly_removed_rate": ratio(wrongly_removed, len(true)),
        "unchecked": sum(1 for verdict in verdicts.values() if verdict == "unchecked"),
    }


def score_grouping(labels: dict[str, str | None], stories: list[list[str]]) -> dict:
    """Pairwise scoring: two bullets are a 'pair' if they report the same story.

    labels: bullet ID -> story label (None for a story only one newsletter covered).
    stories: the editor's top stories, each a list of bullet IDs.
    """
    ids = sorted(labels)
    story_of = {bullet_id: index for index, story in enumerate(stories) for bullet_id in story}
    true_pairs = predicted_pairs = both = 0
    for position, first in enumerate(ids):
        for second in ids[position + 1:]:
            same_truth = labels[first] is not None and labels[first] == labels[second]
            same_predicted = first in story_of and story_of.get(first) == story_of.get(second)
            true_pairs += same_truth
            predicted_pairs += same_predicted
            both += same_truth and same_predicted
    precision, recall = ratio(both, predicted_pairs), ratio(both, true_pairs)
    shared = {label for label in labels.values() if label is not None}
    found = sum(1 for label in shared
                if any({b for b in story} == {b for b, value in labels.items() if value == label} for story in stories))
    return {
        "bullets": len(ids), "shared_stories": len(shared), "shared_stories_found_exactly": found,
        "true_pairs": true_pairs, "predicted_pairs": predicted_pairs, "correct_pairs": both,
        "precision": precision, "recall": recall,
        "f1": 0.0 if true_pairs and not predicted_pairs else f1(precision, recall),
        "top_stories": len(stories),
    }


async def run_team_cases(cases: dict, model: ModelClient, max_model_calls: int, max_tokens: int) -> dict:
    """Run the real fact_check and edit roles on prepared bullets, then score them."""
    budget = Budget(max_model_calls, max_tokens)
    context = TeamContext(gmail=None, model=model, budget=budget, settings={})

    issues = [{"message_id": case["id"], "publication": case["publication"], "subject": case["subject"]}
              for case in cases["fact_check"]]
    checking = BriefingState(issues=issues, texts={case["id"]: case["text"] for case in cases["fact_check"]})
    expected: dict[str, bool] = {}
    for case in cases["fact_check"]:
        for bullet in case["bullets"]:
            bullet_id = f"b{len(expected) + 1}"
            expected[bullet_id] = bullet["supported"]
            checking.bullets.append(Bullet(id=bullet_id, issue_id=case["id"], text=bullet["text"]))
            checking.summary.bullets.setdefault(case["id"], []).append(bullet["text"])
    await fact_check(checking, context)

    editing = BriefingState(issues=[
        {"message_id": f"e{index}", "publication": item["publication"], "subject": ""}
        for index, item in enumerate(cases["editor"], start=1)])
    labels: dict[str, str | None] = {}
    for index, item in enumerate(cases["editor"], start=1):
        labels[f"b{index}"] = item["story"]
        editing.bullets.append(Bullet(id=f"b{index}", issue_id=f"e{index}", text=item["text"], verdict="supported"))
    await edit(editing, context)

    usage = {key: checking.usage["fact_check"][key] + editing.usage["editor"][key] for key in checking.usage["editor"]}
    return {
        "fact_check": {**score_fact_check(expected, {bullet.id: bullet.verdict for bullet in checking.bullets}),
                       "status": checking.status["fact_check"]},
        "editor": {**score_grouping(labels, [[bullet["bullet_id"] for bullet in story["bullets"]]
                                             for story in editing.top_stories]),
                   "status": editing.status["editor"],
                   "headlines": [story["headline"] for story in editing.top_stories]},
        "usage": usage,
    }


def attach_team(result: dict, team: dict) -> dict:
    """Add the team's scores to a result. An unfinished role makes the whole eval incomplete.

    A role that was refused, cut off, failed or hit its budget leaves bullets unchecked or
    stories unchosen, so its numbers are not a measurement of the model and must not look like one.
    """
    result["briefing_team"] = team
    unfinished = [f"{name} ({team[role]['status']})"
                  for role, name in (("fact_check", "fact-checker"), ("editor", "editor"))
                  if team[role]["status"] != "done"]
    if unfinished:
        reason = (f"The briefing-team eval did not finish: {', '.join(unfinished)}; "
                  "its scores are not a full measurement.")
        result["incomplete_reason"] = " ".join(filter(None, [result.get("incomplete_reason"), reason]))
        result["complete"] = False
    return result


def run_team_eval(model_factory: Callable[[], ModelClient], max_model_calls: int = TEAM_MAX_MODEL_CALLS,
                  max_tokens: int = TEAM_MAX_TOKENS) -> dict:
    return asyncio.run(run_team_cases(load_team_cases(), model_factory(), max_model_calls, max_tokens))


# --- Output -----------------------------------------------------------------------

def range_text(values: dict | None) -> str:
    """'97.9% (95.7%–100.0%)': the mean, then the lowest and highest run."""
    if not values:
        return "n/a"
    if values["min"] == values["max"]:
        return percent(values["mean"])
    return f"{percent(values['mean'])} ({percent(values['min'])}–{percent(values['max'])})"


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
{render_hard_html(result.get('hard'))}
{render_team_html(result.get('briefing_team'))}
{render_repeated_html(result.get('repeated'))}
<p class="muted small">"Not sent" means the privacy triage kept the email from the model. For personal email that is
the right outcome and is not counted as a mistake.</p>
"""
    return page("Classification quality", body)


def render_hard_html(hard: dict | None) -> str:
    if not hard:
        return ""
    clear = [mistake for mistake in hard["mistakes"] if not mistake["acceptable"]]
    rows = "".join(
        f"<li>{escape(item['subject'] or '(no subject)')}: expected {escape(CATEGORY_LABELS[item['expected']])}, got "
        f"{escape(CATEGORY_LABELS.get(item['predicted'], item['predicted'].replace('_', ' ')))}"
        f"{' · ' + escape(item['note']) if item['note'] else ''}</li>" for item in clear) or "<li>None.</li>"
    return f"""
<h2>Hard set</h2>
<div class="cards">
  {card(percent(hard['category_accuracy']), 'category accuracy, expected label only')}
  {card(percent(hard['lenient_category_accuracy']), 'counting acceptable alternatives')}
  {card(percent(hard['subscription_detection']['f1']), 'subscription detection F1')}
  {card(percent(hard['triage']['subscriptions_reaching_model']), 'subscriptions reaching the model')}
  {card(hard['clear_mistakes'], 'clear mistakes')}
</div>
<p class="muted small">{hard['cases']} invented emails chosen to be difficult: mixed purposes, misleading subjects,
missing headers, and borderline cases. Where a careful person could give another label, that label also counts in
the lenient score. A real newsletter that the privacy rules never send to the model is counted as a clear mistake.</p>
<h2>Clear mistakes on the hard set (this run)</h2>
<ul class="notes">{rows}</ul>
"""


def render_repeated_html(repeated: dict | None) -> str:
    if not repeated or repeated["runs"] < 2:
        return ""
    rows = "".join(
        f"<tr><td>{escape(name)}</td><td class=\"num\">{summary['cases']}</td>"
        f"<td class=\"num\">{range_text(summary['category_accuracy'])}</td>"
        f"<td class=\"num\">{range_text(summary['lenient_category_accuracy'])}</td>"
        f"<td class=\"num\">{range_text(summary['subscription_f1'])}</td>"
        f"<td class=\"num\">{len(summary['changed_between_runs'])}</td>"
        f"<td class=\"num\">{len(summary['always_wrong'])}</td></tr>"
        for name, summary in repeated["sets"].items())
    changed = "".join(
        f"<li>{escape(entry['subject'] or '(no subject)')} ({escape(name)} set): expected "
        f"{escape(CATEGORY_LABELS[entry['expected']])}; answers "
        + escape(", ".join(f"{answer.replace('_', ' ')} ×{count}"
                           for answer, count in sorted(entry["answers"].items())))
        + "</li>"
        for name, summary in repeated["sets"].items() for entry in summary["changed_between_runs"]) or "<li>None.</li>"
    team = repeated["briefing_team"]
    return f"""
<h2>Repeated runs ({repeated['runs']} runs)</h2>
<p class="muted small">A model does not give the same answers every time. Each figure is the mean over
{repeated['runs']} runs, with the lowest and highest run in brackets.</p>
<div class="table-wrap"><table><thead><tr><th>Set</th><th class="num">Emails</th><th class="num">Category accuracy</th>
<th class="num">Lenient</th><th class="num">Subscription F1</th><th class="num">Changed answer</th>
<th class="num">Always wrong</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="muted small">Briefing team: planted errors caught {range_text(team['catch_rate'])}; true bullets wrongly
removed {range_text(team['wrongly_removed_rate'])}; same-story grouping F1 {range_text(team['grouping_f1'])}.</p>
<h2>Emails whose answer changed between runs</h2>
<ul class="notes">{changed}</ul>
"""


def render_team_html(team: dict | None) -> str:
    if not team:
        return ""
    check, editor = team["fact_check"], team["editor"]
    headlines = "".join(f"<li>{escape(headline)}</li>" for headline in editor["headlines"]) or "<li>None.</li>"
    unfinished = ("" if check["status"] == editor["status"] == "done" else
                  " <strong>A role did not finish, so these scores are not a full measurement.</strong>")
    return f"""
<h2>Briefing team</h2>
<div class="cards">
  {card(f"{check['errors_caught']} / {check['planted_errors']}", 'planted errors caught by the fact-checker')}
  {card(f"{check['true_bullets_wrongly_removed']} / {check['true_bullets']}", 'true bullets wrongly removed')}
  {card(percent(editor['f1']), 'editor: same-story grouping F1')}
  {card(f"{editor['shared_stories_found_exactly']} / {editor['shared_stories']}", 'shared stories grouped exactly')}
  {card(f"{team['usage']['total_tokens']:,}", 'tokens for both roles')}
</div>
<p class="muted small">The fact-checker is given {check['bullets']} bullets, {check['planted_errors']} with a planted
error. The editor is given {editor['bullets']} bullets in which {editor['shared_stories']} stories are covered by
more than one newsletter; grouping is scored over pairs of bullets.
Fact-checker run: {escape(check['status'])}. Editor run: {escape(editor['status'])}.{unfinished}</p>
<h2>Top stories the editor chose</h2>
<ul class="notes">{headlines}</ul>
"""


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
    hard = result.get("hard")
    if hard:
        lines.append(
            f"Hard set ({hard['cases']} emails): category accuracy {percent(hard['category_accuracy'])} strict, "
            f"{percent(hard['lenient_category_accuracy'])} counting acceptable alternatives; subscription F1 "
            f"{percent(hard['subscription_detection']['f1'])}; "
            f"{percent(hard['triage']['subscriptions_reaching_model'])} of subscriptions reach the model")
    team = result.get("briefing_team")
    if team:
        check, editor = team["fact_check"], team["editor"]
        lines.append(
            f"Fact-checker ({check['status']}): caught {check['errors_caught']} of {check['planted_errors']} "
            f"planted errors "
            f"({percent(check['catch_rate'])}); wrongly removed {check['true_bullets_wrongly_removed']} of "
            f"{check['true_bullets']} true bullets; {check['unchecked']} unchecked")
        lines.append(
            f"Editor ({editor['status']}): grouping F1 {percent(editor['f1'])} "
            f"(precision {percent(editor['precision'])}, "
            f"recall {percent(editor['recall'])}); {editor['shared_stories_found_exactly']} of "
            f"{editor['shared_stories']} shared stories grouped exactly")
    repeated = result.get("repeated")
    if repeated and repeated["runs"] > 1:
        for name, summary in repeated["sets"].items():
            lines.append(
                f"Over {repeated['runs']} runs, {name} set: category accuracy "
                f"{range_text(summary['category_accuracy'])}"
                f" strict, {range_text(summary['lenient_category_accuracy'])} lenient; subscription F1 "
                f"{range_text(summary['subscription_f1'])}; {len(summary['changed_between_runs'])} of "
                f"{summary['cases']} emails changed answer between runs, {len(summary['always_wrong'])} always wrong")
        team_summary = repeated["briefing_team"]
        lines.append(
            f"Over {repeated['runs']} runs, briefing team: errors caught {range_text(team_summary['catch_rate'])}; "
            f"true bullets wrongly removed {range_text(team_summary['wrongly_removed_rate'])}; grouping F1 "
            f"{range_text(team_summary['grouping_f1'])}")
    if not result["complete"]:
        lines.append(f"Incomplete: {result['incomplete_reason']}")
    return lines


def save_result(result: dict, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    runs = result.get("repeated", {}).get("runs", 1)
    # A multi-run result gets its own name, so it never replaces a single run saved the same day.
    stem = f"{result['label']}-{result['generated_at'][:10]}" + (f"-{runs}runs" if runs > 1 else "")
    json_path, html_path = directory / f"{stem}.json", directory / f"{stem}.html"
    config.write_private_file(json_path, json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    config.write_private_file(html_path, render_eval_html(result))
    return [json_path, html_path]


def run_eval_command(real_model: bool = False, dataset: str | None = None, output: str | None = None,
                     approved: bool = False, runs: int = 1) -> int:
    if real_model and dataset:
        raise ValueError("The real-model eval only runs on the bundled synthetic dataset, "
                         "so no other email can be sent to OpenAI. Drop --dataset or --real-model.")
    if not 1 <= runs <= MAX_RUNS:
        raise ValueError(f"--runs must be between 1 and {MAX_RUNS}.")
    if dataset:
        sets = {"custom": load_dataset(Path(dataset))}
    else:
        sets = {name: load_dataset(name=name) for name in BUNDLED_SETS}
    directory = Path(output) if output else config.private_dir("evals")

    model_factory: Callable[[], ModelClient]
    if real_model:
        from .briefing import ask_yes_no
        from .classify import make_openai_client

        estimate = runs * (sum(estimate_run_tokens(cases) for cases in sets.values()) + 5_000)
        ceiling = runs * (len(sets) * EVAL_MAX_TOKENS + TEAM_MAX_TOKENS)
        question = (
            f"Send the bundled invented emails to OpenAI ({MODEL}): {runs} run(s) of "
            f"{' + '.join(str(len(cases)) for cases in sets.values())} emails plus the briefing-team cases? "
            f"Roughly {estimate:,} tokens in total; hard limit {ceiling:,}.")
        if not approved and not ask_yes_no(question):
            # Nothing was sent, so there is nothing to score or save (and no earlier result to overwrite).
            if sys.stdin.isatty():
                print("Cancelled. Nothing was sent to OpenAI and no result was saved.")
                return 0
            print("Not interactive: re-run with --yes to approve sending the synthetic emails to OpenAI.")
            return 2
        model_factory, label = make_openai_client, MODEL
    else:
        model = OfflineModel()
        model_factory, label = (lambda: model), "offline"

    per_set: dict[str, list[dict]] = {name: [] for name in sets}
    teams = []
    for number in range(1, runs + 1):
        if runs > 1:
            print(f"Run {number} of {runs}", file=sys.stderr)
        for name, cases in sets.items():
            per_set[name].append(run_eval(cases, model_factory, label))
        teams.append(run_team_eval(model_factory))

    first = next(iter(sets))
    result = per_set[first][0]  # The first run in full, so the saved file keeps its shape.
    if "hard" in per_set:
        result["hard"] = per_set["hard"][0]
    attach_team(result, teams[0])
    result["repeated"] = {
        "runs": runs,
        "sets": {name: aggregate_runs(cases, per_set[name]) for name, cases in sets.items()},
        "briefing_team": aggregate_team(teams),
    }
    unfinished = [f"{name} set: {summary['runs'] - summary['complete_runs']} of {summary['runs']} runs"
                  for name, summary in result["repeated"]["sets"].items() if summary["complete_runs"] < summary["runs"]]
    team_summary = result["repeated"]["briefing_team"]
    if team_summary["complete_runs"] < runs and result["complete"]:
        unfinished.append(f"briefing team: {runs - team_summary['complete_runs']} of {runs} runs")
    if unfinished and result["complete"]:
        result["complete"] = False
        result["incomplete_reason"] = f"Some runs did not finish ({'; '.join(unfinished)}), so the ranges are partial."

    for line in summary_lines(result):
        print(line)
    for path in save_result(result, directory):
        print(f"Saved: {path}", file=sys.stderr)
    return 0 if result["complete"] else 1
