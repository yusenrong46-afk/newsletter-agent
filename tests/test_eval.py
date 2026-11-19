"""The quality eval: dataset checks, scoring arithmetic, and how bad model answers are counted.

Red test (fails on v0.1, which had no eval): `newsletter-agent eval` scores the
bundled dataset and saves results. Failure-mode tests: invalid datasets,
invented or missing IDs, budget stops, refusals, and the rule that the real
model only ever sees the bundled synthetic emails.
"""

import json
from email.utils import parseaddr
from types import SimpleNamespace

import pytest

from helpers import FakeOpenAI, fake_response
from newsletter_agent import cli
from newsletter_agent.classify import ClassificationBatch, MessageClassification
from newsletter_agent.demo import OfflineModel
from newsletter_agent.evaluate import DatasetError, load_dataset, run_eval, run_eval_command, score


def case(case_id, expected, days_ago=1, **extra):
    return {"id": case_id, "from": "A <a@x.example>", "subject": "S", "expected": expected,
            "days_ago": days_ago, "headers": {"List-Id": "<x.example>"}, **extra}


def offline_eval(cases=None):
    model = OfflineModel()
    return run_eval(cases or load_dataset(), lambda: model, "offline")


# --- Red test -----------------------------------------------------------------------

def test_the_eval_command_scores_the_bundled_dataset_and_saves_results(tmp_path, capsys):
    assert cli.main(["eval", "--output", str(tmp_path)]) == 0
    saved = json.loads(next(tmp_path.glob("offline-*.json")).read_text(encoding="utf-8"))
    assert saved["cases"] == len(load_dataset())
    assert saved["complete"] is True
    assert next(tmp_path.glob("offline-*.html")).read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "Subscription detection: F1" in capsys.readouterr().out


# --- The bundled dataset ---------------------------------------------------------------

def test_the_bundled_dataset_is_valid_varied_and_entirely_synthetic():
    cases = load_dataset()
    assert len(cases) >= 90
    assert {item["expected"] for item in cases} >= {"editorial_newsletter", "promotional_subscription",
                                                     "product_update", "transactional", "personal_or_other"}
    for item in cases:  # Every sender is on a reserved .example domain: no real person or company.
        domain = parseaddr(item["from"])[1].rsplit("@", 1)[1]
        assert domain.endswith(".example"), item["from"]


def test_offline_baseline_scores_do_not_regress():
    result = offline_eval()
    assert result["complete"]
    assert result["subscription_detection"]["f1"] >= 0.90
    assert result["category_accuracy"] >= 0.70
    assert result["triage"]["subscriptions_reaching_model"] == 1.0  # Privacy rules never hide a real subscription.
    assert result["triage"]["personal_kept_from_model_or_headers_only"] >= 0.85


# --- Scoring arithmetic --------------------------------------------------------------

def test_scores_match_a_hand_computed_example():
    cases = [case("a", "editorial_newsletter"), case("b", "editorial_newsletter"),
             case("c", "promotional_subscription"), case("d", "transactional"), case("e", "personal_or_other")]
    outcome = {"a": "editorial_newsletter", "b": "promotional_subscription", "c": "promotional_subscription",
               "d": "editorial_newsletter", "e": "not_sent"}
    triage = {"a": "bulk", "b": "bulk", "c": "bulk", "d": "bulk", "e": "not_candidate"}
    result = score(cases, outcome, triage)

    detection = result["subscription_detection"]
    assert (detection["true_positive"], detection["false_positive"],
            detection["false_negative"], detection["true_negative"]) == (3, 1, 0, 1)
    assert detection["precision"] == 0.75 and detection["recall"] == 1.0
    assert detection["f1"] == round(2 * 0.75 / 1.75, 4)
    editorial = result["per_category"]["editorial_newsletter"]
    assert (editorial["precision"], editorial["recall"]) == (0.5, 0.5)
    promotional = result["per_category"]["promotional_subscription"]
    assert (promotional["precision"], promotional["recall"]) == (0.5, 1.0)
    assert result["category_accuracy"] == 0.5  # 2 of the 4 emails that reached the model.
    assert result["confusion"]["transactional"]["editorial_newsletter"] == 1
    assert [item["id"] for item in result["mistakes"]] == ["b", "d"]  # Personal + not_sent is right, not a mistake.


# --- Dataset validation ------------------------------------------------------------------

@pytest.mark.parametrize("lines, message", [
    ([case("a", "newsletter")], "unknown label"),
    ([case("a", "transactional"), case("a", "transactional")], "duplicate id"),
    ([case("a", "transactional", days_ago=120)], "days_ago"),
    ([{"id": "a", "from": "x"}], "missing"),
])
def test_invalid_datasets_are_refused_with_a_clear_reason(tmp_path, lines, message):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines))
    with pytest.raises(DatasetError, match=message):
        load_dataset(path)


def test_the_real_model_never_runs_on_a_custom_dataset(tmp_path):
    path = tmp_path / "mine.jsonl"
    path.write_text(json.dumps(case("a", "transactional")))
    with pytest.raises(ValueError, match="only runs on the bundled synthetic dataset"):
        run_eval_command(real_model=True, dataset=str(path), output=str(tmp_path))


# --- Bad model answers are counted, never hidden -------------------------------------------

CASES = [case(f"n{i}", "editorial_newsletter") for i in range(4)] + [case("t0", "transactional")]


def test_invented_and_missing_ids_count_as_unresolved_and_cost_recall():
    def sloppy(items):
        kept = [MessageClassification(message_id=item["id"], classification="editorial_newsletter",
                                      publication_name=None, confidence="high", rationale="x")
                for item in items if item["id"] not in ("n0", "n1")]
        kept.append(MessageClassification(message_id="invented", classification="editorial_newsletter",
                                          publication_name=None, confidence="high", rationale="x"))
        return ClassificationBatch(classifications=kept)

    fake = FakeOpenAI(sloppy)
    result = run_eval(CASES, lambda: fake, "test")
    assert result["confusion"]["editorial_newsletter"]["unresolved"] == 2
    assert result["subscription_detection"]["recall"] == 0.5
    assert result["unresolved_rate"] == 0.4
    assert not result["complete"]


def test_a_budget_stop_marks_the_eval_incomplete():
    cases = [case(f"n{i:02}", "editorial_newsletter") for i in range(30)]
    result = run_eval(cases, lambda: FakeOpenAI(), "test", max_model_calls=1)
    assert result["status"] == "partial_budget"
    assert not result["complete"] and "incomplete_reason" in result
    assert result["confusion"]["editorial_newsletter"]["pending"] == 5


def test_refusals_and_incomplete_answers_count_as_unresolved():
    refusal = SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal")])

    def refuse(items):
        return fake_response(None, output=[refusal])

    result = run_eval(CASES, lambda: FakeOpenAI(refuse), "test")
    assert result["unresolved_rate"] == 1.0
    assert result["subscription_detection"]["true_positive"] == 0
