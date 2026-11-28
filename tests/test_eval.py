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


def test_a_declined_real_model_eval_saves_nothing_and_keeps_earlier_results(tmp_path, monkeypatch, capsys):
    from newsletter_agent import classify

    def no_client():
        raise AssertionError("A declined eval must not create an OpenAI client.")

    monkeypatch.setattr(classify, "make_openai_client", no_client)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    earlier = tmp_path / "gpt-6-luna-earlier.json"
    earlier.write_text("{}", encoding="utf-8")

    assert run_eval_command(real_model=True, output=str(tmp_path)) == 2  # Not approved, not interactive.
    assert "--yes" in capsys.readouterr().out
    assert [path.name for path in tmp_path.iterdir()] == ["gpt-6-luna-earlier.json"]  # No zero-score "result".


def test_a_category_the_model_never_predicts_counts_as_zero_in_macro_f1():
    """From the PR review: one right newsletter, one receipt wrongly called a newsletter."""
    cases = [case("a", "editorial_newsletter"), case("b", "transactional")]
    outcome = {"a": "editorial_newsletter", "b": "editorial_newsletter"}
    result = score(cases, outcome, {"a": "bulk", "b": "bulk"})
    transactional = result["per_category"]["transactional"]
    assert (transactional["precision"], transactional["recall"], transactional["f1"]) == (None, 0.0, 0.0)
    assert result["per_category"]["editorial_newsletter"]["f1"] == 0.6667
    assert result["macro_f1"] == 0.3333  # Was 0.6667: the failed category was silently left out.
    assert result["per_category"]["uncertain"]["f1"] is None  # No examples: still not scored.


# --- The briefing team's scores ----------------------------------------------------------------

def test_fact_check_scores_match_a_hand_computed_example():
    from newsletter_agent.evaluate import score_fact_check

    expected = {"b1": True, "b2": True, "b3": False, "b4": False, "b5": False}
    verdicts = {"b1": "supported", "b2": "unsupported", "b3": "unsupported", "b4": "supported", "b5": "unchecked"}
    result = score_fact_check(expected, verdicts)
    assert (result["errors_caught"], result["errors_missed"], result["catch_rate"]) == (1, 2, 0.3333)
    assert (result["true_bullets_wrongly_removed"], result["wrongly_removed_rate"]) == (1, 0.5)
    assert result["unchecked"] == 1  # An unchecked planted error counts as missed, never as caught.


def test_grouping_scores_match_a_hand_computed_example():
    from newsletter_agent.evaluate import score_grouping

    labels = {"b1": "x", "b2": "x", "b3": "x", "b4": "y", "b5": "y", "b6": None}
    # The editor grouped two of the three "x" bullets, and wrongly put a "y" bullet with a single.
    result = score_grouping(labels, [["b1", "b2"], ["b4", "b6"], ["b3"]])
    assert (result["true_pairs"], result["predicted_pairs"], result["correct_pairs"]) == (4, 2, 1)
    assert (result["precision"], result["recall"], result["f1"]) == (0.5, 0.25, 0.3333)
    assert result["shared_stories_found_exactly"] == 0
    assert score_grouping(labels, [])["f1"] == 0.0  # Grouping nothing is a failure, not "n/a".
    assert score_grouping(labels, [["b1", "b2", "b3"], ["b4", "b5"]])["shared_stories_found_exactly"] == 2


def test_the_team_eval_runs_the_real_roles_and_offline_scores_do_not_regress(tmp_path):
    from newsletter_agent.evaluate import load_team_cases, run_team_eval

    cases = load_team_cases()
    planted = sum(1 for case in cases["fact_check"] for bullet in case["bullets"] if not bullet["supported"])
    assert planted >= 10 and len({item["story"] for item in cases["editor"]} - {None}) == 3

    model = OfflineModel()
    result = run_team_eval(lambda: model)
    assert result["fact_check"]["status"] == "done" and result["fact_check"]["unchecked"] == 0
    assert result["fact_check"]["catch_rate"] >= 0.7 and result["fact_check"]["wrongly_removed_rate"] <= 0.15
    assert result["editor"]["status"] == "done" and result["editor"]["f1"] >= 0.8
    assert result["editor"]["precision"] == 1.0  # The stand-in never merges different stories.


def test_a_team_eval_without_budget_reports_unchecked_bullets_not_a_score_of_zero_errors():
    from newsletter_agent.evaluate import run_team_eval

    result = run_team_eval(lambda: OfflineModel(), max_model_calls=0)
    assert result["fact_check"]["status"] == "incomplete"
    assert result["fact_check"]["unchecked"] == result["fact_check"]["bullets"]
    assert result["fact_check"]["errors_caught"] == 0 and result["editor"]["status"] == "incomplete"


def test_the_eval_command_saves_the_team_scores(tmp_path):
    assert cli.main(["eval", "--output", str(tmp_path)]) == 0
    saved = json.loads(next(tmp_path.glob("offline-*.json")).read_text(encoding="utf-8"))
    assert saved["briefing_team"]["fact_check"]["planted_errors"] == 12
    assert "Briefing team" in next(tmp_path.glob("offline-*.html")).read_text(encoding="utf-8")


def test_an_unfinished_team_eval_makes_the_whole_eval_incomplete(tmp_path, monkeypatch, capsys):
    from newsletter_agent import evaluate

    real = evaluate.run_team_eval
    # One call only: the fact-checker's first batch uses it, so the rest never runs.
    monkeypatch.setattr(evaluate, "run_team_eval", lambda factory: real(factory, max_model_calls=1))

    assert cli.main(["eval", "--output", str(tmp_path)]) == 1  # Not a clean exit.
    saved = json.loads(next(tmp_path.glob("offline-*.json")).read_text(encoding="utf-8"))
    assert saved["complete"] is False and saved["briefing_team"]["editor"]["status"] == "incomplete"
    assert "editor (incomplete)" in saved["incomplete_reason"]
    assert "Editor (incomplete)" in capsys.readouterr().out
    assert "A role did not finish" in next(tmp_path.glob("offline-*.html")).read_text(encoding="utf-8")


# --- The hard set, lenient scoring, and repeated runs ---------------------------------------------

def test_the_hard_set_is_valid_synthetic_and_actually_harder():
    standard, hard = load_dataset(), load_dataset(name="hard")
    assert len(hard) >= 60 and not {item["id"] for item in hard} & {item["id"] for item in standard}
    assert sum(1 for item in hard if item["also_acceptable"]) >= 20  # Plenty of borderline cases.
    for item in hard:
        assert parseaddr(item["from"])[1].rsplit("@", 1)[1].endswith(".example"), item["from"]

    model = OfflineModel()
    easy = run_eval(standard, lambda: model, "offline")
    difficult = run_eval(hard, lambda: model, "offline")
    assert difficult["complete"] and difficult["category_accuracy"] < easy["category_accuracy"] - 0.2
    assert difficult["lenient_category_accuracy"] > difficult["category_accuracy"]
    assert difficult["triage"]["subscriptions_reaching_model"] < 1.0  # The privacy rules have a measured cost here.


def test_lenient_accuracy_counts_only_the_listed_alternatives():
    cases = [case("a", "transactional", also_acceptable=["product_update"]), case("b", "transactional"),
             case("c", "editorial_newsletter", also_acceptable=["promotional_subscription"])]
    outcome = {"a": "product_update", "b": "product_update", "c": "uncertain"}
    result = score(cases, outcome, dict.fromkeys("abc", "bulk"))
    assert (result["category_accuracy"], result["lenient_category_accuracy"]) == (0.0, 0.3333)
    assert [(m["id"], m["acceptable"]) for m in result["mistakes"]] == [("a", True), ("b", False), ("c", False)]
    assert result["clear_mistakes"] == 2


@pytest.mark.parametrize("extra, message", [
    ({"also_acceptable": ["newsletter"]}, "also_acceptable must be a list of labels"),
    ({"also_acceptable": "transactional"}, "also_acceptable must be a list of labels"),
    ({"also_acceptable": ["transactional"]}, "repeats the expected label"),
])
def test_invalid_alternative_labels_are_refused(tmp_path, extra, message):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(case("a", "transactional", **extra)), encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_dataset(path)


def test_repeated_runs_report_a_range_and_name_the_emails_that_changed():
    from newsletter_agent.evaluate import aggregate_runs, spread

    cases = [case("steady", "editorial_newsletter"), case("flaky", "transactional"),
             case("wrong", "product_update"), case("borderline", "transactional", also_acceptable=["product_update"])]
    triage = dict.fromkeys(["steady", "flaky", "wrong", "borderline"], "bulk")

    def one_run(flaky_answer):
        outcome = {"steady": "editorial_newsletter", "flaky": flaky_answer, "wrong": "uncertain",
                   "borderline": "product_update"}
        return {**score(cases, outcome, triage), "complete": True, "usage": {"total_tokens": 100}}

    summary = aggregate_runs(cases, [one_run("transactional"), one_run("uncertain"), one_run("transactional")])
    assert summary["category_accuracy"] == {"mean": 0.4167, "min": 0.25, "max": 0.5, "values": [0.5, 0.25, 0.5]}
    assert summary["lenient_category_accuracy"]["max"] == 0.75
    assert [entry["id"] for entry in summary["changed_between_runs"]] == ["flaky"]
    assert summary["changed_between_runs"][0]["answers"] == {"transactional": 2, "uncertain": 1}
    assert [entry["id"] for entry in summary["always_wrong"]] == ["wrong"]
    assert summary["stable_right"] == 2 and summary["total_tokens"] == 300  # "borderline" counts as right.
    assert spread([None, None]) is None and spread([0.5, None])["values"] == [0.5]


def test_the_eval_command_repeats_runs_and_saves_the_ranges(tmp_path, capsys):
    assert cli.main(["eval", "--runs", "3", "--output", str(tmp_path)]) == 0
    saved = json.loads(next(tmp_path.glob("offline-*.json")).read_text(encoding="utf-8"))
    repeated = saved["repeated"]
    assert repeated["runs"] == 3 and set(repeated["sets"]) == {"standard", "hard"}
    assert repeated["sets"]["hard"]["complete_runs"] == 3
    assert repeated["sets"]["standard"]["changed_between_runs"] == []  # The offline stand-in is deterministic.
    assert repeated["briefing_team"]["catch_rate"]["values"] == [0.75, 0.75, 0.75]
    assert saved["hard"]["cases"] == len(load_dataset(name="hard"))
    output = capsys.readouterr().out
    assert "Over 3 runs, hard set" in output and "Hard set (" in output
    html = next(tmp_path.glob("offline-*.html")).read_text(encoding="utf-8")
    assert "Repeated runs (3 runs)" in html and "Hard set" in html


@pytest.mark.parametrize("runs", ["0", "11"])
def test_the_number_of_runs_is_limited(tmp_path, runs, capsys):
    assert cli.main(["eval", "--runs", runs, "--output", str(tmp_path)]) == 1
    assert "--runs must be between 1 and 10" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_one_unfinished_run_makes_the_ranges_partial(tmp_path, monkeypatch):
    from newsletter_agent import evaluate

    real, calls = evaluate.run_eval, []

    def second_run_runs_out(cases, factory, label, **kwargs):
        calls.append(label)
        starve = len(calls) == 3  # Run 2 of the standard set (the call order is standard, hard, standard, hard).
        return real(cases, factory, label, max_model_calls=1 if starve else 20)

    monkeypatch.setattr(evaluate, "run_eval", second_run_runs_out)
    assert cli.main(["eval", "--runs", "2", "--output", str(tmp_path)]) == 1
    saved = json.loads(next(tmp_path.glob("offline-*.json")).read_text(encoding="utf-8"))
    assert saved["complete"] is False and "standard set: 1 of 2 runs" in saved["incomplete_reason"]
    assert saved["repeated"]["sets"]["standard"]["complete_runs"] == 1
    html = next(tmp_path.glob("offline-*.html")).read_text(encoding="utf-8")
    assert html.count('<span class="badge warn">Incomplete</span>') == 2  # At the top and beside the ranges.
    assert "standard set: 1 of 2 runs" in html and "1 of 2 runs</td>" in html and "2 of 2 runs</td>" in html


def test_a_finished_eval_page_carries_no_incomplete_warning(tmp_path):
    assert cli.main(["eval", "--runs", "2", "--output", str(tmp_path)]) == 0
    assert "Incomplete</span>" not in next(tmp_path.glob("offline-*.html")).read_text(encoding="utf-8")


def test_an_email_wrong_in_different_ways_each_run_is_both_changed_and_always_wrong():
    from newsletter_agent.evaluate import aggregate_runs

    cases = [case("varies", "editorial_newsletter"), case("fine", "transactional")]
    triage = {"varies": "bulk", "fine": "bulk"}

    def one_run(answer):
        outcome = {"varies": answer, "fine": "transactional"}
        return {**score(cases, outcome, triage), "complete": True, "usage": {"total_tokens": 1}}

    summary = aggregate_runs(cases, [one_run("transactional"), one_run("personal_or_other")])
    assert [entry["id"] for entry in summary["changed_between_runs"]] == ["varies"]
    assert [entry["id"] for entry in summary["always_wrong"]] == ["varies"]  # Never right, so it counts here too.
    assert summary["stable_right"] == 1


def test_a_multi_run_result_never_replaces_a_single_run_from_the_same_day(tmp_path):
    assert cli.main(["eval", "--output", str(tmp_path)]) == 0
    assert cli.main(["eval", "--runs", "2", "--output", str(tmp_path)]) == 0
    names = sorted(path.name for path in tmp_path.glob("*.json"))
    assert len(names) == 2 and names[0].endswith("-2runs.json") and "runs" not in names[1]


def test_retrieval_and_chat_tool_evals_score_and_save_in_eval_report(tmp_path, capsys):
    from newsletter_agent.evaluate import run_chat_tool_eval, run_retrieval_eval

    retrieval = run_retrieval_eval()
    assert retrieval["queries"] >= 14
    modes = retrieval["modes"]
    assert set(modes) == {"lexical", "vector", "hybrid"}
    assert modes["hybrid"]["recall_at_3"] >= modes["lexical"]["recall_at_3"]
    assert modes["hybrid"]["recall_at_3"] >= 0.90
    assert modes["hybrid"]["mrr"] >= 0.90

    chat_tools = run_chat_tool_eval()
    assert chat_tools["cases"] >= 10
    assert chat_tools["tool_accuracy"] >= 0.90
    assert chat_tools["valid_tool_rate"] == 1.0

    assert cli.main(["eval", "--output", str(tmp_path)]) == 0
    saved = json.loads(next(tmp_path.glob("offline-*.json")).read_text(encoding="utf-8"))
    assert "retrieval" in saved and "chat_tools" in saved
    output = capsys.readouterr().out
    assert "Retrieval (" in output and "Chat tool selection (" in output
    html = next(tmp_path.glob("offline-*.html")).read_text(encoding="utf-8")
    assert "Hybrid search retrieval" in html and "Chat agent tool selection" in html

