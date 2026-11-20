"""Cleanup suggestions and weekly trends: plain arithmetic, conservative rules, safe output.

Red test (fails on v0.1): reports carry cleanup suggestions and weekly counts.
Failure-mode tests: who must never be suggested, stale unread flags, ties,
partial weeks, a broken weekly total, hostile publication names, and URLs.
"""

from helpers import DAY_MS, NOW_MS, fake_gmail, newsletter, run_workflow
from newsletter_agent.chat import get_inventory_overview
from newsletter_agent.demo import SyntheticGmail, http_error
from newsletter_agent.html_report import render_html_report
from newsletter_agent.report import check_reconciliation, cleanup_advice, render_report, trend, weekly_counts

WINDOW = (NOW_MS - 90 * DAY_MS, NOW_MS)


def unread_newsletters(prefix: str, count: int, unread: int, list_id: str, **kwargs) -> list[dict]:
    return [newsletter(f"{prefix}{i}", list_id=list_id, when=NOW_MS - (i + 1) * 5 * DAY_MS,
                       labels=("INBOX", "UNREAD") if i < unread else ("INBOX",), **kwargs)
            for i in range(count)]


def group(name="Weekly", status="observed", category="editorial_newsletter", received=10, unread=9, key=None):
    return {"group_key": key or f"list:{name.lower()}", "name": name, "status": status, "category": category,
            "received_count": received, "unread_count": unread, "has_unsubscribe_header": True}


# --- Red test -------------------------------------------------------------------------

def test_reports_carry_cleanup_suggestions_and_weekly_counts(tmp_path):
    mail = unread_newsletters("a", 8, 8, "ignored.example") + unread_newsletters("b", 6, 1, "read.example")
    report, _ = run_workflow(tmp_path, fake_gmail(mail))

    advice = report["cleanup_advice"]
    assert advice["available"] is True
    assert [item["name"] for item in advice["suggestions"]] == ["Ignored"]
    suggestion = advice["suggestions"][0]
    assert suggestion["action"] == "consider_unsubscribing"
    assert suggestion["reason"] == "8 received, 8 still unread (100%)."
    for item in report["publications_observed"]:
        assert sum(item["weekly_counts"]) == item["received_count"]
        assert len(item["weekly_counts"]) == 13 and item["trend"] in {"new", "rising", "steady", "falling", "fading"}
    assert report["run"]["weeks"] == 13 and report["run"]["last_week_days"] == 6.0  # 90 days = 12 weeks + 6 days.
    assert report["reconciliation"]["ok"]
    assert "## Cleanup suggestions" in render_report(report)
    assert "Consider unsubscribing" in render_html_report(report)
    assert get_inventory_overview(report)["cleanup_suggestions"][0]["name"] == "Ignored"


# --- Who is never suggested --------------------------------------------------------------

def test_only_well_evidenced_mostly_unread_publications_are_suggested():
    groups = [
        group("Kept", received=10, unread=9),
        group("Provisional", status="provisional", received=10, unread=10),
        group("Receipts", category="transactional", received=10, unread=10),
        group("Personal", category="personal_or_other", received=10, unread=10),
        group("Too few", received=3, unread=3),
        group("Mostly read", received=10, unread=4),
        group("Half unread", received=10, unread=5),
    ]
    suggestions = cleanup_advice(groups, unread_flags_fresh=True)["suggestions"]
    assert [(item["name"], item["action"]) for item in suggestions] == [
        ("Kept", "consider_unsubscribing"), ("Half unread", "skim_or_reduce")]


def test_stale_unread_flags_turn_suggestions_off_with_a_note(tmp_path):
    class NoUnreadRefresh(SyntheticGmail):
        def list(self, userId, q, pageToken, maxResults, includeSpamTrash):
            if "is:unread" in q:
                self.list_error = http_error(500)
            else:
                self.list_error = None
            return super().list(userId, q, pageToken, maxResults, includeSpamTrash)

    report, _ = run_workflow(tmp_path, NoUnreadRefresh(unread_newsletters("a", 8, 8, "x.example"), page_size=2))
    assert report["cleanup_advice"]["available"] is False
    assert report["cleanup_advice"]["suggestions"] == []
    assert any("Cleanup suggestions are off" in note for note in report["notes"])
    assert "Unread flags could not be refreshed" in render_html_report(report)


def test_ties_are_ordered_the_same_way_every_time():
    groups = [group("beta", key="list:b"), group("Alpha", key="list:a"), group("alpha", key="list:a2")]
    names = [item["group_key"] for item in cleanup_advice(groups, True)["suggestions"]]
    assert names == ["list:a", "list:a2", "list:b"]
    assert names == [item["group_key"] for item in cleanup_advice(list(reversed(groups)), True)["suggestions"]]


# --- Weekly counts and trends ----------------------------------------------------------------

def test_weekly_bins_cover_a_window_that_is_not_whole_weeks():
    dates = [WINDOW[0], WINDOW[0] + 7 * DAY_MS - 1, WINDOW[0] + 7 * DAY_MS, WINDOW[1] - 1]
    counts = weekly_counts(dates, WINDOW)
    assert len(counts) == 13
    assert counts[0] == 2 and counts[1] == 1 and counts[-1] == 1
    assert sum(counts) == len(dates)


def test_a_wrong_weekly_total_is_reported_not_hidden():
    counts = {"discovered": 1, "in_scope": 1, "excluded": 0, "unavailable": 0, "candidates": 1, "not_candidates": 0,
              "classified": 1, "unresolved": 0, "not_yet_classified": 0}
    categories = {"editorial_newsletter": 1, "promotional_subscription": 0, "product_update": 0,
                  "transactional": 0, "personal_or_other": 0, "uncertain": 0}
    broken = {"group_key": "list:x", "received_count": 1, "weekly_counts": [0, 2]}
    assert check_reconciliation(counts, categories, [broken]) == ["weekly counts != received for list:x"]


def days_ago(*days):
    return [NOW_MS - int(day * DAY_MS) for day in days]


def test_trends_from_dates():
    assert trend(days_ago(2, 5, 9), WINDOW) == "new"
    assert trend(days_ago(40, 50, 60, 70), WINDOW) == "fading"
    assert trend(days_ago(5, 10, 15, 20, 25, 30, 70), WINDOW) == "rising"
    assert trend(days_ago(10, 50, 55, 60, 65, 70, 80), WINDOW) == "falling"
    assert trend(days_ago(5, 20, 35, 50, 65, 80), WINDOW) == "steady"
    assert trend(days_ago(1), (NOW_MS - 2 * DAY_MS, NOW_MS)) is None  # A briefing window is too short.


# --- Safe output ------------------------------------------------------------------------------

def test_hostile_publication_names_are_escaped_and_advice_has_no_links(tmp_path):
    def hostile(items):
        from newsletter_agent.classify import ClassificationBatch, MessageClassification

        return ClassificationBatch(classifications=[
            MessageClassification(message_id=item["id"], classification="editorial_newsletter",
                                  publication_name='<script>alert("x")</script> https://evil.example/u',
                                  confidence="high", rationale="x")
            for item in items])

    from helpers import FakeOpenAI

    report, _ = run_workflow(tmp_path, fake_gmail(unread_newsletters("a", 6, 6, "x.example")),
                             fake_openai=FakeOpenAI(hostile))
    html = render_html_report(report)
    assert '<script>alert("x")</script>' not in html
    assert "&lt;script&gt;" in html
    for item in report["cleanup_advice"]["suggestions"]:
        assert "http" not in item["reason"]  # Reasons are built from numbers only.
    assert 'aria-label="Messages per week: ' in html  # Sparklines carry numbers, not email text.
