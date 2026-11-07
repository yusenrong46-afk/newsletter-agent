"""Totals, reconciliation, and the HTML report's safety with hostile email text."""

import re

from helpers import WINDOW_START_MS, DAY_MS, FakeOpenAI, fake_gmail, group_by_key, message, newsletter, run_workflow
from newsletter_agent.classify import ALL_CATEGORIES, ClassificationBatch, MessageClassification
from newsletter_agent.html_report import page, render_html_report
from newsletter_agent.report import check_reconciliation


def test_totals_reconcile_across_a_realistic_mailbox(tmp_path):
    google = fake_gmail([
        newsletter("a1", list_id="alpha.example", labels=("UNREAD",)),
        newsletter("a2", list_id="alpha.example"),
        newsletter("b1", list_id="beta.example", labels=("UNREAD",)),
        newsletter("r1", list_id="beta.example", subject="Your receipt", labels=("UNREAD",)),
        message("solo", sender="Solo <solo@home.example>", labels=("UNREAD",)),
        newsletter("old", when=WINDOW_START_MS - DAY_MS),
    ])
    report, _ = run_workflow(tmp_path, google)
    counts = report["counts"]
    assert (counts["discovered"], counts["in_scope"], counts["excluded"]) == (6, 5, 1)
    assert (counts["candidates"], counts["not_candidates"], counts["classified"]) == (4, 1, 4)
    assert (counts["subscription_messages"], counts["subscription_unread"]) == (3, 2)
    assert counts["in_scope_unread"] == 4  # Includes the receipt and the unclassified message.
    assert group_by_key(report, "list:alpha.example")["unread_count"] == 1
    assert group_by_key(report, "list:beta.example")["received_count"] == 1
    assert report["reconciliation"]["ok"]


def test_reconciliation_detects_a_mismatch():
    counts = {"discovered": 5, "in_scope": 4, "excluded": 0, "unavailable": 0,
              "candidates": 4, "not_candidates": 0, "classified": 4, "unresolved": 0, "not_yet_classified": 0}
    categories = {name: 0 for name in ALL_CATEGORIES}
    categories["editorial_newsletter"] = 4
    assert check_reconciliation(counts, categories, [{"received_count": 3}]) == [
        "discovered != in_scope + excluded + unavailable", "sum of group counts != subscription messages"]


def test_same_name_groups_are_flagged_but_not_merged(tmp_path):
    def same_name(items):
        return ClassificationBatch(classifications=[
            MessageClassification(message_id=item["id"], classification="editorial_newsletter",
                                  publication_name="Daily Brief", confidence="high", rationale="x") for item in items])

    google = fake_gmail([newsletter("a", list_id="one.example"), newsletter("b", list_id="two.example")])
    report, _ = run_workflow(tmp_path, google, FakeOpenAI(same_name))
    assert report["counts"]["publications_observed"] == 2
    assert report["possible_duplicates"][0]["group_keys"] == ["list:one.example", "list:two.example"]


def hostile_report(tmp_path):
    def hostile_name(items):
        return ClassificationBatch(classifications=[
            MessageClassification(message_id=item["id"], classification="editorial_newsletter",
                                  publication_name='</script><script>alert("name")</script>',
                                  confidence="high", rationale="x") for item in items])

    google = fake_gmail([newsletter("x", subject='<img src=x onerror=alert(1)> "quoted"',
                                    sender='"<b>Bold</b>" <evil@example.com>')])
    report, _ = run_workflow(tmp_path, google, FakeOpenAI(hostile_name))
    return report


def test_html_report_escapes_hostile_email_text(tmp_path):
    html = render_html_report(hostile_report(tmp_path))
    assert "<img src=x" not in html and 'alert("name")</script>' not in html
    assert "&lt;img src=x" in html and "&lt;/script&gt;" in html
    assert html.count("<script>") == 1  # Only the page's own sorting script.


def test_html_report_cannot_load_anything_remote(tmp_path):
    html = render_html_report(hostile_report(tmp_path))
    assert "Content-Security-Policy" in html and "default-src 'none'" in html
    assert not re.search(r"(src|href)\s*=\s*[\"']?https?:", html)


def test_html_report_numbers_match_the_json(tmp_path):
    report, _ = run_workflow(tmp_path, fake_gmail([newsletter("a", labels=("UNREAD",)), newsletter("b")]))
    html = render_html_report(report)
    group = report["publications_observed"][0]
    assert f'data-received="{group["received_count"]}"' in html
    assert f'data-unread="{group["unread_count"]}"' in html
    assert "DEMO DATA" not in html and "DEMO DATA" in render_html_report(report, demo=True)


def test_page_title_is_escaped():
    assert "<title>&lt;x&gt;</title>" in page("<x>", "")
