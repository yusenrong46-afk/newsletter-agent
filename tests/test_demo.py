"""The demo runs the real pipeline offline, on invented mail only."""

import asyncio
import json
from datetime import datetime, timezone

from newsletter_agent import classify, demo
from newsletter_agent.cli import main
from newsletter_agent.classify import ClassificationBatch

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


class RecordingModel(demo.OfflineModel):
    def __init__(self):
        super().__init__()
        self.inputs = []

    async def parse(self, text_format, input, **kwargs):
        self.inputs.append((text_format, json.loads(input)))
        return await super().parse(text_format, input, **kwargs)


def run(tmp_path):
    model = RecordingModel()
    report, briefing = asyncio.run(demo.run_demo_async(tmp_path / "site", lambda: model, True, NOW))
    return report, briefing, model


def test_demo_reconciles_and_shows_the_interesting_cases(tmp_path):
    (tmp_path / "site").mkdir()
    report, briefing, model = run(tmp_path)
    counts = report["counts"]
    assert report["reconciliation"]["ok"] and report["run"]["status"] == "complete"
    assert counts["excluded_by_reason"] == {"outside_window": 4, "label_sent": 2}
    assert counts["unavailable"] == 1  # The message deleted mid-scan.

    groups = {g["group_key"]: g for g in report["publications_observed"] + report["publications_provisional"]}
    # Same delivery service (mailhost.example), still two publications.
    assert "list:bytegarden.mailhost.example" in groups and "list:fieldnotes.mailhost.example" in groups
    assert groups["sender:news@trailhead.example"]["other_messages_same_identity"] == {"transactional": 6}

    sent_ids = {item["id"] for fmt, data in model.inputs if fmt is ClassificationBatch for item in data["messages"]}
    assert not any(message_id.startswith("sam") for message_id in sent_ids)  # Personal mail never leaves.
    assert "orbit-injection" in sent_ids
    assert briefing["counts"]["summarized"] >= 4


def test_demo_writes_a_labelled_site(tmp_path):
    (tmp_path / "site").mkdir()
    run(tmp_path)
    for name in ("index.html", "briefing.html"):
        html = (tmp_path / "site" / name).read_text()
        assert "DEMO DATA" in html and "Content-Security-Policy" in html


def test_demo_command_needs_no_accounts(tmp_path, monkeypatch):
    def no_real_model():
        raise AssertionError("The offline demo must not create an OpenAI client.")

    monkeypatch.setattr(classify, "make_openai_client", no_real_model)
    assert main(["demo", "--output", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "index.html").exists()
