"""The golden demo snapshot: the answers an upgrade must not change.

The demo pipeline runs on the synthetic mailbox with a fixed date. Counts,
categories, publication groups and the reconciliation result must match the
saved snapshot exactly. New report keys are fine; changed values are not.

To update the snapshot after a deliberate change: UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from newsletter_agent.demo import OfflineModel, run_demo_async

GOLDEN = Path(__file__).parent / "golden" / "demo_report.json"
NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def demo_answers(output_dir: Path) -> dict:
    model = OfflineModel()
    output_dir.mkdir(parents=True, exist_ok=True)
    report, briefing = asyncio.run(run_demo_async(output_dir, lambda: model, True, NOW))
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


def assert_contains(expected, actual, path="answers"):
    """Every saved value must be present and equal; extra keys in `actual` are allowed."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected an object"
        for key, value in expected.items():
            assert key in actual, f"{path}.{key} is missing"
            assert_contains(value, actual[key], f"{path}.{key}")
    else:
        assert actual == expected, f"{path}: expected {expected!r}, got {actual!r}"


def test_demo_answers_match_the_golden_snapshot(tmp_path):
    answers = demo_answers(tmp_path / "site")
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(answers, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert answers["reconciliation_ok"] is True
    assert_contains(golden, answers)
