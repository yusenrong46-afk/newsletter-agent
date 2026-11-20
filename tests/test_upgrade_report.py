"""The before/after report is generated from the saved results, not typed by hand."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_make_report():
    spec = importlib.util.spec_from_file_location("make_report", ROOT / "bench" / "make_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_report_is_built_from_the_measured_results(tmp_path):
    make_report = load_make_report()
    output = tmp_path / "report.html"
    assert make_report.main(["--output", str(output)]) == 0
    html = output.read_text(encoding="utf-8")
    before = json.loads((ROOT / "bench" / "results" / "before.json").read_text(encoding="utf-8"))
    after = json.loads((ROOT / "bench" / "results" / "after.json").read_text(encoding="utf-8"))
    assert html.startswith("<!doctype html>")
    assert f"{before['scan_speed']['seconds']:.1f} s" in html and f"{after['scan_speed']['seconds']:.1f} s" in html
    assert before["answers"] == after["answers"]  # The upgrade did not change the demo's answers.
    assert "The answers are identical" in html
    assert "http://" not in html and "https://" not in html  # Self-contained: nothing loads from the network.
