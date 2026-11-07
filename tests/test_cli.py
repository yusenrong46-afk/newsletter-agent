"""The command line, setup checks, and importing data from the old single-file layout."""

import stat

import pytest

from newsletter_agent import config
from newsletter_agent.cli import build_parser, main


@pytest.mark.parametrize("argv", [
    ["setup"], ["login"], ["check", "--gmail"], ["inventory", "--days", "90", "--preflight"],
    ["inventory", "--resume", "--yes", "--max-messages", "100"], ["report", "--open"], ["chat"],
    ["brief", "--yes", "--open"], ["schedule", "07:00"], ["unschedule"], ["demo", "--real-model"], ["mcp-server"],
])
def test_every_command_parses(argv):
    assert build_parser().parse_args(argv).handler


def test_invalid_days_is_a_clear_error(capsys):
    assert main(["inventory", "--days", "0"]) == 1
    assert "--days must be at least 1" in capsys.readouterr().err


def test_setup_reports_status_without_printing_secrets(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-SECRET-VALUE")
    assert main(["setup"]) == 0
    output = capsys.readouterr().out
    assert "✓ OpenAI API key" in output and "SECRET-VALUE" not in output
    assert "newsletter-agent login" in output


def test_import_legacy_copies_privately_and_never_overwrites(tmp_path):
    old = tmp_path / "old"
    (old / ".gmail").mkdir(parents=True)
    (old / ".gmail" / "token.json").write_text("old-token")
    (old / ".env.local").write_text("OPENAI_API_KEY=sk-x\n")
    (old / ".inventory" / "reports").mkdir(parents=True)
    (old / ".inventory" / "classification_cache.json").write_text("{}")
    (old / ".inventory" / "reports" / "inventory-1.json").write_text("{}")
    (old / ".inventory" / ".tmp-file").write_text("skip me")

    config.credentials_dir()
    (config.credentials_dir() / "token.json").write_text("newer-token")  # Already present: keep it.
    copied = config.import_legacy_data(old)

    assert sorted(copied) == ["credentials/.env", "inventory/classification_cache.json", "inventory/reports/inventory-1.json"]
    assert (config.credentials_dir() / "token.json").read_text() == "newer-token"
    for name in copied:
        assert stat.S_IMODE((config.app_home() / name).stat().st_mode) == 0o600
