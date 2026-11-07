"""Scheduling needs explicit consent and never runs launchctl in tests."""

import plistlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from newsletter_agent import config, schedule


@pytest.fixture
def mac(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule.sys, "platform", "darwin")
    monkeypatch.setattr(schedule.Path, "home", lambda: tmp_path / "user")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    return commands, fake_run


def test_parse_time_accepts_24_hour_times_only():
    assert schedule.parse_time("07:00") == (7, 0)
    assert schedule.parse_time("23:59") == (23, 59)
    for bad in ("7am", "24:00", "07:60", ""):
        with pytest.raises(ValueError):
            schedule.parse_time(bad)


def test_plist_runs_the_briefing_at_the_chosen_time():
    plist = plistlib.loads(schedule.build_plist(7, 0))
    assert plist["ProgramArguments"][1:] == ["-m", "newsletter_agent", "brief", "--scheduled"]
    assert plist["StartCalendarInterval"] == {"Hour": 7, "Minute": 0}
    assert plist["EnvironmentVariables"]["NEWSLETTER_AGENT_HOME"] == str(config.app_home())


def test_declining_changes_nothing(mac):
    commands, fake_run = mac
    message = schedule.install_schedule("07:00", ask=lambda question: False, run_command=fake_run)
    assert "Nothing was changed" in message
    assert commands == [] and not schedule.plist_path().exists()
    assert config.load_config()["briefing"]["auto_approve"] is False


def test_consent_installs_and_unschedule_removes(mac):
    commands, fake_run = mac
    questions = []
    schedule.install_schedule("07:00", ask=lambda q: questions.append(q) or True, run_command=fake_run)
    assert "send newsletter text to OpenAI" in questions[0] and "300,000 tokens" in questions[0]
    assert schedule.plist_path().exists()
    assert config.load_config()["briefing"]["auto_approve"] is True
    assert commands[-1][:2] == ["launchctl", "bootstrap"]

    schedule.remove_schedule(run_command=fake_run)
    assert not schedule.plist_path().exists()
    assert config.load_config()["briefing"]["auto_approve"] is False


def test_auto_approve_is_off_by_default():
    assert config.load_config()["briefing"]["auto_approve"] is False
