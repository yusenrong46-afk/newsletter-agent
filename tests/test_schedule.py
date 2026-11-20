"""Scheduling on macOS, Linux and Windows: explicit consent, and nothing half-installed on failure.

No test runs launchctl, systemctl or schtasks: a fake run_command records the
commands. Red tests (fail on v0.1): Linux installs a real timer, Windows
installs a task, a failed install leaves auto-approval off, and the store
imports without the Unix-only fcntl module.
"""

import plistlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from newsletter_agent import briefing, config, schedule


class FakeRunner:
    def __init__(self, fail_on: str | None = None):
        self.commands: list[list[str]] = []
        self.fail_on = fail_on

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        failed = self.fail_on is not None and self.fail_on in command
        return SimpleNamespace(returncode=1 if failed else 0, stderr="refused by the test" if failed else "")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule.Path, "home", lambda: tmp_path / "user")
    monkeypatch.setattr(schedule, "gui_domain", lambda: "gui/501")
    return tmp_path / "user"


def on(monkeypatch, platform: str):
    monkeypatch.setattr(schedule.sys, "platform", platform)


def approve(question):
    return True


def auto_approve() -> bool:
    return config.load_config()["briefing"]["auto_approve"]


# --- Shared rules ---------------------------------------------------------------------

def test_parse_time_accepts_24_hour_times_only():
    assert schedule.parse_time("07:00") == (7, 0)
    assert schedule.parse_time("23:59") == (23, 59)
    for bad in ("7am", "24:00", "07:60", ""):
        with pytest.raises(ValueError):
            schedule.parse_time(bad)


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_declining_changes_nothing(home, monkeypatch, platform):
    on(monkeypatch, platform)
    monkeypatch.setattr(schedule.shutil, "which", lambda name: "/usr/bin/" + name)
    runner = FakeRunner()
    message = schedule.install_schedule("07:00", ask=lambda question: False, run_command=runner)
    assert "Nothing was changed" in message
    assert runner.commands == [] and not schedule.is_scheduled() and not auto_approve()


@pytest.mark.parametrize("platform, tool", [("darwin", "bootstrap"), ("linux", "enable"), ("win32", "/Create")])
def test_a_failed_install_leaves_nothing_behind_and_auto_approval_off(home, monkeypatch, platform, tool):
    on(monkeypatch, platform)
    monkeypatch.setattr(schedule.shutil, "which", lambda name: "/usr/bin/" + name)
    with pytest.raises(ValueError, match="could not install the schedule.*refused by the test"):
        schedule.install_schedule("07:00", ask=approve, run_command=FakeRunner(fail_on=tool))
    assert not schedule.is_scheduled()
    assert not auto_approve()  # v0.1 switched this on before launchctl answered.


def test_auto_approve_is_off_by_default():
    assert auto_approve() is False


# --- macOS ------------------------------------------------------------------------------

def test_plist_runs_the_briefing_at_the_chosen_time():
    plist = plistlib.loads(schedule.build_plist(7, 0))
    assert plist["ProgramArguments"][1:] == ["-m", "newsletter_agent", "brief", "--scheduled"]
    assert plist["StartCalendarInterval"] == {"Hour": 7, "Minute": 0}
    assert plist["EnvironmentVariables"]["NEWSLETTER_AGENT_HOME"] == str(config.app_home())


def test_macos_consent_installs_and_unschedule_removes(home, monkeypatch):
    on(monkeypatch, "darwin")
    runner, questions = FakeRunner(), []
    schedule.install_schedule("07:00", ask=lambda q: questions.append(q) or True, run_command=runner)
    assert "send newsletter text to OpenAI" in questions[0] and "600,000 tokens" in questions[0]
    assert schedule.plist_path().exists() and schedule.is_scheduled()
    assert auto_approve()
    assert runner.commands[-1][:2] == ["launchctl", "bootstrap"]

    assert "removed" in schedule.remove_schedule(run_command=runner)
    assert not schedule.plist_path().exists()
    assert not auto_approve()


# --- Linux ------------------------------------------------------------------------------

def test_linux_installs_a_persistent_systemd_user_timer(home, monkeypatch):
    on(monkeypatch, "linux")
    monkeypatch.setattr(schedule.shutil, "which", lambda name: "/usr/bin/systemctl")
    runner = FakeRunner()
    message = schedule.install_schedule("06:45", ask=approve, run_command=runner)

    assert "Scheduled daily at 06:45" in message
    timer = (schedule.systemd_dir() / "newsletter-agent-briefing.timer").read_text(encoding="utf-8")
    service = (schedule.systemd_dir() / "newsletter-agent-briefing.service").read_text(encoding="utf-8")
    assert "OnCalendar=*-*-* 06:45:00" in timer and "Persistent=true" in timer
    assert '"-m" "newsletter_agent" "brief" "--scheduled"' in service
    assert f"Environment={schedule.systemd_quote(f'NEWSLETTER_AGENT_HOME={config.app_home()}')}" in service
    assert runner.commands == [["systemctl", "--user", "daemon-reload"],
                               ["systemctl", "--user", "enable", "--now", "newsletter-agent-briefing.timer"]]
    assert auto_approve() and schedule.is_scheduled()

    assert "removed" in schedule.remove_schedule(run_command=runner)
    assert ["systemctl", "--user", "disable", "--now", "newsletter-agent-briefing.timer"] in runner.commands
    assert not schedule.is_scheduled() and not auto_approve()


def test_linux_without_systemd_prints_a_cron_line_and_changes_nothing(home, monkeypatch):
    on(monkeypatch, "linux")
    monkeypatch.setattr(schedule.shutil, "which", lambda name: None)

    def must_not_ask(question):
        raise AssertionError("Nothing is installed, so there is nothing to approve.")

    message = schedule.install_schedule("07:00", ask=must_not_ask, run_command=FakeRunner())
    assert "0 7 * * *" in message and "brief --scheduled" in message
    assert not schedule.is_scheduled() and not auto_approve()


def test_systemd_values_are_quoted_and_percent_signs_escaped():
    assert schedule.systemd_quote('C:/a b/100%"x') == '"C:/a b/100%%\\"x"'


# --- Windows ----------------------------------------------------------------------------

def test_windows_installs_a_daily_task_and_unschedule_deletes_it(home, monkeypatch):
    on(monkeypatch, "win32")
    runner = FakeRunner()
    schedule.install_schedule("07:30", ask=approve, run_command=runner)

    script = schedule.windows_script_path().read_text(encoding="utf-8")
    assert f'set "NEWSLETTER_AGENT_HOME={config.app_home()}"' in script
    assert "-m newsletter_agent brief --scheduled" in script and "briefing.log" in script
    create = runner.commands[-1]
    assert create[:2] == ["schtasks", "/Create"]
    assert create[create.index("/SC") + 1] == "DAILY" and create[create.index("/ST") + 1] == "07:30"
    assert create[create.index("/TR") + 1] == f'"{schedule.windows_script_path()}"'
    assert auto_approve() and schedule.is_scheduled()

    assert "removed" in schedule.remove_schedule(run_command=runner)
    assert runner.commands[-1] == ["schtasks", "/Delete", "/TN", "newsletter-agent-briefing", "/F"]
    assert not schedule.is_scheduled() and not auto_approve()


def test_the_windows_script_survives_percent_signs_and_non_ascii_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWSLETTER_AGENT_HOME", str(tmp_path / "José 100% data"))
    lines = schedule.build_windows_script().split("\r\n")
    assert lines[:2] == ["@echo off", "chcp 65001 >nul"]  # UTF-8, so the accented folder name is read correctly.
    assert "José 100%% data" in lines[2] and "100% data" not in lines[2].replace("%%", "")
    assert "José 100%% data" in lines[3]


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_unschedule_with_nothing_installed(home, monkeypatch, platform):
    on(monkeypatch, platform)
    runner = FakeRunner()
    assert "No schedule was installed" in schedule.remove_schedule(run_command=runner)
    assert runner.commands == []


# --- Portability of the rest of the app -------------------------------------------------------

def test_the_store_and_cli_import_without_fcntl():
    """Windows has no fcntl; v0.1 imported it at the top of store.py and could not start there."""
    code = ("import sys; sys.modules['fcntl'] = None; "
            "import newsletter_agent.store, newsletter_agent.inventory, newsletter_agent.cli; print('ok')")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.stdout.strip() == "ok", result.stderr


def test_linux_notifications_use_notify_send_with_plain_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(briefing.sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/notify-send")
    monkeypatch.setattr(briefing.subprocess, "run", lambda command, **kwargs: calls.append(command))
    briefing.notify('3 issues; $(rm -rf ~) "quoted"')
    assert calls == [["notify-send", "Newsletter briefing", '3 issues; $(rm -rf ~) "quoted"']]


def test_windows_notifications_do_nothing(monkeypatch):
    monkeypatch.setattr(briefing.sys, "platform", "win32")
    monkeypatch.setattr(briefing.subprocess, "run", lambda *a, **k: pytest.fail("no notifier on Windows"))
    briefing.notify("hello")
