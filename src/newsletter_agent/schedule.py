"""Run the briefing every day: launchd on macOS, a systemd user timer on Linux,
Task Scheduler on Windows.

`newsletter-agent schedule 07:00` asks for consent, installs the schedule, and
only after the scheduler confirms it turns on auto-approval (within the
briefing budget). A failed install changes nothing: its files are removed and
auto-approval stays off.

Missed runs: launchd and the systemd timer (Persistent=true) run the briefing
when the computer wakes. Task Scheduler skips a run while the computer is off.
On Linux without systemd, the command prints a cron line instead.
"""

import os
import plistlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import config

LABEL = "com.newsletter-agent.briefing"
UNIT = "newsletter-agent-briefing"
TASK_NAME = "newsletter-agent-briefing"


def platform_name() -> str:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
    return "linux"


def parse_time(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", text.strip())
    if not match:
        raise ValueError("Use 24-hour HH:MM, for example 07:00.")
    return int(match.group(1)), int(match.group(2))


def briefing_command() -> list[str]:
    return [sys.executable, "-m", "newsletter_agent", "brief", "--scheduled"]


def run(command: list[str], run_command=subprocess.run):
    return run_command(command, capture_output=True, text=True, check=False)


def failure(tool: str, result) -> ValueError:
    detail = (getattr(result, "stderr", "") or "").strip()[:200]
    return ValueError(f"{tool} could not install the schedule{': ' + detail if detail else ''}. Nothing was changed.")


# --- macOS: launchd -------------------------------------------------------------------

def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def gui_domain() -> str:
    return f"gui/{os.getuid()}"  # type: ignore[attr-defined,unused-ignore]  # macOS only.


def build_plist(hour: int, minute: int) -> bytes:
    logs = config.logs_dir()
    return plistlib.dumps({
        "Label": LABEL,
        "ProgramArguments": briefing_command(),
        "StartCalendarInterval": {"Hour": hour, "Minute": minute},
        "EnvironmentVariables": {"NEWSLETTER_AGENT_HOME": str(config.app_home())},
        "StandardOutPath": str(logs / "briefing.log"),
        "StandardErrorPath": str(logs / "briefing.log"),
        "ProcessType": "Background",
    })


def install_launchd(hour: int, minute: int, run_command) -> None:
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_plist(hour, minute))
    run(["launchctl", "bootout", gui_domain(), str(path)], run_command)  # Replace an older schedule if present.
    result = run(["launchctl", "bootstrap", gui_domain(), str(path)], run_command)
    if result.returncode != 0:
        path.unlink(missing_ok=True)
        raise failure("launchctl", result)


def remove_launchd(run_command) -> bool:
    path = plist_path()
    if not path.exists():
        return False
    run(["launchctl", "bootout", gui_domain(), str(path)], run_command)
    path.unlink()
    return True


# --- Linux: systemd user timer -----------------------------------------------------------

def systemd_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def systemd_quote(value: str) -> str:
    """Quote one value for a unit file: % starts a specifier, so it is doubled."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'"{escaped}"'


def build_systemd_units(hour: int, minute: int) -> dict[str, str]:
    log = config.logs_dir() / "briefing.log"
    command = " ".join(systemd_quote(part) for part in briefing_command())
    service = "\n".join([
        "[Unit]",
        "Description=newsletter-agent daily briefing",
        "",
        "[Service]",
        "Type=oneshot",
        f"Environment={systemd_quote('NEWSLETTER_AGENT_HOME=' + str(config.app_home()))}",
        f"ExecStart={command}",
        f"StandardOutput=append:{str(log).replace('%', '%%')}",
        f"StandardError=append:{str(log).replace('%', '%%')}",
        "",
    ])
    timer = "\n".join([
        "[Unit]",
        "Description=Run the newsletter-agent briefing daily",
        "",
        "[Timer]",
        f"OnCalendar=*-*-* {hour:02d}:{minute:02d}:00",
        "Persistent=true",  # Run at the next boot or wake if the computer was off at that time.
        "",
        "[Install]",
        "WantedBy=timers.target",
        "",
    ])
    return {f"{UNIT}.service": service, f"{UNIT}.timer": timer}


def install_systemd(hour: int, minute: int, run_command) -> None:
    directory = systemd_dir()
    directory.mkdir(parents=True, exist_ok=True)
    units = build_systemd_units(hour, minute)
    for name, text in units.items():
        (directory / name).write_text(text, encoding="utf-8")
    for command in (["systemctl", "--user", "daemon-reload"],
                    ["systemctl", "--user", "enable", "--now", f"{UNIT}.timer"]):
        result = run(command, run_command)
        if result.returncode != 0:
            for name in units:
                (directory / name).unlink(missing_ok=True)
            run(["systemctl", "--user", "daemon-reload"], run_command)
            raise failure("systemctl", result)


def remove_systemd(run_command) -> bool:
    directory = systemd_dir()
    names = [f"{UNIT}.service", f"{UNIT}.timer"]
    if not any((directory / name).exists() for name in names):
        return False
    run(["systemctl", "--user", "disable", "--now", f"{UNIT}.timer"], run_command)
    for name in names:
        (directory / name).unlink(missing_ok=True)
    run(["systemctl", "--user", "daemon-reload"], run_command)
    return True


# --- Windows: Task Scheduler -------------------------------------------------------------

def windows_script_path() -> Path:
    return config.app_home() / "briefing.cmd"


def build_windows_script() -> str:
    """Task Scheduler runs one command, so a small script sets the data folder and the log."""
    python, *arguments = briefing_command()
    log = config.logs_dir() / "briefing.log"

    def literal(path) -> str:
        return str(path).replace("%", "%%")  # In a batch file, a single % starts a variable.

    return "\r\n".join([
        "@echo off",
        "chcp 65001 >nul",  # The file is UTF-8, so paths with non-ASCII characters are read correctly.
        f'set "NEWSLETTER_AGENT_HOME={literal(config.app_home())}"',
        f'"{literal(python)}" {" ".join(arguments)} >> "{literal(log)}" 2>&1',
        "",
    ])


def install_windows(hour: int, minute: int, run_command) -> None:
    script = windows_script_path()
    config.write_private_file(script, build_windows_script())
    result = run(["schtasks", "/Create", "/TN", TASK_NAME, "/SC", "DAILY", "/ST", f"{hour:02d}:{minute:02d}",
                  "/TR", f'"{script}"', "/F"], run_command)
    if result.returncode != 0:
        script.unlink(missing_ok=True)
        raise failure("schtasks", result)


def remove_windows(run_command) -> bool:
    script = windows_script_path()
    if not script.exists():
        return False
    run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"], run_command)
    script.unlink()
    return True


# --- Choosing the backend -------------------------------------------------------------------

def is_scheduled() -> bool:
    platform = platform_name()
    if platform == "macos":
        return plist_path().exists()
    if platform == "windows":
        return windows_script_path().exists()
    return (systemd_dir() / f"{UNIT}.timer").exists()


def cron_line(hour: int, minute: int) -> str:
    return f"{minute} {hour} * * * {' '.join(briefing_command())}"


def install_schedule(time_text: str, ask, run_command=subprocess.run) -> str:
    hour, minute = parse_time(time_text)
    platform = platform_name()
    if platform == "linux" and not shutil.which("systemctl"):
        return ("systemd was not found, so nothing was scheduled and auto-approval is still off. "
                "To run the briefing daily anyway, add this line with `crontab -e`, and set "
                f"auto_approve = true under [briefing] in {config.app_home() / 'config.toml'}:\n"
                f"{cron_line(hour, minute)}")
    settings = config.load_config()
    briefing = settings["briefing"]
    question = (
        f"Every day at {hour:02d}:{minute:02d}, newsletter-agent will read new Gmail messages (read-only), "
        f"send newsletter text to OpenAI ({config.MODEL}) without asking, and save a private briefing. "
        f"Each run is capped at {briefing['max_model_calls']} calls and {briefing['max_tokens']:,} tokens. Approve?"
    )
    if not ask(question):
        return "Not scheduled. Nothing was changed."

    installers = {"macos": install_launchd, "linux": install_systemd, "windows": install_windows}
    installers[platform](hour, minute, run_command)  # Raises, with its files removed, if the scheduler refuses.
    briefing["auto_approve"] = True  # Only after the scheduler confirmed the install.
    briefing["time"] = f"{hour:02d}:{minute:02d}"
    config.save_config(settings)
    return f"Scheduled daily at {hour:02d}:{minute:02d}. Briefings: {config.briefings_dir() / 'latest.html'}"


def remove_schedule(run_command=subprocess.run) -> str:
    settings = config.load_config()
    settings["briefing"]["auto_approve"] = False
    config.save_config(settings)
    removers = {"macos": remove_launchd, "linux": remove_systemd, "windows": remove_windows}
    if removers[platform_name()](run_command):
        return "Daily briefing removed, and auto-approval turned off."
    return "No schedule was installed. Auto-approval is off."
