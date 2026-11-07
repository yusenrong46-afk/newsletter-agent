"""Run the briefing every day with macOS launchd.

`newsletter-agent schedule 07:00` asks for consent, turns on auto-approval
(within the briefing budget), and installs a LaunchAgent. If the Mac is asleep
at that time, launchd runs the briefing when it wakes.
"""

import os
import plistlib
import re
import subprocess
import sys
from pathlib import Path

from . import config

LABEL = "com.newsletter-agent.briefing"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def parse_time(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", text.strip())
    if not match:
        raise ValueError("Use 24-hour HH:MM, for example 07:00.")
    return int(match.group(1)), int(match.group(2))


def build_plist(hour: int, minute: int) -> bytes:
    logs = config.logs_dir()
    return plistlib.dumps({
        "Label": LABEL,
        "ProgramArguments": [sys.executable, "-m", "newsletter_agent", "brief", "--scheduled"],
        "StartCalendarInterval": {"Hour": hour, "Minute": minute},
        "EnvironmentVariables": {"NEWSLETTER_AGENT_HOME": str(config.app_home())},
        "StandardOutPath": str(logs / "briefing.log"),
        "StandardErrorPath": str(logs / "briefing.log"),
        "ProcessType": "Background",
    })


def launchctl(arguments: list[str], run_command=subprocess.run):
    return run_command(["launchctl", *arguments], capture_output=True, text=True, check=False)


def install_schedule(time_text: str, ask, run_command=subprocess.run) -> str:
    if sys.platform != "darwin":
        hour, minute = parse_time(time_text)
        return (f"Automatic scheduling is macOS-only. On Linux, add this cron line:\n"
                f"{minute} {hour} * * * {sys.executable} -m newsletter_agent brief --scheduled")
    hour, minute = parse_time(time_text)
    settings = config.load_config()
    briefing = settings["briefing"]
    question = (
        f"Every day at {hour:02d}:{minute:02d}, newsletter-agent will read new Gmail messages (read-only), "
        f"send newsletter text to OpenAI ({config.MODEL}) without asking, and save a private briefing. "
        f"Each run is capped at {briefing['max_model_calls']} calls and {briefing['max_tokens']:,} tokens. Approve?"
    )
    if not ask(question):
        return "Not scheduled. Nothing was changed."

    briefing["auto_approve"] = True
    briefing["time"] = f"{hour:02d}:{minute:02d}"
    config.save_config(settings)
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_plist(hour, minute))
    domain = f"gui/{os.getuid()}"
    launchctl(["bootout", domain, str(path)], run_command)  # Replace an older schedule if present.
    result = launchctl(["bootstrap", domain, str(path)], run_command)
    if result.returncode != 0:
        raise ValueError(f"launchctl could not load the schedule: {result.stderr.strip()[:200]}")
    return f"Scheduled daily at {hour:02d}:{minute:02d}. Briefings: {config.briefings_dir() / 'latest.html'}"


def remove_schedule(run_command=subprocess.run) -> str:
    settings = config.load_config()
    settings["briefing"]["auto_approve"] = False
    config.save_config(settings)
    path = plist_path()
    if path.exists():
        launchctl(["bootout", f"gui/{os.getuid()}", str(path)], run_command)
        path.unlink()
        return "Daily briefing removed, and auto-approval turned off."
    return "No schedule was installed. Auto-approval is off."
