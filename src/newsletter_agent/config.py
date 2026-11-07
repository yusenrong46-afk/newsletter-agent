"""Settings, private file locations, and the OpenAI key.

All personal data lives in one private folder, ~/.newsletter-agent by default
(set NEWSLETTER_AGENT_HOME to use another folder):

    credentials/   Google OAuth client, Gmail token, .env with OPENAI_API_KEY
    inventory/     caches, checkpoint, reports/
    briefings/     daily briefing pages
    logs/          output from scheduled runs
    config.toml    your settings
"""

import contextlib
import getpass
import os
import shutil
import sys
import tempfile
import tomllib
from datetime import tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MODEL = "gpt-6-luna"

DEFAULT_CONFIG = {
    "display_timezone": "",  # Empty means this computer's time zone.
    "briefing": {
        "time": "07:00",
        "categories": ["editorial_newsletter"],
        "include": [],  # Publication names to keep; empty keeps all.
        "exclude": [],  # Publication names to skip.
        "max_model_calls": 30,
        "max_tokens": 300_000,
        "auto_approve": False,  # Only `newsletter-agent schedule` turns this on, after asking.
    },
}


# --- Folders ----------------------------------------------------------------

def app_home() -> Path:
    return Path(os.environ.get("NEWSLETTER_AGENT_HOME") or Path.home() / ".newsletter-agent")


def private_dir(*parts: str) -> Path:
    """Return a folder inside the app home, created with owner-only permissions."""
    path = app_home().joinpath(*parts)
    for folder in [app_home(), *[app_home().joinpath(*parts[:i + 1]) for i in range(len(parts))]]:
        if folder.is_symlink():
            raise ValueError(f"Refusing to use a symlink for private data: {folder}")
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(folder, 0o700)
    return path


def credentials_dir() -> Path:
    return private_dir("credentials")


def inventory_dir() -> Path:
    return private_dir("inventory")


def briefings_dir() -> Path:
    return private_dir("briefings")


def logs_dir() -> Path:
    return private_dir("logs")


def write_private_file(path: Path, text: str):
    """Write atomically with mode 0600: a crash leaves the old file or the new one."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


# --- config.toml ------------------------------------------------------------

def load_config() -> dict:
    """Read config.toml on top of the defaults. A missing file means all defaults."""
    config = {"display_timezone": DEFAULT_CONFIG["display_timezone"], "briefing": dict(DEFAULT_CONFIG["briefing"])}
    path = app_home() / "config.toml"
    if path.exists():
        saved = tomllib.loads(path.read_text(encoding="utf-8"))
        config["display_timezone"] = saved.get("display_timezone", config["display_timezone"])
        config["briefing"].update(saved.get("briefing", {}))
    return config


def toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def save_config(config: dict):
    lines = [f"display_timezone = {toml_value(config['display_timezone'])}", "", "[briefing]"]
    lines += [f"{key} = {toml_value(value)}" for key, value in config["briefing"].items()]
    private_dir()
    write_private_file(app_home() / "config.toml", "\n".join(lines) + "\n")


def display_zone() -> tzinfo:
    """The time zone for dates in reports: config.toml's display_timezone, else this computer's."""
    from datetime import datetime
    from zoneinfo import ZoneInfoNotFoundError

    name = load_config()["display_timezone"]
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            print(f"Warning: unknown display_timezone {name!r} in config.toml; using this computer's.", file=sys.stderr)
    return datetime.now().astimezone().tzinfo


# --- OpenAI key -------------------------------------------------------------

def env_file() -> Path:
    return credentials_dir() / ".env"


def has_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip()) or env_file().exists()


def get_api_key() -> str:
    """Use OPENAI_API_KEY, else credentials/.env, else ask once and save privately."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    path = env_file()
    if path.exists():
        for line in path.read_text().splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "OPENAI_API_KEY":
                return value.strip().strip("\"'")
        raise ValueError(f"{path} exists but does not contain OPENAI_API_KEY.")
    if not sys.stdin.isatty():
        raise ValueError("No OpenAI API key found. Set OPENAI_API_KEY or run `newsletter-agent setup`.")
    key = getpass.getpass("OpenAI API key (saved privately for future runs): ").strip()
    if not key.startswith("sk-") or any(char.isspace() for char in key):
        raise ValueError("The API key format looks invalid; it was not saved.")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # Never overwrites.
    with os.fdopen(fd, "w") as output:
        output.write(f"OPENAI_API_KEY={key}\n")
    return key


# --- One-time import from the single-file version -----------------------------

def import_legacy_data(project_dir: Path) -> list[str]:
    """Copy .gmail/, .env.local and .inventory/ from the old layout. Never overwrites."""
    copies = [
        (project_dir / ".gmail" / "client_secret.json", credentials_dir() / "client_secret.json"),
        (project_dir / ".gmail" / "token.json", credentials_dir() / "token.json"),
        (project_dir / ".env.local", credentials_dir() / ".env"),
    ]
    legacy_inventory = project_dir / ".inventory"
    if legacy_inventory.is_dir():
        for source in legacy_inventory.rglob("*"):
            if source.is_file() and not source.name.startswith("."):
                copies.append((source, inventory_dir() / source.relative_to(legacy_inventory)))

    copied = []
    for source, destination in copies:
        if not source.is_file() or destination.exists():
            continue
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o600)
        copied.append(str(destination.relative_to(app_home())))
    return copied
