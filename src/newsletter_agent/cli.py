"""The `newsletter-agent` command."""

import argparse
import asyncio
import sys
import webbrowser
from pathlib import Path

from . import config


def command_setup(args) -> int:
    """Check each setup step and say what to do next. Never prints secrets."""
    if args.import_legacy:
        copied = config.import_legacy_data(Path(args.import_legacy).expanduser())
        print(f"Imported {len(copied)} files into {config.app_home()} (existing files were not overwritten).")
    credentials = config.credentials_dir()
    from .schedule import plist_path
    from .store import latest_report_path

    checks = [
        ("Data folder", True, str(config.app_home())),
        ("Google OAuth client", (credentials / "client_secret.json").exists(),
         f"save your Desktop OAuth client JSON as {credentials / 'client_secret.json'} (docs/setup-google.md)"),
        ("Gmail authorization", (credentials / "token.json").exists(), "run: newsletter-agent login"),
        ("OpenAI API key", config.has_api_key(), f"set OPENAI_API_KEY or put OPENAI_API_KEY=... in {config.env_file()}"),
        ("Inventory report", latest_report_path() is not None, "run: newsletter-agent inventory --preflight"),
        ("Daily briefing schedule", plist_path().exists(), "optional: newsletter-agent schedule 07:00"),
    ]
    for name, ok, hint in checks:
        print(f"  {'✓' if ok else '·'} {name}" + ("" if ok and name != "Data folder" else f": {hint}"))
    return 0


def command_inventory(args) -> int:
    from .inventory import run_inventory_command
    from .scan import InventoryOptions

    if args.days < 1:
        raise ValueError("--days must be at least 1")
    if args.max_messages is not None and args.max_messages < 1:
        raise ValueError("--max-messages must be at least 1")
    return run_inventory_command(InventoryOptions(
        days=args.days, preflight=args.preflight, max_messages=args.max_messages, resume=args.resume,
        approved=args.yes, max_model_calls=args.max_model_calls, max_tokens=args.max_tokens,
    ))


def command_report(args) -> int:
    from .store import latest_report_path

    path = latest_report_path("html")
    if not path:
        print("No report yet. Run: newsletter-agent inventory")
        return 1
    print(path)
    if args.open:
        webbrowser.open(path.as_uri())
    return 0


def command_check(args) -> int:
    from .gmail_client import check_connection, describe_session_failure

    try:
        asyncio.run(check_connection(live=args.gmail))
    except BaseExceptionGroup as group:
        print(f"Check failed: {describe_session_failure(group)}", file=sys.stderr)
        return 1
    return 0


def command_schedule(args) -> int:
    from .briefing import ask_yes_no
    from .schedule import install_schedule

    print(install_schedule(args.time, ask_yes_no))
    return 0


def command_unschedule(args) -> int:
    from .schedule import remove_schedule

    print(remove_schedule())
    return 0


def command_login(args) -> int:
    from .gmail_server import gmail_login

    gmail_login()
    return 0


def command_chat(args) -> int:
    from .chat import run_chat

    return run_chat()


def command_brief(args) -> int:
    from .briefing import run_briefing_command

    return run_briefing_command(approved=args.yes, scheduled=args.scheduled, open_after=args.open)


def command_demo(args) -> int:
    from .demo import run_demo

    return run_demo(args.output, args.real_model, args.open)


def command_mcp_server(args) -> int:
    from .gmail_server import run_gmail_server

    run_gmail_server()
    return 0


def build_parser() -> argparse.ArgumentParser:
    from .scan import DEFAULT_MAX_MODEL_CALLS, DEFAULT_MAX_TOKENS

    parser = argparse.ArgumentParser(prog="newsletter-agent", description="Inventory and brief your Gmail newsletters.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    setup = commands.add_parser("setup", help="Check setup and show the next step")
    setup.add_argument("--import-legacy", metavar="DIR", help="Copy data from the old single-file version in DIR")
    setup.set_defaults(handler=command_setup)

    commands.add_parser("login", help="Authorize read-only Gmail access").set_defaults(handler=command_login)

    check = commands.add_parser("check", help="Test the local MCP server (add --gmail to test Gmail)")
    check.add_argument("--gmail", action="store_true", help="Also search Gmail and read one message's headers")
    check.set_defaults(handler=command_check)

    inventory = commands.add_parser("inventory", help="Find your newsletters and subscriptions")
    inventory.add_argument("--days", type=int, default=90, help="Days to scan, ending now (default 90)")
    inventory.add_argument("--preflight", action="store_true", help="Scan and estimate only; no OpenAI calls")
    inventory.add_argument("--max-messages", type=int, help="Stop after this many in-scope messages (partial)")
    inventory.add_argument("--resume", action="store_true", help="Continue the saved window, reusing finished work")
    inventory.add_argument("--yes", action="store_true", help="Approve sending email content to OpenAI without a prompt")
    inventory.add_argument("--max-model-calls", type=int, default=DEFAULT_MAX_MODEL_CALLS, help="Hard limit per run")
    inventory.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Hard limit per run")
    inventory.set_defaults(handler=command_inventory)

    report = commands.add_parser("report", help="Show (or --open) the latest HTML report")
    report.add_argument("--open", action="store_true", help="Open it in your browser")
    report.set_defaults(handler=command_report)

    commands.add_parser("chat", help="Ask questions about your newsletters").set_defaults(handler=command_chat)

    brief = commands.add_parser("brief", help="Summarize new newsletter issues")
    brief.add_argument("--yes", action="store_true", help="Approve OpenAI use without a prompt")
    brief.add_argument("--open", action="store_true", help="Open the briefing in your browser")
    brief.add_argument("--scheduled", action="store_true", help=argparse.SUPPRESS)
    brief.set_defaults(handler=command_brief)

    schedule = commands.add_parser("schedule", help="Run the briefing daily (macOS), e.g. schedule 07:00")
    schedule.add_argument("time", help="24-hour time, HH:MM")
    schedule.set_defaults(handler=command_schedule)
    commands.add_parser("unschedule", help="Remove the daily briefing").set_defaults(handler=command_unschedule)

    demo = commands.add_parser("demo", help="Try everything on a synthetic mailbox (no accounts needed)")
    demo.add_argument("--output", metavar="DIR", help="Where to write index.html and briefing.html")
    demo.add_argument("--real-model", action="store_true", help="Send the synthetic emails to GPT-6 Luna (your key)")
    demo.add_argument("--open", action="store_true", help="Open the demo report in your browser")
    demo.set_defaults(handler=command_demo)

    commands.add_parser("mcp-server", help="Run the Gmail MCP server (used internally)").set_defaults(
        handler=command_mcp_server)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
