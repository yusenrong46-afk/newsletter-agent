"""The inventory workflow:

    freeze window -> list IDs -> read metadata -> triage -> preflight
      -> approval -> classify in batches -> count in Python -> report
"""

import asyncio
import sys
from datetime import datetime, timezone

from .classify import build_preflight, classify_pending, confirm_paid_run, make_openai_client, plan_classification
from .gmail_client import describe_session_failure, open_gmail_client
from .html_report import render_html_report
from .report import build_report, render_preflight, render_report
from .scan import (InventoryOptions, InventoryRun, in_scope_records, refresh_unread_flags, scan_mailbox,
                   start_or_resume_state, triage_messages)
from .store import InventoryStore


def finish_run(run: InventoryRun, demo: bool = False) -> dict:
    """Save the checkpoint and reports. Partial runs produce a report too."""
    run.store.save_metadata()
    run.store.save_classifications()
    run.save_checkpoint()
    report = build_report(run)
    if run.options.quiet:
        return report
    text = render_report(report)
    paths = run.store.save_report(report, text, render_html_report(report, demo=demo))
    run.report_paths = [str(path) for path in paths]
    report["report_paths"] = run.report_paths
    if report["run"]["status"] in ("preflight", "awaiting_approval"):
        print(f"Run status: {report['run']['status']}. No email was sent to OpenAI by this step.")
    else:
        print(text)
    print("Saved privately:")
    for path in run.report_paths:
        print(f"  {path}")
    return report


async def run_inventory(gmail, options: InventoryOptions, store: InventoryStore,
                        openai_factory=make_openai_client, confirm=confirm_paid_run,
                        now: datetime | None = None, demo: bool = False) -> dict:
    moment = now or datetime.now(timezone.utc)
    state, resumed = start_or_resume_state(options, store, moment)
    run = InventoryRun(options=options, store=store, state=state, resumed=resumed)
    try:
        await scan_mailbox(run, gmail)

        records = in_scope_records(run)
        triage = triage_messages(records)
        pending = plan_classification(run, records, triage)
        run.preflight = build_preflight(run, records, triage, pending)
        if not options.quiet:
            print(render_preflight(run.preflight))

        if options.preflight:
            run.mode = "preflight"
        elif pending and not run.failures:
            if options.approved or confirm(run.preflight):
                try:
                    openai_client = openai_factory()
                except (ValueError, OSError) as error:
                    run.failures.append({"stage": "openai_setup", "error": "openai_setup", "detail": str(error)})
                else:
                    await classify_pending(run, gmail, openai_client, pending)
            else:
                run.mode = "awaiting_approval"

        await refresh_unread_flags(run, gmail)
    except asyncio.CancelledError:
        # Ctrl-C arrives as a cancellation. Clear it so the partial report can be saved.
        run.interrupted = True
        asyncio.current_task().uncancel()
    return finish_run(run, demo=demo)


async def inventory_with_real_gmail(options: InventoryOptions) -> dict:
    store = InventoryStore()
    store.lock()
    report = None
    try:
        async with open_gmail_client() as gmail:
            report = await run_inventory(gmail, options, store)
    except BaseExceptionGroup:
        # Ctrl-C also stops the server, so closing the connection can fail. The
        # report is already saved by then, and that failure must not hide it.
        if report is None:
            raise
    finally:
        store.unlock()
    return report


def exit_code_for(status: str) -> int:
    if status == "interrupted":
        return 130
    if status == "awaiting_approval" and not sys.stdin.isatty():
        return 2
    return 0 if status in ("complete", "preflight", "awaiting_approval") else 1


def run_inventory_command(options: InventoryOptions) -> int:
    try:
        report = asyncio.run(inventory_with_real_gmail(options))
    except KeyboardInterrupt:
        print("\nInterrupted. Saved work is kept; continue with --resume.", file=sys.stderr)
        return 130
    except BaseExceptionGroup as group:
        print(f"Inventory stopped: {describe_session_failure(group)}", file=sys.stderr)
        return 1
    return exit_code_for(report["run"]["status"])
