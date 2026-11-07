# Architecture

## Modules

| Module | Responsibility |
|---|---|
| `cli.py` | The `newsletter-agent` command and subcommands |
| `config.py` | Data folder, `config.toml`, OpenAI key, private atomic file writes |
| `gmail_server.py` | **The only code that uses Google's libraries.** OAuth, the two MCP tools, MIME → text |
| `gmail_client.py` | MCP client over stdio; turns tool results into small message records |
| `scan.py` | Freeze the window, list every page, read metadata, triage, refresh unread flags |
| `classify.py` | Batches of 25, structured output, validation of the model's answer |
| `report.py` | Grouping, exact counts, reconciliation, text report |
| `html_report.py` | Self-contained HTML pages with escaping and a CSP |
| `inventory.py` | The inventory workflow and its command |
| `briefing.py` | Incremental briefing: select, summarize, validate sources, render, notify |
| `chat.py` | Chat with three local read-only tools |
| `schedule.py` | launchd install/remove after consent |
| `demo.py` | Synthetic mailbox, offline model stand-in, demo site |

## Inventory flow

1. **Freeze the window.** Start and end are fixed once per run. The Gmail query uses epoch `after:`/`before:` with a day of padding, and Python enforces `start <= internalDate < end` exactly.
2. **List every page** through `search_emails`, following `nextPageToken` until it runs out. IDs are deduplicated. `resultSizeEstimate` is never used as a count.
3. **Read metadata only** through `read_email(mode="metadata")`, with headers matched case-insensitively. Results are cached, since Gmail message content is immutable.
4. **Triage** with Python rules on headers, which decide what may be sent to the model (see [privacy.md](privacy.md)).
5. **Preflight:** counts, cache hits, and a token estimate, with no model calls.
6. **Approval and budget:** one confirmation, then a hard limit on calls and tokens, checked before every call using the full output reservation.
7. **Classify** in batches through the Responses API with a strict Pydantic schema. Every answer is checked against the IDs sent. Invented, missing or duplicate IDs, refusals, incomplete responses and invalid output become *unresolved* items. Only successes are cached.
8. **Count and reconcile.** Totals are calculated from stored records and checked a second way:
   - discovered = in scope + excluded + unavailable
   - in scope = candidates + not candidates
   - candidates = classified + unresolved + pending
   - the categories add up to the classified total
   - the groups add up to the subscription messages
9. **Report** as JSON, Markdown, and static HTML.

### Grouping rules

- **Group key:** the List-ID if present, otherwise the exact sender address. Never the domain or delivery service.
- **What counts as a publication:** only messages labelled `editorial_newsletter`, `promotional_subscription`, or `product_update`. Other mail from the same identity is shown beside the group.
- **Observed vs. provisional:** a group is **observed** if it has a List-ID, or at least 2 messages with `List-Unsubscribe` at medium or high confidence. Otherwise it is **provisional**, and the report gives the reason.
- **Possible duplicates:** groups that share a model-given name are flagged as possibly the same publication, never merged.

## Checkpoints and resume

- **Saving:** state is saved after every page and every batch, with an atomic write (temporary file plus `os.replace`).
- **Resume:** `--resume` keeps the frozen window and lists the mailbox again from page one, because an old page token may be invalid. It reuses cached metadata and classifications, and removes messages that disappeared since the last run.
- **Ctrl-C:** it arrives as a task cancellation. The run saves a partial report, and the MCP subprocess is shut down.
- **Concurrent runs:** a lock file stops two runs (for example a scheduled briefing and a manual inventory) from sharing the caches at the same time.

## Briefing flow

1. **Window:** from the end of the last *successful* briefing to now. The first run covers 24 hours, and the lookback is capped at 3 days.
2. **Scan and classify:** the same scan and classification code, so only new mail is classified. The briefing uses its own checkpoint but shares the caches.
3. **Select:** new `editorial_newsletter` issues, with optional include/exclude filters by publication name.
4. **Summarize:** up to 6,000 characters per issue, 8 issues per call. Each bullet must cite a `source_message_id` from its own batch; otherwise it is dropped and counted. Links are removed from both the input and the output.
5. **Save:** a dated HTML page, `latest.html`, and a counts-only macOS notification. The window moves forward only after a complete briefing, so a failure or budget stop never skips mail.

## Testing

`uv run pytest` runs entirely offline. The workflow tests use the **real** FastMCP server and protocol, connected in memory to `SyntheticGmail` (a stand-in for Google's client), and a fake OpenAI client. A few tests run the installed OpenAI SDK against a local `httpx.MockTransport` to check the exact request shapes.
