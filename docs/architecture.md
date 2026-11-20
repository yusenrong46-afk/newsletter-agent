# Architecture

## Modules

| Module | Responsibility |
|---|---|
| `cli.py` | The `newsletter-agent` command and subcommands |
| `config.py` | Data folder, `config.toml`, OpenAI key, private atomic file writes |
| `store.py` | SQLite caches and checkpoints, migration from v0.1 JSON, retention, the run lock |
| `gmail_server.py` | **The only code that uses Google's libraries.** OAuth, the two MCP tools, MIME → text |
| `gmail_client.py` | MCP client over stdio; turns tool results into small message records |
| `scan.py` | Freeze the window, list every page, read metadata, triage, refresh unread flags |
| `classify.py` | Batches of 25, structured output, validation of the model's answer |
| `report.py` | Grouping, exact counts, reconciliation, text report |
| `html_report.py` | Self-contained HTML pages with escaping and a CSP |
| `inventory.py` | The inventory workflow and its command |
| `briefing.py` | Incremental briefing: select, summarize, validate sources, render, notify |
| `briefing_team.py` | The briefing's three model roles (summarize, fact-check, edit), their shared state, and the checks on each answer |
| `chat.py` | Chat with three local read-only tools |
| `schedule.py` | Daily schedule after consent: launchd (macOS), systemd user timer (Linux), Task Scheduler (Windows) |
| `demo.py` | Synthetic mailbox, offline model stand-in, demo site |
| `evaluate.py` | The quality eval: labelled synthetic dataset (`eval_data/dataset.jsonl`), scoring, results page |

## Inventory flow

1. **Freeze the window.** Start and end are fixed once per run. The Gmail query uses epoch `after:`/`before:` with a day of padding, and Python enforces `start <= internalDate < end` exactly.
2. **List every page** through `search_emails`, following `nextPageToken` until it runs out. IDs are deduplicated. `resultSizeEstimate` is never used as a count.
3. **Read metadata only** through `read_email(mode="metadata")`, with headers matched case-insensitively. Results are cached, since Gmail message content is immutable.
   - **In parallel, in order:** each page's new IDs are read 16 at a time, with up to 8 reads in flight (`READ_CONCURRENCY`), and the results are applied in listing order. The parallel reads only fill the header cache; run state (discovered, unavailable) changes one ID at a time as each is counted, so stopping early (a cap, a failed read, Ctrl-C) leaves nothing half-recorded. Duplicates, the `--max-messages` cap, and "an ID counts only once its read finished" behave exactly as with one-at-a-time reads, and the report does not depend on which read finished first. Body excerpts for classification and briefing texts are read the same way.
   - **Server side:** the two MCP tools are async and run Google's blocking client in worker threads (`SERVER_THREADS = 8`), each with its own Gmail client, because httplib2 is not thread-safe. Eight parallel reads stay well under Gmail's per-user rate limit.
   - **Rate limits:** if a read is still rate-limited after Google's own retries, the client switches to slow mode for the rest of the run (one read at a time, backing off 2, 4, then 8 seconds) before giving up, and the report says so.
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

### Trends and cleanup suggestions

Both are plain Python on dates and unread flags, with no model calls.
- **Weekly counts:** each publication's messages per week from the window start (90 days is 12 weeks plus a 6-day last week). Reconciliation checks that every publication's weekly counts add up to its total.
- **Trend:** `new` (first seen in the last 14 days), `fading` (nothing in the last 30 days, at least 3 before), `rising` or `falling` (comparing the two halves of the window), otherwise `steady`. Windows shorter than 28 days, such as a briefing's, get no trend.
- **Cleanup suggestions:** only observed publications with at least 4 messages. 80% or more unread suggests "consider unsubscribing"; 50 to 79% suggests "skim or reduce". Suggestions are ranked by unread count, then name, so the order is stable. If unread flags could not be refreshed, suggestions are turned off and the report says why. Reasons are built from numbers only, and no unsubscribe link is stored or shown.

### Grouping rules

- **Group key:** the List-ID if present, otherwise the exact sender address. Never the domain or delivery service.
- **What counts as a publication:** only messages labelled `editorial_newsletter`, `promotional_subscription`, or `product_update`. Other mail from the same identity is shown beside the group.
- **Observed vs. provisional:** a group is **observed** if it has a List-ID, or at least 2 messages with `List-Unsubscribe` at medium or high confidence. Otherwise it is **provisional**, and the report gives the reason.
- **Possible duplicates:** groups that share a model-given name are flagged as possibly the same publication, never merged.

## Checkpoints and resume

- **Saving:** caches and checkpoints live in one SQLite database (`inventory/newsletter.db`, WAL mode, `synchronous=FULL`). Each page and each batch is one committed transaction, so a crash or a hard kill loses at most the page or batch in progress, and the database is never left half-written. Version 0.1 rewrote whole JSON files after every page, which grew with the square of the mailbox size.
- **Upgrading:** v0.1's JSON caches are imported once and renamed to `*.json.migrated`, so an upgrade never re-reads Gmail or re-bills the model.
- **Corruption:** only a database that SQLite itself reports as damaged is moved aside to `newsletter.db.corrupt-<time>`, with a warning, and the caches are rebuilt. A database that is merely busy (another run is mid-write) is left alone and reported as "another run is using this data"; other failures, such as a full disk, are reported as they are. Opening an up-to-date database only reads, so it never waits on a running scan.
- **Retention:** after a complete listing, header records and labels older than `retention_days` (default 180) are deleted, except inside any saved checkpoint's window.
- **Resume:** `--resume` keeps the frozen window and lists the mailbox again from page one, because an old page token may be invalid. It reuses cached metadata and classifications, and removes messages that disappeared since the last run.
- **Ctrl-C:** it arrives as a task cancellation. The run saves a partial report, and the MCP subprocess is shut down.
- **Concurrent runs:** a lock file stops two runs (for example a scheduled briefing and a manual inventory) from sharing the caches at the same time.

## Briefing flow

1. **Window:** from the end of the last *successful* briefing to now. The first run covers 24 hours, and the lookback is capped at 3 days.
2. **Scan and classify:** the same scan and classification code, so only new mail is classified. The briefing uses its own checkpoint but shares the caches.
3. **Select:** new `editorial_newsletter` issues, with optional include/exclude filters by publication name.
4. **Summarize:** up to 6,000 characters per issue, 8 issues per call. Each bullet must cite a `source_message_id` from its own batch; otherwise it is dropped and counted. Links are removed from both the input and the output.
5. **Fact-check and edit:** see "The briefing team" below.
6. **Save:** a dated HTML page, `latest.html`, and a counts-only desktop notification (macOS `osascript`, Linux `notify-send`; none on Windows). The window moves forward only after a complete briefing, so a failure or budget stop never skips mail.

## Testing

`uv run pytest` runs entirely offline. The workflow tests use the **real** FastMCP server and protocol, connected in memory to `SyntheticGmail` (a stand-in for Google's client), and a fake OpenAI client. A few tests run the installed OpenAI SDK against a local `httpx.MockTransport` to check the exact request shapes.

## The briefing team

Three model roles run in a fixed order, driven by plain Python (`briefing_team.py`):

| Role | Sees | Returns | Python checks |
|---|---|---|---|
| `summarize` | issue text | bullets that cite an issue | the existing source check |
| `fact_check` | issue text and that issue's bullets | supported or not, per bullet ID | A verdict counts only if it cites a real bullet exactly once. Unsupported bullets are removed and counted; the role can never rewrite one. |
| `edit` | the surviving bullets only, never email text | up to 5 stories: a headline of at most 12 words and the bullet IDs it covers | Bullet IDs must be real and used once; headlines are cleaned and links removed; "covered by N newsletters" is counted by Python. |

- **One shape:** every role is `async def role(state, context) -> state`, and all three read and update one `BriefingState`. That is the shape graph frameworks use, so adopting one later (for loops or run-time routing) would be a small change. A fixed three-step list does not need a framework.
- **Bullet IDs come from Python** (`b1`, `b2`, …). The model only ever cites them, which is what makes its answers checkable.
- **Only the summarizer is essential.** If the fact-checker or the editor fails, is refused, runs out of budget, or is switched off, the briefing is still produced, with a note, and still counts as complete. A bullet with no trustworthy verdict is kept and counted as unchecked.
- **One shared budget**, spent in role order. Defaults are 60 calls and 600,000 tokens per briefing; a saved `config.toml` keeps the cap it already has.
- **Issue text stays in memory.** `BriefingState.to_dict()` leaves it out, and it is cleared after the last role.
- **Chat is unchanged:** it summarizes with the summarizer alone.

## Measuring quality

`newsletter-agent eval` runs the real pipeline (MCP server, scan, triage, classification, validation) on 96 invented emails with known labels: 56 from the demo mailbox (at most 6 per sender) and 40 hand-written hard cases, such as receipts from newsletter senders, a friend writing through a newsletter platform, product updates that read like promotions, non-English issues, prompt-injection attempts, and HTML-only or empty bodies. `evals/build_dataset.py` regenerates the dataset.

It reports:
- **Subscription detection**, scored end to end: mail the privacy triage never sends to the model counts as "not a subscription".
- **Per-category precision, recall and F1**, and a confusion matrix, over the emails that reached the model.
- **Triage coverage:** how many real subscriptions reached the model, and how many personal emails were kept from it or sent as headers only.
- **Publication-name accuracy, unresolved answers, and tokens per 1,000 emails.**
- **The briefing team** (`eval_data/briefing_team.json`): the fact-checker is given 34 bullets, 12 with a planted error (a changed number, an invented cause, a reversed fact, and one issue that tells the checker to approve everything), and is scored on errors caught and true bullets wrongly removed. The editor is given 14 bullets in which 3 stories are covered by more than one newsletter, and is scored over pairs of bullets on whether it grouped the right ones.

Invented, missing or duplicate IDs count as unresolved and lower recall. A budget stop marks the result incomplete. `--real-model` refuses any dataset except the bundled synthetic one, so the eval can never send other email to OpenAI. CI runs the offline stand-in and fails if its scores drop. Results are in `evals/results/`.
