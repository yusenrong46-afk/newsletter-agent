# Privacy: exactly what goes where

## Access

| | |
|---|---|
| Gmail permission | `gmail.readonly` only. The code cannot send, draft, delete, label, mark as read or unsubscribe |
| Gmail access path | A local MCP server on your computer with two tools, `search_emails` and `read_email` |
| Tracking | No link, image or unsubscribe URL in an email is ever fetched or visited |

## What is sent to OpenAI

Nothing is sent until you approve: interactively, with `--yes`, or once with `newsletter-agent schedule`, which asks for consent. Each run has a hard budget.

| Triage result (from headers only) | Sent for classification |
|---|---|
| Has mailing-list evidence (`List-Id`, `List-Unsubscribe`, `Precedence: bulk`, `Feedback-ID`, campaign headers, or a Promotions/Updates/Forums label) | Sender, subject, those signals, and up to 2,000 body characters with links removed |
| 3+ messages from one address, no such evidence | Sender, subject, and signals; **no body** |
| Neither | **Nothing** |

- **Briefings** send up to 6,000 characters (links removed) of each new message already classified as an editorial newsletter.
- **Chat** sends your questions, plus totals and lists from your saved report. It sends summaries of specific issues only after you confirm in the terminal.
- **Retention:** all requests use `store=False`, which asks OpenAI not to store the responses. OpenAI's own data policies still apply.
- **Prompt injection:** every prompt labels email text as untrusted data. The classifier and the summarizer have no tools.

## What is stored on your computer

Everything is in `~/.newsletter-agent/`, or the folder set by `NEWSLETTER_AGENT_HOME`. The folders are mode 0700 and the files are 0600.

| Path | Contents |
|---|---|
| `credentials/` | Your OAuth client, Gmail token, and OpenAI key |
| `inventory/newsletter.db` | A SQLite database (plus its `-wal`/`-shm` files) with three tables: **metadata** (sender, subject, date, labels, and List-ID; `List-Unsubscribe` is stored only as yes/no, because the URL is often personal), **classifications** (labels, publication names, and short rationales), and **kv** (checkpoints for `--resume`). Records older than `retention_days` (default 180) are deleted after a complete scan, but never inside the window of a saved inventory or briefing |
| `inventory/*.json.migrated` | Backups of the JSON caches used by version 0.1, kept after their one-time import into `newsletter.db`. Safe to delete |
| `inventory/reports/` | JSON, Markdown, and HTML reports |
| `briefings/` | Briefing pages: summaries, plus the subjects of their sources |
| `logs/` | Output from scheduled briefings (counts, no email content) |
| `evals/` | Results of `newsletter-agent eval`: scores on the bundled invented emails only, never your mail |
| `briefing.cmd` (Windows only) | The small script Task Scheduler runs for the daily briefing |

The daily schedule itself is one small file outside this folder: a LaunchAgent in `~/Library/LaunchAgents/` on macOS, or a service and timer in `~/.config/systemd/user/` on Linux. `newsletter-agent unschedule` removes it.

**Email bodies are never written to disk.** They exist in memory only while a batch is being processed.

To remove everything, delete `~/.newsletter-agent/` and revoke access at <https://myaccount.google.com/permissions>.

## What is printed

The free preflight prints aggregate numbers only. Full reports, which you choose to run, show publication names, senders, and example subjects in your terminal and in the private files.
