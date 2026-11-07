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
| `inventory/metadata_cache.json` | Sender, subject, date, labels, and List-ID. `List-Unsubscribe` is stored only as yes/no, because the URL is often personal |
| `inventory/classification_cache.json` | Labels, publication names, and short rationales |
| `inventory/reports/` | JSON, Markdown, and HTML reports |
| `briefings/` | Briefing pages: summaries, plus the subjects of their sources |
| `logs/` | Output from scheduled briefings (counts, no email content) |

**Email bodies are never written to disk.** They exist in memory only while a batch is being processed.

To remove everything, delete `~/.newsletter-agent/` and revoke access at <https://myaccount.google.com/permissions>.

## What is printed

The free preflight prints aggregate numbers only. Full reports, which you choose to run, show publication names, senders, and example subjects in your terminal and in the private files.
