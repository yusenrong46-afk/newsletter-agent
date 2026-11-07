# Security

newsletter-agent handles personal email, so security reports are very welcome.

## Reporting a problem

Please use **GitHub's private vulnerability reporting**: on this repository, go to **Security → Report a vulnerability**. Please don't open a public issue for security problems.

Helpful details include what you found, how to reproduce it (with synthetic data, never real email), and what could happen.

## Scope

Things worth reporting:
- Email content reaching OpenAI outside the documented triage rules
- Email bodies, tokens, keys, or unsubscribe URLs written to disk or logs
- Prompt-injection paths that let email text cause actions beyond a misleading summary
- HTML report or briefing escaping problems, such as script injection from email text
- Any path to Gmail write access

## Design safeguards

The Gmail scope is read-only. The model has no Gmail tools, and email text is treated as untrusted. Local files are owner-only, and HTML output uses a strict Content-Security-Policy. See [docs/privacy.md](docs/privacy.md).
