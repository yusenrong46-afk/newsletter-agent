# Connect your Gmail (one-time setup, about 10 minutes)

newsletter-agent uses **your own** Google Cloud OAuth client. Your mail goes from Google straight to your computer, and nobody else's app is involved. You'll create a small, free Google Cloud project that only you use.

## 1. Create a project and enable the Gmail API

1. Open <https://console.cloud.google.com/> and sign in with the Gmail account you want to inventory.
2. Create a new project (top bar → project picker → **New project**). Any name works, for example `my-newsletter-agent`.
3. Go to **APIs & Services → Library**, search for **Gmail API**, and click **Enable**.

## 2. Configure the consent screen (testing mode)

1. Go to **APIs & Services → OAuth consent screen** (also called **Google Auth Platform → Branding / Audience**).
2. User type: **External**. App name: anything, for example `newsletter-agent (personal)`. Add your email as the support and developer contact.
3. **Data access / Scopes:** add `https://www.googleapis.com/auth/gmail.readonly`. That's the only scope needed.
4. **Audience / Test users:** add your own Gmail address.
5. Leave the app in **Testing**. You don't need to publish it or go through Google verification, because only you use it.

> **Re-authorizing:** Google may expire testing-mode authorizations after about 7 days. If commands report `auth_expired`, run `newsletter-agent login` again.

## 3. Create a Desktop OAuth client

1. Go to **APIs & Services → Credentials → Create credentials → OAuth client ID**.
2. Application type: **Desktop app**.
3. Download the JSON file and save it as:

```
~/.newsletter-agent/credentials/client_secret.json
```

(`newsletter-agent setup` creates the folder with private permissions and prints the exact path.)

## 4. Authorize

```bash
newsletter-agent login
```

A browser window opens. Choose your account and approve **read-only** Gmail access. Google will show a warning because the app is unverified; that's expected for a personal testing-mode app. The token is saved to `~/.newsletter-agent/credentials/token.json`, readable only by you.

Then check the connection:

```bash
newsletter-agent check --gmail
```

## Revoking access

Visit <https://myaccount.google.com/permissions> and remove the app, and delete `~/.newsletter-agent/credentials/token.json`.
