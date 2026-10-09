# Deploying to Fly.io

## Prerequisites

- `flyctl` installed: `curl -L https://fly.io/install.sh | sh`, then add `~/.fly/bin` to `PATH`.
- A Brave Search API key and an Anthropic API key with a workspace scope (see `keys.md` for the format used locally).
- **A credit card on the Fly account.** On the free trial Fly kills any machine after 5 minutes ("Trial machine stopping" in `fly logs`), which cuts off searches mid-run — a full 10-query search takes ~5 minutes. Add one at https://fly.io/trial.
- Logged in: `fly auth login`. **If your Fly account belongs to an org with SSO enforced**, personal access tokens are blocked (you'll see "Access Tokens cannot be created for your account because an organization you are a member of requires Single Sign On"). Work around it by logging in interactively somewhere with a browser, then minting an org-scoped token instead:
  ```bash
  fly auth login          # completes SSO in the browser
  fly orgs list            # note the org slug, e.g. "personal"
  fly tokens org <org-slug>
  ```
  That prints a token starting with `FlyV1 ...`. Export it as `FLY_API_TOKEN` for any environment (including a headless/CI one) that can't do the interactive browser flow itself — every `fly` command below then works non-interactively. Treat it like any other credential: don't commit it, and revoke it from the Fly dashboard once you're done if it was ever pasted somewhere outside your own terminal.

## First deploy

`fly.toml` is already checked in with the app's config, so skip the `fly launch` wizard (it's built for first-time interactive setup and will prompt for things already decided) and run the pieces directly:

```bash
cd project-recommendation
fly apps create event-recommender --org <org-slug>
fly volumes create data_volume --region ord --size 1 --app event-recommender --yes
fly secrets set ANTHROPIC_KEY=sk-ant-... BRAVE_API=... --app event-recommender
fly deploy --app event-recommender
```

This is the exact sequence that was used for the live deployment — `fly apps create` → `fly volumes create` (must exist before the first deploy, since `fly.toml`'s `[mounts]` block references it) → `fly secrets set` (staged, applied on the next deploy) → `fly deploy` (builds the Docker image and ships it; takes a few minutes the first time because of the sentence-transformers/scikit-learn dependency weight).

If you ever do run `fly launch`, double check it hasn't rewritten `fly.toml` and dropped the `[mounts]` block or the `auto_stop_machines`/`auto_start_machines`/`min_machines_running=0` lines under `[http_service]` — those are what keep this cheap for single-user, occasional traffic.

## Operating it

- `fly status --app event-recommender` — machine state.
- `fly logs --app event-recommender` — live app output. A fast crash on startup almost always means a secret is missing (`config.py` raises a `RuntimeError` naming which one).
- `fly dashboard` — billing/usage view. Worth checking after the first week against the ~$3.86/month estimate (shared-cpu-1x 512MB + 1GB volume, always-on; less with auto-stop given low traffic).
- `fly deploy --app event-recommender` — redeploy after code changes.
- `fly ssh console --app event-recommender` then `ls -la /app/data` — confirm `events.db` is landing on the mounted volume, not the ephemeral root filesystem.

## Sharing the site

The live URL is the showcase: anyone with the link gets a read-only view (EI map, ranked events, live
search progress). Editing — running searches, retraining, labeling, run history — unlocks per browser tab
with a password set as a Fly secret, so it never lives in the code or the repo:

```bash
fly secrets set APP_PASSWORD='something-long' --app event-recommender
```

Without `APP_PASSWORD` the site stays read-only for everyone. Five wrong attempts (from anyone, in any
tab) lock unlocking for 10 minutes. Any tab — visitor or owner — swaps itself for a static "Paused" page
after 15 minutes without mouse/keyboard/scroll activity (`app.idle_pause_seconds` in `config.yaml`), because an open
Streamlit tab keeps reconnecting and would otherwise keep the machine running. Searches already in
progress keep running.

## Pausing and resuming

To pause without losing anything (app name, volume and secrets are kept; only the ~$0.15/month volume is billed):

```bash
fly ssh console --app event-recommender -C "python3 -c \"import sqlite3; s=sqlite3.connect('/app/data/events.db'); d=sqlite3.connect('/tmp/b.db'); s.backup(d)\""
fly ssh sftp get /tmp/b.db data/events_backup_$(date +%F).db --app event-recommender   # optional local backup
fly scale count 0 --app event-recommender --yes
```

With zero machines nothing can wake the app, even with `auto_start_machines` on. To resume, deploy again —
it creates a new machine and reattaches the existing `data_volume`, so all events, labels and models come back:

```bash
fly deploy --app event-recommender
```
