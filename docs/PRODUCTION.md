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

## Private app on your phone

The app is a private, installable web app (PWA): `api.py` (FastAPI, run by `serve.py`) serves the page in `pwa/`
and the JSON API behind its Swipe, Search and Analytics tabs. Everything except the sign-in page, the app's static
files and the daily-job hook needs a sign-in cookie. Sign-in uses a password set as a Fly secret:

```bash
fly secrets set APP_PASSWORD='something-long' --app event-recommender
```

A sign-in lasts 90 days (`auth.cookie_days` in `config.yaml`); changing `APP_PASSWORD` signs out every device.
Five wrong attempts lock sign-in for 10 minutes.

To install on Android: open https://event-recommender.fly.dev in Chrome (not an in-app browser), sign in, then
Chrome's ⋮ menu → **Install app**. It opens full screen from the home screen.

While the app is on screen it pings the server every 30 s (`swipe.heartbeat_seconds`); about 10 minutes after it's
closed, with no search running, the machine stops itself (`server.idle_shutdown_seconds`).

## Daily job

Once a day the app retrains on your latest ratings, searches the default queries (`search.default_queries` in
`config.yaml`, `daily.max_results` events each, in `search.default_location` for the next
`search.default_days_ahead` days) and pre-computes the SHAP colouring for every unrated card. That's about $0.65 a day.
It runs at most once per New York calendar day; the latest run is in the `daily_runs` table.

A Fly machine on a second, tiny app wakes the main app with `POST /api/cron/daily`, authorised by a shared token.
One-time setup (the token goes straight from a file into both apps' secrets, never onto the screen):

```bash
python3 -c "import secrets; print('CRON_TOKEN=' + secrets.token_urlsafe(32))" > /tmp/cron.env
fly secrets import --app event-recommender < /tmp/cron.env
fly apps create event-recommender-cron --org <org-slug>
fly secrets import --stage --app event-recommender-cron < /tmp/cron.env
rm /tmp/cron.env
fly machine run curlimages/curl:8.10.1 --app event-recommender-cron --region ord --schedule daily \
  sh -c 'curl -fsS --max-time 300 -X POST -H "Authorization: Bearer $CRON_TOKEN" https://event-recommender.fly.dev/api/cron/daily'
```

Fly picks the time of day for `--schedule daily`. To run it now: `fly machine list --app event-recommender-cron`, then
`fly machine start <id> --app event-recommender-cron`. Without `CRON_TOKEN` the endpoint refuses every call, so the
daily job is simply off.

## Re-extracting context and cleaning duplicates

New searches extract context and delete generic listings and duplicates on their own. After changing
the context prompt, the theme list or `context.duplicate_similarity` in `config.yaml`, redo it for every
event already stored (Haiku, about 2¢ per 265 events):

```bash
fly ssh console --app event-recommender -C "sh -c 'cd /app && python -u refresh_context.py > /app/data/refresh.log 2>&1'"
fly ssh console --app event-recommender -C "cat /app/data/refresh.log"
```

It re-extracts everything (keeping contexts you edited), deletes generic listings and duplicates, re-embeds,
recomputes the map and retrains. The log lists every merge and any event whose copies had conflicting labels
(those go back to the Swipe queue). Write to a log file as above: the run takes several minutes and the SSH session can
drop before printed output comes back. Back up first (see below).

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
