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

## Publishing a public snapshot

The live app never holds GitHub credentials. The GitHub repo (https://github.com/ZacharyCosenza/project-recommendation) and its Pages setup are already in place — `.github/workflows/pages.yml` deploys whatever's in `dashboard/` on every push to `main`. To publish an update:

1. In the running app, click **Generate public snapshot**, then **Download snapshot.zip**.
2. Locally: `unzip -o snapshot.zip -d dashboard/`, then `git add dashboard/ && git commit -m "snapshot" && git push`.
3. The Pages workflow runs automatically on that push; the published URL is under the repo's Settings → Pages.
