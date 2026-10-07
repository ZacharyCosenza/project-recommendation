# Deploying to Fly.io

## Prerequisites

- `flyctl` installed (https://fly.io/docs/flyctl/install/) and `fly auth login` run once.
- A Brave Search API key and an Anthropic API key with a workspace scope (see `keys.md` for the format used locally).

## First deploy

```bash
cd project-recommendation
fly launch --no-deploy   # confirm app name/region, let it detect fly.toml already present
fly volumes create data_volume --region ord --size 1
fly secrets set ANTHROPIC_KEY=sk-ant-... BRAVE_API=...
fly deploy
```

`fly launch --no-deploy` may offer to rewrite `fly.toml` — check it still has the `[mounts]` block pointing `data_volume` at `/app/data` and the `auto_stop_machines`/`auto_start_machines`/`min_machines_running=0` lines under `[http_service]` before deploying; those are what keep this cheap for single-user, occasional traffic.

## Operating it

- `fly status` — machine state.
- `fly logs` — live app output. A fast crash on startup almost always means a secret is missing (`config.py` raises a `RuntimeError` naming which one).
- `fly dashboard` — billing/usage view. Worth checking after the first week against the ~$3.86/month estimate (shared-cpu-1x 512MB + 1GB volume, always-on; less with auto-stop given low traffic).
- `fly deploy` — redeploy after code changes.
- `fly ssh console` then `ls -la /app/data` — confirm `events.db` is landing on the mounted volume, not the ephemeral root filesystem.

## Publishing a public snapshot

The live app never holds GitHub credentials. To publish:

1. One-time: create a **public** GitHub repo for this project (`gh repo create project-recommendation --public --source=. --push` from this directory, or do it via github.com), then enable Pages under Settings → Pages → Source: GitHub Actions. `.github/workflows/pages.yml` is already in this repo and will pick up from there.
2. In the running app, click **Generate public snapshot**, then **Download snapshot.zip**.
3. Locally: `unzip -o snapshot.zip -d dashboard/`, then `git add dashboard/ && git commit -m "snapshot" && git push`.
4. The Pages workflow runs automatically on that push; the published URL is under the repo's Settings → Pages once the first run completes.
