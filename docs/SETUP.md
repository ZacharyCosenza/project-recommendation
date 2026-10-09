# Running it locally

## 1. Get the code

```bash
git clone https://github.com/ZacharyCosenza/project-recommendation.git
cd project-recommendation
```

## 2. Get two API keys

- Anthropic API key (needs a workspace-scoped key, not an org-level one — see the error you'll get if it's wrong).
- Brave Search API key: https://brave.com/search/api/

## 3. Put them in `keys.md`

Create a file called `keys.md` in the project root (it's gitignored, never committed):

```
ANTHROPIC_KEY=sk-ant-...
BRAVE_API=...
```

No quotes, no spaces around `=`. Add `APP_PASSWORD=...` too if you want to be able to unlock editing
(searching, retraining, labeling) from the sidebar; without it the app is read-only.

## 4. Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. First run creates `data/events.db` automatically.

Every tunable setting (search model, default queries, GP kernel, map size, idle timeouts, ...) lives in
`config.yaml`; secrets never go there.

## Troubleshooting

- **`RuntimeError: ANTHROPIC_KEY / BRAVE_API missing`** — `keys.md` isn't in the project root, or a line is missing/misspelled. Check it reads exactly `ANTHROPIC_KEY=...` and `BRAVE_API=...`, one per line.
- **sentence-transformers download hangs on first run** — it's pulling the `all-MiniLM-L6-v2` model (~90MB) the first time; subsequent runs use the local cache.
- **"Retrain Model" refuses to run** — needs at least 4 events and at least 2 distinct labels (-1/0/1) in the database. Run a search and label a few events first.

For deploying this somewhere other than your own machine, see `PRODUCTION.md`.
