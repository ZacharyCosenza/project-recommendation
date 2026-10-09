import html
import json
import re
import threading
from datetime import date

import anthropic
import requests
from anthropic import beta_tool

from . import config, db, model, viz

cfg = config.search
client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_TAG_RE = re.compile(r"<[^>]+>")
_start_lock = threading.Lock()


class SearchAlreadyRunning(Exception):
    pass


def _clean(s):
    return html.unescape(_TAG_RE.sub("", s)).strip() if s else s


def brave_web_search(query, count=cfg.brave_results):
    headers = {"Accept": "application/json", "X-Subscription-Token": config.BRAVE_API_KEY}
    resp = requests.get(BRAVE_ENDPOINT, headers=headers, params={"q": query, "count": count},
                        timeout=cfg.brave_timeout_seconds)
    resp.raise_for_status()
    results = resp.json().get("web", {}).get("results", [])
    return [{"url": r.get("url"), "title": _clean(r.get("title")), "description": _clean(r.get("description"))}
            for r in results]


@beta_tool
def brave_search(query: str, count: int = cfg.brave_results) -> str:
    """Search the web via Brave Search.

    Args:
        query: search query string
        count: number of results to return, max 10
    """
    return json.dumps(brave_web_search(query, count=count))


def refine_search(topic, location, date_start, date_end, max_results):
    today = date.today().isoformat()
    when = f"on {date_start}" if date_start == date_end else f"between {date_start} and {date_end}"
    runner = client.beta.messages.tool_runner(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        tools=[brave_search],
        messages=[{
            "role": "user",
            "content": (
                f"Today's date: {today}\n\n"
                f"Find the best {max_results} current, specific events for: {topic} in {location}, "
                f"happening {when}.\n\n"
                f"Start with a broad brave_search call, then check the results. If they're too generic "
                f"(no confirmed date/time/venue {when}), refine your query with more specific keywords — "
                f"exact dates, a neighborhood, a venue, an event series name you learned from the first "
                f"search — and search again. Call brave_search up to {cfg.max_brave_calls} times total. When "
                f"done, output ONLY a JSON object mapping url to description for the {max_results} best, most "
                f"specific results, no other text. Each description should name the specific event, date/time, "
                f"and venue when known; if specifics still aren't confirmed, say so rather than inventing them."
            ),
        }],
    )
    last, usage = None, {"input_tokens": 0, "output_tokens": 0}
    for message in runner:
        last = message
        usage["input_tokens"] += message.usage.input_tokens
        usage["output_tokens"] += message.usage.output_tokens
    text = next(b.text for b in last.content if b.type == "text")
    text = re.sub(r"^```(?:json)?\n|\n```$", "", text.strip())
    usage["cost_usd"] = (usage["input_tokens"] * cfg.input_cost_per_mtok
                         + usage["output_tokens"] * cfg.output_cost_per_mtok) / 1e6
    results = {url: {"description": d, "theme": topic, "query_text": topic} for url, d in json.loads(text).items()}
    return results, usage


def execute_run(conn, run_id, location, date_start, date_end, rows):
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    errors = []
    try:
        for i, row in enumerate(rows):
            topic = row["text"].strip()
            try:
                results, q_usage = refine_search(topic, location, date_start, date_end, int(row["max_results"]))
            except Exception as e:
                errors.append(f"{topic}: {e}")
            else:
                for k in usage:
                    usage[k] += q_usage[k]
                if results:
                    embs = model.embed_descriptions([v["description"] for v in results.values()])
                    db.insert_events(conn, run_id, location, results, embs)
                db.update_run_progress(conn, run_id, usage)
            db.set_queries_done(conn, run_id, i + 1)

        status = "failed" if errors and len(errors) == len(rows) else "completed"
        try:
            viz.refresh_map_layout(conn)
        except Exception as e:
            errors.append(f"map layout: {e}")
    except Exception as e:
        # anything outside a single query (DB, embedding model) must still close the run out
        errors.append(f"unexpected: {e}")
        status = "failed"

    db.finish_run(conn, run_id, status, usage, error="; ".join(errors) or None)


def start_search(location, date_start, date_end, query_rows):
    # Runs outside the Streamlit script, so reruns, other pages and closed tabs don't stop it.
    rows = [r for r in query_rows if str(r.get("text", "")).strip()]
    with _start_lock:
        conn = db.get_connection()
        if db.active_run(conn) is not None:
            raise SearchAlreadyRunning("A search is already running.")
        run_id = db.start_run(conn, location, date_start, date_end, rows)

    threading.Thread(
        target=execute_run, args=(db.get_connection(), run_id, location, date_start, date_end, rows),
        name=f"search-{run_id}", daemon=True,
    ).start()
    return run_id
