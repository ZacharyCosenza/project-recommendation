import html
import json
import re
from datetime import date

import anthropic
import requests
from anthropic import beta_tool

from .config import ANTHROPIC_API_KEY, BRAVE_API_KEY, INPUT_COST_PER_MTOK, MODEL, OUTPUT_COST_PER_MTOK

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_TAG_RE = re.compile(r"<[^>]+>")

DEFAULT_QUERY_ROWS = [
    {"text": "concerts", "max_results": 5},
    {"text": "classical music concerts", "max_results": 5},
    {"text": "art gallery shows", "max_results": 5},
    {"text": "museum exhibits", "max_results": 5},
    {"text": "block parties and street fairs", "max_results": 5},
    {"text": "comedy shows", "max_results": 5},
    {"text": "theater and broadway shows", "max_results": 5},
    {"text": "food festivals", "max_results": 5},
    {"text": "nightlife and club events", "max_results": 5},
    {"text": "sports games", "max_results": 5},
]


def _clean(s):
    return html.unescape(_TAG_RE.sub("", s)).strip() if s else s


def brave_web_search(query, count=10):
    headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY}
    params = {"q": query, "count": count}
    resp = requests.get(BRAVE_ENDPOINT, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("web", {}).get("results", [])
    return [{"url": r.get("url"), "title": _clean(r.get("title")), "description": _clean(r.get("description"))} for r in results]


@beta_tool
def brave_search(query: str, count: int = 10) -> str:
    """Search the web via Brave Search.

    Args:
        query: search query string
        count: number of results to return, max 10
    """
    return json.dumps(brave_web_search(query, count=count))


def refine_search(topic, location, date_start, date_end, max_results=5):
    today = date.today().isoformat()
    when = f"on {date_start}" if date_start == date_end else f"between {date_start} and {date_end}"
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=4096,
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
                f"search — and search again. Call brave_search up to 4 times total. When done, output "
                f"ONLY a JSON object mapping url to description for the {max_results} best, most specific "
                f"results, no other text. Each description should name the specific event, date/time, and "
                f"venue when known; if specifics still aren't confirmed, say so rather than inventing them."
            ),
        }],
    )
    last, total_input, total_output = None, 0, 0
    for message in runner:
        last = message
        total_input += message.usage.input_tokens
        total_output += message.usage.output_tokens
    text = next(b.text for b in last.content if b.type == "text")
    text = re.sub(r"^```(?:json)?\n|\n```$", "", text.strip())
    return json.loads(text), total_input, total_output


def search_query(topic, location, date_start, date_end, max_results=5):
    output, in_tok, out_tok = refine_search(topic, location, date_start, date_end, max_results=max_results)
    results = {url: {"description": d, "theme": topic, "query_text": topic} for url, d in output.items()}
    cost = in_tok / 1e6 * INPUT_COST_PER_MTOK + out_tok / 1e6 * OUTPUT_COST_PER_MTOK
    return results, {"input_tokens": in_tok, "output_tokens": out_tok, "cost_usd": cost}
