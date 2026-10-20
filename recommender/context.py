from datetime import date

import anthropic
from pydantic import BaseModel

from . import config

cfg = config.context
client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

PROMPT = """For each numbered event description below, extract four fields.

- date: the first specific calendar date the event happens on, as YYYY-MM-DD. Use {year} when the description gives no year. Leave it empty if the description has no specific date (only a date range, a whole month, or nothing).
- event_type: a short lowercase noun phrase for what kind of event it is, e.g. "comedy show", "orchestra concert", "street festival", "art exhibition", "baseball game". Leave it empty if you can't tell.
- theme: exactly one of: {themes}. Use "other" if none fits.
- city: the city the event takes place in, as its common full name without state or country (e.g. "New York City", "Chicago", "Monterrey"). For a borough or neighborhood, give the city it belongs to (Brooklyn → "New York City"). Leave it empty if the description doesn't say.{hint}

Never guess: an empty string is better than an invented value.

{events}"""


class EventContext(BaseModel):
    index: int
    date: str
    event_type: str
    theme: str
    city: str


class EventContexts(BaseModel):
    events: list[EventContext]


def usable_location(location):
    return location if location and location.strip().lower() not in ("n/a", "na", "none", "any", "-") else ""


def format_context(day, event_type, theme, city):
    try:
        weekday = date.fromisoformat(day).strftime("%A") if day else ""
    except ValueError:
        weekday = ""
    return ", ".join(part for part in (weekday, event_type.strip(), theme, city) if part)


def extract(descriptions, year, location=""):
    """Per description: {"context", "theme", "city"} ("" where unknown), plus API usage."""
    location = usable_location(location)
    hint = f' The search was for events in "{location}", which may help.' if location else ""
    themes = ", ".join(f'"{t}"' for t in cfg.themes)
    out = [{"context": "", "theme": "other", "city": location} for _ in descriptions]
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    for start in range(0, len(descriptions), cfg.batch_size):
        batch = descriptions[start:start + cfg.batch_size]
        listing = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(batch))
        response = client.messages.parse(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            output_config={"effort": cfg.effort},
            messages=[{"role": "user", "content": PROMPT.format(year=year, themes=themes, hint=hint, events=listing)}],
            output_format=EventContexts,
        )
        # A truncated or refused response has no parsed output; those events keep the blank defaults.
        for item in response.parsed_output.events if response.parsed_output else []:
            if 0 <= item.index < len(batch):
                theme = item.theme if item.theme in cfg.themes else "other"
                city = item.city.strip() or location
                out[start + item.index] = {"context": format_context(item.date, item.event_type, theme, city),
                                           "theme": theme, "city": city}
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
    usage["cost_usd"] = (usage["input_tokens"] * cfg.input_cost_per_mtok
                         + usage["output_tokens"] * cfg.output_cost_per_mtok) / 1e6
    return out, usage
