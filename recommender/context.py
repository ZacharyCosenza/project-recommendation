from datetime import date

import anthropic
from pydantic import BaseModel

from . import config

cfg = config.context
client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

PROMPT = """For each numbered event description below, extract six fields.

- date: the first calendar date the event happens on, as YYYY-MM-DD. For a run, exhibition or festival, the date it opens. Use {year} when the description gives no year. Leave it empty if the description gives no date at all (or only a whole month).
- end_date: the last date, as YYYY-MM-DD, if the event spans several days (a run, exhibition, multi-day festival). Leave it empty for a single-day event.
- event_type: a short lowercase noun phrase for what kind of event it is, e.g. "comedy show", "orchestra concert", "street festival", "art exhibition", "baseball game". Leave it empty if you can't tell.
- theme: exactly one of: {themes}. Use "other" if none fits.
- city: the city the event takes place in, as its common full name without state or country (e.g. "New York City", "Chicago", "Monterrey"). For a borough or neighborhood, give the city it belongs to (Brooklyn → "New York City"). Leave it empty if the description doesn't say.{hint}
- specific: true if the description is about one particular event (a show, game, festival, exhibition) on a known date or date range. false if it is a venue page, calendar, season schedule, "things to do" list or community page that doesn't settle on one event.

Never guess: an empty string is better than an invented value.

{events}"""


class EventContext(BaseModel):
    index: int
    date: str
    end_date: str
    event_type: str
    theme: str
    city: str
    specific: bool


class EventContexts(BaseModel):
    events: list[EventContext]


def usable_location(location):
    return location if location and location.strip().lower() not in ("n/a", "na", "none", "any", "-") else ""


def weekdays(start, end):
    """"Friday", "Friday to Sunday", or "any day" for a run of a week or more; "" without a usable start date."""
    try:
        first = date.fromisoformat(start)
    except ValueError:
        return ""
    try:
        last = date.fromisoformat(end) if end else first
    except ValueError:
        last = first
    if (last - first).days >= 6:
        return "any day"
    return f"{first:%A} to {last:%A}" if last > first else f"{first:%A}"


def format_context(start, end, event_type, theme, city):
    return ", ".join(part for part in (weekdays(start, end), event_type.strip(), theme, city) if part)


def extract(descriptions, year, location=""):
    """Per description: {"context", "theme", "city", "generic"} ("" where unknown), plus API usage."""
    location = usable_location(location)
    hint = f' The search was for events in "{location}", which may help.' if location else ""
    themes = ", ".join(f'"{t}"' for t in cfg.themes)
    out = [{"context": "", "theme": "other", "city": location, "generic": False} for _ in descriptions]
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
                context = format_context(item.date, item.end_date, item.event_type, theme, city)
                out[start + item.index] = {"context": context, "theme": theme, "city": city,
                                           "generic": not item.specific}
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
    usage["cost_usd"] = (usage["input_tokens"] * cfg.input_cost_per_mtok
                         + usage["output_tokens"] * cfg.output_cost_per_mtok) / 1e6
    return out, usage
