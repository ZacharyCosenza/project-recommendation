from datetime import date

import anthropic
from pydantic import BaseModel

from . import config

cfg = config.context
client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

PROMPT = """For each numbered event description below, extract two fields.

- date: the first specific calendar date the event happens on, as YYYY-MM-DD. Use {year} when the description gives no year. Leave it empty if the description has no specific date (only a date range, a whole month, or nothing).
- event_type: a short lowercase noun phrase for what kind of event it is, e.g. "comedy show", "orchestra concert", "street festival", "art exhibition", "baseball game". Leave it empty if you can't tell.

Never guess: an empty string is better than an invented value.

{events}"""


class EventContext(BaseModel):
    index: int
    date: str
    event_type: str


class EventContexts(BaseModel):
    events: list[EventContext]


def format_context(day, event_type):
    try:
        weekday = date.fromisoformat(day).strftime("%A") if day else ""
    except ValueError:
        weekday = ""
    return ", ".join(part for part in (weekday, event_type.strip()) if part)


def extract(descriptions, year):
    """Day of week + event type for each description ("" where unknown), plus API usage."""
    contexts = [""] * len(descriptions)
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    for start in range(0, len(descriptions), cfg.batch_size):
        batch = descriptions[start:start + cfg.batch_size]
        listing = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(batch))
        response = client.messages.parse(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            output_config={"effort": cfg.effort},
            messages=[{"role": "user", "content": PROMPT.format(year=year, events=listing)}],
            output_format=EventContexts,
        )
        # A truncated or refused response has no parsed output; those events just keep a blank context.
        for item in response.parsed_output.events if response.parsed_output else []:
            if 0 <= item.index < len(batch):
                contexts[start + item.index] = format_context(item.date, item.event_type)
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
    usage["cost_usd"] = (usage["input_tokens"] * cfg.input_cost_per_mtok
                         + usage["output_tokens"] * cfg.output_cost_per_mtok) / 1e6
    return contexts, usage
