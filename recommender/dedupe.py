"""Merge listings of the same event found on different sites (venue page, ticket sellers, news, guides).

Embedding similarity shortlists candidates, then Claude decides: similarity alone can't tell
"Blackhawks vs Penguins" from "Blackhawks vs Hurricanes", or match the same festival covered by two outlets.
"""
import numpy as np
from pydantic import BaseModel
from scipy.sparse.csgraph import connected_components

from . import context, db

cfg = context.cfg

PROMPT = """These event listings came from web searches and look alike. Group the ones that are the same event: the same thing happening at the same venue. A multi-day run or festival is one event; different games, matches or acts are different events.

For each group of two or more, give the index to keep (prefer the organizer's or venue's own page, then the most specific description) and the indices of its duplicates. Leave out listings that have no duplicate.

{events}"""


class SameEvent(BaseModel):
    keep: int
    duplicates: list[int]


class SameEvents(BaseModel):
    groups: list[SameEvent]


def candidate_groups(rows, only_urls=None):
    """Sets of similar events; with only_urls, just the sets that include one of them."""
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    _, component = connected_components(X @ X.T >= cfg.duplicate_similarity, directed=False)
    groups = [np.flatnonzero(component == c) for c in np.unique(component)]
    return [[rows[i] for i in g] for g in groups
            if len(g) > 1 and (only_urls is None or any(rows[i]["url"] in only_urls for i in g))]


def judge(group):
    """[(keep_url, [duplicate_urls])] for one candidate group, plus API cost."""
    listing = "\n\n".join(f"[{i}] {r['url']}\n{r['description']}" for i, r in enumerate(group))
    response = context.client.messages.parse(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        output_config={"effort": cfg.effort},
        messages=[{"role": "user", "content": PROMPT.format(events=listing)}],
        output_format=SameEvents,
    )
    cost = (response.usage.input_tokens * cfg.input_cost_per_mtok
            + response.usage.output_tokens * cfg.output_cost_per_mtok) / 1e6
    merges, used = [], set()
    for g in response.parsed_output.groups if response.parsed_output else []:
        dups = {i for i in g.duplicates if 0 <= i < len(group)} - {g.keep} - used
        if 0 <= g.keep < len(group) and g.keep not in used and dups:
            used |= dups | {g.keep}
            merges.append((group[g.keep]["url"], [group[i]["url"] for i in sorted(dups)]))
    return merges, cost


def dedupe(conn, only_urls=None):
    """Merge duplicates (the copies are deleted). Returns (merged groups, label conflicts, API cost)."""
    rows = db.all_events(conn)
    if len(rows) < 2:
        return [], [], 0.0
    merged, conflicts, cost = [], [], 0.0
    for group in candidate_groups(rows, only_urls):
        merges, c = judge(group)
        cost += c
        for keep, dups in merges:
            if db.merge_duplicates(conn, keep, dups):
                conflicts.append(keep)
            merged.append((keep, dups))
    return merged, conflicts, cost
