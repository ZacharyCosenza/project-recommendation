"""Re-extract every event's context, theme and city, re-embed as description + context, then rebuild the map and model.

Run after changing what goes into the context: python refresh_context.py
A context the user edited (it differs from the last auto-extracted one) is kept as is.
"""
from collections import defaultdict

import numpy as np

from recommender import context, db, model, viz


def main():
    conn = db.get_connection()
    db.init_db(conn)
    rows = conn.execute(
        "SELECT e.url, e.description, e.context, e.context_auto, e.found_at, r.location "
        "FROM events e JOIN run_log r ON r.id = e.run_id"
    ).fetchall()
    by_location = defaultdict(list)
    for r in rows:
        by_location[r["location"]].append(r)
    print(f"{len(rows)} events across {len(by_location)} search locations")

    cost, kept = 0.0, 0
    for location, group in by_location.items():
        extracted, usage = context.extract([r["description"] for r in group], group[0]["found_at"][:4], location)
        cost += usage["cost_usd"]
        updates = []
        for r, ex in zip(group, extracted):
            edited = r["context_auto"] and r["context"] != r["context_auto"]
            kept += bool(edited)
            updates.append((r["context"] if edited else ex["context"], ex["context"], ex["theme"],
                            ex["city"] or "unknown", r["url"]))
        conn.executemany("UPDATE events SET context=?, context_auto=?, theme=?, city=? WHERE url=?", updates)
        conn.commit()
    print(f"extracted for {len(rows)} events (kept {kept} user-edited contexts), ${cost:.4f}")

    rows = conn.execute("SELECT url, description, context FROM events").fetchall()
    embs = model.embed_events([r["description"] for r in rows], [r["context"] for r in rows])
    conn.executemany("UPDATE events SET embedding=?, embedding_dim=? WHERE url=?",
                     [(e.astype(np.float32).tobytes(), len(e), r["url"]) for e, r in zip(embs, rows)])
    conn.execute("DELETE FROM explanations")
    conn.execute("DELETE FROM map_layout")
    conn.commit()
    print(f"re-embedded {len(rows)} events")

    viz.refresh_map_layout(conn)
    print("map layout recomputed")
    meta = model.fit_and_persist(conn)
    print(f"classifier trained on {meta['n_events']} events: {meta['label_counts']}")


if __name__ == "__main__":
    main()
