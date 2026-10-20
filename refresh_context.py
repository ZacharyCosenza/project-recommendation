"""Fill in each event's context, re-embed every event as description + context, then rebuild the map and model.

Run once after upgrading to context-aware embeddings: python refresh_context.py [--overwrite]
Events that already have a context keep it unless --overwrite is given.
"""
import argparse

import numpy as np

from recommender import context, db, model, viz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="re-extract contexts that are already filled in")
    args = parser.parse_args()

    conn = db.get_connection()
    db.init_db(conn)
    rows = conn.execute("SELECT url, description, context, found_at FROM events").fetchall()
    todo = [r for r in rows if args.overwrite or not r["context"]]
    print(f"{len(rows)} events, extracting context for {len(todo)}")

    if todo:
        contexts, usage = context.extract([r["description"] for r in todo], todo[0]["found_at"][:4])
        conn.executemany("UPDATE events SET context=? WHERE url=?", [(c, r["url"]) for c, r in zip(contexts, todo)])
        conn.commit()
        filled = sum(bool(c) for c in contexts)
        print(f"context filled for {filled}/{len(todo)} events, ${usage['cost_usd']:.4f}")

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
