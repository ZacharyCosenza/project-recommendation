import html
import io
import zipfile
from datetime import datetime, timezone

import numpy as np

from . import db, preference, viz

SNAPSHOT_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Event Recommender</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1200px;
        margin: 2rem auto; padding: 0 1rem; color: #222; }}
p.lede {{ color: #444; max-width: 760px; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 0.75rem 1.25rem; margin: 1.5rem 0; background: #fafafa; }}
.card span {{ margin-right: 2rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin-top: 1rem; }}
th, td {{ text-align: left; padding: 0.45rem 0.6rem; border-bottom: 1px solid #eee; vertical-align: top; }}
th {{ background: #f3f3f3; }}
td:first-child, td:nth-child(5) {{ white-space: nowrap; font-variant-numeric: tabular-nums; }}
.muted {{ color: #777; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>Event Recommender</h1>
<p class="lede">An agent searches the web for upcoming events, each event description is embedded, and a
Gaussian process trained on my likes and dislikes scores new events by Expected Improvement — how likely each
one is to beat the best thing I've liked so far, rewarding events the model is still unsure about.</p>
<div class="card">
<span><b>Model trained:</b> {trained_at}</span>
<span><b>Events in training:</b> {n_labeled}</span>
<span><b>Labels (-1 / 0 / +1):</b> {label_dist}</span>
</div>
<h2>Event map</h2>
<p class="muted">Each dot is an event, placed by how similar its description is to the others (UMAP, cosine
distance). The background is Expected Improvement interpolated from every event — brighter means more worth
trying next; the darkest areas include places with no events nearby. White dots are the most recent search, grey are earlier ones. Hover a dot for details.</p>
{map_html}
<h2>Most recent search, ranked by EI</h2>
{table_html}
<p class="muted">Snapshot generated {generated_at}</p>
</body>
</html>
"""


def build_snapshot(conn):
    gp, model_meta = preference.load_current_model(conn)
    df = viz.to_frame(db.all_events(conn))
    recent_id = db.latest_run_id(conn)

    if df.empty:
        map_html, table_html = "<p>No events yet.</p>", ""
    else:
        X = np.stack(df["embedding"].to_numpy())
        ei = preference.expected_improvement(gp, X, model_meta["y_best"]) if gp is not None else None
        recent_mask = (df["run_id"] == recent_id).to_numpy()

        if len(df) >= viz.MIN_MAP_POINTS:
            fig = viz.event_map(df, viz.map_coords(conn, df, X), recent_mask, ei)
            map_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
        else:
            map_html = f"<p>Need at least {viz.MIN_MAP_POINTS} events to draw the map.</p>"
        recent_ei = ei[recent_mask] if ei is not None else None
        table_html = viz.events_table_html(df[recent_mask], recent_ei)

    label_dist = " / ".join(str(model_meta["label_counts"].get(k, 0)) for k in ("-1", "0", "1")) if model_meta else "—"
    index_html = SNAPSHOT_TEMPLATE.format(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        trained_at=html.escape(model_meta["trained_at"][:16].replace("T", " ")) if model_meta else "not yet",
        n_labeled=model_meta["n_events"] if model_meta else 0,
        label_dist=label_dist,
        map_html=map_html,
        table_html=table_html,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("index.html", index_html)
        z.writestr(".nojekyll", "")
    return buf.getvalue()
