import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go

from . import preference

SNAPSHOT_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Event Recommender — public snapshot</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem; background: #fafafa; }}
.card h2 {{ margin-top: 0; font-size: 1rem; color: #555; }}
</style>
</head>
<body>
<h1>Event Recommender — snapshot</h1>
<p>Generated {generated_at}</p>
<div class="card">
<h2>Model status</h2>
<p>Trained: {trained_at}</p>
<p>Events used in training: {n_labeled}</p>
<p>Label distribution: {label_dist}</p>
</div>
{chart_html}
</body>
</html>
"""


def _short(desc, n=90):
    return desc if len(desc) <= n else desc[:n].rsplit(" ", 1)[0] + "..."


def build_snapshot(conn):
    from . import db

    gp, model_meta = preference.load_current_model(conn)
    events = db.get_latest_run_events(conn)

    if gp is not None and events:
        X = np.stack([np.frombuffer(e["embedding"], dtype=np.float32) for e in events])
        ei = preference.expected_improvement(gp, X, model_meta["y_best"])
    else:
        ei = np.zeros(len(events))

    order = np.argsort(-ei)
    descriptions = [_short(events[i]["description"]) for i in order]
    themes = [events[i]["theme"] for i in order]
    ei_sorted = [float(ei[i]) for i in order]

    fig = go.Figure(go.Bar(
        x=ei_sorted,
        y=descriptions,
        orientation="h",
        customdata=list(zip(themes, descriptions)),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>EI: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="ggplot2",
        title="Top events by Expected Improvement (most recent search run)",
        height=max(400, 24 * len(descriptions)),
        margin=dict(l=320),
        yaxis=dict(autorange="reversed"),
    )
    chart_html = fig.to_html(include_plotlyjs="cdn", full_html=False) if descriptions else "<p>No events yet.</p>"

    index_html = SNAPSHOT_TEMPLATE.format(
        generated_at=datetime.now(timezone.utc).isoformat(),
        trained_at=model_meta["trained_at"] if model_meta else "never",
        n_labeled=model_meta["n_events"] if model_meta else 0,
        label_dist=json.dumps(model_meta["label_counts"]) if model_meta else "{}",
        chart_html=chart_html,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("index.html", index_html)
        z.writestr(".nojekyll", "")
    return buf.getvalue()
