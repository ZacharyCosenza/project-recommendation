import html
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config, db

cfg = config.event_map
# One palette for every chart, matching the SHAP highlight: blue pushes toward a like, red away from it.
BLUE = "#2166ac"
RED = "#b2182b"
GREY = "#9e9e9e"
INK = "#3a3a3a"
FRAME = "#8a8a8a"
GRIDLINE = "#ececec"


def _rgb(hex_color):
    return ",".join(str(int(hex_color[i:i + 2], 16)) for i in (1, 3, 5))


def _layout(fig, **size):
    fig.update_layout(
        template="simple_white", plot_bgcolor="white", font=dict(color=INK), margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0), hoverlabel=dict(align="left"), **size,
    )


def _label(v):
    return f"{v:+d}" if v else "0"


def to_frame(rows):
    return pd.DataFrame([{
        "url": r["url"], "run_id": r["run_id"], "city": r["city"], "theme": r["theme"],
        "description": r["description"], "context": r["context"], "label": r["label"],
        "embedding": np.frombuffer(r["embedding"], dtype=np.float32),
        "found_at": r["found_at"], "created_at": r["created_at"],
    } for r in rows])


def umap_coords(X):
    import umap
    n = len(X)
    reducer = umap.UMAP(
        n_components=2, metric=cfg.umap_metric, n_neighbors=min(cfg.umap_neighbors, n - 1),
        min_dist=cfg.umap_min_dist, random_state=cfg.umap_random_state,
        # spectral init needs a reasonably connected graph; fall back for tiny datasets
        init="spectral" if n > cfg.umap_neighbors else "random",
    )
    return reducer.fit_transform(X)


def map_coords(conn, df, X):
    # UMAP's first call per process spends ~20s compiling numba code, so the layout is computed when
    # the event set changes and stored, rather than on every page load.
    stored = db.get_map_layout(conn)
    urls = df["url"].tolist()
    if stored and set(stored) == set(urls):
        return np.array([stored[u] for u in urls])
    coords = umap_coords(X)
    db.save_map_layout(conn, urls, coords)
    return coords


def refresh_map_layout(conn):
    df = to_frame(db.all_events(conn))
    if len(df) >= cfg.min_points:
        map_coords(conn, df, np.stack(df["embedding"].to_numpy()))


def score_surface(coords, score, neutral, grid=cfg.surface_grid, bandwidth_frac=cfg.surface_bandwidth_frac,
                  min_weight=cfg.surface_min_weight):
    # Scores only exist at the events (they're computed in embedding space), so the surface is a
    # Gaussian-weighted average of nearby events' scores, fading to neutral where no event is close enough.
    lo, hi = coords.min(0), coords.max(0)
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    xs = np.linspace(lo[0], hi[0], grid, dtype=np.float32)
    ys = np.linspace(lo[1], hi[1], grid, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    h = bandwidth_frac * float(np.linalg.norm(hi - lo))
    d2 = ((pts[:, None, :] - coords[None, :, :].astype(np.float32)) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * h * h))
    wsum = w.sum(1)
    z = (w @ score.astype(np.float32)) / np.maximum(wsum, 1e-12)
    z = neutral + (z - neutral) * np.minimum(1.0, wsum / min_weight)
    return xs, ys, z.reshape(grid, grid)


def _hover_text(row, score):
    desc = row["description"]
    if len(desc) > 300:
        desc = desc[:300].rsplit(" ", 1)[0] + "..."
    lines = "<br>".join(html.escape(line) for line in textwrap.wrap(desc, 60))
    shown = "—" if score is None or np.isnan(score) else f"{score:.0%}"
    return (f"<b>{html.escape(row['theme'])}</b> · {html.escape(row['city'])}<br>{lines}"
            f"<br>label {_label(row['label'])} · P(like) {shown}")


def event_map(df, coords, recent_mask, score=None):
    values = score if score is not None else np.full(len(df), np.nan)
    hover = [_hover_text(row, e) for (_, row), e in zip(df.iterrows(), values)]
    recent_mask = np.asarray(recent_mask)
    older = ~recent_mask

    fig = go.Figure()
    if score is not None:
        # White is the average event; blue above it, red below. The top 5% is clipped so a few near-certain
        # likes don't wash the rest of the map out.
        mid = float(score.mean())
        top = max(float(np.quantile(score, 0.95)), mid + 1e-6)
        xs, ys, z = score_surface(coords, score, neutral=mid)
        fig.add_trace(go.Contour(
            x=xs, y=ys, z=z, colorscale=[[0.0, RED], [mid / top, "#f7f7f7"], [1.0, BLUE]], ncontours=14,
            connectgaps=False, zmin=0.0, zmax=top,
            contours=dict(coloring="heatmap"), line=dict(width=0.6, color="rgba(255,255,255,0.35)"),
            colorbar=dict(title="P(like)", thickness=12, tickformat=".0%"), hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=coords[older, 0], y=coords[older, 1], mode="markers", name="Earlier runs",
        marker=dict(size=6, color=GREY, opacity=0.6, line=dict(width=0.5, color="white")),
        hovertext=[h for h, m in zip(hover, older) if m], hoverinfo="text", customdata=df["url"][older],
    ))
    fig.add_trace(go.Scatter(
        x=coords[recent_mask, 0], y=coords[recent_mask, 1], mode="markers", name="Most recent run",
        marker=dict(size=9, color="white", line=dict(width=1.5, color=INK)),
        hovertext=[h for h, m in zip(hover, recent_mask) if m], hoverinfo="text", customdata=df["url"][recent_mask],
    ))
    _layout(fig, width=cfg.width, height=cfg.height)
    frame = dict(showline=True, mirror=True, linecolor=FRAME, linewidth=1, showticklabels=False,
                 ticks="", showgrid=False, zeroline=False, title=None)
    fig.update_xaxes(**frame)
    fig.update_yaxes(**frame)
    return fig


def highlight_html(tokens, values, floor):
    # Shared scale per event, floored so a description with only tiny attributions stays pale.
    scale = max(max((abs(v) for v in values), default=0.0), floor)
    spans = []
    for token, v in zip(tokens, values):
        alpha = 0.65 * min(abs(v) / scale, 1.0)
        rgb = _rgb(BLUE if v > 0 else RED)
        spans.append(f'<span style="background:rgba({rgb},{alpha:.2f});border-radius:3px" '
                     f'title="{v:+.3f}">{html.escape(token)}</span>')
    return f'<div style="line-height:1.9">{"".join(spans)}</div>'


def calibration_plot(p_like, liked, bins=cfg.calibration_bins):
    """Predicted P(like) vs how often those events were actually liked, in equal-count bins."""
    order = np.argsort(p_like)
    groups = [g for g in np.array_split(order, min(bins, len(order))) if len(g)]
    predicted = [p_like[g].mean() for g in groups]
    actual = [liked[g].mean() for g in groups]
    hover = [f"{len(g)} events<br>predicted {p:.0%} · liked {a:.0%}" for g, p, a in zip(groups, predicted, actual)]
    top = max(max(predicted), max(actual), 0.1) * 1.1

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfectly calibrated",
                             line=dict(color=GREY, dash="dash", width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=predicted, y=actual, mode="lines+markers", name="This model",
                             line=dict(color=BLUE, width=2), marker=dict(size=9, line=dict(width=2, color="white")),
                             hovertext=hover, hoverinfo="text"))
    _layout(fig, height=360)
    axis = dict(range=[0, top], tickformat=".0%", showline=True, linecolor=FRAME, showgrid=True, gridcolor=GRIDLINE,
                zeroline=False)
    fig.update_xaxes(title="Predicted P(like), held out", **axis)
    fig.update_yaxes(title="Share actually liked", scaleanchor="x", **axis)
    return fig
