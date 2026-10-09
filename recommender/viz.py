import html
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config, db

cfg = config.event_map
GREY = "#9e9e9e"


def _label(v):
    return f"{v:+d}" if v else "0"


def to_frame(rows):
    return pd.DataFrame([{
        "url": r["url"], "run_id": r["run_id"], "city": r["city"], "theme": r["theme"],
        "description": r["description"], "label": r["label"],
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


def ei_surface(coords, ei, grid=cfg.surface_grid, bandwidth_frac=cfg.surface_bandwidth_frac,
               min_weight=cfg.surface_min_weight):
    # EI only exists at the events (it's computed in embedding space), so the surface is a
    # Gaussian-weighted average of nearby events' EI, fading to 0 where no event is close enough.
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
    z = (w @ ei.astype(np.float32)) / np.maximum(wsum, 1e-12)
    z *= np.minimum(1.0, wsum / min_weight)
    return xs, ys, z.reshape(grid, grid)


def _hover_text(row, ei):
    desc = row["description"]
    if len(desc) > 300:
        desc = desc[:300].rsplit(" ", 1)[0] + "..."
    lines = "<br>".join(html.escape(line) for line in textwrap.wrap(desc, 60))
    score = "—" if ei is None or np.isnan(ei) else f"{ei:.3f}"
    return (f"<b>{html.escape(row['theme'])}</b> · {html.escape(row['city'])}<br>{lines}"
            f"<br>label {_label(row['label'])} · EI {score}")


def event_map(df, coords, recent_mask, ei=None):
    ei_values = ei if ei is not None else np.full(len(df), np.nan)
    hover = [_hover_text(row, e) for (_, row), e in zip(df.iterrows(), ei_values)]
    recent_mask = np.asarray(recent_mask)
    older = ~recent_mask

    fig = go.Figure()
    if ei is not None:
        xs, ys, z = ei_surface(coords, ei)
        fig.add_trace(go.Contour(
            x=xs, y=ys, z=z, colorscale="Viridis", ncontours=14, connectgaps=False,
            # a few high-EI events would otherwise wash the rest of the map into one color
            zmin=0.0, zmax=float(np.quantile(ei, 0.95)),
            contours=dict(coloring="heatmap"), line=dict(width=0.6, color="rgba(255,255,255,0.35)"),
            colorbar=dict(title="EI", thickness=12), hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=coords[older, 0], y=coords[older, 1], mode="markers", name="Earlier runs",
        marker=dict(size=6, color=GREY, opacity=0.6, line=dict(width=0.5, color="white")),
        hovertext=[h for h, m in zip(hover, older) if m], hoverinfo="text",
    ))
    fig.add_trace(go.Scatter(
        x=coords[recent_mask, 0], y=coords[recent_mask, 1], mode="markers", name="Most recent run",
        marker=dict(size=9, color="white", line=dict(width=1.5, color="black")),
        hovertext=[h for h, m in zip(hover, recent_mask) if m], hoverinfo="text",
    ))
    fig.update_layout(
        template="ggplot2", plot_bgcolor="white", width=cfg.width, height=cfg.height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        hoverlabel=dict(align="left"),
    )
    frame = dict(showline=True, mirror=True, linecolor="#8a8a8a", linewidth=1, showticklabels=False,
                 ticks="", showgrid=False, zeroline=False, title=None)
    fig.update_xaxes(**frame)
    fig.update_yaxes(**frame)
    return fig
