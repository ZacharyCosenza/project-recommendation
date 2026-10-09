import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from recommender import background, db, preference, public_snapshot, search, viz

st.set_page_config(page_title="Event Recommender", layout="wide")

SECONDS_PER_QUERY = 30
COST_PER_QUERY = 0.065


@st.cache_resource
def _conn():
    c = db.get_connection()
    db.init_db(c)
    db.fail_stale_runs(c)
    return c


conn = _conn()


def flash(kind, message):
    st.session_state.flash = (kind, message)


def show_flash():
    if "flash" in st.session_state:
        kind, message = st.session_state.pop("flash")
        getattr(st, kind)(message)


def render_event_map(gp, model_meta):
    df = viz.to_frame(db.all_events(conn))
    if len(df) < viz.MIN_MAP_POINTS:
        st.caption(f"The map appears once there are at least {viz.MIN_MAP_POINTS} events (have {len(df)}).")
        return
    X = np.stack(df["embedding"].to_numpy())
    ei = preference.expected_improvement(gp, X, model_meta["y_best"]) if gp is not None else None
    recent_mask = (df["run_id"] == db.latest_run_id(conn)).to_numpy()
    with st.spinner("Laying out event map (first time after new events takes ~20s)..."):
        coords = viz.map_coords(conn, df, X)
    st.plotly_chart(viz.event_map(df, coords, recent_mask, ei), width="content")
    st.caption("Events laid out by description similarity (UMAP, cosine). "
               + ("Background is EI interpolated from every event — a reading aid; the model only scores "
                  "the events themselves, and the darkest areas include places with no events nearby. " if gp is not None else "Train a model to see the EI background. ")
               + "White dots are the most recent run, grey are earlier. Hover a dot for details.")


def model_staleness_text(meta):
    if meta is None:
        return "Model: never trained — click Retrain Model after labeling some events."
    return f"Model: trained on {meta['n_events']} events ({meta['trained_at']})."


def start_search(location, date_start, date_end, query_rows):
    try:
        st.session_state.watching_run = background.start_search(location, date_start, date_end, query_rows)
    except background.SearchAlreadyRunning as e:
        flash("warning", str(e))


def finished_message(run):
    summary = (f"Search finished: {run['total_event_count'] or 0} events ({run['new_event_count'] or 0} new). "
               f"Cost: ${run['cost_usd'] or 0:.4f}")
    if run["status"] == "failed":
        return "error", f"Search failed: {run['error']}"
    if run["error"]:
        return "warning", f"{summary} Some steps failed: {run['error']}"
    return "success", summary


@st.fragment(run_every=5)
def search_status():
    background.heartbeat()
    run = db.active_run(conn)
    if run is not None:
        total = len(json.loads(run["queries_json"])) or 1
        done = run["queries_done"]
        st.progress(done / total, text=(
            f"Searching {run['location']}: {done}/{total} queries done · {run['total_event_count'] or 0} events · "
            f"${run['cost_usd'] or 0:.3f} so far. Runs in the background — you can leave this page."))
        st.session_state.watching_run = run["id"]
        return
    watched = st.session_state.pop("watching_run", None)
    if watched is not None:
        flash(*finished_message(db.get_run(conn, watched)))
        st.rerun(scope="app")


def render_event_table(gp, model_meta, events_df, key):
    if events_df.empty:
        st.info("No events here.")
        return

    if gp is not None:
        X = np.stack(events_df["embedding"].to_numpy())
        ei = preference.expected_improvement(gp, X, model_meta["y_best"])
        events_df = events_df.assign(ei=ei).sort_values("ei", ascending=False)
    else:
        events_df = events_df.assign(ei=np.nan)

    display_df = events_df[["ei", "theme", "description", "label", "url"]].reset_index(drop=True)

    st.data_editor(
        display_df,
        key=key,
        disabled=["ei", "theme", "description", "url"],
        column_config={
            "ei": st.column_config.NumberColumn("EI", format="%.3f"),
            "label": st.column_config.SelectboxColumn("Label", options=[-1, 0, 1], required=True),
            "url": st.column_config.LinkColumn("Link"),
        },
        hide_index=True,
        width="stretch",
    )

    edits = st.session_state.get(key, {}).get("edited_rows", {})
    if edits:
        for row_idx, changes in edits.items():
            if "label" in changes:
                url = display_df.iloc[int(row_idx)]["url"]
                db.update_label(conn, url, int(changes["label"]))
        st.rerun()


gp, model_meta = preference.load_current_model(conn)

st.title("Event Recommender")
show_flash()
search_status()

status_cols = st.columns([2, 2, 1, 1])
status_cols[0].caption(f"Lifetime spend: ${db.total_cost(conn):.4f}")
status_cols[1].caption(model_staleness_text(model_meta))
if status_cols[2].button("Retrain Model"):
    try:
        meta = preference.fit_and_persist(conn)
        flash("success", f"Retrained on {meta['n_events']} events.")
    except ValueError as e:
        flash("warning", str(e))
    st.rerun()
if status_cols[3].button("Generate public snapshot"):
    st.session_state.snapshot_bytes = public_snapshot.build_snapshot(conn)
if "snapshot_bytes" in st.session_state:
    st.download_button("Download snapshot.zip", data=st.session_state.snapshot_bytes,
                        file_name="snapshot.zip", mime="application/zip")

with st.expander("New Search"):
    if "query_rows_df" not in st.session_state:
        st.session_state.query_rows_df = pd.DataFrame(search.DEFAULT_QUERY_ROWS)

    edited_queries = st.data_editor(
        st.session_state.query_rows_df,
        num_rows="dynamic",
        key="query_rows_editor",
        column_config={
            "text": st.column_config.TextColumn("Query", required=True),
            "max_results": st.column_config.NumberColumn("Max results", min_value=1, max_value=10, step=1),
        },
        hide_index=True,
        width="stretch",
    )

    location = st.text_input("Location", value="Chicago")
    date_range = st.date_input("Date range", value=(date.today(), date.today() + timedelta(days=7)))
    date_start, date_end = (date_range if isinstance(date_range, tuple) and len(date_range) == 2
                             else (date_range, date_range))

    query_rows = [r for r in edited_queries.to_dict("records") if str(r.get("text", "")).strip()]
    n_rows = len(query_rows)
    est_minutes = max(1, round(n_rows * SECONDS_PER_QUERY / 60))
    st.warning(f"This calls paid APIs: roughly ${n_rows * COST_PER_QUERY:.2f} and ~{est_minutes} min "
               f"for {n_rows} quer{'y' if n_rows == 1 else 'ies'}. It runs in the background, so you can leave the page.")
    confirmed = st.checkbox("I understand this will call paid APIs")
    busy = db.active_run(conn) is not None
    if busy:
        st.caption("A search is already running; wait for it to finish before starting another.")
    if st.button("Run Search", disabled=busy or not confirmed or n_rows == 0):
        start_search(location, date_start.isoformat(), date_end.isoformat(), query_rows)
        st.rerun()

with st.expander("Run history"):
    runs = db.recent_runs(conn)
    if runs:
        st.dataframe(pd.DataFrame([dict(r) for r in runs]), hide_index=True, width="stretch")
    else:
        st.caption("No runs yet.")

st.header("Event map")
render_event_map(gp, model_meta)

st.header("Events")
view = st.radio("Showing", ["Most recent run", "All events"], horizontal=True)

if view == "Most recent run":
    rows = db.get_latest_run_events(conn)
    render_event_table(gp, model_meta, viz.to_frame(rows), key="recent_editor")
else:
    cities = ["All"] + db.list_cities(conn)
    col1, col2, col3 = st.columns(3)
    city_filter = col1.selectbox("City", cities)
    label_filter = col2.selectbox("Label", ["All", -1, 0, 1])
    page = col3.number_input("Page", min_value=1, value=1)
    rows = db.browse_events(
        conn,
        city=None if city_filter == "All" else city_filter,
        label=None if label_filter == "All" else int(label_filter),
        page=page,
    )
    render_event_table(gp, model_meta, viz.to_frame(rows), key="browse_editor")
