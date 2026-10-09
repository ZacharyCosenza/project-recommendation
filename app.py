import json
from datetime import date, timedelta
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st

from recommender import config, db, model, search, session, viz

st.set_page_config(page_title="Event Recommender", layout="wide")

cfg = config.app


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


def owner_panel():
    with st.sidebar:
        st.subheader("Owner")
        if not session.owner_mode_available():
            st.caption("Editing is disabled: no APP_PASSWORD is set.")
            return False
        if st.session_state.get("owner"):
            st.caption("Editing unlocked for this tab.")
            if st.button("Lock"):
                st.session_state.owner = False
                st.rerun()
            return True
        with st.form("unlock", clear_on_submit=True):
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Unlock editing")
        if submitted:
            result = session.try_unlock(password)
            if result == "ok":
                st.session_state.owner = True
                st.rerun()
            st.error("Too many wrong attempts. Try again in a few minutes." if result == "locked"
                     else "Wrong password.")
        return False


def idle_pause():
    # Swap the tab for a static page after inactivity: an open Streamlit tab keeps reconnecting and
    # would wake the machine forever. Component iframes may not navigate the top page themselves, so
    # this injects the timer into the parent document (allowed: the iframe is same-origin).
    limit_ms = cfg.idle_pause_seconds * 1000
    st.iframe(f"""<script>
const p = window.parent;
if (!p.__idlePauseInstalled) {{
  p.__idlePauseInstalled = true;
  const s = p.document.createElement("script");
  s.textContent = `(function () {{
    let last = Date.now();
    const bump = () => {{ last = Date.now(); }};
    ["mousemove", "mousedown", "keydown", "wheel", "scroll", "touchstart"].forEach(
      (e) => window.addEventListener(e, bump, {{ passive: true, capture: true }}));
    setInterval(() => {{
      if (Date.now() - last > {limit_ms}) location.replace("{cfg.paused_page}");
    }}, 15000);
  }})();`;
  p.document.head.appendChild(s);
}}
</script>""", height=1)


def render_event_map(gp, model_meta):
    df = viz.to_frame(db.all_events(conn))
    if len(df) < config.event_map.min_points:
        st.caption(f"The map appears once there are at least {config.event_map.min_points} events (have {len(df)}).")
        return
    X = np.stack(df["embedding"].to_numpy())
    ei = model.expected_improvement(gp, X, model_meta["y_best"]) if gp is not None else None
    recent_mask = (df["run_id"] == db.latest_run_id(conn)).to_numpy()
    with st.spinner("Laying out event map (first time after new events takes ~20s)..."):
        coords = viz.map_coords(conn, df, X)
    st.plotly_chart(viz.event_map(df, coords, recent_mask, ei), width="content")
    st.caption("Events laid out by description similarity (UMAP, cosine). "
               + ("Background is EI interpolated from every event — a reading aid; the model only scores "
                  "the events themselves, and the darkest areas include places with no events nearby. "
                  if gp is not None else "The EI background appears once a model is trained. ")
               + "White dots are the most recent search, grey are earlier. Hover a dot for details.")


def model_status_text(meta):
    if meta is None:
        return "Model: not trained yet."
    return f"Model: trained on {meta['n_events']} events ({meta['trained_at'][:16].replace('T', ' ')} UTC)."


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
    session.heartbeat()
    run = db.active_run(conn)
    if run is not None:
        total = len(json.loads(run["queries_json"])) or 1
        done = run["queries_done"]
        st.progress(done / total, text=(
            f"Live search in {run['location']}: {done}/{total} queries done · "
            f"{run['total_event_count'] or 0} events found so far."))
        st.session_state.watching_run = run["id"]
        return
    watched = st.session_state.pop("watching_run", None)
    if watched is not None:
        flash(*finished_message(db.get_run(conn, watched)))
        st.rerun(scope="app")


LABEL_ICON = {1: "👍", -1: "👎", 0: "—"}
BRUSHES = {"👍 Like": 1, "👎 Dislike": -1, "Clear": 0}


def apply_brush(key, urls):
    if not st.session_state.get("owner"):
        return
    rows = st.session_state[key].selection.rows
    brush = st.session_state.get("brush")
    if rows and brush:
        for i in rows:
            db.update_label(conn, urls[i], BRUSHES[brush])


def render_event_table(gp, model_meta, events_df, editable, key):
    if events_df.empty:
        st.info("No events here.")
        return

    if gp is not None:
        X = np.stack(events_df["embedding"].to_numpy())
        ei = model.expected_improvement(gp, X, model_meta["y_best"])
        events_df = events_df.assign(ei=ei).sort_values("ei", ascending=False)
    else:
        events_df = events_df.assign(ei=np.nan)
    events_df = events_df.reset_index(drop=True)

    table = pd.DataFrame({
        "Label": events_df["label"].map(LABEL_ICON),
        "EI": events_df["ei"],
        "Theme": events_df["theme"],
        "City": events_df["city"],
        "Event": events_df["description"],
        "Link": events_df["url"].where(events_df["url"].str.startswith(("http://", "https://")), None),
    })
    column_config = {
        "Label": st.column_config.TextColumn("Label", width="small"),
        "EI": st.column_config.NumberColumn("EI", format="%.3f", width="small"),
        "Event": st.column_config.TextColumn("Event", width="large"),
        "Link": st.column_config.LinkColumn("Link", display_text="open", width="small"),
    }

    if not editable:
        st.dataframe(table, column_config=column_config, hide_index=True, height=cfg.table_height)
        return

    st.segmented_control("Clicking a row marks it as", list(BRUSHES), key="brush", default="👍 Like", required=True)
    # Selections report rows by their position in `table`, even after the viewer re-sorts columns.
    st.dataframe(table, column_config=column_config, hide_index=True, height=cfg.table_height, key=key,
                 on_select=partial(apply_brush, key, events_df["url"].tolist()), selection_mode="single-row")


@st.fragment
def events_section(gp, model_meta, editable):
    # A fragment, so labeling a row reruns only this table instead of the whole page (map included).
    view = st.radio("Showing", ["Most recent search", "All events"], horizontal=True)
    if view == "Most recent search":
        rows = db.get_latest_run_events(conn)
    else:
        col1, col2, col3 = st.columns(3)
        city_filter = col1.selectbox("City", ["All"] + db.list_cities(conn))
        label_filter = col2.selectbox("Label", ["All", "👍 liked", "neutral", "👎 disliked"])
        page = col3.number_input("Page", min_value=1, value=1)
        rows = db.browse_events(
            conn,
            city=None if city_filter == "All" else city_filter,
            label={"All": None, "👍 liked": 1, "neutral": 0, "👎 disliked": -1}[label_filter],
            page=page,
            page_size=cfg.page_size,
        )
    render_event_table(gp, model_meta, viz.to_frame(rows), editable,
                       key="recent_table" if view == "Most recent search" else "browse_table")


def owner_controls():
    status_cols = st.columns([3, 1])
    status_cols[0].caption(f"Lifetime API spend: ${db.total_cost(conn):.4f}")
    if status_cols[1].button("Retrain Model"):
        try:
            meta = model.fit_and_persist(conn)
            flash("success", f"Retrained on {meta['n_events']} events.")
        except ValueError as e:
            flash("warning", str(e))
        st.rerun()

    with st.expander("New Search"):
        if "query_rows_df" not in st.session_state:
            st.session_state.query_rows_df = pd.DataFrame(config.search.default_queries)

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

        location = st.text_input("Location", value=config.search.default_location)
        date_range = st.date_input("Date range", value=(date.today(), date.today() + timedelta(days=config.search.default_days_ahead)))
        date_start, date_end = (date_range if isinstance(date_range, tuple) and len(date_range) == 2
                                 else (date_range, date_range))

        query_rows = [r for r in edited_queries.to_dict("records") if str(r.get("text", "")).strip()]
        n_rows = len(query_rows)
        est_minutes = max(1, round(n_rows * cfg.seconds_per_query / 60))
        st.warning(f"This calls paid APIs: roughly ${n_rows * cfg.cost_per_query:.2f} and ~{est_minutes} min for "
                   f"{n_rows} quer{'y' if n_rows == 1 else 'ies'}. It runs in the background, so you can leave the page.")
        confirmed = st.checkbox("I understand this will call paid APIs")
        busy = db.active_run(conn) is not None
        if busy:
            st.caption("A search is already running; wait for it to finish before starting another.")
        if st.button("Run Search", disabled=busy or not confirmed or n_rows == 0):
            try:
                st.session_state.watching_run = search.start_search(
                    location, date_start.isoformat(), date_end.isoformat(), query_rows)
            except search.SearchAlreadyRunning as e:
                flash("warning", str(e))
            st.rerun()

    with st.expander("Run history"):
        runs = db.recent_runs(conn)
        if runs:
            st.dataframe(pd.DataFrame([dict(r) for r in runs]), hide_index=True, width="stretch")
        else:
            st.caption("No runs yet.")


is_owner = owner_panel()
gp, model_meta = model.load_current_model(conn)

st.title("Event Recommender")
st.markdown(
    "An agent searches the web for upcoming events, each event description is embedded, and a Gaussian "
    "process trained on my likes and dislikes scores every event by **Expected Improvement** — how likely "
    "it is to beat the best thing I've liked so far, rewarding events the model is still unsure about."
)
show_flash()
search_status()
st.caption(model_status_text(model_meta))

if is_owner:
    owner_controls()

st.header("Event map")
render_event_map(gp, model_meta)

st.header("Events")
if is_owner:
    st.caption("Pick 👍 Like, 👎 Dislike or Clear, then tick the checkbox at the left of a row to label it. Click column headers to sort.")
events_section(gp, model_meta, is_owner)

idle_pause()
