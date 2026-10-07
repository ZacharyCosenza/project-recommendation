from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from recommender import db, embeddings, preference, public_snapshot, search

st.set_page_config(page_title="Event Recommender", layout="wide")


@st.cache_resource
def _conn():
    c = db.get_connection()
    db.init_db(c)
    return c


conn = _conn()


def rows_to_df(rows):
    data = []
    for r in rows:
        data.append({
            "url": r["url"], "city": r["city"], "theme": r["theme"],
            "description": r["description"], "label": r["label"],
            "embedding": np.frombuffer(r["embedding"], dtype=np.float32),
            "found_at": r["found_at"], "created_at": r["created_at"],
        })
    return pd.DataFrame(data)


def model_staleness_text(meta):
    if meta is None:
        return "Model: never trained — click Retrain Model after labeling some events."
    return f"Model: trained on {meta['n_events']} events ({meta['trained_at']})."


def run_search(location, date_start, date_end, query_rows):
    run_id = db.start_run(conn, location, date_start, date_end, query_rows)
    try:
        with st.spinner("Searching... this can take a minute or two."):
            results, usage = search.search_city(query_rows, location, date_start, date_end)
            descs = [v["description"] for v in results.values()]
            embs = embeddings.embed_descriptions(descs) if descs else np.empty((0, 0))
            new_count, total_count = db.insert_events(conn, run_id, location, results, embs)
        db.finish_run(conn, run_id, status="completed", usage=usage,
                       new_event_count=new_count, total_event_count=total_count)
        st.success(f"Found {total_count} events ({new_count} new). Cost: ${usage['cost_usd']:.4f}")
    except Exception as e:
        db.finish_run(conn, run_id, status="failed", error=str(e))
        st.error(f"Search failed: {e}")


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

    edited = st.data_editor(
        display_df,
        key=key,
        disabled=["ei", "theme", "description", "url"],
        column_config={
            "ei": st.column_config.NumberColumn("EI", format="%.3f"),
            "label": st.column_config.SelectboxColumn("Label", options=[-1, 0, 1], required=True),
            "url": st.column_config.LinkColumn("Link"),
        },
        hide_index=True,
        use_container_width=True,
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

status_cols = st.columns([2, 2, 1, 1])
status_cols[0].caption(f"Lifetime spend: ${db.total_cost(conn):.4f}")
status_cols[1].caption(model_staleness_text(model_meta))
if status_cols[2].button("Retrain Model"):
    try:
        meta = preference.fit_and_persist(conn)
        st.success(f"Retrained on {meta['n_events']} events.")
    except ValueError as e:
        st.warning(str(e))
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
        use_container_width=True,
    )

    location = st.text_input("Location", value="Chicago")
    date_range = st.date_input("Date range", value=(date.today(), date.today() + timedelta(days=7)))
    date_start, date_end = (date_range if isinstance(date_range, tuple) and len(date_range) == 2
                             else (date_range, date_range))

    n_rows = len([r for r in edited_queries.to_dict("records") if str(r.get("text", "")).strip()])
    est_cost = n_rows * 0.065
    st.warning(f"This calls paid APIs: roughly ${est_cost:.2f} and 1-2 minutes for {n_rows} quer{'y' if n_rows == 1 else 'ies'}. Not reversible.")
    confirmed = st.checkbox("I understand this will call paid APIs")
    if st.button("Run Search", disabled=not confirmed or n_rows == 0):
        query_rows = [r for r in edited_queries.to_dict("records") if str(r.get("text", "")).strip()]
        run_search(location, date_start.isoformat(), date_end.isoformat(), query_rows)
        st.rerun()

st.header("Events")
view = st.radio("Showing", ["Most recent run", "All events"], horizontal=True)

if view == "Most recent run":
    rows = db.get_latest_run_events(conn)
    render_event_table(gp, model_meta, rows_to_df(rows), key="recent_editor")
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
    render_event_table(gp, model_meta, rows_to_df(rows), key="browse_editor")
