import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from recommender import config, db, explain, model, search, session, viz

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


def selected_url(chart_event):
    points = chart_event.selection.get("points", []) if chart_event else []
    urls = [pt["customdata"] for pt in points if pt.get("customdata")]
    return urls[0] if urls else None


def event_card(gp, model_meta, event, score, editable):
    st.markdown(f"**{event['theme']}** · {event['city']} · label {LABEL_ICON[event['label']]}")
    if editable:
        with st.form(f"context_{event['url']}", border=False):
            new_context = st.text_input("Context", value=event["context"],
                                        placeholder="e.g. Friday, comedy show", help=CONTEXT_HELP)
            if st.form_submit_button("Save context") and new_context.strip() != event["context"]:
                save_context(event["url"], event["description"], new_context.strip())
                st.rerun()
    elif event["context"]:
        st.caption(f"Context: {event['context']}")
    if gp is None:
        st.caption("Word highlighting appears once a model is trained.")
        st.write(model.event_text(event["description"], event["context"]))
    else:
        st.caption(f"P(like) {score:.0%}")
        with st.spinner("Explaining this event (the first time takes a few seconds)..."):
            tokens, values, base = explain.word_attributions(conn, gp, model_meta["id"], event)
        st.html(viz.highlight_html(tokens, values, config.explain.color_floor))
        st.caption(f"Blue words push P(like) up and red words push it down, measured against the same text with "
                   f"words removed (baseline {base:.0%}). Hover a word for its value. This shows what the model is "
                   "sensitive to, not why you like something.")
    if event["url"].startswith(("http://", "https://")):
        st.markdown(f"[Open event page]({event['url']})")
    if editable:
        cols = st.columns(len(CARD_LABELS))
        for col, (name, value) in zip(cols, CARD_LABELS.items()):
            if col.button(name, key=f"card_{name}", disabled=event["label"] == value):
                db.update_label(conn, event["url"], value)
                st.rerun()


@st.fragment
def map_section(gp, model_meta, editable):
    # A fragment, so clicking a dot reruns only the map and card.
    df = viz.to_frame(db.all_events(conn))
    if len(df) < config.event_map.min_points:
        st.caption(f"The map appears once there are at least {config.event_map.min_points} events (have {len(df)}).")
        return
    X = np.stack(df["embedding"].to_numpy())
    scores = model.p_like(gp, X) if gp is not None else None
    recent_mask = (df["run_id"] == db.latest_run_id(conn)).to_numpy()
    with st.spinner("Laying out event map (first time after new events takes ~20s)..."):
        coords = viz.map_coords(conn, df, X)

    map_col, card_col = st.columns([5, 4])
    with map_col:
        chart_event = st.plotly_chart(viz.event_map(df, coords, recent_mask, scores),
                                      width="content", key="event_map", on_select="rerun", selection_mode="points")
        st.caption("Events laid out by similarity of description + context (UMAP, cosine). "
                   + ("Background is P(like) interpolated from every event — a reading aid; the model only scores "
                      "the events themselves, and the darkest areas include places with no events nearby. "
                      if gp is not None else "The P(like) background appears once a model is trained. ")
                   + "White dots are the most recent search, grey are earlier.")
    with card_col:
        # Labeling from the card rebuilds the chart, which clears its selection; keep showing the same event.
        url = selected_url(chart_event) or st.session_state.get("card_url")
        st.session_state.card_url = url
        if url is None or url not in set(df["url"]):
            st.info("Click a dot to see why the model scores that event the way it does.")
            return
        i = int(np.flatnonzero(df["url"].to_numpy() == url)[0])
        event_card(gp, model_meta, df.iloc[i], scores[i] if scores is not None else None, editable)


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
CARD_LABELS = {"👍 Like": 1, "👎 Dislike": -1, "Clear": 0}
CONTEXT_HELP = ("Added to the description before embedding. Auto-filled with the day of the week and the type "
                "of event when the description says so; edit it to tell the model what matters to you.")


def save_context(url, description, context):
    db.update_context(conn, url, context, model.embed_events([description], [context])[0])


def apply_table_edits(key, events_df):
    if not st.session_state.get("owner"):
        return
    for i, change in st.session_state[key]["edited_rows"].items():
        row = events_df.iloc[int(i)]
        like = change.get("👍", row["label"] == 1)
        dislike = change.get("👎", row["label"] == -1)
        # Ticking one box clears the other: whichever box was just changed wins.
        if "👍" in change and like:
            label = 1
        elif "👎" in change and dislike:
            label = -1
        else:
            label = 1 if like else -1 if dislike else 0
        if label != row["label"]:
            db.update_label(conn, row["url"], label)
        context = change.get("Context", row["context"]).strip()
        if context != row["context"]:
            save_context(row["url"], row["description"], context)


def render_event_table(gp, model_meta, events_df, editable, key):
    if events_df.empty:
        st.info("No events here.")
        return

    if gp is not None:
        X = np.stack(events_df["embedding"].to_numpy())
        events_df = events_df.assign(p_like=model.p_like(gp, X)).sort_values("p_like", ascending=False)
    else:
        events_df = events_df.assign(p_like=np.nan)
    events_df = events_df.reset_index(drop=True)

    link = events_df["url"].where(events_df["url"].str.startswith(("http://", "https://")), None)
    column_config = {
        "P(like)": st.column_config.ProgressColumn("P(like)", format="percent", min_value=0, max_value=1,
                                                   width="small", help="The classifier's probability of 👍"),
        "Context": st.column_config.TextColumn("Context", width="medium", help=CONTEXT_HELP),
        "Event": st.column_config.TextColumn("Event", width="large"),
        "Link": st.column_config.LinkColumn("Link", display_text="open", width="small"),
    }

    if not editable:
        table = pd.DataFrame({"Label": events_df["label"].map(LABEL_ICON), "P(like)": events_df["p_like"],
                              "Theme": events_df["theme"], "City": events_df["city"],
                              "Context": events_df["context"], "Event": events_df["description"], "Link": link})
        column_config["Label"] = st.column_config.TextColumn("Label", width="small")
        st.dataframe(table, column_config=column_config, hide_index=True, height=cfg.table_height)
        return

    table = pd.DataFrame({"👍": events_df["label"] == 1, "👎": events_df["label"] == -1, "P(like)": events_df["p_like"],
                          "Theme": events_df["theme"], "City": events_df["city"], "Context": events_df["context"],
                          "Event": events_df["description"], "Link": link})
    column_config["👍"] = st.column_config.CheckboxColumn("👍", width="small")
    column_config["👎"] = st.column_config.CheckboxColumn("👎", width="small")
    # Edits are stored by row position, so the key changes whenever the row order can change (new events or a
    # retrain); within one ordering, replaying the stored edits is harmless because they match the database.
    order_key = f"{key}_{hash(tuple(events_df['url']))}_{model_meta['id'] if model_meta else 0}"
    st.data_editor(table, column_config=column_config, hide_index=True, height=cfg.table_height, key=order_key,
                   disabled=["P(like)", "Theme", "City", "Event", "Link"],
                   on_change=apply_table_edits, args=(order_key, events_df))


REVIEW_ORDERS = ["Newest search first", "Most uncertain first", "Highest P(like) first"]
REVIEW_BUTTONS = {"👎 Dislike": -1, "😐 Neutral": 0, "👍 Like": 1}


@st.fragment
def review_section(gp, model_meta):
    # A fragment, so each answer only redraws the card, not the map and table.
    reviewed, total = db.review_counts(conn)
    df = viz.to_frame(db.unreviewed_events(conn))
    st.progress(reviewed / total if total else 1.0, text=f"{reviewed} reviewed · {len(df)} left")
    if df.empty:
        st.success("All caught up: every event has a label.")
        return

    order = st.segmented_control("Order", REVIEW_ORDERS, default=REVIEW_ORDERS[0], key="review_order",
                                 label_visibility="collapsed") or REVIEW_ORDERS[0]
    if gp is not None:
        probs = model.class_probabilities(gp, np.stack(df["embedding"].to_numpy()))
        df = df.assign(p_like=probs[:, 2], entropy=-(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1))
        if order == REVIEW_ORDERS[1]:
            df = df.sort_values("entropy", ascending=False, kind="stable")
        elif order == REVIEW_ORDERS[2]:
            df = df.sort_values("p_like", ascending=False, kind="stable")
    # Skipped events go to the back of the queue for this session only.
    skipped = st.session_state.setdefault("review_skipped", [])
    df = df.assign(skip_rank=df["url"].map({u: i + 1 for i, u in enumerate(skipped)}).fillna(0))
    event = df.sort_values("skip_rank", kind="stable").iloc[0]

    with st.container(border=True):
        st.markdown(f"**{event['theme']}** · {event['city']}"
                    + (f" · P(like) {event['p_like']:.0%}" if gp is not None else ""))
        st.write(event["description"])
        new_context = st.text_input("Context", value=event["context"], key=f"review_context_{event['url']}",
                                    placeholder="e.g. Friday, comedy show, comedy, Chicago", help=CONTEXT_HELP)
        if event["url"].startswith(("http://", "https://")):
            st.markdown(f"[Open event page]({event['url']})")
        cols = st.columns(len(REVIEW_BUTTONS) + 1)
        for col, (name, value) in zip(cols, REVIEW_BUTTONS.items()):
            if col.button(name, key=f"review_{value}", width="stretch"):
                if new_context.strip() != event["context"]:
                    save_context(event["url"], event["description"], new_context.strip())
                db.update_label(conn, event["url"], value)
                st.rerun(scope="fragment")
        if cols[-1].button("Skip", key="review_skip", width="stretch"):
            skipped.append(event["url"])
            st.rerun(scope="fragment")


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

        location = st.text_input(
            "Location", value=config.search.default_location,
            help="Added to every query. Use N/A to search each query as written, e.g. \"concerts NYC\" + "
                 "\"concerts Boston\"; each event's city is read from its description either way.",
        ).strip() or "N/A"
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
map_section(gp, model_meta, is_owner)

st.header("Events")
if is_owner:
    review_tab, table_tab = st.tabs(["Review", "All events"])
    with review_tab:
        review_section(gp, model_meta)
    with table_tab:
        st.caption("Tick 👍 or 👎 to label an event (neither = neutral), and edit Context to add what matters to you. "
                   "Click column headers to sort.")
        events_section(gp, model_meta, True)
else:
    events_section(gp, model_meta, False)

idle_pause()
