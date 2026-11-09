"""The daily job: retrain on the latest labels, search the default queries, then queue colouring for every
unrated card (the review worker does it in the background).

Retraining comes first because cached SHAP colouring belongs to one model; colouring after the retrain means
the new cards are ready for the model that will score them.
"""
import threading
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from . import config, db, model, review, search, session


def today():
    return datetime.now(ZoneInfo(config.daily.timezone)).date()


def run(conn, day):
    notes = []
    session.heartbeat()
    try:
        meta = model.fit_and_persist(conn)
        notes.append(f"retrained on {meta['n_events']} events")
    except ValueError as e:
        notes.append(f"retrain skipped: {e}")
    rows = [dict(q, max_results=config.daily.max_results) for q in config.search.default_queries]
    end = date.fromisoformat(day) + timedelta(days=config.search.default_days_ahead)
    try:
        run_id = search.run_search(config.search.default_location, day, end.isoformat(), rows)
        notes.append(f"search run {run_id}")
    except search.SearchAlreadyRunning:
        notes.append("search skipped: one was already running")
    notes.append(f"queued colouring for {review.explain_unrated(conn)} more cards")
    return "; ".join(notes)


def start(day=None):
    """Start today's job in the background. Returns False if it has already run (or started) today."""
    day = day or today().isoformat()
    conn = db.get_connection()
    if not db.claim_daily_run(conn, day):
        return False

    def job():
        try:
            db.finish_daily_run(conn, day, "completed", run(conn, day))
        except Exception as e:
            db.finish_daily_run(conn, day, "failed", repr(e))

    threading.Thread(target=job, name=f"daily-{day}", daemon=True).start()
    return True
