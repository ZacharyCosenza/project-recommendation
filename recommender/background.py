import os
import threading
from pathlib import Path

from . import db, pipeline

HEARTBEAT_PATH = Path(os.environ.get("HEARTBEAT_PATH", "/tmp/app-heartbeat"))

_start_lock = threading.Lock()


def heartbeat():
    # Called from a fragment the browser re-runs every few seconds, so a fresh heartbeat means a tab
    # is genuinely open; open TCP connections aren't reliable since Fly's proxy keeps them around.
    HEARTBEAT_PATH.touch()


class SearchAlreadyRunning(Exception):
    pass


def start_search(location, date_start, date_end, query_rows):
    # Runs outside the Streamlit script, so reruns, other pages and closed tabs don't stop it.
    rows = pipeline.clean_rows(query_rows)
    with _start_lock:
        conn = db.get_connection()
        if db.active_run(conn) is not None:
            raise SearchAlreadyRunning("A search is already running.")
        run_id = db.start_run(conn, location, date_start, date_end, rows)

    threading.Thread(
        target=pipeline.execute_run,
        args=(db.get_connection(), run_id, location, date_start, date_end, rows),
        name=f"search-{run_id}", daemon=True,
    ).start()
    return run_id
