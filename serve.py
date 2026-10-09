"""Container entrypoint: runs Streamlit and stops the machine once nothing is using it.

Fly's built-in auto-stop only counts open connections, so it kills a background search as soon as the
browser tab closes. Instead, auto-stop is off and this wrapper exits cleanly after IDLE_SECONDS with no
open tab (no recent heartbeat from the page) and no search running; with restart policy on-failure that
stops the machine, and auto-start brings it back on the next request.
"""
import os
import signal
import subprocess
import sys
import time

IDLE_SECONDS = int(os.environ.get("IDLE_SHUTDOWN_SECONDS", "600"))
CHECK_SECONDS = 30
HEARTBEAT_FRESH_SECONDS = 90
PORT = 8501


def browser_recently_open():
    from recommender.background import HEARTBEAT_PATH
    try:
        return time.time() - HEARTBEAT_PATH.stat().st_mtime < HEARTBEAT_FRESH_SECONDS
    except FileNotFoundError:
        return False


def search_running():
    from recommender import db
    return db.active_run(db.get_connection()) is not None


def main():
    cmd = ["streamlit", "run", "app.py", f"--server.port={PORT}", "--server.address=0.0.0.0"]
    if not os.environ.get("FLY_APP_NAME"):
        os.execvp(cmd[0], cmd)

    from recommender import db
    conn = db.get_connection()
    db.init_db(conn)
    db.fail_stale_runs(conn)  # this is process start, so anything still "running" was killed
    conn.close()

    child = subprocess.Popen(cmd)

    def on_sigterm(*_):
        child.terminate()
        try:
            child.wait(timeout=10)
        finally:
            sys.exit(0)

    signal.signal(signal.SIGTERM, on_sigterm)
    idle_since = None
    while True:
        time.sleep(CHECK_SECONDS)
        if child.poll() is not None:
            sys.exit(child.returncode or 1)
        if browser_recently_open() or search_running():
            idle_since = None
            continue
        idle_since = idle_since or time.monotonic()
        if time.monotonic() - idle_since >= IDLE_SECONDS:
            print(f"serve.py: idle {IDLE_SECONDS}s (no browser, no search); stopping machine", flush=True)
            child.terminate()
            child.wait(timeout=30)
            sys.exit(0)


if __name__ == "__main__":
    main()
