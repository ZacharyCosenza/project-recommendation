"""Container entrypoint: runs Streamlit and stops the machine once nothing is using it.

Fly's built-in auto-stop only counts open connections, so it kills a background search as soon as the
browser tab closes. Instead, auto-stop is off and this wrapper exits cleanly after a stretch (server settings
in config.yaml) with no open tab (no recent heartbeat from the page) and no search running; with restart
policy on-failure that stops the machine, and auto-start brings it back on the next request.
"""
import os
import signal
import subprocess
import sys
import time

from recommender import config, db, session

cfg = config.server


def search_running():
    return db.active_run(db.get_connection()) is not None


def main():
    cmd = ["streamlit", "run", "app.py", f"--server.port={cfg.port}", "--server.address=0.0.0.0"]
    if not os.environ.get("FLY_APP_NAME"):
        os.execvp(cmd[0], cmd)

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
        time.sleep(cfg.check_seconds)
        if child.poll() is not None:
            sys.exit(child.returncode or 1)
        if session.browser_recently_open() or search_running():
            idle_since = None
            continue
        idle_since = idle_since or time.monotonic()
        if time.monotonic() - idle_since >= cfg.idle_shutdown_seconds:
            print(f"serve.py: idle {cfg.idle_shutdown_seconds}s (no browser, no search); stopping machine", flush=True)
            child.terminate()
            child.wait(timeout=30)
            sys.exit(0)


if __name__ == "__main__":
    main()
