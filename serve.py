"""Entrypoint: runs the app (api.py under uvicorn) and, on Fly, stops the machine once nothing is using it.

Fly's built-in auto-stop only counts open connections, so it would kill a background search or the daily job as
soon as the app closes. Instead, auto-stop is off and this wrapper exits cleanly after a stretch (server settings in
config.yaml) with no recent heartbeat (the open app pings every 30s; long jobs ping as they go) and no search
running; with restart policy on-failure that stops the machine, and auto-start brings it back on the next request.
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
    if not os.environ.get("FLY_APP_NAME"):
        cmd = ["uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(cfg.port)]
        os.execvp(cmd[0], cmd)

    child = subprocess.Popen(["uvicorn", "api:app", "--host", "0.0.0.0", "--port", str(cfg.port)])

    def stop(code):
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
        sys.exit(code)

    signal.signal(signal.SIGTERM, lambda *_: stop(0))
    signal.signal(signal.SIGINT, lambda *_: stop(0))  # Fly stops machines with SIGINT by default
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
            print(f"serve.py: idle {cfg.idle_shutdown_seconds}s (no app open, no search); stopping machine", flush=True)
            stop(0)


if __name__ == "__main__":
    main()
