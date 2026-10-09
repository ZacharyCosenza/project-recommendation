import hmac
import threading
import time

from . import config

_lock = threading.Lock()
_failures = []


def heartbeat():
    # Called from a fragment the browser re-runs every few seconds, so a fresh heartbeat means a tab
    # is genuinely open; open TCP connections aren't reliable since Fly's proxy keeps them around.
    config.HEARTBEAT_PATH.touch()


def browser_recently_open():
    try:
        return time.time() - config.HEARTBEAT_PATH.stat().st_mtime < config.server.heartbeat_fresh_seconds
    except FileNotFoundError:
        return False


def owner_mode_available():
    return bool(config.APP_PASSWORD)


def locked_out():
    # Counted across all sessions: per-session counters reset on page reload.
    with _lock:
        cutoff = time.time() - config.auth.window_seconds
        _failures[:] = [t for t in _failures if t > cutoff]
        return len(_failures) >= config.auth.max_failures


def try_unlock(password):
    if not config.APP_PASSWORD:
        return "disabled"
    if locked_out():
        return "locked"
    if hmac.compare_digest(password.encode(), config.APP_PASSWORD.encode()):
        return "ok"
    with _lock:
        _failures.append(time.time())
    return "wrong"
