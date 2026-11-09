import hashlib
import hmac
import threading
import time

from . import config

_lock = threading.Lock()
_failures = []
TOKEN_COOKIE = "owner"


def heartbeat():
    # Called from a fragment the browser re-runs every few seconds, so a fresh heartbeat means a tab
    # is genuinely open; open TCP connections aren't reliable since Fly's proxy keeps them around.
    config.HEARTBEAT_PATH.touch()


def browser_recently_open():
    try:
        return time.time() - config.HEARTBEAT_PATH.stat().st_mtime < config.server.heartbeat_fresh_seconds
    except FileNotFoundError:
        return False


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


def _token_key():
    # Derived from the password, so changing APP_PASSWORD signs out every device.
    return hmac.new(config.APP_PASSWORD.encode(), b"owner-cookie", hashlib.sha256).digest()


def _sign(expiry):
    return hmac.new(_token_key(), expiry.encode(), hashlib.sha256).hexdigest()


def make_token():
    expiry = str(int(time.time()) + config.auth.cookie_days * 86400)
    return f"{expiry}.{_sign(expiry)}"


def valid_token(token):
    if not config.APP_PASSWORD or not token or "." not in token:
        return False
    expiry, signature = token.split(".", 1)
    return expiry.isdigit() and int(expiry) > time.time() and hmac.compare_digest(signature, _sign(expiry))
