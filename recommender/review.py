"""The swipe queue: unrated events with P(like) and SHAP word colouring computed ahead of time.

All colouring goes through one background worker, most urgent first: the card on screen, then the cards about to
be swiped, then everything else (after a search or retrain). One thread, so SHAP (hundreds of embeddings per
event) never competes with itself for the CPU.
"""
import itertools
import queue as queue_lib
import threading

import numpy as np

from . import config, db, explain, model, session, viz

ON_SCREEN, UPCOMING, BACKGROUND = 0, 1, 2
_pending = queue_lib.PriorityQueue()
_order = itertools.count()
_queued = {}  # url -> most urgent priority it's waiting at
_queued_lock = threading.Lock()


def highlight(conn, model_id, url):
    """The card's SHAP-coloured description, or None if it hasn't been computed for this model yet."""
    cached = db.get_explanation(conn, url, model_id) if model_id else None
    return viz.highlight_html(cached[0], cached[1], config.explain.color_floor) if cached else None


def queue(conn, exclude=(), limit=config.swipe.queue_size):
    """Next unrated events, newest search first, skipping urls the phone already holds or skipped."""
    rows = [r for r in db.unreviewed_events(conn) if r["url"] not in exclude][:limit]
    clf, meta = model.cached_current_model(conn)
    model_id = meta["id"] if meta else None
    scores = [None] * len(rows)
    if clf is not None and rows:
        X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        scores = model.p_like(clf, X).tolist()
    cards = [{"url": r["url"], "theme": r["theme"], "city": r["city"], "description": r["description"],
              "context": r["context"], "p_like": s, "highlight": highlight(conn, model_id, r["url"])}
             for r, s in zip(rows, scores)]
    if model_id:
        for card in cards:
            if card["highlight"] is None:
                request_explanation(card["url"], UPCOMING)
    return cards


def explain_unrated(conn):
    """Queue colouring for every unrated card missing it for the current model. Returns how many were queued."""
    _, meta = model.cached_current_model(conn)
    if meta is None:
        return 0
    missing = [e["url"] for e in db.unreviewed_events(conn) if db.get_explanation(conn, e["url"], meta["id"]) is None]
    for url in missing:
        request_explanation(url, BACKGROUND)
    return len(missing)


def request_explanation(url, priority=ON_SCREEN):
    with _queued_lock:
        if _queued.get(url, BACKGROUND + 1) <= priority:
            return
        _queued[url] = priority
    _pending.put((priority, next(_order), url))


def _worker():
    conn = db.get_connection()
    while True:
        _, _, url = _pending.get()
        with _queued_lock:
            _queued.pop(url, None)
        try:
            clf, meta = model.cached_current_model(conn)
            event = db.get_event(conn, url)
            # A url queued twice (at two priorities) is already done the second time.
            if clf is not None and event is not None:
                explain.word_attributions(conn, clf, meta["id"], event)
            session.heartbeat()  # keep the machine up while this runs unattended
        except Exception as e:
            print(f"review: explaining {url} failed: {e}", flush=True)


threading.Thread(target=_worker, name="shap-prefetch", daemon=True).start()
