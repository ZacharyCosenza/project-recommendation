"""Analytics tab: model status and retraining, the event map and its cards, calibration, and the events table."""
import json
import threading

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from recommender import config, db, model, review, viz
from recommender.web import get_conn, owner

router = APIRouter(prefix="/api", dependencies=[Depends(owner)])
_retrain = {"state": "idle", "message": ""}
_retrain_lock = threading.Lock()


class EventEdit(BaseModel):
    url: str
    label: int | None = None
    context: str | None = None


def _figure(fig):
    return json.loads(fig.to_json())


def _scores(clf, rows):
    if clf is None or not rows:
        return [None] * len(rows)
    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    return model.p_like(clf, X).tolist()


@router.get("/model")
def model_status(conn=Depends(get_conn)):
    _, meta = model.cached_current_model(conn)
    out = {"model": None, "retrain": _retrain}
    if meta:
        out["model"] = {k: meta[k] for k in ("id", "trained_at", "n_events", "label_counts")}
    return out


@router.post("/retrain", status_code=202)
def retrain():
    with _retrain_lock:
        if _retrain["state"] == "running":
            raise HTTPException(409, "Already retraining.")
        _retrain.update(state="running", message="")

    def job():
        conn = db.get_connection()
        try:
            meta = model.fit_and_persist(conn)
        except Exception as e:
            _retrain.update(state="error", message=str(e))
            return
        _retrain.update(state="done", message=f"Retrained on {meta['n_events']} events.")
        review.explain_unrated(conn)  # colouring belongs to one model, so redo it for the new one

    threading.Thread(target=job, name="retrain", daemon=True).start()
    return _retrain


@router.get("/map")
def event_map(conn=Depends(get_conn)):
    df = viz.to_frame(db.all_events(conn))
    if len(df) < config.event_map.min_points:
        return {"figure": None, "reason": f"The map appears once there are at least {config.event_map.min_points} "
                                          f"events (have {len(df)})."}
    clf, _ = model.cached_current_model(conn)
    X = np.stack(df["embedding"].to_numpy())
    scores = model.p_like(clf, X) if clf is not None else None
    recent = (df["run_id"] == db.latest_run_id(conn)).to_numpy()
    return {"figure": _figure(viz.event_map(df, viz.map_coords(conn, df, X), recent, scores)),
            "scored": scores is not None}


@router.get("/event")
def event_card(url: str, conn=Depends(get_conn)):
    event = db.get_event(conn, url)
    if event is None:
        raise HTTPException(404)
    clf, meta = model.cached_current_model(conn)
    highlight = review.highlight(conn, meta["id"], url) if meta else None
    if meta and highlight is None:
        review.request_explanation(url)
    return {"url": url, "theme": event["theme"], "city": event["city"], "label": event["label"],
            "context": event["context"], "description": event["description"],
            "p_like": _scores(clf, [event])[0], "highlight": highlight}


@router.get("/calibration")
def calibration(conn=Depends(get_conn)):
    _, meta = model.cached_current_model(conn)
    cal = meta and meta["calibration"]
    if not cal:
        return {"figure": None}
    p, liked = np.array(cal["p_like"]), np.array(cal["liked"])
    return {"figure": _figure(viz.calibration_plot(p, liked)), "n": len(p), "folds": config.model.calibration_folds,
            "brier": float(np.mean((p - liked) ** 2)), "baseline": float(liked.mean() * (1 - liked.mean()))}


@router.get("/events")
def events(view: str = "recent", city: str | None = None, label: int | None = None, page: int = 1,
           conn=Depends(get_conn)):
    if view == "recent":
        rows = db.get_latest_run_events(conn)
    else:
        rows = db.browse_events(conn, city=city, label=label, page=page, page_size=config.app.page_size)
    clf, _ = model.cached_current_model(conn)
    out = [{"url": r["url"], "theme": r["theme"], "city": r["city"], "context": r["context"],
            "description": r["description"], "label": r["label"], "p_like": s}
           for r, s in zip(rows, _scores(clf, rows))]
    out.sort(key=lambda e: -1 if e["p_like"] is None else e["p_like"], reverse=True)
    return {"events": out, "cities": db.list_cities(conn), "page_size": config.app.page_size}


@router.patch("/events", status_code=204)
def edit_event(body: EventEdit, conn=Depends(get_conn)):
    if db.get_event(conn, body.url) is None:
        raise HTTPException(404)
    if body.label is not None:
        if body.label not in (-1, 0, 1):
            raise HTTPException(422)
        db.update_label(conn, body.url, body.label)
    if body.context is not None:
        model.set_context(conn, body.url, body.context.strip())
    return Response(status_code=204)
