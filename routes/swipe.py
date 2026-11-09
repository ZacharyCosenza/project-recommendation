"""Swipe tab: the queue of unrated cards, their SHAP colouring, and rating / undo."""
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel

from recommender import db, model, review
from recommender.web import get_conn, owner

router = APIRouter(prefix="/api", dependencies=[Depends(owner)])


class Label(BaseModel):
    url: str
    label: int


class EventRef(BaseModel):
    url: str


@router.get("/queue")
def swipe_queue(exclude: list[str] = Query(default=[]), conn=Depends(get_conn)):
    reviewed, total = db.review_counts(conn)
    return {"cards": review.queue(conn, set(exclude)), "reviewed": reviewed, "total": total}


@router.get("/explanation")
def explanation(url: str, conn=Depends(get_conn)):
    _, meta = model.cached_current_model(conn)
    found = review.highlight(conn, meta["id"], url) if meta else None
    if found is None:
        if meta:
            review.request_explanation(url)
        return Response(status_code=202)
    return {"highlight": found}


@router.post("/label", status_code=204)
def label(body: Label, conn=Depends(get_conn)):
    if body.label not in (-1, 0, 1) or db.get_event(conn, body.url) is None:
        raise HTTPException(422)
    db.update_label(conn, body.url, body.label)
    return Response(status_code=204)


@router.post("/undo", status_code=204)
def undo(body: EventRef, conn=Depends(get_conn)):
    db.clear_label(conn, body.url)
    return Response(status_code=204)
