"""Search tab: start a search, follow its progress, and see past runs and spend."""
import json
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from recommender import config, daily, db, search
from recommender.web import get_conn, owner

router = APIRouter(prefix="/api", dependencies=[Depends(owner)])


class Query(BaseModel):
    text: str
    max_results: int = 5


class SearchRequest(BaseModel):
    location: str
    date_start: date
    date_end: date
    queries: list[Query]


def finished_message(run):
    if run["status"] == "failed":
        return "error", f"Search failed: {run['error']}"
    summary = (f"Search finished: {run['total_event_count'] or 0} events ({run['new_event_count'] or 0} new), "
               f"${run['cost_usd'] or 0:.2f}.")
    if run["error"]:
        return "warning", f"{summary} Some steps failed: {run['error']}"
    return "success", summary


@router.get("/search/defaults")
def defaults():
    today = daily.today()
    return {"queries": config.search.default_queries, "location": config.search.default_location,
            "date_start": today.isoformat(),
            "date_end": (today + timedelta(days=config.search.default_days_ahead)).isoformat(),
            "seconds_per_query": config.app.seconds_per_query, "cost_per_query": config.app.cost_per_query}


@router.post("/search", status_code=202)
def start(body: SearchRequest):
    queries = [{"text": q.text.strip(), "max_results": min(max(q.max_results, 1), 10)} for q in body.queries
               if q.text.strip()]
    if not queries or body.date_end < body.date_start:
        raise HTTPException(422, "Add at least one query and a valid date range.")
    try:
        run_id = search.start_search(body.location.strip() or "N/A", body.date_start.isoformat(),
                                     body.date_end.isoformat(), queries)
    except search.SearchAlreadyRunning as e:
        raise HTTPException(409, str(e))
    return {"run_id": run_id}


@router.get("/search/status")
def status(conn=Depends(get_conn)):
    run = db.active_run(conn)
    if run is None:
        return {"active": None}
    return {"active": {"id": run["id"], "location": run["location"], "done": run["queries_done"],
                       "total": len(json.loads(run["queries_json"])), "events": run["total_event_count"] or 0}}


@router.get("/runs/{run_id}")
def run_result(run_id: int, conn=Depends(get_conn)):
    run = db.get_run(conn, run_id)
    if run is None:
        raise HTTPException(404)
    kind, message = finished_message(run) if run["status"] != "running" else ("info", "Still running.")
    return {"status": run["status"], "kind": kind, "message": message}


@router.get("/runs")
def runs(conn=Depends(get_conn)):
    daily = db.last_daily_run(conn)
    return {"runs": [dict(r) for r in db.recent_runs(conn)], "spend": db.total_cost(conn),
            "daily": dict(daily) if daily else None}
