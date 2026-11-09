"""The app: login, the PWA shell, and the JSON API behind its Swipe, Search and Analytics tabs.

Every route except login, static files and the cron hook needs the owner cookie.
"""
import hmac
from pathlib import Path

import plotly
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from recommender import config, daily, db
from recommender.web import owner, signed_in
from routes import analytics, auth, search, swipe

PWA = config.ROOT / "pwa"
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
for module in (auth, swipe, search, analytics):
    app.include_router(module.router)
app.mount("/pwa", StaticFiles(directory=PWA), name="pwa")


@app.middleware("http")
async def revalidate_app_files(request: Request, call_next):
    # The app's own files change with every deploy; make browsers check (a cheap 304 when unchanged).
    response = await call_next(request)
    if request.url.path.startswith("/pwa/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.on_event("startup")
def startup():
    conn = db.get_connection()
    db.init_db(conn)
    db.fail_stale_runs(conn)  # this is process start, so anything still "running" was killed
    conn.close()


@app.get("/")
def shell(request: Request):
    if not signed_in(request):
        return RedirectResponse("/login?next=/", status_code=302)
    return FileResponse(PWA / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/sw.js")
def service_worker():
    # Served from the root so its scope covers the whole app.
    return FileResponse(PWA / "sw.js", media_type="text/javascript")


@app.get("/vendor/plotly.min.js")
def plotly_js():
    # The copy bundled with the Python package, so the browser draws exactly what viz.py builds.
    return FileResponse(PLOTLY_JS, media_type="text/javascript", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/heartbeat", dependencies=[Depends(owner)], status_code=204)
def heartbeat():
    return Response(status_code=204)


@app.get("/api/config", dependencies=[Depends(owner)])
def app_config():
    return {"heartbeat_seconds": config.swipe.heartbeat_seconds}


@app.post("/api/cron/daily", status_code=202)
def cron_daily(request: Request):
    # Called by the scheduler app (see docs/PRODUCTION.md) with a shared token instead of the owner cookie.
    sent = request.headers.get("authorization", "").removeprefix("Bearer ")
    if not config.CRON_TOKEN or not hmac.compare_digest(sent.encode(), config.CRON_TOKEN.encode()):
        raise HTTPException(401)
    return {"started": daily.start()}
