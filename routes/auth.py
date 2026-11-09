"""Sign in and out. The only routes that work without the owner cookie (besides static files and the cron hook)."""
import html
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from recommender import config, session

router = APIRouter()
LOGIN_PAGE = (config.ROOT / "pwa" / "login.html").read_text()
ERRORS = {"wrong": "Wrong password.", "locked": "Too many wrong attempts. Try again in a few minutes.",
          "disabled": "Login is disabled: no APP_PASSWORD is set."}


def _safe_next(path):
    # Only same-site paths, so the login can't be used to bounce to another site.
    return path if path.startswith("/") and not path.startswith("//") else "/"


def _login_page(next_path, error="", status=200):
    body = LOGIN_PAGE.replace("{next}", html.escape(_safe_next(next_path))).replace("{error}", html.escape(error))
    return HTMLResponse(body, status_code=status)


@router.get("/login")
def login_page(next: str = "/"):
    return _login_page(next)


@router.post("/login")
async def login(request: Request):
    form = parse_qs((await request.body()).decode())
    password, next_path = form.get("password", [""])[0], form.get("next", ["/"])[0]
    result = session.try_unlock(password)
    if result != "ok":
        return _login_page(next_path, ERRORS[result], status=429 if result == "locked" else 401)
    response = RedirectResponse(_safe_next(next_path), status_code=303)
    response.set_cookie(session.TOKEN_COOKIE, session.make_token(), max_age=config.auth.cookie_days * 86400,
                        httponly=True, secure=True, samesite="lax")
    return response


@router.get("/logout")
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(session.TOKEN_COOKIE)
    return response
