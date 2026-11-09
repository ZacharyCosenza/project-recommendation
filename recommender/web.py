"""FastAPI dependencies shared by every route module."""
from fastapi import HTTPException, Request

from . import db, session


def signed_in(request):
    return session.valid_token(request.cookies.get(session.TOKEN_COOKIE))


def owner(request: Request):
    if not signed_in(request):
        raise HTTPException(401)
    session.heartbeat()  # any API call means the app is open, so the machine shouldn't idle-stop


def get_conn():
    conn = db.get_connection()
    try:
        yield conn
    finally:
        conn.close()
