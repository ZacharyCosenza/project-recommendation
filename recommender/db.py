import json
import sqlite3
from datetime import datetime, timezone

import numpy as np

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS run_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','completed','failed')),
    location        TEXT NOT NULL,
    date_start      TEXT NOT NULL,
    date_end        TEXT NOT NULL,
    queries_json    TEXT NOT NULL,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        REAL,
    new_event_count INTEGER,
    total_event_count INTEGER,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS events (
    url               TEXT PRIMARY KEY,
    run_id            INTEGER NOT NULL REFERENCES run_log(id) ON DELETE CASCADE,
    first_run_id      INTEGER NOT NULL REFERENCES run_log(id),
    city              TEXT NOT NULL,
    theme             TEXT NOT NULL,
    query_text        TEXT NOT NULL,
    description       TEXT NOT NULL,
    embedding         BLOB NOT NULL,
    embedding_dim     INTEGER NOT NULL,
    label             INTEGER NOT NULL DEFAULT 0 CHECK (label IN (-1, 0, 1)),
    label_updated_at  TEXT,
    found_at          TEXT NOT NULL,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_label  ON events(label);
CREATE INDEX IF NOT EXISTS idx_events_city   ON events(city);

CREATE TABLE IF NOT EXISTS models (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at        TEXT NOT NULL,
    n_events          INTEGER NOT NULL,
    label_counts_json TEXT NOT NULL,
    y_best            REAL NOT NULL,
    model_blob        BLOB NOT NULL,
    sklearn_version   TEXT
);
"""


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def _now():
    return datetime.now(timezone.utc).isoformat()


def start_run(conn, location, date_start, date_end, query_rows):
    started_at = _now()
    cur = conn.execute(
        "INSERT INTO run_log (started_at, status, location, date_start, date_end, queries_json) "
        "VALUES (?, 'running', ?, ?, ?, ?)",
        (started_at, location, date_start, date_end, json.dumps(query_rows)),
    )
    conn.commit()
    return cur.lastrowid


def finish_run(conn, run_id, status, usage=None, new_event_count=None, total_event_count=None, error=None):
    usage = usage or {}
    conn.execute(
        "UPDATE run_log SET finished_at=?, status=?, input_tokens=?, output_tokens=?, cost_usd=?, "
        "new_event_count=?, total_event_count=?, error=? WHERE id=?",
        (
            _now(), status,
            usage.get("input_tokens"), usage.get("output_tokens"), usage.get("cost_usd"),
            new_event_count, total_event_count, error, run_id,
        ),
    )
    conn.commit()


def insert_events(conn, run_id, location, results, embeddings):
    now = _now()
    urls = list(results.keys())
    rows = [
        (
            url, run_id, run_id, location,
            results[url]["theme"], results[url]["query_text"], results[url]["description"],
            embeddings[i].astype(np.float32).tobytes(), embeddings.shape[1], now, now,
        )
        for i, url in enumerate(urls)
    ]
    conn.executemany(
        """
        INSERT INTO events (url, run_id, first_run_id, city, theme, query_text, description,
                             embedding, embedding_dim, found_at, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(url) DO UPDATE SET
            run_id = excluded.run_id,
            city = excluded.city,
            theme = excluded.theme,
            query_text = excluded.query_text,
            description = excluded.description,
            embedding = excluded.embedding,
            embedding_dim = excluded.embedding_dim,
            found_at = excluded.found_at
        """,
        rows,
    )
    conn.commit()
    new_count = conn.execute(
        "SELECT COUNT(*) FROM events WHERE run_id=? AND first_run_id=?", (run_id, run_id)
    ).fetchone()[0]
    return new_count, len(urls)


def total_cost(conn):
    row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM run_log").fetchone()
    return row[0]


def list_cities(conn):
    return [r[0] for r in conn.execute("SELECT DISTINCT city FROM events ORDER BY city")]


def get_latest_run_events(conn):
    row = conn.execute(
        "SELECT id FROM run_log WHERE status='completed' ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return []
    return conn.execute(
        "SELECT url, city, theme, query_text, description, embedding, embedding_dim, label, "
        "label_updated_at, found_at, created_at FROM events WHERE run_id=?",
        (row["id"],),
    ).fetchall()


def browse_events(conn, city=None, label=None, page=1, page_size=50):
    clauses, params = [], []
    if city:
        clauses.append("city = ?")
        params.append(city)
    if label is not None:
        clauses.append("label = ?")
        params.append(label)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    offset = (page - 1) * page_size
    query = (
        f"SELECT url, city, theme, query_text, description, embedding, embedding_dim, label, "
        f"label_updated_at, found_at, created_at FROM events {where} "
        f"ORDER BY created_at DESC LIMIT ? OFFSET ?"
    )
    return conn.execute(query, (*params, page_size, offset)).fetchall()


def update_label(conn, url, label):
    conn.execute(
        "UPDATE events SET label=?, label_updated_at=? WHERE url=?",
        (label, _now(), url),
    )
    conn.commit()
