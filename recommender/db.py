import json
import sqlite3
from datetime import datetime, timezone

import numpy as np

from . import config

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
    context           TEXT NOT NULL DEFAULT '',
    context_auto      TEXT NOT NULL DEFAULT '',
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

CREATE TABLE IF NOT EXISTS map_layout (
    url          TEXT PRIMARY KEY REFERENCES events(url) ON DELETE CASCADE,
    x            REAL NOT NULL,
    y            REAL NOT NULL,
    computed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS explanations (
    url          TEXT NOT NULL REFERENCES events(url) ON DELETE CASCADE,
    model_id     INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    tokens_json  TEXT NOT NULL,
    values_json  TEXT NOT NULL,
    base_value   REAL NOT NULL,
    PRIMARY KEY (url, model_id)
);
"""


def get_connection():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # timeout: the background search and the page can write at the same moment
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(conn):
    conn.executescript(SCHEMA)
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(run_log)")}
    if "queries_done" not in columns:
        conn.execute("ALTER TABLE run_log ADD COLUMN queries_done INTEGER NOT NULL DEFAULT 0")
    event_columns = {r["name"] for r in conn.execute("PRAGMA table_info(events)")}
    if "context" not in event_columns:
        conn.execute("ALTER TABLE events ADD COLUMN context TEXT NOT NULL DEFAULT ''")
    if "context_auto" not in event_columns:
        conn.execute("ALTER TABLE events ADD COLUMN context_auto TEXT NOT NULL DEFAULT ''")
    conn.commit()


def set_queries_done(conn, run_id, n):
    conn.execute("UPDATE run_log SET queries_done=? WHERE id=?", (n, run_id))
    conn.commit()


def active_run(conn):
    return conn.execute(
        "SELECT id, started_at, location, queries_json, queries_done, cost_usd, total_event_count "
        "FROM run_log WHERE status='running' ORDER BY id DESC LIMIT 1"
    ).fetchone()


def get_run(conn, run_id):
    return conn.execute("SELECT * FROM run_log WHERE id=?", (run_id,)).fetchone()


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


def run_counts(conn, run_id):
    total = conn.execute("SELECT COUNT(*) FROM events WHERE run_id=?", (run_id,)).fetchone()[0]
    new = conn.execute("SELECT COUNT(*) FROM events WHERE first_run_id=?", (run_id,)).fetchone()[0]
    return new, total


def update_run_progress(conn, run_id, usage):
    new, total = run_counts(conn, run_id)
    conn.execute(
        "UPDATE run_log SET input_tokens=?, output_tokens=?, cost_usd=?, new_event_count=?, total_event_count=? "
        "WHERE id=?",
        (usage["input_tokens"], usage["output_tokens"], usage["cost_usd"], new, total, run_id),
    )
    conn.commit()
    return new, total


def finish_run(conn, run_id, status, usage, error=None):
    update_run_progress(conn, run_id, usage)
    conn.execute(
        "UPDATE run_log SET finished_at=?, status=?, error=? WHERE id=?",
        (_now(), status, error, run_id),
    )
    conn.commit()


def fail_stale_runs(conn):
    # Single process: any run still 'running' when the app starts was killed mid-run.
    conn.execute(
        "UPDATE run_log SET status='failed', finished_at=?, error='interrupted: app stopped before the run finished' "
        "WHERE status='running'",
        (_now(),),
    )
    conn.commit()


def recent_runs(conn, limit=10):
    return conn.execute(
        "SELECT id, started_at, finished_at, status, location, date_start, date_end, cost_usd, "
        "new_event_count, total_event_count, error FROM run_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()


def get_contexts(conn, urls):
    """{url: (context, context_auto)} for events already in the database."""
    if not urls:
        return {}
    marks = ",".join("?" * len(urls))
    rows = conn.execute(f"SELECT url, context, context_auto FROM events WHERE url IN ({marks})", list(urls))
    return {r["url"]: (r["context"], r["context_auto"]) for r in rows}


def insert_events(conn, run_id, results, embeddings):
    """results: {url: {theme, city, query_text, description, context, context_auto}}."""
    now = _now()
    rows = [
        (
            url, run_id, run_id, r["city"] or "unknown", r["theme"], r["query_text"], r["description"],
            r["context"], r["context_auto"], embeddings[i].astype(np.float32).tobytes(), embeddings.shape[1], now, now,
        )
        for i, (url, r) in enumerate(results.items())
    ]
    conn.executemany(
        """
        INSERT INTO events (url, run_id, first_run_id, city, theme, query_text, description, context, context_auto,
                             embedding, embedding_dim, found_at, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(url) DO UPDATE SET
            run_id = excluded.run_id,
            city = excluded.city,
            theme = excluded.theme,
            query_text = excluded.query_text,
            description = excluded.description,
            context = excluded.context,
            context_auto = excluded.context_auto,
            embedding = excluded.embedding,
            embedding_dim = excluded.embedding_dim,
            found_at = excluded.found_at
        """,
        rows,
    )
    conn.commit()


def total_cost(conn):
    row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM run_log").fetchone()
    return row[0]


def list_cities(conn):
    return [r[0] for r in conn.execute("SELECT DISTINCT city FROM events ORDER BY city")]


EVENT_COLUMNS = (
    "url, run_id, city, theme, query_text, description, context, embedding, embedding_dim, label, "
    "label_updated_at, found_at, created_at"
)


def latest_run_id(conn):
    row = conn.execute(
        "SELECT id FROM run_log WHERE total_event_count > 0 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["id"] if row else None


def get_latest_run_events(conn):
    run_id = latest_run_id(conn)
    if run_id is None:
        return []
    return conn.execute(f"SELECT {EVENT_COLUMNS} FROM events WHERE run_id=?", (run_id,)).fetchall()


def all_events(conn):
    return conn.execute(f"SELECT {EVENT_COLUMNS} FROM events ORDER BY created_at").fetchall()


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
    query = f"SELECT {EVENT_COLUMNS} FROM events {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
    return conn.execute(query, (*params, page_size, offset)).fetchall()


def update_label(conn, url, label):
    conn.execute(
        "UPDATE events SET label=?, label_updated_at=? WHERE url=?",
        (label, _now(), url),
    )
    conn.commit()


def get_map_layout(conn):
    return {r["url"]: (r["x"], r["y"]) for r in conn.execute("SELECT url, x, y FROM map_layout")}


def save_map_layout(conn, urls, coords):
    now = _now()
    conn.execute("DELETE FROM map_layout")
    conn.executemany(
        "INSERT INTO map_layout (url, x, y, computed_at) VALUES (?,?,?,?)",
        [(u, float(x), float(y), now) for u, (x, y) in zip(urls, coords)],
    )
    conn.commit()


def get_explanation(conn, url, model_id):
    row = conn.execute(
        "SELECT tokens_json, values_json, base_value FROM explanations WHERE url=? AND model_id=?", (url, model_id)
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["tokens_json"]), json.loads(row["values_json"]), row["base_value"]


def save_explanation(conn, url, model_id, tokens, values, base_value):
    conn.execute(
        "INSERT OR REPLACE INTO explanations (url, model_id, tokens_json, values_json, base_value) VALUES (?,?,?,?,?)",
        (url, model_id, json.dumps(tokens), json.dumps(values), base_value),
    )
    conn.commit()


def update_context(conn, url, context, embedding):
    conn.execute("UPDATE events SET context=?, embedding=?, embedding_dim=? WHERE url=?",
                 (context, embedding.astype(np.float32).tobytes(), len(embedding), url))
    conn.execute("DELETE FROM explanations WHERE url=?", (url,))
    conn.commit()


def unreviewed_events(conn):
    """Events still on the default neutral label: never given an explicit -1, 0 or 1."""
    return conn.execute(
        f"SELECT {EVENT_COLUMNS} FROM events WHERE label = 0 AND label_updated_at IS NULL "
        "ORDER BY run_id DESC, created_at DESC"
    ).fetchall()


def review_counts(conn):
    reviewed, total = conn.execute(
        "SELECT SUM(label != 0 OR label_updated_at IS NOT NULL), COUNT(*) FROM events").fetchone()
    return reviewed or 0, total
