import json
import pickle
from datetime import datetime, timezone
import threading
from functools import lru_cache

import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import StratifiedKFold

from . import config, db

cfg = config.model
CLASSES = (-1, 0, 1)


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(cfg.embedding)


def embed_texts(texts):
    return _embedder().encode(texts, normalize_embeddings=True)


def event_text(description, context):
    return f"{description} Context: {context}" if context else description


def embed_events(descriptions, contexts):
    return embed_texts([event_text(d, c) for d, c in zip(descriptions, contexts)])


def set_context(conn, url, context):
    """Save an edited context and re-embed the event, so the next retrain and P(like) use it."""
    event = db.get_event(conn, url)
    db.update_context(conn, url, context, embed_events([event["description"]], [context])[0])


def _new_classifier():
    kernel = ConstantKernel(cfg.kernel_constant) * RBF(cfg.kernel_length_scale)
    return GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=cfg.optimizer_restarts,
                                     random_state=cfg.random_state)


def class_probabilities(clf, X):
    """Probability of each label in CLASSES (columns), zero for a class the model never saw."""
    proba = clf.predict_proba(X)
    out = np.zeros((len(X), len(CLASSES)))
    for j, c in enumerate(clf.classes_):
        out[:, CLASSES.index(int(c))] = proba[:, j]
    return out


def p_like(clf, X):
    return class_probabilities(clf, X)[:, CLASSES.index(1)]


def out_of_fold_p_like(X, y):
    """P(like) for each event from a model that never saw its label, for checking calibration."""
    folds = min(cfg.calibration_folds, np.bincount(y + 1).max())
    p = np.zeros(len(y))
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=cfg.random_state).split(X, y):
        p[test] = p_like(_new_classifier().fit(X[train], y[train]), X[test])
    return p


def fit_and_persist(conn):
    rows = conn.execute("SELECT embedding, label FROM events").fetchall()
    if len(rows) < cfg.min_events_to_train:
        raise ValueError(f"Need at least {cfg.min_events_to_train} events to train (have {len(rows)}).")

    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    y = np.array([r["label"] for r in rows], dtype=int)
    if len(set(y.tolist())) < 2:
        raise ValueError("Need at least 2 distinct labels to fit a meaningful model.")

    clf = _new_classifier()
    clf.fit(X, y)
    try:
        calibration = {"p_like": out_of_fold_p_like(X, y).round(4).tolist(), "liked": (y == 1).tolist()}
    except ValueError:
        calibration = None  # too few labels of some class to hold any out

    label_counts = {str(v): int((y == v).sum()) for v in CLASSES}
    trained_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO models (trained_at, n_events, label_counts_json, y_best, model_blob, sklearn_version, "
        "calibration_json) VALUES (?,?,?,?,?,?,?)",
        (trained_at, len(rows), json.dumps(label_counts), float(y.max()), pickle.dumps(clf), sklearn.__version__,
         json.dumps(calibration)),
    )
    conn.commit()
    return {"trained_at": trained_at, "n_events": len(rows), "label_counts": label_counts}


_cache = {"id": None, "model": (None, None)}
_cache_lock = threading.Lock()


def cached_current_model(conn):
    """load_current_model, unpickled again only after a retrain (for long-running processes)."""
    latest = conn.execute("SELECT MAX(id) FROM models").fetchone()[0]
    with _cache_lock:
        if latest != _cache["id"]:
            _cache["id"], _cache["model"] = latest, load_current_model(conn)
        return _cache["model"]


def load_current_model(conn):
    row = conn.execute(
        "SELECT id, trained_at, n_events, label_counts_json, model_blob, calibration_json FROM models "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None, None
    clf = pickle.loads(row["model_blob"])
    if not isinstance(clf, GaussianProcessClassifier):
        # Models from before the switch to classification can't score; retrain to get one.
        return None, None
    meta = {
        "id": row["id"],
        "trained_at": row["trained_at"],
        "n_events": row["n_events"],
        "label_counts": json.loads(row["label_counts_json"]),
        "calibration": json.loads(row["calibration_json"]) if row["calibration_json"] else None,
    }
    return clf, meta
