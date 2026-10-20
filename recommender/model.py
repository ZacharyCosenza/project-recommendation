import json
import pickle
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from . import config

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

    label_counts = {str(v): int((y == v).sum()) for v in CLASSES}
    trained_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO models (trained_at, n_events, label_counts_json, y_best, model_blob, sklearn_version) "
        "VALUES (?,?,?,?,?,?)",
        (trained_at, len(rows), json.dumps(label_counts), float(y.max()), pickle.dumps(clf), sklearn.__version__),
    )
    conn.commit()
    return {"trained_at": trained_at, "n_events": len(rows), "label_counts": label_counts}


def load_current_model(conn):
    row = conn.execute(
        "SELECT id, trained_at, n_events, label_counts_json, model_blob FROM models ORDER BY id DESC LIMIT 1"
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
    }
    return clf, meta
