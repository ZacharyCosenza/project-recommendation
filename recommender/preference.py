import json
import pickle
from datetime import datetime, timezone

import numpy as np
import sklearn
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

MIN_EVENTS_TO_TRAIN = 4


def _build_kernel():
    return ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.3)


def expected_improvement(gp, X, y_best, xi=0.01):
    mu, sigma = gp.predict(X, return_std=True)
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    imp = mu - y_best - xi
    safe_sigma = np.where(sigma > 1e-9, sigma, 1.0)
    z = imp / safe_sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    return np.where(sigma > 1e-9, ei, 0.0)


def fit_and_persist(conn):
    rows = conn.execute("SELECT embedding, embedding_dim, label FROM events").fetchall()
    if len(rows) < MIN_EVENTS_TO_TRAIN:
        raise ValueError(f"Need at least {MIN_EVENTS_TO_TRAIN} events to train (have {len(rows)}).")

    X = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    y = np.array([r["label"] for r in rows], dtype=float)
    if len(set(y.tolist())) < 2:
        raise ValueError("Need at least 2 distinct labels to fit a meaningful model.")

    gp = GaussianProcessRegressor(kernel=_build_kernel(), normalize_y=True, random_state=42, n_restarts_optimizer=3)
    gp.fit(X, y)

    y_best = float(y.max())
    label_counts = {str(int(v)): int((y == v).sum()) for v in (-1.0, 0.0, 1.0)}
    trained_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO models (trained_at, n_events, label_counts_json, y_best, model_blob, sklearn_version) "
        "VALUES (?,?,?,?,?,?)",
        (trained_at, len(rows), json.dumps(label_counts), y_best, pickle.dumps(gp), sklearn.__version__),
    )
    conn.commit()
    return {"trained_at": trained_at, "n_events": len(rows), "label_counts": label_counts, "y_best": y_best}


def load_current_model(conn):
    row = conn.execute(
        "SELECT trained_at, n_events, label_counts_json, y_best, model_blob FROM models ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None, None
    gp = pickle.loads(row["model_blob"])
    meta = {
        "trained_at": row["trained_at"],
        "n_events": row["n_events"],
        "label_counts": json.loads(row["label_counts_json"]),
        "y_best": row["y_best"],
    }
    return gp, meta
