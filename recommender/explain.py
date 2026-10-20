from functools import lru_cache

from . import config, db, model

cfg = config.explain


@lru_cache(maxsize=1)
def _masker():
    import shap
    return shap.maskers.Text(r"\W+", mask_token=cfg.mask_token, collapse_mask_token=True)


def _explain(clf, text):
    import shap

    def like_probability(texts):
        return model.p_like(clf, model.embed_texts(list(texts)))

    sv = shap.Explainer(like_probability, _masker())([text], max_evals=cfg.max_evals, silent=True)[0]
    tokens = list(sv.data)
    # The regex tokenizer drops trailing punctuation; keep it so the card shows the full text.
    if tokens:
        tokens[-1] += text[len("".join(tokens)):]
    return tokens, [float(v) for v in sv.values], float(sv.base_values)


def word_attributions(conn, clf, model_id, event):
    """Per-word SHAP values for P(like) over description + context, cached per event and model."""
    cached = db.get_explanation(conn, event["url"], model_id)
    if cached is not None:
        return cached
    tokens, values, base = _explain(clf, model.event_text(event["description"], event["context"]))
    db.save_explanation(conn, event["url"], model_id, tokens, values, base)
    return tokens, values, base
