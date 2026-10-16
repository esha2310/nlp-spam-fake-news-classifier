from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X))
        return 1.0 / (1.0 + np.exp(-raw))
    raise TypeError("Model must expose predict_proba or decision_function.")


def classification_metrics(y_true, pred, scores=None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }

    if scores is not None and len(set(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))

    return metrics


def save_confusion_matrix(y_true, pred, labels, out_path: Path) -> None:
    cm = confusion_matrix(y_true, pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def extract_logistic_top_terms(model, n: int = 30) -> pd.DataFrame:
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["model"]

    if not hasattr(clf, "coef_"):
        return pd.DataFrame(columns=["term", "weight", "direction"])

    terms = np.asarray(tfidf.get_feature_names_out())
    coef = np.asarray(clf.coef_[0])

    pos_idx = np.argsort(coef)[-n:][::-1]
    neg_idx = np.argsort(coef)[:n]

    pos = pd.DataFrame(
        {
            "term": terms[pos_idx],
            "weight": coef[pos_idx],
            "direction": "class_1",
        }
    )
    neg = pd.DataFrame(
        {
            "term": terms[neg_idx],
            "weight": coef[neg_idx],
            "direction": "class_0",
        }
    )
    return pd.concat([pos, neg], ignore_index=True)
