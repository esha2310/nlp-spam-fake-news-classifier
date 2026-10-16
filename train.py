from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd

from src.data import load_binary_csv, load_sms_spam, make_text_split
from src.evaluation import (
    classification_metrics,
    extract_logistic_top_terms,
    positive_scores,
    save_confusion_matrix,
)
from src.io_utils import ensure_dir, write_json
from src.modeling import build_models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["spam", "fake_news"], required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--artifacts-root", default="artifacts")
    return parser.parse_args()


def load_data(args):
    if args.task == "spam":
        if args.data:
            return load_binary_csv(
                args.data,
                text_col=args.text_column,
                label_col=args.label_column,
            )
        return load_sms_spam()

    if not args.data:
        raise ValueError(
            "--data is required for fake_news. "
            "Provide a CSV with text and label columns."
        )

    return load_binary_csv(
        args.data,
        text_col=args.text_column,
        label_col=args.label_column,
    )


def main():
    args = parse_args()
    df = load_data(args)
    split = make_text_split(df, random_state=args.seed)

    labels = (
        ["ham", "spam"]
        if args.task == "spam"
        else ["real", "fake"]
    )

    artifact_dir = ensure_dir(Path(args.artifacts_root) / args.task)
    models = build_models(random_state=args.seed, fast=args.fast)

    rows = []
    trained = {}

    for name, model in models.items():
        print(f"Training {name}...")
        started = time.perf_counter()
        model.fit(split.X_train, split.y_train)
        elapsed = time.perf_counter() - started

        pred = model.predict(split.X_val)

        try:
            scores = positive_scores(model, split.X_val)
        except TypeError:
            scores = None

        metrics = classification_metrics(split.y_val, pred, scores)

        rows.append(
            {
                "model": name,
                "validation_macro_f1": metrics["macro_f1"],
                "validation_f1": metrics["f1"],
                "validation_accuracy": metrics["accuracy"],
                "validation_roc_auc": metrics.get("roc_auc"),
                "train_seconds": elapsed,
            }
        )
        trained[name] = model

    comparison = pd.DataFrame(rows).sort_values(
        ["validation_macro_f1", "validation_f1"],
        ascending=False,
    )
    comparison.to_csv(artifact_dir / "model_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    best_name = str(comparison.iloc[0]["model"])
    best_model = trained[best_name]

    test_pred = best_model.predict(split.X_test)
    try:
        test_scores = positive_scores(best_model, split.X_test)
    except TypeError:
        test_scores = None

    test_metrics = classification_metrics(
        split.y_test,
        test_pred,
        test_scores,
    )

    joblib.dump(best_model, artifact_dir / "best_model.joblib")

    metadata = {
        "task": args.task,
        "best_model": best_name,
        "random_state": args.seed,
        "class_labels": {"0": labels[0], "1": labels[1]},
        "fast_mode": bool(args.fast),
    }
    write_json(metadata, artifact_dir / "metadata.json")
    write_json({"test": test_metrics}, artifact_dir / "metrics.json")

    save_confusion_matrix(
        split.y_test,
        test_pred,
        labels,
        artifact_dir / "confusion_matrix.png",
    )

    # Top terms from the logistic-regression baseline for interpretability.
    logistic = trained["logistic_regression"]
    terms = extract_logistic_top_terms(logistic)
    terms.to_csv(artifact_dir / "top_terms.csv", index=False)

    print(f"\nBest model: {best_name}")
    print("Held-out test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
