from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class TextSplit:
    X_train: pd.Series
    X_val: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def normalize_for_dedup(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def deduplicate_text(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    clean = df[[text_col, label_col]].dropna().copy()
    clean[text_col] = clean[text_col].astype(str)
    clean["_dedup"] = clean[text_col].map(normalize_for_dedup)

    # If identical normalized text has contradictory labels, drop those ambiguous texts.
    conflict_counts = clean.groupby("_dedup")[label_col].nunique()
    conflicts = set(conflict_counts[conflict_counts > 1].index)
    if conflicts:
        clean = clean[~clean["_dedup"].isin(conflicts)]

    clean = clean.drop_duplicates("_dedup", keep="first")
    clean = clean.drop(columns="_dedup").reset_index(drop=True)

    if clean[label_col].nunique() != 2:
        raise ValueError("Dataset must contain exactly two classes after cleaning.")

    return clean


def load_sms_spam() -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "ucimlrepo is required for the SMS Spam dataset. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    dataset = fetch_ucirepo(id=228)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if X.shape[1] < 1 or y.shape[1] < 1:
        raise ValueError("Unexpected UCI SMS Spam Collection format.")

    text = X.iloc[:, 0].astype(str)
    label = y.iloc[:, 0].astype(str).str.lower().str.strip()

    label = label.map({"ham": 0, "spam": 1})
    if label.isna().any():
        unknown = sorted(set(y.iloc[:, 0].astype(str)) - {"ham", "spam"})
        raise ValueError(f"Unexpected SMS labels: {unknown}")

    return pd.DataFrame({"text": text, "label": label.astype(int)})


def load_binary_csv(
    path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[[text_col, label_col]].rename(
        columns={text_col: "text", label_col: "label"}
    )

    labels = out["label"]

    if labels.dtype == object:
        normalized = labels.astype(str).str.lower().str.strip()
        mapping = {
            "0": 0,
            "1": 1,
            "false": 0,
            "true": 1,
            "real": 0,
            "fake": 1,
            "ham": 0,
            "spam": 1,
        }
        mapped = normalized.map(mapping)
        if mapped.isna().any():
            unique = sorted(normalized.dropna().unique().tolist())
            raise ValueError(
                "Could not infer binary labels. Supported labels include "
                f"0/1, real/fake, ham/spam. Found: {unique}"
            )
        out["label"] = mapped.astype(int)
    else:
        out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)

    if set(out["label"].dropna().unique()) != {0, 1}:
        raise ValueError("Label column must contain exactly binary labels 0 and 1.")

    return out


def make_text_split(
    df: pd.DataFrame,
    random_state: int = 42,
) -> TextSplit:
    df = deduplicate_text(df, "text", "label")

    X_train, X_temp, y_train, y_temp = train_test_split(
        df["text"],
        df["label"],
        test_size=0.30,
        stratify=df["label"],
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    return TextSplit(
        X_train=X_train.copy(),
        X_val=X_val.copy(),
        X_test=X_test.copy(),
        y_train=y_train.copy(),
        y_val=y_val.copy(),
        y_test=y_test.copy(),
    )
