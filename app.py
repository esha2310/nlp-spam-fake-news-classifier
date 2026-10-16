from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="NLP Text Classifier",
    page_icon="🧠",
    layout="wide",
)

st.title("Spam & Fake News NLP Classifier")
st.caption("Inference interface for locally trained TF-IDF text classifiers.")

task = st.selectbox(
    "Task",
    options=["spam", "fake_news"],
    format_func=lambda x: "SMS Spam" if x == "spam" else "Fake News",
)

artifact_dir = Path("artifacts") / task
model_path = artifact_dir / "best_model.joblib"
meta_path = artifact_dir / "metadata.json"

if not model_path.exists() or not meta_path.exists():
    st.warning(
        f"No trained model found for '{task}'. "
        f"Train one first with `python train.py --task {task}`."
    )
    st.stop()

model = joblib.load(model_path)
metadata = json.loads(meta_path.read_text(encoding="utf-8"))
class_labels = metadata["class_labels"]

st.write(f"**Saved model:** {metadata['best_model']}")

tab_single, tab_batch = st.tabs(["Single text", "Batch CSV"])

with tab_single:
    text = st.text_area(
        "Enter text",
        height=180,
        placeholder="Paste a message or article text here...",
    )

    if st.button("Classify", type="primary"):
        if not text.strip():
            st.error("Please enter some text.")
        else:
            pred = int(model.predict([text])[0])
            label = class_labels[str(pred)]

            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = float(model.predict_proba([text])[0, pred])

            st.success(f"Prediction: **{label}**")
            if confidence is not None:
                st.write(f"Model confidence: **{confidence:.3f}**")

with tab_batch:
    uploaded = st.file_uploader(
        "Upload CSV containing a `text` column",
        type=["csv"],
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)

        if "text" not in df.columns:
            st.error("CSV must contain a `text` column.")
            st.stop()

        pred = model.predict(df["text"].astype(str))
        result = df.copy()
        result["prediction"] = [
            class_labels[str(int(p))] for p in pred
        ]

        if hasattr(model, "predict_proba"):
            result["class_1_probability"] = model.predict_proba(
                df["text"].astype(str)
            )[:, 1]

        st.dataframe(result.head(100), use_container_width=True)

        st.download_button(
            "Download predictions",
            result.to_csv(index=False).encode("utf-8"),
            file_name=f"{task}_predictions.csv",
            mime="text/csv",
        )
