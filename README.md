# NLP Spam & Fake News Classification

A reproducible NLP text-classification project with two supported tasks:

1. **SMS spam classification** — ready to run with the UCI SMS Spam Collection.
2. **Fake-news classification** — accepts a user-supplied CSV or a common `Fake.csv` + `True.csv` pair.

The project uses TF-IDF features and classical machine-learning models, performs
exact-text deduplication before splitting, compares validation performance, evaluates
the selected model once on a held-out test set, saves the trained pipeline, and includes
a Streamlit interface for single-text and batch predictions.

## Why this project is useful

Text classification projects can suffer from leakage when the same or near-identical text
appears in multiple splits. This repository removes exact normalized-text duplicates before
the train/validation/test split and keeps model selection separate from final test evaluation.

## Supported models

- Logistic Regression
- Calibrated Linear SVM
- Complement Naive Bayes

All models use the same TF-IDF representation for a fair comparison.

## SMS Spam dataset

The spam task uses the **UCI SMS Spam Collection (dataset ID 228)**.

The dataset is fetched with `ucimlrepo`, so the raw dataset is not committed to GitHub.

## Fake-news dataset

For fake-news classification, provide either:

### Option A — one CSV

A CSV with a text column and a binary label column:

```text
text,label
"article text...",0
"article text...",1
```

Run:

```bash
python train.py --task fake_news --data data/my_fake_news.csv --text-column text --label-column label
```

### Option B — `Fake.csv` and `True.csv`

Prepare a combined dataset:

```bash
python prepare_fake_news.py --fake path/to/Fake.csv --true path/to/True.csv --output data/fake_news.csv
```

Then train:

```bash
python train.py --task fake_news --data data/fake_news.csv
```

The repository intentionally does **not** claim fake-news benchmark results without a
specific external dataset.

## Project structure

```text
nlp-spam-fake-news-classifier/
├── app.py
├── train.py
├── prepare_fake_news.py
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── io_utils.py
├── tests/
│   └── test_smoke.py
├── data/
│   ├── .gitkeep
│   └── demo_fake_news.csv
├── artifacts/
│   └── .gitkeep
└── .github/
    └── workflows/
        └── tests.yml
```

## Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

## Train spam classifier

```bash
python train.py --task spam
```

## Train fake-news classifier

```bash
python train.py --task fake_news --data data/fake_news.csv
```

## Run app

```bash
streamlit run app.py
```

The app automatically loads the model stored under:

```text
artifacts/<task>/
```

## Outputs

Each trained task writes:

```text
artifacts/<task>/
├── best_model.joblib
├── metadata.json
├── metrics.json
├── model_comparison.csv
├── confusion_matrix.png
└── top_terms.csv
```

## Demo fake-news CSV

`data/demo_fake_news.csv` contains a few synthetic examples only for smoke testing the
pipeline. It is **not** a benchmark dataset and must not be used for performance claims.

## CV-ready description

**NLP Spam & Fake News Classification** — Developed a leakage-aware TF-IDF text
classification pipeline with model comparison, duplicate-text control, held-out evaluation,
top-term inspection, and Streamlit inference for spam and fake-news tasks.

## Reproducibility

Default random seed: `42`. Metrics depend on the chosen dataset and split and are therefore
generated at training time rather than hard-coded in this README.
