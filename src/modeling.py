from __future__ import annotations

from collections import OrderedDict

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def make_tfidf(max_features: int = 50000):
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=max_features,
        sublinear_tf=True,
    )


def build_models(random_state: int = 42, fast: bool = False):
    features = 12000 if fast else 50000

    models = OrderedDict()

    models["logistic_regression"] = Pipeline(
        [
            ("tfidf", make_tfidf(features)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    svm = LinearSVC(
        class_weight="balanced",
        random_state=random_state,
    )
    models["calibrated_linear_svm"] = Pipeline(
        [
            ("tfidf", make_tfidf(features)),
            (
                "model",
                CalibratedClassifierCV(
                    svm,
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )

    models["complement_nb"] = Pipeline(
        [
            ("tfidf", make_tfidf(features)),
            ("model", ComplementNB(alpha=0.5)),
        ]
    )

    return models
