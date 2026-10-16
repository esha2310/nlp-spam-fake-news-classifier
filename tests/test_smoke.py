import unittest

import pandas as pd

from src.data import deduplicate_text, make_text_split
from src.evaluation import classification_metrics
from src.modeling import build_models


class NLPClassifierSmokeTest(unittest.TestCase):
    def test_logistic_pipeline(self):
        rows = []
        for i in range(80):
            rows.append(
                {
                    "text": f"meeting schedule project update number {i}",
                    "label": 0,
                }
            )
            rows.append(
                {
                    "text": f"win cash prize claim reward now number {i}",
                    "label": 1,
                }
            )

        df = pd.DataFrame(rows)
        clean = deduplicate_text(df, "text", "label")
        split = make_text_split(clean, random_state=42)

        model = build_models(
            random_state=42,
            fast=True,
        )["logistic_regression"]

        model.fit(split.X_train, split.y_train)
        pred = model.predict(split.X_test)

        metrics = classification_metrics(split.y_test, pred)
        self.assertGreaterEqual(metrics["macro_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
