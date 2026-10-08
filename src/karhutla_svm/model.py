import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data import LABELS, load_dataset
from .preprocessing import preprocess


def split_dataset(data, test_size=0.2, seed=42):
    if not 0 < test_size < 1:
        raise ValueError("test_size harus di antara 0 dan 1.")
    splitter = GroupShuffleSplit(n_splits=500, test_size=test_size, random_state=seed)
    candidates = []
    for train, test in splitter.split(data, groups=data["group"]):
        if any(set(data.iloc[index]["label_sentimen"]) != set(LABELS)
               for index in (train, test)):
            continue
        distribution = data["label_sentimen"].value_counts(normalize=True).reindex(LABELS)
        test_distribution = data.iloc[test]["label_sentimen"].value_counts(normalize=True).reindex(LABELS)
        score = abs(len(test) / len(data) - test_size) + np.abs(distribution - test_distribution).sum()
        candidates.append((score, train, test))
    if not candidates:
        raise ValueError("Tidak dapat membagi grup dengan ketiga kelas di train dan test.")
    _, train, test = min(candidates, key=lambda candidate: candidate[0])
    return data.iloc[train].copy(), data.iloc[test].copy()


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess, lowercase=False,
                                 ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")),
        ("svm", LinearSVC(random_state=42, max_iter=10000)),
    ])


def train_model(dataset, output="artifacts", test_size=0.2, seed=42):
    data = load_dataset(dataset)
    train, test = split_dataset(data, test_size, seed)
    model = build_pipeline()
    model.fit(train["teks"], train["label_sentimen"])
    predicted = model.predict(test["teks"])
    matrix = confusion_matrix(test["label_sentimen"], predicted, labels=LABELS)
    report = {
        "accuracy": accuracy_score(test["label_sentimen"], predicted),
        "classification_report": classification_report(
            test["label_sentimen"], predicted, labels=LABELS, output_dict=True, zero_division=0),
        "labels": LABELS, "confusion_matrix": matrix.tolist(),
        "dataset": str(Path(dataset).resolve()), "seed": seed,
        "requested_test_size": test_size, "train_rows": len(train), "test_rows": len(test),
        "train_ids": train["id"].tolist(), "test_ids": test["id"].tolist(),
        "class_counts": data["label_sentimen"].value_counts().to_dict(),
        "sklearn_version": sklearn.__version__,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(output / "confusion_matrix.csv", index_label="actual")
    predictions = test[["id", "referensi_id", "teks", "label_sentimen"]].copy()
    predictions["prediksi"] = predicted
    predictions.to_csv(output / "test_predictions.csv", index=False)
    return report


def predict_text(text, model_path="artifacts/model.joblib"):
    if not preprocess(text):
        raise ValueError("Teks kosong setelah preprocessing.")
    # Load only trusted local artifacts: joblib can execute code during loading.
    model = joblib.load(model_path)
    return str(model.predict([text])[0])
