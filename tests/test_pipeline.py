import joblib
import pytest

from src.karhutla_svm.cli import DEFAULT_DATASET, main
from src.karhutla_svm.data import LABELS, leakage_groups, load_dataset
from src.karhutla_svm.model import build_pipeline, predict_text, split_dataset, train_model
from src.karhutla_svm.preprocessing import preprocess


def test_preprocessing():
    assert preprocess("Tidak BAIK! @akun https://x.com #Karhutla") == "tidak baik karhutla"
    assert preprocess("  ") == ""


def test_dataset_and_split():
    data = load_dataset(DEFAULT_DATASET)
    assert len(data) == 92
    assert data.label_sentimen.value_counts().to_dict() == {"negatif": 39, "positif": 38, "netral": 15}
    train, test = split_dataset(data)
    assert not set(train.group) & set(test.group)
    assert not set(train.teks.map(preprocess)) & set(test.teks.map(preprocess))
    assert set(train.label_sentimen) == set(test.label_sentimen) == set(LABELS)
    assert test.id.tolist() == split_dataset(data)[1].id.tolist()
    pipeline = build_pipeline().fit(train.teks, train.label_sentimen)
    vocabulary = pipeline.named_steps["tfidf"].vocabulary_
    test_only = set(" ".join(test.teks.map(preprocess)).split()) - set(" ".join(train.teks.map(preprocess)).split())
    assert not test_only.intersection(vocabulary)


def test_reference_chains_and_duplicates():
    data = load_dataset(DEFAULT_DATASET).iloc[:4].copy()
    data["referensi_id"] = ["", data.iloc[0].id, data.iloc[1].id, ""]
    data.iloc[3, data.columns.get_loc("teks")] = data.iloc[2].teks
    assert len(set(leakage_groups(data))) == 1


@pytest.mark.parametrize("change", ["label", "empty_text", "duplicate_id", "missing_column"])
def test_validation(tmp_path, change):
    data = load_dataset(DEFAULT_DATASET)
    if change == "label":
        data.loc[0, "label_sentimen"] = "invalid"
    elif change == "empty_text":
        data.loc[0, "teks"] = "https://x.com"
    elif change == "duplicate_id":
        data.loc[0, "id"] = data.loc[1, "id"]
    else:
        data = data.drop(columns="referensi_id")
    path = tmp_path / "invalid.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_train_reload_and_cli(tmp_path, capsys):
    report = train_model(DEFAULT_DATASET, tmp_path)
    assert report["train_rows"] + report["test_rows"] == 92
    assert sum(map(sum, report["confusion_matrix"])) == report["test_rows"]
    assert 0 <= report["accuracy"] <= 1
    for name in ["model.joblib", "evaluation.json", "confusion_matrix.csv", "test_predictions.csv"]:
        assert (tmp_path / name).is_file()
    text = load_dataset(DEFAULT_DATASET).iloc[0].teks
    model_path = tmp_path / "model.joblib"
    assert predict_text(text, model_path) == joblib.load(model_path).predict([text])[0]
    assert main(["predict", "--text", text, "--model", str(model_path)]) == 0
    assert capsys.readouterr().out.strip().lower() in LABELS
    assert main(["predict", "--text", " "]) == 1
    assert main(["train", "--dataset", str(tmp_path / "missing.csv")]) == 1
    with pytest.raises(ValueError):
        split_dataset(load_dataset(DEFAULT_DATASET), 1)
