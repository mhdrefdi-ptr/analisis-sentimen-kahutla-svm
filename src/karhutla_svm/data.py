from pathlib import Path

import pandas as pd

from .preprocessing import preprocess

LABELS = ["positif", "negatif", "netral"]
COLUMNS = ["id", "referensi_id", "jenis", "tanggal", "teks", "keyword",
           "label_sentimen", "link", "language"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = set(COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak tersedia: {sorted(missing)}")
    if data.empty:
        raise ValueError("Dataset kosong.")
    for column in ["id", "referensi_id", "label_sentimen"]:
        data[column] = data[column].str.strip()
    data["label_sentimen"] = data["label_sentimen"].str.lower()
    if (data["id"] == "").any() or data["id"].duplicated().any():
        raise ValueError("ID harus terisi dan unik.")
    if not data["label_sentimen"].isin(LABELS).all():
        raise ValueError("Label hanya boleh positif, negatif, atau netral.")
    if set(data["label_sentimen"]) != set(LABELS):
        raise ValueError("Dataset harus memiliki ketiga kelas sentimen.")
    if data["teks"].map(preprocess).eq("").any():
        raise ValueError("Teks kosong setelah preprocessing.")
    data["group"] = leakage_groups(data)
    return data


def leakage_groups(data: pd.DataFrame) -> list[str]:
    """Join reference chains and identical normalized texts transitively."""
    parents = {}

    def root(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = root(parents[key])
        return parents[key]

    def join(left, right):
        parents[root(left)] = root(right)

    seen = {}
    for row in data.itertuples():
        root(row.id)
        if row.referensi_id:
            join(row.id, row.referensi_id)
        normalized = preprocess(row.teks)
        if normalized in seen:
            join(row.id, seen[normalized])
        seen[normalized] = row.id
    return [root(key) for key in data["id"]]
