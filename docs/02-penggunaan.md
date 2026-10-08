# Cara menjalankan

Jalankan semua perintah dari root proyek, bukan dari `docs/` atau `src/`.
Selesaikan [setup](01-setup.md) dan aktifkan virtual environment terlebih dahulu.
Perintah `python` berikut berlaku di Windows dan Ubuntu setelah aktivasi.

## 1. Training dan evaluasi

```bash
python -m src.karhutla_svm.cli train
```

Dataset default: `data/dataset_karhutla_normalisasi_final.csv`.
CLI menampilkan JSON berisi accuracy, classification report, confusion matrix,
dan metadata pembagian data. Hasil disimpan ke `artifacts/`:

| File | Isi |
| --- | --- |
| `model.joblib` | Pipeline preprocessing, TF-IDF, dan LinearSVC |
| `evaluation.json` | Metrik, confusion matrix, seed, ID train/test, dan metadata |
| `confusion_matrix.csv` | Baris aktual, kolom prediksi; urutan positif, negatif, netral |
| `test_predictions.csv` | ID, referensi ID, teks, label aktual, dan prediksi holdout |

Training dengan opsi eksplisit:

```bash
python -m src.karhutla_svm.cli train --dataset data/dataset_karhutla_normalisasi_final.csv --output artifacts/percobaan-01 --test-size 0.2 --seed 42
```

- `--dataset`: path CSV.
- `--output`: folder hasil, default `artifacts`.
- `--test-size`: proporsi test yang diminta, default `0.2`; harus lebih dari 0 dan kurang dari 1.
- `--seed`: seed pembagian data, default `42`.

Ukuran test aktual dapat berbeda karena grup tidak dipecah. Kedua bagian wajib
memuat tiga kelas. Seed CLI mengatur split; seed LinearSVC tetap 42.

**Perhatian:** Training ulang menimpa empat file hasil pada folder output yang
sama. Gunakan folder output berbeda untuk mempertahankan hasil sebelumnya.

## 2. Prediksi teks

Training harus berhasil sebelum menjalankan prediksi dengan model default.

```bash
python -m src.karhutla_svm.cli predict --text "Asap kebakaran hutan semakin parah"
```

Output berupa `Positif`, `Negatif`, atau `Netral`, bukan confidence/probabilitas.
Pada model hasil evaluasi yang tercatat, contoh tersebut menghasilkan `Negatif`.

Untuk model dari folder lain:

```bash
python -m src.karhutla_svm.cli predict --model artifacts/percobaan-01/model.joblib --text "Terima kasih petugas pemadam"
```

**Keamanan:** Hanya muat file joblib dari sumber tepercaya. Proses pemuatan
joblib dapat menjalankan kode dari file tersebut.

## 3. Automated tests

```bash
python -m pytest
python -m pytest -v
```

Untuk menyimpan bukti pengujian, buat folder laporan terlebih dahulu.

Ubuntu:

```bash
mkdir -p reports
python -m pytest -v --junitxml=reports/pytest-results.xml > reports/pytest-output.txt
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force reports | Out-Null
python -m pytest -v --junitxml=reports/pytest-results.xml > reports/pytest-output.txt
```

## 4. Bantuan dan masalah umum

```bash
python -m src.karhutla_svm.cli --help
python -m src.karhutla_svm.cli train --help
python -m src.karhutla_svm.cli predict --help
```

| Masalah | Tindakan |
| --- | --- |
| `python` tidak ditemukan di Ubuntu | Aktifkan `.venv`; sebelum aktivasi gunakan `python3` |
| Dependency tidak ditemukan | Aktifkan `.venv`, lalu ulangi `python -m pip install -e ".[test]"` |
| Dataset/model tidak ditemukan | Periksa folder kerja dan path `--dataset`/`--model`; training dahulu jika model belum ada |
| Kolom atau label CSV tidak valid | Sesuaikan CSV dengan [spesifikasi](03-spesifikasi.md) |
| Teks kosong setelah preprocessing | Masukkan teks bermakna, bukan hanya URL/mention/tanda baca |
| Split gagal | Periksa jumlah grup dan distribusi kelas; jangan memecah grup terkait untuk memaksa split |

CLI mengembalikan exit code 0 saat berhasil dan 1 untuk error data/file yang
ditangani. Argumen CLI yang tidak valid menghasilkan exit code 2 dari argparse.
