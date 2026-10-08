# Sentimen Karhutla SVM

MVP CLI klasifikasi positif, negatif, dan netral menggunakan Pandas,
TF-IDF dan LinearSVC. Tidak menggunakan web atau database.

## Dokumentasi

- [Setup Windows dan Ubuntu](docs/01-setup.md)
- [Cara menjalankan](docs/02-penggunaan.md)
- [Spesifikasi implementasi](docs/03-spesifikasi.md)

## Instalasi

Python 3.10 atau lebih baru. Jalankan dari root proyek:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
python -m src.karhutla_svm.cli train
python -m src.karhutla_svm.cli predict --text "Asap kebakaran hutan semakin parah"
```

Dataset default: `data/dataset_karhutla_normalisasi_final.csv`.
File asli dipertahankan tanpa perubahan. Untuk nama/path lain:

```bash
python -m src.karhutla_svm.cli train --dataset data/dataset_karhutla.csv --output artifacts --test-size 0.2 --seed 42
python -m src.karhutla_svm.cli predict --model artifacts/model.joblib --text "Terima kasih petugas pemadam"
```

## Pipeline dan evaluasi

- Validasi sembilan kolom, ID unik, tiga label, dan teks tidak kosong.
  ID dibaca sebagai string agar ID X tidak kehilangan presisi.
- Normalisasi Unicode, huruf kecil, hapus URL, mention, dan tanda baca.
  Pertahankan negasi dan kata hashtag; tidak memakai stemming/stopword removal.
- Gabungkan rantai `id`/`referensi_id` dan teks identik setelah normalisasi
  menjadi grup terhubung, termasuk referensi sumber yang tidak ada di CSV.
- Cari split deterministik dari 500 kandidat berbasis grup, mendekati proporsi
  kelas dan ukuran test. Kedua bagian wajib memiliki tiga kelas; gagal dengan
  pesan jelas jika tidak memungkinkan. Pemilihan hanya memakai label/ukuran,
  bukan skor model. Proporsi test aktual dapat berbeda karena ukuran grup.
- Fit TF-IDF unigram/bigram dan SVM hanya pada train. Pipeline yang sama
  digunakan saat prediksi; tidak ada fit pada test.

Hasil training di `artifacts/`:

- `model.joblib`: pipeline tersimpan.
- `evaluation.json`: accuracy, precision/recall/F1 per kelas dan macro/weighted,
  confusion matrix, seed, ID split, jumlah kelas, serta versi scikit-learn.
- `confusion_matrix.csv`: baris label aktual, kolom prediksi; urutan
  positif, negatif, netral. Matriks juga tampil di output CLI.
- `test_predictions.csv`: prediksi test dengan ID dan referensi sumber.

Training ulang menimpa file hasil pada direktori output yang sama.
Hanya muat model joblib dari sumber tepercaya: pemuatan dapat menjalankan kode.

## Batasan

Dataset kecil (92 baris, termasuk bahasa Indonesia dan Melayu); skor satu
holdout bukan bukti generalisasi produksi. Tidak ada target accuracy yang
mengada-ada, data sintetis, atau confidence palsu. Duplikat dekat/parafrasa
serta posting berbeda tentang kejadian sama belum dikelompokkan otomatis.

PRD Markdown lengkap tidak tersedia dalam repository; implementasi mengikuti
spesifikasi MVP pada pesan. Struktur: `src/karhutla_svm/` untuk preprocessing,
validasi data, model, dan CLI; `tests/` untuk pytest.
