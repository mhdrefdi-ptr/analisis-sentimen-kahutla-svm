# Spesifikasi implementasi MVP

Dokumen ini menjelaskan implementasi saat ini, bukan pengganti PRD lengkap yang
belum tersedia dalam repository.

## Tujuan dan ruang lingkup

Klasifikasi sentimen teks media sosial terkait kebakaran hutan dan lahan
(Karhutla) menjadi:

- **Positif:** dukungan, apresiasi, atau penilaian positif terhadap penanganan.
- **Negatif:** keluhan, kritik, kekhawatiran, atau penilaian negatif.
- **Netral:** informasi faktual tanpa ekspresi sentimen yang jelas.

MVP meliputi baca CSV, validasi, preprocessing, pembagian train/test, TF-IDF,
training SVM, evaluasi, penyimpanan/pemuatan model, dan prediksi CLI.
Web, SQL, crawling X, prediksi batch, serta grafik otomatis pada CLI tidak
termasuk implementasi. Grafik laporan tersedia secara terpisah di `reports/`.

## Stack dan lingkungan

| Komponen | Spesifikasi |
| --- | --- |
| Python | >=3.10; lingkungan pengujian tercatat memakai 3.12.3 |
| Pandas | >=2.0, pembacaan dan pengolahan CSV |
| Scikit-learn | >=1.3, Pipeline, TF-IDF, split grup, LinearSVC, metrik |
| Joblib | >=1.3, serialisasi pipeline |
| Pytest | >=8, dependency opsional `test` |
| Interface | CLI melalui `python -m src.karhutla_svm.cli` |

Dependency ditetapkan dalam `pyproject.toml` dengan batas minimum, bukan versi
terkunci. Perbedaan versi dapat memengaruhi kompatibilitas artifact dan hasil.
Belum ada benchmark RAM/CPU atau pengujian native Windows yang tercatat.

## Dataset dan validasi

Dataset default: `data/dataset_karhutla_normalisasi_final.csv`.
Snapshot yang diverifikasi berisi 92 baris: positif 38, negatif 39, netral 15.
Loader tidak mewajibkan 92 baris; assertion tersebut ada pada tes dataset bawaan.

Sembilan kolom wajib:

```text
id,referensi_id,jenis,tanggal,teks,keyword,label_sentimen,link,language
```

- Encoding UTF-8; BOM didukung melalui `utf-8-sig`.
- Semua kolom dibaca sebagai string, termasuk ID X agar presisi tetap terjaga.
- `id` harus terisi dan unik setelah trim whitespace.
- `referensi_id` boleh kosong atau mengarah ke ID di luar CSV.
- `teks` menjadi input; hasil preprocessing tidak boleh kosong.
- `label_sentimen` di-trim dan diubah ke huruf kecil; hanya tiga label di atas
  yang diterima dan seluruh kelas harus ada dalam dataset.
- CSV tidak boleh kosong dan seluruh kolom wajib harus tersedia.
- Format tanggal, nilai jenis/language, dan keabsahan URL belum divalidasi.

## Pipeline dan pencegahan leakage

1. Validasi dataset.
2. Normalisasi teks untuk identifikasi duplikat: Unicode NFKC, huruf kecil,
   hapus URL/mention, ganti tanda baca dan underscore dengan spasi,
   rapikan whitespace. Kata hashtag, negasi, dan angka dipertahankan;
   emoji terhapus. Tidak menggunakan stemming atau stopword removal.
3. Gabungkan rantai `id`/`referensi_id` dan teks identik setelah normalisasi
   secara transitif ke dalam grup, termasuk referensi yang tidak ada dalam CSV.
4. Buat 500 kandidat `GroupShuffleSplit` dengan seed default 42. Pilih kandidat
   yang memuat tiga kelas pada kedua bagian dan paling mendekati proporsi test
   serta distribusi kelas. Pemilihan tidak memakai skor model.
5. Fit pipeline TF-IDF dan SVM hanya pada train; gunakan pipeline tersimpan
   untuk transform/predict test serta teks baru.

Duplikat dekat, parafrasa, serta posting berbeda tentang kejadian sama belum
dikelompokkan otomatis. Split berbasis grup bukan stratifikasi sempurna.

## Parameter model

| Komponen | Parameter eksplisit |
| --- | --- |
| `TfidfVectorizer` | `preprocessor=preprocess`, `lowercase=False`, `ngram_range=(1, 2)`, `token_pattern=r"(?u)\b\w+\b"` |
| `LinearSVC` | `random_state=42`, `max_iter=10000` |
| Split | `test_size=0.2`, `random_state=42`, 500 kandidat berbasis grup |

Parameter lain memakai default library. Seed CLI hanya mengubah split.
Tidak ada tuning hyperparameter atau keluaran probabilitas.

## Evaluasi dan artifact

Metrik: accuracy; precision, recall, F1-score, dan support per kelas;
macro/weighted average; confusion matrix. `zero_division=0` dipakai untuk
metrik yang tidak terdefinisi. Urutan matriks: positif, negatif, netral;
baris adalah aktual dan kolom adalah prediksi.

Empat artifact: `model.joblib`, `evaluation.json`, `confusion_matrix.csv`, dan
`test_predictions.csv`. JSON juga menyimpan path dataset, seed split, ukuran
test yang diminta, jumlah train/test, ID kedua bagian, jumlah kelas, dan versi
scikit-learn. Training pada folder output yang sama menimpa hasil sebelumnya.

Hasil holdout yang tercatat dalam `artifacts/evaluation.json`:

| Indikator | Nilai |
| --- | --- |
| Train/test | 73 / 19 baris |
| Accuracy | 68,42% (13/19 benar) |
| Macro F1 | 49,56% |
| Weighted F1 | 62,60% |
| Recall netral | 0% dari 3 teks netral |

Angka ini merupakan hasil satu holdout, bukan target acceptance atau jaminan
generalisasi. Hasil delapan kasus pytest lulus membuktikan fungsi yang diuji,
bukan kualitas model pada semua kelas.

## Struktur proyek

```text
data/                       Dataset asli
src/karhutla_svm/
  preprocessing.py          Normalisasi teks
  data.py                   Validasi CSV dan pengelompokan leakage
  model.py                  Split, training, evaluasi, simpan/muat, prediksi
  cli.py                    Subcommand train dan predict
tests/test_pipeline.py      Delapan kasus pengujian, termasuk parametrized tests
artifacts/                  Model dan hasil evaluasi
reports/                    Laporan Word/PDF, chart, dan bukti pytest
docs/                       Setup, penggunaan, dan spesifikasi
pyproject.toml              Konfigurasi paket dan dependency
```

Panduan operasional: [setup](01-setup.md) dan [penggunaan](02-penggunaan.md).
