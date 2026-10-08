# Setup proyek: Windows dan Ubuntu

## Persyaratan

- Python 3.10 atau lebih baru; Python 3.12 direkomendasikan karena digunakan dalam pengujian proyek.
- Internet untuk instalasi dependency.
- Folder proyek yang berisi `pyproject.toml`, `src/`, dan `data/`.
- Dataset: `data/dataset_karhutla_normalisasi_final.csv`.

Tidak memerlukan GPU, database, atau server web. Gunakan virtual environment
baru pada setiap OS; jangan menyalin `.venv` dari komputer lain.

## Windows (PowerShell)

1. Instal Python 3.12 dari https://www.python.org/downloads/windows/.
   Aktifkan opsi **Add python.exe to PATH** dan Python Launcher jika tersedia.
   Buka ulang PowerShell setelah instalasi.
2. Masuk ke folder proyek. Ganti contoh path berikut sesuai lokasi proyek:

   ```powershell
   cd "C:\Projects\analisis-sentimen-kahutla"
   py -3.12 --version
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -e ".[test]"
   python -m pytest
   ```

Jika `py` tidak tersedia tetapi `python --version` menunjukkan versi yang sesuai,
gunakan `python -m venv .venv`.

Jika aktivasi PowerShell diblokir execution policy, tidak perlu mengubah policy.
Gunakan interpreter virtual environment secara langsung:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest
```

Untuk panduan penggunaan tanpa aktivasi, ganti `python` pada setiap perintah
dengan `.\.venv\Scripts\python.exe`.

## Ubuntu (Bash)

Ubuntu 22.04/24.04 menyediakan Python yang memenuhi persyaratan proyek.

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
cd ~/Projects/analisis-sentimen-kahutla
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest
```

Ganti path `~/Projects/analisis-sentimen-kahutla` sesuai lokasi proyek.
Jika versi Python kurang dari 3.10, instal versi yang sesuai sebelum membuat
virtual environment. Jangan memakai `sudo pip` atau instalasi global.

## Pemeriksaan setup

```bash
python -c "import pandas, sklearn, joblib; print('Dependency tersedia')"
python -m src.karhutla_svm.cli --help
```

Perintah tersebut juga berlaku di PowerShell setelah virtual environment aktif.
Pengujian yang tercatat saat dokumentasi dibuat: **8 passed**. Waktu eksekusi
dapat berbeda antar komputer.

Setiap membuka terminal baru, masuk kembali ke root proyek dan aktifkan `.venv`.
Keluar dari virtual environment dengan `deactivate`.

Lanjutkan ke [cara menjalankan](02-penggunaan.md).
