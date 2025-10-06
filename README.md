# ProjectTekberKelompok3

OpenCV model for predict how many people in frame including their emotions and age.

Proyek sederhana untuk **menghitung jumlah orang** berbasis deteksi wajah. Fitur **emosi** dan **umur** belum diimplementasikan, tetapi sudah disiapkan struktur foldernya agar tim yang menangani dapat langsung mengisi bagiannya.

---

## 1) Struktur Proyek

```
face-mini/
├─ requirements.txt
└─ src/
   ├─ main.py                # entrypoint: deteksi wajah + hitung orang
   ├─ emotions/              # placeholder (belum diisi)
   │  └─ __init__.py
   └─ ages/                  # placeholder (belum diisi)
      └─ __init__.py
```

---

## 2) Prasyarat

* Python 3.9 atau lebih baru
* Kamera (jika pakai webcam) atau file video (mp4/avi)
* Pip/venv direkomendasikan

---

## 3) Instalasi Cepat

```bash
# opsional: buat virtual env
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` berisi:

```
opencv-python
mediapipe
```

---

## 4) Cara Menjalankan

### Webcam

```bash
python src/main.py --video 0
```

### File Video

```bash
python src/main.py --video path/ke/video.mp4
```

### Opsi Berguna

* Atur resolusi dan FPS:

  ```bash
  python src/main.py --video 0 --width 1280 --height 720 --fps 30
  ```
* Perketat atau longgarkan kepercayaan deteksi (default 0.55):

  ```bash
  python src/main.py --conf 0.65
  ```

Tekan `ESC` untuk keluar.

---

## 5) Apa yang Sudah Jalan

* Deteksi wajah menggunakan MediaPipe
* Hitung jumlah wajah pada frame saat ini
* Tampilkan bounding box dan jumlah orang di overlay

> Catatan: Hitung orang saat ini = jumlah wajah terdeteksi di frame itu. Belum ada tracking antar frame, karena scope proyek ini memang **count-only** yang ringan.

---

## 6) Pembagian Tugas Tim

Tim dibagi dua bagian yang **belum** diimplementasikan. Folder sudah disiapkan agar integrasi minimal:

### A) Tim Emosi (`src/emotions/`)

* **Target input**: crop wajah (nanti ditarik dari main loop).
* **Format input yang disarankan**: grayscale 64×64 atau 112×112.
* **Output**: `label_emosi (str)`, `confidence (float)`.
* **Saran smoothing**: majority vote window (misal 7 frame) per ID jika kelak tracking ditambahkan.

**Yang perlu disiapkan di folder `src/emotions/`:**

1. `inference.py`

   * Fungsi `load_model(...)`
   * Fungsi `predict_emotions(list_of_gray_faces) -> List[(label, conf)]`
2. `README.md` (opsional) yang jelaskan model, cara pakai, dan dependensi ekstra (kalau ada).

**Contoh antarmuka yang akan dipanggil dari `main.py` nanti:**

```python
from emotions.inference import load_model, predict_emotions
emo_model = load_model("path/opsional.pt")
labels = predict_emotions(gray_faces_64)  # -> [("happy", 0.92), ...]
```

### B) Tim Umur (`src/ages/`)

* **Target input**: crop wajah grayscale 64×64.
* **Output**: `age (float atau int)` 1–90 (clamped).
* **Saran smoothing**: Exponential Weighted Average (EWA) per ID jika tracking ditambahkan nanti.

**Yang perlu disiapkan di folder `src/ages/`:**

1. `inference.py`

   * Fungsi `load_model(...)`
   * Fungsi `predict_ages(list_of_gray_faces) -> List[float]`
2. `README.md` (opsional) model & dependensi.

**Contoh antarmuka:**

```python
from ages.inference import load_model, predict_ages
age_model = load_model("path/opsional.pt")
ages = predict_ages(gray_faces_64)  # -> [22.3, 35.7, ...]
```

> Kedua tim tidak perlu mengubah `main.py` pada tahap awal. Cukup pastikan fungsi antarmuka di atas tersedia. Integrasi akan dilakukan dengan menambahkan 5–10 baris kode di `main.py` saat waktunya.

---

## 7) Alur Data (Saat Ini)

1. Baca frame dari kamera/file.
2. Konversi BGR → RGB (MediaPipe butuh RGB).
3. Deteksi wajah → daftar bounding box.
4. Gambar kotak + tulis “People: N”.
5. Tampilkan ke layar.

> Emosi dan umur akan diinsert di langkah 4 setelah tim masing-masing siap.

---

## 8) Kualitas & Performa

Agar counting stabil:

* **Confidence threshold**: default 0.55. Jika banyak false positive, naikkan ke 0.65. Jika sering miss, turunkan ke 0.45.
* **Resolusi**: 1280×720 cukup. 1920×1080 menurunkan FPS.
* **Exposure**: skrip mencoba menonaktifkan auto exposure/autofocus untuk stabilitas, tapi kemampuan ini tergantung driver kamera.

---

## 9) Troubleshooting

* **Window tidak muncul / crash saat buka kamera**

  * Pastikan `--video 0` sesuai index kamera di perangkat kalian.
  * Coba kurangi resolusi: `--width 640 --height 360`.

* **FPS rendah**

  * Kurangi resolusi.
  * Tutup aplikasi berat lain yang memakai kamera/CPU.

* **Tidak ada wajah terdeteksi**

  * Tambahkan pencahayaan.
  * Turunkan `--conf`.
  * Pastikan wajah menghadap kamera dan tidak terlalu kecil di frame.

---

## 10) Roadmap Integrasi Emosi & Umur (Ringkas)

1. Tim Emosi dan Tim Umur menyelesaikan modul `inference.py` masing-masing.
2. Tambahkan di `main.py`:

   * Crop wajah → ubah ke grayscale → resize ke ukuran yang diminta modul.
   * Panggil `predict_emotions` dan/atau `predict_ages`.
   * Tampilkan label emosi dan prediksi umur di atas masing-masing bbox.
3. (Opsional) Tambahkan tracker sederhana agar label tidak lompat-lompat dan bisa smoothing per ID.

---

## 11) Lisensi

Internal project. Silakan sesuaikan.

---

Kalau butuh contoh potongan kode integrasi nanti, tinggal lihat antarmuka fungsi yang sudah ditentukan di bagian Pembagian Tugas Tim.
