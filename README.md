# ProjectTekberKelompok3 -- README
OpenCV model for predict how many people in frame including their emotions and age.

Proyek ini sekarang **hanya menghitung orang** pakai deteksi wajah OpenCV (Haar).
Dua fitur berikut **belum diimplementasikan** dan jadi tugas tim kamu:

* **Emosi** → folder: `src/emotions/`
* **Umur** → folder: `src/ages/`

Di bawah ini instruksi paling minimal supaya kamu bisa nambahin bagianmu tanpa utak-atik core.

## Cara jalanin (untuk cek deteksi wajah)

```bash
pip install -r requirements.txt
python src/main.py --video 0          # webcam
# atau:
# python src/main.py --video path/to/video.mp4
```

Opsi tuning cepat:

* Banyak false positive: `--neigh 7`
* Wajah kecil sering miss: `--scale 1.1 --min 16`
* FPS turun: `--scale 1.3` atau kecilkan resolusi `--width 960 --height 540`

## Kontrak input untuk modul kalian (sama untuk Emosi & Umur)

Saat integrasi, `main.py` akan memberikan **list wajah grayscale** berukuran **64×64**:

* Tipe: `numpy.ndarray` shape `(64, 64)` dtype `uint8` range `0..255`
* Contoh pembuatan dari deteksi:

  ```python
  face_crop = gray[y1:y2, x1:x2]              # gray dari frame
  gray_64 = cv2.resize(face_crop, (64, 64))   # ini yang akan dikirim ke modul
  ```

> Kamu bebas normalisasi sendiri di modul (misal bagi 255, atau z-score). Yang penting **terima list grayscale 64×64**.

## A. Tugas Tim Emosi (`src/emotions/`)

### Buat file: `src/emotions/inference.py`

Implement 2 fungsi berikut:

```python
# wajib ada
def load_model(model_path: str | None = None):
    """
    Muat model/weight yang diperlukan.
    Return: objek model (boleh apa saja) untuk dipakai di predict_emotions.
    """
    # TODO: load .pt atau inisialisasi model
    return None

# wajib ada
def predict_emotions(gray_faces_64, model=None):
    """
    Param:
      gray_faces_64: List[np.ndarray (64,64) uint8]
      model: objek dari load_model()
    Return:
      List[Tuple[str, float]] -> [(label, confidence), ...] panjangnya = jumlah wajah
    """
    # TODO: preproc -> infer -> softmax -> mapping label
    return [("neutral", 0.99) for _ in gray_faces_64]
```

### Output yang diharapkan

* Contoh untuk 3 wajah:

  ```python
  [("happy", 0.92), ("neutral", 0.75), ("sad", 0.66)]
  ```

## B. Tugas Tim Umur (`src/ages/`)

### Buat file: `src/ages/inference.py`

Implement 2 fungsi berikut:

```python
# wajib ada
def load_model(model_path: str | None = None):
    """
    Muat model/weight umur.
    """
    return None

# wajib ada
def predict_ages(gray_faces_64, model=None):
    """
    Param:
      gray_faces_64: List[np.ndarray (64,64) uint8]
      model: objek dari load_model()
    Return:
      List[float|int] -> usia per wajah, panjang = jumlah wajah
    """
    # TODO: preproc -> infer -> postproc (clamp 1..90 jika perlu)
    return [25.0 for _ in gray_faces_64]
```

### Output yang diharapkan

* Contoh untuk 3 wajah:

  ```python
  [22.4, 37.0, 18.9]
  ```

## Cara ngetes modul kalian sendiri (tanpa main loop)

Bikin skrip kecil sementara (opsional), misal `scratch_emosi.py`:

```python
import cv2, numpy as np
from emotions.inference import load_model, predict_emotions

img = cv2.imread("path/to/face.jpg", cv2.IMREAD_GRAYSCALE)
gray_64 = cv2.resize(img, (64,64))
model = load_model(None)
print(predict_emotions([gray_64], model=model))
```

Untuk umur, sama konsepnya pakai `ages.inference`.

## Integrasi nanti (biar tahu targetnya)

Setelah modul siap, `main.py` akan menambahkan kira-kira seperti ini:

```python
from emotions.inference import load_model as load_emo, predict_emotions
from ages.inference import load_model as load_age, predict_ages

emo_model = load_emo(None)
age_model = load_age(None)

# di loop, setelah dapat list gray_64:
emo_out = predict_emotions(gray_faces_64, model=emo_model)
age_out = predict_ages(gray_faces_64, model=age_model)

# lalu overlay di masing-masing bbox
label, conf = emo_out[i]
age = age_out[i]
```

## Ringkas

* Input ke modul: **list grayscale 64×64**.
* Emosi: kembalikan **(label, confidence)** per wajah.
* Umur: kembalikan **angka umur** per wajah.
* Sediakan `load_model()` dan `predict_*()` masing-masing.
* Tes modulmu sendiri dulu; integrasi gampang begitu interface-nya konsisten.
