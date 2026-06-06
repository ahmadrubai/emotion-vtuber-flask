# Emotion VTuber Flask

Aplikasi Flask untuk VTuber avatar 2D berbasis prediksi emosi dari webcam atau gambar upload.

Project ini diset untuk model Teachable Machine/Keras yang dilatih dari gambar utuh. Backend tidak melakukan crop muka sebelum prediksi.

## Fitur

- Realtime emotion prediction dari webcam.
- Browser Source avatar untuk OBS.
- Debug page untuk melihat webcam, FPS, emosi, dan confidence.
- Model Test page untuk upload gambar dan cek probabilitas emosi.
- Upload model Keras langsung dari browser (`.h5` atau `.keras`).
- Upload `labels.txt` jika urutan/nama label model berubah.

## Struktur Penting

```text
app.py                    Flask app dan endpoint API
config.py                 Konfigurasi model, kamera, dan path
Keras/keras_model.h5      Model default
Keras/labels.txt          Label default model
static/assets/            Gambar avatar per emosi
templates/avatar.html     Halaman OBS Browser Source
templates/model_test.html Halaman test gambar dan upload model
src/emotion_model.py      Loader dan inferensi model
src/camera.py             Realtime webcam pipeline
```

## Install Setelah Extract RAR

1. Extract file `.rar` ke folder bebas.

   Contoh:

   ```text
   C:\Users\NamaUser\Documents\emotion-vtuber-flask
   ```

2. Buka PowerShell atau Command Prompt di folder hasil extract.

   Cara cepat di Windows:

   - Buka folder hasil extract.
   - Klik address bar File Explorer.
   - Ketik `powershell`.
   - Tekan Enter.

3. Buat virtual environment (`.venv`), install dependency, lalu jalankan app:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Catatan:

- `.venv` tidak perlu didownload.
- `.venv` dibuat otomatis oleh command `python -m venv .venv`.
- Kalau command itu gagal atau `python` tidak dikenali, install Python dulu dari https://www.python.org/downloads/windows/
- Saat install Python, centang opsi `Add python.exe to PATH`.

Kalau `.venv` sudah pernah dibuat sebelumnya, cukup aktifkan dan jalankan:

```powershell
.\.venv\Scripts\activate
python app.py
```

Kalau terminal belum berada di folder project, masuk dulu:

```powershell
cd path\ke\emotion-vtuber-flask
```

Buka:

- Home: http://127.0.0.1:5000/
- Avatar OBS: http://127.0.0.1:5000/avatar
- Debug: http://127.0.0.1:5000/debug
- Model Test + Upload Model: http://127.0.0.1:5000/model_test

## OBS Browser Source

Tambahkan Browser Source dengan URL:

```text
http://127.0.0.1:5000/avatar
```

Rekomendasi:

- Width: `1920`
- Height: `1080`
- Background: transparan

## Upload Model Keras

Masuk ke:

```text
http://127.0.0.1:5000/model_test
```

Lalu upload:

- `model`: file `.h5` atau `.keras`
- `labels`: file `.txt` jika label model berbeda

Kalau upload sukses, model langsung aktif tanpa restart Flask.

## Format labels.txt

Format Teachable Machine:

```text
0 Angry
1 Sad
2 Happy
3 Fear
4 Neutral
```

Nama label akan dibuat lowercase di backend, contoh `Happy` menjadi `happy`.

Pastikan nama label cocok dengan file avatar di `static/assets/`.

Contoh:

```text
static/assets/happy.png
static/assets/sad.png
static/assets/neutral.png
```

Kalau model memprediksi label yang tidak punya file avatar, avatar fallback ke `neutral.png`.

## Full Image, Bukan Crop Muka

Konfigurasi utama ada di `config.py`:

```python
MODEL_USE_TM_PREPROCESSING = True
MODEL_USE_FACE_CROP = False
```

Artinya:

- Input webcam/gambar dipakai utuh.
- Tidak ada face crop sebelum model predict.
- Preprocessing tetap mengikuti Teachable Machine: resize ke `224x224`, lalu `(pixel / 127.5) - 1`.

Ini cocok kalau dataset training awalnya bukan hasil crop muka.

## Endpoint API

```text
GET  /emotion_state
GET  /model_status
POST /predict_image
POST /upload_model
GET  /video_feed
```

Contoh status model:

```json
{
  "model_loaded": true,
  "use_face_crop": false,
  "use_tm_preprocessing": true,
  "output_count": 5
}
```

## Catatan Kamera

Index kamera ada di `config.py`:

```python
CAMERA_INDEX = 1
```

Kalau webcam tidak kebuka, coba ubah ke:

```python
CAMERA_INDEX = 0
```

## Troubleshooting

### Avatar blank atau tidak berubah

Cek apakah label model punya gambar di `static/assets/`.

Misal label dari model adalah `happpy`, backend akan mencari:

```text
static/assets/happpy.png
```

Kalau file tidak ada, avatar fallback ke `neutral.png`.

### Jumlah output model tidak sama dengan label

Upload `labels.txt` yang sesuai dengan model.

Contoh model output 5 kelas harus punya 5 baris label.

### Prediksi terasa aneh

Pastikan mode crop tetap mati:

```python
MODEL_USE_FACE_CROP = False
```

Kalau model dari Teachable Machine, biarkan:

```python
MODEL_USE_TM_PREPROCESSING = True
```
