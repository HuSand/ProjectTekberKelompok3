# src/main.py
# jalankan dari root project:  py -m src.main

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
import math
import numpy as np
from collections import deque
from pathlib import Path

from src.umur_model.umur_model import load_model as load_age_model, predict_ages
from src.emosi_model.emosi_model import EmotionRecognizer

# ================== CONFIG ==================
VIDEO_SRC = 2
FRAME_W, FRAME_H, FPS = 1280, 720, 30

FACES_DIR       = "data/faces"
FACE_SAVE_SIZE  = 64

EMO_H5_PATH = "src/emosi_model/fer2013_cnn.h5"
AGE_H5_PATH = "src/umur_model/age_model.h5"

# Deteksi & kotak
FACE_MARGIN     = 0.28
BOX_SHRINK      = 0.90
MIN_FACE_PX     = 110
HAAR_SCALE      = 1.05
HAAR_NEIGH      = 7
HAAR_MIN_SIZE   = 36

# Emosi
EMO_UNKNOWN_TH  = 0.35
EMO_EMA_ALPHA   = 0.5

# Umur (stabilisasi)
AGE_EMA_ALPHA   = 0.12
AGE_ROUND_STEP  = 1.0
AGE_MEDIAN_LEN  = 15
AREA_DRIFT_GATE = 0.30  # skip update kalau bbox beda area >30%

# Umur: quality gates
BLUR_MIN_VAR        = 80.0
BRIGHT_MIN          = 40.0
BRIGHT_MAX          = 210.0
CONTRAST_MIN_STD    = 20.0
AGE_JUMP_MAX_ABS    = 8.0
AGE_JUMP_MAX_RATIO  = 0.30

# Range tampilan (biar gak ngawur)
DISPLAY_MIN_AGE = 12.0
DISPLAY_MAX_AGE = 75.0
ADULT_MODE_MIN  = 16.0   # kalau model sering nembak bocil, dorong minimal ke sini
# ============================================

# ==== AGE SCALING (SATU-SATUNYA) ====
AGE_TARGET = 22.0        # target mean umur default untuk kalibrasi awal
AGE_SCALE_GLOBAL = None  # auto dari batch, lalu dipakai seed untuk tiap orang

_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


class AgeCalibrator:
    """Linear a*x + b, per-person. Pakai hotkeys 1/2/3/4/5 buat isi titik kalibrasi cepat."""
    def __init__(self, l2=1e-3):
        self.X = []  # raw_pred from model (sebelum scale per-ID)
        self.Y = []  # true_age yang kamu input
        self.a = 1.0
        self.b = 0.0
        self.l2 = l2

    def add(self, raw_pred, true_age):
        if raw_pred is None: return
        self.X.append(float(raw_pred))
        self.Y.append(float(true_age))
        self._fit()

    def _fit(self):
        if len(self.X) < 2:
            # fallback: skala kasar agar sekitar 22
            m = np.mean(self.X) if self.X else 1.0
            self.a = (22.0 / max(m,1e-6))
            self.b = 0.0
            return
        X = np.array(self.X, dtype=float)
        Y = np.array(self.Y, dtype=float)
        Sx2 = float(np.dot(X, X)) + self.l2
        Sx  = float(np.sum(X))
        n   = float(len(X))
        Sxy = float(np.dot(X, Y))
        Sy  = float(np.sum(Y))
        det = Sx2*n - Sx*Sx
        if abs(det) < 1e-8:
            self.a, self.b = 1.0, 0.0
        else:
            self.a = ( n*Sxy - Sx*Sy) / det
            self.b = (Sx2*Sy - Sx*Sxy) / det

    def apply(self, raw_pred):
        if raw_pred is None: return None
        return float(self.a*float(raw_pred) + self.b)

def _square_with_margin(x1,y1,x2,y2,W,H,margin=FACE_MARGIN):
    w = x2 - x1; h = y2 - y1
    s = max(w, h)
    cx = x1 + w//2; cy = y1 + h//2
    s = int(s * (1.0 + margin))
    x1n = max(0, cx - s//2)
    y1n = max(0, cy - s//2)
    x2n = min(W, x1n + s)
    y2n = min(H, y1n + s)
    x1n = max(0, x2n - s)
    y1n = max(0, y2n - s)
    return x1n, y1n, x2n, y2n

def _shrink_box(x1,y1,x2,y2, factor=BOX_SHRINK):
    cx = (x1+x2)/2.0
    cy = (y1+y2)/2.0
    w  = (x2-x1)*factor
    h  = (y2-y1)*factor
    x1n = int(round(cx - w/2)); y1n = int(round(cy - h/2))
    x2n = int(round(cx + w/2)); y2n = int(round(cy + h/2))
    return x1n,y1n,x2n,y2n

def _maybe_align_by_eyes(gray, roi):
    gx, gy, gw, gh = roi
    faceROI = gray[gy:gy+gh, gx:gx+gw]
    eyes = _eye_cascade.detectMultiScale(
        faceROI, 1.12, 3, minSize=(int(0.12*gw), int(0.12*gh))
    )
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
        (xA,yA,wA,hA), (xB,yB,wB,hB) = eyes
        ax = gx + xA + wA/2; ay = gy + yA + hA/2
        bx = gx + xB + wB/2; by = gy + yB + hB/2
        angle = math.degrees(math.atan2(by - ay, bx - ax))
        M = cv2.getRotationMatrix2D((gx+gw/2, gy+gh/2), angle, 1.0)
        return M
    return None

def crop_face_square(frame_bgr, box, do_align=True):
    H, W = frame_bgr.shape[:2]
    x1,y1,x2,y2 = box
    x1,y1,x2,y2 = _square_with_margin(x1,y1,x2,y2,W,H,FACE_MARGIN)
    if do_align:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        M = _maybe_align_by_eyes(gray, (x1,y1,x2-x1,y2-y1))
        if M is not None:
            frame_bgr = cv2.warpAffine(frame_bgr, M, (W,H),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT_101)
    x1,y1,x2,y2 = _shrink_box(x1,y1,x2,y2, factor=BOX_SHRINK)
    x1 = max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
    face = frame_bgr[y1:y2, x1:x2]
    return face, (x1,y1,x2,y2)

def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    areaA = (ax2-ax1)*(ay2-ay1); areaB = (bx2-bx1)*(by2-by1)
    union = areaA + areaB - inter + 1e-6
    return inter/union

class EMA:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.v = None
    def update(self, x):
        self.v = float(x) if self.v is None else (self.alpha*float(x) + (1.0-self.alpha)*self.v)
        return self.v

class PersonState:
    def __init__(self):
        self.box = None
        self.age_ema = EMA(alpha=AGE_EMA_ALPHA)
        self.emo_conf_ema = EMA(alpha=EMO_EMA_ALPHA)
        self.emo_lbl = "Unknown"
        self.age_hist = deque(maxlen=AGE_MEDIAN_LEN)
        self.last_area = None
        self.last_good_age = None      # umur valid terakhir
        self.last_display_age = None   # umur yang ditampilkan terakhir
        self.age_scale = None          # per-ID scale (seed dari global)
        self.age_bias = 0.0
        self.cal = AgeCalibrator()     # <<< kalibrator per-ID
        self.last_raw_age = None       # simpan raw buat kalibrasi cepat

def assign_ids(prev_states, boxes):
    assigned = []
    used_prev = set()
    for b in boxes:
        best_id = None; best_iou = 0.0
        for pid, st in prev_states.items():
            if st.box is None or pid in used_prev: continue
            ov = iou(st.box, b)
            if ov > best_iou: best_iou, best_id = ov, pid
        if best_id is not None and best_iou >= 0.25:
            assigned.append((best_id, b)); used_prev.add(best_id)
        else:
            new_id = max(prev_states.keys(), default=0) + 1
            prev_states[new_id] = PersonState()
            assigned.append((new_id, b)); used_prev.add(new_id)
    return assigned, prev_states

def open_cam(src, w, h, fps):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    try: cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except: pass
    try: cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except: pass
    return cap

def build_detector(name="haarcascade_frontalface_default.xml"):
    path = cv2.data.haarcascades + name
    det = cv2.CascadeClassifier(path)
    if det.empty(): raise RuntimeError(f"Gagal load cascade: {path}")
    return det

def detect_faces_gray(detector, gray, scale=HAAR_SCALE, neigh=HAAR_NEIGH, min_size=HAAR_MIN_SIZE):
    eq = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(eq, scaleFactor=scale, minNeighbors=neigh,
                                      flags=cv2.CASCADE_SCALE_IMAGE,
                                      minSize=(min_size, min_size))
    return [(x, y, x+w, y+h) for (x, y, w, h) in faces]

def put_label_with_bg(img, text, org, font_scale=0.6, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = org
    if y - th - 10 < 0: y = y + th + 10
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 2), (0,0,0), -1)
    cv2.putText(img, text, (x + 3, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

# ====== AGE QUALITY & OUTLIER HELPERS ======
def _face_quality(g64):
    blur = cv2.Laplacian(g64, cv2.CV_64F).var()
    mean = float(np.mean(g64))
    std  = float(np.std(g64))
    ok = (blur >= BLUR_MIN_VAR) and (BRIGHT_MIN <= mean <= BRIGHT_MAX) and (std >= CONTRAST_MIN_STD)
    return ok, blur, mean, std

def _is_outlier_age(new_age, last_age):
    if last_age is None: return False
    jump_abs = abs(new_age - last_age)
    jump_lim = max(AGE_JUMP_MAX_ABS, AGE_JUMP_MAX_RATIO * max(1.0, last_age))
    return (jump_abs > jump_lim)

# ====== AGE SCALING GLOBAL ======
def auto_age_scale_from_batch(batch_vals, target=AGE_TARGET):
    """Batch vals: list float output model (biasanya 0..1). Balik: scale float."""
    arr = np.array([v for v in batch_vals if v is not None], dtype=float)
    if arr.size == 0:
        return None
    m = float(np.mean(arr))
    if m < 2.0:
        scale = target / max(m, 1e-6)
        scale = float(np.clip(scale, 25.0, 60.0))
        return scale
    elif m < 10.0:
        return 10.0
    else:
        return 1.0

def apply_scale_per_person(state: "PersonState", raw_age):
    """Kalibrasi per-orang. Kalau scale belum ada, ambil dari global/auto."""
    global AGE_SCALE_GLOBAL
    if raw_age is None:
        return None
    if AGE_SCALE_GLOBAL is None:
        AGE_SCALE_GLOBAL = 40.0  # fallback aman
    if state.age_scale is None:
        state.age_scale = AGE_SCALE_GLOBAL
    # seed scale global -> kemudian dilembutkan lagi oleh calibrator (a*x+b)
    base = float(raw_age) * state.age_scale + state.age_bias
    # dorong minimum dewasa (sering kejadian model ngira bocil)
    if base < ADULT_MODE_MIN:
        base = ADULT_MODE_MIN + 0.3*(base - ADULT_MODE_MIN)  # tarik pelan
    # clamp aman
    base = float(np.clip(base, DISPLAY_MIN_AGE, DISPLAY_MAX_AGE))
    return base

def main():
    global AGE_TARGET, AGE_SCALE_GLOBAL

    print(f"[CHECK] ada .h5 emosi? {Path(EMO_H5_PATH).exists()}")
    print(f"[CHECK] ada .h5 umur?  {Path(AGE_H5_PATH).exists()}")

    cap = open_cam(VIDEO_SRC, FRAME_W, FRAME_H, FPS)
    detector = build_detector()
    outdir = Path(FACES_DIR)

    emo = None
    try:
        emo = EmotionRecognizer(EMO_H5_PATH)
        print("[INFO] Model emosi OK (rebuild+load_weights)")
        _probe = emo.predict_emotion(np.full((48,48), 127, np.uint8))
        print(f"[INFO] Emo warmup: {_probe}")
    except Exception as e:
        print(f"[WARN] Emosi load gagal: {e}")

    age_model = load_age_model(AGE_H5_PATH)

    print(f"[INFO] Tekan 'S' untuk simpan wajah ke {outdir}")
    print("[INFO] Tekan 'ESC' untuk keluar")
    print("[INFO] Hotkeys: '[' turunin target umur, ']' naikin target umur | 1=18y 2=20y 3=22y 4=25y 5=30y (kalibrasi wajah terbesar)")

    states = {}  # pid -> PersonState
    active_pid = None  # id wajah terbesar (buat kalibrasi hotkey)

    while True:
        ok, frame = cap.read()
        if not ok: break

        H, W = frame.shape[:2]
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_boxes = detect_faces_gray(detector, gray_full, scale=HAAR_SCALE, neigh=HAAR_NEIGH, min_size=HAAR_MIN_SIZE)

        # Pilih yang terbesar jadi active_pid
        largest_area = -1
        largest_box  = None

        # Crop + align + filter
        crops = []  # (pid, face_bgr, sq_box)
        id_boxes, states = assign_ids(states, raw_boxes)
        for (pid, b) in id_boxes:
            face_bgr, sq = crop_face_square(frame, b, do_align=True)
            sx1, sy1, sx2, sy2 = sq
            side = min(sx2 - sx1, sy2 - sy1)
            if side < MIN_FACE_PX: continue
            states[pid].box = sq
            crops.append((pid, face_bgr, sq))
            area = (sx2-sx1)*(sy2-sy1)
            if area > largest_area:
                largest_area = area
                largest_box = (pid, sq)

        active_pid = largest_box[0] if largest_box is not None else None

        # AGE batch (CLAHE + resize ke 64)
        gray_faces_pre = []
        for (_, face_bgr, _) in crops:
            g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (64,64), interpolation=cv2.INTER_AREA)
            g = _clahe.apply(g)
            gray_faces_pre.append(g)

        age_raw_out = [None]*len(gray_faces_pre)
        if gray_faces_pre:
            try:
                preds = predict_ages(gray_faces_pre, model=age_model)  # list of floats (RAW)
                age_raw_out = [float(p) for p in preds]
            except Exception as e:
                print("[AGE-ERR]", e)
                age_raw_out = [None]*len(gray_faces_pre)

        # === UPDATE GLOBAL SCALE dari batch RAW ===
        if age_raw_out:
            cand = auto_age_scale_from_batch(age_raw_out, target=AGE_TARGET)
            if cand is not None:
                if AGE_SCALE_GLOBAL is None:
                    AGE_SCALE_GLOBAL = cand
                    print(f"[AGE] init global scale ×{AGE_SCALE_GLOBAL:.1f}")
                else:
                    AGE_SCALE_GLOBAL = 0.2 * cand + 0.8 * AGE_SCALE_GLOBAL

        # ===== Render per-person =====
        for idx, (pid, face_bgr, sq) in enumerate(crops):
            x1,y1,x2,y2 = sq
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)
            st = states[pid]
            labels = []

            # ===== AGE: scale per-ID + calibrator + quality/outlier + median + EMA + rounding =====
            a_disp = None
            a_raw = age_raw_out[idx] if idx < len(age_raw_out) else None
            st.last_raw_age = a_raw

            if a_raw is not None:
                # 1) global-seeded per-ID scaling (dorong minimal dewasa + clamp)
                a_scaled = apply_scale_per_person(st, a_raw)
                # 2) per-ID calibrator linear (a*x+b) — ini ngoreksi bias
                a_cal = st.cal.apply(a_scaled)
                # quality & stabilizer
                g64 = gray_faces_pre[idx]
                area = float((x2-x1)*(y2-y1))
                stable_area = True
                if st.last_area is not None:
                    drift = abs(area - st.last_area) / (st.last_area + 1e-6)
                    if drift > AREA_DRIFT_GATE: stable_area = False
                st.last_area = area

                q_ok, blur, mean, std = _face_quality(g64)
                use_update = stable_area and q_ok and (not _is_outlier_age(a_cal, st.last_good_age))
                if use_update:
                    st.age_hist.append(a_cal)
                    st.last_good_age = a_cal

                base_val = st.last_good_age if st.last_good_age is not None else a_cal
                med_val = base_val
                if len(st.age_hist) >= max(5, AGE_MEDIAN_LEN//3):
                    med_val = float(np.median(list(st.age_hist)))

                a_smooth = st.age_ema.update(med_val)
                a_disp   = max(DISPLAY_MIN_AGE, min(DISPLAY_MAX_AGE, round(a_smooth / AGE_ROUND_STEP) * AGE_ROUND_STEP))
                st.last_display_age = a_disp
                labels.append(f"Age: {a_disp:.1f}y")

            # ===== EMOTION =====
            if emo is not None:
                try:
                    emo_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                    emo_gray = cv2.resize(emo_gray, (48,48), interpolation=cv2.INTER_AREA)
                    emo_gray = _clahe.apply(emo_gray)
                    lbl, conf = emo.predict_emotion(emo_gray)
                    c_smooth = st.emo_conf_ema.update(float(conf))
                    lbl_show = lbl if c_smooth >= EMO_UNKNOWN_TH else "Unknown"
                    if lbl_show != "Unknown": st.emo_lbl = lbl
                    labels.append(f"Emo: {lbl_show} {int(round(c_smooth*100))}%")
                except Exception as e:
                    print(f"[EMO-ERR] {type(e).__name__}: {e}")
                    labels.append("Emo: N/A")

            # tulis stacked
            for j, t in enumerate(labels):
                put_label_with_bg(frame, t, (x1, y1 + 20*j))

        cv2.putText(frame, f"People: {len(crops)}", (10,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # info kalibrasi
        if active_pid in states and states[active_pid].last_display_age is not None:
            put_label_with_bg(frame, f"Active PID: {active_pid}  (kalibrasi: 1/2/3/4/5)", (10, H-10), font_scale=0.6, thickness=2)

        cv2.imshow("face-mini (Age+Emotion)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        if k == ord('s') and crops:
            outdir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time()*1000)
            for idx, (pid, face_bgr, sq) in enumerate(crops, start=1):
                g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                g = cv2.resize(g, (FACE_SAVE_SIZE, FACE_SAVE_SIZE))
                cv2.imwrite(str(outdir/f"face_{ts}_p{idx}.png"), g)
            print(f"[SAVED] {len(crops)} face(s) -> {outdir}")

        # Hotkeys: kalibrasi target umur global
        if k == ord('['):
            AGE_TARGET = max(12.0, AGE_TARGET - 2.0)
            print(f"[AGE] target -> {AGE_TARGET:.1f}")
            AGE_SCALE_GLOBAL = None
        if k == ord(']'):
            AGE_TARGET = min(45.0, AGE_TARGET + 2.0)
            print(f"[AGE] target -> {AGE_TARGET:.1f}")
            AGE_SCALE_GLOBAL = None

        # Hotkeys: kalibrasi per-orang (pakai wajah terbesar)
        if active_pid in states:
            st = states[active_pid]
            rp = st.last_raw_age  # raw pred sebelum scaling
            if rp is not None:
                if k == ord('1'):
                    st.cal.add(rp, 18.0); print("[CAL] PID", active_pid, "-> set 18y")
                if k == ord('2'):
                    st.cal.add(rp, 20.0); print("[CAL] PID", active_pid, "-> set 20y")
                if k == ord('3'):
                    st.cal.add(rp, 22.0); print("[CAL] PID", active_pid, "-> set 22y")
                if k == ord('4'):
                    st.cal.add(rp, 25.0); print("[CAL] PID", active_pid, "-> set 25y")
                if k == ord('5'):
                    st.cal.add(rp, 30.0); print("[CAL] PID", active_pid, "-> set 30y")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
