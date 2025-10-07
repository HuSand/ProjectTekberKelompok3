# src/main.py
# jalankan dari root project:  py -m src.main

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
import math
import numpy as np
from pathlib import Path

from src.umur_model.umur_model import load_model as load_age_model, predict_ages
from src.emosi_model.emosi_model import EmotionRecognizer

# ================== CONFIG ==================
VIDEO_SRC = 0
FRAME_W, FRAME_H, FPS = 1280, 720, 30

FACES_DIR       = "data/faces"
FACE_SAVE_SIZE  = 64

EMO_H5_PATH = "src/emosi_model/fer2013_cnn.h5"   # JSON tidak dipakai
AGE_H5_PATH = "src/umur_model/age_model.h5"

FACE_MARGIN     = 0.45
MIN_FACE_PX     = 120
EMO_UNKNOWN_TH  = 0.42     # sedikit dinaikkan biar gak gampang salah
AGE_EMA_ALPHA   = 0.18     # lebih kalem biar tidak sensitif
AGE_ROUND_STEP  = 0.5      # tampilkan umur dibulatkan ke 0.5 tahun


# ============================================

_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

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

def _maybe_align_by_eyes(gray, roi):
    gx, gy, gw, gh = roi
    faceROI = gray[gy:gy+gh, gx:gx+gw]
    eyes = _eye_cascade.detectMultiScale(faceROI, 1.12, 3,
                                         minSize=(int(0.12*gw), int(0.12*gh)))
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
        self.emo_lbl = "Unknown"
        self.emo_conf_ema = EMA(alpha=0.5)


def assign_ids(prev_states, boxes):
    assigned = []
    used_prev = set()
    for b in boxes:
        best_id = None; best_iou = 0.0
        for pid, st in prev_states.items():
            if st.box is None or pid in used_prev:
                continue
            ov = iou(st.box, b)
            if ov > best_iou:
                best_iou, best_id = ov, pid
        if best_id is not None and best_iou >= 0.25:
            assigned.append((best_id, b))
            used_prev.add(best_id)
        else:
            new_id = max(prev_states.keys(), default=0) + 1
            prev_states[new_id] = PersonState()
            assigned.append((new_id, b))
            used_prev.add(new_id)
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
    if det.empty():
        raise RuntimeError(f"Gagal load cascade: {path}")
    return det

def detect_faces_gray(detector, gray, scale=1.08, neigh=6, min_size=36):
    eq = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(eq, scaleFactor=scale, minNeighbors=neigh,
                                      flags=cv2.CASCADE_SCALE_IMAGE,
                                      minSize=(min_size, min_size))
    return [(x, y, x+w, y+h) for (x, y, w, h) in faces]

def put_label_with_bg(img, text, org, font_scale=0.6, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = org
    if y - th - 10 < 0:
        y = y + th + 10
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 2), (0,0,0), -1)
    cv2.putText(img, text, (x + 3, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

def main():
    print(f"[CHECK] ada .h5 emosi? {Path(EMO_H5_PATH).exists()}")
    print(f"[CHECK] ada .h5 umur? {Path(AGE_H5_PATH).exists()}")

    cap = open_cam(VIDEO_SRC, FRAME_W, FRAME_H, FPS)
    detector = build_detector()
    outdir = Path(FACES_DIR)

    emo = None
    try:
        emo = EmotionRecognizer(EMO_H5_PATH)  # rebuild + load_weights
        print("[INFO] Model emosi OK (rebuild+load_weights)")
    except Exception as e:
        print(f"[WARN] Emosi load gagal: {e}")

    age_model = load_age_model(AGE_H5_PATH)

    print(f"[INFO] Tekan 'S' untuk simpan wajah ke {outdir}")
    print("[INFO] Tekan 'ESC' untuk keluar")

    states = {}  # pid -> PersonState

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_boxes = detect_faces_gray(detector, gray_full, scale=1.08, neigh=6, min_size=36)

        # Crop square + align + filter min size + assign IDs
        crops = []  # (pid, face_bgr, sq_box)
        id_boxes, states = assign_ids(states, raw_boxes)
        for (pid, b) in id_boxes:
            face_bgr, sq = crop_face_square(frame, b, do_align=True)
            sx1, sy1, sx2, sy2 = sq
            side = min(sx2 - sx1, sy2 - sy1)
            if side < MIN_FACE_PX:
                continue
            states[pid].box = sq
            crops.append((pid, face_bgr, sq))

        # AGE batch
        gray_faces_64 = []
        id_order = []
        for (pid, face_bgr, sq) in crops:
            g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            gray_faces_64.append(g)  # predict_ages akan resize sendiri (64x64, 0..1)
            id_order.append(pid)

        age_out = []
        if gray_faces_64:
            try:
                age_out = predict_ages(gray_faces_64, model=age_model)
            except Exception as e:
                print("[AGE-ERR]", e)
                age_out = [25.0]*len(gray_faces_64)

        # Tulis hasil
        idx_age = 0
        for (pid, face_bgr, sq) in crops:
            x1,y1,x2,y2 = sq
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)

            st = states.get(pid, None)
            labels = []

            # AGE smoothing
            age_val = None
            if idx_age < len(age_out):
                age_val = float(age_out[idx_age]); idx_age += 1
            if st and age_val is not None:
                age_val = st.age_ema.update(age_val)
                # round to nearest 0.5
                age_val = round(age_val / AGE_ROUND_STEP) * AGE_ROUND_STEP
                labels.append(f"Age: {age_val:.1f}y")


            # EMO + smoothing + threshold Unknown
            if emo is not None and st is not None:
                try:
                    emo_lbl, emo_p = emo.predict_emotion(face_bgr, unknown_threshold=EMO_UNKNOWN_TH)
                    st.emo_lbl = emo_lbl
                    emo_p = st.emo_conf_ema.update(emo_p)
                    labels.append(f"Emo: {st.emo_lbl} {emo_p*100:.0f}%")
                except Exception:
                    labels.append("Emo: N/A")

            for j, t in enumerate(labels):
                put_label_with_bg(frame, t, (x1, y1 + 20*j))

        cv2.putText(frame, f"People: {len(crops)}", (10,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
