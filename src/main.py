import cv2
import argparse
from pathlib import Path
import time
from src.umur_model.umur_model import load_model as load_age_model, predict_ages


def open_cam(src, w, h, fps):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    # kunci auto exposure/focus kalau didukung driver (kalem, gak semua kamera nurut)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return cap

def build_detector(name="haarcascade_frontalface_default.xml"):
    path = cv2.data.haarcascades + name
    det = cv2.CascadeClassifier(path)
    if det.empty():
        raise RuntimeError(f"Gagal load cascade: {path}")
    return det

def detect_faces_gray(detector, gray, scale=1.2, neigh=5, min_size=24):
    eq = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        eq,
        scaleFactor=scale,
        minNeighbors=neigh,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(min_size, min_size)
    )
    # (x,y,w,h) -> (x1,y1,x2,y2)
    return [(x, y, x+w, y+h) for (x, y, w, h) in faces]

def save_faces(gray_frame, boxes, outdir: Path, face_size=64, prefix="faces"):
    """
    Simpan semua wajah grayscale sebagai {prefix}_{ts_ms}_p{idx}.png ke data/faces.
    idx = urutan orang di frame (1-based).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    saved = []
    for idx, (x1,y1,x2,y2) in enumerate(boxes, start=1):
        crop = gray_frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        face = cv2.resize(crop, (face_size, face_size), interpolation=cv2.INTER_AREA)
        filename = f"{prefix}_{ts}_p{idx}.png"
        path = outdir / filename
        cv2.imwrite(str(path), face)
        saved.append(filename)
    return saved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="0", help="'0' webcam atau path file")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=float, default=1.2)
    ap.add_argument("--neigh", type=int, default=5)
    ap.add_argument("--min", type=int, default=24)
    ap.add_argument("--faces-dir", default="data/faces", help="folder output wajah (default: data/faces)")
    ap.add_argument("--face-size", type=int, default=64)
    args = ap.parse_args()

    src = int(args.video) if args.video.isdigit() else args.video
    cap = open_cam(src, args.width, args.height, args.fps)
    detector = build_detector()
    outdir = Path(args.faces_dir)
    age_model= load_age_model("src/umur_model/age_model.h5")


    print("[INFO] Tekan 'S' untuk simpan semua wajah (grayscale) ke", outdir)
    print("[INFO] Tekan 'ESC' untuk keluar")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = detect_faces_gray(detector, gray, args.scale, args.neigh, args.min)

        # gambar bbox + label Person #i
        for idx, (x1,y1,x2,y2) in enumerate(boxes, start=1):
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)
            label = f"Person #{idx}"
            ytxt = y1 - 8 if y1 - 8 > 18 else y1 + 18
            cv2.putText(frame, label, (x1, ytxt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        people = len(boxes)
        cv2.putText(frame, f"People: {people}", (10,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow("face-mini (press S to save faces)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('s') and people > 0:
            saved = save_faces(gray, boxes, outdir, face_size=args.face_size, prefix="face")
            print(f"[SAVED] {len(saved)} file → {outdir}/  ({', '.join(saved)})")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
