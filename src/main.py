import cv2
import argparse

def open_cam(src, w, h, fps):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    # coba kunci auto exposure/focus kalau didukung driver
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return cap

def build_detector(cascade_name="haarcascade_frontalface_default.xml"):
    # OpenCV bawa path data cascades di cv2.data.haarcascades
    cascade_path = cv2.data.haarcascades + cascade_name
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Gagal load cascade: {cascade_path}")
    return face_cascade

def detect_faces_gray(face_cascade, gray, scale=1.2, neigh=5, min_size=24):
    # equalize biar kontras stabil; Haar suka yang gini
    eq = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        eq,
        scaleFactor=scale,
        minNeighbors=neigh,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(min_size, min_size)
    )
    # convert (x,y,w,h) -> (x1,y1,x2,y2)
    boxes = [(x, y, x+w, y+h) for (x, y, w, h) in faces]
    return boxes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="0", help="'0' webcam atau path file")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    # tuning detektor Haar
    ap.add_argument("--scale", type=float, default=1.2, help="scaleFactor (1.05-1.4). Kecil = lebih teliti, tapi lebih lambat")
    ap.add_argument("--neigh", type=int, default=5, help="minNeighbors (3-8). Besar = lebih ketat, lebih sedikit false positive")
    ap.add_argument("--min", type=int, default=24, help="minSize piksel wajah (12-64). Besar = buang wajah kecil jauh di belakang")
    args = ap.parse_args()

    src = int(args.video) if args.video.isdigit() else args.video
    cap = open_cam(src, args.width, args.height, args.fps)
    face_cascade = build_detector()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Grayscale for detection & any future model inputs
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        boxes = detect_faces_gray(face_cascade, gray, scale=args.scale, neigh=args.neigh, min_size=args.min)

        # draw
        for (x1,y1,x2,y2) in boxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)

        people = len(boxes)
        cv2.putText(frame, f"People: {people}", (10,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow("face-mini (OpenCV Haar, grayscale)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
