import cv2, argparse
import mediapipe as mp

def open_cam(src, w, h, fps):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    # coba kunci exposure biar stabil (ignore kalau device gak support)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return cap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="0", help="'0' webcam atau path file")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.55)
    args = ap.parse_args()

    src = int(args.video) if args.video.isdigit() else args.video
    cap = open_cam(src, args.width, args.height, args.fps)

    face = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=args.conf
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # MediaPipe pakainya RGB, tapi kita juga bikin versi grayscale
        # cuma buat overlay text “low cost”. Deteksi tetap pakai RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        res = face.process(rgb)
        boxes = []
        if res.detections:
            h, w = frame.shape[:2]
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x1 = max(int(bb.xmin*w), 0)
                y1 = max(int(bb.ymin*h), 0)
                x2 = min(int((bb.xmin+bb.width)*w),  w-1)
                y2 = min(int((bb.ymin+bb.height)*h), h-1)
                if (x2-x1) >= 12 and (y2-y1) >= 12:
                    boxes.append((x1,y1,x2,y2))

        # gambar kotak di frame BGR biar keliatan
        for (x1,y1,x2,y2) in boxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)

        # tampilkan count di pojok pakai grayscale sebagai background tipis
        people = len(boxes)
        cv2.putText(frame, f"People: {people}", (10,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow("face-mini (count only)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC buat keluar
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
