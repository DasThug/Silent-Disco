from frame.FrameSource import VideoSource
from frame.frame_selection import best_frame_in_interval
import cv2

from analysis.vision import *

# Wherever your current detect_candidates lives
from main import detect_candidates


path = "/Users/aleks/Downloads/PF_silent_disco/GX010808.MP4"

video = VideoSource(path)

person_detector = PersonDetector(
    model_path="yolo11n.pt",
    conf=0.25
)


while True:

    result = best_frame_in_interval(
        video,
        duration=30
    )

    if result is None:
        break

    frames, score = result

    if not frames:
        break

    selected = frames[0]
    frame = selected.image

    # --------------------------
    # 1. YOLO people
    # --------------------------

    people = person_detector.detect(frame)

    # --------------------------
    # 2. Your RGB detector
    # --------------------------

    candidates = detect_candidates(frame)

    counts = {
        "red": 0,
        "green": 0,
        "blue": 0
    }

    display = frame.copy()

    # --------------------------
    # 3. Head regions
    # --------------------------

    for person in people:

        head_box = person_to_head_region(
            person["bbox"],
            frame.shape
        )

        head_box = expand_box(
            head_box,
            frame.shape,
            scale=1.30
        )

        color, scores = classify_head_color(
            head_box,
            candidates
        )

        if color is not None:
            counts[color] += 1

        x1, y1, x2, y2 = head_box

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        if color is not None:

            cv2.putText(
                display,
                color,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    print(
        f"\nFrame {selected.frame_number} "
        f"@ {selected.timestamp:.2f}s"
    )

    print(
        f"R={counts['red']} | "
        f"G={counts['green']} | "
        f"B={counts['blue']}"
    )

    cv2.namedWindow(
        "Headset detections",
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        "Headset detections",
        display
    )

    cv2.resizeWindow(
        "Headset detections",
        1280,
        720
    )

    if cv2.waitKey(0) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()

import cv2
from pathlib import Path

from frame.FrameSource import VideoSource
from frame.frame_selection import best_frame_in_interval


class BoxAnnotator:

    def __init__(self, image):
        self.image = image
        self.boxes = []

        self.start = None
        self.current = None
        self.drawing = False

    def mouse(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            self.start = (x, y)
            self.current = (x, y)

            self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:

            if self.drawing:
                self.current = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:

            if not self.drawing:
                return

            self.current = (x, y)

            x1 = min(self.start[0], x)
            y1 = min(self.start[1], y)

            x2 = max(self.start[0], x)
            y2 = max(self.start[1], y)

            if x2 - x1 > 3 and y2 - y1 > 3:

                self.boxes.append(
                    (x1, y1, x2, y2)
                )

            self.drawing = False

    def draw(self):

        display = self.image.copy()

        for i, (x1, y1, x2, y2) in enumerate(self.boxes):

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                str(i),
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        if self.drawing and self.current:

            cv2.rectangle(
                display,
                self.start,
                self.current,
                (150, 150, 150),
                2
            )

        return display


    
class PersonDetector:
    def __init__(self, model_path="yolo11n.pt", conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf, classes=[0], verbose=False)

        boxes = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            boxes.append({
                "bbox": (int(x1),int(y1),int(x2),int(y2)),
                "confidence": float(box.conf[0])
            })

        return boxes

def person_to_head_region(person_box, frame_shape, head_fraction=0.35):
    x1, y1, x2, y2 = person_box

    h = y2 - y1

    head_y2 = int(y1 + h * head_fraction)

    return (x1, y1, x2, head_y2)

def expand_box(box, frame_shape, scale=1.25):
    """ The enlargement matters because an earcup can lie slightly outside a strict head box. """
    x1, y1, x2, y2 = box

    h, w = frame_shape[:2]

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    nx1 = max(0, int(cx - bw / 2))
    ny1 = max(0, int(cy - bh / 2))
    nx2 = min(w - 1, int(cx + bw / 2))
    ny2 = min(h - 1, int(cy + bh / 2))

    return nx1, ny1, nx2, ny2

def blob_inside_box(blob, box):
    cx, cy = blob["center"]

    x1, y1, x2, y2 = box

    return (
        x1 <= cx <= x2 and
        y1 <= cy <= y2
    )

def blobs_in_head(head_box, candidates):

    result = {}

    for color in ("red", "green", "blue"):

        blobs = candidates[color]["blobs"]

        result[color] = [
            blob
            for blob in blobs
            if blob_inside_box(blob, head_box)
        ]

    return result

def classify_head_color(head_box, candidates):

    color_scores = {}

    for color in ("red", "green", "blue"):

        blobs = [
            blob
            for blob in candidates[color]["blobs"]
            if blob_inside_box(blob, head_box)
        ]

        score = sum(blob["area"] for blob in blobs)

        color_scores[color] = score

    best_color = max(
        color_scores,
        key=color_scores.get
    )

    if color_scores[best_color] == 0:
        return None, color_scores

    return best_color, color_scores

# score = sum(blob["area"] * max(blob["delta_v"], 0) * max(blob["delta_dominance"], 0) for blob in blobs)