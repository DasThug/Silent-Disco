import cv2
import json
import math
from pathlib import Path

from analysis.color_segmentation import RGBBlobDetector


SAMPLES_DIR = Path(".samples")
OUTPUT_JSON = Path("headset_annotations.json")
WINDOW_NAME = "Headset Candidate Annotator"

CROP_s = 6
CROP_SIZE = 96
COLS = 10

DELTA_V_RANGE = (40, 200)
DELTA_DOMINANCE_RANGE = (40, 200)


# -----------------------------
# JSON helpers
# -----------------------------
def load_annotations(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_annotations(path, annotations):
    with open(path, "w") as f:
        json.dump(annotations, f, indent=2)


def upsert_annotation(annotations, image_name, bounds):
    entry = {
        "imageName": image_name,
        "bounds": bounds
    }

    for i, item in enumerate(annotations):
        if item["imageName"] == image_name:
            annotations[i] = entry
            return

    annotations.append(entry)


def get_existing_bounds(annotations, image_name):
    for item in annotations:
        if item["imageName"] == image_name:
            return item.get("bounds", [])
    return []


# -----------------------------
# Blob/crop helpers
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def extract_blob_bbox(blob):
    """
    Try to infer bbox from your blob dict.
    Adjust here if your blob structure uses different names.
    """

    # Case 1: blob["bbox"] exists
    if "bbox" in blob:
        bbox = blob["bbox"]

        # tuple/list format
        if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            x, y, w, h = bbox
            return int(x), int(y), int(w), int(h)

        # dict format
        if isinstance(bbox, dict):
            if all(k in bbox for k in ("x", "y", "w", "h")):
                return int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
            if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    # Case 2: direct xywh
    if all(k in blob for k in ("x", "y", "w", "h")):
        return int(blob["x"]), int(blob["y"]), int(blob["w"]), int(blob["h"])

    # Case 3: direct x1y1x2y2
    if all(k in blob for k in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = blob["x1"], blob["y1"], blob["x2"], blob["y2"]
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    # Case 4: OpenCV connected components style keys
    if all(k in blob for k in ("left", "top", "width", "height")):
        return int(blob["left"]), int(blob["top"]), int(blob["width"]), int(blob["height"])

    raise KeyError(
        "Could not infer bbox from blob. "
        "Please inspect one blob and adapt extract_blob_bbox()."
    )


def crop_from_blob(frame, blob, scale=6):
    """
    Build the same kind of crop as detector.get_blob_crops(),
    while also returning the ORIGINAL frame coordinates.
    """
    H, W = frame.shape[:2]

    x, y, w, h = extract_blob_bbox(blob)

    cx = x + w / 2.0
    cy = y + h / 2.0

    crop_w = max(1, int(round(w * scale)))
    crop_h = max(1, int(round(h * scale)))

    x1 = int(round(cx - crop_w / 2))
    y1 = int(round(cy - crop_h / 2))
    x2 = int(round(cx + crop_w / 2))
    y2 = int(round(cy + crop_h / 2))

    x1 = clamp(x1, 0, W - 1)
    y1 = clamp(y1, 0, H - 1)
    x2 = clamp(x2, x1 + 1, W)
    y2 = clamp(y2, y1 + 1, H)

    crop = frame[y1:y2, x1:x2].copy()

    bounds = {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
    }

    return crop, bounds


def build_candidates(frame, detector):
    """
    Runs the full RGB blob pipeline on one frame and returns:
    [
        {
            "crop": ...,
            "bounds": {...},
            "color": "red"/"green"/"blue"
        },
        ...
    ]
    """
    candidates = []

    masks = detector.segment_colors(frame)

    for color, mask in masks.items():
        cleaned = detector.clean_mask(mask)
        blobs = detector.extract_blob_features(cleaned)
        blobs = detector.filter_blobs(blobs)
        blobs = detector.add_local_contrast_features(frame, blobs, color, debug=False)
        blobs = detector.filter_led_blobs(
            blobs,
            delta_v_range=DELTA_V_RANGE,
            delta_dominance_range=DELTA_DOMINANCE_RANGE
        )

        for blob in blobs:
            try:
                crop, bounds = crop_from_blob(frame, blob, scale=CROP_SCALE)
            except Exception as e:
                print(f"[WARN] Failed to crop blob in {color}: {e}")
                continue

            if crop is None or crop.size == 0:
                continue

            candidates.append({
                "crop": crop,
                "bounds": bounds,
                "color": color,
            })

    return candidates


# -----------------------------
# Collage UI
# -----------------------------
class CandidateAnnotator:
    def __init__(self, crop_size=96, cols=10):
        self.crop_size = crop_size
        self.cols = cols

        self.colors = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
        }

        self.candidates = []
        self.selected = set()
        self.tile_rects = []
        self.collage = None
        self.window_name = WINDOW_NAME

    def load_candidates(self, candidates, preselected_bounds=None):
        self.candidates = candidates
        self.selected = set()

        if preselected_bounds:
            bound_keys = {
                (b["x1"], b["y1"], b["x2"], b["y2"])
                for b in preselected_bounds
            }
            for i, cand in enumerate(candidates):
                b = cand["bounds"]
                key = (b["x1"], b["y1"], b["x2"], b["y2"])
                if key in bound_keys:
                    self.selected.add(i)

        self.render()

    def render(self):
        self.tile_rects = []

        if not self.candidates:
            self.collage = 255 * (cv2.UMat(300, 600, cv2.CV_8UC3).get())
            cv2.putText(
                self.collage,
                "No candidates found",
                (40, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )
            return

        rows = math.ceil(len(self.candidates) / self.cols)
        self.collage = cv2.cvtColor(
            cv2.UMat(rows * self.crop_size, self.cols * self.crop_size, cv2.CV_8UC1).get(),
            cv2.COLOR_GRAY2BGR
        )

        for i, cand in enumerate(self.candidates):
            tile = cv2.resize(cand["crop"], (self.crop_size, self.crop_size))
            color = self.colors[cand["color"]]

            # base border by RGB channel
            cv2.rectangle(tile, (0, 0), (self.crop_size - 1, self.crop_size - 1), color, 2)

            # draw candidate index
            cv2.putText(
                tile,
                str(i),
                (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # mark selected
            if i in self.selected:
                cv2.rectangle(tile, (3, 3), (self.crop_size - 4, self.crop_size - 4), (0, 255, 255), 3)

            y = (i // self.cols) * self.crop_size
            x = (i % self.cols) * self.crop_size

            self.collage[y:y + self.crop_size, x:x + self.crop_size] = tile
            self.tile_rects.append((x, y, x + self.crop_size, y + self.crop_size))

    def click(self, x, y):
        for i, (x1, y1, x2, y2) in enumerate(self.tile_rects):
            if x1 <= x < x2 and y1 <= y < y2:
                if i in self.selected:
                    self.selected.remove(i)
                else:
                    self.selected.add(i)
                self.render()
                break

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click(x, y)

    def selected_bounds(self):
        return [self.candidates[i]["bounds"] for i in sorted(self.selected)]

    def show(self):
        cv2.imshow(self.window_name, self.collage)


# -----------------------------
# Main
# -----------------------------
def main():
    detector = RGBBlobDetector()
    image_paths = sorted(SAMPLES_DIR.glob("*.png"))

    if not image_paths:
        print(f"No PNG files found in {SAMPLES_DIR}")
        return

    annotations = load_annotations(OUTPUT_JSON)
    annotator = CandidateAnnotator(crop_size=CROP_SIZE, cols=COLS)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1400, 900)
    cv2.setMouseCallback(WINDOW_NAME, annotator.mouse_callback)

    print("Controls:")
    print("  Left click : select/deselect candidate")
    print("  s          : save selected crops and go next")
    print("  c          : clear current selection")
    print("  n          : skip image")
    print("  q          : save progress and quit")

    for idx, image_path in enumerate(image_paths):
        print(f"\n[{idx+1}/{len(image_paths)}] {image_path.name}")

        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Could not read {image_path}")
            continue

        candidates = build_candidates(frame, detector)
        print(f"  Candidates found: {len(candidates)}")

        preselected_bounds = get_existing_bounds(annotations, image_path.name)
        annotator.load_candidates(candidates, preselected_bounds=preselected_bounds)
        annotator.show()

        while True:
            annotator.show()
            key = cv2.waitKey(20) & 0xFF

            if key == ord("c"):
                annotator.selected.clear()
                annotator.render()

            elif key == ord("s"):
                bounds = annotator.selected_bounds()
                upsert_annotation(annotations, image_path.name, bounds)
                save_annotations(OUTPUT_JSON, annotations)
                print(f"  Saved {len(bounds)} headset box(es)")
                break

            elif key == ord("n"):
                print("  Skipped")
                break

            elif key == ord("q"):
                save_annotations(OUTPUT_JSON, annotations)
                print("Saved progress and quit")
                cv2.destroyAllWindows()
                return

    save_annotations(OUTPUT_JSON, annotations)
    cv2.destroyAllWindows()
    print(f"\nDone. Saved annotations to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()