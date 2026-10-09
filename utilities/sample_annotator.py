import cv2
import json
from pathlib import Path

SAMPLES_DIR = Path(".samples")
OUTPUT_JSON = Path("annotations.json")
WINDOW_NAME = "Annotator"

drawing = False
start_point = None
current_box = None
current_boxes = []
base_image = None
display_image = None


def load_annotations(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_annotations(path, annotations):
    with open(path, "w") as f:
        json.dump(annotations, f, indent=2)


def normalize_box(pt1, pt2):
    x1 = min(pt1[0], pt2[0])
    y1 = min(pt1[1], pt2[1])
    x2 = max(pt1[0], pt2[0])
    y2 = max(pt1[1], pt2[1])
    return {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}


def draw_box(img, box, color=(0, 255, 0), thickness=2, label=None):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label is not None:
        cv2.putText(
            img,
            str(label),
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )


def refresh_display():
    global display_image
    display_image = base_image.copy()

    for i, box in enumerate(current_boxes):
        draw_box(display_image, box, color=(0, 255, 0), thickness=2, label=i + 1)

    if current_box is not None:
        draw_box(display_image, current_box, color=(0, 255, 255), thickness=2)

    cv2.imshow(WINDOW_NAME, display_image)


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, current_box, current_boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_box = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_box = normalize_box(start_point, (x, y))
        refresh_display()

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_box = normalize_box(start_point, (x, y))

        # ignore tiny accidental boxes
        w = current_box["x2"] - current_box["x1"]
        h = current_box["y2"] - current_box["y1"]
        if w > 3 and h > 3:
            current_boxes.append(current_box)

        current_box = None
        refresh_display()


def get_existing_bounds(annotations, image_name):
    for item in annotations:
        if item["imageName"] == image_name:
            return item.get("bounds", [])
    return []


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


def main():
    global base_image, current_boxes, current_box

    image_paths = sorted(SAMPLES_DIR.glob("*.png"))
    if not image_paths:
        print(f"No PNG files found in {SAMPLES_DIR}")
        return

    annotations = load_annotations(OUTPUT_JSON)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("Controls:")
    print("  Click + drag + release : add bounding box")
    print("  s : save current image annotations and go next")
    print("  r : remove last box")
    print("  c : clear all boxes on current image")
    print("  n : skip image")
    print("  q : save progress and quit")

    for idx, image_path in enumerate(image_paths):
        print(f"\n[{idx+1}/{len(image_paths)}] {image_path.name}")

        base_image = cv2.imread(str(image_path))
        if base_image is None:
            print(f"Could not read {image_path}")
            continue

        current_boxes = list(get_existing_bounds(annotations, image_path.name))
        current_box = None

        refresh_display()

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == ord("r"):
                if current_boxes:
                    removed = current_boxes.pop()
                    print(f"Removed last box: {removed}")
                    refresh_display()

            elif key == ord("c"):
                current_boxes = []
                print("Cleared all boxes")
                refresh_display()

            elif key == ord("s"):
                upsert_annotation(annotations, image_path.name, current_boxes)
                save_annotations(OUTPUT_JSON, annotations)
                print(f"Saved {len(current_boxes)} box(es) for {image_path.name}")
                break

            elif key == ord("n"):
                print("Skipped")
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