
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from analysis.color_segmentation import RGBBlobDetector


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / ".samples"
CSV_PATH = SAMPLES_DIR / "selected_blobs.csv"

COLORS = ("red", "green", "blue")

BOX_COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
}

SELECTED_COLOR = (0, 165, 255)  # orange in BGR


def contour_mean_hsv(frame, contour):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    cv2.drawContours(
        mask,
        [contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    mean_h, mean_s, mean_v, _ = cv2.mean(hsv, mask=mask)

    return float(mean_h), float(mean_s), float(mean_v)


def make_blob_id(filename, color, blob_index, bbox):
    x, y, w, h = bbox
    return f"{filename}|{color}|{blob_index}|{x},{y},{w},{h}"


def serialize_blob(frame, filename, color, blob_index, blob):
    x, y, w, h = map(int, blob["bbox"])
    cx, cy = map(int, blob["center"])

    mean_h, mean_s, mean_v = contour_mean_hsv(
        frame,
        blob["contour"]
    )

    return {
        "blob_id": make_blob_id(
            filename,
            color,
            blob_index,
            (x, y, w, h)
        ),
        "filename": filename,
        "color": color,
        "blob_index": blob_index,

        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": w,
        "bbox_h": h,

        "center_x": cx,
        "center_y": cy,

        "area": float(blob["area"]),
        "perimeter": float(blob["perimeter"]),
        "circularity": float(blob["circularity"]),
        "aspect_ratio": float(blob["aspect_ratio"]),
        "extent": float(blob["extent"]),
        "solidity": float(blob["solidity"]),

        "blob_v": float(blob["blob_v"]),
        "surrounding_v": float(blob["surrounding_v"]),
        "delta_v": float(blob["delta_v"]),

        "blob_dominance": float(blob["blob_dominance"]),
        "surrounding_dominance": float(blob["surrounding_dominance"]),
        "delta_dominance": float(blob["delta_dominance"]),

        "ring_size": int(blob["ring_size"]),

        "mean_h": mean_h,
        "mean_s": mean_s,
        "mean_v": mean_v,

        "_contour": blob["contour"],
    }


def process_frame(detector, path, debug=True):
    if debug:
        print(f"\n[LOAD] {path}")

    frame = cv2.imread(str(path))

    if frame is None:
        raise RuntimeError(f"Could not read image: {path}")

    if debug:
        print(f"[IMAGE] shape={frame.shape}, dtype={frame.dtype}")

    masks = detector.segment_colors(frame)

    if debug:
        print("[SEGMENT] masks created")

    all_blobs = []

    for color in COLORS:
        mask = masks[color]

        if debug:
            print(
                f"[MASK:{color}] "
                f"nonzero={cv2.countNonZero(mask)}"
            )

        cleaned = detector.clean_mask(mask)

        if debug:
            print(
                f"[CLEAN:{color}] "
                f"nonzero={cv2.countNonZero(cleaned)}"
            )

        blobs = detector.extract_blob_features(cleaned)

        if debug:
            print(
                f"[BLOBS:{color}] extracted={len(blobs)}"
            )

        blobs = detector.add_local_contrast_features(
            frame,
            blobs,
            color,
            debug=False
        )

        if debug:
            print(
                f"[FEATURES:{color}] local contrast added"
            )

        for blob_index, blob in enumerate(blobs):
            serialized = serialize_blob(
                frame,
                path.name,
                color,
                blob_index,
                blob
            )

            all_blobs.append(serialized)

    if debug:
        print(
            f"[FRAME DONE] total blobs={len(all_blobs)}"
        )

    return frame, all_blobs


def write_csv(selected):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "blob_id",
        "filename",
        "color",
        "blob_index",

        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",

        "center_x",
        "center_y",

        "area",
        "perimeter",
        "circularity",
        "aspect_ratio",
        "extent",
        "solidity",

        "blob_v",
        "surrounding_v",
        "delta_v",

        "blob_dominance",
        "surrounding_dominance",
        "delta_dominance",

        "ring_size",

        "mean_h",
        "mean_s",
        "mean_v"
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        for row in selected.values():
            writer.writerow(row)

    print(
        f"[CSV] saved {len(selected)} selected blobs -> {CSV_PATH}"
    )


def print_blob_features(blob):
    print("\n[SELECTED BLOB]")
    print(f"  id:                {blob['blob_id']}")
    print(f"  color:             {blob['color']}")
    print(f"  area:              {blob['area']:.2f}")
    print(f"  perimeter:         {blob['perimeter']:.2f}")
    print(f"  circularity:       {blob['circularity']:.3f}")
    print(f"  aspect_ratio:      {blob['aspect_ratio']:.3f}")
    print(f"  extent:            {blob['extent']:.3f}")
    print(f"  solidity:          {blob['solidity']:.3f}")
    print(f"  delta_v:           {blob['delta_v']:.2f}")
    print(f"  delta_dominance:   {blob['delta_dominance']:.2f}")
    print(f"  ring_size:         {blob['ring_size']}")
    print(f"  mean_h:            {blob['mean_h']:.2f}")
    print(f"  mean_s:            {blob['mean_s']:.2f}")
    print(f"  mean_v:            {blob['mean_v']:.2f}")


def run_blob_selector(debug=True):
    print(f"[ROOT]    {PROJECT_ROOT}")
    print(f"[SAMPLES] {SAMPLES_DIR}")
    print(f"[CSV]     {CSV_PATH}")

    if not SAMPLES_DIR.exists():
        raise RuntimeError(
            f"Samples folder does not exist: {SAMPLES_DIR}"
        )

    image_paths = sorted(SAMPLES_DIR.glob("*.png"))

    print(f"[FOUND] {len(image_paths)} PNG files")

    if not image_paths:
        raise RuntimeError(
            f"No PNG files found in {SAMPLES_DIR}"
        )

    detector = RGBBlobDetector()

    print(
        f"[DETECTOR] {detector} "
        f"type={type(detector)}"
    )

    selected = {}

    if CSV_PATH.exists():
        print(f"[CSV] existing file found: {CSV_PATH}")

        with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                blob_id = row.get("blob_id")

                if blob_id:
                    selected[blob_id] = row

        print(f"[CSV] restored {len(selected)} selections")

    current_index = 0
    frame = None
    blobs = []
    hovered_blob = None

    window_name = "Blob Selector"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        1400,
        900
    )

    def load_current():
        nonlocal frame, blobs, hovered_blob

        path = image_paths[current_index]

        frame, blobs = process_frame(
            detector,
            path,
            debug=debug
        )

        hovered_blob = None

        print(
            f"[FRAME] {current_index + 1}/{len(image_paths)} "
            f"{path.name}"
        )

    def draw():
        if frame is None:
            return

        display = frame.copy()

        for blob in blobs:
            x = blob["bbox_x"]
            y = blob["bbox_y"]
            w = blob["bbox_w"]
            h = blob["bbox_h"]

            is_selected = blob["blob_id"] in selected

            color = (
                SELECTED_COLOR
                if is_selected
                else BOX_COLORS[blob["color"]]
            )

            thickness = 5 if is_selected else 2

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                color,
                thickness
            )

        if hovered_blob is not None:
            blob = hovered_blob

            x = blob["bbox_x"]
            y = blob["bbox_y"]
            w = blob["bbox_w"]
            h = blob["bbox_h"]

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                3
            )

            lines = [
                f"{blob['color']} blob",
                f"area: {blob['area']:.1f}",
                f"circ: {blob['circularity']:.3f}",
                f"aspect: {blob['aspect_ratio']:.3f}",
                f"extent: {blob['extent']:.3f}",
                f"solidity: {blob['solidity']:.3f}",
                f"dV: {blob['delta_v']:.1f}",
                f"dDom: {blob['delta_dominance']:.1f}",
                f"H: {blob['mean_h']:.1f}",
                f"S: {blob['mean_s']:.1f}",
                f"V: {blob['mean_v']:.1f}",
            ]

            for j, text in enumerate(lines):
                cv2.putText(
                    display,
                    text,
                    (30, 45 + j * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        cv2.putText(
            display,
            (
                f"{current_index + 1}/{len(image_paths)} "
                f"| blobs: {len(blobs)} "
                f"| selected: {len(selected)} "
                f"| A/D prev/next | click select | S save | Q quit"
            ),
            (20, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(window_name, display)

    def mouse(event, mx, my, flags, param):
        nonlocal hovered_blob

        if event == cv2.EVENT_MOUSEMOVE:
            matches = []

            for blob in blobs:
                x = blob["bbox_x"]
                y = blob["bbox_y"]
                w = blob["bbox_w"]
                h = blob["bbox_h"]

                if x <= mx <= x + w and y <= my <= y + h:
                    matches.append(
                        (w * h, blob)
                    )

            if matches:
                _, hovered_blob = min(
                    matches,
                    key=lambda item: item[0]
                )
            else:
                hovered_blob = None

            draw()

        elif event == cv2.EVENT_LBUTTONDOWN:
            matches = []

            for blob in blobs:
                x = blob["bbox_x"]
                y = blob["bbox_y"]
                w = blob["bbox_w"]
                h = blob["bbox_h"]

                if x <= mx <= x + w and y <= my <= y + h:
                    matches.append(
                        (w * h, blob)
                    )

            if not matches:
                print(
                    f"[CLICK] no blob at ({mx}, {my})"
                )
                return

            _, blob = min(
                matches,
                key=lambda item: item[0]
            )

            blob_id = blob["blob_id"]

            if blob_id in selected:
                del selected[blob_id]

                print(
                    f"[DESELECT] {blob_id}"
                )
            else:
                row = {
                    key: value
                    for key, value in blob.items()
                    if not key.startswith("_")
                }

                selected[blob_id] = row

                print(
                    f"[SELECT] {blob_id}"
                )

                print_blob_features(blob)

            write_csv(selected)
            draw()

    cv2.setMouseCallback(
        window_name,
        mouse
    )

    load_current()
    draw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            print("[QUIT]")
            break

        elif key == ord("d"):
            if current_index + 1 < len(image_paths):
                current_index += 1
                load_current()
                draw()

        elif key == ord("a"):
            if current_index > 0:
                current_index -= 1
                load_current()
                draw()

        elif key == ord("s"):
            write_csv(selected)

    write_csv(selected)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_blob_selector(debug=True)
