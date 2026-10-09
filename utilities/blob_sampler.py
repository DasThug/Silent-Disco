"""Interactive RGB blob sampling into CSV. Keep RGBBlobDetector unchanged."""
import csv
from pathlib import Path

import cv2
import numpy as np

# Change only this import to match the module containing YOUR RGBBlobDetector.
from analysis.color_segmentation import RGBBlobDetector

IMAGE_DIR = Path('.samples')
CSV_PATH = Path('blob_samples.csv')
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 850
PANEL_WIDTH = 355
COLORS = ('red', 'green', 'blue')
BGR = {'red': (40, 40, 255), 'green': (50, 220, 50), 'blue': (255, 140, 40)}

FIELDS = [
    'frame_id', 'filename', 'color', 'blob_id',
    'center_x', 'center_y', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
    'area', 'perimeter', 'circularity', 'aspect_ratio', 'extent', 'solidity',
    'hue_min', 'hue_mean', 'hue_max',
    'saturation_min', 'saturation_mean', 'saturation_max',
    'value_min', 'value_mean', 'value_max',
    'blob_v', 'surrounding_v', 'delta_v',
    'blob_dominance', 'surrounding_dominance', 'delta_dominance', 'ring_size'
]


def contour_hsv_stats(hsv, blob):
    """Statistics for pixels INSIDE the filled contour, excluding its bbox background."""
    x, y, w, h = blob['bbox']
    roi = hsv[y:y+h, x:x+w]
    contour = blob['contour'].copy()
    contour[:, :, 0] -= x
    contour[:, :, 1] -= y
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    pixels = roi[mask != 0]
    result = {}
    for i, name in enumerate(('hue', 'saturation', 'value')):
        v = pixels[:, i]
        result[f'{name}_min'] = int(v.min())
        result[f'{name}_mean'] = float(v.mean())
        result[f'{name}_max'] = int(v.max())
    return result


def choose_blobs(frame, blobs, color, frame_id, old_ids):
    """Interactive selection. Returns (selected indices, quit_requested)."""
    h, w = frame.shape[:2]
    scale = min(1., MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h)
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    background = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    window = f'Blob sampler - {color}'
    selected = set()
    hovered = None
    dirty = True

    def nearest(mx, my):
        x0, y0 = mx / scale, my / scale
        options = []
        for i, blob in enumerate(blobs):
            x, y, bw, bh = blob['bbox']
            if x <= x0 < x + bw and y <= y0 < y + bh:
                options.append((bw * bh, i))
        return min(options)[1] if options else None

    def mouse(event, mx, my, flags, userdata):
        nonlocal hovered, dirty
        if mx >= sw or my >= sh:
            target = None
        else:
            target = nearest(mx, my)
        if event == cv2.EVENT_MOUSEMOVE and target != hovered:
            hovered = target
            dirty = True
        elif event == cv2.EVENT_LBUTTONDOWN and target is not None:
            if target in selected:
                selected.remove(target)
            else:
                selected.add(target)
            hovered = target
            dirty = True

    def render():
        canvas = np.zeros((sh, sw + PANEL_WIDTH, 3), np.uint8)
        canvas[:, :sw] = background
        overlay = canvas.copy()
        for i, blob in enumerate(blobs):
            x, y, bw, bh = blob['bbox']
            xy1 = (round(x * scale), round(y * scale))
            xy2 = (round((x + bw) * scale), round((y + bh) * scale))
            is_selected = i in selected
            cv2.rectangle(overlay, xy1, xy2, (255, 255, 255),
                          2 if is_selected or i == hovered else 1)
        canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)
        if hovered is not None:
            blob = blobs[hovered]
            contour = np.round(blob['contour'] * scale).astype(np.int32)
            cv2.drawContours(canvas, [contour], -1, (255, 255, 255), 2)

        def line(text, number, color_rgb=(230, 230, 230)):
            cv2.putText(canvas, text, (sw + 12, 27 + number * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, color_rgb, 1, cv2.LINE_AA)

        line(f'{frame_id} / {color.upper()}', 0, BGR[color])
        line(f'Blobs: {len(blobs)} | Selected: {len(selected)}', 1)
        line('Click bbox to select / deselect', 2)
        line('N: save & next color/image', 3)
        line('Q: save & quit', 4)
        line('C: clear selection', 5)
        if hovered is not None:
            blob = blobs[hovered]
            x, y, bw, bh = blob['bbox']
            line(f'Blob #{hovered}', 7, (0, 255, 255))
            line(f'Already in CSV: {hovered in old_ids}', 8)
            for j, key in enumerate(('area', 'perimeter', 'circularity',
                                     'aspect_ratio', 'extent', 'solidity')):
                line(f'{key}: {blob[key]:.3f}', 9 + j)
            line(f'Center: {blob["center"]}', 15)
            line(f'BBox: {x}, {y}, {bw}, {bh}', 16)
        return canvas

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(sw + PANEL_WIDTH, 1600), min(sh, 900))
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            if dirty:
                cv2.imshow(window, render())
                dirty = False
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('n'), ord('q')):
                return sorted(selected), key == ord('q')
            if key == ord('c'):
                selected.clear()
                dirty = True
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return sorted(selected), True
    finally:
        cv2.destroyWindow(window)


def existing_sample_ids():
    ids = set()
    if CSV_PATH.exists():
        with CSV_PATH.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                ids.add((row['frame_id'], row['color'], int(row['blob_id'])))
    return ids


def append_samples(frame, frame_path, color, blobs, indices, detector, seen):
    fresh = [i for i in indices if (frame_path.stem, color, i) not in seen]
    if not fresh:
        print(f'  [{color}] No new samples selected')
        return 0

    # The slow routine operates ONLY on the selected blobs.
    chosen = [blobs[i].copy() for i in fresh]
    detector.add_local_contrast_features(
        frame, chosen, color, min_ring_size=8, debug=True
    )
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rows = []
    for i, blob in zip(fresh, chosen):
        x, y, w, h = blob['bbox']
        cx, cy = blob['center']
        row = {
            'frame_id': frame_path.stem, 'filename': frame_path.name,
            'color': color, 'blob_id': i,
            'center_x': cx, 'center_y': cy,
            'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
        }
        row.update({k: blob[k] for k in (
            'area', 'perimeter', 'circularity', 'aspect_ratio', 'extent', 'solidity',
            'blob_v', 'surrounding_v', 'delta_v',
            'blob_dominance', 'surrounding_dominance', 'delta_dominance', 'ring_size'
        )})
        row.update(contour_hsv_stats(hsv, blob))
        rows.append(row)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with CSV_PATH.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if header:
            writer.writeheader()
        writer.writerows(rows)
    for i in fresh:
        seen.add((frame_path.stem, color, i))
    print(f'  [{color}] Saved {len(rows)} samples -> {CSV_PATH}')
    return len(rows)


def main():
    paths = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() == '.png')
    if not paths:
        print(f'No PNG images in {IMAGE_DIR.resolve()}')
        return
    detector = RGBBlobDetector()
    seen = existing_sample_ids()
    print(f'Found {len(paths)} images; existing samples: {len(seen)}')
    added = 0
    for image_index, path in enumerate(paths, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f'Cannot read {path}, skipping')
            continue
        print(f'\n[{image_index}/{len(paths)}] {path.name} | shape: {frame.shape}')
        masks = detector.segment_colors(frame)
        for color in COLORS:
            cleaned = detector.clean_mask(masks[color])
            blobs = detector.extract_blob_features(cleaned)
            old_ids = {id_ for name, col, id_ in seen if name == path.stem and col == color}
            print(f'  {color}: {len(blobs)} blobs. Select with mouse, N=next, Q=quit')
            chosen, quit_requested = choose_blobs(frame, blobs, color, path.stem, old_ids)
            added += append_samples(frame, path, color, blobs, chosen, detector, seen)
            if quit_requested:
                print(f'Finished. Added {added} samples.')
                return
    print(f'Finished all images. Added {added} samples.')


if __name__ == '__main__':
    main()
