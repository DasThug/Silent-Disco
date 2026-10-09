"""Local RGB blob annotation server. No modifications to RGBBlobDetector required."""

# NOTE: Run like: python blob_web_sampler/server.py --images .samples --csv blob_samples.csv
import argparse
import csv
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

# Run from your project root so the existing module resolves.
from analysis.color_segmentation import RGBBlobDetector

COLORS = ("red", "green", "blue")
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
GEO = ('area', 'perimeter', 'circularity', 'aspect_ratio', 'extent', 'solidity')
DELTA = ('blob_v', 'surrounding_v', 'delta_v', 'blob_dominance',
         'surrounding_dominance', 'delta_dominance', 'ring_size')
ROOT = Path(__file__).resolve().parent


def hsv_within_contour(hsv, blob):
    x, y, w, h = blob['bbox']
    roi = hsv[y:y+h, x:x+w]
    contour = blob['contour'].copy()
    contour[:, :, 0] -= x
    contour[:, :, 1] -= y
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    pixels = roi[mask != 0]
    data = {}
    for i, channel in enumerate(('hue', 'saturation', 'value')):
        v = pixels[:, i]
        data.update({f'{channel}_min': int(v.min()),
                     f'{channel}_mean': float(v.mean()),
                     f'{channel}_max': int(v.max())})
    return data


class Sampler:
    def __init__(self, folder, csv_path):
        self.paths = sorted(p for p in folder.iterdir() if p.suffix.lower() == '.png')
        self.csv_path = csv_path
        self.detector = RGBBlobDetector()
        self.lock = threading.RLock()
        self.cached_index = None
        self.frame = None
        self.blobs_by_color = {}
        self.existing = set()
        if csv_path.exists():
            with csv_path.open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != FIELDS:
                    raise ValueError(f'CSV header mismatch in {csv_path}. Use a new output CSV.')
                for row in reader:
                    self.existing.add((row['frame_id'], row['color'], int(row['blob_id'])))

    def load(self, index):
        if not 0 <= index < len(self.paths):
            raise ValueError('Image index out of range')
        if self.cached_index == index:
            return
        path = self.paths[index]
        print(f'Loading {path.name} and segmenting RGB...', flush=True)
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Unable to decode {path.name}')
        masks = self.detector.segment_colors(frame)
        blobs = {}
        for color in COLORS:
            cleaned = self.detector.clean_mask(masks[color])
            blobs[color] = self.detector.extract_blob_features(cleaned)
            print(f'  {color}: {len(blobs[color])} blobs', flush=True)
        self.frame, self.blobs_by_color, self.cached_index = frame, blobs, index

    def metadata(self, index, color):
        self.load(index)
        if color not in COLORS:
            raise ValueError('Invalid color')
        path = self.paths[index]
        blobs = self.blobs_by_color[color]
        return {
            'filename': path.name, 'frame_id': path.stem, 'index': index,
            'color': color, 'width': self.frame.shape[1], 'height': self.frame.shape[0],
            'blobs': [{
                'id': i, 'bbox': [int(v) for v in b['bbox']],
                'center': [int(v) for v in b['center']],
                **{key: float(b[key]) for key in GEO}
            } for i, b in enumerate(blobs)],
            'saved': [i for i in range(len(blobs))
                      if (path.stem, color, i) in self.existing]
        }

    def save(self, index, color, ids):
        self.load(index)
        if color not in COLORS or not isinstance(ids, list):
            raise ValueError('Invalid selection')
        blobs = self.blobs_by_color[color]
        unique = sorted({int(i) for i in ids})
        if any(i < 0 or i >= len(blobs) for i in unique):
            raise ValueError('Invalid blob ID')
        path = self.paths[index]
        new = [i for i in unique if (path.stem, color, i) not in self.existing]
        if not new:
            return {'added': 0, 'total_saved': len(self.existing)}
        chosen = [blobs[i].copy() for i in new]
        # This expensive method is called ONLY AFTER manual selection.
        print(f'Contrast features: {path.name} / {color} / {len(chosen)} selected', flush=True)
        self.detector.add_local_contrast_features(
            self.frame, chosen, color, min_ring_size=8, debug=True
        )
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        rows = []
        for i, blob in zip(new, chosen):
            x, y, w, h = blob['bbox']
            cx, cy = blob['center']
            row = {
                'frame_id': path.stem, 'filename': path.name, 'color': color, 'blob_id': i,
                'center_x': cx, 'center_y': cy, 'bbox_x': x, 'bbox_y': y,
                'bbox_w': w, 'bbox_h': h,
                **{key: blob[key] for key in GEO + DELTA},
                **hsv_within_contour(hsv, blob)
            }
            rows.append(row)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if header:
                writer.writeheader()
            writer.writerows(rows)
        for i in new:
            self.existing.add((path.stem, color, i))
        print(f'Saved {len(new)} samples to {self.csv_path}', flush=True)
        return {'added': len(new), 'total_saved': len(self.existing)}


def create_handler(sampler):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, payload, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, obj, status=200):
            self.send(status, json.dumps(obj).encode())

        def do_GET(self):
            parsed = urlsplit(self.path)
            q = parse_qs(parsed.query)
            try:
                if parsed.path == '/':
                    self.send(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
                elif parsed.path == '/api/frames':
                    self.send_json({'frames': [p.name for p in sampler.paths],
                                    'total_saved': len(sampler.existing)})
                elif parsed.path == '/api/frame':
                    with sampler.lock:
                        self.send_json(sampler.metadata(int(q['index'][0]), q['color'][0]))
                elif parsed.path == '/image':
                    index = int(q['index'][0])
                    if not 0 <= index < len(sampler.paths):
                        raise ValueError('Invalid index')
                    self.send(200, sampler.paths[index].read_bytes(), 'image/png')
                else:
                    self.send_json({'error': 'Not found'}, 404)
            except (ValueError, KeyError, IndexError) as exc:
                self.send_json({'error': str(exc)}, 400)
            except Exception as exc:
                print('GET error:', repr(exc), flush=True)
                self.send_json({'error': str(exc)}, 500)

        def do_POST(self):
            if urlsplit(self.path).path != '/api/save':
                return self.send_json({'error': 'Not found'}, 404)
            try:
                length = int(self.headers.get('Content-Length', 0))
                if length > 1_000_000:
                    return self.send_json({'error': 'Request too large'}, 413)
                data = json.loads(self.rfile.read(length))
                with sampler.lock:
                    self.send_json(sampler.save(int(data['index']), data['color'], data['ids']))
            except (KeyError, ValueError, TypeError) as exc:
                self.send_json({'error': str(exc)}, 400)
            except Exception as exc:
                print('POST error:', repr(exc), flush=True)
                self.send_json({'error': str(exc)}, 500)

    return Handler


def main():
    parser = argparse.ArgumentParser(description='Full-resolution browser blob sampler')
    parser.add_argument('--images', type=Path, default=Path('.samples'))
    parser.add_argument('--csv', type=Path, default=Path('blob_samples.csv'))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not args.images.is_dir():
        parser.error(f'Image folder not found: {args.images}')
    sampler = Sampler(args.images, args.csv)
    if not sampler.paths:
        parser.error(f'No PNGs in {args.images}')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), create_handler(sampler))
    url = f'http://127.0.0.1:{args.port}'
    print(f'{len(sampler.paths)} frames | CSV: {args.csv} | Open {url}', flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
