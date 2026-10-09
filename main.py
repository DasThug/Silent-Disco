import cv2
import numpy as np
import time
import csv
import sys
from pathlib import Path
import torch

from VideoStream_python.streamInterface import StreamReader
from frame.FrameSource import VideoSource, Frame
from frame.frame_selection import FrameSelector
from analysis.color_segmentation import RGBBlobDetector

from PF_silent_disco.src.silent_disco import resolve_device, load_model, predict_probabilities

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

model_path = Path("PF_silent_disco/models/presence.pt")
presence_model = load_model(model_path, 2, device)
if presence_model is None:
    raise FileNotFoundError(f"Could not load {model_path}")

INTERVAL = 30
VIDEO_FPS = 30
SAMPLE_FPS = 5
CROP_SIZE = 96

def show_crop_collage(candidates, crop_size=96, cols=10):
    tiles = []

    colors = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0)
    }

    for color, crops in candidates.items():
        for image, probability in crops:
            crop = cv2.resize(image, (crop_size, crop_size))

            cv2.rectangle(crop, (0, 0), (crop_size-1, crop_size-1),
                        colors[color], 2)

            cv2.putText(crop, f"{probability:.2f}", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

            tiles.append(crop)

    if not tiles:
        print("No crops found")
        return

    rows = int(np.ceil(len(tiles) / cols))
    collage = np.zeros((rows * crop_size, cols * crop_size, 3),
                       dtype=np.uint8)

    for i, tile in enumerate(tiles):
        y = (i // cols) * crop_size
        x = (i % cols) * crop_size
        collage[y:y+crop_size, x:x+crop_size] = tile

    cv2.namedWindow("Headset Candidates", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Headset Candidates", 1280, 720)
    cv2.imshow("Headset Candidates", collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def classify_headset(crop, color, threshold=0.5):
    """Return True if a headset is detected in the crop."""
    if crop is None or crop.size == 0:
        return False

    crop = cv2.resize(crop, (255, 255))
    probabilities = predict_probabilities(presence_model, [crop], device)

    probability = probabilities[0, 1].item()
    return probability >= threshold

def classify_headsets(crops, threshold=0.5, batch_size=4):
    """ Classify multiple crops (batches) and return presence probabilities. """
    if not crops:
        return np.array([], dtype=np.float32)

    resized = [
        cv2.resize(crop, (255, 255))
        for crop in crops
    ]

    probabilities = predict_probabilities(
        presence_model, resized, device, batch_size=batch_size
    )

    return probabilities[:, 1].numpy()

def save_distribution(history, path="distributions.csv"):
    """Save all interval distributions."""

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "interval", "timestamp",
                "red", "green", "blue"
            ]
        )
        writer.writeheader()
        writer.writerows(history)



if __name__ == "__main__":
    INTERVAL = 30
    VIDEO_FPS = 30
    SAMPLE_FPS = 2
    CROP_SIZE = 96
    CONFIDENCE_THRESHOLD = 0.7
    DEBUG = True

    path = "/Users/aleks/Downloads/PF_silent_disco/GX010808.MP4"

    stream = StreamReader(path, StreamReader.STREAM_TYPE_FILE)
    video = VideoSource(path=path)
    selector = FrameSelector()
    detector = RGBBlobDetector()

    history = []
    interval_index = 0

    result = selector.best_frame_in_interval(video, duration=30, sample_fps=SAMPLE_FPS, debug=False)
    if result is None:
        raise RuntimeError("Result is none")

    frame, score = result
    print(f"Selected frame: {frame.frame_number}")
    print(f"Time: {frame.timestamp:.2f}s")
    print(f"Score: {score:.3f}")


    candidates = {}
    rejected = {}
    all_crops = {}
    counts = {}
    masks = detector.segment_colors(frame.image)

    for color, mask in masks.items():
        print(f"\nProcessing {color}...")

        cleaned = detector.clean_mask(mask)
        blobs = detector.extract_blob_features(cleaned)
        filtered_blobs = detector.filter_blobs(blobs)
        blobs = detector.add_local_contrast_features(frame.image, filtered_blobs, color, debug=False)
        blobs = detector.filter_led_blobs(
            blobs,
            delta_v_range=(40, 200),
            delta_dominance_range=(40, 200)
        )

        crops = detector.get_blob_crops(frame.image, blobs, scale=6)

        # Remove invalid crops before inference
        crops = [crop for crop in crops if crop is not None and crop.size > 0]
        print(f"  Candidate blobs: {len(crops)}")

        # Presence model
        probabilities = classify_headsets(crops, batch_size=32)

        accepted = []
        rejected_crops = []

        for crop, probability in zip(crops, probabilities):
            item = (crop, float(probability))

            if probability >= CONFIDENCE_THRESHOLD:
                accepted.append(item)
            else:
                rejected_crops.append(item)

        all_crops[color] = list(zip(crops, probabilities))
        candidates[color] = accepted
        rejected[color] = rejected_crops
        counts[color] = len(accepted)

        print(f"  Accepted: {len(accepted)}")
        print(f"  Rejected: {len(rejected_crops)}")


    total = sum(counts.values())

    distribution = {
        color: count / total * 100 if total > 0 else 0
        for color, count in counts.items()
    }

    print(f"\nTotal headsets: {total}")
    print("Color distribution:")

    for color in counts:
        print(f"  {color.capitalize()}: {counts[color]} ({distribution[color]:.1f}%)")

    

    

    


