import cv2
import numpy as np
import time
import csv

from VideoStream_python.streamInterface import StreamReader
from frame.FrameSource import VideoSource, Frame
from frame.frame_selection import FrameSelector
from analysis.color_segmentation import RGBBlobDetector

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
        for image in crops:
            crop = cv2.resize(image, (crop_size, crop_size))
            cv2.rectangle(crop, (0, 0), (crop_size-1, crop_size-1),
                          colors[color], 2)
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

def classify_headset(crop, color):
    """
    Placeholder for perception model.

    Return:
        True  - headset detected
        False - not a headset
        None  - model not implemented
    """
    return None


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

    path = "/Users/aleks/Downloads/PF_silent_disco/GX010808.MP4"

    stream = StreamReader(path, StreamReader.STREAM_TYPE_FILE)
    video = VideoSource(path=path)
    selector = FrameSelector()
    detector = RGBBlobDetector()

    history = []
    interval_index = 0

    result = selector.best_frame_in_interval(video, duration=30, sample_fps=SAMPLE_FPS)
    if result is None:
        raise RuntimeError("Result is none")

    frame, score = result
    print(f"Selected frame: {frame.frame_number}")
    print(f"Time: {frame.timestamp:.2f}s")
    print(f"Score: {score:.3f}")

    candidates = {}
    masks = detector.segment_colors(frame.image)
    for color, mask in masks.items():
        cleaned = detector.clean_mask(mask)
        blobs = detector.extract_blob_features(cleaned)
        filtered_blobs = detector.filter_blobs(blobs)
        blobs = detector.add_local_contrast_features(frame.image, filtered_blobs, color, debug=True)
        blobs = detector.filter_led_blobs(blobs, delta_v_range = (40, 200), delta_dominance_range = (40, 200))

        # DEBUG:
        filtered_mask = detector.blobs_to_mask(cleaned, blobs)
        masked_image = cv2.bitwise_and(frame.image, frame.image, mask=filtered_mask)
        cv2.namedWindow(f"{color}_image", cv2.WINDOW_NORMAL)
        cv2.imshow(f"{color}_image", masked_image)
        cv2.resizeWindow(f"{color}_image", 1280, 720)
        # -----
        
        crops = detector.get_blob_crops(frame.image, blobs, scale=6)
        candidates[color] = crops

    show_crop_collage(candidates)
    
    

    #if not stream.start():
        #raise RuntimeError("Could not start stream")

    # try:
    #     while True:
    #         print(f"\n--- Interval {interval_index + 1} ---")

    #         result = selector.best_frame_in_interval(
    #             stream,
    #             duration=INTERVAL,
    #             video_fps=VIDEO_FPS,
    #             sample_fps=SAMPLE_FPS,
    #             debug=True
    #         )

    #         if result is None:
    #             print("Stream ended or no valid frame")
    #             break

    #         best_frame, score = result

    #         counts, candidates, unclassified = process_frame(best_frame.image, detector)

    #         timestamp = interval_index * INTERVAL

    #         distribution = {
    #             "interval": interval_index,
    #             "timestamp": timestamp,
    #             **counts
    #         }

    #         history.append(distribution)
    #         save_distribution(history)

    #         print(f"Candidate counts: "
    #               f"R={len(candidates['red'])}, "
    #               f"G={len(candidates['green'])}, "
    #               f"B={len(candidates['blue'])}")

    #         print(f"Unclassified candidates: {unclassified}")
    #         print(f"Confirmed counts: {counts}")

    #         interval_index += 1

    # except KeyboardInterrupt:
    #     print("\nStopping pipeline")

    # finally:
    #     stream.stop()
    #     print(f"Processed {len(history)} intervals")


