import cv2
import numpy as np

from frame.FrameSource import VideoSource
from frame.frame_selection import best_frame_in_interval

from analysis.color_segmentation import (
    segment_colors,
    clean_mask,
    extract_blob_features,
    filter_blobs,
    add_local_contrast_features,
    filter_led_blobs,
    blobs_to_mask,
)

COLORS = ("red", "green", "blue")

def detect_candidates(frame):
    """
    Run current colour/blob pipeline.

    Returns:
        {
            "red": {
                "blobs": [...],
                "mask": binary mask
            },
            ...
        }
    """

    masks = segment_colors(frame)

    results = {}

    for color, mask in masks.items():

        cleaned = clean_mask(mask)

        blobs = extract_blob_features(cleaned)

        # Geometry
        blobs = filter_blobs(blobs)

        # Local LED contrast
        blobs = add_local_contrast_features(frame, blobs, color, ring_size=8)

        blobs = filter_led_blobs(blobs, min_delta_v=15, min_delta_dominance=10)

        candidate_mask = blobs_to_mask(cleaned, blobs)

        results[color] = {"blobs": blobs, "mask": candidate_mask}

    return results

class StaticBackgroundModel:

    def __init__(
        self,
        shape,
        frequency_threshold=0.80,
        min_samples=8
    ):
        """
        frequency_threshold:
            Fraction of calibration frames in which a location must
            contain a candidate before being considered static.

        min_samples:
            Don't construct a static mask before this many frames.
        """

        self.frequency_threshold = frequency_threshold
        self.min_samples = min_samples

        self.samples = 0

        self.hits = {color: np.zeros(shape, dtype=np.uint16) for color in COLORS}

    def update(self, candidate_results):
        """
        Add one selected frame to the calibration history.
        """

        for color in COLORS:

            mask = candidate_results[color]["mask"]

            # Only count presence once per frame.
            self.hits[color][mask > 0] += 1

        self.samples += 1

    def get_frequency(self, color):

        if self.samples == 0:
            return np.zeros_like(self.hits[color], dtype=np.float32)

        return (self.hits[color].astype(np.float32) / self.samples)

    def get_static_mask(self, color):

        if self.samples < self.min_samples:
            return np.zeros_like(self.hits[color], dtype=np.uint8)

        frequency = self.get_frequency(color)

        static = (frequency >= self.frequency_threshold).astype(np.uint8) * 255

        return static

def reject_static_background_blobs(blobs, static_mask, frame_shape, max_overlap=0.70):
    """
    Reject a blob only when a large fraction of its area overlaps
    a high-confidence static background region.
    """

    kept = []

    for blob in blobs:

        blob_mask = np.zeros(frame_shape, dtype=np.uint8)

        cv2.drawContours(blob_mask, [blob["contour"]], -1, 255, thickness=cv2.FILLED)

        blob_pixels = blob_mask > 0

        area = np.count_nonzero(blob_pixels)

        if area == 0:
            continue

        overlap = np.count_nonzero(blob_pixels &(static_mask > 0))

        overlap_fraction = overlap / area

        blob["background_overlap"] = overlap_fraction

        if overlap_fraction < max_overlap:
            kept.append(blob)

    return kept



if __name__ == "__main__":

    path = r"C:\Users\light\Downloads\pf\PF_silent_disco\GX010808.MP4"

    video = VideoSource(path)

    INTERVAL_SECONDS = 10
    CALIBRATION_SECONDS = 120

    n_intervals = (CALIBRATION_SECONDS // INTERVAL_SECONDS)

    background_model = None

    print(
        f"Starting background calibration\n"
        f"Intervals: {n_intervals}\n"
        f"Interval size: {INTERVAL_SECONDS}s\n"
    )

    for i in range(n_intervals):

        print(
            f"\n===== Calibration "
            f"{i + 1}/{n_intervals} ====="
        )

        result = best_frame_in_interval(video, duration=INTERVAL_SECONDS)

        if result is None:
            print("End of video")
            break

        frames, score = result

        if not frames:
            print("No selected frame")
            break

        # Temporal sampling disabled -> one frame
        selected = frames[0]

        frame = selected.image

        print(
            f"Selected frame "
            f"{selected.frame_number} | "
            f"{selected.timestamp:.2f}s | "
            f"score={score:.2f}"
        )

        # Initialise once frame dimensions are known
        if background_model is None:

            h, w = frame.shape[:2]

            background_model = StaticBackgroundModel(shape=(h, w), frequency_threshold=0.80, min_samples=8)

        # Current detector
        candidates = detect_candidates(frame)

        # Add this frame to calibration
        background_model.update(candidates)

        for color in COLORS:

            print(
                f"{color:5s}: "
                f"{len(candidates[color]['blobs'])} candidates"
            )

    print(
        f"\nCalibration complete: "
        f"{background_model.samples} frames"
    )


    for color in COLORS:

        frequency = background_model.get_frequency(color)

        static_mask = background_model.get_static_mask(color)

        # Visualise frequency nicely
        frequency_display = np.clip(frequency * 255, 0, 255).astype(np.uint8)

        cv2.namedWindow(f"{color} background frequency",
            cv2.WINDOW_NORMAL
        )

        cv2.imshow(
            f"{color} background frequency",
            frequency_display
        )

        cv2.resizeWindow(
            f"{color} background frequency",
            1280,
            720
        )

        cv2.namedWindow(
            f"{color} static mask",
            cv2.WINDOW_NORMAL
        )

        cv2.imshow(
            f"{color} static mask",
            static_mask
        )

        cv2.resizeWindow(
            f"{color} static mask",
            1280,
            720
        )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    print("\nTesting calibrated background rejection...")

    for test_i in range(6):

        result = best_frame_in_interval(
            video,
            duration=INTERVAL_SECONDS
        )

        if result is None:
            break

        frames, score = result

        if not frames:
            break

        selected = frames[0]
        frame = selected.image

        candidates = detect_candidates(frame)

        for color in COLORS:

            static_mask = (
                background_model.get_static_mask(color)
            )

            blobs = candidates[color]["blobs"]

            filtered = reject_static_background_blobs(
                blobs,
                static_mask,
                frame.shape[:2],
                max_overlap=0.70
            )

            filtered_mask = blobs_to_mask(
                candidates[color]["mask"],
                filtered
            )

            display = cv2.bitwise_and(
                frame,
                frame,
                mask=filtered_mask
            )

            print(
                f"{color}: "
                f"{len(blobs)} -> "
                f"{len(filtered)}"
            )

            cv2.namedWindow(
                f"{color} after background",
                cv2.WINDOW_NORMAL
            )

            cv2.imshow(
                f"{color} after background",
                display
            )

            cv2.resizeWindow(
                f"{color} after background",
                1280,
                720
            )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    video.release()
    cv2.destroyAllWindows()