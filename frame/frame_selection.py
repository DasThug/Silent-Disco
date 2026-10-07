import cv2
import numpy as np
from frame import FrameSource
import time

from pathlib import Path
from datetime import datetime

def save_debug_frame(frame):
    cache = Path(".cache")
    cache.mkdir(exist_ok=True)

    filename = cache / f"frame_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    cv2.imwrite(str(filename), frame)

    print(f"Saved: {filename}")

def make_sample_points(width, height, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, width, n)
    ys = rng.integers(0, height, n)
    return xs, ys

def frame_score(frame,points=None,
                sampling="points", # "resize" or "points"
                resize_shape=(64, 36), weights=None, return_details=False
):
    
    if weights is None:
        weights = {
            "low": 0.15,
            "mean": 0.20,
            "bright": 0.10,
            "color": 0.20,
            "stage_light": 0.35
        }

    if sampling == "resize":
        pixels = cv2.resize(frame, resize_shape).reshape(-1, 3)
    elif sampling == "points":
        xs, ys = points
        pixels = frame[ys, xs]
    else:
        raise ValueError(f"Unknown sampling mode: {sampling}")

    pixels_3d = pixels.reshape(-1, 1, 3)
    gray = cv2.cvtColor(pixels_3d, cv2.COLOR_BGR2GRAY).ravel()
    hsv = cv2.cvtColor(pixels_3d, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    pixels_f = pixels.astype(np.float32)
    pixels_f = pixels_f[pixels_f[:, 1] > 10]
    pixels_f = pixels_f[pixels_f[:, 1] < 250]

    low = np.percentile(gray, 10)
    mean = np.mean(gray)
    bright = np.mean(gray > 240) * 255
    color = np.mean(pixels_f[:, 1])

    # Fraction of pixels strongly affected by coloured stage lighting
    s = hsv[:, 1]
    v = hsv[:, 2]
    stage_light = np.mean((s > 150) & (v > 180)) * 255

    score = (
        weights["low"] * low +
        weights["mean"] * mean +
        weights["bright"] * bright +
        weights["color"] * color +
        weights["stage_light"] * stage_light
    )

    if return_details:
        return score, {"low": low, "mean": mean, "bright": bright, "color": color, "stage_light": stage_light}

    return score


def get_temporal_frames(source, center_frame, temporal_step=None, temporal_samples=None):

    # Temporal sampling disabled
    if temporal_step is None and temporal_samples is None:
        return [center_frame]

    # Don't allow only one parameter
    if temporal_step is None or temporal_samples is None:
        raise ValueError(
            "temporal_step and temporal_samples must either "
            "both be None or both be provided"
        )

    if temporal_step <= 0:
        raise ValueError("temporal_step must be > 0")

    if temporal_samples < 0:
        raise ValueError("temporal_samples must be >= 0")

    step_frames = max(1, round(temporal_step * source.fps))

    frames = []
    for offset in range(-temporal_samples, temporal_samples + 1):
        # Main selected frame
        if offset == 0:
            frames.append(center_frame)
            continue

        frame_number = (center_frame.frame_number + offset * step_frames)

        # Outside video bounds
        if frame_number < 0:
            continue

        if (hasattr(source, "frame_count") and frame_number >= source.frame_count):
            continue

        frame = source.seek(frame_number)

        if frame is not None:
            frames.append(frame)

    # Sort once, after all frames have been collected
    frames.sort(key=lambda f: f.frame_number)

    return frames


def best_frame_in_interval(source: FrameSource.Frame, duration=30, sample_fps=5, point_samples=2000,
                           temporal_step=None, temporal_samples=None):
    
    start_time = time.perf_counter()

    video_fps = source.fps

    frame_step = max(1, round(video_fps / sample_fps))
    actual_sample_fps = video_fps / frame_step
    n_samples = round(duration * actual_sample_fps)

    frame = source.read()
    if frame is None:
        return None

    h, w = frame.image.shape[:2]
    points = make_sample_points(w, h, point_samples)

    best_frame = frame
    best_score = frame_score(frame.image, points)
    best_frame_number = frame.frame_number

    print(
        f"Video FPS: {video_fps:.2f} | "
        f"Sampling at ~{actual_sample_fps:.2f} FPS"
    )

    for i in range(1, n_samples):
        if not source.skip(frame_step - 1):
            print("\nEnd of video reached")
            break

        frame = source.read()
        if frame is None:
            print("\nEnd of video reached")
            break

        score = frame_score(frame.image, points)

        if score < best_score:
            best_score = score
            best_frame = frame
            best_frame_number = frame.frame_number

        progress = (i + 1) / n_samples * 100

        print(
            f"\rProcessing: {i + 1}/{n_samples} samples "
            f"({progress:.1f}%) | Best score: {best_score:.2f}",
            end=""
        )

    best_time = best_frame_number / video_fps
    elapsed = time.perf_counter() - start_time

    print()
    print(
        f"Best frame: {best_frame_number} "
        f"({best_time:.2f}s) | Score: {best_score:.2f}"
    )
    print(f"Frame selection took {elapsed:.2f} seconds")

    temporal_frames = get_temporal_frames(
        source,
        best_frame,
        temporal_step,
        temporal_samples
    )

    return temporal_frames, best_score




if __name__ == "__main__":
    path = "/Users/aleks/Downloads/PF_silent_disco/GX010808.MP4"
    video = FrameSource.VideoSource(path)

    frames, score = best_frame_in_interval(
        video,
        duration=10,
        sample_fps=5,
        point_samples=2000,
        temporal_step=None,
        temporal_samples=None
    )

    if frames is None:
        print("No result")

    else:
        # Main frame is the middle frame
        main_frame = frames[len(frames) // 2]

        print(f"\nSelected frame score: {score:.2f}")
        print(f"Total frames: {len(frames)}")
        print("A/D = previous/next | S = save | Q = quit\n")

        i = 0
        while True:
            frame = frames[i]

            # Time relative to selected frame
            offset = frame.timestamp - main_frame.timestamp

            if abs(offset) < 1e-6:
                label = "MAIN (t)"
            elif offset < 0:
                label = f"t{offset:.2f}s"
            else:
                label = f"t+{offset:.2f}s"

            print(
                f"\r[{i + 1}/{len(frames)}] "
                f"{label} | "
                f"Frame {frame.frame_number} | "
                f"Video time {frame.timestamp:.2f}s",
                end=""
            )

            display = frame.image.copy()

            # Put temporal position directly on image
            cv2.putText(
                display,
                label,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

            cv2.namedWindow("Temporal Frames", cv2.WINDOW_NORMAL)
            cv2.imshow("Temporal Frames", display)
            cv2.resizeWindow("Temporal Frames", 1280, 720)

            key = cv2.waitKey(0) & 0xFF

            if key == ord("d"):
                i = min(i + 1, len(frames) - 1)

            elif key == ord("a"):
                i = max(i - 1, 0)

            elif key == ord("s"):
                save_debug_frame(frame.image)

            elif key == ord("q"):
                break

    video.release()
    cv2.destroyAllWindows()