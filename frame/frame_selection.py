import cv2
import numpy as np
from frame import FrameSource
from VideoStream_python.streaminterface import StreamReader

import time

from pathlib import Path
from datetime import datetime

def save_debug_frame(frame):
    cache = Path(".cache")
    cache.mkdir(exist_ok=True)

    filename = cache / f"frame_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    cv2.imwrite(str(filename), frame)

    print(f"Saved: {filename}")

class FrameSelector():
    def __init__(self) -> None:
        pass

    def _make_sample_points(self, width, height, n=2000, seed=42):
        rng = np.random.default_rng(seed)
        xs = rng.integers(0, width, n)
        ys = rng.integers(0, height, n)
        return xs, ys

    def frame_score(self, frame, points: tuple[np.ndarray, np.ndarray],
                    sampling="points", # "resize" or "points"
                    resize_shape=(64, 36), return_details=False):
        
        if sampling == "resize":
            pixels = cv2.resize(frame, resize_shape).reshape(-1, 3)
        elif sampling == "points":
            xs, ys = points
            pixels = frame[ys, xs]
        else:
            raise ValueError(f"Unknown sampling mode: {sampling}")

        pixels_3d = pixels.reshape(-1, 1, 3) # BGR Pixels
        hsv = cv2.cvtColor(pixels_3d, cv2.COLOR_BGR2HSV).reshape(-1, 3)

        saturation = hsv[:, 1].astype(np.float32)
        value = hsv[:, 2].astype(np.float32)

        # HARD REJECTION: excessive overexposure
        white_fraction = np.mean((value > 245) & (saturation < 60))
        clipped_fraction = np.mean(np.all(pixels >= 245, axis=1))

        if white_fraction > 0.03 or clipped_fraction > 0.02:
            return float("inf")

        visible_pixels = value > 30 # Keep visible pixels
        mean_sat = np.mean(saturation[visible_pixels]) # Mean Saturation
        mean_value = np.mean(value) / 255 # Mean Illumniation
        colored_fraction = np.mean((saturation > 80) & (value > 50)) # Fraction of meaningful colored light
        bright_colored_fraction = np.mean((saturation > 120) & (value > 120)) # Fraction of bright colored light
        sat_p90 = np.percentile(saturation[visible_pixels], 90) / 255 # Percentile Saturation
        bright_fraction = np.mean(value > 160) # Fraction of strong illumniation


        score = (
            0.20 * mean_sat +
            0.25 * colored_fraction +
            0.30 * bright_colored_fraction +
            0.10 * sat_p90 +
            0.05 * mean_value +
            0.10 * bright_fraction
        )

        return score

    def frame_score_mean_pixel(self, frame, points):
        pixel = cv2.resize(frame, (1, 1), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]

        saturation = hsv[1] / 255
        value = hsv[2] / 255

        score = 0.8 * saturation + 0.2 * value

        return score

    def best_frame_in_interval(self, source: FrameSource.VideoSource, duration=30, sample_fps=5, point_samples=2000, debug=False):
        
        start_time = time.perf_counter()

        video_fps = source.fps

        frame_step = max(1, round(video_fps / sample_fps))
        actual_sample_fps = video_fps / frame_step
        n_samples = round(duration * actual_sample_fps)

        frame = source.read()
        if frame is None:
            return None

        h, w = frame.image.shape[:2]
        if point_samples:
            points = self._make_sample_points(w, h, point_samples)
        else:
            points = self._make_sample_points(1, 1, 1)

        best_frame = frame
        # FRAME SCORE
        best_score = self.frame_score(frame.image, points)
        # -----------
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

            # FRAME SCORE
            score = self.frame_score(frame.image, points)
            # -----------

            if score < best_score:
                best_score = score
                best_frame = frame
                best_frame_number = frame.frame_number

            progress = (i + 1) / n_samples * 100

            if debug:
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


        return best_frame, best_score

    
    def best_frame_in_interval_stream(self, source, duration=30, video_fps=30,
                                    sample_fps=5, point_samples=2000, debug=False):

        start_time = time.perf_counter()

        frame_step = max(1, round(video_fps / sample_fps))
        actual_sample_fps = video_fps / frame_step
        n_samples = max(1, round(duration * actual_sample_fps))

        best_frame = None
        best_score = float("inf")
        best_frame_number = None

        points = None
        processed = 0

        print(
            f"Video FPS: {video_fps:.2f} | "
            f"Sampling at ~{actual_sample_fps:.2f} FPS"
        )

        for i in range(n_samples):
            # Consume intermediate frames without scoring them.
            count = 1 if i == 0 else frame_step
            frames = source.getFrames(count)

            if not frames:
                print("\nEnd of stream reached")
                break

            frame = frames[-1]

            if frame.image is None:
                continue

            if points is None:
                h, w = frame.image.shape[:2]
                points = self._make_sample_points(w, h, point_samples)

            score = self.frame_score(frame.image, points)

            frame_number = i * frame_step

            if score < best_score:
                best_score = score
                best_frame = frame
                best_frame_number = frame_number

            processed += 1

            if debug:
                progress = (i + 1) / n_samples * 100
                print(
                    f"\rProcessing: {i + 1}/{n_samples} samples "
                    f"({progress:.1f}%) | Best score: {best_score:.2f}",
                    end="",
                    flush=True
                )

        elapsed = time.perf_counter() - start_time

        if best_frame is None:
            print("\nNo valid frame found")
            return None

        best_time = best_frame_number / video_fps

        print()
        print(
            f"Best frame: {best_frame_number} "
            f"({best_time:.2f}s) | Score: {best_score:.2f}"
        )
        print(f"Frame selection took {elapsed:.2f} seconds")

        return best_frame, best_score



if __name__ == "__main__":
    path = r"C:\Users\light\Downloads\pf\PF_silent_disco\GX010808.MP4"
    Selector = FrameSelector()
    video = FrameSource.VideoSource(path)

    result = Selector.best_frame_in_interval(video, duration=30, sample_fps=2)

    if result is not None:
        frame, score = result

        print(f"Selected frame: {frame.frame_number}")
        print(f"Time: {frame.timestamp:.2f}s")
        print(f"Score: {score:.3f}")
        print("S = save | Q = quit")

        cv2.namedWindow("Selected Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Selected Frame", 1280, 720)
        cv2.imshow("Selected Frame", frame.image)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord("s"):
                save_debug_frame(frame.image)

            elif key == ord("q"):
                break

    else:
        print("No result")

    video.release()
    cv2.destroyAllWindows()