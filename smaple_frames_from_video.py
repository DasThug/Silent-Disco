from frame.frame_selection import FrameSelector
from frame.FrameSource import VideoSource

import cv2
import numpy as np
from pathlib import Path


path = r"C:\Users\light\Downloads\pf\PF_silent_disco\GX010808.MP4"

selector = FrameSelector()
video = VideoSource(path)

output = Path(".samples")
output.mkdir(parents=True, exist_ok=True)

duration = 30
sample_fps = 2

interval = 0
try:
    while True:
        print(f"\n--- Interval {interval + 1} ---")

        result = selector.best_frame_in_interval(
            video,
            duration=duration,
            sample_fps=sample_fps
        )

        if result is None:
            break

        frame, score = result

        filename = output / f"frame_{frame.frame_number:08d}_{frame.timestamp:.2f}s.png"
        cv2.imwrite(str(filename), frame.image)

        print(f"Saved: {filename}")

        # Advance to the next 30-second interval
        frame_step = max(1, round(video.fps / sample_fps))
        n_samples = round(duration * video.fps / frame_step)
        remaining = max(0, round(duration * video.fps) - ((n_samples - 1) * frame_step + 1))

        if not video.skip(remaining):
            break

        interval += 1

finally:
    video.release()

print(f"\nFinished! Saved {interval + 1} frames.")