import cv2
from dataclasses import dataclass
import numpy as np

@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    frame_number: int

class ImageSource:
    def __init__(self, path):
        self.path = path

    def read(self):
        image = cv2.imread(self.path)
        if image is None:
            return None
        
        return Frame(image=image, timestamp=0.0, frame_number=0)

class VideoSource:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.fps <= 0:
            raise ValueError("Could not determine video FPS")

    def read(self):
        success, frame = self.cap.read()
        if not success:
            return None

        frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        timestamp = frame_number / self.fps
        
        return Frame(image=frame, timestamp=timestamp, frame_number=frame_number)

    def seek(self, frame_number):
        """ Seek to a frame and return it. """
        frame_number = int(frame_number)
        if frame_number < 0 or frame_number >= self.frame_count:
            return None
        
        success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        if not success:
            return None

        return self.read()

    def skip(self, n):
        """ Skip n frames without returning them. """
        for _ in range(n):
            if not self.cap.grab():
                return False
        return True

    def release(self):
        self.cap.release()


