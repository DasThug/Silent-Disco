import cv2
from ultralytics import YOLO
from frame.frame_selection import best_frame_in_interval
from frame.FrameSource import VideoSource

if __name__ == "__main__":
    path = "/Users/aleks/Downloads/PF_silent_disco/GX010808.MP4"
    video = VideoSource(path)

    frames, score = best_frame_in_interval(
        video,
        duration=5,
        sample_fps=5,
        point_samples=2000,
        temporal_step=None,
        temporal_samples=None
    )
    if frames is None:
        print("No result")
    frame = frames[0].image
    print(frame)

    model = YOLO("yolo11n.pt")
    results = model.predict(
        frame,
        conf=0.25,
        classes=[0],      # COCO class 0 = person
        verbose=False
    )

    display = frame.copy()
    people = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0])

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        people.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": confidence
        })

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"person {confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


    print(f"Detected {len(people)} people")

    for i, person in enumerate(people):
        print(
            f"{i}: "
            f"bbox={person['bbox']} | "
            f"confidence={person['confidence']:.3f}"
        )


    cv2.namedWindow(
        "YOLO People",
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        "YOLO People",
        display
    )

    cv2.resizeWindow(
        "YOLO People",
        1280,
        720
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    




