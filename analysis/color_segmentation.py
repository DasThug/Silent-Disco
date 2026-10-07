import cv2
import numpy as np

def segment_colors(frame, local_size=15):
    """Create adaptive red, green and blue masks using
    HSV hue gating + local contrast + Otsu thresholding.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bgr = frame.astype(np.float32)

    H, S, V = cv2.split(hsv)
    B, G, R = cv2.split(bgr)

    # Broad hue masks - only determines WHICH colour a pixel can be
    red_hue = ((H <= 15) | (H >= 165))
    green_hue = ((H >= 40) & (H <= 75))
    blue_hue = ((H >= 90) & (H <= 135))

    # Colour dominance
    dominance = {
        "red": R - np.maximum(G, B),
        "green": G - np.maximum(R, B),
        "blue": B - np.maximum(R, G)
    }

    hue_masks = {
        "red": red_hue,
        "green": green_hue,
        "blue": blue_hue
    }

    V = V.astype(np.float32)
    local_v = cv2.blur(V, (local_size, local_size))
    brightness_contrast = np.maximum(V - local_v, 0)

    masks = {}

    for color in ("red", "green", "blue"):
        d = dominance[color]

        # Compare colour dominance against local surroundings
        local_d = cv2.blur(d, (local_size, local_size))
        color_contrast = np.maximum(d - local_d, 0)

        # LED-likeness:
        # locally brighter + locally more colour dominant
        score = color_contrast + brightness_contrast

        # Only allow broadly correct hue + reasonable saturation/value
        valid = hue_masks[color] & (S >= 80) & (V >= 70)

        score[~valid] = 0

        # Convert score to 0-255 for OpenCV Otsu
        score_8u = cv2.normalize(
            score, None, 0, 255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Otsu automatically chooses threshold
        _, mask = cv2.threshold(
            score_8u,
            0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Ensure rejected hues cannot reappear
        mask[~valid] = 0

        masks[color] = mask

    return masks

def clean_mask(mask):
    """Light morphology to connect small gaps without destroying LED shapes."""

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

def extract_blob_features(mask):
    # TODO: Can discard computations for unused blob features

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    blobs = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area == 0 or perimeter == 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        circularity = 4 * np.pi * area / (perimeter ** 2)
        aspect_ratio = w / h
        extent = area / (w * h)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        blobs.append({
            "contour": contour,
            "center": (cx, cy),
            "bbox": (x, y, w, h),
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "extent": extent,
            "solidity": solidity
        })

    return blobs

def inspect_blobs(frame, blobs, window_name="Blob Inspector"):
    display = frame.copy()
    hovered = None

    def draw():
        img = frame.copy()

        # Draw all blob bounding boxes
        for i, blob in enumerate(blobs):
            x, y, w, h = blob["bbox"]
            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (150, 150, 150),
                3
            )

        # Highlight hovered blob + display features
        if hovered is not None:
            i, blob = hovered
            x, y, w, h = blob["bbox"]

            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (255, 255, 255),
                6
            )

            cv2.drawContours(
                img,
                [blob["contour"]],
                -1,
                (255, 255, 255),
                4
            )

            lines = [
                f"Blob {i}",
                f"Area:        {blob['area']:.1f}",
                f"Perimeter:   {blob['perimeter']:.1f}",
                f"Circularity: {blob['circularity']:.3f}",
                f"Aspect:      {blob['aspect_ratio']:.3f}",
                f"Extent:      {blob['extent']:.3f}",
                f"Solidity:    {blob['solidity']:.3f}",
                f"Size:        {w} x {h}",
            ]

            for j, text in enumerate(lines):
                cv2.putText(
                    img,
                    text,
                    (30, 60 + j * 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,        # bigger text
                    (255, 255, 255),
                    3,          # thicker text
                    cv2.LINE_AA
                )

        cv2.imshow(window_name, img)

    def mouse(event, mx, my, flags, param):
        nonlocal hovered

        if event != cv2.EVENT_MOUSEMOVE:
            return

        hovered = None

        # Smallest containing bbox wins if boxes overlap
        matches = []

        for i, blob in enumerate(blobs):
            x, y, w, h = blob["bbox"]

            if x <= mx <= x+w and y <= my <= y+h:
                matches.append((w*h, i, blob))

        if matches:
            _, i, blob = min(matches)
            hovered = (i, blob)

        draw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, mouse)

    draw()

    while True:
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cv2.destroyWindow(window_name)

def filter_blobs(blobs):
    return [
        blob for blob in blobs
        if 0.20 <= blob["circularity"] <= 0.70
        and 0.2 <= blob["solidity"] <= 0.95
        and 0.30 <= blob["aspect_ratio"] <= 2.05
        and 0.15 <= blob["extent"] <= 0.68
        and 40 <= blob["area"] <= 2000
        and 45 <= blob["perimeter"] <= 400
    ]

def blobs_to_mask(mask, blobs):
    filtered_mask = np.zeros_like(mask)

    for blob in blobs:
        cv2.drawContours(
            filtered_mask,
            [blob["contour"]],
            -1,
            255,
            thickness=cv2.FILLED
        )

    return filtered_mask

def add_local_contrast_features(frame, blobs, color, ring_size=8):
    """
    Compare each blob against a ring immediately surrounding it.

    Adds:
        blob_v
        surrounding_v
        delta_v

        blob_dominance
        surrounding_dominance
        delta_dominance
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)

    bgr = frame.astype(np.float32)
    B, G, R = cv2.split(bgr)

    dominance = {
        "red": R - np.maximum(G, B),
        "green": G - np.maximum(R, B),
        "blue": B - np.maximum(R, G)
    }[color]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * ring_size + 1, 2 * ring_size + 1)
    )

    for blob in blobs:

        # Mask containing only this blob
        blob_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        cv2.drawContours(
            blob_mask,
            [blob["contour"]],
            -1,
            255,
            thickness=cv2.FILLED
        )

        # Expand blob
        dilated = cv2.dilate(blob_mask, kernel)

        # Ring = expanded region minus blob itself
        ring_mask = cv2.subtract(dilated, blob_mask)

        blob_pixels = blob_mask > 0
        ring_pixels = ring_mask > 0

        if not np.any(ring_pixels):
            blob["delta_v"] = 0
            blob["delta_dominance"] = 0
            continue

        # Brightness
        blob_v = np.mean(V[blob_pixels])
        surrounding_v = np.mean(V[ring_pixels])

        # Colour dominance
        blob_d = np.mean(dominance[blob_pixels])
        surrounding_d = np.mean(dominance[ring_pixels])

        blob["blob_v"] = blob_v
        blob["surrounding_v"] = surrounding_v
        blob["delta_v"] = blob_v - surrounding_v

        blob["blob_dominance"] = blob_d
        blob["surrounding_dominance"] = surrounding_d
        blob["delta_dominance"] = blob_d - surrounding_d

    return blobs

def filter_led_blobs(blobs, min_delta_v=15, min_delta_dominance=10):
    return [
        blob for blob in blobs
        if blob["delta_v"] >= min_delta_v
        and blob["delta_dominance"] >= min_delta_dominance
    ]


if __name__ == "__main__":
    frame = cv2.imread(".cache/frame_20261004_190911_130956.png")

    if frame is None:
        raise FileNotFoundError("Could not load image")

    masks = segment_colors(frame)

    # INSPECTING BLOBS:
    # color = "blue"
    # cleaned = clean_mask(masks[color])
    # blobs = extract_blob_features(cleaned)
    # print(f"{len(blobs)} {color} blobs")
    # inspect_blobs(frame, blobs, window_name=f"{color.upper()} Blob Inspector")
    # cv2.destroyAllWindows()


    for color, mask in masks.items():
        cleaned = clean_mask(mask)
        blobs = extract_blob_features(cleaned)

        # Geometric Feature filtering
        filtered_blobs = filter_blobs(blobs)

        # Measure local LED properties
        blobs = add_local_contrast_features(
            frame,
            filtered_blobs,
            color,
            ring_size=8
        )

        # Local LED filtering
        blobs = filter_led_blobs(
            blobs,
            min_delta_v=15,
            min_delta_dominance=10
        )

        filtered_mask = blobs_to_mask(cleaned, blobs)

        # Apply binary mask to original image
        masked_image = cv2.bitwise_and(
            frame,
            frame,
            mask=filtered_mask
        )

    #     # White binary mask
    #     #cv2.namedWindow(f"{color}_mask", cv2.WINDOW_NORMAL)
    #     #cv2.imshow(f"{color}_mask", cleaned)
    #     #cv2.resizeWindow(f"{color}_mask", 1280, 720)

        # Original image visible only where mask is white
        cv2.namedWindow(f"{color}_image", cv2.WINDOW_NORMAL)
        cv2.imshow(f"{color}_image", masked_image)
        from frame.frame_selection import save_debug_frame
        save_debug_frame(masked_image)
        cv2.resizeWindow(f"{color}_image", 1280, 720)

    cv2.waitKey(0)
    cv2.destroyAllWindows()