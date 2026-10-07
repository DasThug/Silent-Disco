import cv2

path = ".cache/frame_20261004_190911_130956.png"

image = cv2.imread(path)
if image is None:
    raise FileNotFoundError(path)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

zoom = 1.0
offset_x = 0
offset_y = 0
mouse_x = 0
mouse_y = 0


def show():
    h, w = image.shape[:2]

    view_w = max(1, int(w / zoom))
    view_h = max(1, int(h / zoom))

    x1 = max(0, min(offset_x, w - view_w))
    y1 = max(0, min(offset_y, h - view_h))

    crop = image[y1:y1 + view_h, x1:x1 + view_w]
    display = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)

    # Convert window coordinates back to original image coordinates
    ix = min(w - 1, x1 + int(mouse_x / zoom))
    iy = min(h - 1, y1 + int(mouse_y / zoom))

    H, S, V = hsv[iy, ix]

    cv2.putText(
        display,
        f"HSV ({H}; {S}; {V})   XY ({ix}; {iy})   Zoom {zoom:.1f}x",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        2, (255, 255, 255), 2
    )

    cv2.imshow("HSV Inspector", display)


def mouse(event, x, y, flags, param):
    global zoom, offset_x, offset_y, mouse_x, mouse_y

    mouse_x, mouse_y = x, y

    h, w = image.shape[:2]

    if event == cv2.EVENT_MOUSEWHEEL:
        # Original image coordinate currently under cursor
        old_ix = offset_x + x / zoom
        old_iy = offset_y + y / zoom

        if flags > 0:
            zoom = min(zoom * 1.25, 20.0)
        else:
            zoom = max(zoom / 1.25, 1.0)

        # Keep same image pixel underneath cursor
        offset_x = int(old_ix - x / zoom)
        offset_y = int(old_iy - y / zoom)

        view_w = int(w / zoom)
        view_h = int(h / zoom)

        offset_x = max(0, min(offset_x, w - view_w))
        offset_y = max(0, min(offset_y, h - view_h))

    show()


cv2.namedWindow("HSV Inspector", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("HSV Inspector", mouse)

show()

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):
        zoom = 1.0
        offset_x = 0
        offset_y = 0
        show()

cv2.destroyAllWindows()