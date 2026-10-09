import sys
import cv2
import time
from VideoStream_python.streamInterface import StreamReader as sr

if __name__ == "__main__":
  args = sys.argv[1:]
  if len(args) >= 1:
    networkPath = args[0]
  else:
    networkPath = input("Enter network path: ") or "rtsp://127.0.0.1:8554/latest"
  print(f"Using network path: {networkPath}")
  stream = sr(networkPath,sr.STREAM_TYPE_NETWORK)
  if not stream.start():
    print(f"Failed to connect to stream: {networkPath}")
    raise SystemExit(1)
  try:
    initialFrames = stream.getFrames(1, True)
    if not initialFrames or initialFrames[0].image is None:
      print(f"Connected but did not receive an initial frame: {networkPath}")
      raise SystemExit(1)
    firstFrame = initialFrames[0]
    info = stream.getStreamInfo()
    fps = info.get("fps", 0.0)
    fpsDisplay = f"{fps:.2f}" if isinstance(fps, (int, float)) and fps > 0 else "unknown"
    widthValue = info.get("width", 0)
    heightValue = info.get("height", 0)
    width = int(widthValue) if isinstance(widthValue, (int, float)) else 0
    height = int(heightValue) if isinstance(heightValue, (int, float)) else 0
    megapixels = width * height / 1_000_000 if width and height else 0.0
    print(
      f"Stream info: {width or 'unknown'}x{height or 'unknown'} "
      f"({megapixels:.2f} MP) at {fpsDisplay} FPS, codec {info.get('codec', 'unknown')}"
    )
    if megapixels and megapixels < 24:
      print("Warning: PC transfer is below ~26 MP. Set Remote Save Image Size to Large in the camera's remote-shooting settings.")
    while True:
      frames = [firstFrame] if firstFrame is not None else stream.getFrames(1, True)
      firstFrame = None
      if frames is None:
        print("no frames retrieved")
        time.sleep(0.005)
        continue
      frame = frames[0].image if frames else None
      if frame is None:
        print("Failed to retrieve frame")
        time.sleep(0.005)
        continue
      cv2.imshow("Stream", frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  finally:
    stream.stop()
    cv2.destroyAllWindows()
