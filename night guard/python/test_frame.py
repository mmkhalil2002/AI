python3 - <<'PY'
import cv2

cameras = {}

for device in (0, 2):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cameras[device] = cap

for device, cap in cameras.items():
    success, frame = cap.read()
    print(f"/dev/video{device}: Opened={cap.isOpened()}, Frame={success}")
    cap.release()
PY