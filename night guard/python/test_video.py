import cv2
import time
from camera_resolutions import display_resolutions

cameras = {
    'FRONT': '/dev/video0',
    'BACK': '/dev/video2',
    'LEFT': '/dev/video4',
    'RIGHT': '/dev/video6',
}

opened = {}
for name, device in cameras.items():
    display_resolutions(device, name)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f'{name}: FAILED to open {device}')
        cap.release()
        continue
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    opened[name] = cap
    print(f'{name}: OPENED {device}')

if not opened:
    raise SystemExit('No cameras could be opened')

print('Press Q to quit.')
try:
    last_report = 0
    while True:
        for name, cap in opened.items():
            ok, frame = cap.read()
            if ok:
                cv2.imshow(name, frame)
            elif time.monotonic() - last_report > 2:
                print(f'{name}: failed to read a frame')
        if time.monotonic() - last_report > 2:
            last_report = time.monotonic()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    for cap in opened.values():
        cap.release()
    cv2.destroyAllWindows()
