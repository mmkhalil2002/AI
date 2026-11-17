#!/usr/bin/env python3

import cv2
import time
from camera_resolutions import display_resolutions

cameras = {}

for device in (0, 2):
    display_resolutions(device)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print(f"Camera {device}: FAILED")
        continue

    cameras[device] = cap
    print(f"Camera {device}: OPENED")

if not cameras:
    raise SystemExit("No cameras available")

print("Live video started. Press Q to quit.")

try:
    while True:
        for device, cap in cameras.items():
            success, frame = cap.read()

            if success:
                cv2.imshow(f"Camera {device}", frame)
            else:
                print(f"Camera {device}: Frame failed")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.01)

except KeyboardInterrupt:
    pass

finally:
    for cap in cameras.values():
        cap.release()

    cv2.destroyAllWindows()
