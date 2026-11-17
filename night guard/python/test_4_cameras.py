#!/usr/bin/env python3



#!/usr/bin/env python3

import cv2
import time
import numpy as np
from camera_resolutions import display_resolutions

# Night Guard camera configuration
CAMERA_DEVICES = {
    "FRONT": 0,
    "BACK": 2,
    "LEFT SIDE": 4,
    "RIGHT SIDE": 8,
}

WIDTH = 640
HEIGHT = 480
FPS = 15

cameras = {}
frame_counts = {}
last_frames = {}

def get_fourcc(cap):
    value = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(
        chr((value >> (8 * i)) & 0xFF)
        for i in range(4)
    )

# Open all four cameras
for name, device in CAMERA_DEVICES.items():
    display_resolutions(device, name)
    print(f"Opening {name}: /dev/video{device}", flush=True)

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"{name}: FAILED to open", flush=True)
        cap.release()
        continue

    # Request compressed MJPEG format
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(
        f"{name}: OPENED | "
        f"Format={get_fourcc(cap)} | "
        f"Resolution="
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} | "
        f"FPS={cap.get(cv2.CAP_PROP_FPS):.1f}",
        flush=True
    )

    cameras[name] = cap
    frame_counts[name] = 0
    last_frames[name] = np.zeros(
        (HEIGHT, WIDTH, 3), dtype=np.uint8
    )

if not cameras:
    raise SystemExit("No cameras opened.")

# Create video windows
for name in cameras:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, WIDTH, HEIGHT)

print("\nFour-camera live test started.")
print("Press Q in a video window to quit.")
print("Or Ctrl+C in the terminal.\n")

last_report = time.monotonic()

try:
    while True:
        for name, cap in cameras.items():
            success, frame = cap.read()

            if success:
                frame_counts[name] += 1

                frame = cv2.resize(frame, (WIDTH, HEIGHT))

                cv2.putText(
                    frame,
                    name,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                last_frames[name] = frame

            else:
                # Keep the window visible if a frame fails
                frame = last_frames[name].copy()

                cv2.putText(
                    frame,
                    "FRAME FAILED",
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                last_frames[name] = frame

            cv2.imshow(name, last_frames[name])

        # Report measured FPS every 5 seconds
        now = time.monotonic()

        if now - last_report >= 5:
            elapsed = now - last_report

            print("\n--- Camera FPS ---", flush=True)

            for name, count in frame_counts.items():
                print(
                    f"{name}: {count / elapsed:.1f} FPS",
                    flush=True
                )
                frame_counts[name] = 0

            last_report = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    for cap in cameras.values():
        cap.release()

    cv2.destroyAllWindows()
    print("All cameras released.")
