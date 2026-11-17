#!/usr/bin/env python3


#!/usr/bin/env python3

import cv2
import time
from camera_resolutions import display_resolutions

# Camera configuration
CAMERA_DEVICES = {
    "Camera 1": 0,
    "Camera 2": 2,
    "Camera 3": 4,
}

WIDTH = 640
HEIGHT = 480
FPS = 15

cameras = {}

# Open all three cameras
for name, device in CAMERA_DEVICES.items():
    display_resolutions(device, name)
    print(f"Opening {name}: /dev/video{device}", flush=True)

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"{name}: FAILED to open", flush=True)
        continue

    # Use MJPEG to reduce USB bandwidth
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    actual_format = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(
        chr((actual_format >> (8 * i)) & 0xFF)
        for i in range(4)
    )

    print(
        f"{name}: OPENED, "
        f"Format={fourcc}, "
        f"Resolution={int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
        flush=True
    )

    cameras[name] = cap

if not cameras:
    raise SystemExit("No cameras opened.")

print("\nLive video started.")
print("Press Q in a video window to quit.")
print("Or press Ctrl+C in the terminal.\n")

last_report = time.monotonic()
frame_counts = {name: 0 for name in cameras}

try:
    while True:
        for name, cap in cameras.items():
            success, frame = cap.read()

            if success:
                frame_counts[name] += 1

                cv2.putText(
                    frame,
                    name,
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(name, frame)

            else:
                print(f"{name}: FAILED to read frame", flush=True)

        # Report frame counts every 5 seconds
        now = time.monotonic()
        if now - last_report >= 5:
            elapsed = now - last_report
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
