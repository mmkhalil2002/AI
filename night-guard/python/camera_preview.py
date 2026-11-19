"""Live OpenCV previews for the Night Guard camera tests."""

import argparse
import platform
import time

import cv2
import numpy as np

from camera_devices import DirectShowCamera, assign_cameras, camera_source, discover_cameras
from camera_resolutions import display_resolutions


WIDTH, HEIGHT, FPS = 640, 480, 15
POSITIONS = ("FRONT", "BACK", "LEFT SIDE", "RIGHT SIDE")
PREVIEW_SECONDS = 0.5


def show_status(title, message):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.putText(frame, message, (24, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2)
    cv2.imshow(title, frame)


def show_frame(title, frame, label):
    image = frame.copy()
    cv2.rectangle(image, (0, 0), (min(image.shape[1] - 1, 205), 32), (0, 0, 0), -1)
    cv2.putText(image, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.imshow(title, image)


def exit_requested(titles):
    if cv2.waitKey(1) & 0xff in (ord("q"), ord("Q"), 27):
        return True
    try:
        return any(cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
                   for title in titles)
    except cv2.error:
        # Some HighGUI backends remove the window handle as soon as it closes.
        return True


def open_camera(device, title, backend):
    try:
        source = camera_source(device)
    except (OSError, ValueError) as error:
        show_status(title, "Camera not detected")
        print(f"{title}: {error}", flush=True)
        return None
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        cap.release()
        show_status(title, "Could not open camera")
        print(f"{title}: could not open camera", flush=True)
        return None
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((code >> (8 * offset)) & 255) for offset in range(4))
        print(f"{title}: opened, format={fourcc!r}, "
              f"FPS={cap.get(cv2.CAP_PROP_FPS):g}", flush=True)
    except Exception:
        cap.release()
        raise
    return cap


def preview_in_turn(devices, titles, backend):
    for title in titles:
        show_status(title, "Waiting for camera...")
    while True:
        for device, title in zip(devices, titles):
            if exit_requested(titles):
                return
            cap = open_camera(device, title, backend)
            if cap is None:
                time.sleep(0.1)
                continue
            try:
                deadline = time.monotonic() + PREVIEW_SECONDS
                last_frame = None
                while time.monotonic() < deadline:
                    if exit_requested(titles):
                        return
                    ok, frame = cap.read()
                    if not ok:
                        show_status(title, "Could not read frame")
                        last_frame = None
                        break
                    last_frame = frame
                    show_frame(title, frame, "LIVE")
                if last_frame is not None:
                    show_frame(title, last_frame, "LAST FRAME")
            finally:
                cap.release()


def run_preview(count):
    """Open one live preview per requested camera, including offline slots."""
    windows = platform.system() == "Windows"
    backend = cv2.CAP_DSHOW if windows else cv2.CAP_V4L2
    parser = argparse.ArgumentParser(description=f"Preview {count} USB camera(s).")
    parser.add_argument("devices", nargs="*", metavar="DEVICE",
                        help=f"Override all {count} camera devices (default: auto-detect).")
    parser.add_argument("--list-cameras", action="store_true", help="List detected cameras and exit.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--sequential", dest="sequential", action="store_true",
                       help="Open one camera at a time and cycle through the previews.")
    modes.add_argument("--simultaneous", dest="sequential", action="store_false",
                       help="Keep every camera open at the same time.")
    parser.set_defaults(sequential=False)
    args = parser.parse_args()
    if args.devices and len(args.devices) != count:
        parser.error(f"provide exactly {count} camera device(s)")
    try:
        if args.list_cameras:
            for device in discover_cameras():
                print(f"{device} | dshow:{device.identifier}" if isinstance(device, DirectShowCamera)
                      else device)
            return
        devices = assign_cameras(args.devices or ["auto"] * count)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    titles = [f"{position} [{device if device is not None else 'undetected'}]"
              for position, device in zip(POSITIONS, devices)]
    cameras = {}
    offline = set()
    frame_counts = dict.fromkeys(titles, 0)
    print("Press Q or Esc in any preview window to exit.", flush=True)
    try:
        for index, title in enumerate(titles):
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 360, 270)
            cv2.moveWindow(title, 40 + index % 2 * 380, 40 + index // 2 * 310)
            show_status(title, "Opening camera...")

        if args.sequential and count > 1:
            if not windows:
                for title, device in zip(titles, devices):
                    if device is not None:
                        display_resolutions(device, title)
            preview_in_turn(devices, titles, backend)
            return

        for title, device in zip(titles, devices):
            if exit_requested(titles):
                return
            if not windows and device is not None:
                display_resolutions(device, title)
            cap = open_camera(device, title, backend)
            if cap is None:
                offline.add(title)
                continue
            cameras[title] = cap

        last_report = time.monotonic()
        while not exit_requested(titles):
            for title, cap in cameras.items():
                ok, frame = cap.read()
                if ok:
                    cv2.imshow(title, frame)
                    frame_counts[title] += 1
                    offline.discard(title)
                elif title not in offline:
                    show_status(title, "Could not read frame")
                    offline.add(title)
            now = time.monotonic()
            elapsed = now - last_report
            if elapsed >= 5:
                print(" | ".join(f"{title}: {frame_counts[title] / elapsed:.1f} FPS"
                                 for title in titles), flush=True)
                frame_counts = dict.fromkeys(titles, 0)
                last_report = now
            if not cameras:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        for cap in cameras.values():
            cap.release()
        cv2.destroyAllWindows()
