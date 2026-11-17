#!/usr/bin/env python3
"""List capture modes and preview one Linux USB camera. Press Q to exit."""
import argparse
import cv2
from camera_resolutions import display_resolutions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('device', nargs='?', default='/dev/video0',
                        help='V4L2 capture device (default: /dev/video0)')
    args = parser.parse_args()
    device = int(args.device) if args.device.isdecimal() else args.device
    display_resolutions(device, 'Camera 1')
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise SystemExit(f'Cannot open camera {device}')
        print('Preview uses camera default resolution. Press Q to quit.', flush=True)
        reported = False
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Camera frame read failed.', flush=True)
                break
            if not reported:
                print(f'Actual frame: {frame.shape[1]}x{frame.shape[0]}', flush=True)
                reported = True
            cv2.imshow('Camera 1', frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
