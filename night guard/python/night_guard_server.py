"""Live USB camera MJPEG server; frames are retained only in memory."""
import argparse
import logging
import platform
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json

import cv2
from dotenv import dotenv_values

POSITIONS = (('front', 'FRONT_CAMERA'), ('back', 'BACK_CAMERA'),
             ('left', 'LEFT_CAMERA'), ('right', 'RIGHT_CAMERA'))


def integer(settings, key, default, low, high):
    """Read a bounded integer, e.g. integer({}, 'FPS', 15, 1, 120)."""
    try:
        value = int(settings.get(key, default))
        if not low <= value <= high:
            raise ValueError()
        return value
    except (TypeError, ValueError):
        raise ValueError(f'{key} must be an integer from {low} to {high}') from None


class Camera:
    def __init__(self, name, device, enabled, settings, stop):
        self.name, self.device, self.enabled = name, device, enabled
        self.settings, self.stop = settings, stop
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.connected = False
        self.thread = None

    def start(self):
        if self.enabled:
            self.thread = threading.Thread(target=self.capture, name=self.name, daemon=True)
            self.thread.start()

    def capture(self):
        """Reconnect independently and replace the single latest JPEG frame."""
        width, height, fps, quality = self.settings
        backend = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_V4L2
        while not self.stop.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self.device, backend)
                if not cap.isOpened():
                    raise RuntimeError('unable to open device; check mapping and permissions')
                for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, width),
                                    (cv2.CAP_PROP_FRAME_HEIGHT, height),
                                    (cv2.CAP_PROP_FPS, fps), (cv2.CAP_PROP_BUFFERSIZE, 1)):
                    cap.set(prop, value)
                while not self.stop.is_set():
                    started = time.monotonic()
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError('camera read failed; reconnecting')
                    ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if not ok:
                        raise RuntimeError('JPEG encoding failed')
                    with self.condition:
                        self.frame = jpeg.tobytes()
                        self.sequence += 1
                        self.connected = True
                        self.condition.notify_all()
                    self.stop.wait(max(0, 1 / fps - (time.monotonic() - started)))
            except Exception as exc:
                logging.warning('%s: %s', self.name, exc)
            finally:
                with self.condition:
                    self.connected = False
                    self.frame = None
                    self.condition.notify_all()
                if cap is not None:
                    cap.release()
            self.stop.wait(2)

    def status(self):
        with self.condition:
            return {'enabled': self.enabled, 'connected': self.connected,
                    'state': 'disabled' if not self.enabled else
                    ('connected' if self.connected else 'offline')}


def handler_for(cameras, stop):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/status':
                body = json.dumps({name: camera.status() for name, camera in cameras.items()}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
                return
            camera = cameras.get(self.path.removeprefix('/stream/')) if self.path.startswith('/stream/') else None
            if camera is None or not camera.enabled:
                self.send_error(404, 'Camera not configured')
                return
            self.connection.settimeout(5)
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            sequence = -1
            try:
                while not stop.is_set():
                    with camera.condition:
                        camera.condition.wait_for(lambda: stop.is_set() or
                                                  (camera.frame is not None and camera.sequence != sequence), 1)
                        if camera.frame is None or camera.sequence == sequence:
                            continue
                        frame, sequence = camera.frame, camera.sequence
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                     str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')
                    self.wfile.flush()
            except (OSError, TimeoutError):
                pass

        def log_message(self, fmt, *args):
            logging.debug(fmt, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', type=int, choices=(0, 1, 2), default=1)
    parser.add_argument('-p', type=int, help='video port override')
    parser.add_argument('-c', type=int, help='reserved control port (no listener)')
    args = parser.parse_args()
    logging.basicConfig(level=(logging.ERROR, logging.INFO, logging.DEBUG)[args.v],
                        format='%(asctime)s %(levelname)s %(message)s')
    settings = dotenv_values(Path(__file__).with_name('server.env'))
    try:
        count = integer(settings, 'CAMERA_COUNT', 4, 1, 4)
        capture = tuple(integer(settings, key, default, low, high) for key, default, low, high in
                        [('WIDTH', 640, 1, 7680), ('HEIGHT', 480, 1, 4320),
                         ('FPS', 15, 1, 120), ('JPEG_QUALITY', 80, 1, 100)])
        port = integer({'PORT': args.p if args.p is not None else settings.get('SERVER_PORT', 14095)}, 'PORT', 14095, 1, 65535)
        control = integer({'CONTROL_PORT': args.c if args.c is not None else settings.get('CONTROL_PORT', 14098)}, 'CONTROL_PORT', 14098, 1, 65535)
        stop = threading.Event()
        cameras = {}
        for index, (name, key) in enumerate(POSITIONS):
            device = str(settings.get(key) or (index if platform.system() == 'Windows' else f'/dev/video{index * 2}'))
            if device.isdecimal():
                device = int(device)
            elif platform.system() == 'Windows' and index < count:
                raise ValueError(f'{key} must be a nonnegative Windows camera index')
            cameras[name] = Camera(name, device, index < count, capture, stop)
        server = ThreadingHTTPServer((settings.get('SERVER_HOST') or '127.0.0.1', port), handler_for(cameras, stop))
        server.daemon_threads = True
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    def shutdown(signum, frame):
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        for camera in cameras.values():
            camera.start()
        logging.info('Video listening on %s:%s; control port %s is reserved and NOT listening', *server.server_address, control)
        server.serve_forever(poll_interval=0.25)
    finally:
        stop.set()
        server.server_close()
        for camera in cameras.values():
            with camera.condition:
                camera.condition.notify_all()
            if camera.thread:
                camera.thread.join(timeout=5)


if __name__ == '__main__':
    main()
