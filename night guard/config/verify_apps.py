"""Hardware-free server integration and offscreen GUI smoke checks.

Run: python config/verify_apps.py
"""
import importlib.util
import os
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import cv2
import numpy as np
import requests
import night_guard_server as server


def eventually(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError('Timed out waiting for expected state')


class Capture:
    online = True
    opened = []

    def __init__(self, device, backend):
        self.opened.append(device)
        self.released = False

    def isOpened(self):
        return self.online

    def set(self, *args):
        return True

    def read(self):
        return self.online, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def main():
    for invalid in ('0', '5', 'abc'):
        try:
            server.integer({'CAMERA_COUNT': invalid}, 'CAMERA_COUNT', 4, 1, 4)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid count accepted')
    with patch.object(cv2, 'VideoCapture', Capture):
        for count in range(1, 5):
            Capture.opened = []
            stop = threading.Event()
            cameras = {name: server.Camera(name, index, index < count,
                       (64, 48, 15, 80), stop) for index, (name, _) in enumerate(server.POSITIONS)}
            for camera in cameras.values():
                camera.start()
            try:
                eventually(lambda: sum(c.status()['connected'] for c in cameras.values()) == count)
                assert sorted(Capture.opened) == list(range(count)), 'Disabled camera opened'
            finally:
                stop.set()
                for camera in cameras.values():
                    if camera.thread:
                        camera.thread.join(3)
                        assert not camera.thread.is_alive()
        stop = threading.Event()
        cameras = {name: server.Camera(name, index, index == 0, (64, 48, 15, 80), stop)
                   for index, (name, _) in enumerate(server.POSITIONS)}
        http = server.ThreadingHTTPServer(('127.0.0.1', 0), server.handler_for(cameras, stop))
        http.daemon_threads = True
        service = threading.Thread(target=http.serve_forever, daemon=True)
        service.start()
        cameras['front'].start()
        url = f'http://127.0.0.1:{http.server_port}'
        window = None
        try:
            eventually(lambda: cameras['front'].status()['connected'])
            status = requests.get(url + '/status', timeout=2).json()
            assert status['front']['state'] == 'connected'
            assert status['back']['state'] == 'disabled'
            assert requests.get(url + '/stream/back', timeout=2).status_code == 404
            assert requests.get(url + '/missing', timeout=2).status_code == 404
            with requests.get(url + '/stream/front', stream=True, timeout=2) as response:
                assert response.status_code == 200
                assert 'multipart' in response.headers['Content-Type']
                data = next(response.iter_content(2048))
                assert b'Content-Type: image/jpeg' in data and b'\xff\xd8' in data
            Capture.online = False
            eventually(lambda: cameras['front'].status()['state'] == 'offline')
            assert cameras['front'].frame is None
            Capture.online = True
            eventually(lambda: cameras['front'].status()['connected'])
            if importlib.util.find_spec('PySide6') is None:
                raise RuntimeError('Server checks passed; install PySide6 for GUI checks')
            from PySide6.QtWidgets import QApplication
            import night_guard_client as client
            app = QApplication.instance() or QApplication([])
            window = client.Window({'SERVER_URL': url})
            window.show()
            assert window.connect_button.text() == 'Connect'
            assert window.network is None
            assert window.status.text() == ''
            assert window.control_area.width() == 200
            assert not window.control_area.children()
            assert window.control_area.geometry().right() < window.server_address.parentWidget().x()
            window.server_address.setText('not a valid address')
            window.connect_button.click()
            assert window.connect_button.text() == 'Failed'
            assert window.network is None
            window.server_address.setText(url.removeprefix('http://'))
            window.connect_button.click()
            def received():
                app.processEvents()
                return not window.panels['front'].source.isNull()
            eventually(received)
            assert window.connect_button.text() == 'Connected'
            assert '#15803d' in window.connect_button.styleSheet()
            assert len(window.panels) == 4
            assert window.panels['back'].text() == 'Not configured'
            previous = window.network
            started = time.monotonic()
            window.connect_button.click()
            assert time.monotonic() - started < 1
            assert previous.stop.is_set()
            assert window.network is None
            assert window.server_address.text() == ''
            assert window.connect_button.text() == 'Connect'
            assert window.status.text() == 'Disconnected'
            # A late frame from retired readers cannot repopulate the display.
            for _ in range(5):
                app.processEvents()
                window.refresh()
            assert all(panel.source.isNull() and not panel.text() for panel in window.panels.values())
            window.connect_button.click()
            assert window.server_address.text() == url
            assert window.network.url == url
            eventually(received)
            assert window.connect_button.text() == 'Connected'
            http.shutdown()
            http.server_close()
            def unavailable():
                app.processEvents()
                return 'unavailable' in window.status.text()
            eventually(unavailable, 8)
            assert window.connect_button.text() == 'Failed'
            assert '#b91c1c' in window.connect_button.styleSheet()
            # A failed connection can be changed and retried without blocking the GUI.
            previous = window.network
            window.server_address.setText('127.0.0.1:1')
            started = time.monotonic()
            window.connect_button.click()
            assert time.monotonic() - started < 1
            assert previous.stop.is_set()
            eventually(lambda: (app.processEvents() or window.connect_button.text() == 'Failed'), 8)
            window.close()
            assert all(not thread.is_alive() for thread in window.network.threads)
            window = None
        finally:
            if window is not None:
                window.close()
            Capture.online = True
            stop.set()
            http.shutdown()
            http.server_close()
            service.join(3)
            cameras['front'].thread.join(3)
    print('PASS: camera counts 1–4, MJPEG/status, reconnect, GUI frames, connection button, disconnect clears display, empty address uses default, retries, worker shutdown')


if __name__ == '__main__':
    main()
