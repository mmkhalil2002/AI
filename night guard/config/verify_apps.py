"""Hardware-free server integration and offscreen GUI smoke checks.

Run: python config/verify_apps.py
"""
import importlib.util
import faulthandler
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
from camera_modes import configure_resolution, Mode


def eventually(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    faulthandler.dump_traceback()
    raise AssertionError('Timed out waiting for expected state')


class Capture:
    online = True
    opened = []
    properties = []

    def __init__(self, device, backend):
        self.opened.append(device)
        self.released = False

    def isOpened(self):
        return self.online

    def set(self, *args):
        self.properties.append(args)
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
            Capture.properties = []
            stop = threading.Event()
            cameras = {name: server.Camera(name, index, index < count,
                       (0, 0, 15, 80), stop) for index, (name, _) in enumerate(server.POSITIONS)}
            with patch('camera_modes.read_modes', return_value={Mode((64, 64, 1), (48, 48, 1), 'MJPG')}):
                assert configure_resolution(cameras, (0, 0, 15, 80)) == (64, 48)
            for camera in cameras.values():
                camera.start()
            try:
                eventually(lambda: sum(c.status()['connected'] for c in cameras.values()) == count)
                assert sorted(Capture.opened) == list(range(count)), 'Disabled camera opened'
                assert (cv2.CAP_PROP_FRAME_WIDTH, 64) in Capture.properties
                assert (cv2.CAP_PROP_FRAME_HEIGHT, 48) in Capture.properties
                decoded = cv2.imdecode(np.frombuffer(cameras['front'].frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                assert decoded.shape[:2] == (48, 64), 'Native frame dimensions changed'
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
            custom = client.Window({'WIDTH': '400', 'HEIGHT': '300'})
            assert all((panel.width(), panel.height()) == (400, 300) for panel in custom.panels.values())
            custom.close()
            for invalid in ('0', '-1', 'abc'):
                try:
                    client.Window({'WIDTH': invalid})
                except ValueError:
                    pass
                else:
                    raise AssertionError('Invalid client panel width accepted')
            window = client.Window({'SERVER_URL': url})
            window.show()
            assert window.connect_button.text() == 'Connect'
            assert window.network is None
            assert window.status.text() == ''
            assert window.control_area.width() == 200
            assert window.zoom_button.text() == 'Zoom Out'
            assert '#facc15' in window.zoom_button.styleSheet()
            assert window.connect_button.parentWidget() is window.zoom_button.parentWidget()
            assert window.zoom_button.parentWidget() is window.split_button.parentWidget()
            assert window.connect_button.x() < window.zoom_button.x() < window.split_button.x()
            assert window.split_button.text() == 'Merge'
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
            assert all((panel.width(), panel.height()) == (480, 320) for panel in window.panels.values())
            assert window.panels['back'].text() == 'Not configured'
            window.zoom_button.click()
            app.processEvents()
            zoom = window.zoom_window
            assert zoom is not None and zoom.isVisible()
            assert len(zoom.panels) == 4
            assert window.zoom_button.text() == 'Zoom In'
            assert not zoom.panels['front'].source.isNull()
            # Split reuses live feeds and retains camera-specific magnification.
            zoom.panels['front'].magnification = 1.25
            zoom.controls['right'].click()
            previous_network = window.network
            window.split_button.click()
            app.processEvents()
            assert not zoom.isVisible()
            assert window.split_button.text() == 'Split'
            assert '#2563eb' in window.split_button.styleSheet()
            assert set(window.split_windows) == set(client.NAMES)
            assert window.network is previous_network
            split_front = window.split_windows['front']
            assert all(view.isVisible() and len(view.panels) == 1 for view in window.split_windows.values())
            assert not split_front.panels['front'].source.isNull()
            window.sync_zoom_views()
            assert split_front.panels['front'].source.cacheKey() == window.panels['front'].source.cacheKey()
            assert window.split_windows['back'].panels['back'].text() == 'Not configured'
            assert not {'left', 'right', 'up', 'down'}.intersection(split_front.controls)
            assert {'left', 'right', 'up', 'down'}.issubset(zoom.controls)
            assert split_front.panels['front'].magnification == 1.25
            split_front.controls['plus'].click()
            assert split_front.panels['front'].magnification == 1.5625
            # Closing one view leaves the rest open; Merge restores all four cameras.
            window.split_windows['back'].close()
            assert len(window.split_windows) == 3
            window.split_button.click()
            assert not window.split_windows
            assert zoom.isVisible()
            assert zoom.stack.currentIndex() == 1
            assert zoom.panels['front'].magnification == 1.5625
            assert window.split_button.text() == 'Merge'
            zoom.change_camera(-1)
            zoom.actual_size()
            assert zoom.panels['back'].text() == 'Not configured'
            assert zoom.stack.currentIndex() == 0
            assert zoom.camera_title.text() == 'FRONT (1/4)'
            assert zoom.stack.currentWidget() is zoom.scroll_areas['front']
            assert zoom.panels['front'].pixmap().size() == zoom.panels['front'].source.size()
            assert window.panels['front'].pixmap().width() <= window.panels['front'].width()
            assert window.panels['front'].pixmap().height() <= window.panels['front'].height()
            zoom.controls['plus'].click()
            assert zoom.panels['front'].magnification == 1.25
            assert zoom.panels['back'].magnification == 1
            for index, name in enumerate(client.NAMES[1:], 1):
                zoom.controls['right'].click()
                assert zoom.stack.currentIndex() == index
                assert zoom.stack.currentWidget() is zoom.scroll_areas[name]
                assert all(not panel.isVisible() for other, panel in zoom.panels.items() if other != name)
            zoom.controls['right'].click()
            assert zoom.stack.currentIndex() == 0
            assert zoom.current_panel().magnification == 1.25
            zoom.controls['left'].click()
            assert zoom.stack.currentIndex() == 3
            zoom.controls['right'].click()
            for _ in range(30):
                zoom.controls['plus'].click()
            assert zoom.panels['front'].magnification == 8
            for _ in range(30):
                zoom.controls['minus'].click()
            assert zoom.panels['front'].magnification == 1
            # Native-size display and magnification enlarge the whole image, with scrollbars.
            from PySide6.QtGui import QPixmap
            window.timer.stop()
            original = zoom.panels['front'].source
            zoom.panels['front'].source = QPixmap(640, 480)
            zoom.panels['front'].source.fill()
            zoom.panels['front'].fit()
            assert zoom.panels['front'].pixmap().size().width() == 640
            assert zoom.panels['front'].pixmap().size().height() == 480
            zoom.panels['front'].change_zoom(2)
            app.processEvents()
            assert zoom.panels['front'].size().width() == 1280
            assert zoom.panels['front'].size().height() == 960
            scroll = zoom.scroll_areas['front']
            assert scroll.horizontalScrollBar().maximum() > 0
            assert scroll.verticalScrollBar().maximum() > 0
            zoom.panels['front'].source = original
            zoom.panels['front'].change_zoom(0.5)
            window.timer.start(50)
            # Fresh frames reach the zoom window without additional network readers.
            sequence = cameras['front'].sequence
            eventually(lambda: cameras['front'].sequence > sequence)
            window.refresh()
            assert not zoom.panels['front'].source.isNull()
            window.split_button.click()
            split_views = list(window.split_windows.values())
            previous = window.network
            started = time.monotonic()
            window.connect_button.click()
            assert time.monotonic() - started < 1
            assert previous.stop.is_set()
            assert window.network is None
            assert window.server_address.text() == ''
            assert window.connect_button.text() == 'Connect'
            assert window.status.text() == 'Disconnected'
            assert all(panel.source.isNull() and not panel.text() for panel in zoom.panels.values())
            assert all(panel.source.isNull() and not panel.text()
                       for view in split_views for panel in view.panels.values())
            window.zoom_button.click()
            assert window.zoom_window is None
            assert not window.split_windows
            assert all(not view.isVisible() for view in split_views)
            assert window.zoom_button.text() == 'Zoom Out'
            window.zoom_button.click()
            window.zoom_window.close()
            assert window.zoom_window is None
            assert '#facc15' in window.zoom_button.styleSheet()
            window.split_button.click()
            assert len(window.split_windows) == 4
            for view in list(window.split_windows.values()):
                view.close()
            assert window.zoom_window is None
            assert window.split_button.text() == 'Merge'
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
            window.split_button.click()
            window.quit_button.click()
            assert not window.split_windows
            assert window.zoom_window is None
            assert not window.isVisible()
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
    print('PASS: streaming, reconnect, connection controls, split/merge views, preserved magnification and selected camera, individual/all-window close, disconnect cleanup, zoom controls, worker shutdown')


if __name__ == '__main__':
    main()
