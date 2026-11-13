"""Four-panel network camera monitor. No local camera access."""
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import dotenv_values
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QGridLayout, QVBoxLayout

NAMES = ('front', 'back', 'left', 'right')
TITLES = ('FRONT', 'BACK', 'LEFT SIDE', 'RIGHT SIDE')


class Network:
    """Keep bounded latest-frame mailboxes; GUI polls instead of queuing signals."""
    def __init__(self, url):
        self.url = url
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.frames = {}
        self.states = {name: 'Server unavailable' for name in NAMES}
        self.connection = 'Connecting to server…'
        self.threads = [threading.Thread(target=self.status_loop, daemon=True)]
        self.threads += [threading.Thread(target=self.stream_loop, args=(name,), daemon=True) for name in NAMES]
        for thread in self.threads:
            thread.start()

    def status_loop(self):
        with requests.Session() as session:
            while not self.stop.is_set():
                try:
                    with session.get(self.url + '/status', timeout=(2, 2)) as response:
                        response.raise_for_status()
                        status = response.json()
                    states = {name: ('Not configured' if not status[name]['enabled'] else
                                     ('' if status[name]['connected'] else 'Camera offline')) for name in NAMES}
                    with self.lock:
                        self.states = states
                        self.connection = 'Connected to ' + self.url
                except (requests.RequestException, ValueError, KeyError, TypeError):
                    with self.lock:
                        self.connection = 'Server unavailable — reconnecting'
                        self.states = {name: 'Server unavailable' for name in NAMES}
                self.stop.wait(1)

    def stream_loop(self, name):
        with requests.Session() as session:
            while not self.stop.is_set():
                try:
                    with session.get(self.url + '/stream/' + name, stream=True, timeout=(2, 2)) as response:
                        response.raise_for_status()
                        buffer = b''
                        for chunk in response.iter_content(4096):
                            if self.stop.is_set():
                                break
                            buffer += chunk
                            while True:
                                start = buffer.find(b'\xff\xd8')
                                end = buffer.find(b'\xff\xd9', max(start, 0))
                                if start < 0 or end < 0:
                                    break
                                frame, buffer = buffer[start:end + 2], buffer[end + 2:]
                                with self.lock:
                                    self.frames[name] = frame
                            if len(buffer) > 8 * 1024 * 1024:
                                raise ValueError('Oversized stream frame')
                except (requests.RequestException, ValueError):
                    pass
                with self.lock:
                    self.frames.pop(name, None)
                self.stop.wait(1)

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=5)


class VideoPanel(QLabel):
    def __init__(self):
        super().__init__('Connecting…')
        self.source = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(100, 80)
        self.setStyleSheet('background: #111827; color: #d1d5db; font-size: 18px;')

    def show_frame(self, data):
        image = QImage.fromData(data)
        if not image.isNull():
            self.source = QPixmap.fromImage(image)
            self.fit()

    def fit(self):
        if not self.source.isNull():
            self.setPixmap(self.source.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()


class Window(QMainWindow):
    def __init__(self, settings):
        super().__init__()
        self.setWindowTitle(settings.get('GUI_TITLE') or 'Four USB Camera Monitor')
        width = positive(settings, 'GUI_WIDTH', 1400)
        height = positive(settings, 'GUI_HEIGHT', 900)
        control_height = positive(settings, 'CONTROL_PANEL_HEIGHT', 200)
        self.resize(width, height)
        root = QWidget()
        layout = QVBoxLayout(root)
        self.status = QLabel('Connecting…')
        layout.addWidget(self.status)
        grid = QGridLayout()
        self.panels = {}
        for index, (name, title) in enumerate(zip(NAMES, TITLES)):
            box = QVBoxLayout()
            box.addWidget(QLabel(title))
            panel = VideoPanel()
            box.addWidget(panel, 1)
            grid.addLayout(box, index // 2, index % 2)
            self.panels[name] = panel
        for index in (0, 1):
            grid.setRowStretch(index, 1)
            grid.setColumnStretch(index, 1)
        layout.addLayout(grid, 1)
        reserved = QLabel('RESERVED CONTROL AREA\nControls will be defined later')
        reserved.setMinimumHeight(control_height)
        reserved.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reserved.setStyleSheet('background: #e5e7eb; color: #374151;')
        layout.addWidget(reserved)
        self.setCentralWidget(root)
        url = (settings.get('SERVER_URL') or 'http://127.0.0.1:14095').rstrip('/')
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise ValueError('SERVER_URL must be an http:// or https:// server URL')
        self.network = Network(url)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(50)

    def refresh(self):
        with self.network.lock:
            frames = self.network.frames.copy()
            self.network.frames.clear()
            states = self.network.states.copy()
            connection = self.network.connection
        self.status.setText(connection)
        for name, panel in self.panels.items():
            if states[name]:
                panel.source = QPixmap()
                panel.setText(states[name])
            elif name in frames:
                panel.show_frame(frames[name])

    def closeEvent(self, event):
        self.timer.stop()
        self.network.close()
        event.accept()


def positive(settings, key, default):
    """Validate positive GUI dimensions, e.g. positive({}, 'GUI_WIDTH', 1400)."""
    try:
        value = int(settings.get(key, default))
        if not 1 <= value <= 16384:
            raise ValueError()
        return value
    except (ValueError, TypeError):
        raise ValueError(f'{key} must be an integer from 1 to 16384') from None


def main():
    app = QApplication(sys.argv)
    try:
        window = Window(dotenv_values(Path(__file__).with_name('client.env')))
    except ValueError as exc:
        sys.exit(f'Configuration error: {exc}')
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
