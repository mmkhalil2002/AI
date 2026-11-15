"""Four-panel network camera monitor. No local camera access."""
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import dotenv_values
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget, QGridLayout,
                               QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton)

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
        self.connection_state = 'connecting'
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
                        self.connection_state = 'connected'
                except (requests.RequestException, ValueError, KeyError, TypeError):
                    with self.lock:
                        self.connection = 'Server unavailable — reconnecting'
                        self.connection_state = 'failed'
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
        control_width = positive(settings, 'CONTROL_PANEL_WIDTH', 200)
        self.resize(width, height)
        root = QWidget()
        outer_layout = QHBoxLayout(root)
        self.control_area = QWidget()
        self.control_area.setFixedWidth(control_width)
        self.control_area.setStyleSheet('background: #e5e7eb;')
        outer_layout.addWidget(self.control_area)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(content, 1)
        connection_bar = QHBoxLayout()
        connection_bar.addWidget(QLabel('Server IP / URL:'))
        self.default_url = (settings.get('SERVER_URL') or '').strip() or 'http://127.0.0.1:14095'
        self.server_address = QLineEdit(self.default_url)
        self.server_address.setPlaceholderText('192.168.1.100:14095')
        self.server_address.setAccessibleName('Server IP or URL')
        self.server_address.textChanged.connect(self.fit_address_field)
        self.fit_address_field()
        connection_bar.addWidget(self.server_address)
        self.connect_button = QPushButton('Connect')
        self.connect_button.setMinimumWidth(120)
        self.set_connection_button('idle')
        self.connect_button.clicked.connect(self.toggle_connection)
        self.server_address.returnPressed.connect(self.connect_server)
        connection_bar.addWidget(self.connect_button)
        connection_bar.addStretch(1)
        layout.addLayout(connection_bar)
        self.status = QLabel('')
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
        self.setCentralWidget(root)
        self.network = None
        self.retired_networks = []
        try:
            self.default_port = urlparse(settings.get('SERVER_URL') or '').port or 14095
        except ValueError:
            self.default_port = 14095
        for panel in self.panels.values():
            panel.setText('Not connected')
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(50)

    def fit_address_field(self):
        """Keep the address input compact while fitting its current text."""
        text = self.server_address.text() or self.server_address.placeholderText()
        self.server_address.setFixedWidth(self.server_address.fontMetrics().horizontalAdvance(text) + 28)

    def set_connection_button(self, state):
        text, color = {'idle': ('Connect', '#facc15'), 'connecting': ('Connecting...', '#a16207'),
                       'connected': ('Connected', '#15803d'),
                       'failed': ('Failed', '#b91c1c')}[state]
        self.connect_button.setText(text)
        self.connect_button.setToolTip('Click to disconnect and clear the display' if state == 'connected'
                                       else 'Click to connect to the server')
        foreground = '#111827' if state == 'idle' else 'white'
        self.connect_button.setStyleSheet(
            f'QPushButton {{ background-color: {color}; color: {foreground}; padding: 8px; }}')

    def toggle_connection(self):
        """Click the green Connected button to disconnect; otherwise connect."""
        if self.connect_button.text() == 'Connected':
            self.disconnect_server()
        else:
            self.connect_server()

    def disconnect_server(self):
        """Stop readers and immediately clear the address and displayed frames."""
        if self.network is not None:
            self.network.stop.set()
            self.retired_networks.append(self.network)
            self.network = None
        self.server_address.clear()
        for panel in self.panels.values():
            panel.source = QPixmap()
            panel.clear()
        self.status.setText('Disconnected')
        self.set_connection_button('idle')

    def connect_server(self):
        """Connect asynchronously; a bare IP uses the configured/default video port."""
        # Stop old readers without blocking the GUI while their requests time out.
        if self.network is not None:
            self.network.stop.set()
            self.retired_networks.append(self.network)
            self.network = None
        for panel in self.panels.values():
            panel.source = QPixmap()
            panel.setText('Not connected')
        try:
            address = self.server_address.text().strip() or self.default_url
            explicit_scheme = '://' in address
            parsed = urlparse(address if explicit_scheme else 'http://' + address)
            if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                    or parsed.username or parsed.password or parsed.path not in ('', '/')
                    or parsed.query or parsed.fragment or any(c.isspace() for c in address)):
                raise ValueError('Enter a server IP or HTTP URL, for example 192.168.1.100:14095')
            port = parsed.port
            if port is not None and not 1 <= port <= 65535:
                raise ValueError('Server port must be from 1 to 65535')
            host = f'[{parsed.hostname}]' if ':' in parsed.hostname else parsed.hostname
            if not explicit_scheme and port is None:
                port = self.default_port
            url = f'{parsed.scheme}://{host}' + (f':{port}' if port is not None else '')
        except ValueError as exc:
            self.status.setText(str(exc))
            self.set_connection_button('failed')
            return
        self.server_address.setText(url)
        self.status.setText('Connecting to ' + url)
        self.set_connection_button('connecting')
        self.network = Network(url)

    def refresh(self):
        self.retired_networks = [network for network in self.retired_networks
                                 if any(thread.is_alive() for thread in network.threads)]
        if self.network is None:
            return
        with self.network.lock:
            frames = self.network.frames.copy()
            self.network.frames.clear()
            states = self.network.states.copy()
            connection = self.network.connection
            connection_state = self.network.connection_state
        self.status.setText(connection)
        self.set_connection_button(connection_state)
        for name, panel in self.panels.items():
            if states[name]:
                panel.source = QPixmap()
                panel.setText(states[name])
            elif name in frames:
                panel.show_frame(frames[name])

    def closeEvent(self, event):
        self.timer.stop()
        networks = self.retired_networks + ([self.network] if self.network else [])
        for network in networks:
            network.stop.set()
        for network in networks:
            network.close()
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
