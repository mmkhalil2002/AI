"""Network camera monitor. No local camera access."""
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import dotenv_values
from PySide6.QtCore import Qt, QTimer, QSize, QByteArray
from PySide6.QtGui import QImage, QPixmap, QShortcut, QKeySequence, QIcon, QPainter, QPen, QColor
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget, QGridLayout,
                               QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QDialog,
                               QStackedWidget, QScrollArea, QSizePolicy)

NAMES = ('front', 'back', 'left', 'right')
TITLES = ('FRONT', 'BACK', 'LEFT SIDE', 'RIGHT SIDE')


class Network:
    """Keep bounded latest-frame mailboxes; GUI polls instead of queuing signals."""
    def __init__(self, url, camera_names):
        self.url = url
        self.camera_names = camera_names
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.frames = {}
        self.states = {name: 'Server unavailable' for name in camera_names}
        self.connection = 'Connecting to server…'
        self.connection_state = 'connecting'
        self.threads = [threading.Thread(target=self.status_loop, daemon=True)]
        self.threads += [threading.Thread(target=self.stream_loop, args=(name,), daemon=True) for name in camera_names]
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
                                     ('' if status[name]['connected'] else 'Camera offline')) for name in self.camera_names}
                    with self.lock:
                        self.states = states
                        self.connection = 'Connected to ' + self.url
                        self.connection_state = 'connected'
                except (requests.RequestException, ValueError, KeyError, TypeError):
                    with self.lock:
                        self.connection = 'Server unavailable — reconnecting'
                        self.connection_state = 'failed'
                        self.states = {name: 'Server unavailable' for name in self.camera_names}
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
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
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


class ZoomPanel(VideoPanel):
    """Display native image pixels at 1x, enlarging the full image up to 8x."""
    def __init__(self):
        super().__init__()
        self.magnification = 1.0

    def change_zoom(self, multiplier):
        self.magnification = min(8.0, max(1.0, self.magnification * multiplier))
        self.fit()

    def fit(self):
        if self.source.isNull():
            if self.size() != QSize(100, 80):
                self.setFixedSize(100, 80)
            return
        target_size = QSize(max(1, round(self.source.width() * self.magnification)),
                            max(1, round(self.source.height() * self.magnification)))
        if self.size() != target_size:
            self.setFixedSize(target_size)
        self.setPixmap(self.source if self.magnification == 1 else self.source.scaled(
            target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))



# White line-art on rounded black buttons, matching the supplied toolbar reference.
# SVGs are embedded: no external icon files or internet connection are needed.
TOOLBAR_PATHS = {
    'left': '<path d="M19 12H5m7-7-7 7 7 7"/>',
    'right': '<path d="M5 12h14m-7-7 7 7-7 7"/>',
    'up': '<path d="M12 19V5m-7 7 7-7 7 7"/>',
    'down': '<path d="M12 5v14m-7-7 7 7 7-7"/>',
    'fit': '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="9" r="1.5"/><path d="m5 17 5-5 3 3 3-4 3 6"/>',
    'actual': '<rect x="2" y="4" width="17" height="15" rx="2"/><path d="m5 16 4-4 3 3 2-2"/><circle cx="14" cy="9" r="1"/><circle cx="18" cy="18" r="4"/><path d="m21 21 2 2"/>',
    'plus': '<circle cx="10.5" cy="10.5" r="7"/><path d="M10.5 7v7m-3.5-3.5h7M16 16l5 5"/>',
    'minus': '<circle cx="10.5" cy="10.5" r="7"/><path d="M7 10.5h7M16 16l5 5"/>',
    'save': '<path d="M5 3h12l4 4v14H3V3h2zm2 0v7h10V3M7 21v-8h10v8"/>',
    'copy': '<rect x="7" y="3" width="14" height="18" rx="2"/><path d="M10 3V2h8v1M3 7v13h4"/>',
    'print': '<path d="M6 9V3h12v6M6 18H3v-8h18v8h-3M6 15h12v7H6zM17 12h1"/>',
}


def toolbar_icon(key):
    """Render a 24px white SVG icon (e.g. toolbar_icon('plus'))."""
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
           'viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.9" '
           'stroke-linecap="round" stroke-linejoin="round">'
           + TOOLBAR_PATHS[key] + '</svg>')
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    QSvgRenderer(QByteArray(svg.encode('utf-8'))).render(painter)
    painter.end()
    return QIcon(pixmap)


class ZoomWindow(QDialog):
    def __init__(self, parent, camera_name=None):
        super().__init__(parent)
        self.camera_names = (camera_name,) if camera_name else parent.camera_names
        self.setWindowTitle('Night Guard — Camera Zoom')
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(1280, 960)
        layout = QVBoxLayout(self)
        self.camera_title = QLabel()
        self.camera_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.camera_title)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.panels = {}
        self.scroll_areas = {}
        self.controls = {}
        for name in self.camera_names:
            panel = ZoomPanel()
            self.panels[name] = panel
            scroll = QScrollArea()
            scroll.setWidgetResizable(False)
            scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scroll.setStyleSheet('QScrollArea { background: #111827; }')
            scroll.setWidget(panel)
            self.scroll_areas[name] = scroll
            self.stack.addWidget(scroll)
        # Compact icon-only toolbar in the exact left-to-right order of the reference.
        # Save, copy and print remain disabled: this application intentionally does
        # not capture, persist, or export camera frames.
        controls = QHBoxLayout()
        controls.setSpacing(7)
        controls.setContentsMargins(0, 4, 0, 4)
        items = (
            ('left', 'Previous camera', lambda: self.change_camera(-1)),
            ('right', 'Next camera', lambda: self.change_camera(1)),
            ('up', 'Scroll image up', lambda: self.pan_vertical(-1)),
            ('down', 'Scroll image down', lambda: self.pan_vertical(1)),
            ('fit', 'Fit whole image in window', self.fit_to_window),
            ('actual', 'Actual image size (1×)', self.actual_size),
            ('plus', 'Zoom in', lambda: self.current_panel().change_zoom(1.25)),
            ('minus', 'Zoom out', lambda: self.current_panel().change_zoom(1 / 1.25)),
            ('save', 'Save disabled — no snapshots or recordings', None),
            ('copy', 'Copy disabled — no image export', None),
            ('print', 'Print disabled — no image export', None),
        )
        for key, tip, action in items:
            if camera_name and key in ('left', 'right', 'up', 'down'):
                continue
            if len(self.camera_names) == 1 and key in ('left', 'right'):
                continue
            button = QPushButton()
            button.setIcon(toolbar_icon(key))
            button.setIconSize(QSize(20, 20))
            button.setFixedSize(25, 25)
            button.setAutoDefault(False)
            button.setToolTip(tip)
            button.setAccessibleName(tip)
            button.setStyleSheet('''
                QPushButton { background: #080808; border: 0; border-radius: 5px; padding: 2px; }
                QPushButton:hover:enabled { background: #333333; }
                QPushButton:pressed:enabled { background: #555555; }
                QPushButton:disabled { background: #777777; }
            ''')
            if action is None:
                button.setEnabled(False)
            else:
                button.clicked.connect(action)
            controls.addWidget(button)
            self.controls[key] = button
        controls.addStretch(1)
        layout.insertLayout(1, controls)
        for key, direction in ((('Left', -1), ('Right', 1)) if len(self.camera_names) > 1 else ()):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda step=direction: self.change_camera(step))
        self.change_camera(0)

    def current_panel(self):
        return self.panels[self.camera_names[self.stack.currentIndex()]]

    def pan_vertical(self, direction):
        """Scroll the selected zoom image by 20% of its visible height."""
        scrollbar = self.scroll_areas[self.camera_names[self.stack.currentIndex()]].verticalScrollBar()
        step = max(30, round(scrollbar.pageStep() * 0.2))
        scrollbar.setValue(scrollbar.value() + direction * step)

    def actual_size(self):
        """Return the selected image to native 1× resolution."""
        self.current_panel().magnification = 1.0
        self.current_panel().fit()

    def fit_to_window(self):
        """Fit the entire selected image into the visible scroll area (max 1×)."""
        panel = self.current_panel()
        if panel.source.isNull():
            return
        scroll = self.scroll_areas[self.camera_names[self.stack.currentIndex()]]
        area = scroll.viewport().size()
        panel.magnification = min(1.0, max(0.01, area.width() / panel.source.width()),
                                  max(0.01, area.height() / panel.source.height()))
        panel.fit()

    def change_camera(self, step):
        """Cycle FRONT → BACK → LEFT SIDE → RIGHT SIDE, wrapping at the ends."""
        index = (self.stack.currentIndex() + step) % len(self.camera_names)
        self.stack.setCurrentIndex(index)
        title = TITLES[NAMES.index(self.camera_names[index])]
        self.camera_title.setText(f'{title} ({index + 1}/{len(self.camera_names)})' if len(self.camera_names) > 1 else title)
        self.setWindowTitle('Night Guard — Camera Zoom — ' + title)
        self.current_panel().fit()


class Window(QMainWindow):
    def __init__(self, settings):
        super().__init__()
        self.setWindowTitle(settings.get('GUI_TITLE') or 'Four USB Camera Monitor')
        self.camera_names = NAMES[:positive(settings, 'CAMERA_COUNT', 2, maximum=len(NAMES))]
        width = positive(settings, 'GUI_WIDTH', 880)
        height = positive(settings, 'GUI_HEIGHT', 620)
        panel_width = positive(settings, 'WIDTH', 480)
        panel_height = positive(settings, 'HEIGHT', 320)
        control_width = positive(settings, 'CONTROL_PANEL_WIDTH', 200)
        self.resize(width, height)
        root = QWidget()
        outer_layout = QHBoxLayout(root)
        self.control_area = QWidget()
        self.control_area.setFixedWidth(control_width)
        self.control_area.setStyleSheet('background: #e5e7eb;')
        self.zoom_window = None
        self.split_windows = {}
        self.changing_zoom = False
        self.zoom_button = QPushButton('Zoom Out')
        self.zoom_button.setStyleSheet('background-color: #facc15; color: #111827; padding: 8px;')
        self.zoom_button.clicked.connect(self.toggle_zoom)
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
        self.server_address.setToolTip('Enter a server address, then press Enter to connect.')
        self.server_address.textChanged.connect(self.fit_address_field)
        self.fit_address_field()
        connection_bar.addWidget(self.server_address)
        self.connect_button = QPushButton('Connect')
        self.connect_button.setMinimumWidth(120)
        self.set_connection_button('idle')
        self.connect_button.clicked.connect(self.toggle_connection)
        self.server_address.returnPressed.connect(self.connect_server)
        connection_bar.addWidget(self.connect_button)
        connection_bar.addWidget(self.zoom_button)
        self.split_button = QPushButton('Merge')
        self.split_button.clicked.connect(self.toggle_split)
        self.set_split_button(False)
        connection_bar.addWidget(self.split_button)
        connection_bar.addStretch(1)
        self.quit_button = QPushButton('Exit')
        self.quit_button.setMinimumHeight(self.connect_button.sizeHint().height())
        self.quit_button.setToolTip('Close the client')
        self.quit_button.clicked.connect(self.close)
        connection_bar.addWidget(self.quit_button)
        layout.addLayout(connection_bar)
        self.status = QLabel('')
        layout.addWidget(self.status)
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.panels = {}
        for index, (name, title) in enumerate(zip(self.camera_names, TITLES)):
            box = QVBoxLayout()
            box.addWidget(QLabel(title))
            panel = VideoPanel()
            panel.setFixedSize(panel_width, panel_height)
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
        self.server_address.setFocus()
        self.server_address.selectAll()

    def fit_address_field(self):
        """Keep the address input compact while fitting its current text."""
        metrics = self.server_address.fontMetrics()
        self.server_address.setFixedWidth(max(
            metrics.horizontalAdvance(self.server_address.text()),
            metrics.horizontalAdvance(self.server_address.placeholderText()),
        ) + 28)

    def toggle_zoom(self):
        """Open the configured live zoom views, or close the existing zoom window."""
        if self.zoom_window is not None:
            self.zoom_window.reject()
            return
        self.zoom_window = ZoomWindow(self)
        self.zoom_window.finished.connect(self.zoom_closed)
        self.sync_zoom_views()
        self.zoom_button.setText('Zoom In')
        self.zoom_button.setStyleSheet('background-color: #15803d; color: white; padding: 8px;')
        self.zoom_window.show()

    def zoom_closed(self, result):
        self.changing_zoom = True
        for window in list(self.split_windows.values()):
            window.close()
        self.split_windows.clear()
        self.changing_zoom = False
        self.zoom_window = None
        self.zoom_button.setText('Zoom Out')
        self.zoom_button.setStyleSheet('background-color: #facc15; color: #111827; padding: 8px;')
        self.set_split_button(False)

    def set_split_button(self, split):
        self.split_button.setText('Split' if split else 'Merge')
        self.split_button.setToolTip('Click to merge the camera windows' if split
                                     else 'Click to split cameras into separate windows')
        color, foreground = ('#2563eb', 'white') if split else ('#facc15', '#111827')
        self.split_button.setStyleSheet(f'background-color: {color}; color: {foreground}; padding: 8px;')

    def remember_split_view(self, name, window):
        if self.zoom_window is not None:
            self.zoom_window.panels[name].magnification = window.panels[name].magnification

    def split_view_closed(self, name, window):
        if self.changing_zoom:
            return
        self.remember_split_view(name, window)
        self.split_windows.pop(name, None)
        if not self.split_windows and self.zoom_window is not None:
            self.zoom_window.reject()

    def toggle_split(self):
        """Split the configured cameras into windows, or restore the merged viewer."""
        if self.split_windows:
            self.changing_zoom = True
            for name, window in list(self.split_windows.items()):
                self.remember_split_view(name, window)
                window.close()
            self.split_windows.clear()
            self.changing_zoom = False
            self.set_split_button(False)
            self.sync_zoom_views()
            self.zoom_window.show()
            self.zoom_window.raise_()
            return
        if self.zoom_window is None:
            self.toggle_zoom()
        self.zoom_window.hide()
        for name in self.camera_names:
            window = ZoomWindow(self, name)
            window.panels[name].magnification = self.zoom_window.panels[name].magnification
            window.finished.connect(lambda result, n=name, w=window: self.split_view_closed(n, w))
            self.split_windows[name] = window
        self.sync_zoom_views()
        for index, window in enumerate(self.split_windows.values()):
            window.move(self.x() + 40 + index * 32, self.y() + 40 + index * 32)
            window.show()
        self.set_split_button(True)

    def sync_zoom_views(self):
        if self.zoom_window is None:
            return
        viewers = [self.zoom_window] + list(self.split_windows.values())
        for viewer in viewers:
            for name, target in viewer.panels.items():
                source = self.panels[name]
                target.source = source.source
                if target.source.isNull():
                    target.setText(source.text())
                if viewer is not self.zoom_window or not self.split_windows or target.source.isNull():
                    target.fit()

    def set_connection_button(self, state):
        self.connection_button_state = state
        text, color = {'idle': ('Connect', '#facc15'), 'connecting': ('Connecting...', '#a16207'),
                       'connected': ('Disconnect', '#15803d'),
                       'failed': ('Failed', '#b91c1c')}[state]
        self.connect_button.setText(text)
        self.connect_button.setToolTip('Click to disconnect, or edit the address to switch servers' if state == 'connected'
                                       else 'Click to connect to the server')
        foreground = '#111827' if state == 'idle' else 'white'
        self.connect_button.setStyleSheet(
            f'QPushButton {{ background-color: {color}; color: {foreground}; padding: 8px; }}')

    def toggle_connection(self):
        """Disconnect an unchanged connection, or connect to the edited address."""
        address = self.server_address.text().strip()
        if (self.connection_button_state == 'connected' and self.network is not None
                and (not address or address == self.network.url)):
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
        self.sync_zoom_views()
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
        self.sync_zoom_views()
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
        self.network = Network(url, self.camera_names)

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
        self.sync_zoom_views()

    def closeEvent(self, event):
        if self.zoom_window is not None:
            self.zoom_window.reject()
        self.timer.stop()
        networks = self.retired_networks + ([self.network] if self.network else [])
        for network in networks:
            network.stop.set()
        for network in networks:
            network.close()
        event.accept()


def positive(settings, key, default, *, maximum=16384):
    """Validate positive GUI dimensions, e.g. positive({}, 'GUI_WIDTH', 1400)."""
    try:
        value = int(settings.get(key, default))
        if not 1 <= value <= maximum:
            raise ValueError()
        return value
    except (ValueError, TypeError):
        raise ValueError(f'{key} must be an integer from 1 to {maximum}') from None


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
