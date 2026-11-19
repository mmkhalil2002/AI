"""Select a common Linux V4L2 capture resolution from driver-reported modes."""
from dataclasses import dataclass
from math import gcd
import platform
import re
import shutil
import subprocess


@dataclass(frozen=True)
class Mode:
    width: tuple
    height: tuple
    fourcc: str

    def supports(self, width, height):
        return all(low <= value <= high and (value - low) % step == 0
                   for value, (low, high, step) in ((width, self.width), (height, self.height)))


def parse_modes(output):
    """Parse discrete sizes and stepwise/continuous ranges from v4l2-ctl."""
    modes = set()
    fourcc = ''
    for line in output.splitlines():
        format_match = re.search(r"\[\d+\]:\s*'([^']{4})'", line)
        if format_match:
            fourcc = format_match[1]
        discrete = re.search(r'Size:\s*Discrete\s+(\d+)x(\d+)', line)
        ranged = re.search(r'Size:\s*(Stepwise|Continuous)\s+(\d+)x(\d+)\s*-\s*(\d+)x(\d+)(?:\s+with step\s+(\d+)/(\d+))?', line)
        if discrete:
            w, h = map(int, discrete.groups())
            mode = Mode((w, w, 1), (h, h, 1), fourcc)
        elif ranged:
            kind, w, h, maxw, maxh, sw, sh = ranged.groups()
            if kind == 'Stepwise' and (sw is None or sh is None):
                continue
            mode = Mode((int(w), int(maxw), int(sw or 1)),
                        (int(h), int(maxh), int(sh or 1)), fourcc)
        else:
            continue
        if fourcc and all(0 < low <= high and step > 0 for low, high, step in (mode.width, mode.height)):
            modes.add(mode)
    return modes


def read_modes(device):
    if platform.system() != 'Linux':
        raise ValueError('Automatic common resolution requires Linux V4L2; set positive WIDTH and HEIGHT on this OS')
    executable = shutil.which('v4l2-ctl')
    if executable is None:
        raise ValueError('Automatic resolution requires v4l-utils: sudo apt install v4l-utils')
    path = f'/dev/video{device}' if isinstance(device, int) else str(device)
    try:
        result = subprocess.run([executable, '--device', path, '--list-formats-ext'],
                                capture_output=True, text=True, errors='replace', timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f'Cannot query camera modes for {path}: {exc}') from exc
    if result.returncode:
        raise ValueError(f'Cannot query camera modes for {path}: {result.stderr.strip() or "check device and permissions"}')
    modes = parse_modes(result.stdout)
    if not modes:
        raise ValueError(f'No supported frame sizes reported for {path}; verify the capture node')
    return modes


def intersect_axis(a, b):
    """Intersect two bounded arithmetic progressions without enumerating pixels."""
    low, high = max(a[0], b[0]), min(a[1], b[1])
    divisor = gcd(a[2], b[2])
    difference = b[0] - a[0]
    if low > high or difference % divisor:
        return None
    modulus = b[2] // divisor
    offset = ((difference // divisor) * pow(a[2] // divisor, -1, modulus)) % modulus if modulus > 1 else 0
    step = a[2] * modulus
    first = a[0] + a[2] * offset
    first += ((low - first + step - 1) // step) * step
    return (first, high, step) if first <= high else None


def lowest_common_resolution(mode_sets):
    """Choose minimum pixel area, then width/height, shared by every camera."""
    if not mode_sets or any(not modes for modes in mode_sets):
        raise ValueError('Cannot choose a common resolution without modes for every enabled camera')
    common = {(mode.width, mode.height) for mode in mode_sets[0]}
    for modes in mode_sets[1:]:
        intersections = set()
        for width, height in common:
            for mode in modes:
                w = intersect_axis(width, mode.width)
                h = intersect_axis(height, mode.height)
                if w and h:
                    intersections.add((w, h))
        common = intersections
        if not common:
            raise ValueError('Enabled cameras have no common supported resolution; configure positive WIDTH and HEIGHT')
    return min(((w[0], h[0]) for w, h in common), key=lambda size: (size[0] * size[1], *size))


def configure_resolution(cameras, capture):
    """Resolve auto dimensions once and apply the same size to every worker."""
    width, height, fps, quality = capture
    if bool(width) != bool(height):
        raise ValueError('WIDTH and HEIGHT must both be 0 for automatic selection, or both positive')
    enabled = [camera for camera in cameras.values() if camera.enabled and camera.device is not None]
    if width == height == 0:
        mode_sets = []
        for camera in enabled:
            modes = read_modes(camera.device)
            if camera.pixel_format:
                modes = {mode for mode in modes if mode.fourcc == camera.pixel_format}
                if not modes:
                    raise ValueError(f'Camera {camera.device} does not advertise {camera.pixel_format} capture')
            mode_sets.append(modes)
        width, height = lowest_common_resolution(mode_sets)
        for camera, modes in zip(enabled, mode_sets):
            formats = {mode.fourcc for mode in modes if mode.supports(width, height)}
            camera.pixel_format = 'MJPG' if 'MJPG' in formats else sorted(formats)[0]
    for camera in cameras.values():
        camera.settings = (width, height, fps, quality)
    return width, height
