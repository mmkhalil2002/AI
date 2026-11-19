"""Report Linux camera capture modes without changing the camera settings."""
import platform
import shutil
import subprocess


def display_resolutions(device, name=None):
    """Print driver-reported formats, frame sizes/ranges, and frame rates.

    Example: display_resolutions(0, 'FRONT') queries /dev/video0.
    Returns True when modes were reported; preview can continue on failure.
    """
    path = f'/dev/video{device}' if isinstance(device, int) else str(device)
    print(f'\n=== {name or path}: supported resolutions ({path}) ===', flush=True)
    if platform.system() != 'Linux':
        print('Mode enumeration requires Linux V4L2 for these camera tests.', flush=True)
        return False
    executable = shutil.which('v4l2-ctl')
    if executable is None:
        print('Cannot enumerate modes: install v4l-utils with: sudo apt install v4l-utils', flush=True)
        return False
    try:
        result = subprocess.run([executable, '--device', path, '--list-formats-ext'],
                                capture_output=True, text=True, errors='replace', timeout=15,
                                check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f'Cannot enumerate modes for {path}: {exc}', flush=True)
        return False
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    if result.returncode != 0:
        print(f'Mode query failed for {path}: {result.stderr.strip() or "check device and permissions"}', flush=True)
        return False
    if 'Size:' not in result.stdout:
        print('No frame sizes reported by this device; verify it is a capture node.', flush=True)
        return False
    return True
