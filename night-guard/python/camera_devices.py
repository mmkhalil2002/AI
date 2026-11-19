"""Discover V4L2 capture nodes and DirectShow capture devices."""

from dataclasses import dataclass
from pathlib import Path
import platform
import shutil

from camera_modes import read_modes


INTEGRATED_NAMES = ("integrated", "built-in", "builtin", "truevision", "facetime", "surface camera")


def is_integrated(name):
    return any(word in name.casefold() for word in INTEGRATED_NAMES)


@dataclass(frozen=True)
class DirectShowCamera:
    index: int
    identifier: str
    name: str

    def __str__(self):
        return f"{self.index}: {self.name or 'DirectShow camera'}"


def directshow_cameras():
    """Return DirectShow video monikers in the same order OpenCV indexes them."""
    from ctypes import (POINTER, WINFUNCTYPE, WinDLL, byref, c_byte, c_long,
                        c_ulong, c_void_p, c_wchar_p, cast, wstring_at)

    ole = WinDLL("ole32")
    automation = WinDLL("oleaut32")
    pointer = c_void_p

    def guid(value):
        result = (c_byte * 16)()
        ole.CLSIDFromString(c_wchar_p(value), byref(result))
        return result

    def method(obj, slot, result, *args):
        address = cast(obj, POINTER(POINTER(pointer)))[0][slot]
        return WINFUNCTYPE(result, pointer, *args)(address)

    def check(result):
        if result < 0:
            raise OSError(f"DirectShow enumeration failed (HRESULT {result & 0xffffffff:#010x})")

    def release(obj):
        if obj:
            method(obj, 2, c_ulong)(obj)

    def friendly_name(moniker):
        bag = pointer()
        property_bag = guid("{55272A00-42CB-11CE-8135-00AA004BB851}")
        result = method(moniker, 9, c_long, pointer, pointer, pointer, POINTER(pointer))(
            moniker, None, None, byref(property_bag), byref(bag))
        if result < 0:
            return ""
        # A VARIANT stores its type at offset 0 and its BSTR pointer at offset 8.
        value = (c_byte * 24)()
        try:
            result = method(bag, 3, c_long, c_wchar_p, pointer, pointer)(
                bag, "FriendlyName", byref(value), None)
            if result >= 0 and value[0] == 8:  # VT_BSTR
                address = pointer.from_buffer(value, 8).value
                return wstring_at(address) if address else ""
            return ""
        finally:
            automation.VariantClear(byref(value))
            release(bag)

    check(ole.CoInitialize(None))
    device_enum, monikers = pointer(), pointer()
    try:
        clsid = guid("{62BE5D10-60EB-11D0-BD3B-00A0C911CE86}")
        iid = guid("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
        category = guid("{860BB310-5D01-11D0-BD3B-00A0C911CE86}")
        check(ole.CoCreateInstance(byref(clsid), None, 1, byref(iid), byref(device_enum)))
        check(method(device_enum, 3, c_long, pointer, POINTER(pointer), c_ulong)(
            device_enum, byref(category), byref(monikers), 0))
        cameras = []
        while monikers:
            moniker = pointer()
            result = method(monikers, 3, c_long, c_ulong, POINTER(pointer), pointer)(
                monikers, 1, byref(moniker), None)
            check(result)
            if result == 1:  # S_FALSE means the enumeration is exhausted.
                break
            display_name = pointer()
            try:
                check(method(moniker, 20, c_long, pointer, pointer, POINTER(pointer))(
                    moniker, None, None, byref(display_name)))
                cameras.append(DirectShowCamera(len(cameras), wstring_at(display_name),
                                                friendly_name(moniker)))
            finally:
                ole.CoTaskMemFree(display_name)
                release(moniker)
        return cameras
    finally:
        release(monikers)
        release(device_enum)
        ole.CoUninitialize()


def linux_cameras():
    """Prefer stable links, skip metadata nodes, and keep one node per USB camera."""
    if shutil.which("v4l2-ctl") is None:
        raise ValueError("Ubuntu camera discovery requires v4l-utils: sudo apt install v4l-utils")
    devices = Path("/dev")
    links = {}
    for directory in ("by-id", "by-path"):
        for link in sorted((devices / "v4l" / directory).glob("*")):
            if link.exists():
                links.setdefault(link.resolve(), link)
    capture_nodes = {}
    nodes = (node for node in devices.glob("video*") if node.name[5:].isdecimal())
    for node in sorted(nodes, key=lambda path: int(path.name[5:])):
        sys_node = Path("/sys/class/video4linux") / node.name
        hardware = (sys_node / "device").resolve()
        usb_device = next((parent for parent in (hardware, *hardware.parents)
                           if (parent / "idVendor").is_file()), None)
        identity = usb_device or (hardware if hardware.is_dir() else node)
        if identity in capture_nodes:
            continue
        device = str(links.get(node, node))
        try:
            read_modes(device)
        except ValueError:
            continue
        try:
            name = (sys_node / "name").read_text(encoding="utf-8", errors="replace")
        except OSError:
            name = ""
        capture_nodes[identity] = (device, usb_device is not None and not is_integrated(name))
    external = [device for device, usb in capture_nodes.values() if usb]
    return sorted(external or [device for device, _ in capture_nodes.values()])


def discover_cameras():
    if platform.system() == "Windows":
        cameras = directshow_cameras()
        external = [camera for camera in cameras
                    if "usb#" in camera.identifier.casefold()
                    and not is_integrated(camera.name)]
        return sorted(external or cameras, key=lambda camera: camera.identifier.casefold())
    return linux_cameras()


def camera_source(device):
    """Resolve a stable DirectShow moniker to the index it has at open time."""
    if isinstance(device, DirectShowCamera):
        for camera in directshow_cameras():
            if camera.identifier.casefold() == device.identifier.casefold():
                return camera.index
        raise ValueError(f"Camera is unavailable: {device.name or device.identifier}")
    if device is None:
        raise ValueError("No camera detected for this position")
    return device


def assign_cameras(values):
    """Fill auto positions from discovery, reserving explicitly selected devices."""
    windows = platform.system() == "Windows"
    auto = [not value or str(value).strip().casefold() == "auto" for value in values]
    devices = [None] * len(values)
    reserved = set()
    detected = discover_cameras() if any(auto) else []

    def key(device):
        if windows:
            return device.index if isinstance(device, DirectShowCamera) else device
        path = f"/dev/video{device}" if isinstance(device, int) else device
        return Path(path).resolve()

    for index, value in enumerate(values):
        if auto[index]:
            continue
        value = str(value).strip()
        if value.isdecimal():
            device = int(value)
        elif not windows:
            device = value
        elif value.casefold().startswith("dshow:"):
            identifier = value[6:]
            device = next((camera for camera in directshow_cameras()
                           if camera.identifier.casefold() == identifier.casefold()),
                          DirectShowCamera(-1, identifier, ""))
        else:
            raise ValueError(f"Expected auto, a Windows camera index, or a dshow: identifier, got {value!r}")
        devices[index] = device
        reserved.add(key(device))

    available = iter(device for device in detected if key(device) not in reserved)
    for index, automatic in enumerate(auto):
        if automatic:
            devices[index] = next(available, None)
    return devices
