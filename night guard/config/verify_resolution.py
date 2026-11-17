"""Hardware-free checks: python config/verify_resolution.py."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import camera_modes as modes


def discrete(*sizes):
    return {modes.Mode((w, w, 1), (h, h, 1), 'MJPG') for w, h in sizes}


class ResolutionTests(unittest.TestCase):
    def test_common_not_individual_minimum(self):
        sets = [discrete((160, 120), (640, 480), (1280, 720)),
                discrete((320, 240), (640, 480), (1280, 720))]
        self.assertEqual(modes.lowest_common_resolution(sets), (640, 480))
        self.assertEqual(modes.lowest_common_resolution(sets[:1]), (160, 120))
        self.assertEqual(modes.lowest_common_resolution(sets * 2), (640, 480))

    def test_no_common(self):
        with self.assertRaisesRegex(ValueError, 'no common'):
            modes.lowest_common_resolution([discrete((320, 240)), discrete((640, 480))])

    def test_parse_ranges_and_formats(self):
        parsed = modes.parse_modes("""[0]: 'MJPG' (Motion-JPEG)
            Size: Discrete 640x480
            [1]: 'YUYV' (YUYV 4:2:2)
            Size: Stepwise 160x120 - 1920x1080 with step 16/8
            [2]: 'RGB3' (RGB)
            Size: Continuous 32x32 - 1024x768
        """)
        self.assertEqual(len(parsed), 3)
        self.assertEqual(modes.lowest_common_resolution([parsed, discrete((640, 480))]), (640, 480))
        ranged = {modes.Mode((162, 1920, 6), (122, 1080, 6), 'MJPG')}
        stepped = {modes.Mode((160, 1920, 4), (120, 1080, 4), 'MJPG')}
        self.assertEqual(modes.lowest_common_resolution([ranged, stepped]), (168, 128))

    def test_intersections_against_enumeration(self):
        for low_a in range(1, 5):
            for low_b in range(1, 5):
                for step_a in range(1, 5):
                    for step_b in range(1, 5):
                        a, b = (low_a, 24, step_a), (low_b, 20, step_b)
                        expected = set(range(*[a[0], a[1] + 1, a[2]])) & set(range(b[0], b[1] + 1, b[2]))
                        result = modes.intersect_axis(a, b)
                        actual = set(range(result[0], result[1] + 1, result[2])) if result else set()
                        self.assertEqual(actual, expected)

    def test_configure_auto_and_explicit(self):
        cameras = {str(i): SimpleNamespace(device=i, enabled=i < 2, pixel_format=None) for i in range(4)}
        with patch.object(modes, 'read_modes', side_effect=[discrete((64, 48), (640, 480)), discrete((64, 48))]) as read:
            self.assertEqual(modes.configure_resolution(cameras, (0, 0, 15, 80)), (64, 48))
            self.assertEqual([call.args[0] for call in read.call_args_list], [0, 1])
            self.assertTrue(all(camera.settings == (64, 48, 15, 80) for camera in cameras.values()))
            self.assertEqual(cameras['0'].pixel_format, 'MJPG')
        with patch.object(modes, 'read_modes') as read:
            self.assertEqual(modes.configure_resolution(cameras, (1280, 720, 15, 80)), (1280, 720))
            read.assert_not_called()
        for capture in ((0, 480, 15, 80), (640, 0, 15, 80)):
            with self.assertRaisesRegex(ValueError, 'both'):
                modes.configure_resolution(cameras, capture)

    def test_query_errors(self):
        with patch.object(modes.platform, 'system', return_value='Linux'), patch.object(modes.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(ValueError, 'v4l-utils'):
                modes.read_modes(0)
        with patch.object(modes.platform, 'system', return_value='Linux'), patch.object(modes.shutil, 'which', return_value='v4l2-ctl'), patch.object(modes.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr='Permission denied')):
            with self.assertRaisesRegex(ValueError, '/dev/video2.*Permission denied'):
                modes.read_modes(2)


if __name__ == '__main__':
    unittest.main()
