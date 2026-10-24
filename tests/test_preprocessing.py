import unittest
from retrieval_not_binding.preprocessing import apply_gso_controlled_context, extract_detail_tiles

def rgba(width: int, height: int, pixel=(20, 40, 60, 255)) -> bytes:
    return bytes(pixel) * (width * height)


class PreprocessingTests(unittest.TestCase):

    def test_context_replaces_only_boundary_connected_gray(self) -> None:
        width = height = 256
        raw = bytearray(bytes((127, 127, 127, 255)) * (width * height))
        for y in range(64, 192):
            for x in range(64, 192):
                raw[(y * width + x) * 4:(y * width + x) * 4 + 4] = bytes((20, 40, 60, 255))
        raw[(128 * width + 128) * 4:(128 * width + 128) * 4 + 4] = bytes((127, 127, 127, 255))
        result = apply_gso_controlled_context(bytes(raw), width, height)
        self.assertEqual(result.background_pixel_count, width * height - 128 * 128)
        self.assertEqual(result.object_pixel_count, 128 * 128)
        center = (128 * width + 128) * 4
        self.assertEqual(result.rgba[center:center + 4], bytes((127, 127, 127, 255)))
        self.assertEqual(result.rgba[(80 * width + 80) * 4:(80 * width + 80) * 4 + 4], bytes((20, 40, 60, 255)))
        self.assertNotEqual(result.rgba[:4], bytes((127, 127, 127, 255)))

    def test_detail_tiles_match_frozen_overlapping_bounds(self) -> None:
        width, height = (256, 320)
        raw = bytes((index * 17 & 255 for index in range(width * height * 4)))
        raw = bytearray(raw)
        raw[3::4] = bytes((255,)) * (width * height)
        tiles = extract_detail_tiles(bytes(raw), width, height)
        self.assertEqual(tuple((item.bounds_xyxy_exclusive for item in tiles)), ((0, 0, 141, 176), (115, 0, 256, 176), (0, 144, 141, 320), (115, 144, 256, 320)))
        for item in tiles:
            left, top, right, bottom = item.bounds_xyxy_exclusive
            expected = b''.join((bytes(raw[(y * width + left) * 4:(y * width + right) * 4]) for y in range(top, bottom)))
            self.assertEqual(item.rgba, expected)
