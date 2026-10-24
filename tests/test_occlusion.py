import unittest
from unittest.mock import patch
import retrieval_not_binding.occlusion as occlusion
from retrieval_not_binding.occlusion import (
    FAMILIES, OcclusionError, apply_foreground_aware_occlusion,
    gso_foreground_mask, uco3d_foreground_mask,
)

def gso_fixture() -> tuple[bytes, int, int]:
    width = height = 256
    raw = bytearray(bytes((127, 127, 127, 255)) * (width * height))
    for y in range(48, 208):
        for x in range(56, 200):
            offset = (y * width + x) * 4
            raw[offset:offset + 4] = bytes((48 + x // 4, 64 + y // 5, 96, 255))
    raw[(128 * width + 128) * 4:(128 * width + 128) * 4 + 4] = bytes((127, 127, 127, 255))
    return bytes(raw), width, height


class OcclusionTests(unittest.TestCase):

    def test_gso_foreground_is_boundary_component_complement(self) -> None:
        raw, width, height = gso_fixture()
        result = gso_foreground_mask(raw, width, height)
        self.assertEqual(result.foreground_pixel_count, 160 * 144)
        self.assertEqual(result.bounds_xyxy_exclusive, (56, 48, 200, 208))
        self.assertEqual(result.binary[128 * width + 128], 1)
        self.assertEqual(result.binary[0], 0)

    def test_uco_mask_threshold_is_frozen(self) -> None:
        width = height = 256
        gray = bytearray(width * height)
        for y in range(64, 192):
            gray[y * width + 64:y * width + 192] = bytes((128,)) * 128
        result = uco3d_foreground_mask(bytes(gray), width, height)
        self.assertEqual(result.foreground_pixel_count, 128 * 128)
        self.assertEqual(result.bounds_xyxy_exclusive, (64, 64, 192, 192))

    def test_four_families_are_deterministic_and_coverage_bounded(self) -> None:
        raw, width, height = gso_fixture()
        foreground = gso_foreground_mask(raw, width, height)
        observed = []
        with patch.object(occlusion, '_PILLOW_VERSION_OVERRIDE', '12.0.0'):
            for rank in range(4):
                first = apply_foreground_aware_occlusion(raw, width, height, foreground, source_id='gso', entity_role='seen-memory', source_role_rank=rank, entity_handle=f'fixture-{rank}')
                second = apply_foreground_aware_occlusion(raw, width, height, foreground, source_id='gso', entity_role='seen-memory', source_role_rank=rank, entity_handle=f'fixture-{rank}')
                self.assertEqual(first, second)
                self.assertGreaterEqual(first.occluded_foreground_micros, 180000)
                self.assertLessEqual(first.occluded_foreground_micros, 300000)
                self.assertNotEqual(first.rgba, raw)
                observed.append(first.family)
        self.assertEqual(tuple(observed), FAMILIES)

    def test_invalid_or_tiny_foreground_fails_closed(self) -> None:
        with self.assertRaises(OcclusionError):
            uco3d_foreground_mask(bytes(256 * 256), 256, 256)
