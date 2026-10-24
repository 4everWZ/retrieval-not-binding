"""Controlled context, detail views, and native video frame selection."""
from __future__ import annotations
import dataclasses
from array import array

_BACKGROUND = bytes((127, 127, 127, 255))


_CHECKER = (bytes((96, 112, 128, 255)), bytes((176, 160, 144, 255)))


class PreprocessingError(ValueError):
    def __init__(self, code: str = "input_invalid") -> None:
        self.code = code if code in {
            "input_invalid", "invariant_failed", "resource_limit",
        } else "input_invalid"
        super().__init__(self.code)


def _fail(code: str) -> None:
    raise PreprocessingError(code)


def _validate_rgba(raw: bytes, width: int, height: int) -> None:
    if (
        type(raw) is not bytes or type(width) is not int or type(height) is not int
        or not 256 <= width <= 32768 or not 256 <= height <= 32768
        or width * height > 32 * 1024 * 1024
        or len(raw) != width * height * 4
        or any(raw[index] != 255 for index in range(3, len(raw), 4))
    ):
        _fail("input_invalid")


@dataclasses.dataclass(frozen=True, slots=True)
class ControlledContextPixels:
    rgba: bytes = dataclasses.field(repr=False)
    background_pixel_count: int
    object_pixel_count: int
    changed_pixel_count: int


@dataclasses.dataclass(frozen=True, slots=True)
class DetailTilePixels:
    bounds_xyxy_exclusive: tuple[int, int, int, int]
    width: int
    height: int
    rgba: bytes = dataclasses.field(repr=False)


def apply_gso_controlled_context(raw: bytes, width: int, height: int) -> ControlledContextPixels:
    """Replace only boundary-connected exact renderer background pixels."""

    _validate_rgba(raw, width, height)
    count = width * height
    visited = bytearray(count)
    queue = array("I")

    def add(index: int) -> None:
        if not visited[index] and raw[index * 4:index * 4 + 4] == _BACKGROUND:
            visited[index] = 1
            queue.append(index)

    for x in range(width):
        add(x)
        add((height - 1) * width + x)
    for y in range(1, height - 1):
        add(y * width)
        add(y * width + width - 1)
    head = 0
    while head < len(queue):
        index = queue[head]
        head += 1
        y, x = divmod(index, width)
        if x:
            add(index - 1)
        if x + 1 < width:
            add(index + 1)
        if y:
            add(index - width)
        if y + 1 < height:
            add(index + width)
    background = len(queue)
    if background < 1 or background >= count:
        _fail("invariant_failed")
    output = bytearray(raw)
    for index, selected in enumerate(visited):
        if selected:
            y, x = divmod(index, width)
            output[index * 4:index * 4 + 4] = _CHECKER[(x // 128 + y // 128) & 1]
    if any(
        output[index * 4:index * 4 + 4] != raw[index * 4:index * 4 + 4]
        for index, selected in enumerate(visited) if not selected
    ):
        _fail("invariant_failed")
    return ControlledContextPixels(bytes(output), background, count - background, background)


def extract_detail_tiles(raw: bytes, width: int, height: int) -> tuple[DetailTilePixels, ...]:
    _validate_rgba(raw, width, height)
    result = []
    for left, top, right, bottom in detail_tile_bounds(width, height):
        tile_width, tile_height = right - left, bottom - top
        rows = tuple(
            raw[(y * width + left) * 4:(y * width + right) * 4]
            for y in range(top, bottom)
        )
        payload = b"".join(rows)
        if len(payload) != tile_width * tile_height * 4:
            _fail("invariant_failed")
        result.append(DetailTilePixels(
            (left, top, right, bottom), tile_width, tile_height, payload,
        ))
    if len(result) != 4:
        _fail("invariant_failed")
    return tuple(result)


def ten_even_temporal_bin_indices(decoded_frame_count: int) -> tuple[int, ...]:
    """Return ten deterministic frames spanning the full decoded timeline."""

    if type(decoded_frame_count) is not int or not 10 <= decoded_frame_count <= 2**31 - 1:
        _fail("input_mismatch")
    result = tuple(
        (ordinal * (decoded_frame_count - 1)) // 9
        for ordinal in range(10)
    )
    if (
        result != tuple(sorted(set(result)))
        or result[0] != 0 or result[-1] != decoded_frame_count - 1
    ):
        _fail("invariant_failed")
    return result


def detail_tile_bounds(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    """Return the frozen row-major 2x2 bounds with 10% full-axis overlap."""

    if (
        type(width) is not int or type(height) is not int
        or not 256 <= width <= 32768 or not 256 <= height <= 32768
    ):
        _fail("input_mismatch")
    x0 = (9 * width + 10) // 20
    x1 = (11 * width + 10) // 20
    y0 = (9 * height + 10) // 20
    y1 = (11 * height + 10) // 20
    if not 0 < x0 < x1 < width or not 0 < y0 < y1 < height:
        _fail("invariant_failed")
    return (
        (0, 0, x1, y1), (x0, 0, width, y1),
        (0, y0, x1, height), (x0, y0, width, height),
    )
