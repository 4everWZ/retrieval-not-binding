"""Foreground-aware occlusion used for the paper cohort."""
from __future__ import annotations
import dataclasses
import hashlib
from array import array

FAMILIES = ("ellipse", "rotated-quadrilateral", "boundary-polygon", "diagonal-band")


PALETTES = (
    ((74, 64, 55), (128, 112, 91)),
    ((47, 67, 83), (101, 126, 143)),
    ((36, 39, 43), (82, 87, 91)),
    ((170, 161, 143), (218, 211, 192)),
)


_BACKGROUND = bytes((127, 127, 127, 255))


_ANCHORS = (150_000, 350_000, 650_000, 850_000)


_SCALES = (350_000, 425_000, 500_000, 575_000, 650_000, 725_000, 800_000, 875_000, 950_000)


_PILLOW_VERSION_OVERRIDE: str | None = None


class OcclusionError(ValueError):
    def __init__(self, code: str = "input_invalid") -> None:
        self.code = code if code in {
            "input_invalid", "foreground_invalid", "candidate_invalid", "invariant_failed",
        } else "input_invalid"
        super().__init__(self.code)


def _fail(code: str) -> None:
    raise OcclusionError(code)


def _validate_dimensions(width: int, height: int) -> None:
    if (
        type(width) is not int or type(height) is not int
        or not 256 <= width <= 32768 or not 256 <= height <= 32768
        or width * height > 32 * 1024 * 1024
    ):
        _fail("input_invalid")


def _validate_rgba(raw: bytes, width: int, height: int) -> None:
    _validate_dimensions(width, height)
    if (
        type(raw) is not bytes or len(raw) != width * height * 4
        or any(raw[index] != 255 for index in range(3, len(raw), 4))
    ):
        _fail("input_invalid")


@dataclasses.dataclass(frozen=True, slots=True)
class ForegroundMask:
    width: int
    height: int
    binary: bytes = dataclasses.field(repr=False)
    foreground_pixel_count: int
    bounds_xyxy_exclusive: tuple[int, int, int, int]


@dataclasses.dataclass(frozen=True, slots=True)
class OccludedPixels:
    rgba: bytes = dataclasses.field(repr=False)
    family: str
    palette_index: int
    candidate_index: int
    foreground_pixel_count: int
    occluded_foreground_pixel_count: int
    occluded_foreground_micros: int
    mask_pixel_count: int
    mask_bounds_xyxy_exclusive: tuple[int, int, int, int]


def _foreground(binary: bytes, width: int, height: int) -> ForegroundMask:
    _validate_dimensions(width, height)
    if type(binary) is not bytes or len(binary) != width * height or any(value not in (0, 1) for value in binary):
        _fail("foreground_invalid")
    selected = 0
    left, top, right, bottom = width, height, -1, -1
    for index, value in enumerate(binary):
        if not value:
            continue
        selected += 1
        y, x = divmod(index, width)
        left, top, right, bottom = min(left, x), min(top, y), max(right, x), max(bottom, y)
    if selected < 256:
        _fail("foreground_invalid")
    return ForegroundMask(
        width, height, binary, selected, (left, top, right + 1, bottom + 1),
    )


def gso_foreground_mask(raw: bytes, width: int, height: int) -> ForegroundMask:
    """Use the complement of the exact boundary-connected renderer background."""

    _validate_rgba(raw, width, height)
    count = width * height
    background = bytearray(count)
    queue = array("I")

    def add(index: int) -> None:
        if not background[index] and raw[index * 4:index * 4 + 4] == _BACKGROUND:
            background[index] = 1
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
    if not queue or len(queue) == count:
        _fail("foreground_invalid")
    return _foreground(bytes(0 if value else 1 for value in background), width, height)


def uco3d_foreground_mask(gray8: bytes, width: int, height: int) -> ForegroundMask:
    """Threshold one official uCO3D mask-video frame decoded as gray8."""

    _validate_dimensions(width, height)
    if type(gray8) is not bytes or len(gray8) != width * height:
        _fail("foreground_invalid")
    return _foreground(bytes(value >= 128 for value in gray8), width, height)


def _candidate_spec(family: str, variant: int, scale: int) -> tuple[str, tuple[int, ...]]:
    if family in {"ellipse", "rotated-quadrilateral"}:
        ax = _ANCHORS[variant]
        ay = _ANCHORS[(variant + 1) % len(_ANCHORS)]
        return family, (ax, ay, scale)
    return family, (variant, scale)


def _draw_candidate(Image, ImageDraw, width: int, height: int, bounds: tuple[int, int, int, int], spec, factor: int):
    family, values = spec
    left, top, right, bottom = bounds
    bw, bh = right - left, bottom - top
    canvas = Image.new("L", (width * factor, height * factor), 0)
    draw = ImageDraw.Draw(canvas)

    def p(value: int) -> int:
        return value * factor

    if family in {"ellipse", "rotated-quadrilateral"}:
        ax, ay, scale = values
        cx = left + (bw * ax + 500_000) // 1_000_000
        cy = top + (bh * ay + 500_000) // 1_000_000
        rw = max(2, (bw * scale + 1_999_999) // 2_000_000)
        rh = max(2, (bh * scale * 3 + 7_999_999) // 8_000_000)
        if family == "ellipse":
            draw.ellipse((p(cx - rw), p(cy - rh), p(cx + rw), p(cy + rh)), fill=255)
        else:
            draw.polygon(tuple((p(x), p(y)) for x, y in (
                (cx - rw, cy - rh // 3), (cx - rw // 4, cy - rh),
                (cx + rw, cy + rh // 3), (cx + rw // 4, cy + rh),
            )), fill=255)
    elif family == "boundary-polygon":
        side, scale = values
        depth = max(3, (min(bw, bh) * scale + 999_999) // 1_000_000)
        if side == 0:
            points = ((0, top + bh // 5), (left + depth, top + bh // 3), (left + depth * 3 // 4, top + 4 * bh // 5), (0, bottom))
        elif side == 1:
            points = ((width, top), (right - depth * 3 // 4, top + bh // 5), (right - depth, top + 2 * bh // 3), (width, top + 4 * bh // 5))
        elif side == 2:
            points = ((left, 0), (left + bw // 5, top + depth), (left + 4 * bw // 5, top + depth * 3 // 4), (right, 0))
        else:
            points = ((left + bw // 5, height), (left + bw // 3, bottom - depth), (right, bottom - depth * 3 // 4), (right, height))
        draw.polygon(tuple((p(x), p(y)) for x, y in points), fill=255)
    else:
        orientation, scale = values
        band = max(3, (min(bw, bh) * scale + 2_999_999) // 3_000_000)
        if orientation & 1:
            points = ((left - band, top), (left, top - band), (right + band, bottom), (right, bottom + band))
        else:
            points = ((left, bottom + band), (left - band, bottom), (right, top - band), (right + band, top))
        if orientation >= 2:
            shift = (bw - bh) // 6
            points = tuple((x, y + shift) for x, y in points)
        draw.polygon(tuple((p(x), p(y)) for x, y in points), fill=255)
    return canvas


def _mask_bounds(mask: bytes, width: int) -> tuple[int, int, int, int]:
    count = 0
    left, top, right, bottom = width, len(mask) // width, -1, -1
    for index, value in enumerate(mask):
        if value < 128:
            continue
        count += 1
        y, x = divmod(index, width)
        left, top, right, bottom = min(left, x), min(top, y), max(right, x), max(bottom, y)
    if not count:
        _fail("candidate_invalid")
    return left, top, right + 1, bottom + 1


def apply_foreground_aware_occlusion(
    raw: bytes,
    width: int,
    height: int,
    foreground: ForegroundMask,
    *,
    source_id: str,
    entity_role: str,
    source_role_rank: int,
    entity_handle: str,
) -> OccludedPixels:
    """Select a foreground-coverage-constrained shape and composite textured pixels."""

    _validate_rgba(raw, width, height)
    if (
        type(foreground) is not ForegroundMask or foreground.width != width or foreground.height != height
        or source_id not in {"gso", "uco3d"}
        or entity_role not in {"seen-memory", "pseudo-unseen-query-only"}
        or type(source_role_rank) is not int or source_role_rank < 0
        or type(entity_handle) is not str or not entity_handle or len(entity_handle) > 256
    ):
        _fail("input_invalid")
    try:
        from PIL import Image, ImageChops, ImageDraw, __version__ as pillow_version
    except ImportError:
        _fail("input_invalid")
    if (_PILLOW_VERSION_OVERRIDE or pillow_version) != "12.0.0":
        _fail("input_invalid")

    family = FAMILIES[source_role_rank % 4]
    source_offset = 0 if source_id == "gso" else 2
    role_offset = 0 if entity_role == "seen-memory" else 1
    palette_index = (source_role_rank // 4 + source_offset + role_offset) % 4
    specs = tuple(_candidate_spec(family, variant, scale) for variant in range(4) for scale in _SCALES)
    rotation = int.from_bytes(hashlib.sha256(
        b"episodic-occlusion-v2-candidate-order\0" + entity_handle.encode("utf-8")
    ).digest()[:8], "big") % len(specs)
    specs = specs[rotation:] + specs[:rotation]

    fg_image = Image.frombytes("L", (width, height), bytes(value * 255 for value in foreground.binary))
    ranked = []
    for index, spec in enumerate(specs):
        mask = _draw_candidate(Image, ImageDraw, width, height, foreground.bounds_xyxy_exclusive, spec, 1)
        overlap = ImageChops.multiply(mask, fg_image).histogram()[255]
        micros = overlap * 1_000_000 // foreground.foreground_pixel_count
        ranked.append((abs(micros - 240_000), index, spec))
    ranked.sort()

    chosen = None
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for _, index, spec in ranked:
        large = _draw_candidate(Image, ImageDraw, width, height, foreground.bounds_xyxy_exclusive, spec, 4)
        softened = large.resize((width, height), resample=resampling)
        binary_mask = softened.point(lambda value: 255 if value >= 128 else 0)
        overlap = ImageChops.multiply(binary_mask, fg_image).histogram()[255]
        micros = overlap * 1_000_000 // foreground.foreground_pixel_count
        if 180_000 <= micros <= 300_000:
            chosen = index, softened.tobytes(), overlap, micros
            break
    if chosen is None:
        _fail("candidate_invalid")
    candidate_index, mask, overlap, micros = chosen

    first, second = PALETTES[palette_index]
    output = bytearray(raw)
    mask_pixels = 0
    for index, alpha in enumerate(mask):
        if not alpha:
            continue
        mask_pixels += int(alpha >= 128)
        y, x = divmod(index, width)
        tx = x * 65_535 // max(1, width - 1)
        ty = y * 65_535 // max(1, height - 1)
        weight = (tx + ty) // 2
        weave = 5 if ((x // 11 + y // 7) & 1) else -5
        offset = index * 4
        for channel in range(3):
            texture = first[channel] + ((second[channel] - first[channel]) * weight + 32_767) // 65_535 + weave
            texture = min(255, max(0, texture))
            output[offset + channel] = (raw[offset + channel] * (255 - alpha) + texture * alpha + 127) // 255
    if mask_pixels < 1 or bytes(output) == raw or not 180_000 <= micros <= 300_000:
        _fail("invariant_failed")
    return OccludedPixels(
        bytes(output), family, palette_index, candidate_index,
        foreground.foreground_pixel_count, overlap, micros, mask_pixels,
        _mask_bounds(mask, width),
    )
