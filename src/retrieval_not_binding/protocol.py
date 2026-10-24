"""Typed, gold-blind inputs and fixed-size memory envelopes."""
from __future__ import annotations
import dataclasses
import hashlib
import math
import re

DIMENSION = 768


FOLD_COUNT = 4


ARMS = (
    "correct",
    "memory_off",
    "shuffled_binding",
    "wrong_entity",
    "raw_multiset",
)


_ENVELOPE_MAGIC = b"ERMENV01"


_SLOT_COUNT = 5


ENVELOPE_SIZE_BYTES = len(_ENVELOPE_MAGIC) + _SLOT_COUNT * 34


_FEATURE = re.compile(r"feature-run:[0-9a-f]{64}\Z", re.ASCII)


_EPISODE = re.compile(r"episode-run:[0-9a-f]{64}\Z", re.ASCII)


_QUERY = re.compile(r"query-run:[0-9a-f]{64}\Z", re.ASCII)


_CANDIDATE = re.compile(r"candidate-run:[0-9a-f]{64}\Z", re.ASCII)


_CLUSTER = re.compile(r"cluster-run:[0-9a-f]{64}\Z", re.ASCII)


_SOURCES = {"gso", "uco3d"}


_KINDS = {"view", "occlusion", "environment", "detail"}


class BindingError(ValueError):
    def __init__(self, code: str = "input_invalid") -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise BindingError(code)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Profile:
    profile_id: str
    beta_micros: int
    gamma_micros: int
    alpha_micros: int
    lambda_micros: int


def _token(value: str, pattern: re.Pattern[str], code: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        _fail(code)
    return value


def _unit(values: tuple[float, ...]) -> tuple[float, ...]:
    if type(values) is not tuple or len(values) != DIMENSION or any(type(item) is not float or not math.isfinite(item) for item in values):
        _fail("feature_invalid")
    norm = math.sqrt(math.fsum(item * item for item in values))
    if not 0.9999 <= norm <= 1.0001:
        _fail("feature_invalid")
    return values


def _normalize(values) -> tuple[float, ...]:
    values = tuple(float(item) for item in values)
    if len(values) != DIMENSION or any(not math.isfinite(item) for item in values):
        _fail("score_invalid")
    norm = math.sqrt(math.fsum(item * item for item in values))
    if not math.isfinite(norm) or norm <= 1e-12:
        _fail("score_invalid")
    return tuple(item / norm for item in values)


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _micros(value: float) -> int:
    if not math.isfinite(value):
        _fail("score_invalid")
    return int(math.floor(min(1.0, max(-1.0, value)) * 500_000.0 + 500_000.5))


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FeatureVector:
    feature_token: str
    values: tuple[float, ...] = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        _token(self.feature_token, _FEATURE, "feature_invalid")
        _unit(self.values)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CandidateRef:
    candidate_token: str
    feature_token: str

    def __post_init__(self) -> None:
        _token(self.candidate_token, _CANDIDATE, "candidate_invalid")
        _token(self.feature_token, _FEATURE, "candidate_invalid")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeRef:
    episode_token: str
    feature_token: str

    def __post_init__(self) -> None:
        _token(self.episode_token, _EPISODE, "episode_invalid")
        _token(self.feature_token, _FEATURE, "episode_invalid")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MemoryApplication:
    arm: str
    envelope: bytes = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        if self.arm not in ARMS or type(self.envelope) is not bytes or len(self.envelope) != ENVELOPE_SIZE_BYTES:
            _fail("control_invalid")
        decode_envelope(self.envelope)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Query:
    query_token: str
    query_feature_tokens: tuple[str, ...]
    candidates: tuple[CandidateRef, ...]
    applications: tuple[MemoryApplication, ...]

    def __post_init__(self) -> None:
        _token(self.query_token, _QUERY, "query_invalid")
        if (
            type(self.query_feature_tokens) is not tuple or len(self.query_feature_tokens) not in {1, 5}
            or any(_FEATURE.fullmatch(item or "") is None for item in self.query_feature_tokens)
            or type(self.candidates) is not tuple or len(self.candidates) != 8
            or len({item.candidate_token for item in self.candidates}) != 8
            or type(self.applications) is not tuple
            or tuple(item.arm for item in self.applications) != ARMS
            or len({len(item.envelope) for item in self.applications}) != 1
        ):
            _fail("input_invalid")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ReaderInput:
    features: tuple[FeatureVector, ...]
    episode_features: tuple[EpisodeRef, ...]
    queries: tuple[Query, ...]

    def __post_init__(self) -> None:
        if (
            type(self.features) is not tuple or not self.features
            or len({item.feature_token for item in self.features}) != len(self.features)
            or type(self.episode_features) is not tuple or not self.episode_features
            or len({item.episode_token for item in self.episode_features}) != len(self.episode_features)
            or type(self.queries) is not tuple or not self.queries
            or len({item.query_token for item in self.queries}) != len(self.queries)
        ):
            _fail("input_invalid")
        known = {item.feature_token for item in self.features}
        episodes = {item.episode_token for item in self.episode_features}
        if any(item.feature_token not in known for item in self.episode_features):
            _fail("input_invalid")
        for query in self.queries:
            if not set(query.query_feature_tokens) <= known or any(item.feature_token not in known for item in query.candidates):
                _fail("input_invalid")
            for application in query.applications:
                if any(token not in episodes for token, _ in decode_envelope(application.envelope)):
                    _fail("input_invalid")


def encode_envelope(
    episodes: tuple[tuple[str, int], ...], *, reserve_seed: bytes,
) -> bytes:
    if (
        type(episodes) is not tuple or len(episodes) > _SLOT_COUNT
        or type(reserve_seed) is not bytes or len(reserve_seed) != 32
        or any(_EPISODE.fullmatch(token or "") is None or type(position) is not int or not 0 <= position < 10 for token, position in episodes)
    ):
        _fail("control_invalid")
    result = bytearray(_ENVELOPE_MAGIC)
    for index in range(_SLOT_COUNT):
        if index < len(episodes):
            token, position = episodes[index]
            result.extend((1, position)); result.extend(bytes.fromhex(token.split(":", 1)[1]))
        else:
            result.extend((0, 255)); result.extend(hashlib.sha256(reserve_seed + bytes((index,))).digest())
    if len(result) != ENVELOPE_SIZE_BYTES:
        _fail("control_invalid")
    return bytes(result)


def decode_envelope(payload: bytes) -> tuple[tuple[str, int], ...]:
    try:
        if type(payload) is not bytes or len(payload) != ENVELOPE_SIZE_BYTES or payload[:8] != _ENVELOPE_MAGIC:
            _fail("control_invalid")
        result = []
        cursor = 8
        inactive = False
        for _ in range(_SLOT_COUNT):
            active = payload[cursor]; position = payload[cursor + 1]; digest = payload[cursor + 2:cursor + 34]; cursor += 34
            if active == 1 and not inactive and 0 <= position < 10:
                result.append(("episode-run:" + digest.hex(), position))
            elif active == 0 and position == 255:
                inactive = True
            else:
                _fail("control_invalid")
        return tuple(result)
    except BindingError:
        raise
    except Exception:
        _fail("control_invalid")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Ranking:
    query_token: str
    profile_id: str
    arm: str
    ranked_candidate_tokens: tuple[str, ...]
    ranked_score_micros: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ReaderOutput:
    rankings: tuple[Ranking, ...]
