"""Fixed 24-entity candidate retrieval before the shared top-eight readout."""
from __future__ import annotations
import dataclasses
import math
import re
from .protocol import _fail, _SOURCES, _KINDS
_QUERY = re.compile(r"query:[0-9a-f]{64}\Z", re.ASCII)

@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CandidateVector:
    entity_handle: str
    values: tuple[float, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CandidateRanking:
    query_handle: str
    source_id: str
    query_kind: str
    ranked_entity_handles: tuple[str, ...]
    ranked_score_micros: tuple[int, ...]


def _validated_unit(values: tuple[float, ...]) -> tuple[float, ...]:
    if type(values) is not tuple or len(values) != 768 or any(type(item) is not float or not math.isfinite(item) for item in values):
        _fail("candidate_invalid")
    norm = math.sqrt(math.fsum(item * item for item in values))
    if not 0.99999 <= norm <= 1.00001:
        _fail("candidate_invalid")
    return values


def rank_candidates(
    *, query_handle: str, source_id: str, query_kind: str,
    query_views: tuple[tuple[float, ...], ...], candidates: tuple[CandidateVector, ...],
) -> CandidateRanking:
    if (
        _QUERY.fullmatch(query_handle or "") is None or source_id not in _SOURCES or query_kind not in _KINDS
        or type(query_views) is not tuple or len(query_views) != (5 if query_kind == "detail" else 1)
        or type(candidates) is not tuple or len(candidates) != 24
        or len({item.entity_handle for item in candidates}) != 24
    ):
        _fail("candidate_invalid")
    views = tuple(_validated_unit(item) for item in query_views)
    scored = []
    for candidate in candidates:
        if type(candidate) is not CandidateVector or not candidate.entity_handle:
            _fail("candidate_invalid")
        vector = _validated_unit(candidate.values)
        mean = math.fsum(math.fsum(q * c for q, c in zip(view, vector, strict=True)) for view in views) / len(views)
        micros = int(math.floor(min(1.0, max(-1.0, mean)) * 500_000.0 + 500_000.5))
        scored.append((micros, candidate.entity_handle))
    ordered = tuple(sorted(scored, key=lambda item: (-item[0], item[1].encode())))
    entities = tuple(item[1] for item in ordered); scores = tuple(item[0] for item in ordered)
    return CandidateRanking(
        query_handle=query_handle, source_id=source_id, query_kind=query_kind,
        ranked_entity_handles=entities, ranked_score_micros=scores,
    )
