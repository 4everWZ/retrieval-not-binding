"""Source-stratified entity bootstrap for the profile-selected evaluation."""
from __future__ import annotations
import dataclasses
import random
from .protocol import ARMS, _SOURCES, _fail
from .evaluation import QueryOutcome

@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BootstrapSummary:
    rounds: int
    seed: int
    point_delta_micros: int
    lower_micros: int
    upper_micros: int
    defined_rounds: int


def _source_delta(rows: tuple[QueryOutcome, ...]) -> int:
    if not rows:
        _fail("summary_invalid")
    primary = sum(dict(item.ranks_by_arm)[ARMS[0]] == 1 for item in rows)
    off = sum(dict(item.ranks_by_arm)[ARMS[1]] == 1 for item in rows)
    difference = primary - off
    magnitude = (abs(difference) * 1_000_000 + len(rows) // 2) // len(rows)
    return magnitude if difference >= 0 else -magnitude


def bootstrap_summary(outcomes: tuple[QueryOutcome, ...], *, rounds: int, seed: int) -> BootstrapSummary:
    seen = tuple(item for item in outcomes if not item.pseudo_unseen)
    if type(rounds) is not int or rounds != 2000 or type(seed) is not int or seed != 20260803 or len(seen) != 128:
        _fail("summary_invalid")
    groups = {
        source: {cluster: tuple(item for item in seen if item.source_id == source and item.entity_cluster_token == cluster)
                 for cluster in sorted({item.entity_cluster_token for item in seen if item.source_id == source})}
        for source in sorted(_SOURCES)
    }
    if any(len(values) != 16 or any(len(rows) != 4 for rows in values.values()) for values in groups.values()):
        _fail("summary_invalid")
    source_point = tuple(_source_delta(tuple(item for rows in groups[source].values() for item in rows)) for source in sorted(_SOURCES))
    point_total = sum(source_point)
    point = (abs(point_total) + 1) // 2
    point = point if point_total >= 0 else -point
    generator = random.Random(seed); replicates = []
    for _ in range(rounds):
        deltas = []
        for source in sorted(_SOURCES):
            keys = tuple(groups[source])
            sampled = tuple(item for _ in keys for item in groups[source][keys[generator.randrange(len(keys))]])
            deltas.append(_source_delta(sampled))
        total = sum(deltas); value = (abs(total) + 1) // 2
        replicates.append(value if total >= 0 else -value)
    ordered = sorted(replicates)
    return BootstrapSummary(
        rounds=rounds, seed=seed, point_delta_micros=point,
        lower_micros=ordered[rounds * 25 // 1000],
        upper_micros=ordered[min(rounds - 1, (rounds * 975 + 999) // 1000 - 1)],
        defined_rounds=rounds,
    )
