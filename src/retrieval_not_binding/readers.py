"""Fixed mean, reservoir, k-center, and ERM readers (manuscript Table 2)."""
from __future__ import annotations
import dataclasses
import hashlib
import math
from .protocol import (ARMS, DIMENSION, ReaderInput, ReaderOutput, Profile, Ranking, decode_envelope, _dot, _fail, _micros, _normalize)
from .memory import run_erm_profiles

# Preserve the original sampling namespace to retain exemplar priorities.
BASELINES = ("mean_prototype", "reservoir", "kcenter", "erm")


CONTROL_ARMS = ("correct", "memory_off", "wrong_entity", "shuffled_binding")


_SOURCE_ARMS = {
    "correct": ARMS[0],
    "memory_off": ARMS[1],
    "wrong_entity": ARMS[3],
    "shuffled_binding": ARMS[2],
}


_RESERVOIR_SEED = 20260806


_REPRESENTATIVE_COUNT = 2


_GAIN = 0.25


def _mean(rows: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    if not rows:
        _fail("control_invalid")
    return _normalize(math.fsum(row[index] for row in rows) for index in range(DIMENSION))


def _select_reservoir(
    query_token: str,
    episodes: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    seed = _RESERVOIR_SEED.to_bytes(8, "big")
    return tuple(sorted(
        episodes,
        key=lambda item: (
            hashlib.sha256(b"icassp-reservoir-v1\0" + seed + query_token.encode() + b"\0" + item[0].encode()).digest(),
            item[0].encode(),
        ),
    )[:_REPRESENTATIVE_COUNT])


def _select_kcenter(
    episodes: tuple[tuple[str, int], ...],
    features: dict[str, tuple[float, ...]],
) -> tuple[tuple[str, int], ...]:
    if len(episodes) <= _REPRESENTATIVE_COUNT:
        return episodes
    centroid = _mean(tuple(features[token] for token, _ in episodes))
    first = min(episodes, key=lambda item: (-_dot(features[item[0]], centroid), item[0].encode()))
    selected = [first]
    while len(selected) < _REPRESENTATIVE_COUNT:
        remaining = tuple(item for item in episodes if item not in selected)
        chosen = min(remaining, key=lambda item: (
            max(_dot(features[item[0]], features[other[0]]) for other in selected),
            item[0].encode(),
        ))
        selected.append(chosen)
    return tuple(selected)


def _classic_rankings(supply: ReaderInput, baseline: str) -> tuple[Ranking, ...]:
    features = {item.feature_token: item.values for item in supply.features}
    episode_features = {item.episode_token: features[item.feature_token] for item in supply.episode_features}
    rankings: list[Ranking] = []
    for query in supply.queries:
        query_vector = _mean(tuple(features[token] for token in query.query_feature_tokens))
        applications = {item.arm: item for item in query.applications}
        for arm in CONTROL_ARMS:
            source_arm = _SOURCE_ARMS[arm]
            if arm == "memory_off":
                fused = query_vector
            else:
                episodes = decode_envelope(applications[source_arm].envelope)
                if baseline == "reservoir":
                    episodes = _select_reservoir(query.query_token, episodes)
                elif baseline == "kcenter":
                    episodes = _select_kcenter(episodes, episode_features)
                memory = _mean(tuple(episode_features[token] for token, _ in episodes))
                fused = _normalize((1.0 - _GAIN) * query_vector[index] + _GAIN * memory[index] for index in range(DIMENSION))
            scored = tuple(
                (_micros(_dot(fused, features[item.feature_token])), item.candidate_token)
                for item in query.candidates
            )
            ordered = tuple(sorted(scored, key=lambda item: (-item[0], item[1].encode())))
            rankings.append(Ranking(
                query_token=query.query_token,
                profile_id=baseline,
                arm=arm,
                ranked_candidate_tokens=tuple(item[1] for item in ordered),
                ranked_score_micros=tuple(item[0] for item in ordered),
            ))
    return tuple(rankings)


def run_readers(supply: ReaderInput, profiles: tuple[Profile, ...]) -> ReaderOutput:
    """Run methods without evaluator gold; returned output is safe for a later join."""
    if type(supply) is not ReaderInput or type(profiles) is not tuple:
        _fail("input_invalid")
    rankings: list[Ranking] = []
    for baseline in BASELINES[:3]:
        rankings.extend(_classic_rankings(supply, baseline))
    erm = run_erm_profiles(supply, profiles)
    selected = tuple(item for item in erm.rankings if item.profile_id == "balanced-recency")
    if len(selected) != len(supply.queries) * len(ARMS):
        _fail("input_invalid")
    for item in selected:
        if item.arm in _SOURCE_ARMS.values():
            arm = next(name for name, source in _SOURCE_ARMS.items() if source == item.arm)
            rankings.append(dataclasses.replace(item, profile_id="erm", arm=arm))
    result = ReaderOutput(rankings=tuple(rankings))
    expected = len(supply.queries) * len(BASELINES) * len(CONTROL_ARMS)
    if len(result.rankings) != expected:
        _fail("determinism_mismatch")
    return result
