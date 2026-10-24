"""Content- and age-weighted episodic residual memory (manuscript Eq. 3–5)."""
from __future__ import annotations
import math
from .protocol import (ARMS, DIMENSION, ReaderInput, ReaderOutput, Profile, Ranking, decode_envelope, _dot, _fail, _micros, _normalize)

def _query_vector(tokens: tuple[str, ...], features: dict[str, tuple[float, ...]]) -> tuple[float, ...]:
    rows = tuple(features[item] for item in tokens)
    return _normalize(math.fsum(row[index] for row in rows) / len(rows) for index in range(DIMENSION))


def _fused_query(
    query: tuple[float, ...], episodes: tuple[tuple[str, int], ...],
    features: dict[str, tuple[float, ...]], profile: Profile,
) -> tuple[float, ...]:
    if not episodes:
        _fail("control_invalid")
    rows = tuple(features[token] for token, _ in episodes)
    prototype = _normalize(math.fsum(row[index] for row in rows) for index in range(DIMENSION))
    maximum_age = max(10 - position for _, position in episodes)
    beta = profile.beta_micros / 1_000_000.0
    gamma = profile.gamma_micros / 1_000_000.0
    logits = tuple(beta * _dot(query, row) - gamma * ((10 - position) / maximum_age) for row, (_, position) in zip(rows, episodes, strict=True))
    maximum = max(logits)
    exps = tuple(math.exp(item - maximum) for item in logits); denominator = math.fsum(exps)
    weights = tuple(item / denominator for item in exps)
    alpha = profile.alpha_micros / 1_000_000.0
    memory = _normalize(
        prototype[index] + alpha * math.fsum(weight * (row[index] - prototype[index]) for weight, row in zip(weights, rows, strict=True))
        for index in range(DIMENSION)
    )
    gain = profile.lambda_micros / 1_000_000.0
    return _normalize((1.0 - gain) * query[index] + gain * memory[index] for index in range(DIMENSION))


def run_erm_profiles(supply: ReaderInput, profiles: tuple[Profile, ...]) -> ReaderOutput:
    if type(supply) is not ReaderInput or type(profiles) is not tuple or len(profiles) != 4:
        _fail("input_invalid")
    features = {item.feature_token: item.values for item in supply.features}
    episode_features = {
        item.episode_token: features[item.feature_token] for item in supply.episode_features
    }
    rankings = []
    for query in supply.queries:
        base_query = _query_vector(query.query_feature_tokens, features)
        for profile in profiles:
            for application in query.applications:
                episodes = decode_envelope(application.envelope)
                fused = base_query if application.arm == "memory_off" else _fused_query(
                    base_query, episodes, episode_features, profile
                )
                scored = tuple(
                    (_micros(_dot(fused, features[candidate.feature_token])), candidate.candidate_token)
                    for candidate in query.candidates
                )
                ordered = tuple(sorted(scored, key=lambda item: (-item[0], item[1].encode())))
                rankings.append(Ranking(
                    query_token=query.query_token, profile_id=profile.profile_id, arm=application.arm,
                    ranked_candidate_tokens=tuple(item[1] for item in ordered),
                    ranked_score_micros=tuple(item[0] for item in ordered),
                ))
    result = ReaderOutput(rankings=tuple(rankings))
    if len(result.rankings) != len(supply.queries) * len(profiles) * len(ARMS):
        _fail("determinism_mismatch")
    return result
