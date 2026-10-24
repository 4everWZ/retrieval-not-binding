"""Separate feature-only reader inputs, evaluator labels, and saved rankings."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .evaluation import GoldRow
from .protocol import (
    BindingError, CandidateRef, EpisodeRef, FeatureVector, MemoryApplication,
    Query, Ranking, ReaderInput, ReaderOutput,
)
from .serialization import canonical_bytes


def _keys(value: dict, expected: set[str]) -> None:
    if type(value) is not dict or set(value) != expected:
        raise BindingError("input_schema_invalid")


def reader_input_dict(value: ReaderInput) -> dict:
    return {
        "schema_version": "retrieval_not_binding.reader_input.v1",
        "features": [dataclasses.asdict(item) for item in value.features],
        "episode_features": [dataclasses.asdict(item) for item in value.episode_features],
        "queries": [{
            "query_token": query.query_token,
            "query_feature_tokens": query.query_feature_tokens,
            "candidates": [dataclasses.asdict(item) for item in query.candidates],
            "applications": [{"arm": app.arm, "envelope_hex": app.envelope.hex()}
                             for app in query.applications],
        } for query in value.queries],
    }


def parse_reader_input(raw: dict) -> ReaderInput:
    _keys(raw, {"schema_version", "features", "episode_features", "queries"})
    if raw["schema_version"] != "retrieval_not_binding.reader_input.v1":
        raise BindingError("input_schema_invalid")
    features = []
    for item in raw["features"]:
        _keys(item, {"feature_token", "values"})
        features.append(FeatureVector(feature_token=item["feature_token"],
                                      values=tuple(float(x) for x in item["values"])))
    episodes = []
    for item in raw["episode_features"]:
        _keys(item, {"episode_token", "feature_token"})
        episodes.append(EpisodeRef(**item))
    queries = []
    for item in raw["queries"]:
        _keys(item, {"query_token", "query_feature_tokens", "candidates", "applications"})
        candidates = []
        for candidate in item["candidates"]:
            _keys(candidate, {"candidate_token", "feature_token"})
            candidates.append(CandidateRef(**candidate))
        applications = []
        for app in item["applications"]:
            _keys(app, {"arm", "envelope_hex"})
            applications.append(MemoryApplication(arm=app["arm"],
                                                   envelope=bytes.fromhex(app["envelope_hex"])))
        queries.append(Query(query_token=item["query_token"],
                             query_feature_tokens=tuple(item["query_feature_tokens"]),
                             candidates=tuple(candidates), applications=tuple(applications)))
    return ReaderInput(features=tuple(features), episode_features=tuple(episodes), queries=tuple(queries))


def load_reader_input(path: str | Path) -> ReaderInput:
    return parse_reader_input(json.loads(Path(path).read_text(encoding="utf-8")))


def load_gold(path: str | Path) -> tuple[GoldRow, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _keys(raw, {"schema_version", "rows"})
    if raw["schema_version"] != "retrieval_not_binding.gold.v1":
        raise BindingError("gold_schema_invalid")
    rows = tuple(GoldRow(**row) for row in raw["rows"])
    if len({row.query_token for row in rows}) != len(rows):
        raise BindingError("gold_query_duplicate")
    entity_roles = {}
    for row in rows:
        role = (row.fold, row.source_id, row.pseudo_unseen)
        if entity_roles.setdefault(row.entity_cluster_token, role) != role:
            raise BindingError("entity_fold_isolation_invalid")
    return rows


def parse_rankings(raw: dict) -> ReaderOutput:
    _keys(raw, {"rankings"})
    fields = {field.name for field in dataclasses.fields(Ranking)}
    rankings = []
    for item in raw["rankings"]:
        _keys(item, fields)
        rankings.append(Ranking(**{**item,
            "ranked_candidate_tokens": tuple(item["ranked_candidate_tokens"]),
            "ranked_score_micros": tuple(item["ranked_score_micros"]),
        }))
    return ReaderOutput(rankings=tuple(rankings))


def write_json(path: str | Path, value: object) -> None:
    # Preserve existing outputs; a new invocation must choose a fresh path.
    with Path(path).open("xb") as stream:
        stream.write(canonical_bytes(value))
