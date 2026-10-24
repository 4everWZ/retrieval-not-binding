"""Assemble frozen cohort features into separate reader and evaluator inputs."""
from __future__ import annotations

import hashlib

from .cohort import CohortSelection, ObservationRecord, build_cohort_views
from .evaluation import GoldRow
from .protocol import (
    ARMS, BindingError, CandidateRef, EpisodeRef, FeatureVector, MemoryApplication,
    Query, ReaderInput, encode_envelope,
)
from .retrieval import CandidateVector, rank_candidates


ENROLLMENT_KIND = {"view": "occlusion", "occlusion": "environment",
                   "environment": "detail", "detail": "view"}


def _opaque(domain: bytes, nonce: bytes, value: str, prefix: str) -> str:
    return prefix + hashlib.sha256(domain + nonce + b"\0" + value.encode()).hexdigest()


def _folds(selection: CohortSelection, nonce: bytes) -> dict[str, int]:
    result = {}
    for source in selection.sources:
        for role in ("seen-memory", "pseudo-unseen-query-only"):
            rows = tuple(item for item in source.selected if item.role == role)
            ordered = sorted(rows, key=lambda item: hashlib.sha256(
                b"erm-fold-v1\0" + nonce + item.entity_handle.encode()).digest())
            expected = 16 if role == "seen-memory" else 8
            if len(ordered) != expected:
                raise BindingError("fold_invalid")
            for index, item in enumerate(ordered):
                result[item.entity_handle] = index % 4
    return result


def prepare_inputs(
    selection: CohortSelection,
    observations: tuple[ObservationRecord, ...],
    features_by_content: dict[str, tuple[float, ...]],
    query_views: dict[str, tuple[str, ...]],
    enrollment_content: dict[tuple[str, str, str], str],
    *, token_nonce: bytes,
) -> tuple[ReaderInput, tuple[GoldRow, ...]]:
    """Use precomputed global features; never select new entities or download assets.

    ``query_views`` maps observation handles to one or five feature-content IDs.
    ``enrollment_content`` maps (source, condition, entity) to the modifier-off
    enrollment feature-content ID. The nonce must be retained for exact replay.
    """
    if type(token_nonce) is not bytes or len(token_nonce) != 32:
        raise BindingError("input_invalid")
    views = build_cohort_views(selection, observations)
    metadata = {row.query_handle: row for row in views.evaluator_queries}
    feature_tokens = {key: _opaque(b"erm-feature-v1", token_nonce, key, "feature-run:")
                      for key in features_by_content}
    entities = tuple(entity for source in selection.sources for entity in source.selected)
    candidate_tokens = {row.entity_handle: _opaque(b"erm-candidate-v1", token_nonce,
                        row.entity_handle, "candidate-run:") for row in entities}
    cluster_tokens = {row.entity_handle: _opaque(b"erm-cluster-v1", token_nonce,
                      row.entity_handle, "cluster-run:") for row in entities}
    history_content = {}
    for row in views.method_queries:
        for observation in (*row.correct, *row.shuffled_binding, *row.wrong_entity, *row.raw_multiset):
            previous = history_content.setdefault(observation.episode_handle, observation.content_sha256)
            if previous != observation.content_sha256:
                raise BindingError("input_invalid")
    episode_tokens = {key: _opaque(b"erm-episode-v1", token_nonce, key, "episode-run:")
                      for key in history_content}
    query_tokens = {row.query_handle: _opaque(b"erm-query-v1", token_nonce,
                    row.query_handle, "query-run:") for row in views.method_queries}
    folds = _folds(selection, token_nonce)
    query_observations = {(row.entity_handle, row.query_kind): row.observation_handle
                          for row in observations if row.observation_role == "query"}
    queries, gold = [], []
    for row in sorted(views.method_queries, key=lambda item: query_tokens[item.query_handle].encode()):
        label = metadata[row.query_handle]
        # query_views uses observation handles, while the reader uses opaque query tokens.
        content_ids = query_views[query_observations[(label.entity_handle, label.query_kind)]]
        expected_views = 5 if label.query_kind == "detail" else 1
        if len(content_ids) != expected_views:
            raise BindingError("input_invalid")
        gallery = tuple(CandidateVector(entity_handle=entity.entity_handle,
            values=features_by_content[enrollment_content[(label.source_id,
                ENROLLMENT_KIND[label.query_kind], entity.entity_handle)]])
            for entity in entities if entity.source_id == label.source_id)
        ranked = rank_candidates(query_handle=row.query_handle, source_id=label.source_id,
            query_kind=label.query_kind, query_views=tuple(features_by_content[key] for key in content_ids),
            candidates=gallery)
        applications = []
        for arm, history in zip(ARMS, (row.correct, row.correct, row.shuffled_binding,
                                      row.wrong_entity, row.raw_multiset), strict=True):
            references = tuple((episode_tokens[item.episode_handle], item.event_position) for item in history)
            seed = hashlib.sha256(token_nonce + query_tokens[row.query_handle].encode() + arm.encode()).digest()
            applications.append(MemoryApplication(arm=arm,
                envelope=encode_envelope(references, reserve_seed=seed)))
        queries.append(Query(query_token=query_tokens[row.query_handle],
            query_feature_tokens=tuple(feature_tokens[key] for key in content_ids),
            candidates=tuple(CandidateRef(candidate_token=candidate_tokens[entity],
                feature_token=feature_tokens[enrollment_content[(label.source_id,
                    ENROLLMENT_KIND[label.query_kind], entity)]])
                for entity in ranked.ranked_entity_handles[:8]), applications=tuple(applications)))
        gold.append(GoldRow(query_token=query_tokens[row.query_handle],
            entity_cluster_token=cluster_tokens[label.entity_handle],
            gold_candidate_token=candidate_tokens[label.entity_handle], fold=folds[label.entity_handle],
            source_id=label.source_id, pseudo_unseen=label.pseudo_unseen,
            query_kind=label.query_kind, memory_count=label.memory_count))
    reader_input = ReaderInput(
        features=tuple(FeatureVector(feature_token=feature_tokens[key], values=features_by_content[key])
                       for key in sorted(features_by_content, key=bytes.fromhex)),
        episode_features=tuple(EpisodeRef(episode_token=episode_tokens[key],
            feature_token=feature_tokens[history_content[key]])
            for key in sorted(history_content, key=lambda key: episode_tokens[key].encode())),
        queries=tuple(queries))
    return reader_input, tuple(sorted(gold, key=lambda row: row.query_token.encode()))
