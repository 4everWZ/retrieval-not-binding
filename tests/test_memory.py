from __future__ import annotations
import dataclasses
import hashlib
import unittest
from pathlib import Path
from retrieval_not_binding.protocol import (
    ARMS, DIMENSION, ENVELOPE_SIZE_BYTES, MemoryApplication, CandidateRef,
    EpisodeRef, FeatureVector, Query, ReaderInput, encode_envelope, decode_envelope,
)
from retrieval_not_binding.memory import run_erm_profiles
from retrieval_not_binding.evaluation import (
    ProfileScore, GoldRow, QueryOutcome, evaluate_selected_profiles, select_profiles,
)
from retrieval_not_binding.statistics import bootstrap_summary
from retrieval_not_binding.settings import load_config, profiles_from_config

ROOT = Path(__file__).resolve().parents[1]

def config():
    return load_config(ROOT / "configs/benchmark.json")

def token(kind: str, index: int) -> str:
    return f"{kind}-run:" + hashlib.sha256(f"{kind}/{index}".encode()).hexdigest()


def unit(index: int) -> tuple[float, ...]:
    return tuple(1.0 if item == index else 0.0 for item in range(DIMENSION))


def supply() -> ReaderInput:
    features = tuple(FeatureVector(feature_token=token("feature", index), values=unit(index)) for index in range(12))
    candidates = tuple(CandidateRef(candidate_token=token("candidate", index), feature_token=token("feature", index)) for index in range(8))
    episode_features = (
        EpisodeRef(episode_token=token("episode", 0), feature_token=token("feature", 8)),
        EpisodeRef(episode_token=token("episode", 1), feature_token=token("feature", 1)),
        EpisodeRef(episode_token=token("episode", 2), feature_token=token("feature", 9)),
        EpisodeRef(episode_token=token("episode", 3), feature_token=token("feature", 10)),
    )
    primary = ((token("episode", 0), 1), (token("episode", 1), 9))
    shuffled = ((primary[1][0], primary[0][1]), (primary[0][0], primary[1][1]))
    wrong = ((token("episode", 2), 1), (token("episode", 3), 9))
    raw = (*primary, primary[0])
    rows = []
    for query_index in range(2):
        episode_sets = (primary, primary, shuffled, wrong, raw)
        applications = tuple(MemoryApplication(
            arm=arm,
            envelope=encode_envelope(episodes, reserve_seed=hashlib.sha256(f"{query_index}/{arm}".encode()).digest()),
        ) for arm, episodes in zip(ARMS, episode_sets, strict=True))
        rows.append(Query(
            query_token=token("query", query_index), query_feature_tokens=(token("feature", query_index),),
            candidates=candidates, applications=applications,
        ))
    return ReaderInput(features=features, episode_features=episode_features, queries=tuple(rows))


class MemoryTests(unittest.TestCase):

    def test_fixed_envelope_round_trip_and_duplicate_are_explicit(self) -> None:
        one = ((token('episode', 0), 9),)
        payload = encode_envelope(one, reserve_seed=b'a' * 32)
        raw = encode_envelope((*one, one[0]), reserve_seed=b'b' * 32)
        self.assertEqual((len(payload), len(raw)), (ENVELOPE_SIZE_BYTES, ENVELOPE_SIZE_BYTES))
        self.assertEqual(decode_envelope(payload), one)
        self.assertEqual(decode_envelope(raw), (*one, one[0]))
        self.assertTrue(decode_envelope(payload)[0][0].startswith('episode-run:'))
        self.assertFalse(decode_envelope(payload)[0][0].startswith('feature-run:'))

    def test_method_has_no_gold_and_reuses_identical_off_ranking_across_profiles(self) -> None:
        value = supply()
        output = run_erm_profiles(value, profiles_from_config(config()))
        self.assertNotIn('gold', Query.__dataclass_fields__)
        self.assertNotIn('source', Query.__dataclass_fields__)
        self.assertEqual(len(output.rankings), 2 * 4 * 5)
        off = [item for item in output.rankings if item.query_token == token('query', 0) and item.arm == ARMS[1]]
        self.assertEqual(len({item.ranked_candidate_tokens for item in off}), 1)
        self.assertEqual(len({item.ranked_score_micros for item in off}), 1)
        primary = next((item for item in output.rankings if item.query_token == token('query', 0) and item.profile_id == 'balanced-recency' and (item.arm == ARMS[0])))
        shuffled = next((item for item in output.rankings if item.query_token == token('query', 0) and item.profile_id == 'balanced-recency' and (item.arm == ARMS[2])))
        self.assertNotEqual(primary.ranked_score_micros, shuffled.ranked_score_micros)

    def test_profile_selector_uses_only_aggregate_receipts_and_stable_tie_break(self) -> None:
        profiles = profiles_from_config(config())
        receipts = []
        for fold in range(4):
            for index, profile in enumerate(profiles):
                receipts.append(ProfileScore(held_fold=fold, profile_id=profile.profile_id, fit_seen_query_count=96, correct_count=80 + int(index == 1), memory_off_correct_count=78, rescue_count=3, harm_count=1))
        selected = select_profiles(tuple(receipts), profiles)
        self.assertEqual({item.profile_id for item in selected}, {'balanced-recency'})
        tied = tuple((dataclasses.replace(item, correct_count=80) for item in receipts))
        self.assertEqual({item.profile_id for item in select_profiles(tied, profiles)}, {'content-strong'})

    def test_held_evaluator_uses_only_the_profile_selected_for_each_fold(self) -> None:
        value = supply()
        profiles = profiles_from_config(config())
        output = run_erm_profiles(value, profiles)
        gold = tuple((GoldRow(query_token=token('query', index), entity_cluster_token=token('cluster', index), gold_candidate_token=token('candidate', index), fold=index, source_id='gso', pseudo_unseen=False, query_kind='view', memory_count=2) for index in range(2)))
        receipts = tuple((ProfileScore(held_fold=fold, profile_id=profile.profile_id, fit_seen_query_count=1, correct_count=int(profile.profile_id == 'balanced-recency'), memory_off_correct_count=0, rescue_count=1, harm_count=0) for fold in range(4) for profile in profiles))
        outcomes = evaluate_selected_profiles(output, gold, select_profiles(receipts, profiles))
        self.assertEqual({item.profile_id for item in outcomes}, {'balanced-recency'})
        self.assertEqual(len(outcomes), 2)

    def test_bootstrap_resamples_complete_entity_clusters_by_source(self) -> None:
        outcomes = []
        for source_index, source in enumerate(('gso', 'uco3d')):
            for entity in range(16):
                for query in range(4):
                    rescue = entity == 0 and query == 0
                    ranks = {arm: 1 for arm in ARMS}
                    ranks[ARMS[1]] = 2 if rescue else 1
                    outcomes.append(QueryOutcome(query_token=token('query', source_index * 64 + entity * 4 + query), entity_cluster_token=token('cluster', source_index * 16 + entity), fold=entity % 4, source_id=source, pseudo_unseen=False, query_kind=('view', 'occlusion', 'environment', 'detail')[query], memory_count=1 + query, profile_id='content-strong', ranks_by_arm=tuple(((arm, ranks[arm]) for arm in ARMS))))
        value = bootstrap_summary(tuple(outcomes), rounds=2000, seed=20260803)
        self.assertEqual(value.defined_rounds, 2000)
        self.assertEqual(value, bootstrap_summary(tuple(outcomes), rounds=2000, seed=20260803))
        self.assertGreater(value.point_delta_micros, 0)
