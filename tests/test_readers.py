import unittest
from retrieval_not_binding.protocol import ReaderInput
from retrieval_not_binding.evaluation import GoldRow
from retrieval_not_binding.settings import profiles_from_config
from retrieval_not_binding.readers import BASELINES, CONTROL_ARMS, run_readers
from retrieval_not_binding.evaluation import summarize_readers
from tests.test_memory import supply, token, config

class ReaderTests(unittest.TestCase):

    def test_methods_run_without_gold_and_classic_controls_ignore_position_shuffle(self) -> None:
        value = supply()
        profiles = profiles_from_config(config())
        output = run_readers(value, profiles)
        self.assertEqual(len(output.rankings), len(value.queries) * len(BASELINES) * len(CONTROL_ARMS))
        lookup = {(item.query_token, item.profile_id, item.arm): item for item in output.rankings}
        for query in value.queries:
            for baseline in BASELINES[:3]:
                self.assertEqual(lookup[query.query_token, baseline, 'correct'].ranked_candidate_tokens, lookup[query.query_token, baseline, 'shuffled_binding'].ranked_candidate_tokens)

    def test_evaluator_only_oracle_detects_nontrivial_shuffle(self) -> None:
        value = supply()
        profiles = profiles_from_config(config())
        output = run_readers(value, profiles)
        gold = tuple((GoldRow(query_token=token('query', index), entity_cluster_token=token('cluster', index), gold_candidate_token=token('candidate', index), fold=index, source_id='gso' if index == 0 else 'uco3d', pseudo_unseen=False, query_kind='view', memory_count=2) for index in range(2)))
        summaries, oracle = summarize_readers(output, gold, value)
        self.assertEqual(tuple((item.baseline for item in summaries)), BASELINES)
        self.assertTrue(all((item.correct_minus_shuffled_count == 0 for item in summaries[:3])))
        self.assertEqual((oracle.eligible_query_count, oracle.correct_mapping_correct_count), (2, 2))
        self.assertEqual(oracle.shuffled_mapping_correct_count, 0)
        self.assertEqual(oracle.status, 'ready_protocol_sanity_control_v1')
