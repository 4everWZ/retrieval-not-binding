import json
import unittest

from retrieval_not_binding.dataset import prepare_inputs
from retrieval_not_binding.io import reader_input_dict
from retrieval_not_binding.protocol import BindingError, decode_envelope
from tests.test_cohort import observations, selection
from tests.test_memory import unit


class DatasetTests(unittest.TestCase):
    def test_assembly_preserves_fold_isolation_candidates_and_first_duplicate(self):
        cohort = selection()
        rows = observations(cohort)
        entity_indices = {entity.entity_handle: index
                          for index, entity in enumerate(entity for source in cohort.sources for entity in source.selected)}
        features = {row.content_sha256: unit(entity_indices[row.entity_handle]) for row in rows}
        views = {row.observation_handle: (row.content_sha256,) * (5 if row.query_kind == "detail" else 1)
                 for row in rows if row.observation_role == "query"}
        gallery = {(row.source_id, row.query_kind, row.entity_handle): row.content_sha256
                   for row in rows if row.observation_role == "query"}
        inputs, gold = prepare_inputs(cohort, rows, features, views, gallery, token_nonce=b"n" * 32)
        self.assertEqual((len(inputs.queries), len(gold)), (192, 192))
        by_query = {row.query_token: row for row in gold}
        folds_by_entity = {}
        for query in inputs.queries:
            row = by_query[query.query_token]
            self.assertIn(row.gold_candidate_token, [item.candidate_token for item in query.candidates])
            self.assertEqual(len(query.query_feature_tokens), 5 if row.query_kind == "detail" else 1)
            previous = folds_by_entity.setdefault(row.entity_cluster_token, row.fold)
            self.assertEqual(previous, row.fold)
            arms = {app.arm: decode_envelope(app.envelope) for app in query.applications}
            self.assertEqual(arms["raw_multiset"], (*arms["correct"], arms["correct"][0]))
            self.assertEqual({len(app.envelope) for app in query.applications}, {178})
        self.assertEqual(len(folds_by_entity), 48)
        encoded = json.dumps(reader_input_dict(inputs))
        self.assertNotIn('"gold_candidate_token"', encoded)
        self.assertNotIn('"source_id"', encoded)
        self.assertNotIn('"fold"', encoded)

    def test_preparation_requires_a_replayable_nonce(self):
        with self.assertRaises(BindingError):
            prepare_inputs(selection(), (), {}, {}, {}, token_nonce=b"short")
