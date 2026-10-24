import unittest

from retrieval_not_binding.preprocessing import ten_even_temporal_bin_indices
from retrieval_not_binding.retrieval import CandidateVector, rank_candidates
from tests.test_memory import unit


class CandidateRetrievalTests(unittest.TestCase):
    def test_fixed_gallery_is_ranked_deterministically_before_top_eight(self):
        candidates = tuple(CandidateVector(entity_handle=f"entity-{index:02}", values=unit(index))
                           for index in range(24))
        args = dict(query_handle="query:" + "a" * 64, source_id="gso", query_kind="view",
                    query_views=(unit(3),))
        first = rank_candidates(**args, candidates=candidates)
        second = rank_candidates(**args, candidates=tuple(reversed(candidates)))
        self.assertEqual(first, second)
        self.assertEqual(first.ranked_entity_handles[0], "entity-03")
        self.assertEqual(len(first.ranked_entity_handles[:8]), 8)

    def test_native_frames_include_endpoints_without_resizing(self):
        self.assertEqual(ten_even_temporal_bin_indices(10), tuple(range(10)))
        self.assertEqual(ten_even_temporal_bin_indices(100), tuple(range(0, 100, 11)))
