import dataclasses
import hashlib
import unittest
from collections import Counter
from retrieval_not_binding.cohort import (
    CohortEntity, CohortSource, CohortSelection, CohortError, ObservationRecord,
    QUERY_KINDS, build_cohort_views, observation_handle, method_views_bytes,
)

def selection():
    return CohortSelection(sources=tuple(CohortSource(source_id=source, selected=tuple(
        CohortEntity(source_id=source,
            entity_handle="entity:" + hashlib.sha256(f"{source}/{index}".encode()).hexdigest(),
            role="seen-memory" if index < 16 else "pseudo-unseen-query-only")
        for index in range(24))) for source in ("gso", "uco3d")))

def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def observations(report):
    result = []
    for source in report.sources:
        for entity in source.selected:
            ordinals = range(10) if entity.role == "seen-memory" else range(6, 10)
            for ordinal in ordinals:
                parent = digest(f"parent:{entity.entity_handle}:{ordinal}")
                content = digest(f"content:{entity.entity_handle}:{ordinal}")
                result.append(
                    ObservationRecord(
                        observation_handle=observation_handle(
                            entity_handle=entity.entity_handle,
                            ordinal=ordinal,
                            parent_sha256=parent,
                            content_sha256=content,
                        ),
                        entity_handle=entity.entity_handle,
                        source_id=source.source_id,
                        entity_role=entity.role,
                        ordinal=ordinal,
                        observation_role="memory" if ordinal < 6 else "query",
                        query_kind=None if ordinal < 6 else QUERY_KINDS[ordinal - 6],
                        parent_sha256=parent,
                        content_sha256=content,
                        processing_version="fixture-v1",
                    )
                )
    return tuple(result)


class CohortTests(unittest.TestCase):

    def setUp(self):
        self.selection = selection()
        self.observations = observations(self.selection)

    def test_exact_counts_balance_controls_and_gold_blind_method_view(self):
        views = build_cohort_views(self.selection, self.observations)
        self.assertEqual(len(self.observations), 384)
        self.assertEqual(len(views.method_queries), 192)
        self.assertEqual(len(views.evaluator_queries), 192)
        self.assertEqual(sum((not item.pseudo_unseen for item in views.evaluator_queries)), 128)
        self.assertEqual(sum((item.pseudo_unseen for item in views.evaluator_queries)), 64)
        cells = Counter(((item.source_id, item.memory_count, item.recency, item.spacing) for item in views.evaluator_queries if not item.pseudo_unseen))
        self.assertEqual(len(cells), 32)
        self.assertEqual(set(cells.values()), {4})
        for method in views.method_queries:
            self.assertEqual(method.memory_off_slots, len(method.correct))
            self.assertEqual(len(method.raw_multiset), len(method.correct) + 1)
            self.assertEqual({item.event_position for item in method.wrong_entity}, {item.event_position for item in method.correct})
            if len(method.correct) > 1:
                self.assertNotEqual(method.shuffled_binding, method.correct)
        payload = method_views_bytes(views)
        forbidden_fields = (b'source_id', b'entity_handle', b'query_kind', b'pseudo_unseen', b'cell_index')
        for forbidden in forbidden_fields:
            self.assertNotIn(forbidden, payload)

    def test_input_order_is_irrelevant(self):
        first = build_cohort_views(self.selection, self.observations)
        second = build_cohort_views(self.selection, tuple(reversed(self.observations)))
        self.assertEqual(first, second)

    def test_duplicate_content_or_parent_fails_closed(self):
        rows = list(self.observations)
        original = rows[1]
        rows[1] = dataclasses.replace(original, parent_sha256=rows[0].parent_sha256, observation_handle=observation_handle(entity_handle=original.entity_handle, ordinal=original.ordinal, parent_sha256=rows[0].parent_sha256, content_sha256=original.content_sha256))
        with self.assertRaisesRegex(CohortError, 'content_duplicate'):
            build_cohort_views(self.selection, tuple(rows))

    def test_missing_or_cross_role_observation_fails_closed(self):
        with self.assertRaisesRegex(CohortError, 'observation_invalid'):
            build_cohort_views(self.selection, self.observations[:-1])
        row = self.observations[0]
        with self.assertRaisesRegex(CohortError, 'observation_invalid'):
            ObservationRecord(observation_handle=row.observation_handle, entity_handle=row.entity_handle, source_id=row.source_id, entity_role=row.entity_role, ordinal=row.ordinal, observation_role='query', query_kind='view', parent_sha256=row.parent_sha256, content_sha256=row.content_sha256, processing_version=row.processing_version)

    def test_raw_multiset_duplicates_first_observation(self):
        views = build_cohort_views(self.selection, self.observations)
        for query in views.method_queries:
            self.assertEqual(query.raw_multiset[:-1], query.correct)
            self.assertEqual(query.raw_multiset[-1], query.correct[0])
