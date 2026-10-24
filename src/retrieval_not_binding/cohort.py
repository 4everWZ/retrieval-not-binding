"""Frozen cohort schedule and matched memory interventions."""
from __future__ import annotations
import dataclasses
import hashlib
import re
from collections import Counter, defaultdict
from .serialization import canonical_bytes

@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CohortEntity:
    source_id: str
    entity_handle: str
    role: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CohortSource:
    source_id: str
    selected: tuple[CohortEntity, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CohortSelection:
    sources: tuple[CohortSource, ...]


# Sampling namespaces retain the published observation and episode assignments.
SOURCES = ("gso", "uco3d")


QUERY_KINDS = ("view", "occlusion", "environment", "detail")


MEMORY_COUNTS = (1, 2, 3, 4)


RECENCIES = ("near", "far")


SPACINGS = ("clustered", "distributed")


SEEN_PER_SOURCE = 16


UNSEEN_PER_SOURCE = 8


SELECTED_PER_SOURCE = SEEN_PER_SOURCE + UNSEEN_PER_SOURCE


_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


_ENTITY_HANDLE = re.compile(r"entity:[0-9a-f]{64}\Z", re.ASCII)


_OBSERVATION_HANDLE = re.compile(r"observation:[0-9a-f]{64}\Z", re.ASCII)


_QUERY_HANDLE = re.compile(r"query:[0-9a-f]{64}\Z", re.ASCII)


_OBSERVATION_DOMAIN = b"reencounter-episodic-development-observation-v1\0"


_QUERY_DOMAIN = b"reencounter-episodic-development-query-v1\0"


_EPISODE_ORDER_DOMAIN = b"reencounter-episodic-development-episode-order-v1\0"


_VIEWS_SCHEMA = "retrieval_not_binding.cohort_views.v1"


class CohortError(RuntimeError):
    """Stable fail-closed cohort error."""

    def __init__(self, code: str) -> None:
        allowed = {
            "input_invalid",
            "license_invalid",
            "exclusion_invalid",
            "cohort_insufficient",
            "observation_invalid",
            "content_duplicate",
            "schedule_invalid",
            "control_invalid",
        }
        self.code = code if code in allowed else "input_invalid"
        super().__init__(self.code)


def _fail(code: str) -> None:
    raise CohortError(code)


def observation_handle(
    *, entity_handle: str, ordinal: int, parent_sha256: str, content_sha256: str
) -> str:
    if (
        type(entity_handle) is not str
        or _ENTITY_HANDLE.fullmatch(entity_handle) is None
        or type(ordinal) is not int
        or not 0 <= ordinal <= 9
        or type(parent_sha256) is not str
        or _SHA256.fullmatch(parent_sha256) is None
        or type(content_sha256) is not str
        or _SHA256.fullmatch(content_sha256) is None
    ):
        _fail("observation_invalid")
    payload = (
        _OBSERVATION_DOMAIN
        + entity_handle.encode("ascii")
        + b"\0"
        + bytes((ordinal,))
        + bytes.fromhex(parent_sha256)
        + bytes.fromhex(content_sha256)
    )
    return "observation:" + hashlib.sha256(payload).hexdigest()


def query_handle(observation_id: str) -> str:
    if type(observation_id) is not str or _OBSERVATION_HANDLE.fullmatch(observation_id) is None:
        _fail("observation_invalid")
    return "query:" + hashlib.sha256(
        _QUERY_DOMAIN + observation_id.encode("ascii")
    ).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ObservationRecord:
    observation_handle: str
    entity_handle: str
    source_id: str
    entity_role: str
    ordinal: int
    observation_role: str
    query_kind: str | None
    parent_sha256: str
    content_sha256: str
    processing_version: str

    def __post_init__(self) -> None:
        if (
            self.source_id not in SOURCES
            or self.entity_role not in {"seen-memory", "pseudo-unseen-query-only"}
            or self.observation_role not in {"memory", "query"}
            or type(self.processing_version) is not str
            or not self.processing_version
            or self.observation_handle
            != observation_handle(
                entity_handle=self.entity_handle,
                ordinal=self.ordinal,
                parent_sha256=self.parent_sha256,
                content_sha256=self.content_sha256,
            )
        ):
            _fail("observation_invalid")
        if self.observation_role == "memory":
            if (
                self.entity_role != "seen-memory"
                or not 0 <= self.ordinal <= 5
                or self.query_kind is not None
            ):
                _fail("observation_invalid")
        elif self.ordinal not in (6, 7, 8, 9) or self.query_kind != QUERY_KINDS[self.ordinal - 6]:
            _fail("observation_invalid")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class HistoryObservation:
    episode_handle: str
    event_position: int
    content_sha256: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QueryHistory:
    query_handle: str
    query_content_sha256: str
    correct: tuple[HistoryObservation, ...]
    memory_off_slots: int
    shuffled_binding: tuple[HistoryObservation, ...]
    wrong_entity: tuple[HistoryObservation, ...]
    raw_multiset: tuple[HistoryObservation, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QueryMetadata:
    query_handle: str
    source_id: str
    entity_handle: str
    pseudo_unseen: bool
    query_kind: str
    cell_index: int | None
    memory_count: int
    recency: str | None
    spacing: str | None
    correct_episode_handles: tuple[str, ...]
    wrong_entity_handle: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CohortViews:
    schema_version: str
    method_queries: tuple[QueryHistory, ...]
    evaluator_queries: tuple[QueryMetadata, ...]


def _cells() -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (count, recency, spacing)
        for count in MEMORY_COUNTS
        for recency in RECENCIES
        for spacing in SPACINGS
    )


def _event_positions(count: int, recency: str, spacing: str) -> tuple[int, ...]:
    if recency == "near" and spacing == "clustered":
        result = tuple(range(10 - count, 10))
    elif recency == "far" and spacing == "clustered":
        result = tuple(range(1, 1 + count))
    elif recency == "near" and spacing == "distributed":
        result = (0, 3, 6, 9)[-count:]
    elif recency == "far" and spacing == "distributed":
        result = (0, 1, 3, 4)[-count:]
    else:
        _fail("schedule_invalid")
    if len(result) != count or tuple(sorted(result)) != result or result[-1] >= 10:
        _fail("schedule_invalid")
    return result


def _episode_subset(
    memory: tuple[ObservationRecord, ...], query_id: str, count: int
) -> tuple[ObservationRecord, ...]:
    ranked = sorted(
        memory,
        key=lambda item: hashlib.sha256(
            _EPISODE_ORDER_DOMAIN
            + query_id.encode("ascii")
            + item.observation_handle.encode("ascii")
        ).digest(),
    )
    return tuple(ranked[:count])


def _as_method_episodes(
    observations: tuple[ObservationRecord, ...], positions: tuple[int, ...]
) -> tuple[HistoryObservation, ...]:
    if len(observations) != len(positions):
        _fail("schedule_invalid")
    return tuple(
        HistoryObservation(
            episode_handle=item.observation_handle,
            event_position=position,
            content_sha256=item.content_sha256,
        )
        for item, position in zip(observations, positions, strict=True)
    )


def build_cohort_views(
    selection: CohortSelection, observations: tuple[ObservationRecord, ...]
) -> CohortViews:
    """Validate materialization commitments and build paired gold-blind controls."""

    if type(selection) is not CohortSelection or type(observations) is not tuple:
        _fail("input_invalid")
    selected = tuple(item for source in selection.sources for item in source.selected)
    by_handle = {item.entity_handle: item for item in selected}
    if len(by_handle) != 48:
        _fail("input_invalid")
    grouped: dict[str, list[ObservationRecord]] = defaultdict(list)
    all_content: set[str] = set()
    all_parent: set[str] = set()
    all_observation_handles: set[str] = set()
    for item in observations:
        if type(item) is not ObservationRecord or item.entity_handle not in by_handle:
            _fail("observation_invalid")
        entity = by_handle[item.entity_handle]
        if item.source_id != entity.source_id or item.entity_role != entity.role:
            _fail("observation_invalid")
        if (
            item.content_sha256 in all_content
            or item.parent_sha256 in all_parent
            or item.observation_handle in all_observation_handles
        ):
            _fail("content_duplicate")
        all_content.add(item.content_sha256)
        all_parent.add(item.parent_sha256)
        all_observation_handles.add(item.observation_handle)
        grouped[item.entity_handle].append(item)
    if len(observations) != 384:
        _fail("observation_invalid")
    memory_by_entity: dict[str, tuple[ObservationRecord, ...]] = {}
    query_by_entity: dict[str, tuple[ObservationRecord, ...]] = {}
    for entity in selected:
        rows = tuple(sorted(grouped[entity.entity_handle], key=lambda item: item.ordinal))
        expected_ordinals = tuple(range(10)) if entity.role == "seen-memory" else (6, 7, 8, 9)
        if tuple(item.ordinal for item in rows) != expected_ordinals:
            _fail("observation_invalid")
        memory_by_entity[entity.entity_handle] = tuple(
            item for item in rows if item.observation_role == "memory"
        )
        query_by_entity[entity.entity_handle] = tuple(
            item for item in rows if item.observation_role == "query"
        )

    preliminary: list[
        tuple[CohortEntity, ObservationRecord, int | None, tuple[int, str, str] | None]
    ] = []
    for source_report in selection.sources:
        seen = tuple(item for item in source_report.selected if item.role == "seen-memory")
        unseen = tuple(
            item
            for item in source_report.selected
            if item.role == "pseudo-unseen-query-only"
        )
        if len(seen) != SEEN_PER_SOURCE or len(unseen) != UNSEEN_PER_SOURCE:
            _fail("schedule_invalid")
        for rank, entity in enumerate(seen):
            for query_index, query in enumerate(query_by_entity[entity.entity_handle]):
                cell_index = (rank + 4 * query_index) % 16
                preliminary.append((entity, query, cell_index, _cells()[cell_index]))
        for entity in unseen:
            for query in query_by_entity[entity.entity_handle]:
                preliminary.append((entity, query, None, None))

    seen_by_cell: dict[
        tuple[str, int], list[tuple[CohortEntity, ObservationRecord]]
    ] = defaultdict(list)
    for entity, query, cell_index, _ in preliminary:
        if cell_index is not None:
            seen_by_cell[(entity.source_id, cell_index)].append((entity, query))
    if any(len(items) != 4 for items in seen_by_cell.values()) or len(seen_by_cell) != 32:
        _fail("schedule_invalid")
    wrong_provider: dict[str, str] = {}
    for items in seen_by_cell.values():
        ordered = sorted(items, key=lambda item: QUERY_KINDS.index(item[1].query_kind or ""))
        for index, (entity, query) in enumerate(ordered):
            provider = ordered[(index + 1) % len(ordered)][0]
            if provider.entity_handle == entity.entity_handle:
                _fail("control_invalid")
            wrong_provider[query.observation_handle] = provider.entity_handle

    method_queries: list[QueryHistory] = []
    evaluator_queries: list[QueryMetadata] = []
    seen_entities_by_source = {
        source.source_id: tuple(item for item in source.selected if item.role == "seen-memory")
        for source in selection.sources
    }
    for entity, query, cell_index, cell in preliminary:
        q_handle = query_handle(query.observation_handle)
        if cell is None:
            providers = seen_entities_by_source[entity.source_id]
            provider_index = int(q_handle[-8:], 16) % len(providers)
            provider = providers[provider_index]
            alternate = providers[(provider_index + 1) % len(providers)]
            count, recency, spacing = 4, "far", "distributed"
            correct_handles: tuple[str, ...] = ()
        else:
            count, recency, spacing = cell
            provider = entity
            alternate = by_handle[wrong_provider[query.observation_handle]]
            correct_handles = tuple(
                item.observation_handle
                for item in _episode_subset(memory_by_entity[entity.entity_handle], q_handle, count)
            )
        positions = _event_positions(count, recency, spacing)
        primary_source = _episode_subset(memory_by_entity[provider.entity_handle], q_handle, count)
        primary = _as_method_episodes(primary_source, positions)
        shuffled = (
            tuple(
                dataclasses.replace(item, episode_handle=primary[-1 - index].episode_handle,
                                    content_sha256=primary[-1 - index].content_sha256)
                for index, item in enumerate(primary)
            )
            if count > 1
            else primary
        )
        wrong_source = _episode_subset(memory_by_entity[alternate.entity_handle], q_handle, count)
        wrong = _as_method_episodes(wrong_source, positions)
        raw_multiset = primary + (dataclasses.replace(primary[0]),)
        method_queries.append(
            QueryHistory(
                query_handle=q_handle,
                query_content_sha256=query.content_sha256,
                correct=primary,
                memory_off_slots=len(primary),
                shuffled_binding=shuffled,
                wrong_entity=wrong,
                raw_multiset=raw_multiset,
            )
        )
        evaluator_queries.append(
            QueryMetadata(
                query_handle=q_handle,
                source_id=entity.source_id,
                entity_handle=entity.entity_handle,
                pseudo_unseen=entity.role == "pseudo-unseen-query-only",
                query_kind=query.query_kind or "",
                cell_index=cell_index,
                memory_count=count,
                recency=None if cell is None else recency,
                spacing=None if cell is None else spacing,
                correct_episode_handles=correct_handles,
                wrong_entity_handle=alternate.entity_handle,
            )
        )

    ordered_pairs = sorted(
        zip(method_queries, evaluator_queries, strict=True), key=lambda pair: pair[0].query_handle
    )
    method_tuple = tuple(pair[0] for pair in ordered_pairs)
    evaluator_tuple = tuple(pair[1] for pair in ordered_pairs)
    if len(method_tuple) != 192 or len({item.query_handle for item in method_tuple}) != 192:
        _fail("schedule_invalid")
    cell_counts = Counter(
        (item.source_id, item.memory_count, item.recency, item.spacing)
        for item in evaluator_tuple
        if not item.pseudo_unseen
    )
    if set(cell_counts.values()) != {4} or len(cell_counts) != 32:
        _fail("schedule_invalid")
    return CohortViews(
        schema_version=_VIEWS_SCHEMA,
        method_queries=method_tuple,
        evaluator_queries=evaluator_tuple,
    )


def method_views_bytes(views: CohortViews) -> bytes:
    if type(views) is not CohortViews:
        _fail("input_invalid")
    return canonical_bytes(tuple(dataclasses.asdict(item) for item in views.method_queries))


def evaluator_views_bytes(views: CohortViews) -> bytes:
    if type(views) is not CohortViews:
        _fail("input_invalid")
    return canonical_bytes(tuple(dataclasses.asdict(item) for item in views.evaluator_queries))
