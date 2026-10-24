"""Gold joins, paired contrasts, mapping oracle, and fold-isolated profile selection."""
from __future__ import annotations
import dataclasses
from .protocol import (ARMS, FOLD_COUNT, ReaderInput, ReaderOutput, Profile, Ranking, decode_envelope, _CLUSTER, _QUERY, _CANDIDATE, _KINDS, _SOURCES, _fail, _token)
from .readers import BASELINES, CONTROL_ARMS

@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class GoldRow:
    query_token: str
    entity_cluster_token: str
    gold_candidate_token: str
    fold: int
    source_id: str
    pseudo_unseen: bool
    query_kind: str
    memory_count: int

    def __post_init__(self) -> None:
        _token(self.query_token, _QUERY, "gold_binding_invalid")
        _token(self.gold_candidate_token, _CANDIDATE, "gold_binding_invalid")
        _token(self.entity_cluster_token, _CLUSTER, "gold_binding_invalid")
        if (
            type(self.fold) is not int or not 0 <= self.fold < FOLD_COUNT or self.source_id not in _SOURCES
            or type(self.pseudo_unseen) is not bool or self.query_kind not in _KINDS
            or type(self.memory_count) is not int or not 1 <= self.memory_count <= 4
        ):
            _fail("gold_binding_invalid")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProfileScore:
    held_fold: int
    profile_id: str
    fit_seen_query_count: int
    correct_count: int
    memory_off_correct_count: int
    rescue_count: int
    harm_count: int


def _ranking_map(output: ReaderOutput) -> dict[tuple[str, str, str], Ranking]:
    result = {(item.query_token, item.profile_id, item.arm): item for item in output.rankings}
    if len(result) != len(output.rankings):
        _fail("candidate_invalid")
    return result


def _correct(ranking: Ranking, gold: str) -> bool:
    return bool(ranking.ranked_candidate_tokens and ranking.ranked_candidate_tokens[0] == gold)


def fit_profile_scores(
    output: ReaderOutput, gold_rows: tuple[GoldRow, ...], profiles: tuple[Profile, ...],
) -> tuple[ProfileScore, ...]:
    lookup = _ranking_map(output); receipts = []
    for held in range(FOLD_COUNT):
        rows = tuple(item for item in gold_rows if not item.pseudo_unseen and item.fold != held)
        for profile in profiles:
            primary = tuple(_correct(lookup[(row.query_token, profile.profile_id, ARMS[0])], row.gold_candidate_token) for row in rows)
            off = tuple(_correct(lookup[(row.query_token, profile.profile_id, ARMS[1])], row.gold_candidate_token) for row in rows)
            receipts.append(ProfileScore(
                held_fold=held, profile_id=profile.profile_id, fit_seen_query_count=len(rows),
                correct_count=sum(primary), memory_off_correct_count=sum(off),
                rescue_count=sum(not before and after for before, after in zip(off, primary, strict=True)),
                harm_count=sum(before and not after for before, after in zip(off, primary, strict=True)),
            ))
    return tuple(receipts)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProfileSelection:
    held_fold: int
    profile_id: str


def select_profiles(receipts: tuple[ProfileScore, ...], profiles: tuple[Profile, ...]) -> tuple[ProfileSelection, ...]:
    if type(receipts) is not tuple or len(receipts) != FOLD_COUNT * len(profiles):
        _fail("profile_selection_invalid")
    indices = {item.profile_id: index for index, item in enumerate(profiles)}; result = []
    for held in range(FOLD_COUNT):
        rows = tuple(item for item in receipts if item.held_fold == held)
        if len(rows) != len(profiles) or {item.profile_id for item in rows} != set(indices):
            _fail("profile_selection_invalid")
        chosen = max(rows, key=lambda item: (
            item.correct_count, item.rescue_count - item.harm_count,
            item.rescue_count, -indices[item.profile_id],
        ))
        result.append(ProfileSelection(
            held_fold=held, profile_id=chosen.profile_id,
        ))
    return tuple(result)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QueryOutcome:
    query_token: str
    entity_cluster_token: str
    fold: int
    source_id: str
    pseudo_unseen: bool
    query_kind: str
    memory_count: int
    profile_id: str
    ranks_by_arm: tuple[tuple[str, int], ...]


def evaluate_selected_profiles(
    output: ReaderOutput, gold_rows: tuple[GoldRow, ...], selections: tuple[ProfileSelection, ...],
) -> tuple[QueryOutcome, ...]:
    lookup = _ranking_map(output); selected = {item.held_fold: item.profile_id for item in selections}
    if set(selected) != set(range(FOLD_COUNT)):
        _fail("profile_selection_invalid")
    result = []
    for row in gold_rows:
        profile_id = selected[row.fold]; ranks = []
        for arm in ARMS:
            ranking = lookup[(row.query_token, profile_id, arm)]
            try:
                rank = ranking.ranked_candidate_tokens.index(row.gold_candidate_token) + 1
            except ValueError:
                rank = 9
            ranks.append((arm, rank))
        result.append(QueryOutcome(
            query_token=row.query_token, entity_cluster_token=row.entity_cluster_token,
            fold=row.fold, source_id=row.source_id,
            pseudo_unseen=row.pseudo_unseen, query_kind=row.query_kind,
            memory_count=row.memory_count, profile_id=profile_id, ranks_by_arm=tuple(ranks),
        ))
    return tuple(sorted(result, key=lambda item: item.query_token.encode()))


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BaselineSummary:
    baseline: str
    seen_query_count: int
    correct_by_arm: tuple[tuple[str, int], ...]
    rescue_count: int
    harm_count: int
    correct_minus_wrong_count: int
    correct_minus_shuffled_count: int
    source_correct_minus_memory_off: tuple[tuple[str, int], ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class OracleSummary:
    eligible_query_count: int
    correct_mapping_correct_count: int
    shuffled_mapping_correct_count: int
    correct_minus_shuffled_count: int
    status: str


def summarize_readers(
    output: ReaderOutput,
    gold_rows: tuple[GoldRow, ...],
    supply: ReaderInput,
) -> tuple[tuple[BaselineSummary, ...], OracleSummary]:
    """Join gold only after method execution; oracle is evaluator-only."""
    if type(output) is not ReaderOutput or type(gold_rows) is not tuple or type(supply) is not ReaderInput:
        _fail("evaluation_invalid")
    lookup = {(item.query_token, item.profile_id, item.arm): item for item in output.rankings}
    gold = {item.query_token: item for item in gold_rows}
    if len(lookup) != len(output.rankings) or set(gold) != {item.query_token for item in supply.queries}:
        _fail("evaluation_invalid")
    seen = tuple(item for item in gold_rows if not item.pseudo_unseen)
    summaries = []
    for baseline in BASELINES:
        correct = {
            arm: tuple(
                lookup[(row.query_token, baseline, arm)].ranked_candidate_tokens[0] == row.gold_candidate_token
                for row in seen
            )
            for arm in CONTROL_ARMS
        }
        source_delta = []
        for source in ("gso", "uco3d"):
            indices = tuple(index for index, row in enumerate(seen) if row.source_id == source)
            source_delta.append((source, sum(correct["correct"][i] for i in indices) - sum(correct["memory_off"][i] for i in indices)))
        summaries.append(BaselineSummary(
            baseline=baseline,
            seen_query_count=len(seen),
            correct_by_arm=tuple((arm, sum(correct[arm])) for arm in CONTROL_ARMS),
            rescue_count=sum(not before and after for before, after in zip(correct["memory_off"], correct["correct"], strict=True)),
            harm_count=sum(before and not after for before, after in zip(correct["memory_off"], correct["correct"], strict=True)),
            correct_minus_wrong_count=sum(correct["correct"]) - sum(correct["wrong_entity"]),
            correct_minus_shuffled_count=sum(correct["correct"]) - sum(correct["shuffled_binding"]),
            source_correct_minus_memory_off=tuple(source_delta),
        ))
    queries = {item.query_token: item for item in supply.queries}
    eligible = tuple(row for row in seen if row.memory_count >= 2)
    shuffled_correct = 0
    for row in eligible:
        applications = {item.arm: item for item in queries[row.query_token].applications}
        correct_mapping = decode_envelope(applications[ARMS[0]].envelope)
        shuffled_mapping = decode_envelope(applications[ARMS[2]].envelope)
        if correct_mapping == shuffled_mapping:
            shuffled_correct += 1
    oracle = OracleSummary(
        eligible_query_count=len(eligible),
        correct_mapping_correct_count=len(eligible),
        shuffled_mapping_correct_count=shuffled_correct,
        correct_minus_shuffled_count=len(eligible) - shuffled_correct,
        status="ready_protocol_sanity_control_v1" if eligible and shuffled_correct < len(eligible) else "protocol_invalid",
    )
    if oracle.status != "ready_protocol_sanity_control_v1":
        _fail("protocol_invalid")
    return tuple(summaries), oracle

@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SelectedProfileSummary:
    seen_query_count: int
    pseudo_unseen_query_count: int
    rescue_count: int
    harm_count: int
    net_rescue_count: int
    correct_minus_wrong_count: int
    correct_minus_shuffled_count: int
    correct_minus_raw_multiset_count: int
    false_familiarity_count: int
    source_net_rescue: tuple[tuple[str, int], ...]

def summarize_selected_profiles(outcomes: tuple[QueryOutcome, ...]) -> SelectedProfileSummary:
    if type(outcomes) is not tuple or len(outcomes) != 192 or len({item.query_token for item in outcomes}) != 192:
        _fail("input_invalid")
    ranks = lambda row: dict(row.ranks_by_arm)
    seen = tuple(item for item in outcomes if not item.pseudo_unseen)
    unseen = tuple(item for item in outcomes if item.pseudo_unseen)
    rescue = sum(ranks(item)[ARMS[1]] != 1 and ranks(item)[ARMS[0]] == 1 for item in seen)
    harm = sum(ranks(item)[ARMS[1]] == 1 and ranks(item)[ARMS[0]] != 1 for item in seen)
    primary_correct = sum(ranks(item)[ARMS[0]] == 1 for item in seen)
    wrong_correct = sum(ranks(item)[ARMS[3]] == 1 for item in seen)
    shuffled_correct = sum(ranks(item)[ARMS[2]] == 1 for item in seen)
    raw_correct = sum(ranks(item)[ARMS[4]] == 1 for item in seen)
    false_familiarity = sum(ranks(item)[ARMS[1]] == 1 and ranks(item)[ARMS[0]] != 1 for item in unseen)
    source_net = []
    for source in sorted(_SOURCES):
        rows = tuple(item for item in seen if item.source_id == source)
        source_rescue = sum(ranks(item)[ARMS[1]] != 1 and ranks(item)[ARMS[0]] == 1 for item in rows)
        source_harm = sum(ranks(item)[ARMS[1]] == 1 and ranks(item)[ARMS[0]] != 1 for item in rows)
        source_net.append((source, source_rescue - source_harm))
    return SelectedProfileSummary(
        seen_query_count=len(seen), pseudo_unseen_query_count=len(unseen),
        rescue_count=rescue, harm_count=harm, net_rescue_count=rescue-harm,
        correct_minus_wrong_count=primary_correct-wrong_correct,
        correct_minus_shuffled_count=primary_correct-shuffled_correct,
        correct_minus_raw_multiset_count=primary_correct-raw_correct,
        false_familiarity_count=false_familiarity,
        source_net_rescue=tuple(source_net),
    )
