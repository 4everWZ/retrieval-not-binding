"""Feature-only ranking and a separate label-aware evaluation command."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .evaluation import (
    evaluate_selected_profiles, fit_profile_scores, select_profiles,
    summarize_readers, summarize_selected_profiles,
)
from .io import load_gold, load_reader_input, parse_rankings, write_json
from .memory import run_erm_profiles
from .protocol import ENVELOPE_SIZE_BYTES
from .readers import run_readers
from .settings import load_config, profiles_from_config
from .statistics import bootstrap_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrieval-not-binding", description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    rank = commands.add_parser("rank", help="Read feature inputs only and save all rankings")
    rank.add_argument("--input", type=Path, required=True)
    rank.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Join saved rankings with separate evaluator labels")
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--rankings", type=Path, required=True)
    evaluate.add_argument("--gold", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        profiles = profiles_from_config(config)
        supply = load_reader_input(args.input)
        if args.command == "rank":
            fixed = run_readers(supply, profiles)
            selected = run_erm_profiles(supply, profiles)
            write_json(args.output, {
                "schema_version": "retrieval_not_binding.rankings.v1",
                "configuration": config,
                "fixed_readers": dataclasses.asdict(fixed),
                "erm_profiles": dataclasses.asdict(selected),
            })
        else:
            raw = json.loads(args.rankings.read_text(encoding="utf-8"))
            if (set(raw) != {"schema_version", "configuration", "fixed_readers", "erm_profiles"}
                or raw["schema_version"] != "retrieval_not_binding.rankings.v1"
                or raw["configuration"] != config):
                raise ValueError("rankings_schema_invalid")
            fixed = parse_rankings(raw["fixed_readers"])
            selected = parse_rankings(raw["erm_profiles"])
            gold = load_gold(args.gold)
            summaries, oracle = summarize_readers(fixed, gold, supply)
            receipts = fit_profile_scores(selected, gold, profiles)
            selections = select_profiles(receipts, profiles)
            outcomes = evaluate_selected_profiles(selected, gold, selections)
            statistics = config["statistics"]
            interval = bootstrap_summary(outcomes, rounds=statistics["bootstrap_rounds"],
                                         seed=statistics["bootstrap_seed"])
            baseline = {(row.query_token, row.arm): row for row in fixed.rankings
                        if row.profile_id == "mean_prototype"}
            coverage = {}
            for source in ("gso", "uco3d", "overall"):
                rows = tuple(row for row in gold if source == "overall" or row.source_id == source)
                coverage[source] = {
                    "query_count": len(rows),
                    "candidate_covered_count": sum(row.gold_candidate_token in baseline[(row.query_token, "memory_off")].ranked_candidate_tokens for row in rows),
                    "memory_off_correct_count": sum(row.gold_candidate_token == baseline[(row.query_token, "memory_off")].ranked_candidate_tokens[0] for row in rows),
                }
            write_json(args.output, {
                "schema_version": "retrieval_not_binding.evaluation.v1",
                "configuration": config,
                "cohort": coverage,
                "envelope_bytes_per_application": ENVELOPE_SIZE_BYTES,
                "envelope_bytes_per_arm": len(supply.queries) * ENVELOPE_SIZE_BYTES,
                "fixed_readers": [dataclasses.asdict(item) for item in summaries],
                "mapping_oracle": dataclasses.asdict(oracle),
                "profile_selections": [dataclasses.asdict(item) for item in selections],
                "profile_selected_erm": dataclasses.asdict(summarize_selected_profiles(outcomes)),
                "entity_bootstrap": dataclasses.asdict(interval),
            })
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    return 0
