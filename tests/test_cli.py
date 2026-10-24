from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

from retrieval_not_binding.cli import main
from retrieval_not_binding.io import load_gold, parse_reader_input, reader_input_dict, write_json
from retrieval_not_binding.protocol import ARMS, BindingError, Query, Ranking, ReaderInput, ReaderOutput
from retrieval_not_binding.readers import BASELINES, CONTROL_ARMS
from retrieval_not_binding.settings import load_config, profiles_from_config
from tests.test_memory import config, supply, token


CONFIG = Path(__file__).resolve().parents[1] / "configs/benchmark.json"


class CliTests(unittest.TestCase):
    def test_gold_rejects_an_entity_split_across_folds(self):
        first = {"query_token": token("query", 0), "entity_cluster_token": token("cluster", 0),
            "gold_candidate_token": token("candidate", 0), "fold": 0, "source_id": "gso",
            "pseudo_unseen": False, "query_kind": "view", "memory_count": 2}
        second = {**first, "query_token": token("query", 1), "fold": 1, "query_kind": "detail"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.json"
            write_json(path, {"schema_version": "retrieval_not_binding.gold.v1", "rows": [first, second]})
            with self.assertRaisesRegex(BindingError, "entity_fold_isolation_invalid"):
                load_gold(path)

    def test_reader_input_round_trip_and_rejects_evaluator_labels(self):
        original = supply()
        raw = json.loads(json.dumps(reader_input_dict(original)))
        self.assertEqual(parse_reader_input(raw), original)
        raw["gold"] = []
        with self.assertRaises(BindingError):
            parse_reader_input(raw)

    def test_rank_requires_no_gold_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input.json", root / "rankings.json"
            write_json(source, reader_input_dict(supply()))
            args = ["--config", str(CONFIG), "rank", "--input", str(source), "--output", str(output)]
            self.assertEqual(main(args), 0)
            raw = json.loads(output.read_text())
            self.assertEqual(len(raw["fixed_readers"]["rankings"]), 2 * 4 * 4)
            self.assertEqual(len(raw["erm_profiles"]["rankings"]), 2 * 4 * 5)
            self.assertEqual(raw["configuration"], config())
            original_bytes = output.read_bytes()
            with self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(output.read_bytes(), original_bytes)

    def test_evaluate_reports_known_synthetic_counts_and_source_bootstrap(self):
        # Synthetic rankings exercise the full evaluation path without encoding images.
        template = supply()
        queries, gold, fixed, profiles = [], [], [], []
        seen_index = 0
        for source_index, source in enumerate(("gso", "uco3d")):
            for entity in range(24):
                for kind_index, kind in enumerate(("view", "occlusion", "environment", "detail")):
                    index = source_index * 96 + entity * 4 + kind_index
                    query_token = token("query", index)
                    original = template.queries[0]
                    queries.append(Query(query_token=query_token,
                        query_feature_tokens=original.query_feature_tokens,
                        candidates=original.candidates, applications=original.applications))
                    pseudo = entity >= 16
                    gold.append({"query_token": query_token,
                        "entity_cluster_token": token("cluster", source_index * 24 + entity),
                        "gold_candidate_token": token("candidate", 0),
                        "fold": entity % 4, "source_id": source, "pseudo_unseen": pseudo,
                        "query_kind": kind, "memory_count": 2})
                    correct = {
                        "correct": seen_index != 2,
                        "memory_off": seen_index not in (0, 1),
                        "wrong_entity": seen_index not in (0, 1, 2, 3),
                        "shuffled_binding": seen_index not in (2, 5),
                        "raw_multiset": seen_index != 2,
                    }
                    if pseudo:
                        correct = {arm: True for arm in ARMS}
                        correct["correct"] = not (source_index == 0 and entity == 16 and kind_index == 0)
                    candidates = tuple(token("candidate", item) for item in range(8))
                    def ranking(profile_id, arm):
                        order = candidates if correct[arm] else (candidates[1], candidates[0], *candidates[2:])
                        return Ranking(query_token=query_token, profile_id=profile_id, arm=arm,
                            ranked_candidate_tokens=order,
                            ranked_score_micros=tuple(range(900000, 100000, -100000)))
                    fixed.extend(ranking(reader, arm) for reader in BASELINES for arm in CONTROL_ARMS)
                    profiles.extend(ranking(profile.profile_id, arm)
                        for profile in profiles_from_config(config()) for arm in ARMS)
                    if not pseudo:
                        seen_index += 1
        value = ReaderInput(features=template.features, episode_features=template.episode_features,
                            queries=tuple(queries))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "input.json", reader_input_dict(value))
            write_json(root / "gold.json", {"schema_version": "retrieval_not_binding.gold.v1", "rows": gold})
            write_json(root / "rankings.json", {"schema_version": "retrieval_not_binding.rankings.v1",
                "configuration": config(),
                "fixed_readers": dataclasses.asdict(ReaderOutput(rankings=tuple(fixed))),
                "erm_profiles": dataclasses.asdict(ReaderOutput(rankings=tuple(profiles)))})
            self.assertEqual(main(["--config", str(CONFIG), "evaluate",
                "--input", str(root / "input.json"), "--rankings", str(root / "rankings.json"),
                "--gold", str(root / "gold.json"), "--output", str(root / "report.json")]), 0)
            report = json.loads((root / "report.json").read_text())
            summary = report["profile_selected_erm"]
            self.assertEqual((summary["rescue_count"], summary["harm_count"], summary["net_rescue_count"]), (2, 1, 1))
            self.assertEqual(summary["false_familiarity_count"], 1)
            self.assertEqual(report["cohort"]["overall"]["query_count"], 192)
            self.assertEqual(report["envelope_bytes_per_arm"], 34176)
            self.assertEqual(report["mapping_oracle"]["eligible_query_count"], 128)
            self.assertEqual(report["entity_bootstrap"]["defined_rounds"], 2000)
