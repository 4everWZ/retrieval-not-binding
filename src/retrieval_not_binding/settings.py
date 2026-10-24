"""Load the paper's reader profiles from its public benchmark configuration."""
from pathlib import Path
import json
from .protocol import ARMS, DIMENSION, ENVELOPE_SIZE_BYTES, FOLD_COUNT, Profile, BindingError


def load_config(path: str | Path = "configs/benchmark.json") -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config.get("schema_version") != "retrieval_not_binding.benchmark.v1"
        or config["controls"]["applications"] != list(ARMS)
        or config["controls"]["capacity_slots"] != 5
        or config["controls"]["serialized_bytes_per_application"] != ENVELOPE_SIZE_BYTES
        or config["controls"]["raw_multiset_duplicate"] != "first-observation"
        or config["representation"]["dimension"] != DIMENSION
        or config["statistics"]["fold_count"] != FOLD_COUNT):
        raise BindingError("configuration_invalid")
    if (config["readers"]["reservoir"]["seed"] != 20260806
        or config["readers"]["reservoir"]["representative_count"] != 2
        or config["readers"]["kcenter"]["representative_count"] != 2
        or config["readers"]["erm"]["profile_id"] != "balanced-recency"
        or config["statistics"]["bootstrap_rounds"] != 2000
        or config["statistics"]["bootstrap_seed"] != 20260803
        or config["candidates"]["candidate_k"] != 8
        or config["candidates"]["candidate_count_per_source_query_kind"] != 24):
        raise BindingError("configuration_invalid")
    profiles_from_config(config)
    return config


def profiles_from_config(config: dict) -> tuple[Profile, ...]:
    profiles = tuple(Profile(**item) for item in config["algorithm"]["profiles"])
    expected = (
        Profile(profile_id="content-strong", alpha_micros=1000000, beta_micros=8000000, gamma_micros=0, lambda_micros=250000),
        Profile(profile_id="balanced-recency", alpha_micros=1000000, beta_micros=8000000, gamma_micros=500000, lambda_micros=250000),
        Profile(profile_id="recency-strong", alpha_micros=1000000, beta_micros=8000000, gamma_micros=1000000, lambda_micros=250000),
        Profile(profile_id="residual-conservative", alpha_micros=500000, beta_micros=4000000, gamma_micros=500000, lambda_micros=250000),
    )
    if profiles != expected:
        raise BindingError("configuration_invalid")
    return profiles
