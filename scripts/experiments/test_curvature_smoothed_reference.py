#!/usr/bin/env python3
"""Pure validation for the isolated curvature-smoothed reference."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from curvature_smoothed_reference import build_smoothed_reference  # noqa: E402
from reference_trajectory_v1 import build_reference_trajectory, load_config  # noqa: E402


def require(condition: bool, message: str) -> None:
  if not condition:
    raise AssertionError(message)


def leaf_values(value, prefix=()):
  if isinstance(value, dict):
    result = {}
    for key, child in value.items():
      result.update(leaf_values(child, prefix + (str(key),)))
    return result
  if isinstance(value, list):
    result = {}
    for index, child in enumerate(value):
      result.update(leaf_values(child, prefix + (str(index),)))
    return result
  return {".".join(prefix): value}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, required=True)
  args = parser.parse_args()
  result = build_smoothed_reference(args.config.resolve())
  canonical_path = REPO_ROOT / result["experiment_config"]["canonical_reference_config"]
  canonical = build_reference_trajectory(load_config(canonical_path))
  rows = result["trajectory"]
  tests = []
  def check(name, function):
    function(); tests.append(name); print(f"PASS: {name}")
  def configuration_isolation() -> None:
    reference_config = result["experiment_config"]
    require(set(reference_config) == {"metadata", "canonical_reference_config", "smoothing"},
            "experimental reference config duplicates unrelated canonical settings")
    baseline_stage = json.loads((
      REPO_ROOT / "config/experiments/isaac_closed_loop_v1_command_headroom.json").read_text())
    experimental_stage = json.loads((
      REPO_ROOT / "config/experiments/isaac_closed_loop_v1_command_headroom_curvature_smoothed.json").read_text())
    baseline_leaves, experimental_leaves = leaf_values(baseline_stage), leaf_values(experimental_stage)
    differences = {key for key in baseline_leaves.keys() | experimental_leaves.keys()
                   if baseline_leaves.get(key) != experimental_leaves.get(key)}
    require(differences == {"baseline_configs.reference_trajectory"},
            f"Stage config is not isolated to its reference link: {sorted(differences)}")
  check("configuration isolation", configuration_isolation)
  check("start pose", lambda: require(max(abs(rows[0][key]) for key in ("x_m", "y_m", "yaw_rad")) < 2e-6, "start changed"))
  check("final pose", lambda: require(math.hypot(rows[-1]["x_m"] - 5.3, rows[-1]["y_m"] - 2.8) < 2e-8 and abs(rows[-1]["yaw_rad"]) < 2e-8, "final pose changed"))
  check("turn headings", lambda: require(abs(result["corners"][0]["heading_change_rad"] - math.pi / 2) < 1e-12 and abs(result["corners"][1]["heading_change_rad"] + math.pi / 2) < 1e-12, "turn angle mismatch"))
  check("peak curvature", lambda: require(abs(max(row["curvature_ref_1_m"] for row in rows) - 1 / .75) < 1e-10 and abs(min(row["curvature_ref_1_m"] for row in rows) + 1 / .55) < 1e-10, "peak curvature mismatch"))
  check("analytic curvature continuity", lambda: require(
    all(abs(result["boundaries"][i]["curvature_end_1_m"] -
            result["boundaries"][i + 1]["curvature_start_1_m"]) < 1e-14
        for i in range(len(result["boundaries"]) - 1)),
    "curvature boundary discontinuity"))
  check("positive adjusted straights", lambda: require(min(result["adjusted_straight_lengths_m"]) > 0, "non-positive straight"))
  check("ramp heading", lambda: require(
    all(abs(.5 * abs(corner["peak_curvature_1_m"]) * corner["ramp_length_m"] - .1) < 1e-14
        for corner in result["corners"]), "ramp heading mismatch"))
  ds = [rows[i + 1]["s_m"] - rows[i]["s_m"] for i in range(len(rows) - 1)]
  check("uniform resampling", lambda: require(all(abs(value - .02) < 1e-9 for value in ds[:-1]) and 0 < ds[-1] <= .02 + 1e-9, "resampling mismatch"))
  check("finite", lambda: require(all(math.isfinite(float(value)) for row in rows for value in row.values() if isinstance(value, (int, float))), "non-finite value"))
  check("start and stop", lambda: require(rows[0]["v_ref_m_s"] == 0 and rows[-1]["v_ref_m_s"] == 0, "endpoint speed mismatch"))
  limits = result["canonical_config"]["trajectory_limits"]
  check("dynamic limits", lambda: require(max(row["v_ref_m_s"] for row in rows) <= limits["maximum_body_speed_m_s"] + 1e-10 and max(abs(row["omega_ref_rad_s"]) for row in rows) <= limits["maximum_yaw_rate_rad_s"] + 1e-10 and max(row["v_ref_m_s"] ** 2 * abs(row["curvature_ref_1_m"]) for row in rows) <= limits["maximum_lateral_acceleration_m_s2"] + 1e-10 and max(abs(row["v_left_ref_m_s"]) for row in rows) <= limits["maximum_track_surface_speed_m_s"] + 1e-10 and max(abs(row["v_right_ref_m_s"]) for row in rows) <= limits["maximum_track_surface_speed_m_s"] + 1e-10, "dynamic limit violation"))
  first = [row["v_ref_m_s"] for row in rows if row["curvature_ref_1_m"] > 1.3]
  second = [row["v_ref_m_s"] for row in rows if row["curvature_ref_1_m"] < -1.8]
  check("tighter corner slower", lambda: require(min(second) < min(first), "tighter corner not slower"))
  print(json.dumps({"passed": len(tests), "total": len(tests),
                    "canonical_path_length_m": canonical["summary"]["path_length_m"],
                    "smoothed_path_length_m": rows[-1]["s_m"],
                    "path_length_difference_m": rows[-1]["s_m"] - canonical["summary"]["path_length_m"],
                    "corners": result["corners"],
                    "adjusted_straight_lengths_m": result["adjusted_straight_lengths_m"],
                    "total_duration_s": rows[-1]["t_s"]}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
