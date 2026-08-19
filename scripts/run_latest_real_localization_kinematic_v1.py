#!/usr/bin/env python3
"""Run exactly one nominal exact-kinematic V2 case for the latest Bag C/D route."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from map_route_kinematic_validation_v1 import load_validation_config, scenario_summary
from map_route_reference_v1 import build_map_route_reference
from simulate_map_route_kinematic_validation_v2 import simulate_v2
from tracking_controller_v2_direction_aware import load_direction_aware_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def evaluate_gate(rows: list[dict], metrics: dict) -> dict:
  active_end = float(metrics["reference_end_time_s"])
  reverse = [row for row in rows if float(row["time_s"]) <= active_end + 1e-12 and
             int(row["motion_direction"]) == -1 and float(row["v_ref_m_s"]) < -1e-6]
  negative_fraction = sum(float(row["v_cmd_m_s"]) < 0.0 for row in reverse) / len(reverse)
  checks = {
    "finite_and_bounded_xy": metrics["active"]["maximum_abs_time_aligned_xy_error"] < .5,
    "reverse_command_direction_dominantly_negative": negative_fraction > .5,
    "all_modes_present": set(metrics["mode_metrics"]) == {"forward", "reverse", "pivot"},
  }
  return {"passed": all(checks.values()), "checks": checks,
          "reverse_negative_reference_sample_count": len(reverse),
          "reverse_negative_command_fraction": negative_fraction,
          "policy": "A reverse maneuver passes signed-direction compatibility only when negative commands are the majority while v_ref is materially negative."}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/latest_real_localization_kinematic_v1.json")
  parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs/latest_real_localization_kinematic_v1")
  args = parser.parse_args(); config = json.loads(args.config.resolve().read_text())
  reference_result = build_map_route_reference(REPO_ROOT / config["baseline_configs"]["map_route_reference"])
  reference = reference_result["trajectory"]
  _, controller = load_direction_aware_config(REPO_ROOT / config["baseline_configs"]["direction_aware_controller"])
  validation = load_validation_config(REPO_ROOT / config["baseline_configs"]["v1_kinematic_validation"])
  initial = (float(reference[0]["x_m"]), float(reference[0]["y_m"]), float(reference[0]["yaw_rad"]))
  rows = simulate_v2(reference, initial, controller, validation, "map_nominal")
  metrics = scenario_summary(rows, reference, validation)
  output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
  with (output_dir / config["output"]["csv_filename"]).open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
  summary = {
    "metadata": config["metadata"], "source_config_paths": config["baseline_configs"],
    "map_reference_summary": reference_result["summary"], "v2_scenarios": {"map_nominal": metrics},
    "kinematic_gate": evaluate_gate(rows, metrics),
    "run_policy": "exactly one nominal full-route scenario; no perturbation and no parameter sweep",
    "projection_rule": f"metrics-only ordered +/-{validation['evaluation']['ordered_projection_half_window_segments']}-segment window",
    "limitations": ["Exact unicycle kinematics only; no actuator, force, contact, slip, noise, or Isaac Sim.",
                    "Large source-vs-generated pivot yaw disagreement is a reference confound, not automatically a controller failure."]}
  with (output_dir / config["output"]["summary_filename"]).open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
  print(f"Latest route nominal kinematic complete: samples={len(rows)}, active XY RMS={metrics['active']['rms_time_aligned_xy_error']:.6f}, final={metrics['final_xy_goal_error_m']:.6f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
