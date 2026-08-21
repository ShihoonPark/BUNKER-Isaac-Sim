#!/usr/bin/env python3
"""Replay accepted Bag-D poses through Real Tracking V1 without ROS or commands."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from map_route_reference_v1 import parse_tum
from real_global_path_tracking_v1 import (
  REPO_ROOT, ROW_SOURCES, ROW_SOURCE_RECORDED_POSE,
  LocalizationSample, ShadowSession, sample_from_xy_yaw, write_shadow_csv,
)


def _load_json(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    return json.load(stream)


def _all_safe_zero(result: dict[str, Any]) -> bool:
  fields = ("real_safe_v_cmd", "real_safe_omega_cmd",
            "real_safe_v_left", "real_safe_v_right")
  return all(all(abs(float(row[field])) <= 1e-15 for field in fields)
             for row in result["rows"])


def synthetic_safety_injections(config_path: Path) -> dict[str, Any]:
  context = ShadowSession(config_path).context
  start = context["start_map_reference"]
  base = sample_from_xy_yaw(10.0, float(start["x_m"]), float(start["y_m"]),
                            float(start["yaw_rad"]))
  safety = context["config"]["initial_safety_envelope_not_calibrated"]
  cases: dict[str, dict[str, Any]] = {}

  def capture(name: str, result: dict[str, Any]) -> None:
    cases[name] = {"safety_reason": result["rows"][0]["safety_reason"],
                   "real_safe_candidate_zero": _all_safe_zero(result)}

  capture("no_localization", ShadowSession(config_path).evaluate(None, 10.0))
  capture("stale", ShadowSession(config_path).evaluate(
    base, 10.0 + float(safety["maximum_localization_age_s"]) + 0.01))
  capture("wrong_frame", ShadowSession(config_path).evaluate(
    LocalizationSample(**{**base.__dict__, "frame_id": "odom"}), 10.0))
  capture("nonfinite", ShadowSession(config_path).evaluate(
    LocalizationSample(**{**base.__dict__, "x_m": math.nan}), 10.0))
  capture("invalid_quaternion", ShadowSession(config_path).evaluate(
    LocalizationSample(base.timestamp_s, base.frame_id, base.x_m, base.y_m, base.z_m,
                       0.0, 0.0, 0.0, 0.0), 10.0))
  jump_session = ShadowSession(config_path)
  jump_session.evaluate(base, 10.0)
  jumped = sample_from_xy_yaw(10.1,
                               base.x_m + float(safety["maximum_pose_jump_m"]) + 0.5,
                               base.y_m,
                               float(start["yaw_rad"]) +
                               float(safety["maximum_pose_jump_yaw_rad"]) + 0.5)
  capture("pose_jump", jump_session.evaluate(jumped, 10.1))
  return {"cases": cases,
          "all_injected_unsafe_cases_zero": all(
            value["real_safe_candidate_zero"] for value in cases.values())}


def run_recorded_audit(config_path: Path, output_dir: Path) -> dict[str, Any]:
  session = ShadowSession(config_path)
  config = session.context["config"]
  dataset_path = REPO_ROOT / config["baseline_configs"]["dataset_provenance"]
  latest_path = REPO_ROOT / config["baseline_configs"]["latest_localization_reference"]
  dataset = _load_json(dataset_path)
  latest = _load_json(latest_path)
  localization = dataset["localization"]
  source_path = Path(localization["accepted_trajectory_artifact"])
  if (str(source_path) != latest["source"]["trajectory_path"] or
      localization["pose_semantics"] != "T_map_lidar; p_map = T_map_lidar * p_lidar" or
      latest["source"]["pose_semantics"] != "T_map_lidar"):
    raise RuntimeError("latest accepted Bag-D provenance disagrees across canonical configs")
  source = parse_tum(source_path,
                     float(config["recorded_audit"]["quaternion_norm_tolerance"]))
  expected_count = int(localization["accepted_pose_count"])
  if len(source) != expected_count or expected_count != int(latest["source"]["accepted_pose_count"]):
    raise RuntimeError("accepted Bag-D pose count changed")

  rows: list[dict[str, Any]] = []
  source_valid_count = 0
  for pose in source:
    timestamp = float(pose["source_timestamp_s"])
    sample = LocalizationSample(
      timestamp, str(localization["parent_frame"]),
      float(pose["x_m"]), float(pose["y_m"]), float(pose["z_m"]),
      float(pose["qx"]), float(pose["qy"]), float(pose["qz"]), float(pose["qw"]))
    result = session.evaluate(sample, timestamp, row_source=ROW_SOURCE_RECORDED_POSE)
    source_valid_count += int(result["validation"]["valid"])
    rows.extend(result["rows"])

  output_dir.mkdir(parents=True, exist_ok=True)
  csv_path = output_dir / config["output"]["recorded_csv_filename"]
  write_shadow_csv(csv_path, rows)
  reasons = Counter(str(row["safety_reason"]) for row in rows)
  row_sources = Counter(str(row["row_source"]) for row in rows)
  safety = config["initial_safety_envelope_not_calibrated"]
  injection = synthetic_safety_injections(config_path)
  finite_commands = all(math.isfinite(float(row[field])) for row in rows for field in (
    "canonical_v_cmd", "canonical_omega_cmd", "canonical_v_left", "canonical_v_right",
    "real_safe_v_cmd", "real_safe_omega_cmd", "real_safe_v_left", "real_safe_v_right"))
  safe_limits_hold = all(
    abs(float(row["real_safe_v_cmd"])) <= float(safety["maximum_real_body_speed_m_s"]) + 1e-12 and
    abs(float(row["real_safe_omega_cmd"])) <= float(safety["maximum_real_yaw_rate_rad_s"]) + 1e-12 and
    abs(float(row["real_safe_v_left"])) <= float(safety["maximum_real_track_speed_m_s"]) + 1e-12 and
    abs(float(row["real_safe_v_right"])) <= float(safety["maximum_real_track_speed_m_s"]) + 1e-12
    for row in rows)
  checks = {
    "canonical_dataset_identity": len(source) == 2176,
    "all_recorded_poses_pass_interface_safety": source_valid_count == len(source),
    "two_start_modes_per_pose": len(rows) == 2 * len(source),
    "all_rows_are_recorded_pose_source": row_sources == {ROW_SOURCE_RECORDED_POSE: len(rows)},
    "commands_finite": finite_commands,
    "initial_real_safety_caps_hold": safe_limits_hold,
    "unsafe_injections_force_zero": injection["all_injected_unsafe_cases_zero"],
    "shadow_only": all(int(row["shadow_only"]) == 1 for row in rows),
  }
  summary = {
    "stage": "Real Global Path Tracking V1 — recorded shadow audit",
    "gate": {"passed": all(checks.values()), "checks": checks},
    "dataset_provenance": {"config": str(dataset_path.relative_to(REPO_ROOT)),
                           "artifact": str(source_path),
                           "pose_semantics": "accepted-only T_map_lidar",
                           "accepted_pose_count": len(source)},
    "reference": {"preset": config["runtime"]["preset"],
                  "lap_length_m": session.context["lap_length_m"],
                  "start_modes": list(config["runtime"]["evaluated_start_modes"])},
    "recorded_interface": {"valid_pose_count": source_valid_count,
                           "shadow_row_count": len(rows),
                           "row_source_counts": {source: row_sources.get(source, 0)
                                                 for source in ROW_SOURCES},
                           "safety_reason_counts": dict(sorted(reasons.items())),
                           "real_safety_clamped_row_count": sum(
                             float(row["real_safety_scale"]) < 1.0 - 1e-12 for row in rows),
                           "maximum_abs_canonical_v_m_s": max(abs(float(row["canonical_v_cmd"])) for row in rows),
                           "maximum_abs_canonical_omega_rad_s": max(abs(float(row["canonical_omega_cmd"])) for row in rows),
                           "maximum_abs_real_safe_v_m_s": max(abs(float(row["real_safe_v_cmd"])) for row in rows),
                           "maximum_abs_real_safe_omega_rad_s": max(abs(float(row["real_safe_omega_cmd"])) for row in rows)},
    "synthetic_safety_injections": injection,
    "interpretation": config["recorded_audit"]["interpretation"],
    "tracking_performance_interpretation_prohibited": True,
    "physical_command_publication": False,
    "output_csv": str(csv_path),
  }
  summary_path = output_dir / config["output"]["recorded_summary_filename"]
  with summary_path.open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, allow_nan=False)
    stream.write("\n")
  print(f"recorded Bag-D shadow rows={len(rows)}, gate={'PASS' if summary['gate']['passed'] else 'FAIL'}")
  print("interpretation: interface/sign/projection/safety audit only; not rounded-loop tracking performance")
  print(f"output={output_dir}")
  return summary


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path,
                      default=REPO_ROOT / "config/real_global_path_tracking_v1.json")
  parser.add_argument("--output-dir", type=Path)
  args = parser.parse_args()
  config_path = args.config.resolve()
  config = _load_json(config_path)
  default = REPO_ROOT / config["output"]["default_directory"] / "recorded_bag_d"
  summary = run_recorded_audit(config_path, (args.output_dir or default).resolve())
  return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
