#!/usr/bin/env python3
"""Deterministic tests for Real Global Path Tracking V1 shadow preparation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from map_route_reference_v1 import parse_tum
from real_global_path_tracking_v1 import (
  CSV_COLUMNS, NEAREST_PATH_MODE, REPO_ROOT, ROW_SOURCE_POSE_UPDATE,
  ROW_SOURCE_RECORDED_POSE, ROW_SOURCE_STATUS_TIMER, START_POSE_MODE,
  LocalizationSample, ShadowSession, sample_from_xy_yaw, wrap_to_pi,
)


def require(condition: bool, message: str) -> None:
  if not condition:
    raise AssertionError(message)


def safe_zero(result: dict[str, Any]) -> bool:
  fields = ("real_safe_v_cmd", "real_safe_omega_cmd",
            "real_safe_v_left", "real_safe_v_right")
  return all(all(abs(float(row[field])) <= 1e-15 for field in fields)
             for row in result["rows"])


def mode_row(result: dict[str, Any], mode: str) -> dict[str, Any]:
  return next(row for row in result["rows"] if row["start_mode"] == mode)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path,
                      default=REPO_ROOT / "config/real_global_path_tracking_v1.json")
  parser.add_argument("--require-recorded-output", action="store_true")
  args = parser.parse_args()
  config_path = args.config.resolve()
  base_session = ShadowSession(config_path)
  context = base_session.context
  config = context["config"]
  start = context["start_map_reference"]
  periodic = context["periodic_map_reference"]
  safety = config["initial_safety_envelope_not_calibrated"]
  tests: list[str] = []

  def check(name: str, function: Callable[[], None]) -> None:
    function()
    tests.append(name)
    print(f"PASS: {name}")

  def configuration() -> None:
    require(config["runtime"]["shadow_only"] and
            not config["runtime"]["create_command_publisher"] and
            config["runtime"]["arm_state"] == "NOT_ARMABLE", "shadow lock changed")
    require(not safety["startup_alignment"]["pass_thresholds_defined"],
            "start alignment threshold invented")
    require(float(safety["maximum_real_body_speed_m_s"]) < .60 and
            float(safety["maximum_real_yaw_rate_rad_s"]) < .70 and
            float(safety["maximum_real_track_speed_m_s"]) < .80,
            "real caps are not below simulation limits")
  check("shadow-only non-calibrated configuration", configuration)

  def shared_modules() -> None:
    core = (REPO_ROOT / "scripts/real_global_path_tracking_v1.py").read_text()
    require("from global_path_demo_v1 import build_demo_reference" in core and
            "from tracking_controller_v2_direction_aware import controller_command" in core,
            "reference/controller implementation was duplicated instead of imported")
    require(not any(token in core.lower() for token in ("isaacsim", "physx", "omni.usd")),
            "pure real adapter imports Isaac/PhysX")
  check("simulation and real share reference/controller core", shared_modules)

  def exact_start() -> None:
    session = ShadowSession(config_path)
    result = session.evaluate(sample_from_xy_yaw(
      1.0, float(start["x_m"]), float(start["y_m"]), float(start["yaw_rad"])), 1.0)
    alignment = result["alignment"]
    require(result["validation"]["valid"] and
            alignment["start_xy_distance_m"] < 1e-12 and
            abs(alignment["start_heading_error_rad"]) < 1e-12 and
            alignment["nearest"]["distance_m"] < 1e-12 and
            abs(alignment["nearest"]["s_within_lap_m"]) < 1e-12,
            "exact canonical start audit mismatch")
    start_row = mode_row(result, START_POSE_MODE)
    nearest_row = mode_row(result, NEAREST_PATH_MODE)
    require(abs(float(start_row["canonical_v_cmd"])) < 1e-12 and
            float(nearest_row["canonical_v_cmd"]) > 0.0,
            "two start modes did not remain distinct")
  check("exact rounded start with correct yaw", exact_start)

  def heading_offset() -> None:
    offset = 0.4
    result = ShadowSession(config_path).evaluate(sample_from_xy_yaw(
      2.0, float(start["x_m"]), float(start["y_m"]),
      float(start["yaw_rad"]) + offset), 2.0)
    require(abs(result["alignment"]["start_heading_error_rad"] + offset) < 1e-12 and
            abs(mode_row(result, START_POSE_MODE)["heading_error"] + offset) < 1e-12,
            "heading error sign/wrap mismatch")
  check("rounded start with heading offset", heading_offset)

  def lateral_offset() -> None:
    reference = next(row for row in periodic
                     if row["segment_type"] == "line" and
                     .5 < float(row["s_m"]) < context["lap_length_m"] - .5)
    offset = .12
    yaw = float(reference["yaw_rad"])
    actual_x = float(reference["x_m"]) - math.sin(yaw) * offset
    actual_y = float(reference["y_m"]) + math.cos(yaw) * offset
    result = ShadowSession(config_path).evaluate(
      sample_from_xy_yaw(3.0, actual_x, actual_y, yaw), 3.0)
    nearest = result["alignment"]["nearest"]
    row = mode_row(result, NEAREST_PATH_MODE)
    require(abs(float(nearest["cross_track_error_m"]) - offset) < 1e-8 and
            abs(float(row["e_y"]) + offset) < 1e-8,
            "lateral CTE/body-error sign mismatch")
  check("lateral offset from straight has deterministic signs", lateral_offset)

  def periodic_boundary() -> None:
    session = ShadowSession(config_path)
    before, after = periodic[-2], periodic[1]
    first = session.evaluate(sample_from_xy_yaw(
      4.0, float(before["x_m"]), float(before["y_m"]), float(before["yaw_rad"])), 4.0)
    second = session.evaluate(sample_from_xy_yaw(
      4.1, float(after["x_m"]), float(after["y_m"]), float(after["yaw_rad"])), 4.1)
    first_s = float(first["alignment"]["nearest"]["s_total_m"])
    second_s = float(second["alignment"]["nearest"]["s_total_m"])
    require(first_s > context["lap_length_m"] - .05 and
            second_s > context["lap_length_m"] and
            0.0 < second_s - first_s < .10,
            "closed-path boundary selected the wrong progress branch")
  check("near-lap-boundary projection unwrap", periodic_boundary)

  def corner_projection() -> None:
    reference = next(row for row in periodic if row["segment_type"] == "arc")
    result = ShadowSession(config_path).evaluate(sample_from_xy_yaw(
      5.0, float(reference["x_m"]), float(reference["y_m"]),
      float(reference["yaw_rad"])), 5.0)
    require(result["alignment"]["nearest"]["distance_m"] < 1e-10 and
            abs(wrap_to_pi(result["alignment"]["nearest"]["path_yaw_rad"] -
                           float(reference["yaw_rad"]))) < .03,
            "corner projection/yaw mismatch")
  check("near-corner continuous projection", corner_projection)

  def far_pose() -> None:
    result = ShadowSession(config_path).evaluate(sample_from_xy_yaw(
      6.0, float(start["x_m"]) + 20.0, float(start["y_m"]) + 20.0,
      float(start["yaw_rad"])), 6.0)
    require(result["validation"]["valid"] and
            result["alignment"]["start_xy_distance_m"] > 20.0 and
            all(row["arm_state"] == "NOT_ARMABLE" and
                row["start_alignment_state"] == "START_ALIGNMENT_NOT_CHECKED"
                for row in result["rows"]),
            "far pose was silently treated as aligned/armable")
  check("far pose remains diagnostics-only", far_pose)

  base = sample_from_xy_yaw(10.0, float(start["x_m"]), float(start["y_m"]),
                            float(start["yaw_rad"]))

  def stale() -> None:
    result = ShadowSession(config_path).evaluate(
      base, 10.0 + float(safety["maximum_localization_age_s"]) + .01)
    require("STALE_LOCALIZATION" in result["validation"]["reasons"] and safe_zero(result) and
            all(row["localization_state"] == "STALE" for row in result["rows"]),
            "stale localization did not force zero")
  check("stale localization forces zero", stale)

  def wrong_frame() -> None:
    result = ShadowSession(config_path).evaluate(replace(base, frame_id="odom"), 10.0)
    require("WRONG_FRAME" in result["validation"]["reasons"] and safe_zero(result) and
            all(row["localization_state"] == "INVALID" for row in result["rows"]),
            "wrong frame did not force zero")
  check("wrong frame forces zero", wrong_frame)

  def nonfinite() -> None:
    result = ShadowSession(config_path).evaluate(replace(base, x_m=math.nan), 10.0)
    require("NONFINITE_POSE" in result["validation"]["reasons"] and safe_zero(result) and
            all(not int(row["canonical_candidate_computed"]) for row in result["rows"]),
            "NaN pose did not suppress candidates")
  check("NaN pose forces zero", nonfinite)

  def pose_jump() -> None:
    session = ShadowSession(config_path)
    session.evaluate(base, 10.0)
    jumped = sample_from_xy_yaw(
      10.1, base.x_m + float(safety["maximum_pose_jump_m"]) + .1, base.y_m,
      float(start["yaw_rad"]) + float(safety["maximum_pose_jump_yaw_rad"]) + .1)
    result = session.evaluate(jumped, 10.1)
    require("POSE_JUMP_TRANSLATION" in result["validation"]["reasons"] and
            "POSE_JUMP_YAW" in result["validation"]["reasons"] and safe_zero(result),
            "sudden pose jump did not force zero")
  check("large pose jump forces zero", pose_jump)

  def missing_and_quaternion() -> None:
    missing = ShadowSession(config_path).evaluate(None, 12.0)
    invalid = ShadowSession(config_path).evaluate(
      LocalizationSample(12.0, "map", base.x_m, base.y_m, 0.0,
                         0.0, 0.0, 0.0, 0.0), 12.0)
    require("NO_LOCALIZATION" in missing["validation"]["reasons"] and safe_zero(missing) and
            "INVALID_QUATERNION" in invalid["validation"]["reasons"] and safe_zero(invalid),
            "missing/invalid quaternion condition did not force zero")
  check("missing pose and invalid quaternion force zero", missing_and_quaternion)

  def clamp_preserves_canonical() -> None:
    result = ShadowSession(config_path).evaluate(base, 10.0)
    row = mode_row(result, NEAREST_PATH_MODE)
    require(abs(float(row["canonical_v_cmd"]) - .6) < 1e-12 and
            abs(float(row["real_safe_v_cmd"])) <= float(safety["maximum_real_body_speed_m_s"]) + 1e-12 and
            abs(float(row["real_safe_omega_cmd"])) <= float(safety["maximum_real_yaw_rate_rad_s"]) + 1e-12 and
            abs(float(row["real_safe_v_left"])) <= float(safety["maximum_real_track_speed_m_s"]) + 1e-12 and
            abs(float(row["real_safe_v_right"])) <= float(safety["maximum_real_track_speed_m_s"]) + 1e-12 and
            0.0 < float(row["real_safety_scale"]) < 1.0,
            "canonical and real-safe candidates were not separately preserved")
  check("real safety clamp preserves canonical candidate", clamp_preserves_canonical)

  def csv_schema() -> None:
    required = {"timestamp", "localization_timestamp", "localization_age_s",
      "actual_map_x", "actual_map_y", "actual_yaw", "reference_x", "reference_y",
      "reference_yaw", "reference_s", "reference_v", "reference_omega",
      "nearest_path_s", "e_x", "e_y", "cross_track_error", "heading_error",
      "canonical_v_cmd", "canonical_omega_cmd", "canonical_v_left", "canonical_v_right",
      "real_safe_v_cmd", "real_safe_omega_cmd", "real_safe_v_left", "real_safe_v_right",
      "command_scale", "localization_valid", "safety_reason", "shadow_only", "row_source"}
    require(required <= set(CSV_COLUMNS), "required Sim/Real comparison CSV fields missing")
  check("future Sim-to-Real CSV schema", csv_schema)

  def row_source_semantics() -> None:
    sample = sample_from_xy_yaw(20.0, float(start["x_m"]), float(start["y_m"]),
                                float(start["yaw_rad"]))
    pose = ShadowSession(config_path).evaluate(
      sample, 20.0, row_source=ROW_SOURCE_POSE_UPDATE)
    timer = ShadowSession(config_path).evaluate(
      sample, 20.0, update_history=False, row_source=ROW_SOURCE_STATUS_TIMER)
    recorded = ShadowSession(config_path).evaluate(
      sample, 20.0, row_source=ROW_SOURCE_RECORDED_POSE)
    semantics = config["output"]["row_source_semantics"]
    require(all(row["row_source"] == ROW_SOURCE_POSE_UPDATE for row in pose["rows"]) and
            all(row["row_source"] == ROW_SOURCE_STATUS_TIMER for row in timer["rows"]) and
            all(row["row_source"] == ROW_SOURCE_RECORDED_POSE for row in recorded["rows"]) and
            set(semantics) == {ROW_SOURCE_POSE_UPDATE, ROW_SOURCE_STATUS_TIMER,
                               ROW_SOURCE_RECORDED_POSE} and
            config["output"]["future_tracking_performance_filter"] ==
            "row_source == POSE_UPDATE",
            "row-source provenance/filter semantics mismatch")
  check("row_source provenance and metric filter", row_source_semantics)

  def no_command_publisher() -> None:
    wrapper = (REPO_ROOT / "scripts/run_real_global_path_tracking_shadow_v1.py").read_text()
    require("create_subscription" in wrapper and "create_publisher" not in wrapper and
            "cmd_vel" not in wrapper, "live wrapper contains a command-publisher path")
  check("live wrapper has no cmd_vel publisher", no_command_publisher)

  def provenance() -> None:
    dataset_path = REPO_ROOT / config["baseline_configs"]["dataset_provenance"]
    dataset = json.loads(dataset_path.read_text())
    source = parse_tum(Path(dataset["localization"]["accepted_trajectory_artifact"]),
                       float(config["recorded_audit"]["quaternion_norm_tolerance"]))
    require(len(source) == dataset["localization"]["accepted_pose_count"] == 2176 and
            dataset["localization"]["parent_frame"] == "map" and
            dataset["localization"]["source_type"] == "accepted independent localization",
            "recorded accepted Bag-D provenance mismatch")
  check("recorded Bag-D interface provenance", provenance)

  def frozen_simulation_baseline() -> None:
    paths = ["config/global_path_demo_v1.json", "scripts/global_path_demo_v1.py",
             "scripts/run_global_path_demo_v1.py",
             "config/tracking_controller_v2_direction_aware.json",
             "scripts/tracking_controller_v2_direction_aware.py",
             "config/tracking_controller_v1.json", "scripts/tracking_controller_v1.py"]
    completed = subprocess.run(["git", "diff", "--exit-code", "--", *paths], cwd=REPO_ROOT,
                               check=False, capture_output=True, text=True)
    require(completed.returncode == 0 and not completed.stdout,
            "validated simulation reference/controller baseline changed")
  check("validated simulation baseline frozen", frozen_simulation_baseline)

  if args.require_recorded_output:
    def recorded_output() -> None:
      directory = REPO_ROOT / config["output"]["default_directory"] / "recorded_bag_d"
      summary_path = directory / config["output"]["recorded_summary_filename"]
      csv_path = directory / config["output"]["recorded_csv_filename"]
      require(summary_path.is_file() and csv_path.is_file(), "recorded audit output missing")
      summary = json.loads(summary_path.read_text())
      with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        output_rows = list(reader)
        row_count = len(output_rows)
        fields = set(reader.fieldnames or [])
      row_sources = Counter(row["row_source"] for row in output_rows)
      require(summary["gate"]["passed"] and
              summary["dataset_provenance"]["accepted_pose_count"] == 2176 and
              summary["recorded_interface"]["shadow_row_count"] == 4352 and
              summary["recorded_interface"]["row_source_counts"] ==
              {ROW_SOURCE_POSE_UPDATE: 0, ROW_SOURCE_STATUS_TIMER: 0,
               ROW_SOURCE_RECORDED_POSE: 4352} and
              row_sources == {ROW_SOURCE_RECORDED_POSE: 4352} and
              row_count == 4352 and set(CSV_COLUMNS) == fields and
              summary["tracking_performance_interpretation_prohibited"] and
              not summary["physical_command_publication"],
              "recorded shadow gate/schema mismatch")
    check("recorded Bag-D shadow output gate", recorded_output)

  print(f"result: {len(tests)}/{len(tests)} checks passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
