#!/usr/bin/env python3
"""Pure and post-hoc validation for the Stage-E map-route Isaac closed loop."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable

from map_route_kinematic_validation_v1 import ordered_project
from map_route_reference_v1 import build_map_route_reference
from run_isaac_closed_loop_v1 import DeadlineScheduler
from run_map_route_isaac_closed_loop_v1 import REPO_ROOT, load_stage_config, reference_context_index
from test_tracked_force_plant_v2 import load_config as load_plant_config
from tracking_controller_v1 import ReferenceInterpolator, wrap_to_pi
from tracking_controller_v2_direction_aware import controller_command, load_direction_aware_config


def require(condition: bool, message: str) -> None:
  if not condition:
    raise AssertionError(message)


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
  return abs(left - right) <= tolerance


def load_rows(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as stream:
    return list(csv.DictReader(stream))


def numeric(row: dict[str, str], key: str) -> float:
  return float(row[key])


def independently_feasible(v_raw: float, omega_raw: float, controller: dict[str, Any]) -> tuple[float, ...]:
  limits = controller["command_limits"]
  spacing = float(controller["measured_fixed"]["track_center_distance_m"])
  left_raw = v_raw - .5 * spacing * omega_raw
  right_raw = v_raw + .5 * spacing * omega_raw
  scale = min(1.0,
              float(limits["maximum_abs_body_speed_m_s"]) / abs(v_raw) if v_raw else math.inf,
              float(limits["maximum_abs_yaw_rate_rad_s"]) / abs(omega_raw) if omega_raw else math.inf,
              float(limits["maximum_abs_track_surface_speed_m_s"]) / abs(left_raw) if left_raw else math.inf,
              float(limits["maximum_abs_track_surface_speed_m_s"]) / abs(right_raw) if right_raw else math.inf)
  v, omega = scale * v_raw, scale * omega_raw
  return scale, v, omega, v - .5 * spacing * omega, v + .5 * spacing * omega


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/map_route_isaac_closed_loop_v1.json")
  parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs/map_route_isaac_closed_loop_v1")
  parser.add_argument("--pure-only", action="store_true")
  parser.add_argument("--require-output", action="store_true")
  args = parser.parse_args()
  stage = load_stage_config(args.config.resolve())
  reference_result = build_map_route_reference(REPO_ROOT / stage["baseline_configs"]["map_route_reference"])
  reference = reference_result["trajectory"]
  _, controller = load_direction_aware_config(REPO_ROOT / stage["baseline_configs"]["direction_aware_controller"])
  plant = load_plant_config(REPO_ROOT / stage["baseline_configs"]["tracked_force_plant_v2"])
  tests: list[str] = []

  def check(name: str, function: Callable[[], None]) -> None:
    function(); tests.append(name); print(f"PASS: {name}")

  def config_isolation() -> None:
    require(stage["baseline_configs"] == {
      "map_route_reference": "config/map_route_reference_v1.json",
      "direction_aware_controller": "config/tracking_controller_v2_direction_aware.json",
      "tracked_force_plant_v2": "config/tracked_force_plant_v2.json"}, "noncanonical Stage-E input")
    tunable = plant["tunable_uncalibrated"]
    expected = {"actuator_time_constant_s": .10, "command_delay_s": 0.0, "deadzone_m_s": 0.0,
                "acceleration_limit_m_s2": 0.0, "maximum_surface_speed_m_s": 1.5,
                "k_long_n_per_m_s": 900.0, "k_lat_n_per_m_s": 180.0, "mu_long": .8,
                "mu_lat": .18, "maximum_track_force_n": 250.0,
                "physics_dt_s": 1.0 / 120.0, "settle_duration_s": 1.0}
    require(all(close(float(tunable[key]), value, 1e-14) for key, value in expected.items()), "plant baseline changed")
    require(tunable["support_mode"] == "flat_track_boxes", "wrong support mode")
    require(close(float(plant["measured_fixed"]["track_center_distance_b_m"]), .434), "plant spacing")
    require(close(float(controller["simulation"]["time_step_s"]), .02), "controller period")
    require(close(float(controller["simulation"]["terminal_hold_s"]), 3.0), "terminal hold")
  check("canonical configuration isolation", config_isolation)

  def discrete_metadata() -> None:
    times = [float(row["t_s"]) for row in reference]
    require(set(int(row["motion_direction"]) for row in reference) == {-1, 0, 1}, "missing maneuver mode")
    boundaries = [index for index in range(len(reference) - 1)
                  if int(reference[index]["segment_id"]) != int(reference[index + 1]["segment_id"])]
    require(len(boundaries) == 22, "expected 23 ordered segments")
    for index in boundaries:
      exact = reference_context_index(reference, times, times[index])
      later_time = .5 * (times[index] + times[index + 1])
      later = reference_context_index(reference, times, later_time)
      require(exact == index and later == index + 1, "discrete boundary ownership failed")
  check("discrete mode lookup never interpolates", discrete_metadata)

  def reverse_controller_and_tracks() -> None:
    reverse = next(row for row in reference if int(row["motion_direction"]) == -1 and float(row["v_ref_m_s"]) < -.05)
    sampled = {key: float(reverse[key]) for key in ("t_s", "s_m", "x_m", "y_m", "yaw_rad",
                                                     "curvature_ref_1_m", "v_ref_m_s", "omega_ref_rad_s")}
    command = controller_command((sampled["x_m"], sampled["y_m"], sampled["yaw_rad"]), sampled, -1, controller)
    require(command["motion_direction"] == -1 and close(command["v_raw_m_s"], sampled["v_ref_m_s"]),
            "negative reference was not passed unchanged")
    scale, v, omega, left, right = independently_feasible(-.3, 0.0, controller)
    require(close(scale, 1.0) and close(v, -.3) and close(omega, 0.0) and left < 0 and right < 0,
            "reverse track mapping sign failed")
  check("negative reference and reverse track mapping", reverse_controller_and_tracks)

  def pivot_track_convention() -> None:
    _, _, _, left_positive, right_positive = independently_feasible(0.0, .4, controller)
    _, _, _, left_negative, right_negative = independently_feasible(0.0, -.4, controller)
    require(left_positive < 0 < right_positive and right_negative < 0 < left_negative,
            "pivot differential convention failed")
  check("pivot track direction convention", pivot_track_convention)

  def ordered_projection() -> None:
    progress = []
    half_window = int(stage["runtime"]["ordered_projection_half_window_segments"])
    for index, row in enumerate(reference):
      result = ordered_project(float(row["x_m"]), float(row["y_m"]), reference, index, half_window)
      progress.append(float(result["s_projected_m"]))
    require(all(right + 1e-10 >= left for left, right in zip(progress, progress[1:])), "projection progress reversed")
    require(abs(progress[-1] - float(reference[-1]["s_m"])) < 1e-9, "final projection jumped to start")
  check("ordered projection preserves final branch", ordered_projection)

  def scheduler() -> None:
    schedule = DeadlineScheduler(.02, 1e-12)
    updates = [schedule.maybe_update(index / 120.0) for index in range(120)]
    actual = [row for row in updates if row is not None]
    intervals = [right["control_actual_update_time_s"] - left["control_actual_update_time_s"]
                 for left, right in zip(actual, actual[1:])]
    require(len(actual) == 50 and close(min(intervals), 2 / 120.0, 1e-14) and
            close(max(intervals), 3 / 120.0, 1e-14), "scheduler quantization changed")
  check("Stage-C deadline scheduler semantics", scheduler)

  def frozen_baselines() -> None:
    paths = ["config/map_route_reference_v1.json", "scripts/map_route_reference_v1.py",
             "scripts/generate_map_route_reference_v1.py", "scripts/test_map_route_reference_v1.py",
             "config/tracking_controller_v2_direction_aware.json", "scripts/tracking_controller_v2_direction_aware.py",
             "scripts/test_tracking_controller_v2_direction_aware.py", "config/tracked_force_plant_v2.json",
             "scripts/run_isaac_closed_loop_v1.py", "scripts/test_isaac_closed_loop_v1.py"]
    result = subprocess.run(["git", "diff", "--exit-code", "--", *paths], cwd=REPO_ROOT,
                            check=False, capture_output=True, text=True)
    require(result.returncode == 0 and not result.stdout, "frozen baseline modified")
  check("frozen baseline files unchanged", frozen_baselines)

  output_csv = args.output_dir.resolve() / stage["output"]["csv_filename"]
  output_summary = args.output_dir.resolve() / stage["output"]["summary_filename"]
  if args.pure_only:
    print(f"result: {len(tests)}/{len(tests)} checks passed (pure)")
    return 0
  if not output_csv.exists() or not output_summary.exists():
    if args.require_output:
      raise FileNotFoundError("Stage-E output is required but missing")
    print(f"result: {len(tests)}/{len(tests)} checks passed (output absent; post-hoc skipped)")
    return 0
  rows = load_rows(output_csv)
  with output_summary.open(encoding="utf-8") as stream:
    summary = json.load(stream)

  required = {"physics_time_s", "control_update", "reference_segment_id", "reference_motion_direction",
              "controller_reference_motion_direction", "actual_local_x_m", "actual_local_y_m",
              "actual_local_yaw_rad", "e_x_m", "e_y_m", "heading_error_rad", "v_raw_m_s",
              "omega_raw_rad_s", "v_cmd_m_s", "omega_cmd_rad_s", "command_scale", "v_left_cmd_m_s",
              "v_right_cmd_m_s", "v_left_state_m_s", "v_right_state_m_s", "body_vy_m_s",
              "left_longitudinal_force_n", "right_longitudinal_force_n", "left_lateral_force_n",
              "right_lateral_force_n", "left_normal_load_n", "right_normal_load_n",
              "applied_yaw_moment_n_m", "max_abs_custom_force_normal_component_n", "projected_s_m"}
  check("required output schema", lambda: require(required <= set(rows[0]), "missing output columns"))

  def finite_output() -> None:
    for row in rows:
      for key, value in row.items():
        if value != "": require(math.isfinite(float(value)) if key != "controller_saturation_reasons" else True,
                                f"non-finite {key}")
    require("NaN" not in output_summary.read_text() and "Infinity" not in output_summary.read_text(), "non-finite JSON")
  check("finite CSV and JSON", finite_output)

  def timing_and_final() -> None:
    dt = float(plant["tunable_uncalibrated"]["physics_dt_s"])
    require(close(numeric(rows[-1], "physics_time_s"), float(summary["last_logged_pre_step_time_s"]), 1e-12), "last pre-step")
    require(close(float(summary["actual_final_simulation_time_s"]) - numeric(rows[-1], "physics_time_s"), dt, 1e-12), "post-step time")
    require(close(float(summary["final_post_step_observation"]["physics_time_s"]),
                  float(summary["actual_final_simulation_time_s"]), 1e-12), "final observation time")
    update_rows = [row for row in rows if int(row["control_update"])]
    intervals = [numeric(right, "control_actual_update_time_s") - numeric(left, "control_actual_update_time_s")
                 for left, right in zip(update_rows, update_rows[1:])]
    require(close(min(intervals), 2 * dt, 1e-11) and close(max(intervals), 3 * dt, 1e-11), "update grid")
  check("physics, scheduler, and final post-step timing", timing_and_final)

  def error_and_control_reconstruction() -> None:
    gains = controller["controller"]
    for row in rows:
      dx = numeric(row, "reference_x_m") - numeric(row, "actual_local_x_m")
      dy = numeric(row, "reference_y_m") - numeric(row, "actual_local_y_m")
      yaw = numeric(row, "actual_local_yaw_rad")
      ex = math.cos(yaw) * dx + math.sin(yaw) * dy
      ey = -math.sin(yaw) * dx + math.cos(yaw) * dy
      eh = wrap_to_pi(numeric(row, "reference_yaw_rad") - yaw)
      require(close(ex, numeric(row, "e_x_m")) and close(ey, numeric(row, "e_y_m")) and
              close(eh, numeric(row, "heading_error_rad")), "current error semantics")
      factor = int(float(row["controller_reference_motion_direction"]))
      factor = -1 if factor == -1 else 1
      vraw = numeric(row, "controller_reference_v_ref_m_s") * math.cos(numeric(row, "controller_update_heading_error_rad")) + float(gains["k_longitudinal_1_s"]) * numeric(row, "controller_update_e_x_m")
      oraw = numeric(row, "controller_reference_omega_ref_rad_s") + factor * float(gains["k_lateral_rad_s_per_m"]) * numeric(row, "controller_update_e_y_m") + float(gains["k_heading_1_s"]) * math.sin(numeric(row, "controller_update_heading_error_rad"))
      require(close(vraw, numeric(row, "v_raw_m_s")) and close(oraw, numeric(row, "omega_raw_rad_s")), "held V2 law")
      scale, v, omega, left, right = independently_feasible(vraw, oraw, controller)
      require(all(close(a, b) for a, b in ((scale, numeric(row, "command_scale")), (v, numeric(row, "v_cmd_m_s")),
                                          (omega, numeric(row, "omega_cmd_rad_s")), (left, numeric(row, "v_left_cmd_m_s")),
                                          (right, numeric(row, "v_right_cmd_m_s")))), "feasibility reconstruction")
  check("current errors and held V2 command reconstruct", error_and_control_reconstruction)

  def held_context() -> None:
    for index, row in enumerate(rows):
      require(close(numeric(row, "held_command_age_s"), numeric(row, "physics_time_s") - numeric(row, "controller_reference_t_s")), "held age")
      if int(row["control_update"]):
        require(close(numeric(row, "controller_reference_t_s"), numeric(row, "control_actual_update_time_s")), "update context time")
      elif index:
        for key in ("controller_reference_t_s", "controller_reference_segment_id", "controller_reference_motion_direction",
                    "controller_update_e_x_m", "controller_update_e_y_m", "controller_update_heading_error_rad",
                    "v_raw_m_s", "omega_raw_rad_s", "v_cmd_m_s", "omega_cmd_rad_s", "command_scale"):
          require(row[key] == rows[index - 1][key], f"held field changed: {key}")
  check("controller context is held between updates", held_context)

  def route_and_projection() -> None:
    require(set(int(float(row["reference_segment_id"])) for row in rows) == set(range(23)), "segments missing")
    require(set(int(float(row["reference_motion_direction"])) for row in rows) == {-1, 0, 1}, "modes missing")
    end_s = float(reference[-1]["s_m"])
    final_active = max((row for row in rows if numeric(row, "reference_t_s") < float(reference[-1]["t_s"]) + 1e-10), key=lambda row: numeric(row, "physics_time_s"))
    require(numeric(final_active, "projected_s_m") > .8 * end_s, "final projection jumped to route start")
  check("all maneuver segments and ordered projection", route_and_projection)

  def direction_sanity() -> None:
    reverse = [row for row in rows if int(float(row["reference_motion_direction"])) == -1 and numeric(row, "v_cmd_m_s") < -.05]
    require(reverse and sum(numeric(row, "v_left_cmd_m_s") + numeric(row, "v_right_cmd_m_s") for row in reverse) / (2 * len(reverse)) < 0,
            "reverse track command sign")
    require(sum(numeric(row, "v_left_state_m_s") + numeric(row, "v_right_state_m_s") for row in reverse) / (2 * len(reverse)) < 0,
            "reverse actuator sign")
    require(sum(numeric(row, "v_actual_m_s") for row in reverse) / len(reverse) < 0, "reverse body direction")
    positive = [row for row in rows if int(float(row["reference_motion_direction"])) == 0 and numeric(row, "omega_cmd_rad_s") > .05]
    negative = [row for row in rows if int(float(row["reference_motion_direction"])) == 0 and numeric(row, "omega_cmd_rad_s") < -.05]
    require(all(numeric(row, "v_left_cmd_m_s") < numeric(row, "v_right_cmd_m_s") for row in positive), "positive pivot sign")
    require(all(numeric(row, "v_left_cmd_m_s") > numeric(row, "v_right_cmd_m_s") for row in negative), "negative pivot sign")
  check("reverse and pivot direction sanity", direction_sanity)

  def plant_health() -> None:
    tolerance = float(stage["runtime"]["custom_force_normal_tolerance_n"])
    require(max(abs(numeric(row, "max_abs_custom_force_normal_component_n")) for row in rows) <= tolerance, "force tangent invariant")
    require(all(int(float(row["left_supported"])) and int(float(row["right_supported"])) for row in rows), "support loss")
  check("plant support and tangent-force invariant", plant_health)

  def summary_consistency() -> None:
    reference_end = float(reference[-1]["t_s"])
    active = [row for row in rows if numeric(row, "physics_time_s") <= reference_end + 1e-12]
    fields = (("e_y_m", "rms_e_y_m"), ("heading_error_rad", "rms_heading_error_rad"),
              ("time_aligned_xy_error_m", "rms_time_aligned_xy_error_m"),
              ("cross_track_error_m", "rms_cross_track_error_m"), ("progress_error_m", "rms_progress_error_m"))
    for csv_key, summary_key in fields:
      expected = math.sqrt(sum(numeric(row, csv_key) ** 2 for row in active) / len(active))
      require(close(expected, float(summary["active"][summary_key]), 1e-11), f"summary mismatch {summary_key}")
    final = summary["final_post_step_observation"]
    require(close(math.hypot(float(final["actual_local_x_m"]) - float(reference[-1]["x_m"]),
                             float(final["actual_local_y_m"]) - float(reference[-1]["y_m"])),
                  float(summary["final_xy_goal_error_m"]), 1e-11), "final goal summary")
  check("summary metrics independently recompute", summary_consistency)

  print(f"result: {len(tests)}/{len(tests)} checks passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
