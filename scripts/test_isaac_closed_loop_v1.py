#!/usr/bin/env python3
"""Pure-helper and post-hoc validation for Isaac Closed Loop V1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_isaac_closed_loop_v1 import (  # noqa: E402
  DeadlineScheduler, continuous_angle_update, load_stage_config,
  world_to_settled_local_xy,
)


def require(condition: bool, message: str) -> None:
  if not condition:
    raise AssertionError(message)


def rms(values: list[float]) -> float:
  return math.sqrt(sum(value * value for value in values) / len(values))


def load_json(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    return json.load(stream)


def pure_tests(stage_path: Path) -> tuple[list[tuple[str, Callable[[], None]]], dict[str, Any]]:
  stage = load_stage_config(stage_path)
  paths = {name: REPO_ROOT / value for name, value in stage["baseline_configs"].items()}
  controller = load_json(paths["tracking_controller"])
  plant = load_json(paths["tracked_force_plant_v2"])
  dt = float(plant["tunable_uncalibrated"]["physics_dt_s"])
  period = float(controller["simulation"]["time_step_s"])
  tolerance = float(stage["runtime"]["scheduler_time_tolerance_s"])

  def configuration_links() -> None:
    require(all(path.is_file() for path in paths.values()), "a baseline config link does not exist")
    b_controller = float(controller["measured_fixed"]["track_center_distance_m"])
    b_plant = float(plant["measured_fixed"]["track_center_distance_b_m"])
    require(b_controller == b_plant == 0.434, "controller/plant B mismatch")
    require(dt == 0.008333333333333333, "V2 physics timestep changed")
    require(period == 0.02, "Tracking Controller V1 period changed")
    require(plant["tunable_uncalibrated"]["support_mode"] == "flat_track_boxes",
            "flat-track support baseline changed")

  def local_frame() -> None:
    x0, y0, yaw0 = 2.0, -1.0, 0.7
    require(max(abs(value) for value in world_to_settled_local_xy(x0, y0, x0, y0, yaw0)) < 1e-14,
            "settled origin does not map to zero")
    forward = world_to_settled_local_xy(
      x0 + math.cos(yaw0), y0 + math.sin(yaw0), x0, y0, yaw0)
    left = world_to_settled_local_xy(
      x0 - math.sin(yaw0), y0 + math.cos(yaw0), x0, y0, yaw0)
    require(abs(forward[0] - 1.0) < 1e-14 and abs(forward[1]) < 1e-14,
            "settled forward does not map to local +X")
    require(abs(left[0]) < 1e-14 and abs(left[1] - 1.0) < 1e-14,
            "settled left does not map to local +Y")
    require(abs((yaw0 + 0.3) - yaw0 - 0.3) < 1e-14, "relative yaw convention failed")

  def continuous_yaw() -> None:
    previous = math.pi - 0.05
    updated = continuous_angle_update(previous, -math.pi + 0.04)
    require(abs(updated - (math.pi + 0.04)) < 1e-14, "+pi crossing unwrap failed")
    reverse = continuous_angle_update(-math.pi + 0.05, math.pi - 0.04)
    require(abs(reverse - (-math.pi - 0.04)) < 1e-14, "-pi crossing unwrap failed")

  def scheduler() -> None:
    scheduler_state = DeadlineScheduler(period, tolerance)
    updates = []
    for step in range(round(20.0 / dt)):
      update = scheduler_state.maybe_update(step * dt)
      if update is not None:
        updates.append(update)
    times = [float(update["control_actual_update_time_s"]) for update in updates]
    lateness = [float(update["control_update_lateness_s"]) for update in updates]
    intervals = [times[index] - times[index - 1] for index in range(1, len(times))]
    require(times[0] == 0.0, "first scheduler update is not at zero")
    require(all(times[index] > times[index - 1] for index in range(1, len(times))),
            "scheduler updates are not monotonic")
    require(all(0.0 <= value < dt + tolerance for value in lateness), "scheduler lateness bound failed")
    expected = (2.0 * dt, 3.0 * dt)
    require(all(any(abs(value - candidate) < 1e-12 for candidate in expected) for value in intervals),
            "scheduler interval is not a two/three-step quantization")
    require(abs(sum(intervals) / len(intervals) - period) < dt / len(intervals) + 1e-12,
            "long-run scheduler mean is not near target")

  return ([("configuration links", configuration_links), ("settled local frame", local_frame),
           ("continuous yaw", continuous_yaw), ("deadline scheduler", scheduler)],
          {"stage": stage, "controller": controller, "plant": plant, "dt": dt, "period": period})


def posthoc_tests(output_dir: Path, context: dict[str, Any]) -> list[tuple[str, Callable[[], None]]]:
  stage, controller, plant = context["stage"], context["controller"], context["plant"]
  csv_path = output_dir / stage["output"]["csv_filename"]
  summary_path = output_dir / stage["output"]["summary_filename"]
  with csv_path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
  summary = load_json(summary_path)
  dt, period = context["dt"], context["period"]
  tolerance = float(stage["runtime"]["scheduler_time_tolerance_s"])

  def all_finite() -> None:
    required = [key for key in rows[0] if key not in (
      "control_target_time_s", "control_actual_update_time_s", "control_update_lateness_s")]
    for index, row in enumerate(rows):
      require(all(math.isfinite(float(row[key])) for key in required),
              f"row {index} contains a non-finite required value")

  def csv_timing() -> None:
    times = [float(row["physics_time_s"]) for row in rows]
    require(all(times[index] > times[index - 1] for index in range(1, len(times))),
            "physics timestamps are not strictly increasing")
    require(all(abs((times[index] - times[index - 1]) - dt) < 1e-10
                for index in range(1, len(times))), "physics interval differs from V2 dt")
    reference_end = float(summary["reference_duration_s"])
    require(any(time <= reference_end for time in times) and any(time > reference_end for time in times),
            "active or terminal-hold samples are missing")
    updates = [row for row in rows if int(row["control_update"]) == 1]
    require(all(0.0 <= float(row["control_update_lateness_s"]) < dt + tolerance for row in updates),
            "generated scheduler lateness bound failed")

  def command_feasibility_and_mapping() -> None:
    limits = controller["command_limits"]
    body_max = float(limits["maximum_abs_body_speed_m_s"])
    yaw_max = float(limits["maximum_abs_yaw_rate_rad_s"])
    track_max = float(limits["maximum_abs_track_surface_speed_m_s"])
    spacing = float(controller["measured_fixed"]["track_center_distance_m"])
    for index, row in enumerate(rows):
      v, omega = float(row["v_cmd_m_s"]), float(row["omega_cmd_rad_s"])
      left, right = float(row["v_left_cmd_m_s"]), float(row["v_right_cmd_m_s"])
      require(abs(v) <= body_max + 1e-10 and abs(omega) <= yaw_max + 1e-10,
              f"row {index} body command infeasible")
      require(abs(left) <= track_max + 1e-10 and abs(right) <= track_max + 1e-10,
              f"row {index} track command infeasible")
      require(0.0 < float(row["command_scale"]) <= 1.0, f"row {index} invalid scale")
      require(abs(left - (v - 0.5 * spacing * omega)) < 1e-10 and
              abs(right - (v + 0.5 * spacing * omega)) < 1e-10,
              f"row {index} differential-track mapping mismatch")

  def current_diagnostic_errors() -> None:
    def wrap(angle: float) -> float:
      return (angle + math.pi) % (2.0 * math.pi) - math.pi
    for index, row in enumerate(rows):
      actual_x, actual_y = float(row["actual_local_x_m"]), float(row["actual_local_y_m"])
      actual_yaw = float(row["actual_local_yaw_rad"])
      dx = float(row["reference_x_m"]) - actual_x
      dy = float(row["reference_y_m"]) - actual_y
      expected_x = math.cos(actual_yaw) * dx + math.sin(actual_yaw) * dy
      expected_y = -math.sin(actual_yaw) * dx + math.cos(actual_yaw) * dy
      expected_heading = wrap(float(row["reference_yaw_rad"]) - actual_yaw)
      require(abs(float(row["e_x_m"]) - expected_x) < 1e-12,
              f"row {index} current e_x timestamp mismatch")
      require(abs(float(row["e_y_m"]) - expected_y) < 1e-12,
              f"row {index} current e_y timestamp mismatch")
      require(abs(float(row["heading_error_rad"]) - expected_heading) < 1e-12,
              f"row {index} current heading-error timestamp mismatch")

  def held_controller_context() -> None:
    gains = controller["controller"]
    held_fields = (
      "controller_reference_t_s", "controller_reference_s_m",
      "controller_reference_v_ref_m_s", "controller_reference_omega_ref_rad_s",
      "controller_update_e_x_m", "controller_update_e_y_m",
      "controller_update_heading_error_rad", "v_raw_m_s", "omega_raw_rad_s",
      "v_cmd_m_s", "omega_cmd_rad_s", "v_left_cmd_m_s", "v_right_cmd_m_s",
      "command_scale")
    for index, row in enumerate(rows):
      physics_time = float(row["physics_time_s"])
      controller_time = float(row["controller_reference_t_s"])
      require(controller_time <= physics_time + tolerance,
              f"row {index} controller context comes from the future")
      require(abs(float(row["held_command_age_s"]) - (physics_time - controller_time)) < 1e-12,
              f"row {index} held-command age mismatch")
      if int(row["control_update"]) == 1:
        require(abs(controller_time - float(row["control_actual_update_time_s"])) < 1e-12,
                f"row {index} controller reference/update time mismatch")
        for context_field, current_field in (
            ("controller_reference_s_m", "reference_s_m"),
            ("controller_reference_v_ref_m_s", "v_ref_m_s"),
            ("controller_reference_omega_ref_rad_s", "omega_ref_rad_s"),
            ("controller_update_e_x_m", "e_x_m"),
            ("controller_update_e_y_m", "e_y_m"),
            ("controller_update_heading_error_rad", "heading_error_rad")):
          require(abs(float(row[context_field]) - float(row[current_field])) < 1e-12,
                  f"row {index} update context {context_field} mismatch")
      else:
        require(index > 0, "first row is not a control update")
        require(all(row[field] == rows[index - 1][field] for field in held_fields),
                f"row {index} held controller context/command changed without update")
      heading = float(row["controller_update_heading_error_rad"])
      expected_v_raw = (float(row["controller_reference_v_ref_m_s"]) * math.cos(heading)
                        + float(gains["k_longitudinal_1_s"]) *
                        float(row["controller_update_e_x_m"]))
      expected_omega_raw = (float(row["controller_reference_omega_ref_rad_s"])
                            + float(gains["k_lateral_rad_s_per_m"]) *
                            float(row["controller_update_e_y_m"])
                            + float(gains["k_heading_1_s"]) * math.sin(heading))
      require(abs(float(row["v_raw_m_s"]) - expected_v_raw) < 1e-12,
              f"row {index} held v_raw is not reconstructable")
      require(abs(float(row["omega_raw_rad_s"]) - expected_omega_raw) < 1e-12,
              f"row {index} held omega_raw is not reconstructable")

  def evaluation_summary() -> None:
    reference_end = float(summary["reference_duration_s"])
    active = [row for row in rows if float(row["physics_time_s"]) <= reference_end +
              float(stage["runtime"]["active_window_time_tolerance_s"])]
    for prefix, window in (("active", active), ("full_run", rows)):
      require(int(summary[f"{prefix}_sample_count"]) == len(window), f"{prefix} sample count mismatch")
      for field in ("cross_track_error_m", "heading_error_rad", "progress_error_m",
                    "speed_error_m_s", "yaw_rate_error_rad_s"):
        values = [float(row[field]) for row in window]
        require(abs(float(summary[f"{prefix}_rms_{field}"]) - rms(values)) < 1e-12,
                f"{prefix} RMS {field} mismatch")
        require(abs(float(summary[f"{prefix}_maximum_abs_{field}"]) - max(abs(v) for v in values)) < 1e-12,
                f"{prefix} maximum {field} mismatch")

  def update_saturation() -> None:
    updates = [row for row in rows if int(row["control_update"]) == 1]
    scale_tolerance = float(stage["runtime"]["saturation_scale_tolerance"])
    count = sum(float(row["command_scale"]) < 1.0 - scale_tolerance for row in updates)
    timing = summary["control_timing"]
    require(int(timing["update_count"]) == len(updates), "control update count mismatch")
    require(int(timing["saturated_control_update_count"]) == count, "update saturation count mismatch")
    require(abs(float(timing["saturated_control_update_fraction"]) - count / len(updates)) < 1e-12,
            "update saturation fraction mismatch")

  def terminal_and_initial() -> None:
    require(float(rows[-1]["v_ref_m_s"]) == 0.0 and float(rows[-1]["omega_ref_rad_s"]) == 0.0,
            "final reference is not stopped")
    initial_tolerance = float(stage["runtime"]["local_initial_pose_tolerance"])
    require(max(abs(float(rows[0][key])) for key in
                ("actual_local_x_m", "actual_local_y_m", "actual_local_yaw_rad")) <= initial_tolerance,
            "settled local initial pose is not near zero")

  def final_post_step_observation() -> None:
    final = summary["final_post_step_observation"]
    required = ("physics_time_s", "reference_t_s", "reference_s_m", "reference_x_m",
                "reference_y_m", "reference_yaw_rad", "v_ref_m_s", "omega_ref_rad_s",
                "actual_local_x_m", "actual_local_y_m", "actual_local_yaw_rad",
                "v_actual_m_s", "omega_actual_rad_s", "v_cmd_m_s", "omega_cmd_rad_s")
    require(all(key in final and math.isfinite(float(final[key])) for key in required),
            "final post-step observation is missing or non-finite")
    last_pre = float(summary["last_logged_pre_step_time_s"])
    final_time = float(summary["actual_final_simulation_time_s"])
    require(last_pre == float(rows[-1]["physics_time_s"]), "last pre-step summary time mismatch")
    require(final_time > last_pre, "final post-step time is not after last pre-step time")
    require(abs((final_time - last_pre) - dt) < 1e-12,
            "final post-step/pre-step time difference is not one physics dt")
    require(float(final["physics_time_s"]) == final_time, "final observation time mismatch")
    require(float(final["v_ref_m_s"]) == 0.0 and float(final["omega_ref_rad_s"]) == 0.0,
            "final post-step reference is not stopped")
    goal_x, goal_y, goal_yaw = (float(final["reference_x_m"]), float(final["reference_y_m"]),
                                float(final["reference_yaw_rad"]))
    xy_error = math.hypot(float(final["actual_local_x_m"]) - goal_x,
                          float(final["actual_local_y_m"]) - goal_y)
    heading_error = (goal_yaw - float(final["actual_local_yaw_rad"]) + math.pi) % (2 * math.pi) - math.pi
    comparisons = {
      "final_xy_error_to_reference_goal_m": xy_error,
      "final_heading_error_rad": heading_error,
      "final_abs_v_cmd_m_s": abs(float(final["v_cmd_m_s"])),
      "final_abs_omega_cmd_rad_s": abs(float(final["omega_cmd_rad_s"])),
      "final_actual_body_speed_m_s": float(final["v_actual_m_s"]),
      "final_actual_body_yaw_rate_rad_s": float(final["omega_actual_rad_s"]),
    }
    require(all(abs(float(summary[key]) - value) < 1e-12 for key, value in comparisons.items()),
            "top-level final metric does not match final post-step observation")

  def plant_structure() -> None:
    required = ("left_contact_count", "right_contact_count", "left_normal_load_n",
                "right_normal_load_n", "left_supported", "right_supported",
                "max_abs_custom_force_normal_component_n")
    require(all(key in rows[0] for key in required), "support/contact diagnostics are missing")
    require(all(int(row["left_supported"]) or int(row["right_supported"]) for row in rows),
            "both sides lost support simultaneously")
    normal_tolerance = float(stage["runtime"]["custom_force_normal_tolerance_n"])
    require(max(abs(float(row["max_abs_custom_force_normal_component_n"])) for row in rows)
            <= normal_tolerance, "custom force violated contact-tangent invariant")

  return [("generated finite values", all_finite), ("generated CSV timing", csv_timing),
          ("generated command feasibility and mapping", command_feasibility_and_mapping),
          ("current diagnostic errors", current_diagnostic_errors),
          ("held controller context", held_controller_context),
          ("active/full evaluation metrics", evaluation_summary),
          ("control-update saturation", update_saturation),
          ("terminal and local-origin behavior", terminal_and_initial),
          ("final post-step observation", final_post_step_observation),
          ("plant structural sanity", plant_structure)]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/isaac_closed_loop_v1.json")
  parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs/isaac_closed_loop_v1")
  parser.add_argument("--pure-only", action="store_true")
  args = parser.parse_args()
  tests, context = pure_tests(args.config.resolve())
  if not args.pure_only:
    tests += posthoc_tests(args.output_dir.resolve(), context)
  failures = []
  for name, test in tests:
    try:
      test(); print(f"PASS: {name}")
    except Exception as error:
      failures.append((name, error)); print(f"FAIL: {name}: {error}")
  print(f"result: {len(tests) - len(failures)}/{len(tests)} checks passed")
  return 1 if failures else 0


if __name__ == "__main__":
  raise SystemExit(main())
