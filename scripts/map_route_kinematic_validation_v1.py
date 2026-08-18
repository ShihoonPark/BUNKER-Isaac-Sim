#!/usr/bin/env python3
"""Pure kinematic compatibility diagnostics for maneuver-aware map references."""

from __future__ import annotations

import bisect
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

from map_route_reference_v1 import build_map_route_reference
from tracking_controller_v1 import (
  ReferenceInterpolator, body_frame_errors, controller_command,
  integrate_unicycle_exact, load_controller_config, project_to_polyline,
  wrap_to_pi,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_validation_config(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    config = json.load(stream)
  for relative in config["baseline_configs"].values():
    if not (REPO_ROOT / relative).is_file():
      raise ValueError(f"missing baseline config: {relative}")
  if int(config["evaluation"]["ordered_projection_half_window_segments"]) < 2:
    raise ValueError("ordered projection window must contain at least two segments")
  return config


def _profile_speed(time_s: float, sign: int, speed: float, cruise: float,
                   accel: float, decel: float) -> tuple[float, float]:
  accel_time = speed / accel
  decel_time = speed / decel
  if time_s < accel_time:
    return sign * accel * time_s, sign * accel
  if time_s < accel_time + cruise:
    return sign * speed, 0.0
  if time_s < accel_time + cruise + decel_time:
    remaining = accel_time + cruise + decel_time - time_s
    return sign * decel * remaining, -sign * decel
  return 0.0, 0.0


def _sample_times(duration: float, dt: float) -> list[float]:
  count = math.floor(duration / dt)
  times = [index * dt for index in range(count + 1)]
  if duration - times[-1] > 1e-12:
    times.append(duration)
  else:
    times[-1] = duration
  return times


def build_synthetic_reference(kind: str, controller_config: dict[str, Any],
                              validation_config: dict[str, Any]) -> list[dict[str, Any]]:
  """Build deterministic piecewise-command references using exact signed integration."""
  settings = validation_config["synthetic_references"]
  limits = controller_config["simulation"]
  trajectory_limits = {
    "accel": 0.35,
    "decel": 0.45,
  }
  dt = float(limits["time_step_s"])
  speed = float(settings["straight_speed_m_s"])
  cruise = float(settings["straight_cruise_duration_s"])
  accel, decel = trajectory_limits["accel"], trajectory_limits["decel"]
  straight_duration = speed / accel + cruise + speed / decel

  if kind in ("forward_straight", "reverse_straight", "reverse_curve"):
    sign = 1 if kind == "forward_straight" else -1
    duration = straight_duration
    curvature = float(settings["reverse_curve_curvature_1_m"]) if kind == "reverse_curve" else 0.0

    def command(time_s: float) -> tuple[float, float, float]:
      v, a = _profile_speed(time_s, sign, speed, cruise, accel, decel)
      omega = abs(v) * curvature
      return v, omega, a
  elif kind == "forward_reverse_transition":
    stop = float(settings["transition_stop_duration_s"])
    first_duration = speed / accel + 3.0 + speed / decel
    reverse_duration = speed / accel + 5.0 + speed / decel
    duration = first_duration + stop + reverse_duration

    def command(time_s: float) -> tuple[float, float, float]:
      if time_s <= first_duration:
        return (*_profile_speed(time_s, 1, speed, 3.0, accel, decel)[:1], 0.0,
                _profile_speed(time_s, 1, speed, 3.0, accel, decel)[1])
      if time_s < first_duration + stop:
        return 0.0, 0.0, 0.0
      v, a = _profile_speed(time_s - first_duration - stop, -1, speed, 5.0, accel, decel)
      return v, 0.0, a
  elif kind == "pivot":
    duration = float(settings["pivot_duration_s"])
    angle = float(settings["pivot_angle_rad"])

    def command(time_s: float) -> tuple[float, float, float]:
      r = min(1.0, max(0.0, time_s / duration))
      omega = angle * (6.0 * r - 6.0 * r * r) / duration
      return 0.0, omega, 0.0
  else:
    raise ValueError(f"unknown synthetic reference: {kind}")

  times = _sample_times(duration, dt)
  poses = [(0.0, 0.0, 0.0)]
  s_values = [0.0]
  commands = [command(time_s) for time_s in times]
  for index in range(len(times) - 1):
    interval = times[index + 1] - times[index]
    v, omega, _ = commands[index]
    poses.append(integrate_unicycle_exact(poses[-1], v, omega, interval,
                                          float(limits["near_zero_yaw_rate_rad_s"])))
    s_values.append(s_values[-1] + abs(v) * interval)
  rows = []
  for index, (time_s, pose, values) in enumerate(zip(times, poses, commands)):
    v, omega, acceleration = values
    if index == len(times) - 1:
      v, omega, acceleration = 0.0, 0.0, 0.0
    direction = 0 if kind == "pivot" else (1 if v > 0.0 else (-1 if v < 0.0 else 0))
    curvature = float(settings["reverse_curve_curvature_1_m"]) if kind == "reverse_curve" and direction else 0.0
    rows.append({"index": index, "t_s": time_s, "dt_s": 0.0 if index == 0 else time_s - times[index - 1],
                 "s_m": s_values[index], "x_m": pose[0], "y_m": pose[1], "yaw_rad": pose[2],
                 "curvature_ref_1_m": curvature, "curvature_numeric_1_m": curvature,
                 "v_ref_m_s": v, "omega_ref_rad_s": omega, "a_ref_m_s2": acceleration,
                 "segment_id": 0, "segment_type": kind, "motion_direction": direction,
                 "curvature_valid": 0 if kind == "pivot" else 1})
  return rows


def ordered_project(x: float, y: float, reference_rows: list[dict[str, Any]],
                    expected_index: int, half_window: int) -> dict[str, float]:
  """Project only onto an ordered local reference neighborhood for metrics."""
  start = max(0, expected_index - half_window)
  end = min(len(reference_rows), expected_index + half_window + 2)
  local = reference_rows[start:end]
  if not any(math.hypot(float(second["x_m"]) - float(first["x_m"]),
                        float(second["y_m"]) - float(first["y_m"])) > 1e-15
             for first, second in zip(local, local[1:])):
    reference = reference_rows[expected_index]
    return {"projected_x_m": float(reference["x_m"]),
            "projected_y_m": float(reference["y_m"]),
            "s_projected_m": float(reference["s_m"]),
            "cross_track_error_m": math.hypot(x - float(reference["x_m"]),
                                                y - float(reference["y_m"])),
            "projection_segment_index": expected_index}
  projection = project_to_polyline(x, y, local)
  projection["projection_segment_index"] += start
  return projection


def simulate(reference_rows: list[dict[str, Any]], initial_pose: tuple[float, float, float],
             controller_config: dict[str, Any], validation_config: dict[str, Any],
             name: str) -> list[dict[str, Any]]:
  dt = float(controller_config["simulation"]["time_step_s"])
  hold = float(controller_config["simulation"]["terminal_hold_s"])
  end_time = float(reference_rows[-1]["t_s"]) + hold
  times = _sample_times(end_time, dt)
  interpolator = ReferenceInterpolator(reference_rows)
  reference_times = [float(row["t_s"]) for row in reference_rows]
  half_window = int(validation_config["evaluation"]["ordered_projection_half_window_segments"])
  pose = initial_pose
  rows = []
  for step, time_s in enumerate(times):
    reference = interpolator.sample(time_s)
    command = controller_command(pose, reference, controller_config)
    limits = controller_config["command_limits"]
    raw_left = float(command["v_raw_m_s"]) - .5 * float(controller_config["measured_fixed"]["track_center_distance_m"]) * float(command["omega_raw_rad_s"])
    raw_right = float(command["v_raw_m_s"]) + .5 * float(controller_config["measured_fixed"]["track_center_distance_m"]) * float(command["omega_raw_rad_s"])
    saturation_reasons = []
    if abs(float(command["v_raw_m_s"])) > float(limits["maximum_abs_body_speed_m_s"]): saturation_reasons.append("body")
    if abs(float(command["omega_raw_rad_s"])) > float(limits["maximum_abs_yaw_rate_rad_s"]): saturation_reasons.append("yaw")
    if abs(raw_left) > float(limits["maximum_abs_track_surface_speed_m_s"]): saturation_reasons.append("left_track")
    if abs(raw_right) > float(limits["maximum_abs_track_surface_speed_m_s"]): saturation_reasons.append("right_track")
    expected_index = max(0, min(len(reference_rows) - 1,
                                bisect.bisect_right(reference_times, min(time_s, reference_times[-1])) - 1))
    if (expected_index + 1 < len(reference_rows) and
        int(reference_rows[expected_index].get("segment_id", 0)) !=
        int(reference_rows[expected_index + 1].get("segment_id", 0)) and
        time_s > reference_times[expected_index] + 1e-12):
      expected_index += 1
    projection = ordered_project(pose[0], pose[1], reference_rows, expected_index, half_window)
    source_row = reference_rows[expected_index]
    xy_error = math.hypot(reference["x_m"] - pose[0], reference["y_m"] - pose[1])
    rows.append({
      "scenario": name, "time_s": time_s, "reference_index": expected_index,
      "segment_id": int(source_row.get("segment_id", 0)),
      "segment_type": str(source_row.get("segment_type", "synthetic")),
      "motion_direction": int(source_row.get("motion_direction", 0)),
      "reference_s_m": reference["s_m"], "reference_x_m": reference["x_m"],
      "reference_y_m": reference["y_m"], "reference_yaw_rad": reference["yaw_rad"],
      "actual_x_m": pose[0], "actual_y_m": pose[1], "actual_yaw_rad": pose[2],
      "e_x_m": command["e_x_m"], "e_y_m": command["e_y_m"],
      "heading_error_rad": command["e_heading_rad"], "time_aligned_xy_error_m": xy_error,
      "projected_s_m": projection["s_projected_m"],
      "cross_track_error_m": projection["cross_track_error_m"],
      "progress_error_m": reference["s_m"] - projection["s_projected_m"],
      "projection_segment_index": int(projection["projection_segment_index"]),
      "v_ref_m_s": reference["v_ref_m_s"], "omega_ref_rad_s": reference["omega_ref_rad_s"],
      "v_raw_m_s": command["v_raw_m_s"], "omega_raw_rad_s": command["omega_raw_rad_s"],
      "v_cmd_m_s": command["v_cmd_m_s"], "omega_cmd_rad_s": command["omega_cmd_rad_s"],
      "v_left_cmd_m_s": command["v_left_cmd_m_s"], "v_right_cmd_m_s": command["v_right_cmd_m_s"],
      "command_scale": command["command_scale"],
      "saturation_reasons": "+".join(saturation_reasons) if saturation_reasons else "none"})
    if step + 1 < len(times):
      pose = integrate_unicycle_exact(pose, command["v_cmd_m_s"], command["omega_cmd_rad_s"],
                                      times[step + 1] - time_s,
                                      float(controller_config["simulation"]["near_zero_yaw_rate_rad_s"]))
  return rows


def rms(values: list[float]) -> float:
  return math.sqrt(sum(value * value for value in values) / len(values))


def metric_block(rows: list[dict[str, Any]], saturation_tolerance: float) -> dict[str, Any]:
  fields = ("e_x_m", "e_y_m", "heading_error_rad", "time_aligned_xy_error_m",
            "cross_track_error_m", "progress_error_m")
  result: dict[str, Any] = {"sample_count": len(rows)}
  for field in fields:
    values = [float(row[field]) for row in rows]
    label = field.removesuffix("_m").removesuffix("_rad")
    result[f"rms_{label}"] = rms(values)
    result[f"maximum_abs_{label}"] = max(abs(value) for value in values)
  saturated = sum(float(row["command_scale"]) < 1.0 - saturation_tolerance for row in rows)
  reason_counts = {reason: sum(reason in str(row["saturation_reasons"]).split("+") for row in rows)
                   for reason in ("body", "yaw", "left_track", "right_track")}
  result.update({
    "v_ref_range_m_s": [min(float(row["v_ref_m_s"]) for row in rows), max(float(row["v_ref_m_s"]) for row in rows)],
    "v_cmd_range_m_s": [min(float(row["v_cmd_m_s"]) for row in rows), max(float(row["v_cmd_m_s"]) for row in rows)],
    "omega_cmd_range_rad_s": [min(float(row["omega_cmd_rad_s"]) for row in rows), max(float(row["omega_cmd_rad_s"]) for row in rows)],
    "rms_v_cmd_m_s": rms([float(row["v_cmd_m_s"]) for row in rows]),
    "rms_omega_cmd_rad_s": rms([float(row["omega_cmd_rad_s"]) for row in rows]),
    "saturated_sample_count": saturated, "saturated_sample_fraction": saturated / len(rows),
    "saturation_reason_counts": reason_counts,
    "minimum_command_scale": min(float(row["command_scale"]) for row in rows)})
  return result


def scenario_summary(rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]],
                     validation_config: dict[str, Any]) -> dict[str, Any]:
  end = float(reference_rows[-1]["t_s"])
  active = [row for row in rows if float(row["time_s"]) <= end + 1e-12]
  early_s = float(validation_config["evaluation"]["early_window_s"])
  late_s = float(validation_config["evaluation"]["late_window_s"])
  early = [row for row in active if float(row["time_s"]) <= early_s]
  late = [row for row in active if float(row["time_s"]) >= end - late_s]
  tolerance = float(validation_config["evaluation"]["saturation_tolerance"])
  error_norm = lambda row: math.hypot(float(row["e_y_m"]), float(row["heading_error_rad"]))
  segment_metrics = {}
  for segment_id in sorted({int(row["segment_id"]) for row in active}):
    segment_rows = [row for row in active if int(row["segment_id"]) == segment_id]
    segment_metrics[str(segment_id)] = {"segment_type": segment_rows[0]["segment_type"],
                                        **metric_block(segment_rows, tolerance)}
    if str(segment_rows[0]["segment_type"]).lower() == "pivot":
      segment_metrics[str(segment_id)].update({
        "reference_yaw_change_rad": float(segment_rows[-1]["reference_yaw_rad"]) - float(segment_rows[0]["reference_yaw_rad"]),
        "actual_yaw_change_rad": float(segment_rows[-1]["actual_yaw_rad"]) - float(segment_rows[0]["actual_yaw_rad"]),
        "final_pivot_heading_error_rad": float(segment_rows[-1]["heading_error_rad"]),
        "maximum_abs_translational_command_m_s": max(abs(float(row["v_cmd_m_s"])) for row in segment_rows)})
  modes = {}
  for mode in ("forward", "reverse", "pivot"):
    selected = [row for row in active if str(row["segment_type"]).lower() == mode]
    if selected:
      modes[mode] = metric_block(selected, tolerance)
  return {
    "reference_end_time_s": end,
    "active": metric_block(active, tolerance), "full_run": metric_block(rows, tolerance),
    "early_error_norm_rms": rms([error_norm(row) for row in early]),
    "late_active_error_norm_rms": rms([error_norm(row) for row in late]),
    "initial_error_norm": error_norm(rows[0]),
    "final_active_error_norm": error_norm(active[-1]),
    "terminal_error_norm": error_norm(rows[-1]),
    "final_xy_goal_error_m": math.hypot(float(rows[-1]["actual_x_m"]) - float(reference_rows[-1]["x_m"]),
                                         float(rows[-1]["actual_y_m"]) - float(reference_rows[-1]["y_m"])),
    "final_heading_error_rad": float(rows[-1]["heading_error_rad"]),
    "mode_metrics": modes, "segment_metrics": segment_metrics}


def error_dynamics(state: tuple[float, float, float], v_ref: float,
                   controller_config: dict[str, Any]) -> tuple[float, float, float]:
  """Exact continuous straight-reference body-error dynamics under Controller V1."""
  e_x, e_y, e_heading = state
  reference = {"x_m": e_x, "y_m": e_y, "yaw_rad": e_heading,
               "v_ref_m_s": v_ref, "omega_ref_rad_s": 0.0}
  command = controller_command((0.0, 0.0, 0.0), reference, controller_config)
  v, omega = command["v_cmd_m_s"], command["omega_cmd_rad_s"]
  return (omega * e_y - v + v_ref * math.cos(e_heading),
          -omega * e_x + v_ref * math.sin(e_heading), -omega)


def analytic_jacobian(v_ref: float, controller_config: dict[str, Any]) -> list[list[float]]:
  gains = controller_config["controller"]
  return [[-float(gains["k_longitudinal_1_s"]), 0.0, 0.0],
          [0.0, 0.0, v_ref],
          [0.0, -float(gains["k_lateral_rad_s_per_m"]), -float(gains["k_heading_1_s"])]]


def finite_difference_jacobian(v_ref: float, controller_config: dict[str, Any],
                               epsilon: float) -> list[list[float]]:
  result = [[0.0] * 3 for _ in range(3)]
  for column in range(3):
    positive = [0.0, 0.0, 0.0]; negative = [0.0, 0.0, 0.0]
    positive[column] = epsilon; negative[column] = -epsilon
    plus = error_dynamics(tuple(positive), v_ref, controller_config)
    minus = error_dynamics(tuple(negative), v_ref, controller_config)
    for row in range(3):
      result[row][column] = (plus[row] - minus[row]) / (2.0 * epsilon)
  return result


def straight_eigenvalues(v_ref: float, controller_config: dict[str, Any]) -> list[float]:
  gains = controller_config["controller"]
  kx = float(gains["k_longitudinal_1_s"]); ky = float(gains["k_lateral_rad_s_per_m"])
  kh = float(gains["k_heading_1_s"])
  discriminant = kh * kh - 4.0 * ky * v_ref
  root = math.sqrt(discriminant)
  return sorted([-kx, (-kh - root) / 2.0, (-kh + root) / 2.0])


def empirical_growth_rate(rows: list[dict[str, Any]], minimum_error: float) -> dict[str, Any]:
  eligible = [row for row in rows if abs(float(row["v_ref_m_s"])) >= .29 and
              abs(float(row["omega_ref_rad_s"])) < 1e-12 and
              float(row["command_scale"]) >= 1.0 - 1e-12 and
              math.hypot(float(row["e_y_m"]), float(row["heading_error_rad"])) >= minimum_error]
  if len(eligible) < 3:
    return {"sample_count": len(eligible), "rate_1_s": None}
  times = [float(row["time_s"]) for row in eligible]
  logs = [math.log(math.hypot(float(row["e_y_m"]), float(row["heading_error_rad"]))) for row in eligible]
  mean_t = sum(times) / len(times); mean_y = sum(logs) / len(logs)
  denominator = sum((value - mean_t) ** 2 for value in times)
  rate = sum((time - mean_t) * (value - mean_y) for time, value in zip(times, logs)) / denominator
  return {"sample_count": len(eligible), "start_time_s": times[0], "end_time_s": times[-1], "rate_1_s": rate}


CSV_FIELDS = [
  "scenario", "time_s", "reference_index", "segment_id", "segment_type", "motion_direction",
  "reference_s_m", "reference_x_m", "reference_y_m", "reference_yaw_rad",
  "actual_x_m", "actual_y_m", "actual_yaw_rad", "e_x_m", "e_y_m", "heading_error_rad",
  "time_aligned_xy_error_m", "projected_s_m", "cross_track_error_m", "progress_error_m",
  "projection_segment_index", "v_ref_m_s", "omega_ref_rad_s", "v_raw_m_s", "omega_raw_rad_s",
  "v_cmd_m_s", "omega_cmd_rad_s", "v_left_cmd_m_s", "v_right_cmd_m_s", "command_scale",
  "saturation_reasons"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader(); writer.writerows(rows)


def run_validation(config_path: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
  validation_config = load_validation_config(config_path)
  controller_path = REPO_ROOT / validation_config["baseline_configs"]["tracking_controller"]
  map_path = REPO_ROOT / validation_config["baseline_configs"]["map_route_reference"]
  controller_config = load_controller_config(controller_path)
  map_result = build_map_route_reference(map_path)
  map_reference = map_result["trajectory"]
  perturb = validation_config["perturbations"]
  definitions: list[tuple[str, list[dict[str, Any]], tuple[float, float, float]]] = []
  synthetic = {kind: build_synthetic_reference(kind, controller_config, validation_config)
               for kind in ("forward_straight", "reverse_straight", "reverse_curve",
                            "forward_reverse_transition", "pivot")}
  definitions.extend([
    ("forward_nominal", synthetic["forward_straight"], (0.0, 0.0, 0.0)),
    ("forward_lateral", synthetic["forward_straight"], (0.0, float(perturb["large_lateral_m"]), 0.0)),
    ("forward_heading", synthetic["forward_straight"], (0.0, 0.0, math.radians(float(perturb["heading_deg"])))),
    ("forward_combined", synthetic["forward_straight"], (0.0, float(perturb["small_lateral_m"]), math.radians(float(perturb["heading_deg"])))),
    ("reverse_nominal", synthetic["reverse_straight"], (0.0, 0.0, 0.0)),
    ("reverse_lateral_small", synthetic["reverse_straight"], (0.0, float(perturb["small_lateral_m"]), 0.0)),
    ("reverse_lateral_large", synthetic["reverse_straight"], (0.0, float(perturb["large_lateral_m"]), 0.0)),
    ("reverse_heading", synthetic["reverse_straight"], (0.0, 0.0, math.radians(float(perturb["heading_deg"])))),
    ("reverse_combined", synthetic["reverse_straight"], (0.0, float(perturb["small_lateral_m"]), math.radians(float(perturb["heading_deg"])))),
  ])
  for suffix, pose in (("nominal", (0.0, 0.0, 0.0)),
                       ("lateral", (0.0, float(perturb["small_lateral_m"]), 0.0)),
                       ("heading", (0.0, 0.0, math.radians(float(perturb["heading_deg"])))),
                       ("combined", (0.0, float(perturb["small_lateral_m"]), math.radians(float(perturb["heading_deg"]))))):
    definitions.append((f"reverse_curve_{suffix}", synthetic["reverse_curve"], pose))
  definitions.extend([
    ("transition_nominal", synthetic["forward_reverse_transition"], (0.0, 0.0, 0.0)),
    ("transition_lateral", synthetic["forward_reverse_transition"], (0.0, float(perturb["small_lateral_m"]), 0.0)),
    ("pivot_nominal", synthetic["pivot"], (0.0, 0.0, 0.0)),
    ("pivot_heading", synthetic["pivot"], (0.0, 0.0, math.radians(float(perturb["heading_deg"])))),
    ("pivot_xy", synthetic["pivot"], (float(perturb["small_lateral_m"]), float(perturb["small_lateral_m"]), 0.0)),
    ("map_nominal", map_reference, (float(map_reference[0]["x_m"]), float(map_reference[0]["y_m"]), float(map_reference[0]["yaw_rad"]))),
    ("map_perturbed", map_reference, (float(map_reference[0]["x_m"]), float(map_reference[0]["y_m"]) + float(perturb["full_route_lateral_m"]), float(map_reference[0]["yaw_rad"]) + math.radians(float(perturb["full_route_heading_deg"]))))])
  output_dir.mkdir(parents=True, exist_ok=True)
  scenarios = {}; summaries = {}
  for name, reference, pose in definitions:
    rows = simulate(reference, pose, controller_config, validation_config, name)
    scenarios[name] = rows
    summaries[name] = scenario_summary(rows, reference, validation_config)
    if name.startswith("reverse_") and "curve" not in name:
      summaries[name]["unsaturated_reverse_interior_growth_fit"] = empirical_growth_rate(
        rows, float(validation_config["evaluation"]["growth_fit_min_error"]))
    write_csv(output_dir / f"{name}.csv", rows)
  epsilon = float(validation_config["evaluation"]["finite_difference_epsilon"])
  stability = {}
  for label, speed_value in (("forward", .30), ("reverse", -.30)):
    stability[label] = {"v_ref_m_s": speed_value,
                        "analytic_jacobian": analytic_jacobian(speed_value, controller_config),
                        "finite_difference_jacobian": finite_difference_jacobian(speed_value, controller_config, epsilon),
                        "eigenvalues_1_s": straight_eigenvalues(speed_value, controller_config)}
  summary = {"metadata": validation_config["metadata"],
             "controller_config_path": str(controller_path), "map_reference_config_path": str(map_path),
             "controller_gains": controller_config["controller"],
             "command_limits": controller_config["command_limits"],
             "simulation": controller_config["simulation"],
             "ordered_projection_rule": f"nearest segment restricted to +/- {validation_config['evaluation']['ordered_projection_half_window_segments']} reference segments around the time-indexed reference sample; metrics only",
             "local_stability": stability, "map_reference_summary": map_result["summary"]["reference"],
             "scenarios": summaries,
             "limitations": ["Exact kinematic unicycle only: no actuator, force, slip, delay, noise, or PhysX.",
                             "Tracking Controller V1 is unchanged and was not previously validated for signed reverse or pivot references.",
                             "Stage-D pivot 14 reference yaw remains a known representation confound."]}
  with (output_dir / validation_config["output"]["summary_filename"]).open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
  return summary, scenarios
