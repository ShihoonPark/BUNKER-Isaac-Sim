#!/usr/bin/env python3
"""Pure kinematic A/B validation for Direction-Aware Tracking Controller V2."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import Any

from map_route_kinematic_validation_v1 import (
  REPO_ROOT, _sample_times, build_synthetic_reference, empirical_growth_rate,
  load_validation_config as load_v1_validation_config, ordered_project,
  scenario_summary, simulate as simulate_v1,
)
from map_route_reference_v1 import build_map_route_reference
from tracking_controller_v1 import ReferenceInterpolator, integrate_unicycle_exact
from tracking_controller_v2_direction_aware import (
  controller_command, load_direction_aware_config,
)


def load_config(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    config = json.load(stream)
  for relative in config["baseline_configs"].values():
    if not (REPO_ROOT / relative).is_file():
      raise ValueError(f"missing baseline input: {relative}")
  return config


def explicit_synthetic_modes(kind: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Attach semantic maneuver modes without inferring them from instantaneous speed."""
  result = [dict(row) for row in rows]
  if kind == "pivot":
    for row in result:
      row.update({"motion_direction": 0, "segment_type": "pivot"})
  elif kind == "forward_straight":
    for row in result:
      row.update({"motion_direction": 1, "segment_type": "forward"})
  elif kind in ("reverse_straight", "reverse_curve"):
    for row in result:
      row.update({"motion_direction": -1, "segment_type": "reverse"})
  elif kind == "forward_reverse_transition":
    first_reverse = next(index for index, row in enumerate(result) if float(row["v_ref_m_s"]) < 0.0)
    reverse_boundary = first_reverse - 1
    for index, row in enumerate(result):
      direction = -1 if index >= reverse_boundary else 1
      row.update({"motion_direction": direction,
                  "segment_type": "reverse" if direction < 0 else "forward",
                  "segment_id": 1 if direction < 0 else 0})
  else:
    raise ValueError(f"unknown synthetic mode annotation: {kind}")
  return result


def simulate_v2(reference_rows: list[dict[str, Any]], initial_pose: tuple[float, float, float],
                controller_config: dict[str, Any], v1_validation_config: dict[str, Any],
                name: str) -> list[dict[str, Any]]:
  dt = float(controller_config["simulation"]["time_step_s"])
  hold = float(controller_config["simulation"]["terminal_hold_s"])
  end_time = float(reference_rows[-1]["t_s"]) + hold
  times = _sample_times(end_time, dt)
  interpolator = ReferenceInterpolator(reference_rows)
  reference_times = [float(row["t_s"]) for row in reference_rows]
  half_window = int(v1_validation_config["evaluation"]["ordered_projection_half_window_segments"])
  pose = initial_pose
  rows = []
  spacing = float(controller_config["measured_fixed"]["track_center_distance_m"])
  limits = controller_config["command_limits"]
  for step, time_s in enumerate(times):
    reference = interpolator.sample(time_s)
    expected_index = max(0, min(len(reference_rows) - 1,
                                bisect.bisect_right(reference_times, min(time_s, reference_times[-1])) - 1))
    if (expected_index + 1 < len(reference_rows) and
        int(reference_rows[expected_index].get("segment_id", 0)) !=
        int(reference_rows[expected_index + 1].get("segment_id", 0)) and
        time_s > reference_times[expected_index] + 1e-12):
      expected_index += 1
    source_row = reference_rows[expected_index]
    direction = int(source_row["motion_direction"])
    command = controller_command(pose, reference, direction, controller_config)
    projection = ordered_project(pose[0], pose[1], reference_rows, expected_index, half_window)
    raw_left = float(command["v_raw_m_s"]) - .5 * spacing * float(command["omega_raw_rad_s"])
    raw_right = float(command["v_raw_m_s"]) + .5 * spacing * float(command["omega_raw_rad_s"])
    reasons = []
    if abs(float(command["v_raw_m_s"])) > float(limits["maximum_abs_body_speed_m_s"]): reasons.append("body")
    if abs(float(command["omega_raw_rad_s"])) > float(limits["maximum_abs_yaw_rate_rad_s"]): reasons.append("yaw")
    if abs(raw_left) > float(limits["maximum_abs_track_surface_speed_m_s"]): reasons.append("left_track")
    if abs(raw_right) > float(limits["maximum_abs_track_surface_speed_m_s"]): reasons.append("right_track")
    rows.append({
      "scenario": name, "time_s": time_s, "reference_index": expected_index,
      "segment_id": int(source_row.get("segment_id", 0)),
      "segment_type": str(source_row.get("segment_type", "synthetic")),
      "motion_direction": direction, "lateral_feedback_factor": int(command["lateral_feedback_factor"]),
      "reference_s_m": reference["s_m"], "reference_x_m": reference["x_m"],
      "reference_y_m": reference["y_m"], "reference_yaw_rad": reference["yaw_rad"],
      "actual_x_m": pose[0], "actual_y_m": pose[1], "actual_yaw_rad": pose[2],
      "e_x_m": command["e_x_m"], "e_y_m": command["e_y_m"],
      "heading_error_rad": command["e_heading_rad"],
      "time_aligned_xy_error_m": math.hypot(reference["x_m"] - pose[0], reference["y_m"] - pose[1]),
      "projected_s_m": projection["s_projected_m"], "cross_track_error_m": projection["cross_track_error_m"],
      "progress_error_m": reference["s_m"] - projection["s_projected_m"],
      "projection_segment_index": int(projection["projection_segment_index"]),
      "v_ref_m_s": reference["v_ref_m_s"], "omega_ref_rad_s": reference["omega_ref_rad_s"],
      "longitudinal_feedback_m_s": command["longitudinal_feedback_m_s"],
      "lateral_feedback_rad_s": command["lateral_feedback_rad_s"],
      "heading_feedback_rad_s": command["heading_feedback_rad_s"],
      "v_raw_m_s": command["v_raw_m_s"], "omega_raw_rad_s": command["omega_raw_rad_s"],
      "v_cmd_m_s": command["v_cmd_m_s"], "omega_cmd_rad_s": command["omega_cmd_rad_s"],
      "v_left_cmd_m_s": command["v_left_cmd_m_s"], "v_right_cmd_m_s": command["v_right_cmd_m_s"],
      "command_scale": command["command_scale"],
      "saturation_reasons": "+".join(reasons) if reasons else "none"})
    if step + 1 < len(times):
      pose = integrate_unicycle_exact(pose, command["v_cmd_m_s"], command["omega_cmd_rad_s"],
                                      times[step + 1] - time_s,
                                      float(controller_config["simulation"]["near_zero_yaw_rate_rad_s"]))
  return rows


def analytic_jacobian(v_ref: float, motion_direction: int,
                      controller_config: dict[str, Any]) -> list[list[float]]:
  gains = controller_config["controller"]
  factor = 1 if motion_direction in (0, 1) else -1
  return [[-float(gains["k_longitudinal_1_s"]), 0.0, 0.0],
          [0.0, 0.0, v_ref],
          [0.0, -factor * float(gains["k_lateral_rad_s_per_m"]),
           -float(gains["k_heading_1_s"])]]


def _error_dynamics(state: tuple[float, float, float], v_ref: float, motion_direction: int,
                    controller_config: dict[str, Any]) -> tuple[float, float, float]:
  e_x, e_y, heading = state
  reference = {"x_m": e_x, "y_m": e_y, "yaw_rad": heading,
               "v_ref_m_s": v_ref, "omega_ref_rad_s": 0.0}
  command = controller_command((0.0, 0.0, 0.0), reference, motion_direction, controller_config)
  v, omega = float(command["v_cmd_m_s"]), float(command["omega_cmd_rad_s"])
  return (omega * e_y - v + v_ref * math.cos(heading),
          -omega * e_x + v_ref * math.sin(heading), -omega)


def finite_difference_jacobian(v_ref: float, motion_direction: int,
                               controller_config: dict[str, Any], epsilon: float) -> list[list[float]]:
  matrix = [[0.0] * 3 for _ in range(3)]
  for column in range(3):
    plus = [0.0] * 3; minus = [0.0] * 3
    plus[column] = epsilon; minus[column] = -epsilon
    upper = _error_dynamics(tuple(plus), v_ref, motion_direction, controller_config)
    lower = _error_dynamics(tuple(minus), v_ref, motion_direction, controller_config)
    for row in range(3):
      matrix[row][column] = (upper[row] - lower[row]) / (2.0 * epsilon)
  return matrix


def eigenvalues(v_ref: float, motion_direction: int,
                controller_config: dict[str, Any]) -> list[float]:
  gains = controller_config["controller"]
  factor = 1 if motion_direction in (0, 1) else -1
  kx = float(gains["k_longitudinal_1_s"]); ky = float(gains["k_lateral_rad_s_per_m"])
  kh = float(gains["k_heading_1_s"])
  root = math.sqrt(kh * kh - 4.0 * factor * ky * v_ref)
  return sorted([-kx, (-kh - root) / 2.0, (-kh + root) / 2.0])


def maximum_differences(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> dict[str, float]:
  if len(first) != len(second):
    raise ValueError("A/B row counts differ")
  fields = ("actual_x_m", "actual_y_m", "actual_yaw_rad", "v_cmd_m_s", "omega_cmd_rad_s")
  return {f"maximum_abs_difference_{field}": max(
    abs(float(left[field]) - float(right[field])) for left, right in zip(first, second)) for field in fields}


def _scenario_definitions(controller_config: dict[str, Any], validation_config: dict[str, Any],
                          map_reference: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]], tuple[float, float, float]]]:
  perturb = validation_config["perturbations"]
  synthetic = {kind: explicit_synthetic_modes(
    kind, build_synthetic_reference(kind, controller_config, validation_config)) for kind in
    ("forward_straight", "reverse_straight", "reverse_curve", "forward_reverse_transition", "pivot")}
  result = [
    ("forward_nominal", synthetic["forward_straight"], (0.0, 0.0, 0.0)),
    ("forward_lateral", synthetic["forward_straight"], (0.0, float(perturb["large_lateral_m"]), 0.0)),
    ("forward_heading", synthetic["forward_straight"], (0.0, 0.0, math.radians(float(perturb["heading_deg"])))),
    ("forward_combined", synthetic["forward_straight"], (0.0, float(perturb["small_lateral_m"]), math.radians(float(perturb["heading_deg"])))),
    ("reverse_nominal", synthetic["reverse_straight"], (0.0, 0.0, 0.0)),
    ("reverse_lateral_small", synthetic["reverse_straight"], (0.0, float(perturb["small_lateral_m"]), 0.0)),
    ("reverse_lateral_large", synthetic["reverse_straight"], (0.0, float(perturb["large_lateral_m"]), 0.0)),
    ("reverse_heading", synthetic["reverse_straight"], (0.0, 0.0, math.radians(float(perturb["heading_deg"])))),
    ("reverse_combined", synthetic["reverse_straight"], (0.0, float(perturb["small_lateral_m"]), math.radians(float(perturb["heading_deg"]))))]
  for suffix, pose in (("nominal", (0.0, 0.0, 0.0)),
                       ("lateral", (0.0, float(perturb["small_lateral_m"]), 0.0)),
                       ("heading", (0.0, 0.0, math.radians(float(perturb["heading_deg"])))),
                       ("combined", (0.0, float(perturb["small_lateral_m"]), math.radians(float(perturb["heading_deg"]))))):
    result.append((f"reverse_curve_{suffix}", synthetic["reverse_curve"], pose))
  result.extend([
    ("transition_nominal", synthetic["forward_reverse_transition"], (0.0, 0.0, 0.0)),
    ("transition_lateral", synthetic["forward_reverse_transition"], (0.0, float(perturb["small_lateral_m"]), 0.0)),
    ("pivot_nominal", synthetic["pivot"], (0.0, 0.0, 0.0)),
    ("pivot_heading", synthetic["pivot"], (0.0, 0.0, math.radians(float(perturb["heading_deg"])))),
    ("pivot_xy", synthetic["pivot"], (float(perturb["small_lateral_m"]), float(perturb["small_lateral_m"]), 0.0)),
    ("map_nominal", map_reference, (float(map_reference[0]["x_m"]), float(map_reference[0]["y_m"]), float(map_reference[0]["yaw_rad"]))),
    ("map_perturbed", map_reference, (float(map_reference[0]["x_m"]), float(map_reference[0]["y_m"]) + float(perturb["full_route_lateral_m"]), float(map_reference[0]["yaw_rad"]) + math.radians(float(perturb["full_route_heading_deg"]))))])
  return result


def run(config_path: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
  config = load_config(config_path)
  _, controller_config = load_direction_aware_config(
    REPO_ROOT / config["baseline_configs"]["direction_aware_controller"])
  v1_validation_config = load_v1_validation_config(
    REPO_ROOT / config["baseline_configs"]["v1_kinematic_validation"])
  map_result = build_map_route_reference(REPO_ROOT / config["baseline_configs"]["map_route_reference"])
  definitions = _scenario_definitions(controller_config, v1_validation_config, map_result["trajectory"])
  output_dir.mkdir(parents=True, exist_ok=True)
  v1_rows = {}; v2_rows = {}; v1_summary = {}; v2_summary = {}; equivalence = {}
  for name, reference, pose in definitions:
    baseline = simulate_v1(reference, pose, controller_config, v1_validation_config, name)
    current = simulate_v2(reference, pose, controller_config, v1_validation_config, name)
    v1_rows[name] = baseline; v2_rows[name] = current
    v1_summary[name] = scenario_summary(baseline, reference, v1_validation_config)
    v2_summary[name] = scenario_summary(current, reference, v1_validation_config)
    if name.startswith("reverse_") and "curve" not in name:
      minimum = float(v1_validation_config["evaluation"]["growth_fit_min_error"])
      v1_summary[name]["unsaturated_reverse_interior_rate"] = empirical_growth_rate(baseline, minimum)
      v2_summary[name]["unsaturated_reverse_interior_rate"] = empirical_growth_rate(current, minimum)
    if name.startswith("forward_") or name.startswith("pivot_"):
      equivalence[name] = maximum_differences(baseline, current)
    with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(current[0].keys())); writer.writeheader(); writer.writerows(current)
  epsilon = float(v1_validation_config["evaluation"]["finite_difference_epsilon"])
  stability = {}
  for name, speed, direction in (("forward", .3, 1), ("reverse", -.3, -1)):
    stability[name] = {"v_ref_m_s": speed, "motion_direction": direction,
                       "analytic_jacobian": analytic_jacobian(speed, direction, controller_config),
                       "finite_difference_jacobian": finite_difference_jacobian(speed, direction, controller_config, epsilon),
                       "eigenvalues_1_s": eigenvalues(speed, direction, controller_config)}
  segment_ab = {}
  for segment_id in range(23):
    before = v1_summary["map_nominal"]["segment_metrics"][str(segment_id)]
    after = v2_summary["map_nominal"]["segment_metrics"][str(segment_id)]
    segment_ab[str(segment_id)] = {"mode": after["segment_type"],
      "v1_rms_e_y_m": before["rms_e_y"], "v2_rms_e_y_m": after["rms_e_y"],
      "v1_rms_heading_error_rad": before["rms_heading_error"], "v2_rms_heading_error_rad": after["rms_heading_error"],
      "v1_rms_xy_m": before["rms_time_aligned_xy_error"], "v2_rms_xy_m": after["rms_time_aligned_xy_error"],
      "v1_saturated": before["saturated_sample_count"], "v2_saturated": after["saturated_sample_count"],
      "sample_count": after["sample_count"]}
  summary = {"metadata": config["metadata"], "controller_gains": controller_config["controller"],
             "command_limits": controller_config["command_limits"], "local_stability": stability,
             "forward_and_pivot_equivalence": equivalence,
             "v1_scenarios": v1_summary, "v2_scenarios": v2_summary,
             "map_nominal_segment_ab": segment_ab,
             "projection_rule": f"Same merged V1 metrics-only ordered +/-{v1_validation_config['evaluation']['ordered_projection_half_window_segments']}-segment window.",
             "limitations": ["Exact kinematic unicycle only; no Isaac Sim, actuator, contact, slip, delay, or noise.",
                             "Stage-D pivot 14 reference representation remains unchanged and confounding."]}
  with (output_dir / config["output"]["summary_filename"]).open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
  return summary, v2_rows


def plot(summary: dict[str, Any], rows: dict[str, list[dict[str, Any]]], path: Path) -> str:
  try:
    import matplotlib.pyplot as plt
  except ImportError as error:
    return f"skipped; matplotlib unavailable: {error}"
  figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
  for name in ("forward_lateral", "reverse_lateral_small", "reverse_curve_lateral", "transition_lateral"):
    data = rows[name]
    axes[0, 0].plot([row["time_s"] for row in data], [row["e_y_m"] for row in data], label=name)
  axes[0, 0].set_title("V2 lateral body error"); axes[0, 0].legend(fontsize=8)
  for name in ("reverse_lateral_small", "reverse_lateral_large", "reverse_combined"):
    data = rows[name]
    axes[0, 1].plot([row["time_s"] for row in data], [row["heading_error_rad"] for row in data], label=name)
  axes[0, 1].set_title("V2 reverse heading error"); axes[0, 1].legend(fontsize=8)
  data = rows["map_perturbed"]
  axes[1, 0].plot([row["reference_x_m"] for row in data], [row["reference_y_m"] for row in data], "k--", label="reference")
  axes[1, 0].plot([row["actual_x_m"] for row in data], [row["actual_y_m"] for row in data], label="V2 actual")
  axes[1, 0].set_aspect("equal", adjustable="box"); axes[1, 0].set_title("Map route V2"); axes[1, 0].legend()
  axes[1, 1].plot([row["time_s"] for row in data], [row["e_y_m"] for row in data], label="e_y")
  axes[1, 1].plot([row["time_s"] for row in data], [row["heading_error_rad"] for row in data], label="heading")
  axes[1, 1].set_title("Map-route errors"); axes[1, 1].legend()
  for axis in axes.flat: axis.grid(True, alpha=.25)
  figure.savefig(path, dpi=160); plt.close(figure)
  return str(path)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path,
                      default=REPO_ROOT / "config/map_route_kinematic_validation_v2.json")
  parser.add_argument("--output-dir", type=Path,
                      default=REPO_ROOT / "logs/map_route_kinematic_validation_v2")
  group = parser.add_mutually_exclusive_group(); group.add_argument("--plot", dest="plot_enabled", action="store_true"); group.add_argument("--no-plot", dest="plot_enabled", action="store_false")
  parser.set_defaults(plot_enabled=False); args = parser.parse_args()
  summary, rows = run(args.config.resolve(), args.output_dir.resolve())
  print("Direction-Aware Controller V2 pure kinematic A/B complete (no Isaac Sim)")
  print(f"forward eigenvalues: {summary['local_stability']['forward']['eigenvalues_1_s']}")
  print(f"reverse eigenvalues: {summary['local_stability']['reverse']['eigenvalues_1_s']}")
  for name in ("reverse_lateral_small", "reverse_lateral_large", "reverse_curve_lateral",
               "transition_lateral", "map_nominal", "map_perturbed"):
    before, after = summary["v1_scenarios"][name], summary["v2_scenarios"][name]
    print(f"{name}: V1 late={before['late_active_error_norm_rms']:.6f}, "
          f"V2 late={after['late_active_error_norm_rms']:.6f}, "
          f"V2 sat={after['active']['saturated_sample_count']}/{after['active']['sample_count']}")
  if args.plot_enabled:
    print(f"plot: {plot(summary, rows, args.output_dir.resolve() / 'map_route_kinematic_validation_v2.png')}")
  print(f"outputs: {args.output_dir.resolve()}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
