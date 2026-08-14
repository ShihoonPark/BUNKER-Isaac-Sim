#!/usr/bin/env python3
"""Dependency-free validation harness for Tracking Controller V1."""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from reference_trajectory_v1 import build_reference_trajectory, load_config as load_reference_config  # noqa: E402
from simulate_tracking_controller_v1 import scenario_pose, simulate_scenario, summarize_scenario  # noqa: E402
from tracking_controller_v1 import (  # noqa: E402
  ReferenceInterpolator, body_frame_errors, controller_command, differential_track_speeds,
  feasible_command, integrate_unicycle_exact, load_controller_config, project_to_polyline,
  raw_control, wrap_to_pi,
)


def require(condition: bool, message: str) -> None:
  if not condition:
    raise AssertionError(message)


def rms(values: list[float]) -> float:
  return math.sqrt(sum(value * value for value in values) / len(values))


def synthetic_reference(x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
                        speed: float = 0.3, omega: float = 0.0,
                        curvature: float = 0.0) -> dict[str, float]:
  return {"t_s": 0.0, "s_m": 0.0, "x_m": x, "y_m": y, "yaw_rad": yaw,
          "curvature_ref_1_m": curvature, "v_ref_m_s": speed,
          "omega_ref_rad_s": omega, "a_ref_m_s2": 0.0}


def validate(config: dict, reference_rows: list[dict]) -> list[tuple[str, Callable[[], None]]]:
  tolerance = float(config["validation"]["floating_point_tolerance"])
  spacing = float(config["measured_fixed"]["track_center_distance_m"])
  limits = config["command_limits"]

  def configuration() -> None:
    require(spacing == 0.434, "measured B changed")
    gains = [float(value) for value in config["controller"].values()]
    simulation = config["simulation"]
    require(all(math.isfinite(value) for value in gains), "non-finite gain")
    require(math.isfinite(float(simulation["time_step_s"])) and
            math.isfinite(float(simulation["terminal_hold_s"])), "non-finite simulation value")
    require(all(float(value) > 0.0 for key, value in limits.items() if key.startswith("maximum_")),
            "command limit is not positive")

  def angle_wrapping() -> None:
    require(abs(wrap_to_pi(math.pi + 0.1) - (-math.pi + 0.1)) <= tolerance,
            "+pi crossing did not choose shortest angle")
    require(abs(wrap_to_pi(-math.pi - 0.1) - (math.pi - 0.1)) <= tolerance,
            "-pi crossing did not choose shortest angle")
    require(wrap_to_pi(math.pi) == -math.pi, "pi boundary is not deterministic")
    require(abs(wrap_to_pi(4.0 * math.pi + 0.2) - 0.2) <= tolerance, "multi-turn wrap failed")

  def zero_error_feedforward() -> None:
    references = [synthetic_reference(speed=0.4),
                  synthetic_reference(yaw=0.5, speed=0.35, omega=0.5, curvature=1.0 / 0.75),
                  synthetic_reference(yaw=-0.4, speed=0.3, omega=-0.5, curvature=-1.0 / 0.55)]
    for reference in references:
      raw = raw_control((reference["x_m"], reference["y_m"], reference["yaw_rad"]), reference, config)
      require(abs(raw["v_raw_m_s"] - reference["v_ref_m_s"]) <= tolerance,
              "zero-error v feedforward mismatch")
      require(abs(raw["omega_raw_rad_s"] - reference["omega_ref_rad_s"]) <= tolerance,
              "zero-error omega feedforward mismatch")

  def error_signs() -> None:
    require(body_frame_errors((0, 0, 0), synthetic_reference(y=0.2))["e_y_m"] > 0, "left e_y sign")
    require(body_frame_errors((0, 0, 0), synthetic_reference(y=-0.2))["e_y_m"] < 0, "right e_y sign")
    require(body_frame_errors((0, 0, 0), synthetic_reference(yaw=0.2))["e_heading_rad"] > 0,
            "CCW heading sign")
    require(body_frame_errors((0, 0, 0), synthetic_reference(yaw=-0.2))["e_heading_rad"] < 0,
            "CW heading sign")

  def steering_signs() -> None:
    require(raw_control((0, 0, 0), synthetic_reference(y=0.2, speed=0), config)["omega_raw_rad_s"] > 0,
            "positive lateral correction sign")
    require(raw_control((0, 0, 0), synthetic_reference(y=-0.2, speed=0), config)["omega_raw_rad_s"] < 0,
            "negative lateral correction sign")
    require(raw_control((0, 0, 0), synthetic_reference(yaw=0.2, speed=0), config)["omega_raw_rad_s"] > 0,
            "positive heading correction sign")
    require(raw_control((0, 0, 0), synthetic_reference(yaw=-0.2, speed=0), config)["omega_raw_rad_s"] < 0,
            "negative heading correction sign")

  def track_mapping() -> None:
    for v, omega in ((0.4, 0.3), (0.4, -0.3), (-0.2, 0.1)):
      left, right = differential_track_speeds(v, omega, spacing)
      require(abs(left - (v - 0.5 * spacing * omega)) <= tolerance, "left mapping mismatch")
      require(abs(right - (v + 0.5 * spacing * omega)) <= tolerance, "right mapping mismatch")
    require(differential_track_speeds(0.4, 0.3, spacing)[1] >
            differential_track_speeds(0.4, 0.3, spacing)[0], "positive omega track ordering")
    require(differential_track_speeds(0.4, -0.3, spacing)[0] >
            differential_track_speeds(0.4, -0.3, spacing)[1], "negative omega track ordering")

  def command_feasibility() -> None:
    body_max = float(limits["maximum_abs_body_speed_m_s"])
    yaw_max = float(limits["maximum_abs_yaw_rate_rad_s"])
    track_max = float(limits["maximum_abs_track_surface_speed_m_s"])
    for v_raw, omega_raw in ((10.0, 8.0), (-7.0, 9.0), (5.0, -12.0), (0.2, 0.1)):
      command = feasible_command(v_raw, omega_raw, config)
      require(0.0 < command["command_scale"] <= 1.0, "invalid command scale")
      require(abs(command["v_cmd_m_s"]) <= body_max + tolerance, "body limit exceeded")
      require(abs(command["omega_cmd_rad_s"]) <= yaw_max + tolerance, "yaw limit exceeded")
      require(abs(command["v_left_cmd_m_s"]) <= track_max + tolerance, "left limit exceeded")
      require(abs(command["v_right_cmd_m_s"]) <= track_max + tolerance, "right limit exceeded")
    require(feasible_command(0.2, 0.1, config)["command_scale"] == 1.0,
            "unsaturated scale is not one")
    require(feasible_command(0.0, 0.0, config)["command_scale"] == 1.0,
            "zero-command scale is not one")

  def exact_kinematics() -> None:
    dt = 0.5
    straight = integrate_unicycle_exact((1.0, 2.0, 0.3), 0.4, 0.0, dt)
    require(abs(straight[0] - (1.0 + 0.4 * math.cos(0.3) * dt)) <= tolerance, "straight x")
    require(abs(straight[1] - (2.0 + 0.4 * math.sin(0.3) * dt)) <= tolerance, "straight y")
    for omega in (0.7, -0.7):
      pose = (0.2, -0.1, 0.4)
      v = 0.5
      result = integrate_unicycle_exact(pose, v, omega, dt)
      yaw_next = pose[2] + omega * dt
      expected_x = pose[0] + (v / omega) * (math.sin(yaw_next) - math.sin(pose[2]))
      expected_y = pose[1] - (v / omega) * (math.cos(yaw_next) - math.cos(pose[2]))
      require(max(abs(result[0] - expected_x), abs(result[1] - expected_y),
                  abs(result[2] - yaw_next)) <= tolerance, "arc integration mismatch")

  def path_projection() -> None:
    polyline = [{"x_m": 0.0, "y_m": 0.0, "s_m": 0.0},
                {"x_m": 2.0, "y_m": 0.0, "s_m": 2.0}]
    above = project_to_polyline(0.75, 0.2, polyline)
    below = project_to_polyline(1.25, -0.3, polyline)
    before = project_to_polyline(-1.0, 0.1, polyline)
    after = project_to_polyline(3.0, -0.1, polyline)
    require(above["cross_track_error_m"] > 0 and below["cross_track_error_m"] < 0,
            "cross-track sign mismatch")
    require(abs(above["s_projected_m"] - 0.75) <= tolerance, "projected s mismatch")
    require(before["s_projected_m"] == 0.0 and after["s_projected_m"] == 2.0,
            "projection endpoint clamping failed")

  def time_interpolation() -> None:
    first = synthetic_reference(yaw=3.10, speed=0.2)
    first.update({"t_s": 1.0, "s_m": 2.0, "x_m": 1.0, "a_ref_m_s2": 0.1})
    second = synthetic_reference(yaw=3.30, speed=0.0)
    second.update({"t_s": 3.0, "s_m": 4.0, "x_m": 5.0, "a_ref_m_s2": 0.0})
    interpolator = ReferenceInterpolator([first, second])
    exact = interpolator.sample(1.0)
    midpoint = interpolator.sample(2.0)
    require(exact["x_m"] == first["x_m"] and exact["yaw_rad"] == first["yaw_rad"],
            "exact interpolation mismatch")
    require(abs(midpoint["x_m"] - 3.0) <= tolerance and
            abs(midpoint["yaw_rad"] - 3.20) <= tolerance, "midpoint interpolation mismatch")
    require(interpolator.sample(-2.0)["x_m"] == first["x_m"], "before-start clamp failed")
    require(interpolator.sample(9.0)["x_m"] == second["x_m"], "after-end clamp failed")
    require(midpoint["yaw_rad"] > math.pi, "continuous yaw was incorrectly wrapped")

  scenario_rows = {name: simulate_scenario(name, scenario_pose(initial), reference_rows, config)
                   for name, initial in config["simulation"]["scenarios"].items()}

  def generated_command_mapping_and_feasibility() -> None:
    body_max = float(limits["maximum_abs_body_speed_m_s"])
    yaw_max = float(limits["maximum_abs_yaw_rate_rad_s"])
    track_max = float(limits["maximum_abs_track_surface_speed_m_s"])
    for name, rows in scenario_rows.items():
      for index, row in enumerate(rows):
        expected_left = row["v_cmd_m_s"] - 0.5 * spacing * row["omega_cmd_rad_s"]
        expected_right = row["v_cmd_m_s"] + 0.5 * spacing * row["omega_cmd_rad_s"]
        require(abs(row["v_left_cmd_m_s"] - expected_left) <= tolerance,
                f"{name} row {index} left command mapping mismatch")
        require(abs(row["v_right_cmd_m_s"] - expected_right) <= tolerance,
                f"{name} row {index} right command mapping mismatch")
        require(abs(row["v_cmd_m_s"]) <= body_max + tolerance,
                f"{name} row {index} body-speed limit exceeded")
        require(abs(row["omega_cmd_rad_s"]) <= yaw_max + tolerance,
                f"{name} row {index} yaw-rate limit exceeded")
        require(abs(row["v_left_cmd_m_s"]) <= track_max + tolerance and
                abs(row["v_right_cmd_m_s"]) <= track_max + tolerance,
                f"{name} row {index} track-speed limit exceeded")
        require(0.0 < row["command_scale"] <= 1.0,
                f"{name} row {index} invalid command scale")

  def nominal_behavior() -> None:
    rows = scenario_rows["nominal"]
    metrics = summarize_scenario(rows, reference_rows)
    require(all(math.isfinite(float(value)) for row in rows for value in row.values()),
            "nominal contains non-finite values")
    validation = config["validation"]
    reference_end_time = float(reference_rows[-1]["t_s"])
    active_rows = [row for row in rows if row["time_s"] <= reference_end_time + 1e-12]

    def independently_recomputed(window_rows: list[dict]) -> dict[str, float | int]:
      saturated = sum(row["command_scale"] < 1.0 - 1e-12 for row in window_rows)
      return {
        "rms_cross_track_error_m": rms([row["cross_track_error_m"] for row in window_rows]),
        "rms_heading_error_rad": rms([row["heading_error_rad"] for row in window_rows]),
        "rms_progress_error_m": rms([row["progress_error_m"] for row in window_rows]),
        "rms_speed_error_m_s": rms([row["speed_error_m_s"] for row in window_rows]),
        "saturated_sample_count": saturated,
        "saturated_sample_fraction": saturated / len(window_rows),
      }

    for prefix, window_rows in (("active", active_rows), ("full_run", rows)):
      independent = independently_recomputed(window_rows)
      for field, expected in independent.items():
        actual = metrics[f"{prefix}_{field}"]
        require(abs(actual - expected) <= tolerance,
                f"{prefix} summary {field} disagrees with raw rows")
    require(metrics["active_sample_count"] > 0, "active sample count is empty")
    require(metrics["active_sample_count"] < metrics["full_run_sample_count"],
            "active window was not separated from terminal hold")
    require(metrics["full_run_sample_count"] == len(rows), "full-run sample count mismatch")
    require(metrics["active_sample_count"] == len(active_rows), "active sample count mismatch")
    require(metrics["active_window_last_time_s"] <= reference_end_time + tolerance,
            "active window extends beyond reference end")
    require(metrics["active_rms_cross_track_error_m"]
            <= validation["nominal_max_active_rms_cross_track_error_m"],
            "nominal RMS cross-track error too high")
    require(metrics["final_xy_position_error_to_goal_m"] <= validation["nominal_max_final_xy_error_m"],
            "nominal final XY error too high")
    require(abs(metrics["final_heading_error_rad"]) <= validation["nominal_max_final_heading_error_rad"],
            "nominal final heading error too high")
    require(rows[-1]["v_ref_m_s"] == 0.0 and rows[-1]["omega_ref_rad_s"] == 0.0,
            "terminal reference is not stopped")
    require(abs(metrics["final_abs_v_cmd_m_s"] - abs(rows[-1]["v_cmd_m_s"])) <= tolerance,
            "final body-speed summary disagrees with final row")
    require(abs(metrics["final_abs_omega_cmd_rad_s"] - abs(rows[-1]["omega_cmd_rad_s"])) <= tolerance,
            "final yaw-rate summary disagrees with final row")
    require(metrics["final_abs_v_cmd_m_s"] <= validation["terminal_max_abs_body_speed_m_s"],
            "nominal terminal body-speed command too large")
    require(metrics["final_abs_omega_cmd_rad_s"] <= validation["terminal_max_abs_yaw_rate_rad_s"],
            "nominal terminal yaw-rate command too large")

  def injected_error_reduction(name: str, key: str) -> None:
    rows = scenario_rows[name]
    reference_end_time = float(reference_rows[-1]["t_s"])
    active_rows = [row for row in rows if row["time_s"] <= reference_end_time + 1e-12]
    early_window = float(config["validation"]["early_window_s"])
    late_window = float(config["validation"]["late_window_s"])
    early = [abs(row[key]) for row in active_rows if row["time_s"] <= early_window]
    late_active = [abs(row[key]) for row in active_rows
                   if row["time_s"] >= reference_end_time - late_window]
    fraction = float(config["validation"]["error_reduction_fraction"])
    require(early, f"{name} early active trajectory window is empty")
    require(late_active, f"{name} late active trajectory window is empty")
    require(active_rows[-1]["time_s"] <= reference_end_time + tolerance,
            f"{name} last active sample exceeds the reference end")
    require(rms(late_active) < fraction * rms(early),
            f"{name} late active trajectory {key} was not substantially reduced")
    require(abs(active_rows[-1][key]) < fraction * abs(rows[0][key]),
            f"{name} last active trajectory {key} was not meaningfully reduced")
    require(abs(rows[-1][key]) < fraction * abs(rows[0][key]),
            f"{name} final terminal-hold {key} was not meaningfully reduced")

  return [
    ("configuration", configuration), ("angle wrapping", angle_wrapping),
    ("zero-error feedforward", zero_error_feedforward), ("body-frame error signs", error_signs),
    ("steering correction signs", steering_signs), ("differential-track mapping", track_mapping),
    ("command feasibility", command_feasibility), ("exact kinematic integration", exact_kinematics),
    ("path projection", path_projection), ("time interpolation", time_interpolation),
    ("generated command mapping and feasibility", generated_command_mapping_and_feasibility),
    ("nominal simulation behavior", nominal_behavior),
    ("lateral-offset reduction", lambda: injected_error_reduction("lateral_offset", "cross_track_error_m")),
    ("heading-offset reduction", lambda: injected_error_reduction("heading_offset", "heading_error_rad")),
  ]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/tracking_controller_v1.json")
  parser.add_argument("--reference-config", type=Path,
                      default=REPO_ROOT / "config/reference_trajectory_v1.json")
  args = parser.parse_args()
  config = load_controller_config(args.config.resolve())
  reference_rows = build_reference_trajectory(load_reference_config(args.reference_config.resolve()))["trajectory"]
  tests = validate(config, reference_rows)
  failures = []
  for name, test in tests:
    try:
      test()
      print(f"PASS: {name}")
    except Exception as error:
      failures.append((name, error))
      print(f"FAIL: {name}: {error}")
  print(f"result: {len(tests) - len(failures)}/{len(tests)} checks passed")
  return 1 if failures else 0


if __name__ == "__main__":
  raise SystemExit(main())
