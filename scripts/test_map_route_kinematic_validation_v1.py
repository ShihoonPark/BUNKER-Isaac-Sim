#!/usr/bin/env python3
"""Deterministic tests for pure maneuver-aware controller compatibility validation."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

from map_route_kinematic_validation_v1 import (
  REPO_ROOT, analytic_jacobian, build_synthetic_reference,
  finite_difference_jacobian, load_validation_config, ordered_project,
  run_validation,
)
from map_route_reference_v1 import build_map_route_reference
from tracking_controller_v1 import (
  ReferenceInterpolator, body_frame_errors, controller_command,
  integrate_unicycle_exact, load_controller_config,
)


def require(condition: bool, message: str) -> None:
  if not condition:
    raise AssertionError(message)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path,
                      default=REPO_ROOT / "config/map_route_kinematic_validation_v1.json")
  parser.add_argument("--output-dir", type=Path,
                      default=REPO_ROOT / "logs/map_route_kinematic_validation_v1")
  parser.add_argument("--generate", action="store_true",
                      help="Regenerate scenarios before testing generated results.")
  args = parser.parse_args()
  config = load_validation_config(args.config.resolve())
  controller = load_controller_config(REPO_ROOT / config["baseline_configs"]["tracking_controller"])
  checks = []

  def check(name, function):
    function(); checks.append(name); print(f"PASS: {name}")

  def signed_integration():
    pose = integrate_unicycle_exact((0.0, 0.0, 0.0), -.3, 0.0, 2.0)
    require(abs(pose[0] + .6) < 1e-12 and pose[1] == 0.0 and pose[2] == 0.0,
            "negative speed did not move backward")
  check("exact signed backward integration", signed_integration)

  def pivot_integration():
    pose = integrate_unicycle_exact((1.0, 2.0, .3), 0.0, .4, 2.0)
    require(pose[:2] == (1.0, 2.0) and abs(pose[2] - 1.1) < 1e-12,
            "zero-v pivot translated")
  check("exact zero-v pivot integration", pivot_integration)

  references = {name: build_synthetic_reference(name, controller, config) for name in
                ("forward_straight", "reverse_straight", "reverse_curve",
                 "forward_reverse_transition", "pivot")}
  check("forward reference consistency", lambda: require(
    any(row["v_ref_m_s"] > 0.0 for row in references["forward_straight"]) and
    all(abs(row["omega_ref_rad_s"]) < 1e-15 for row in references["forward_straight"]),
    "invalid forward reference"))
  check("reverse body orientation", lambda: require(
    min(row["x_m"] for row in references["reverse_straight"]) < 0.0 and
    max(abs(row["yaw_rad"]) for row in references["reverse_straight"]) < 1e-15 and
    any(row["v_ref_m_s"] < 0.0 for row in references["reverse_straight"]),
    "reverse reference flipped body yaw"))
  check("reverse-curve Stage-D convention", lambda: require(max(abs(
    row["omega_ref_rad_s"] - abs(row["v_ref_m_s"]) * row["curvature_ref_1_m"])
    for row in references["reverse_curve"]) < 1e-12, "reverse omega convention mismatch"))

  def transition_zero() -> None:
    speeds = [row["v_ref_m_s"] for row in references["forward_reverse_transition"]]
    require(any(value > 0.0 for value in speeds) and any(value < 0.0 for value in speeds),
            "transition lacks a direction")
    require(not any(first * second < 0.0 for first, second in zip(speeds, speeds[1:])),
            "transition jumped through zero")
  check("forward-stop-reverse transition", transition_zero)

  def interpolation() -> None:
    reverse = references["reverse_straight"]
    sample = ReferenceInterpolator(reverse).sample(reverse[len(reverse) // 2]["t_s"])
    require(sample["v_ref_m_s"] < 0.0 and sample["yaw_rad"] == 0.0,
            "interpolation lost signed speed or yaw")
    pivot = references["pivot"]
    pivot_sample = ReferenceInterpolator(pivot).sample(pivot[len(pivot) // 2]["t_s"])
    require(pivot_sample["v_ref_m_s"] == 0.0 and pivot_sample["omega_ref_rad_s"] > 0.0,
            "pivot interpolation coupled omega to speed")
  check("signed and pivot interpolation", interpolation)

  def reconstruction() -> None:
    actual = (.2, -.1, .3)
    reference = {"x_m": .5, "y_m": .2, "yaw_rad": .4, "v_ref_m_s": -.3,
                 "omega_ref_rad_s": .15}
    errors = body_frame_errors(actual, reference)
    gains = controller["controller"]
    expected_v = reference["v_ref_m_s"] * math.cos(errors["e_heading_rad"]) + float(gains["k_longitudinal_1_s"]) * errors["e_x_m"]
    expected_omega = reference["omega_ref_rad_s"] + float(gains["k_lateral_rad_s_per_m"]) * errors["e_y_m"] + float(gains["k_heading_1_s"]) * math.sin(errors["e_heading_rad"])
    command = controller_command(actual, reference, controller)
    require(abs(command["v_raw_m_s"] - expected_v) < 1e-15 and
            abs(command["omega_raw_rad_s"] - expected_omega) < 1e-15,
            "controller reconstruction mismatch")
  check("controller-law reconstruction", reconstruction)

  def jacobian_match() -> None:
    for speed in (.3, -.3):
      analytic = analytic_jacobian(speed, controller)
      numeric = finite_difference_jacobian(speed, controller, 1e-7)
      require(max(abs(analytic[row][column] - numeric[row][column])
                  for row in range(3) for column in range(3)) < 1e-8,
              f"finite-difference Jacobian mismatch at v={speed}")
  check("analytic/numeric local Jacobian", jacobian_match)

  map_result = build_map_route_reference(REPO_ROOT / config["baseline_configs"]["map_route_reference"])
  map_reference = map_result["trajectory"]
  def ordered_projection() -> None:
    projected = []
    half = int(config["evaluation"]["ordered_projection_half_window_segments"])
    for index, row in enumerate(map_reference):
      result = ordered_project(row["x_m"], row["y_m"], map_reference, index, half)
      projected.append(result["s_projected_m"])
    require(all(projected[index] >= projected[index - 1] - 1e-10 for index in range(1, len(projected))),
            "ordered projection moved backward")
    require(abs(projected[-1] - map_reference[-1]["s_m"]) < 1e-10,
            "near-closed endpoint projected to initial branch")
  check("ordered near-closed projection", ordered_projection)

  if args.generate or not (args.output_dir / "summary.json").is_file():
    summary, scenarios = run_validation(args.config.resolve(), args.output_dir.resolve())
  else:
    summary, scenarios = run_validation(args.config.resolve(), args.output_dir.resolve())
  def generated_finite() -> None:
    require(all(math.isfinite(float(value)) for rows in scenarios.values() for row in rows
                for value in row.values() if isinstance(value, (int, float))), "non-finite generated value")
  check("generated values finite", generated_finite)
  check("full route has all maneuver modes", lambda: require(
    {row["segment_type"] for row in scenarios["map_nominal"] if row["time_s"] <= map_reference[-1]["t_s"] + 1e-12}
    >= {"forward", "reverse", "pivot"}, "full route omitted a maneuver mode"))
  check("canonical command feasibility", lambda: require(all(
    0.0 < float(row["command_scale"]) <= 1.0 + 1e-12 for rows in scenarios.values() for row in rows),
    "invalid command scale"))
  check("map reference remains open", lambda: require(
    math.hypot(map_reference[-1]["x_m"] - map_reference[0]["x_m"],
               map_reference[-1]["y_m"] - map_reference[0]["y_m"]) > .08,
    "map endpoint was artificially closed"))
  def frozen_controller_source() -> None:
    result = subprocess.run(
      ["git", "diff", "--exit-code", "--", "scripts/tracking_controller_v1.py"],
      cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    require(result.returncode == 0 and not result.stdout, "Tracking Controller V1 source was modified")
  check("controller source remains frozen", frozen_controller_source)
  print(f"result: {len(checks)}/{len(checks)} checks passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
