#!/usr/bin/env python3
"""Post-generation and pure tests for Direction-Aware Controller V2 kinematics."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

from simulate_map_route_kinematic_validation_v2 import (
  REPO_ROOT, analytic_jacobian, eigenvalues, explicit_synthetic_modes,
  finite_difference_jacobian, load_config, run,
)
from map_route_kinematic_validation_v1 import build_synthetic_reference, load_validation_config
from tracking_controller_v2_direction_aware import load_direction_aware_config


def require(condition: bool, message: str) -> None:
  if not condition:
    raise AssertionError(message)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path,
                      default=REPO_ROOT / "config/map_route_kinematic_validation_v2.json")
  parser.add_argument("--output-dir", type=Path,
                      default=REPO_ROOT / "logs/map_route_kinematic_validation_v2")
  args = parser.parse_args()
  config = load_config(args.config.resolve())
  _, controller = load_direction_aware_config(
    REPO_ROOT / config["baseline_configs"]["direction_aware_controller"])
  v1_validation = load_validation_config(
    REPO_ROOT / config["baseline_configs"]["v1_kinematic_validation"])
  tests = []

  def check(name, function):
    function(); tests.append(name); print(f"PASS: {name}")

  def stable_jacobians() -> None:
    expected_forward = [[-1.0, 0.0, 0.0], [0.0, 0.0, .3], [0.0, -1.5, -2.0]]
    expected_reverse = [[-1.0, 0.0, 0.0], [0.0, 0.0, -.3], [0.0, 1.5, -2.0]]
    require(analytic_jacobian(.3, 1, controller) == expected_forward, "forward Jacobian changed")
    require(analytic_jacobian(-.3, -1, controller) == expected_reverse, "reverse Jacobian incorrect")
    require(max(eigenvalues(.3, 1, controller)) < 0.0 and
            max(eigenvalues(-.3, -1, controller)) < 0.0, "V2 Jacobian is not stable")
  check("forward/reverse analytic stability", stable_jacobians)

  def finite_difference() -> None:
    for speed, direction in ((.3, 1), (-.3, -1)):
      analytic = analytic_jacobian(speed, direction, controller)
      numeric = finite_difference_jacobian(speed, direction, controller, 1e-7)
      require(max(abs(analytic[row][column] - numeric[row][column])
                  for row in range(3) for column in range(3)) < 1e-8,
              "finite-difference Jacobian mismatch")
  check("finite-difference Jacobian confirmation", finite_difference)

  def explicit_boundary_mode() -> None:
    base = build_synthetic_reference("forward_reverse_transition", controller, v1_validation)
    rows = explicit_synthetic_modes("forward_reverse_transition", base)
    first_reverse = next(index for index, row in enumerate(rows) if row["v_ref_m_s"] < 0.0)
    boundary = rows[first_reverse - 1]
    require(boundary["v_ref_m_s"] == 0.0 and boundary["motion_direction"] == -1,
            "zero-speed reverse boundary lacks explicit reverse metadata")
  check("zero-speed transition has explicit mode", explicit_boundary_mode)

  summary, scenarios = run(args.config.resolve(), args.output_dir.resolve())
  def forward_equivalence() -> None:
    maximum = max(value for scenario in summary["forward_and_pivot_equivalence"].values()
                  for key, value in scenario.items() if key.startswith("maximum_abs_difference"))
    require(maximum < 1e-14, f"forward/pivot V1 equivalence failed: {maximum}")
  check("forward and pivot A/B equivalence", forward_equivalence)

  def reverse_decay() -> None:
    values = summary["v2_scenarios"]["reverse_lateral_small"]
    require(values["late_active_error_norm_rms"] < values["early_error_norm_rms"],
            "small reverse lateral perturbation did not decay")
    require(values["active"]["saturated_sample_count"] == 0,
            "small reverse diagnostic saturated")
    require(values["unsaturated_reverse_interior_rate"]["rate_1_s"] < 0.0,
            "empirical reverse rate is not negative")
  check("unsaturated reverse lateral perturbation decays", reverse_decay)

  check("synthetic forward remains convergent", lambda: require(
    summary["v2_scenarios"]["forward_lateral"]["late_active_error_norm_rms"] <
    summary["v2_scenarios"]["forward_lateral"]["early_error_norm_rms"],
    "forward regression did not converge"))

  def feasibility() -> None:
    limits = controller["command_limits"]
    spacing = float(controller["measured_fixed"]["track_center_distance_m"])
    for rows in scenarios.values():
      for row in rows:
        v, omega = float(row["v_cmd_m_s"]), float(row["omega_cmd_rad_s"])
        require(abs(v) <= float(limits["maximum_abs_body_speed_m_s"]) + 1e-12, "body limit")
        require(abs(omega) <= float(limits["maximum_abs_yaw_rate_rad_s"]) + 1e-12, "yaw limit")
        require(abs(float(row["v_left_cmd_m_s"]) - (v - .5 * spacing * omega)) < 1e-12, "left mapping")
        require(abs(float(row["v_right_cmd_m_s"]) - (v + .5 * spacing * omega)) < 1e-12, "right mapping")
        require(abs(float(row["v_left_cmd_m_s"])) <= float(limits["maximum_abs_track_surface_speed_m_s"]) + 1e-12, "left limit")
        require(abs(float(row["v_right_cmd_m_s"])) <= float(limits["maximum_abs_track_surface_speed_m_s"]) + 1e-12, "right limit")
  check("signed command feasibility and track mapping", feasibility)

  check("generated values finite", lambda: require(all(
    math.isfinite(float(value)) for rows in scenarios.values() for row in rows
    for value in row.values() if isinstance(value, (int, float))), "non-finite output"))

  def frozen_inputs() -> None:
    paths = ["config/tracking_controller_v1.json", "scripts/tracking_controller_v1.py",
             "scripts/simulate_tracking_controller_v1.py", "scripts/test_tracking_controller_v1.py",
             "config/map_route_reference_v1.json", "scripts/map_route_reference_v1.py",
             "config/map_route_kinematic_validation_v1.json", "scripts/map_route_kinematic_validation_v1.py"]
    result = subprocess.run(["git", "diff", "--exit-code", "--", *paths], cwd=REPO_ROOT,
                            check=False, capture_output=True, text=True)
    require(result.returncode == 0 and not result.stdout, "frozen V1 input changed")
  check("frozen V1 and Stage-D inputs unchanged", frozen_inputs)
  print(f"result: {len(tests)}/{len(tests)} checks passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
