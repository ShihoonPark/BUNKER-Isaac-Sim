#!/usr/bin/env python3
"""Deterministic unit tests for Direction-Aware Tracking Controller V2."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

from tracking_controller_v1 import controller_command as controller_command_v1
from tracking_controller_v2_direction_aware import (
  REPO_ROOT, controller_command, lateral_feedback_factor,
  load_direction_aware_config, raw_control,
)


def require(condition: bool, message: str) -> None:
  if not condition:
    raise AssertionError(message)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path,
                      default=REPO_ROOT / "config/tracking_controller_v2_direction_aware.json")
  args = parser.parse_args()
  v2_config, controller_config = load_direction_aware_config(args.config.resolve())
  tests = []

  def check(name, function):
    function(); tests.append(name); print(f"PASS: {name}")

  check("V2 configuration loads", lambda: require(v2_config["schema_version"] == 1, "bad schema"))
  check("forward factor", lambda: require(lateral_feedback_factor(1) == 1, "bad forward factor"))
  check("reverse factor", lambda: require(lateral_feedback_factor(-1) == -1, "bad reverse factor"))
  check("pivot preserves V1 factor", lambda: require(lateral_feedback_factor(0) == 1, "bad pivot factor"))

  actual = (.1, -.2, .3)
  reference = {"x_m": .5, "y_m": .1, "yaw_rad": .45,
               "v_ref_m_s": .3, "omega_ref_rad_s": .15}
  def forward_equivalence() -> None:
    v1 = controller_command_v1(actual, reference, controller_config)
    v2 = controller_command(actual, reference, 1, controller_config)
    for field in ("e_x_m", "e_y_m", "e_heading_rad", "v_raw_m_s", "omega_raw_rad_s",
                  "v_cmd_m_s", "omega_cmd_rad_s", "v_left_cmd_m_s", "v_right_cmd_m_s",
                  "command_scale"):
      require(abs(float(v1[field]) - float(v2[field])) < 1e-15, f"forward mismatch: {field}")
  check("forward V2 equals V1", forward_equivalence)

  def reverse_only_lateral_sign() -> None:
    forward = raw_control(actual, reference, 1, controller_config)
    reverse = raw_control(actual, reference, -1, controller_config)
    require(reverse["lateral_feedback_rad_s"] == -forward["lateral_feedback_rad_s"],
            "reverse lateral contribution did not flip")
    for field in ("e_x_m", "e_y_m", "e_heading_rad", "longitudinal_feedback_m_s",
                  "heading_feedback_rad_s", "v_raw_m_s"):
      require(abs(float(reverse[field]) - float(forward[field])) < 1e-15,
              f"unexpected reverse change: {field}")
    expected_difference = -2.0 * forward["lateral_feedback_rad_s"]
    require(abs((reverse["omega_raw_rad_s"] - forward["omega_raw_rad_s"]) - expected_difference) < 1e-15,
            "omega change is not solely the lateral sign")
  check("reverse changes only lateral-feedback sign", reverse_only_lateral_sign)

  def pivot_equivalence() -> None:
    pivot_reference = dict(reference, v_ref_m_s=0.0, omega_ref_rad_s=.2)
    v1 = controller_command_v1(actual, pivot_reference, controller_config)
    v2 = controller_command(actual, pivot_reference, 0, controller_config)
    for field in ("v_raw_m_s", "omega_raw_rad_s", "v_cmd_m_s", "omega_cmd_rad_s",
                  "v_left_cmd_m_s", "v_right_cmd_m_s", "command_scale"):
      require(abs(float(v1[field]) - float(v2[field])) < 1e-15, f"pivot mismatch: {field}")
  check("pivot V2 equals V1", pivot_equivalence)

  def feasibility_equivalence() -> None:
    large_reference = dict(reference, v_ref_m_s=.6, omega_ref_rad_s=.7)
    v1 = controller_command_v1(actual, large_reference, controller_config)
    v2 = controller_command(actual, large_reference, 1, controller_config)
    require(v1["command_scale"] < 1.0 and v1["command_scale"] == v2["command_scale"],
            "feasibility scaling changed")
  check("feasibility scaling unchanged", feasibility_equivalence)

  def explicit_zero_mode() -> None:
    zero_reference = dict(reference, v_ref_m_s=0.0)
    forward = controller_command(actual, zero_reference, 1, controller_config)
    reverse = controller_command(actual, zero_reference, -1, controller_config)
    require(forward["lateral_feedback_factor"] == 1 and reverse["lateral_feedback_factor"] == -1,
            "zero speed overrode explicit motion direction")
    require(forward["omega_raw_rad_s"] != reverse["omega_raw_rad_s"],
            "zero-speed mode metadata was ignored")
  check("zero speed retains explicit translational mode", explicit_zero_mode)

  def finite_commands() -> None:
    for direction in (-1, 0, 1):
      result = controller_command(actual, reference, direction, controller_config)
      require(all(math.isfinite(float(value)) for value in result.values()), "non-finite command")
  check("finite commands", finite_commands)

  def frozen_v1() -> None:
    paths = ["config/tracking_controller_v1.json", "scripts/tracking_controller_v1.py",
             "scripts/simulate_tracking_controller_v1.py", "scripts/test_tracking_controller_v1.py"]
    result = subprocess.run(["git", "diff", "--exit-code", "--", *paths], cwd=REPO_ROOT,
                            check=False, capture_output=True, text=True)
    require(result.returncode == 0 and not result.stdout, "canonical Controller V1 changed")
  check("canonical Controller V1 frozen", frozen_v1)
  print(f"result: {len(tests)}/{len(tests)} checks passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
