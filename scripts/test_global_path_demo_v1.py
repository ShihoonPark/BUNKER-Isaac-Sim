#!/usr/bin/env python3
"""Pure and post-hoc regression checks for Multiple Closed-Loop Global Paths V1."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable

from global_path_demo_v1 import build_demo_reference
from map_route_visualization_v1 import load_ply


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS = ("rounded_loop", "zigzag_loop", "lawnmower_loop")


def require(condition: bool, message: str) -> None:
  if not condition: raise AssertionError(message)


def finite_tree(value: Any) -> bool:
  if isinstance(value, dict): return all(finite_tree(item) for item in value.values())
  if isinstance(value, list): return all(finite_tree(item) for item in value)
  return not isinstance(value, float) or math.isfinite(value)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/global_path_demo_v1.json")
  parser.add_argument("--require-kinematic", action="store_true")
  parser.add_argument("--require-isaac", action="store_true")
  args = parser.parse_args(); config_path = args.config.resolve()
  results = {name: build_demo_reference(config_path, name) for name in PRESETS}
  config = results[PRESETS[0]]["config"]; tests: list[str] = []

  def check(name: str, function: Callable[[], None]) -> None:
    function(); tests.append(name); print(f"PASS: {name}")

  ply = load_ply(Path(config["map"]["point_cloud_path"]))
  check("latest Bag C PLY identity", lambda: require(
    ply["vertex_count"] == 11215 and ply["format"] == "binary_little_endian" and
    ply["fields"] == ["x", "y", "z", "intensity"], "Bag C PLY mismatch"))
  check("Bag C PLY remains finite XY display context", lambda: require(
    all(math.isfinite(value) for point in ply["points"] for value in point), "nonfinite PLY"))

  def waypoint_coverage() -> None:
    ply_x, ply_y = config["map"]["measured_ply_xy_bounds_m"]["x"], config["map"]["measured_ply_xy_bounds_m"]["y"]
    sanity_x = config["map"]["accepted_bag_d_xy_sanity_bounds_m"]["x"]
    sanity_y = config["map"]["accepted_bag_d_xy_sanity_bounds_m"]["y"]
    for result in results.values():
      points = result["preset"]["waypoints_map_xy_m"]
      require(all(ply_x[0] < x < ply_x[1] and ply_y[0] < y < ply_y[1] for x, y in points),
              "waypoint outside measured PLY XY bounds")
      require(all(sanity_x[0] - .01 <= x <= sanity_x[1] + .01 and
                  sanity_y[0] - .01 <= y <= sanity_y[1] + .01 for x, y in points),
              "waypoint outside accepted-route coverage sanity envelope")
  check("manual waypoints use observed classroom coverage", waypoint_coverage)
  check("provider is explicit manual non-planner", lambda: require(all(
    not result["summary"]["provider"]["planner_implemented"] and
    result["summary"]["provider"]["waypoint_frame"] == "map" for result in results.values()),
    "planner semantics changed"))

  def closed_geometry() -> None:
    for result in results.values():
      geometry = result["summary"]["geometry"]
      require(geometry["closed"] and geometry["raw_self_intersection_count"] == 0,
              "core path is not a simple closed loop")
      require(geometry["position_closure_error_m"] < 1e-12 and
              geometry["tangent_closure_error_rad"] < 1e-12 and
              geometry["curvature_closure_error_1_m"] < 1e-12,
              "periodic geometry closure mismatch")
      require(abs(abs(geometry["yaw_change_per_lap_rad"]) - 2.0 * math.pi) < 1e-8,
              "closed yaw winding mismatch")
      require(max(abs(float(row["curvature_ref_1_m"])) for row in result["periodic_lap"]) < 3.0,
              "curvature is not finite/trackable")
  check("position tangent curvature and yaw are closed", closed_geometry)

  def periodic_profiles() -> None:
    for result in results.values():
      periodic = result["summary"]["periodic_profile"]
      execution = result["summary"]["execution_profile"]
      require(periodic["speed_closure_error_m_s"] < 1e-12 and
              periodic["omega_closure_error_rad_s"] < 1e-12, "periodic command closure")
      require(execution["laps"] == 2 and execution["interior_lap_stop_count"] == 0 and
              max(execution["interior_lap_boundary_speed_errors_from_periodic_m_s"], default=0.0) < 1e-12,
              "lap boundary stopped or departed from steady periodic profile")
      require(execution["launch_speed_m_s"] == 0.0 and execution["final_speed_m_s"] == 0.0,
              "launch/final stop envelope missing")
      require(all(periodic["phase_sample_counts"][phase] > 0
                  for phase in ("accel", "cruise", "decel", "curve_limited")),
              "corner-aware speed phases incomplete")
  check("periodic interior plus launch and final stop semantics", periodic_profiles)

  def constraints() -> None:
    for result in results.values():
      values = result["summary"]["execution_profile"]; limits = values["limits"]; tolerance = 1e-9
      require(values["maximum_reference_speed_m_s"] <= limits["maximum_body_speed_m_s"] + tolerance and
              values["maximum_positive_acceleration_m_s2"] <= limits["maximum_acceleration_m_s2"] + tolerance and
              values["maximum_deceleration_magnitude_m_s2"] <= limits["maximum_deceleration_m_s2"] + tolerance and
              values["maximum_lateral_acceleration_m_s2"] <= limits["maximum_lateral_acceleration_m_s2"] + tolerance and
              values["maximum_abs_yaw_rate_rad_s"] <= limits["maximum_yaw_rate_rad_s"] + tolerance and
              values["maximum_abs_left_track_speed_m_s"] <= limits["maximum_track_surface_speed_m_s"] + tolerance and
              values["maximum_abs_right_track_speed_m_s"] <= limits["maximum_track_surface_speed_m_s"] + tolerance,
              "trajectory constraint violation")
  check("all canonical dynamic limits hold", constraints)

  def deterministic() -> None:
    repeated = {name: build_demo_reference(config_path, name) for name in PRESETS}
    require(all(results[name]["trajectory"] == repeated[name]["trajectory"] and
                results[name]["summary"] == repeated[name]["summary"] for name in PRESETS),
            "global path build is nondeterministic")
  check("all presets build deterministically", deterministic)

  def frozen_baselines() -> None:
    paths = ["config/latest_real_localization_reference_v1.json",
             "scripts/map_route_reference_v1.py", "scripts/run_latest_real_localization_kinematic_v1.py",
             "config/tracking_controller_v2_direction_aware.json",
             "scripts/tracking_controller_v2_direction_aware.py", "config/tracked_force_plant_v2.json",
             "scripts/run_map_route_isaac_closed_loop_v1.py"]
    completed = subprocess.run(["git", "diff", "--exit-code", "--", *paths], cwd=REPO_ROOT,
                               check=False, capture_output=True, text=True)
    require(completed.returncode == 0 and not completed.stdout, "existing open-route/controller/plant baseline changed")
  check("existing open-route controller and plant baselines are frozen", frozen_baselines)

  def observer_only_source() -> None:
    source = (REPO_ROOT / "scripts/global_path_demo_visualization_v1.py").read_text()
    forbidden = ("CollisionAPI", "RigidBodyAPI", "MassAPI", "PhysxSchema", "ApplyForce")
    require(not any(token in source for token in forbidden),
            "visualization observer contains a physics-authoring API")
  check("visualization source is observer-only", observer_only_source)

  if args.require_kinematic or args.require_isaac:
    def kinematic_outputs() -> None:
      for name in PRESETS:
        path = REPO_ROOT / config["output"]["default_directory"] / name / config["output"]["kinematic_summary_filename"]
        require(path.is_file(), f"missing {name} kinematic summary")
        summary = json.loads(path.read_text())
        require(finite_tree(summary) and summary["gate"]["passed"], f"{name} kinematic gate failed")
        require(summary["projection"]["nonmonotonic_step_count"] == 0,
                f"{name} lap-aware projected progress moved backward")
        require(set(summary["lap_metrics"]["laps"]) == {"1", "2"} and
                summary["lap_metrics"]["lap_1_to_2"]["lap_2_nondivergent"],
                f"{name} second lap diverged")
    check("three nominal kinematic gates pass", kinematic_outputs)

  if args.require_isaac:
    def isaac_outputs() -> None:
      for name in PRESETS:
        directory = REPO_ROOT / config["output"]["default_directory"] / name
        summary_path = directory / config["output"]["isaac_summary_filename"]
        require(summary_path.is_file(), f"missing {name} Isaac summary")
        summary = json.loads(summary_path.read_text())
        require(finite_tree(summary) and summary["gate"]["passed"], f"{name} Isaac gate failed")
        require(summary["plant"]["left_unsupported_sample_count"] == 0 and
                summary["plant"]["right_unsupported_sample_count"] == 0 and
                summary["plant"]["maximum_abs_custom_force_normal_component_n"] <= 1e-7,
                f"{name} Isaac support/force invariant failed")
        require(summary["projection"]["nonmonotonic_step_count"] == 0 and
                summary["lap_metrics"]["lap_1_to_2"]["lap_2_nondivergent"],
                f"{name} Isaac progress or second lap failed")
    check("three nominal Isaac headless gates pass", isaac_outputs)

  print(f"result: {len(tests)}/{len(tests)} checks passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
