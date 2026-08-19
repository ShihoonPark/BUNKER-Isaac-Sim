#!/usr/bin/env python3
"""Pure and staged post-hoc validation for Latest Real Localization Sync V1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any, Callable

from map_route_reference_v1 import build_map_route_reference, parse_tum, wrap_to_pi
from map_route_visualization_v1 import alignment_diagnostics, load_ply, transform_map_points

REPO_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
  if not condition: raise AssertionError(message)


def finite_tree(value: Any) -> bool:
  if isinstance(value, dict): return all(finite_tree(item) for item in value.values())
  if isinstance(value, list): return all(finite_tree(item) for item in value)
  return not isinstance(value, float) or math.isfinite(value)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--reference-config", type=Path, default=REPO_ROOT / "config/latest_real_localization_reference_v1.json")
  parser.add_argument("--require-kinematic", action="store_true")
  parser.add_argument("--require-isaac", action="store_true")
  parser.add_argument("--require-visualization", action="store_true")
  args = parser.parse_args(); result = build_map_route_reference(args.reference_config.resolve())
  config, source, trajectory, summary = result["config"], result["source"], result["trajectory"], result["summary"]
  ply = load_ply(Path(config["source"]["map_artifact"])); tests: list[str] = []

  def check(name: str, function: Callable[[], None]) -> None:
    function(); tests.append(name); print(f"PASS: {name}")

  check("Bag C PLY identity", lambda: require(ply["format"] == "binary_little_endian" and
    ply["vertex_count"] == 11215 and ply["fields"] == ["x", "y", "z", "intensity"], "Bag C PLY mismatch"))
  check("Bag C PLY finite", lambda: require(all(math.isfinite(value) for point in ply["points"] for value in point), "PLY nonfinite"))
  check("Bag D accepted TUM identity", lambda: require(len(source) == 2176 and
    abs(summary["source_data"]["duration_s"] - 218.29528260231018) < 1e-6, "Bag D TUM mismatch"))
  check("Bag C/D same-frame provenance", lambda: require(config["source"]["source_parent_frame"] == "map" and
    config["source"]["pose_semantics"] == "T_map_lidar" and config["source"]["base_transform_status"] == "unknown", "frame metadata"))
  check("XY-only open traversal", lambda: require(config["processing"] == {"use_xy_only": True,
    "preserve_traversal_order": True, "artificial_loop_closure": False} and
    summary["source_data"]["start_end_xy_distance_m"] > .4, "route policy"))

  def accepted_gaps() -> None:
    timestamps = [float(row["source_timestamp_s"]) for row in source]
    dt = [b-a for a,b in zip(timestamps,timestamps[1:])]
    ds = [math.hypot(float(b["x_m"])-float(a["x_m"]), float(b["y_m"])-float(a["y_m"]))
          for a,b in zip(source,source[1:])]
    yaw = [float(row["recorded_yaw_rad"]) for row in source]
    dyaw = [abs(wrap_to_pi(b-a)) for a,b in zip(yaw,yaw[1:])]
    require(max(dt) < .201 and max(ds) < .111 and max(dyaw) < math.radians(7.75), "material accepted-only gap")
    require(statistics.median(dt) > .099 and statistics.median(dt) < .101, "unexpected accepted rate")
  check("accepted-only gaps remain isolated", accepted_gaps)
  check("maneuver segmentation derives latest counts", lambda: require(
    summary["reference"]["maneuvers"]["segment_counts"] == {"FORWARD": 3, "REVERSE": 2, "PIVOT": 3}, "latest counts"))
  check("signed and pivot semantics", lambda: require(set(int(row["motion_direction"]) for row in trajectory) == {-1,0,1} and
    all(int(row["curvature_valid"]) == (int(row["motion_direction"]) != 0) for row in trajectory) and
    min(float(row["v_ref_m_s"]) for row in trajectory) < 0 < max(float(row["v_ref_m_s"]) for row in trajectory), "mode semantics"))
  check("dynamic limits and track consistency", lambda: require(
    summary["reference"]["maximum_reference_speed_m_s"] <= .6 + 1e-12 and
    summary["reference"]["maximum_abs_yaw_rate_rad_s"] <= .7 + 1e-12 and
    summary["reference"]["maximum_abs_left_track_speed_m_s"] <= .8 + 1e-12 and
    summary["reference"]["maximum_abs_right_track_speed_m_s"] <= .8 + 1e-12, "dynamic limit"))
  check("finite generated reference", lambda: require(finite_tree(summary) and all(
    math.isfinite(float(value)) for row in trajectory for value in row.values() if isinstance(value,(int,float))), "nonfinite"))
  check("geometry preservation guard", lambda: require(abs(
    summary["reference"]["maneuver_geometry_distortion"]["length_change_percent"]) <= 2.0 and
    summary["reference"]["maneuver_geometry_distortion"]["maximum_displacement_m"] < .01, "geometry distortion"))
  check("ordered projection valid", lambda: require(summary["reference"]["projection_diagnostic"]["nonmonotonic_jump_count"] == 0, "projection"))
  pivot_reports = summary["reference"]["maneuvers"]["pivot_reference_reports"]
  report_by_id = {int(row["segment_id"]): row for row in pivot_reports}

  def deterministic_policy() -> None:
    repeated = build_map_route_reference(args.reference_config.resolve())
    require(repeated["summary"]["reference"]["maneuvers"]["pivot_reference_reports"] == pivot_reports,
            "pivot boundary classification is nondeterministic")
  check("pivot boundary classification deterministic", deterministic_policy)
  check("long translation retains geometric yaw", lambda: require(all(
    row["pivot_boundary_yaw_method"] == "GEOMETRIC" for row in trajectory
    if int(row["segment_id"]) == 0), "long segment fallback"))
  check("short pivot-adjacent translation uses fallback", lambda: require(all(
    row["pivot_boundary_yaw_method"] == "RECORDED_YAW_FALLBACK" for row in trajectory
    if int(row["segment_id"]) in (2, 4, 6)), "short segment did not fallback"))
  check("fallback yaw remains continuously unwrapped", lambda: require(max(abs(
    float(trajectory[index]["yaw_rad"]) - float(trajectory[index-1]["yaw_rad"]))
    for index in range(1, len(trajectory))) < math.pi, "fallback yaw branch wrap"))
  def reliable_tangent_semantics() -> None:
    legacy_latest_config = json.loads(args.reference_config.resolve().read_text())
    legacy_latest_config["translational_geometry"].pop("pivot_boundary_yaw_policy")
    with tempfile.TemporaryDirectory() as directory:
      temporary = Path(directory) / "legacy_latest.json"
      temporary.write_text(json.dumps(legacy_latest_config), encoding="utf-8")
      legacy_latest = build_map_route_reference(temporary)
    forward = [row for row in trajectory if int(row["segment_id"]) == 0]
    legacy_forward = [row for row in legacy_latest["trajectory"] if int(row["segment_id"]) == 0]
    old = build_map_route_reference(REPO_ROOT / "config/map_route_reference_v1.json")
    reverse_error = old["summary"]["reference"]["path_tangent_vs_body_yaw"]["reverse"]["maximum_abs_error_from_expected_rad"]
    require(len(forward) == len(legacy_forward) and max(abs(float(a["yaw_rad"])-float(b["yaw_rad"]))
      for a,b in zip(forward,legacy_forward)) < 1e-12 and reverse_error < .3,
      "reliable tangent semantics changed")
  check("reliable forward/reverse tangent semantics retained", reliable_tangent_semantics)
  check("pivot branches contain no approximately-pi mismatch", lambda: require(max(abs(
    float(row["yaw_change_difference_rad"])) for row in pivot_reports) < .7, "pivot branch mismatch"))
  check("latest Pivot 5 branch fixed", lambda: require(abs(
    float(report_by_id[5]["yaw_change_difference_rad"])) < 1e-12 and
    report_by_id[5]["entry_boundary_method"] == "RECORDED_YAW_FALLBACK" and
    report_by_id[5]["exit_boundary_method"] == "RECORDED_YAW_FALLBACK", "Pivot 5 fallback"))

  def old_baseline_policy() -> None:
    old = build_map_route_reference(REPO_ROOT / "config/map_route_reference_v1.json")
    require("pivot_boundary_yaw_policy" not in old["config"]["translational_geometry"], "old policy changed")
    require(old["summary"]["reference"]["maneuvers"]["segment_counts"] ==
            {"FORWARD": 10, "REVERSE": 9, "PIVOT": 4}, "old route changed")
    require(all(row.get("pivot_boundary_yaw_method") == "GEOMETRIC" for row in old["trajectory"]
                if int(row["motion_direction"]) != 0), "old route used fallback")
  check("old 20260814 legacy policy reproducible", old_baseline_policy)

  if args.require_kinematic or args.require_isaac or args.require_visualization:
    path = REPO_ROOT / "logs/latest_real_localization_kinematic_v1/summary.json"
    require(path.exists(), "kinematic output missing"); kin = json.loads(path.read_text()); metrics = kin["v2_scenarios"]["map_nominal"]
    check("one nominal kinematic route finite", lambda: require(finite_tree(kin) and
      kin["run_policy"].startswith("exactly one nominal"), "kinematic nonfinite/policy"))
    check("kinematic route remains numerically bounded", lambda: require(
      metrics["active"]["maximum_abs_time_aligned_xy_error"] < .5 and
      metrics["final_xy_goal_error_m"] < .5 and set(metrics["mode_metrics"]) == {"forward","reverse","pivot"}, "kinematic divergence"))
    check("kinematic signed reverse gate passes", lambda: require(
      kin["kinematic_gate"]["passed"] and
      kin["kinematic_gate"]["checks"]["reverse_command_direction_dominantly_negative"] and
      float(kin["kinematic_gate"]["reverse_negative_command_fraction"]) > .98,
      "corrected latest route failed signed reverse gate"))
    check("all latest kinematic segments reported", lambda: require(set(metrics["segment_metrics"]) == set(map(str,range(8))), "kinematic segments"))

  if args.require_isaac or args.require_visualization:
    path = REPO_ROOT / "logs/latest_real_localization_isaac_v1/summary.json"
    require(path.exists(), "Isaac output missing"); isaac = json.loads(path.read_text())
    check("one nominal Isaac route finite", lambda: require(finite_tree(isaac), "Isaac nonfinite"))
    check("latest provenance embedded", lambda: require(isaac["dataset_provenance"]["localization"]["accepted_pose_count"] == 2176, "provenance"))
    check("all latest Isaac segments reported", lambda: require(set(isaac["segments"]) == set(map(str,range(8))), "Isaac segments"))
    check("Isaac support stable", lambda: require(isaac["plant"]["left_unsupported_sample_count"] == 0 and
      isaac["plant"]["right_unsupported_sample_count"] == 0 and
      isaac["plant"]["maximum_abs_custom_force_normal_component_n"] < 1e-7, "plant support"))

  if args.require_visualization:
    directory = REPO_ROOT / "logs/latest_real_localization_visualization_v1"
    visual = json.loads((directory / "summary.json").read_text())["visualization"]
    check("latest visualization artifacts", lambda: require((directory/"latest_real_localization_visualization_v1.usd").exists() and
      visual["ply"]["vertex_count"] == 11215 and visual["observer"]["actual_trail_display_point_count"] > 1, "visual output"))
    check("latest map/reference alignment", lambda: require(visual["coordinate_alignment"]["start_difference_m"] < 1e-12 and
      visual["coordinate_alignment"]["endpoint_difference_m"] < 1e-12, "alignment"))
    check("visual map remains nonphysical", lambda: require(not visual["observer"]["physics_apis_authored"], "visual physics API"))
  print(f"result: {len(tests)}/{len(tests)} checks passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
