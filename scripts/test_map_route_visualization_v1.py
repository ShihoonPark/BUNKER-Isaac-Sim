#!/usr/bin/env python3
"""Pure and post-hoc checks for Map Route Visualization V1."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Callable

from map_route_reference_v1 import build_map_route_reference, parse_tum
from map_route_visualization_v1 import (
  ActualTrailSampler, alignment_diagnostics, load_ply, map_xy_to_reference, transform_map_points,
)
from run_map_route_isaac_closed_loop_v1 import _notify_observer
from run_map_route_visualization_v1 import REPO_ROOT, load_visualization_config


def require(condition: bool, message: str) -> None:
  if not condition: raise AssertionError(message)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/map_route_visualization_v1.json")
  parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs/map_route_visualization_v1")
  parser.add_argument("--pure-only", action="store_true"); parser.add_argument("--require-output", action="store_true")
  args = parser.parse_args(); config = load_visualization_config(args.config.resolve())
  stage = json.loads((REPO_ROOT / config["stage_e_config"]).read_text())
  result = build_map_route_reference(REPO_ROOT / stage["baseline_configs"]["map_route_reference"])
  reference = result["trajectory"]; transform = result["summary"]["reference"]["maneuver_normalization_transform"]
  raw = parse_tum(Path(config["raw_trajectory_path"]), float(result["config"]["diagnostic_guards"]["quaternion_norm_tolerance"]))
  ply = load_ply(Path(config["point_cloud_path"])); tests: list[str] = []

  def check(name: str, function: Callable[[], None]) -> None:
    function(); tests.append(name); print(f"PASS: {name}")

  check("visualization config loads", lambda: require(config["metadata"]["all_parameters_are_visualization_only"], "not display-only"))
  check("real binary PLY parses", lambda: require(ply["format"] == "binary_little_endian" and ply["vertex_count"] == 13058, "PLY identity"))
  check("PLY XYZ finite and plausible", lambda: require(ply["fields"] == ["x", "y", "z", "intensity"] and all(
    math.isfinite(value) for point in ply["points"] for value in point), "PLY fields/finite"))

  source_text = (REPO_ROOT / "scripts/map_route_visualization_v1.py").read_text()
  check("visual prims author no physics API", lambda: require(all(token not in source_text for token in
    ("CollisionAPI", "RigidBodyAPI", "MassAPI", "PhysxCollisionAPI")), "visual code authors physics API"))

  def exact_transform() -> None:
    first = map_xy_to_reference(float(raw[0]["x_m"]), float(raw[0]["y_m"]), transform)
    require(max(abs(value) for value in first) < 1e-12, "raw origin not transformed by Stage-D transform")
    require(abs(float(transform["map_to_local_rotation_rad"]) + float(transform["map_initial_body_yaw_rad"])) < 1e-12,
            "Stage-D transform inconsistent")
  check("exact Stage-D normalization transform", exact_transform)

  diagnostics = alignment_diagnostics(raw, reference, transform_map_points(ply["points"], transform, 0.0), transform)
  check("raw and processed starts align", lambda: require(diagnostics["start_difference_m"] < 1e-12 and
    math.hypot(float(reference[0]["x_m"]), float(reference[0]["y_m"])) < 1e-12, "start mismatch"))
  check("raw traversal remains open", lambda: require(diagnostics["raw_start_end_distance_m"] > .08, "raw route closed"))
  check("reference traversal remains open", lambda: require(diagnostics["reference_start_end_distance_m"] > .08 and
    math.dist((float(reference[0]["x_m"]), float(reference[0]["y_m"])),
              (float(reference[-1]["x_m"]), float(reference[-1]["y_m"]))) > .08, "reference closed"))
  check("all maneuver segments preserved", lambda: require(len({int(row["segment_id"]) for row in reference}) == 23, "segment count"))
  check("maneuver semantics preserved", lambda: require({direction: len({int(row["segment_id"]) for row in reference
    if int(row["motion_direction"]) == direction}) for direction in (1, -1, 0)} == {1: 10, -1: 9, 0: 4}, "mode counts"))
  check("mode values remain discrete", lambda: require(set(int(row["motion_direction"]) for row in reference) == {-1, 0, 1}, "interpolated mode"))
  check("ordered endpoint markers differ", lambda: require((reference[0]["x_m"], reference[0]["y_m"]) !=
    (reference[-1]["x_m"], reference[-1]["y_m"]), "marker endpoints collapsed"))

  def z_only() -> None:
    original = transform_map_points(ply["points"][:10], transform, 0.0)
    shifted = transform_map_points(ply["points"][:10], transform, .123)
    require(all(abs(a[0]-b[0]) < 1e-15 and abs(a[1]-b[1]) < 1e-15 and abs((b[2]-a[2])-.123) < 1e-12
                for a, b in zip(original, shifted)), "Z offset affected XY")
  check("visual Z offset leaves XY invariant", z_only)

  def trail_is_observer_only() -> None:
    rows = [{"physics_time_s": index * .05, "actual_local_x_m": index * .01, "actual_local_y_m": 0.0}
            for index in range(8)]
    before = copy.deepcopy(rows); sampler = ActualTrailSampler(.1, .005, .05)
    for row in rows: sampler.consider(row)
    require(rows == before and all(point[:2] == (rows[index]["actual_local_x_m"], rows[index]["actual_local_y_m"])
                                   for point, index in zip(sampler.points, (0, 2, 4, 6))), "trail changed numeric rows")
  check("trail downsampling is read-only", trail_is_observer_only)

  def observer_hook() -> None:
    class Recorder:
      def event(self, **payload): self.payload = payload
    _notify_observer(None, "event", value=1)
    recorder = Recorder(); _notify_observer(recorder, "event", value=2)
    require(recorder.payload == {"value": 2}, "observer hook")
  check("optional observer defaults to no-op", observer_hook)

  def frozen_values() -> None:
    paths = ["config/map_route_reference_v1.json", "scripts/map_route_reference_v1.py",
             "scripts/generate_map_route_reference_v1.py", "config/tracking_controller_v2_direction_aware.json",
             "scripts/tracking_controller_v2_direction_aware.py", "config/tracked_force_plant_v2.json",
             "config/map_route_isaac_closed_loop_v1.json"]
    completed = subprocess.run(["git", "diff", "--exit-code", "--", *paths], cwd=REPO_ROOT,
                               check=False, capture_output=True, text=True)
    require(completed.returncode == 0 and not completed.stdout, "frozen baseline changed")
  check("controller plant reference values frozen", frozen_values)

  if args.pure_only:
    print(f"result: {len(tests)}/{len(tests)} checks passed (pure)"); return 0
  csv_path = args.output_dir / config["output"]["csv_filename"]
  summary_path = args.output_dir / config["output"]["summary_filename"]
  usd_path = args.output_dir / config["output"]["stage_filename"]
  if not csv_path.exists() or not summary_path.exists() or not usd_path.exists():
    if args.require_output: raise FileNotFoundError("visualization output is required but missing")
    print(f"result: {len(tests)}/{len(tests)} checks passed (output absent; post-hoc skipped)"); return 0
  with csv_path.open(newline="", encoding="utf-8") as stream: rows = list(csv.DictReader(stream))
  summary = json.loads(summary_path.read_text()); visual = summary["visualization"]
  check("visual output layer inventory", lambda: require(visual["observer"]["point_count"] == 13058 and
    visual["observer"]["actual_trail_display_point_count"] > 1 and not visual["observer"]["physics_apis_authored"], "layer output"))
  check("visual run preserves numeric Stage-E schema", lambda: require(len(rows) > 9000 and
    {"actual_local_x_m", "actual_local_y_m", "cross_track_error_m", "reference_segment_id"} <= set(rows[0]), "CSV schema"))
  print(f"result: {len(tests)}/{len(tests)} checks passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
