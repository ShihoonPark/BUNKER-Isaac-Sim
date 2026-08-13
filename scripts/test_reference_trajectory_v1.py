#!/usr/bin/env python3
"""Dependency-free validation harness for BUNKER Reference Trajectory V1."""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from reference_trajectory_v1 import (  # noqa: E402
  build_reference_trajectory, generate_dense_path, load_config, write_outputs,
)


def require(condition: bool, message: str) -> None:
  if not condition:
    raise AssertionError(message)


def finite_tree(value: Any) -> bool:
  if isinstance(value, dict):
    return all(finite_tree(item) for item in value.values())
  if isinstance(value, (list, tuple)):
    return all(finite_tree(item) for item in value)
  if isinstance(value, float):
    return math.isfinite(value)
  return True


def validate(config: dict[str, Any], result: dict[str, Any]) -> list[tuple[str, Callable[[], None]]]:
  dense, rows = result["dense"], result["trajectory"]
  boundaries = result["boundaries"]
  path, limits, tolerances = config["path"], config["trajectory_limits"], config["validation"]
  constraint_tolerance = float(tolerances["constraint_tolerance"])
  track_spacing = float(config["measured_fixed"]["track_center_distance_m"])
  ds_target = float(path["resampling_interval_m"])
  arc_boundaries = [boundary for boundary in boundaries if boundary["segment_type"] == "arc"]

  def config_loaded() -> None:
    require(track_spacing == 0.434, "measured track spacing changed")

  def primitives_valid() -> None:
    require(len(boundaries) == 5 and all(boundary["length_m"] > 0 for boundary in boundaries),
            "invalid primitive boundaries")

  def no_duplicate_joins() -> None:
    distances = [math.hypot(dense[i]["x_m"] - dense[i - 1]["x_m"],
                            dense[i]["y_m"] - dense[i - 1]["y_m"]) for i in range(1, len(dense))]
    require(min(distances) > float(path["duplicate_distance_tolerance_m"]), "duplicate dense sample")

  def finite_endpoint() -> None:
    require(all(math.isfinite(rows[-1][key]) for key in ("x_m", "y_m", "yaw_rad")),
            "non-finite endpoint")

  def increasing_s() -> None:
    require(all(rows[i]["s_m"] > rows[i - 1]["s_m"] for i in range(1, len(rows))),
            "s is not strictly increasing")

  def uniform_resampling() -> None:
    intervals = [rows[i]["s_m"] - rows[i - 1]["s_m"] for i in range(1, len(rows))]
    tolerance = float(tolerances["resampling_tolerance_m"])
    require(all(abs(value - ds_target) <= tolerance for value in intervals[:-1]),
            "non-final resampling interval differs from configured ds")
    require(0 < intervals[-1] <= ds_target + tolerance, "invalid final resampling remainder")

  def increasing_time() -> None:
    require(all(rows[i]["t_s"] > rows[i - 1]["t_s"] for i in range(1, len(rows))),
            "time is not strictly increasing")

  def independent_interval_acceleration() -> None:
    for index in range(len(rows) - 1):
      ds = rows[index + 1]["s_m"] - rows[index]["s_m"]
      require(ds > 0.0, f"interval {index} has non-positive ds")
      speed = rows[index]["v_ref_m_s"]
      next_speed = rows[index + 1]["v_ref_m_s"]
      recomputed = (next_speed ** 2 - speed ** 2) / (2.0 * ds)
      require(recomputed <= limits["maximum_acceleration_m_s2"] + constraint_tolerance,
              f"interval {index} independently exceeds acceleration limit")
      require(recomputed >= -limits["maximum_deceleration_m_s2"] - constraint_tolerance,
              f"interval {index} independently exceeds deceleration limit")
      require(abs(rows[index]["a_ref_m_s2"] - recomputed) <= constraint_tolerance,
              f"stored acceleration disagrees on interval {index}")

  def independent_timestamps() -> None:
    timestamp_tolerance = max(constraint_tolerance, 1e-12)
    require(rows[0]["dt_s"] == 0.0, "first row dt_s is not zero")
    for index in range(len(rows) - 1):
      ds = rows[index + 1]["s_m"] - rows[index]["s_m"]
      denominator = rows[index]["v_ref_m_s"] + rows[index + 1]["v_ref_m_s"]
      require(denominator > 0.0, f"interval {index} has non-positive speed sum")
      expected_dt = 2.0 * ds / denominator
      actual_dt = rows[index + 1]["t_s"] - rows[index]["t_s"]
      require(abs(actual_dt - expected_dt) <= timestamp_tolerance,
              f"timestamp difference disagrees on interval {index}")
      require(abs(rows[index + 1]["dt_s"] - expected_dt) <= timestamp_tolerance,
              f"stored dt_s disagrees on interval {index}")

  def analytic_final_pose_regression() -> None:
    pose_tolerance = 1e-9
    expected = {"x_m": 5.3, "y_m": 2.8, "yaw_rad": 0.0}
    for key, value in expected.items():
      require(abs(rows[-1][key] - value) <= pose_tolerance,
              f"configured V1 final {key} differs from {value}")

  def required_cruise_profile() -> None:
    speed_tolerance = float(limits["phase_speed_tolerance_m_s"])
    maximum_speed = max(row["v_ref_m_s"] for row in rows)
    require(abs(maximum_speed - limits["maximum_body_speed_m_s"]) <= speed_tolerance,
            "profile does not reach configured maximum body speed")
    require(any(row["motion_phase"] == "cruise" for row in rows[1:-1]),
            "profile has no non-terminal cruise sample")

  def endpoint_speeds() -> None:
    require(abs(rows[0]["v_ref_m_s"] - limits["start_speed_m_s"]) <= constraint_tolerance,
            "start speed mismatch")
    require(abs(rows[-1]["v_ref_m_s"] - limits["end_speed_m_s"]) <= constraint_tolerance,
            "end speed mismatch")

  def start_speed() -> None:
    require(abs(rows[0]["v_ref_m_s"] - limits["start_speed_m_s"]) <= constraint_tolerance,
            "start speed mismatch")

  def end_speed() -> None:
    require(abs(rows[-1]["v_ref_m_s"] - limits["end_speed_m_s"]) <= constraint_tolerance,
            "end speed mismatch")

  def all_finite() -> None:
    require(all(finite_tree(row) for row in rows),
            "non-finite trajectory value")

  def within_speed_limit() -> None:
    require(all(row["v_ref_m_s"] <= row["v_limit_m_s"] + constraint_tolerance for row in rows),
            "speed limit exceeded")

  def acceleration_limits() -> None:
    accelerations = [row["a_ref_m_s2"] for row in rows[:-1]]
    require(max(accelerations) <= limits["maximum_acceleration_m_s2"] + constraint_tolerance,
            "acceleration limit exceeded")
    require(min(accelerations) >= -limits["maximum_deceleration_m_s2"] - constraint_tolerance,
            "deceleration limit exceeded")

  def acceleration_limit() -> None:
    require(max(row["a_ref_m_s2"] for row in rows[:-1])
            <= limits["maximum_acceleration_m_s2"] + constraint_tolerance,
            "acceleration limit exceeded")

  def deceleration_limit() -> None:
    require(min(row["a_ref_m_s2"] for row in rows[:-1])
            >= -limits["maximum_deceleration_m_s2"] - constraint_tolerance,
            "deceleration limit exceeded")

  def lateral_limit() -> None:
    require(all(row["v_ref_m_s"] ** 2 * abs(row["curvature_ref_1_m"])
                <= limits["maximum_lateral_acceleration_m_s2"] + constraint_tolerance for row in rows),
            "lateral acceleration limit exceeded")

  def yaw_limit() -> None:
    require(max(abs(row["omega_ref_rad_s"]) for row in rows)
            <= limits["maximum_yaw_rate_rad_s"] + constraint_tolerance, "yaw-rate limit exceeded")

  def track_limits() -> None:
    maximum = limits["maximum_track_surface_speed_m_s"] + constraint_tolerance
    require(all(abs(row[side]) <= maximum for row in rows
                for side in ("v_left_ref_m_s", "v_right_ref_m_s")), "track-speed limit exceeded")

  def one_track_limit(key: str) -> None:
    maximum = limits["maximum_track_surface_speed_m_s"] + constraint_tolerance
    require(all(abs(row[key]) <= maximum for row in rows), f"{key} limit exceeded")

  def mapping() -> None:
    for row in rows:
      require(abs(row["omega_ref_rad_s"] - row["curvature_ref_1_m"] * row["v_ref_m_s"])
              <= constraint_tolerance, "omega mapping mismatch")
      require(abs(row["v_left_ref_m_s"] - (row["v_ref_m_s"] - 0.5 * track_spacing * row["omega_ref_rad_s"]))
              <= constraint_tolerance, "left mapping mismatch")
      require(abs(row["v_right_ref_m_s"] - (row["v_ref_m_s"] + 0.5 * track_spacing * row["omega_ref_rad_s"]))
              <= constraint_tolerance, "right mapping mismatch")

  def omega_mapping() -> None:
    require(all(abs(row["omega_ref_rad_s"] - row["curvature_ref_1_m"] * row["v_ref_m_s"])
                <= constraint_tolerance for row in rows), "omega mapping mismatch")

  def track_mapping(key: str, sign: float) -> None:
    require(all(abs(row[key] - (row["v_ref_m_s"] + sign * 0.5 * track_spacing * row["omega_ref_rad_s"]))
                <= constraint_tolerance for row in rows), f"{key} mapping mismatch")

  def analytic_curvatures() -> None:
    line_values = {row["curvature_ref_1_m"] for row in rows if row["segment_type"] == "line"}
    require(line_values == {0.0}, "line analytic curvature is nonzero")
    require(abs(arc_boundaries[0]["curvature_ref_1_m"] - 1 / 0.75) <= 1e-12,
            "first arc curvature mismatch")
    require(abs(arc_boundaries[1]["curvature_ref_1_m"] + 1 / 0.55) <= 1e-12,
            "second arc curvature mismatch")

  def line_curvature() -> None:
    require(all(row["curvature_ref_1_m"] == 0.0 for row in rows if row["segment_type"] == "line"),
            "line analytic curvature is nonzero")

  def arc_curvature(index: int, expected: float) -> None:
    require(abs(arc_boundaries[index]["curvature_ref_1_m"] - expected) <= 1e-12,
            f"arc {index} curvature mismatch")

  def numeric_curvature() -> None:
    stats = result["summary"]["curvature_error_away_from_boundaries"]
    require(stats["maximum_abs_error_1_m"] <= tolerances["numeric_curvature_interior_max_error_1_m"],
            "numeric curvature maximum error too high")
    require(stats["rms_error_1_m"] <= tolerances["numeric_curvature_interior_rms_error_1_m"],
            "numeric curvature RMS error too high")

  def tighter_arc_slower() -> None:
    corners = result["summary"]["corner_statistics"]
    require(corners[1]["minimum_speed_inside_m_s"] < corners[0]["minimum_speed_inside_m_s"],
            "tighter arc is not slower")

  def corner_transitions() -> None:
    for corner in result["summary"]["corner_statistics"]:
      require(corner["deceleration_began_before_entry_m"] > 0.0,
              f"deceleration did not begin before arc {corner['segment_index']}")
      require(corner["acceleration_resumed_after_exit_m"] is not None,
              f"acceleration did not resume after arc {corner['segment_index']}")

  def deceleration_before_arcs() -> None:
    require(all(corner["deceleration_began_before_entry_m"] > 0.0
                for corner in result["summary"]["corner_statistics"]),
            "deceleration did not begin before every constrained arc")

  def acceleration_after_arcs() -> None:
    require(all(corner["acceleration_resumed_after_exit_m"] is not None
                for corner in result["summary"]["corner_statistics"]),
            "acceleration did not resume after every arc")

  def goal_stop() -> None:
    require(rows[-1]["motion_phase"] == "stop" and rows[-1]["v_ref_m_s"] == 0.0,
            "trajectory does not stop at goal")

  def deterministic_outputs() -> None:
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
      first_paths = write_outputs(build_reference_trajectory(config), Path(first))
      second_paths = write_outputs(build_reference_trajectory(config), Path(second))
      for key in first_paths:
        require(Path(first_paths[key]).read_bytes() == Path(second_paths[key]).read_bytes(),
                f"non-deterministic output: {key}")

  return [
    ("JSON config loads", config_loaded), ("path primitives valid", primitives_valid),
    ("no duplicate joins", no_duplicate_joins), ("finite endpoint", finite_endpoint),
    ("strict cumulative s", increasing_s), ("uniform resampling", uniform_resampling),
    ("strict timestamps", increasing_time),
    ("independent interval acceleration", independent_interval_acceleration),
    ("independent timestamp consistency", independent_timestamps),
    ("configured V1 final pose", analytic_final_pose_regression),
    ("required maximum-speed cruise profile", required_cruise_profile),
    ("configured start speed", start_speed),
    ("configured end speed", end_speed),
    ("finite values", all_finite), ("pointwise speed limits", within_speed_limit),
    ("acceleration limit", acceleration_limit), ("deceleration limit", deceleration_limit),
    ("lateral limit", lateral_limit), ("yaw-rate limit", yaw_limit),
    ("left track-speed limit", lambda: one_track_limit("v_left_ref_m_s")),
    ("right track-speed limit", lambda: one_track_limit("v_right_ref_m_s")),
    ("omega mapping", omega_mapping),
    ("left differential-track mapping", lambda: track_mapping("v_left_ref_m_s", -1.0)),
    ("right differential-track mapping", lambda: track_mapping("v_right_ref_m_s", 1.0)),
    ("line analytic curvature", line_curvature),
    ("first arc analytic curvature", lambda: arc_curvature(0, 1 / 0.75)),
    ("second arc analytic curvature", lambda: arc_curvature(1, -1 / 0.55)),
    ("numeric curvature accuracy", numeric_curvature), ("tighter arc slower", tighter_arc_slower),
    ("deceleration before constrained arcs", deceleration_before_arcs),
    ("acceleration after arcs", acceleration_after_arcs), ("final goal stop", goal_stop),
    ("deterministic CSV/JSON", deterministic_outputs),
  ]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/reference_trajectory_v1.json")
  args = parser.parse_args()
  config = load_config(args.config.resolve())
  # Direct primitive generation is intentionally exercised separately from the full pipeline.
  generate_dense_path(config)
  result = build_reference_trajectory(config)
  tests = validate(config, result)
  failures = []
  for name, test in tests:
    try:
      test()
      print(f"PASS: {name}")
    except Exception as error:  # Report the complete small harness rather than stopping early.
      failures.append((name, error))
      print(f"FAIL: {name}: {error}")
  print(f"result: {len(tests) - len(failures)}/{len(tests)} checks passed")
  return 1 if failures else 0


if __name__ == "__main__":
  raise SystemExit(main())
