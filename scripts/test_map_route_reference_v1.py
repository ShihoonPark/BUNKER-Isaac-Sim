#!/usr/bin/env python3
"""Standard-library validation harness for Stage-D map-route reference generation."""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from typing import Callable

from map_route_reference_v1 import (
  MapRouteError, build_map_route_reference, cumulative_arc_length,
  parse_tum, segment_maneuvers, wrap_to_pi,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
  if not condition:
    raise AssertionError(message)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/map_route_reference_v1.json")
  args = parser.parse_args()
  result = build_map_route_reference(args.config.resolve())
  config, source, cleaned, uniform = (result[key] for key in ("config", "source", "cleaned", "uniform"))
  trajectory, summary = result["trajectory"], result["summary"]
  source_summary, processing, reference = (summary[key] for key in ("source_data", "processing", "reference"))
  canonical = result["canonical_config"]
  tolerance = float(canonical["validation"]["constraint_tolerance"])
  tests: list[str] = []

  def check(name: str, function: Callable[[], None]) -> None:
    function(); tests.append(name); print(f"PASS: {name}")

  def parser_schema() -> None:
    with tempfile.TemporaryDirectory() as directory:
      bad = Path(directory) / "bad.tum"
      bad.write_text("0 0 0 0 0 0 1\n", encoding="utf-8")
      try:
        parse_tum(bad, .01)
      except MapRouteError as error:
        require("exactly 8 fields" in str(error), "wrong schema error")
      else:
        raise AssertionError("seven-field TUM row was accepted")
  check("TUM eight-field schema", parser_schema)
  check("strict source timestamps", lambda: require(all(
    float(source[index]["source_timestamp_s"]) > float(source[index - 1]["source_timestamp_s"])
    for index in range(1, len(source))), "timestamps not increasing"))
  check("finite source", lambda: require(all(math.isfinite(float(value)) for row in source for key, value in row.items()
                                              if key != "source_index"), "non-finite source value"))
  check("quaternion validity", lambda: require(max(abs(float(row["quaternion_norm"]) - 1.0) for row in source) <=
                                                 config["diagnostic_guards"]["quaternion_norm_tolerance"],
                                                 "invalid quaternion norm"))
  check("actual source count", lambda: require(len(source) == 867, "actual source count changed"))
  check("raw XY length", lambda: require(abs(source_summary["raw_xy_polyline_length_m"] - 25.6533) <= .01,
                                           "raw XY length mismatch"))
  check("raw endpoint distance", lambda: require(abs(source_summary["start_end_xy_distance_m"] - .09156) <= .002,
                                                   "raw endpoint distance mismatch"))
  check("traversal order preserved", lambda: require(
    all(int(cleaned[index]["source_index"]) > int(cleaned[index - 1]["source_index"])
        for index in range(1, len(cleaned))), "cleaning reordered source"))
  check("open traversal", lambda: require(config["processing"]["artificial_loop_closure"] is False and
                                            (source[-1]["x_m"], source[-1]["y_m"]) !=
                                            (source[0]["x_m"], source[0]["y_m"]), "route was closed"))
  check("cleaning endpoints", lambda: require(
    int(cleaned[0]["source_index"]) == int(source[0]["source_index"]) and
    int(cleaned[-1]["source_index"]) == int(source[-1]["source_index"]) and
    cleaned[0]["x_m"] == source[0]["x_m"] and cleaned[-1]["x_m"] == source[-1]["x_m"],
    "cleaning did not preserve endpoints"))
  def uniform_resampling() -> None:
    ds = [float(uniform[index]["s_m"]) - float(uniform[index - 1]["s_m"])
          for index in range(1, len(uniform))]
    require(all(abs(value - .02) < 1e-10 for value in ds[:-1]), "regular resampling interval changed")
    require(0.0 < ds[-1] <= .02 + 1e-10, "invalid final resampling remainder")
  check("uniform resampling", uniform_resampling)
  check("smoothing endpoints", lambda: require(
    result["smoothed_map"][0]["x_m"] == uniform[0]["x_m"] and
    result["smoothed_map"][0]["y_m"] == uniform[0]["y_m"] and
    result["smoothed_map"][-1]["x_m"] == uniform[-1]["x_m"] and
    result["smoothed_map"][-1]["y_m"] == uniform[-1]["y_m"], "smoothing moved endpoint"))
  check("smoothing distortion guards", lambda: require(all(
    processing["smoothing_distortion"]["guards"].values()), "smoothing guard failed"))
  check("normalized start", lambda: require(max(abs(float(trajectory[0][key])) for key in
                                                  ("x_m", "y_m", "yaw_rad")) < 1e-10,
                                                  "normalized start mismatch"))
  check("endpoint separation preserved", lambda: require(abs(
    reference["processed_start_end_distance_m"] - source_summary["start_end_xy_distance_m"]) < 1e-12,
    "rigid normalization/smoothing changed endpoint separation"))
  check("finite final reference", lambda: require(all(
    math.isfinite(float(value)) for row in trajectory for value in row.values()
    if isinstance(value, (int, float))), "non-finite reference"))
  check("continuous unwrapped yaw", lambda: require(max(abs(float(trajectory[index]["yaw_rad"]) -
                                                            float(trajectory[index - 1]["yaw_rad"]))
                                                        for index in range(1, len(trajectory))) < math.pi,
                                                        "yaw contains wrap discontinuity"))
  check("endpoint speeds", lambda: require(trajectory[0]["v_ref_m_s"] == 0.0 and
                                               trajectory[-1]["v_ref_m_s"] == 0.0,
                                               "reference endpoint speed mismatch"))
  check("new time monotonic", lambda: require(trajectory[0]["t_s"] == 0.0 and all(
    float(trajectory[index]["t_s"]) > float(trajectory[index - 1]["t_s"])
    for index in range(1, len(trajectory))), "new timestamps invalid"))

  def independent_limits() -> None:
    limits = canonical["trajectory_limits"]
    b = float(canonical["measured_fixed"]["track_center_distance_m"])
    for index, row in enumerate(trajectory):
      v, omega, curvature = (float(row[key]) for key in
                             ("v_ref_m_s", "omega_ref_rad_s", "curvature_ref_1_m"))
      require(abs(v) <= float(limits["maximum_body_speed_m_s"]) + tolerance, f"body limit at {index}")
      require(abs(omega) <= float(limits["maximum_yaw_rate_rad_s"]) + tolerance, f"yaw limit at {index}")
      if int(row["curvature_valid"]):
        require(v * v * abs(curvature) <= float(limits["maximum_lateral_acceleration_m_s2"]) + tolerance,
                f"lateral limit at {index}")
      require(abs(v - .5 * b * omega) <= float(limits["maximum_track_surface_speed_m_s"]) + tolerance,
              f"left track limit at {index}")
      require(abs(v + .5 * b * omega) <= float(limits["maximum_track_surface_speed_m_s"]) + tolerance,
              f"right track limit at {index}")
    for index in range(len(trajectory) - 1):
      dt = float(trajectory[index + 1]["t_s"]) - float(trajectory[index]["t_s"])
      acceleration = (float(trajectory[index + 1]["v_ref_m_s"]) -
                      float(trajectory[index]["v_ref_m_s"])) / dt
      require(acceleration <= float(limits["maximum_acceleration_m_s2"]) + tolerance,
              f"acceleration limit at {index}")
      require(acceleration >= -float(limits["maximum_deceleration_m_s2"]) - tolerance,
              f"deceleration limit at {index}")
  check("independent canonical dynamic limits", independent_limits)
  def omega_mapping() -> None:
    for index, row in enumerate(trajectory):
      direction = int(row["motion_direction"])
      if direction == 0:
        require(int(row["curvature_valid"]) == 0 and float(row["curvature_ref_1_m"]) == 0.0,
                f"pivot curvature semantics at {index}")
        continue
      expected = abs(float(row["v_ref_m_s"])) * float(row["curvature_ref_1_m"])
      require(abs(float(row["omega_ref_rad_s"]) - expected) < 1e-12,
              f"signed omega mapping at {index}")
  check("signed translational omega mapping", omega_mapping)
  check("final arc length", lambda: require(abs(float(trajectory[-1]["s_m"]) -
                                                sum(math.hypot(float(trajectory[index]["x_m"]) -
                                                               float(trajectory[index - 1]["x_m"]),
                                                               float(trajectory[index]["y_m"]) -
                                                               float(trajectory[index - 1]["y_m"]))
                                                    for index in range(1, len(trajectory)))) < 1e-10,
                                                "final s does not match processed polyline"))
  check("source timing discarded", lambda: require(reference["new_duration_s"] != source_summary["duration_s"] and
                                                      source_summary["source_timestamps_used_for_reference_timing"] is False,
                                                      "source timing was reused"))
  event_index = int(source_summary["suspicious_largest_step"]["source_index"])
  check("large-step sample preserved", lambda: require(any(int(row["source_index"]) == event_index for row in cleaned),
                                                         "large-step sample silently deleted"))
  check("projection diagnostic finite", lambda: require(math.isfinite(float(
    reference["projection_diagnostic"]["maximum_abs_progress_mismatch_m"])), "projection diagnostic invalid"))

  segments = result["segments"]
  def deterministic_segmentation() -> None:
    repeated, repeated_intervals = segment_maneuvers(source, config)
    stable_fields = ("segment_id", "mode", "source_index_start", "source_index_end",
                     "source_relative_time_start_s", "source_relative_time_end_s",
                     "source_polyline_length_m", "recorded_yaw_change_rad")
    require([[segment[key] for key in stable_fields] for segment in repeated] ==
            [[segment[key] for key in stable_fields] for segment in segments],
            "maneuver segmentation is not deterministic")
    require(repeated_intervals == result["motion_intervals"], "motion intervals changed")
  check("deterministic maneuver segmentation", deterministic_segmentation)
  check("known reverse intervals", lambda: require(sum(
    int(segment["moving_reverse_interval_count"]) for segment in segments) == 39,
    "known reverse moving intervals not preserved"))
  check("known pivot segments", lambda: require(
    any(segment["mode"] == "PIVOT" for segment in segments), "no pivots detected"))
  check("source maneuver order", lambda: require(all(
    int(segments[index]["source_index_start"]) == int(segments[index - 1]["source_index_end"])
    for index in range(1, len(segments))), "maneuver source order has a gap or reorder"))

  def stencil_boundaries() -> None:
    by_segment = {int(segment["segment_id"]): [row for row in trajectory
                  if int(row["segment_id"]) == int(segment["segment_id"])] for segment in segments}
    for segment in segments:
      if segment["mode"] == "PIVOT":
        continue
      rows = by_segment[int(segment["segment_id"])]
      local_length = max(float(row["geometry_stencil_s_max_m"]) for row in rows)
      for row in rows:
        require(-1e-12 <= float(row["geometry_stencil_s_min_m"]) <=
                float(row["geometry_stencil_s_max_m"]) <= local_length + 1e-12,
                "curvature stencil escaped its translational segment")
  check("geometry stencils remain inside maneuvers", stencil_boundaries)

  def body_yaw_alignment() -> None:
    errors = {1: [], -1: []}
    for index in range(len(trajectory) - 1):
      row, following = trajectory[index], trajectory[index + 1]
      direction = int(row["motion_direction"])
      if direction == 0 or int(row["segment_id"]) != int(following["segment_id"]):
        continue
      dx = float(following["x_m"]) - float(row["x_m"])
      dy = float(following["y_m"]) - float(row["y_m"])
      if math.hypot(dx, dy) <= 1e-12:
        continue
      difference = wrap_to_pi(math.atan2(dy, dx) - float(row["yaw_rad"]))
      errors[direction].append(abs(difference) if direction > 0 else abs(math.pi - abs(difference)))
    require(errors[1] and errors[-1], "missing generated forward/reverse chords")
    require(sorted(errors[1])[int(.95 * (len(errors[1]) - 1))] < .20,
            "forward body yaw is not aligned with its path")
    require(sorted(errors[-1])[int(.95 * (len(errors[-1]) - 1))] < .20,
            "reverse body yaw is not opposed to its path")
  check("forward/reverse body-yaw convention", body_yaw_alignment)

  check("signed translational speeds", lambda: require(
    any(int(row["motion_direction"]) == 1 and float(row["v_ref_m_s"]) > 0.0 for row in trajectory) and
    any(int(row["motion_direction"]) == -1 and float(row["v_ref_m_s"]) < 0.0 for row in trajectory),
    "signed translational interiors missing"))

  def boundaries() -> None:
    changes = [index for index in range(1, len(trajectory))
               if int(trajectory[index]["segment_id"]) != int(trajectory[index - 1]["segment_id"])]
    require(changes, "no maneuver boundaries")
    for index in changes:
      before, after = trajectory[index - 1], trajectory[index]
      require(abs(float(before["v_ref_m_s"])) < 1e-12, f"boundary {index} lacks zero speed")
      require(float(after["t_s"]) > float(before["t_s"]), f"boundary {index} time discontinuity")
      require(math.hypot(float(after["x_m"]) - float(before["x_m"]),
                         float(after["y_m"]) - float(before["y_m"])) < .03,
              f"boundary {index} XY discontinuity")
      require(abs(float(after["yaw_rad"]) - float(before["yaw_rad"])) < math.pi,
              f"boundary {index} contains a yaw branch discontinuity")
      require(math.isfinite(float(after["omega_ref_rad_s"])), f"boundary {index} omega invalid")
    for index in range(len(trajectory) - 1):
      first, second = float(trajectory[index]["v_ref_m_s"]), float(trajectory[index + 1]["v_ref_m_s"])
      require(not (first * second < 0.0), f"instantaneous signed-speed reversal at {index}")
  check("maneuver-boundary continuity", boundaries)

  def pivot_semantics() -> None:
    pivots = [row for row in trajectory if int(row["motion_direction"]) == 0]
    require(pivots and any(abs(float(row["omega_ref_rad_s"])) > 1e-6 for row in pivots),
            "pivots lack independent angular motion")
    require(all(int(row["curvature_valid"]) == 0 and float(row["curvature_ref_1_m"]) == 0.0
                for row in pivots), "pivot curvature marked valid")
    translations = [row for row in trajectory if int(row["motion_direction"]) != 0]
    require(all(int(row["curvature_valid"]) == 1 for row in translations),
            "translational curvature marked invalid")
  check("pivot and curvature-valid semantics", pivot_semantics)

  check("high-motion region represented", lambda: require(any(
    float(row.get("source_progress_index", -1.0)) <= event_index <=
    float(row.get("source_progress_index", -1.0)) + 1.0
    for row in trajectory), "known high-motion source region absent"))
  print(f"result: {len(tests)}/{len(tests)} checks passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
