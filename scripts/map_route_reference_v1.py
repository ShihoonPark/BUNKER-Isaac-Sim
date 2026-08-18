#!/usr/bin/env python3
"""Reusable Stage-D GLIM map-route preprocessing and trajectory generation."""

from __future__ import annotations

import csv
import copy
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from reference_trajectory_v1 import (  # noqa: E402
  _speed_profile, load_config as load_reference_config, parameterize_trajectory,
)
from tracking_controller_v1 import project_to_polyline, wrap_to_pi  # noqa: E402


class MapRouteError(ValueError):
  """Raised when source or processed route data is invalid."""


def load_config(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    config = json.load(stream)
  if config["source"]["trajectory_format"] != "TUM: timestamp x y z qx qy qz qw":
    raise MapRouteError("unsupported trajectory format")
  if not config["processing"]["use_xy_only"] or not config["processing"]["preserve_traversal_order"]:
    raise MapRouteError("Stage D requires ordered XY processing")
  if config["processing"]["artificial_loop_closure"]:
    raise MapRouteError("artificial loop closure is prohibited")
  smoothing = config["smoothing"]
  if (smoothing["method"] != "one_pass_binomial_5point" or smoothing["wrap_closed_path"] or
      smoothing["weights"] != [1, 4, 6, 4, 1] or smoothing["normalization"] != 16):
    raise MapRouteError("Stage D smoothing configuration changed")
  for section, key in (("cleaning", "minimum_retained_xy_spacing_m"),
                       ("resampling", "spacing_m")):
    value = float(config[section][key])
    if not math.isfinite(value) or value <= 0.0:
      raise MapRouteError(f"{section}.{key} must be positive and finite")
  segmentation = config["maneuver_segmentation"]
  pivot = segmentation["pivot"]
  if (float(segmentation["moving_interval_min_xy_m"]) <= 0.0 or
      float(pivot["max_per_interval_xy_m"]) <= 0.0 or
      int(pivot["minimum_consecutive_intervals"]) < 1 or
      float(pivot["minimum_abs_net_yaw_rad"]) <= 0.0):
    raise MapRouteError("invalid maneuver-segmentation assumptions")
  if (config["translational_geometry"]["curvature_method"] !=
      "fixed_arc_length_chord_and_three_point_circumcircle" or
      float(config["translational_geometry"]["half_span_m"]) <= 0.0):
    raise MapRouteError("invalid translational geometry estimator")
  return config


def quaternion_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
  return math.atan2(2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz))


def parse_tum(path: Path, quaternion_norm_tolerance: float) -> list[dict[str, float | int]]:
  rows = []
  with path.open(encoding="utf-8") as stream:
    for line_number, raw_line in enumerate(stream, 1):
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      fields = line.split()
      if len(fields) != 8:
        raise MapRouteError(f"{path}:{line_number}: expected exactly 8 fields, got {len(fields)}")
      try:
        values = [float(field) for field in fields]
      except ValueError as error:
        raise MapRouteError(f"{path}:{line_number}: non-numeric field") from error
      if not all(math.isfinite(value) for value in values):
        raise MapRouteError(f"{path}:{line_number}: non-finite value")
      timestamp, x, y, z, qx, qy, qz, qw = values
      if rows and timestamp <= float(rows[-1]["source_timestamp_s"]):
        raise MapRouteError(f"{path}:{line_number}: timestamps are not strictly increasing")
      norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
      if abs(norm - 1.0) > quaternion_norm_tolerance:
        raise MapRouteError(f"{path}:{line_number}: quaternion norm {norm} is unreasonable")
      rows.append({"source_index": len(rows), "source_timestamp_s": timestamp,
                   "x_m": x, "y_m": y, "z_m": z, "qx": qx, "qy": qy,
                   "qz": qz, "qw": qw, "recorded_yaw_rad": quaternion_yaw(qx, qy, qz, qw),
                   "quaternion_norm": norm})
  if len(rows) < 2:
    raise MapRouteError("TUM source has fewer than two poses")
  return rows


def _distances(rows: list[dict[str, Any]]) -> list[float]:
  return [math.hypot(float(rows[index]["x_m"]) - float(rows[index - 1]["x_m"]),
                     float(rows[index]["y_m"]) - float(rows[index - 1]["y_m"]))
          for index in range(1, len(rows))]


def _percentile(values: list[float], fraction: float) -> float:
  ordered = sorted(values)
  position = fraction * (len(ordered) - 1)
  lower = math.floor(position); upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _unwrap(values: Iterable[float]) -> list[float]:
  iterator = iter(values)
  first = next(iterator)
  result = [first]
  for value in iterator:
    result.append(result[-1] + wrap_to_pi(value - result[-1]))
  return result


def source_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
  spacing = _distances(rows)
  spacing_3d = [math.sqrt(
    (float(rows[index]["x_m"]) - float(rows[index - 1]["x_m"])) ** 2 +
    (float(rows[index]["y_m"]) - float(rows[index - 1]["y_m"])) ** 2 +
    (float(rows[index]["z_m"]) - float(rows[index - 1]["z_m"])) ** 2)
    for index in range(1, len(rows))]
  timestamps = [float(row["source_timestamp_s"]) for row in rows]
  time_steps = [timestamps[index] - timestamps[index - 1] for index in range(1, len(rows))]
  yaws = _unwrap(float(row["recorded_yaw_rad"]) for row in rows)
  event_index = 1 + max(range(len(spacing)), key=spacing.__getitem__)
  neighborhood = []
  for index in range(max(0, event_index - 2), min(len(rows), event_index + 2)):
    full_rotation = None
    yaw_change = None
    step = None
    if index:
      previous, current = rows[index - 1], rows[index]
      dot = abs(sum(float(previous[key]) * float(current[key]) for key in ("qx", "qy", "qz", "qw")))
      full_rotation = 2.0 * math.acos(min(1.0, max(-1.0, dot)))
      yaw_change = wrap_to_pi(yaws[index] - yaws[index - 1])
      step = spacing[index - 1]
    neighborhood.append({"source_index": index,
                         "relative_time_s": timestamps[index] - timestamps[0],
                         "timestamp_s": timestamps[index], "x_m": rows[index]["x_m"],
                         "y_m": rows[index]["y_m"], "xy_step_from_previous_m": step,
                         "recorded_yaw_change_rad": yaw_change,
                         "full_quaternion_rotation_rad": full_rotation})
  neighbor_steps = spacing[max(0, event_index - 3):min(len(spacing), event_index + 2)]
  continuity = "continuous high-motion region" if sum(
    value >= 0.5 * spacing[event_index - 1] for value in neighbor_steps) >= 3 else "uncertain"
  return {
    "pose_count": len(rows), "duration_s": timestamps[-1] - timestamps[0],
    "start_xyz_m": [rows[0][key] for key in ("x_m", "y_m", "z_m")],
    "end_xyz_m": [rows[-1][key] for key in ("x_m", "y_m", "z_m")],
    "raw_xy_polyline_length_m": sum(spacing),
    "raw_3d_polyline_length_m": sum(spacing_3d),
    "source_z_range_m": {"minimum": min(float(row["z_m"]) for row in rows),
                         "maximum": max(float(row["z_m"]) for row in rows)},
    "start_end_xy_distance_m": math.hypot(float(rows[-1]["x_m"]) - float(rows[0]["x_m"]),
                                            float(rows[-1]["y_m"]) - float(rows[0]["y_m"])),
    "source_dt_s": {"minimum": min(time_steps), "median": statistics.median(time_steps),
                    "maximum": max(time_steps)},
    "source_xy_spacing_m": {"minimum": min(spacing), "median": statistics.median(spacing),
                            "maximum": max(spacing),
                            "below_0p001_count": sum(value < .001 for value in spacing),
                            "below_0p005_count": sum(value < .005 for value in spacing),
                            "below_0p010_count": sum(value < .010 for value in spacing)},
    "recorded_yaw": {"start_rad": yaws[0], "end_rad": yaws[-1],
                     "end_wrapped_rad": wrap_to_pi(yaws[-1]),
                     "wrapped_difference_rad": wrap_to_pi(yaws[-1] - yaws[0])},
    "suspicious_largest_step": {"source_index": event_index,
                                "relative_time_s": timestamps[event_index] - timestamps[0],
                                "classification": continuity,
                                "neighborhood": neighborhood}}


def greedy_spatial_thinning(rows: list[dict[str, Any]], minimum_spacing: float
                            ) -> list[dict[str, Any]]:
  retained = [dict(rows[0])]
  for row in rows[1:-1]:
    if math.hypot(float(row["x_m"]) - float(retained[-1]["x_m"]),
                  float(row["y_m"]) - float(retained[-1]["y_m"])) >= minimum_spacing:
      retained.append(dict(row))
  if int(retained[-1]["source_index"]) != int(rows[-1]["source_index"]):
    retained.append(dict(rows[-1]))
  return retained


def cumulative_arc_length(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  result = []
  cumulative = 0.0
  for index, source in enumerate(rows):
    row = dict(source)
    if index:
      cumulative += math.hypot(float(row["x_m"]) - float(result[-1]["x_m"]),
                               float(row["y_m"]) - float(result[-1]["y_m"]))
      if cumulative <= float(result[-1]["s_m"]):
        raise MapRouteError("polyline contains a duplicate XY point")
    row["s_m"] = cumulative
    result.append(row)
  return result


def resample_polyline(rows: list[dict[str, Any]], spacing: float) -> list[dict[str, Any]]:
  source = cumulative_arc_length(rows)
  total = float(source[-1]["s_m"])
  targets = [index * spacing for index in range(math.floor(total / spacing) + 1)]
  if total - targets[-1] > 1e-12:
    targets.append(total)
  else:
    targets[-1] = total
  source_yaw = _unwrap(float(row["recorded_yaw_rad"]) for row in source)
  result = []
  cursor = 0
  for target in targets:
    while cursor + 1 < len(source) and float(source[cursor + 1]["s_m"]) < target:
      cursor += 1
    left, right = source[cursor], source[min(cursor + 1, len(source) - 1)]
    span = float(right["s_m"]) - float(left["s_m"])
    fraction = 0.0 if span == 0.0 else (target - float(left["s_m"])) / span
    result.append({"x_m": float(left["x_m"]) + fraction * (float(right["x_m"]) - float(left["x_m"])),
                   "y_m": float(left["y_m"]) + fraction * (float(right["y_m"]) - float(left["y_m"])),
                   "s_m": target,
                   "source_progress_index": float(left["source_index"]) + fraction * (
                     float(right["source_index"]) - float(left["source_index"])),
                   "source_timestamp_s": float(left["source_timestamp_s"]) + fraction * (
                     float(right["source_timestamp_s"]) - float(left["source_timestamp_s"])),
                   "recorded_yaw_unwrapped_rad": source_yaw[cursor] + fraction * (
                     source_yaw[min(cursor + 1, len(source_yaw) - 1)] - source_yaw[cursor])})
  return result


def smooth_open_route(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  if len(rows) <= 2:
    return [dict(row) for row in rows]
  result = [dict(row) for row in rows]
  for index in range(1, len(rows) - 1):
    if index in (1, len(rows) - 2):
      indices, weights, normalization = (index - 1, index, index + 1), (1, 2, 1), 4.0
    else:
      indices, weights, normalization = range(index - 2, index + 3), (1, 4, 6, 4, 1), 16.0
    result[index]["x_m"] = sum(weights[offset] * float(rows[source]["x_m"])
                                for offset, source in enumerate(indices)) / normalization
    result[index]["y_m"] = sum(weights[offset] * float(rows[source]["y_m"])
                                for offset, source in enumerate(indices)) / normalization
  return result


def smoothing_diagnostics(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, float]:
  displacement = [math.hypot(float(a["x_m"]) - float(b["x_m"]),
                             float(a["y_m"]) - float(b["y_m"])) for a, b in zip(before, after)]
  before_length = sum(_distances(before)); after_length = sum(_distances(after))
  change = after_length - before_length
  return {"rms_point_displacement_m": math.sqrt(sum(value * value for value in displacement) / len(displacement)),
          "mean_point_displacement_m": sum(displacement) / len(displacement),
          "maximum_point_displacement_m": max(displacement),
          "pre_smoothing_path_length_m": before_length,
          "smoothed_path_length_m": after_length,
          "path_length_change_m": change,
          "path_length_change_percent": 100.0 * change / before_length,
          "start_displacement_m": displacement[0], "end_displacement_m": displacement[-1]}


def normalize_start_frame(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
  x0, y0 = float(rows[0]["x_m"]), float(rows[0]["y_m"])
  forward = next((row for row in rows[1:]
                  if math.hypot(float(row["x_m"]) - x0, float(row["y_m"]) - y0) > 1e-12), None)
  if forward is None:
    raise MapRouteError("processed route has no initial forward tangent")
  initial_tangent = math.atan2(float(forward["y_m"]) - y0, float(forward["x_m"]) - x0)
  cosine, sine = math.cos(initial_tangent), math.sin(initial_tangent)
  result = []
  for source in rows:
    dx, dy = float(source["x_m"]) - x0, float(source["y_m"]) - y0
    row = dict(source)
    row["x_m"] = cosine * dx + sine * dy
    row["y_m"] = -sine * dx + cosine * dy
    row["recorded_yaw_local_rad"] = float(source["recorded_yaw_unwrapped_rad"]) - initial_tangent
    result.append(row)
  return result, {"map_origin_x_m": x0, "map_origin_y_m": y0,
                  "map_initial_geometric_tangent_yaw_rad": initial_tangent,
                  "map_to_local_rotation_rad": -initial_tangent}


def geometric_heading_and_curvature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  with_s = cumulative_arc_length(rows)
  headings = []
  for index in range(len(with_s)):
    before, after = max(0, index - 1), min(len(with_s) - 1, index + 1)
    headings.append(math.atan2(float(with_s[after]["y_m"]) - float(with_s[before]["y_m"]),
                               float(with_s[after]["x_m"]) - float(with_s[before]["x_m"])))
  headings = _unwrap(headings)
  result = []
  for index, source in enumerate(with_s):
    before, after = max(0, index - 1), min(len(with_s) - 1, index + 1)
    ds = float(with_s[after]["s_m"]) - float(with_s[before]["s_m"])
    curvature = (headings[after] - headings[before]) / ds
    row = dict(source)
    row.update({"index": index, "yaw_rad": headings[index],
                "curvature_ref_1_m": curvature, "curvature_numeric_1_m": curvature,
                "segment_index": 0, "segment_type": "recorded_map_route"})
    result.append(row)
  return result


def curvature_statistics(rows: list[dict[str, Any]]) -> dict[str, float]:
  absolute = [abs(float(row["curvature_ref_1_m"])) for row in rows]
  changes = [abs(float(rows[index]["curvature_ref_1_m"]) - float(rows[index - 1]["curvature_ref_1_m"]))
             for index in range(1, len(rows))]
  return {"maximum_abs_1_m": max(absolute), "median_abs_1_m": statistics.median(absolute),
          "p95_abs_1_m": _percentile(absolute, .95), "p99_abs_1_m": _percentile(absolute, .99),
          "maximum_discrete_change_1_m": max(changes)}


def projection_diagnostic(reference: list[dict[str, Any]]) -> dict[str, Any]:
  projected = [project_to_polyline(float(row["x_m"]), float(row["y_m"]), reference)["s_projected_m"]
               for row in reference]
  decreases = [(index, projected[index] - projected[index - 1]) for index in range(1, len(projected))
               if projected[index] < projected[index - 1] - 1e-8]
  errors = [projected[index] - float(reference[index]["s_m"]) for index in range(len(reference))]
  endpoint_distance = math.hypot(float(reference[-1]["x_m"]) - float(reference[0]["x_m"]),
                                 float(reference[-1]["y_m"]) - float(reference[0]["y_m"]))
  nominal_spacing = statistics.median(
    float(reference[index]["s_m"]) - float(reference[index - 1]["s_m"])
    for index in range(1, len(reference)))
  return {"nonmonotonic_jump_count": len(decreases),
          "largest_backward_jump_m": min((value for _, value in decreases), default=0.0),
          "maximum_abs_progress_mismatch_m": max(abs(value) for value in errors),
          "final_projected_s_m": projected[-1],
          "requires_progress_aware_projection_for_exact_reference_points": bool(decreases),
          "near_closure_endpoint_distance_m": endpoint_distance,
          "near_closure_distance_in_nominal_sample_intervals": endpoint_distance / nominal_spacing,
          "stage_e_progress_aware_projection_recommended": True,
          "stage_e_note": "Exact reference-point projections are monotonic, but the open endpoint returns only about five samples from the start. A noisy actual pose can select the wrong traversal branch, so Stage E should use progress-aware projection."}


def source_motion_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Resolve ordered source displacements in the midpoint recorded-yaw frame."""
  yaw = _unwrap(float(row["recorded_yaw_rad"]) for row in rows)
  intervals = []
  for end_index in range(1, len(rows)):
    dx = float(rows[end_index]["x_m"]) - float(rows[end_index - 1]["x_m"])
    dy = float(rows[end_index]["y_m"]) - float(rows[end_index - 1]["y_m"])
    ds = math.hypot(dx, dy)
    midpoint_yaw = 0.5 * (yaw[end_index - 1] + yaw[end_index])
    motion_heading = math.atan2(dy, dx) if ds > 0.0 else midpoint_yaw
    intervals.append({"interval_index": end_index - 1, "source_start_index": end_index - 1,
                      "source_end_index": end_index, "dx_m": dx, "dy_m": dy, "ds_xy_m": ds,
                      "midpoint_recorded_yaw_rad": midpoint_yaw,
                      "recorded_yaw_change_rad": yaw[end_index] - yaw[end_index - 1],
                      "motion_heading_rad": motion_heading,
                      "motion_heading_minus_yaw_rad": wrap_to_pi(motion_heading - midpoint_yaw),
                      "d_forward_m": math.cos(midpoint_yaw) * dx + math.sin(midpoint_yaw) * dy,
                      "d_lateral_m": -math.sin(midpoint_yaw) * dx + math.cos(midpoint_yaw) * dy})
  return intervals


def segment_maneuvers(rows: list[dict[str, Any]], config: dict[str, Any]
                      ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Detect pivots first, then preserve every contiguous forward/reverse run."""
  intervals = source_motion_intervals(rows)
  segmentation = config["maneuver_segmentation"]
  pivot_cfg = segmentation["pivot"]
  stationary_limit = float(pivot_cfg["max_per_interval_xy_m"])
  minimum_count = int(pivot_cfg["minimum_consecutive_intervals"])
  minimum_yaw = float(pivot_cfg["minimum_abs_net_yaw_rad"])
  moving_limit = float(segmentation["moving_interval_min_xy_m"])
  pivot_ranges: list[tuple[int, int]] = []
  start = None
  for index, interval in enumerate(intervals + [{"ds_xy_m": stationary_limit}]):
    stationary = index < len(intervals) and float(interval["ds_xy_m"]) < stationary_limit
    if stationary and start is None:
      start = index
    if not stationary and start is not None:
      end = index - 1
      net_yaw = sum(float(intervals[item]["recorded_yaw_change_rad"])
                    for item in range(start, end + 1))
      if end - start + 1 >= minimum_count and abs(net_yaw) >= minimum_yaw:
        pivot_ranges.append((start, end))
      start = None
  modes: list[str | None] = [None] * len(intervals)
  for start, end in pivot_ranges:
    for index in range(start, end + 1):
      modes[index] = "PIVOT"
  for index, interval in enumerate(intervals):
    if modes[index] is None and float(interval["ds_xy_m"]) >= moving_limit:
      modes[index] = "FORWARD" if float(interval["d_forward_m"]) >= 0.0 else "REVERSE"
  # Non-pivot sub-threshold jitter inherits its nearest translational neighbor.
  # Pivot ranges remain immutable. A tie deterministically favors the previous mode.
  for index, mode in enumerate(modes):
    if mode is not None:
      continue
    previous = next((modes[item] for item in range(index - 1, -1, -1)
                     if modes[item] in ("FORWARD", "REVERSE")), None)
    following = next((modes[item] for item in range(index + 1, len(modes))
                      if modes[item] in ("FORWARD", "REVERSE")), None)
    modes[index] = previous or following or "FORWARD"
  segments = []
  segment_start = 0
  for index in range(1, len(modes) + 1):
    if index == len(modes) or modes[index] != modes[segment_start]:
      interval_start, interval_end = segment_start, index - 1
      source_start, source_end = interval_start, interval_end + 1
      segment_intervals = intervals[interval_start:index]
      yaw_unwrapped = _unwrap(float(row["recorded_yaw_rad"]) for row in rows[source_start:source_end + 1])
      segments.append({
        "segment_id": len(segments), "mode": modes[segment_start],
        "source_index_start": source_start, "source_index_end": source_end,
        "source_relative_time_start_s": float(rows[source_start]["source_timestamp_s"]) - float(rows[0]["source_timestamp_s"]),
        "source_relative_time_end_s": float(rows[source_end]["source_timestamp_s"]) - float(rows[0]["source_timestamp_s"]),
        "source_xy_start_m": [rows[source_start]["x_m"], rows[source_start]["y_m"]],
        "source_xy_end_m": [rows[source_end]["x_m"], rows[source_end]["y_m"]],
        "source_polyline_length_m": sum(float(item["ds_xy_m"]) for item in segment_intervals),
        "net_xy_displacement_m": math.hypot(float(rows[source_end]["x_m"]) - float(rows[source_start]["x_m"]),
                                             float(rows[source_end]["y_m"]) - float(rows[source_start]["y_m"])),
        "recorded_yaw_start_rad": yaw_unwrapped[0], "recorded_yaw_end_rad": yaw_unwrapped[-1],
        "recorded_yaw_change_rad": yaw_unwrapped[-1] - yaw_unwrapped[0],
        "positive_d_forward_accumulation_m": sum(max(0.0, float(item["d_forward_m"])) for item in segment_intervals),
        "negative_d_forward_magnitude_m": sum(max(0.0, -float(item["d_forward_m"])) for item in segment_intervals),
        "moving_forward_interval_count": sum(float(item["ds_xy_m"]) >= moving_limit and float(item["d_forward_m"]) >= 0.0 for item in segment_intervals),
        "moving_reverse_interval_count": sum(float(item["ds_xy_m"]) >= moving_limit and float(item["d_forward_m"]) < 0.0 for item in segment_intervals),
      })
      segment_start = index
  return segments, intervals


def _ensure_minimum_resampled_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Insert a deterministic midpoint so a short zero-boundary-speed run is feasible."""
  if len(rows) != 2:
    return rows
  first, last = rows
  midpoint = {}
  for key in first:
    if key in ("x_m", "y_m", "s_m", "source_progress_index", "source_timestamp_s",
               "recorded_yaw_unwrapped_rad"):
      midpoint[key] = 0.5 * (float(first[key]) + float(last[key]))
    else:
      midpoint[key] = first[key]
  return [first, midpoint, last]


def _interpolate_xy(rows: list[dict[str, Any]], progress: float) -> tuple[float, float]:
  progress = max(0.0, min(float(rows[-1]["s_m"]), progress))
  index = 0
  while index + 1 < len(rows) and float(rows[index + 1]["s_m"]) < progress:
    index += 1
  left, right = rows[index], rows[min(index + 1, len(rows) - 1)]
  span = float(right["s_m"]) - float(left["s_m"])
  fraction = 0.0 if span <= 0.0 else (progress - float(left["s_m"])) / span
  return (float(left["x_m"]) + fraction * (float(right["x_m"]) - float(left["x_m"])),
          float(left["y_m"]) + fraction * (float(right["y_m"]) - float(left["y_m"])))


def _fixed_scale_geometry(rows: list[dict[str, Any]], half_span: float,
                          direction: int) -> list[dict[str, Any]]:
  """Fixed-scale chord tangent and signed three-point circumcircle curvature.

  For points A, B, C sampled at a fixed available spatial scale, curvature is
  2*cross(B-A,C-A)/(|A-B| |B-C| |C-A|). Degenerate triples return zero.
  """
  rows = cumulative_arc_length(rows)
  total = float(rows[-1]["s_m"])
  recorded = _unwrap(float(row["recorded_yaw_unwrapped_rad"]) for row in rows)
  result = []
  previous_body_yaw = None
  for index, source in enumerate(rows):
    s_value = float(source["s_m"])
    if total <= 0.0:
      raise MapRouteError("zero-length translational maneuver")
    if s_value < half_span:
      step = min(half_span, 0.5 * total)
      positions = (0.0, step, min(total, 2.0 * step))
    elif total - s_value < half_span:
      step = min(half_span, 0.5 * total)
      positions = (max(0.0, total - 2.0 * step), total - step, total)
    else:
      positions = (s_value - half_span, s_value, s_value + half_span)
    a, b, c = (_interpolate_xy(rows, value) for value in positions)
    tangent = math.atan2(c[1] - a[1], c[0] - a[0])
    candidate = tangent if direction > 0 else tangent + math.pi
    guide = recorded[index]
    if previous_body_yaw is None:
      candidate += round((guide - candidate) / (2.0 * math.pi)) * 2.0 * math.pi
    else:
      candidate = previous_body_yaw + wrap_to_pi(candidate - previous_body_yaw)
    previous_body_yaw = candidate
    ab, bc, ca = math.hypot(b[0] - a[0], b[1] - a[1]), math.hypot(c[0] - b[0], c[1] - b[1]), math.hypot(a[0] - c[0], a[1] - c[1])
    denominator = ab * bc * ca
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    curvature = 0.0 if denominator <= 1e-12 else 2.0 * cross / denominator
    row = dict(source)
    row.update({"yaw_rad": candidate, "curvature_ref_1_m": curvature,
                "curvature_numeric_1_m": curvature, "curvature_valid": 1,
                "motion_direction": direction, "geometry_stencil_s_min_m": positions[0],
                "geometry_stencil_s_max_m": positions[2]})
    result.append(row)
  return result


def _signed_pointwise_limits(rows: list[dict[str, Any]], canonical: dict[str, Any],
                             direction: int) -> None:
  limits = canonical["trajectory_limits"]
  epsilon = float(canonical["path"]["curvature_near_zero_epsilon_1_m"])
  body_max = float(limits["maximum_body_speed_m_s"])
  lateral_max = float(limits["maximum_lateral_acceleration_m_s2"])
  yaw_max = float(limits["maximum_yaw_rate_rad_s"])
  track_max = float(limits["maximum_track_surface_speed_m_s"])
  spacing = float(canonical["measured_fixed"]["track_center_distance_m"])
  priority = list(limits["limit_reason_priority"])
  tie_tolerance = float(limits["limit_tie_tolerance_m_s"])
  for row in rows:
    curvature = float(row["curvature_ref_1_m"]); absolute = abs(curvature)
    candidates = {
      "global": body_max,
      "curvature": body_max if absolute < epsilon else math.sqrt(lateral_max / absolute),
      "yaw_rate": body_max if absolute < epsilon else yaw_max / absolute,
      "track_speed": track_max / max(abs(direction - .5 * spacing * curvature),
                                      abs(direction + .5 * spacing * curvature)),
    }
    limit = min(candidates.values())
    reason = next(name for name in priority if candidates[name] <= limit + tie_tolerance)
    row.update({"v_limit_global_m_s": body_max,
                "v_limit_curvature_m_s": candidates["curvature"],
                "v_limit_yaw_rate_m_s": candidates["yaw_rate"],
                "v_limit_track_m_s": candidates["track_speed"],
                "v_limit_m_s": limit, "speed_limit_reason": reason})


def build_translational_reference(segment: dict[str, Any], source: list[dict[str, Any]],
                                  config: dict[str, Any], canonical: dict[str, Any]
                                  ) -> list[dict[str, Any]]:
  direction = 1 if segment["mode"] == "FORWARD" else -1
  subset = source[int(segment["source_index_start"]):int(segment["source_index_end"]) + 1]
  cleaned = greedy_spatial_thinning(subset, float(config["cleaning"]["minimum_retained_xy_spacing_m"]))
  uniform = _ensure_minimum_resampled_points(
    resample_polyline(cleaned, float(config["resampling"]["spacing_m"])))
  smoothed = smooth_open_route(uniform)
  geometry = _fixed_scale_geometry(
    smoothed, float(config["translational_geometry"]["half_span_m"]), direction)
  for row in geometry:
    row.update({"segment_id": int(segment["segment_id"]), "segment_type": segment["mode"].lower(),
                "segment_index": int(segment["segment_id"])})
  _signed_pointwise_limits(geometry, canonical, direction)
  profile_config = canonical
  if direction < 0:
    profile_config = copy.deepcopy(canonical)
    limits = profile_config["trajectory_limits"]
    limits["maximum_acceleration_m_s2"], limits["maximum_deceleration_m_s2"] = (
      limits["maximum_deceleration_m_s2"], limits["maximum_acceleration_m_s2"])
  speeds = _speed_profile(geometry, profile_config)
  times = [0.0]
  for index in range(len(geometry) - 1):
    ds = float(geometry[index + 1]["s_m"]) - float(geometry[index]["s_m"])
    denominator = speeds[index] + speeds[index + 1]
    if denominator <= 1e-12:
      raise MapRouteError(f"zero speed over nonzero translational interval in segment {segment['segment_id']}")
    times.append(times[-1] + 2.0 * ds / denominator)
  rows = []
  spacing = float(canonical["measured_fixed"]["track_center_distance_m"])
  for index, source_row in enumerate(geometry):
    row = dict(source_row); u = speeds[index]; v = direction * u
    omega = u * float(row["curvature_ref_1_m"])
    row.update({"t_s": times[index], "dt_s": 0.0 if index == 0 else times[index] - times[index - 1],
                "u_ref_m_s": u, "v_ref_m_s": v, "omega_ref_rad_s": omega,
                "v_left_ref_m_s": v - .5 * spacing * omega,
                "v_right_ref_m_s": v + .5 * spacing * omega,
                "v_lateral_reference_m_s": 0.0,
                "motion_phase": "start" if index == 0 else ("stop" if index == len(geometry) - 1 else segment["mode"].lower())})
    rows.append(row)
  for index in range(len(rows)):
    if index + 1 < len(rows):
      rows[index]["a_ref_m_s2"] = (float(rows[index + 1]["v_ref_m_s"]) - float(rows[index]["v_ref_m_s"])) / (
        float(rows[index + 1]["t_s"]) - float(rows[index]["t_s"]))
    else:
      rows[index]["a_ref_m_s2"] = rows[index - 1]["a_ref_m_s2"]
  return rows


def build_pivot_reference(segment: dict[str, Any], source: list[dict[str, Any]],
                          yaw_entry: float, yaw_exit_candidate: float,
                          canonical: dict[str, Any], sample_period: float = .02
                          ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  start = source[int(segment["source_index_start"])]
  end = source[int(segment["source_index_end"])]
  dx, dy = float(end["x_m"]) - float(start["x_m"]), float(end["y_m"]) - float(start["y_m"])
  distance = math.hypot(dx, dy)
  recorded_delta = float(segment["recorded_yaw_change_rad"])
  yaw_delta = yaw_exit_candidate - yaw_entry
  yaw_delta += round((recorded_delta - yaw_delta) / (2.0 * math.pi)) * 2.0 * math.pi
  yaw_exit = yaw_entry + yaw_delta
  limits = canonical["trajectory_limits"]
  body_max = float(limits["maximum_body_speed_m_s"]); yaw_max = float(limits["maximum_yaw_rate_rad_s"])
  track_max = float(limits["maximum_track_surface_speed_m_s"])
  accel = min(float(limits["maximum_acceleration_m_s2"]), float(limits["maximum_deceleration_m_s2"]))
  b = float(canonical["measured_fixed"]["track_center_distance_m"]); angle = abs(yaw_delta)
  duration = max(1.5 * distance / body_max, 1.5 * angle / yaw_max,
                 1.5 * (distance + .5 * b * angle) / track_max,
                 math.sqrt((6.0 * distance + 2.25 * distance * angle) / accel))
  steps = max(2, math.ceil(duration / sample_period)); duration = max(duration, steps * 1e-6)
  rows = []
  for index in range(steps + 1):
    r = index / steps; time = duration * r
    q = 3.0 * r * r - 2.0 * r * r * r
    qdot = (6.0 * r - 6.0 * r * r) / duration
    qddot = (6.0 - 12.0 * r) / (duration * duration)
    x, y, yaw = float(start["x_m"]) + q * dx, float(start["y_m"]) + q * dy, yaw_entry + q * yaw_delta
    vx, vy, omega = qdot * dx, qdot * dy, qdot * yaw_delta
    v = math.cos(yaw) * vx + math.sin(yaw) * vy
    lateral = -math.sin(yaw) * vx + math.cos(yaw) * vy
    ax, ay = qddot * dx, qddot * dy
    acceleration = math.cos(yaw) * ax + math.sin(yaw) * ay + omega * lateral
    rows.append({"segment_id": int(segment["segment_id"]), "segment_index": int(segment["segment_id"]),
                 "segment_type": "pivot", "motion_direction": 0, "curvature_valid": 0,
                 "x_m": x, "y_m": y, "yaw_rad": yaw, "curvature_ref_1_m": 0.0,
                 "curvature_numeric_1_m": 0.0, "t_s": time,
                 "dt_s": 0.0 if index == 0 else duration / steps,
                 "u_ref_m_s": math.hypot(vx, vy), "v_ref_m_s": v,
                 "v_lateral_reference_m_s": lateral, "omega_ref_rad_s": omega,
                 "a_ref_m_s2": acceleration,
                 "v_left_ref_m_s": v - .5 * b * omega,
                 "v_right_ref_m_s": v + .5 * b * omega,
                 "v_limit_global_m_s": body_max, "v_limit_curvature_m_s": body_max,
                 "v_limit_yaw_rate_m_s": yaw_max, "v_limit_track_m_s": track_max,
                 "v_limit_m_s": body_max, "speed_limit_reason": "pivot",
                 "motion_phase": "pivot", "source_progress_index": float(segment["source_index_start"]) + q * (
                   float(segment["source_index_end"]) - float(segment["source_index_start"]))})
  report = {"duration_s": duration, "reference_yaw_change_rad": yaw_delta,
            "recorded_yaw_change_rad": recorded_delta,
            "yaw_change_difference_rad": yaw_delta - recorded_delta,
            "maximum_abs_v_ref_m_s": max(abs(float(row["v_ref_m_s"])) for row in rows),
            "maximum_abs_lateral_reference_m_s": max(abs(float(row["v_lateral_reference_m_s"])) for row in rows),
            "maximum_abs_omega_ref_rad_s": max(abs(float(row["omega_ref_rad_s"])) for row in rows),
            "maximum_abs_left_track_speed_m_s": max(abs(float(row["v_left_ref_m_s"])) for row in rows),
            "maximum_abs_right_track_speed_m_s": max(abs(float(row["v_right_ref_m_s"])) for row in rows)}
  return rows, report


def assemble_maneuver_reference(segments: list[dict[str, Any]], source: list[dict[str, Any]],
                                config: dict[str, Any], canonical: dict[str, Any]
                                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
  recorded_all = _unwrap(float(row["recorded_yaw_rad"]) for row in source)
  source_guided = [dict(row, recorded_yaw_rad=recorded_all[index])
                   for index, row in enumerate(source)]
  translational: dict[int, list[dict[str, Any]]] = {}
  for segment in segments:
    if segment["mode"] != "PIVOT":
      translational[int(segment["segment_id"])] = build_translational_reference(
        segment, source_guided, config, canonical)
  pieces = []
  pivot_reports = []
  for index, segment in enumerate(segments):
    segment_id = int(segment["segment_id"])
    if segment["mode"] != "PIVOT":
      pieces.append(translational[segment_id]); continue
    previous_rows = translational.get(segment_id - 1)
    next_rows = translational.get(segment_id + 1)
    yaw_entry = float(previous_rows[-1]["yaw_rad"]) if previous_rows else recorded_all[int(segment["source_index_start"])]
    yaw_exit = float(next_rows[0]["yaw_rad"]) if next_rows else recorded_all[int(segment["source_index_end"])]
    pivot_rows, report = build_pivot_reference(segment, source, yaw_entry, yaw_exit, canonical)
    segment.update({"pivot_reference": report}); pivot_reports.append({"segment_id": segment_id, **report})
    pieces.append(pivot_rows)
  combined = []
  time_offset = 0.0
  for piece_index, piece in enumerate(pieces):
    for row_index, source_row in enumerate(piece):
      if piece_index and row_index == 0:
        continue
      row = dict(source_row); row["t_s"] = time_offset + float(source_row["t_s"])
      combined.append(row)
    time_offset = float(combined[-1]["t_s"])
  # Rigidly normalize the assembled sensor-center route using the initial body yaw.
  x0, y0, yaw0 = float(combined[0]["x_m"]), float(combined[0]["y_m"]), float(combined[0]["yaw_rad"])
  cosine, sine = math.cos(yaw0), math.sin(yaw0)
  cumulative = 0.0
  for index, row in enumerate(combined):
    map_x, map_y = float(row["x_m"]), float(row["y_m"])
    dx, dy = map_x - x0, map_y - y0
    row["x_m"], row["y_m"] = cosine * dx + sine * dy, -sine * dx + cosine * dy
    row["yaw_rad"] = float(row["yaw_rad"]) - yaw0
    if index:
      cumulative += math.hypot(float(row["x_m"]) - float(combined[index - 1]["x_m"]),
                               float(row["y_m"]) - float(combined[index - 1]["y_m"]))
    row["s_m"] = cumulative; row["index"] = index
    row["dt_s"] = 0.0 if index == 0 else float(row["t_s"]) - float(combined[index - 1]["t_s"])
  # Stored point acceleration uses the independently meaningful signed time derivative.
  for index in range(len(combined)):
    if index + 1 < len(combined):
      combined[index]["a_ref_m_s2"] = (float(combined[index + 1]["v_ref_m_s"]) -
                                         float(combined[index]["v_ref_m_s"])) / float(combined[index + 1]["dt_s"])
    elif index:
      combined[index]["a_ref_m_s2"] = combined[index - 1]["a_ref_m_s2"]
  transform = {"map_origin_x_m": x0, "map_origin_y_m": y0,
               "map_initial_body_yaw_rad": yaw0,
               "map_to_local_rotation_rad": -yaw0}
  return combined, pivot_reports, transform


def maneuver_geometry_distortion(trajectory: list[dict[str, Any]], source: list[dict[str, Any]],
                                 transform: dict[str, Any]) -> dict[str, Any]:
  """Compare generated sensor-center XY with source XY at recorded-index progress."""
  cosine = math.cos(float(transform["map_initial_body_yaw_rad"]))
  sine = math.sin(float(transform["map_initial_body_yaw_rad"]))
  x0 = float(transform["map_origin_x_m"]); y0 = float(transform["map_origin_y_m"])
  displacements = []
  pivot_displacements = []
  for row in trajectory:
    progress = max(0.0, min(float(len(source) - 1), float(row["source_progress_index"])))
    left = int(math.floor(progress)); right = min(left + 1, len(source) - 1)
    fraction = progress - left
    map_x = float(source[left]["x_m"]) + fraction * (float(source[right]["x_m"]) - float(source[left]["x_m"]))
    map_y = float(source[left]["y_m"]) + fraction * (float(source[right]["y_m"]) - float(source[left]["y_m"]))
    dx, dy = map_x - x0, map_y - y0
    source_x, source_y = cosine * dx + sine * dy, -sine * dx + cosine * dy
    displacement = math.hypot(float(row["x_m"]) - source_x, float(row["y_m"]) - source_y)
    displacements.append(displacement)
    if int(row["motion_direction"]) == 0:
      pivot_displacements.append(displacement)
  raw_length = sum(_distances(source)); generated_length = float(trajectory[-1]["s_m"])
  return {
    "correspondence_definition": "linear source XY interpolation by source_progress_index, compared after the same rigid normalization",
    "rms_displacement_m": math.sqrt(sum(value * value for value in displacements) / len(displacements)),
    "mean_displacement_m": sum(displacements) / len(displacements),
    "maximum_displacement_m": max(displacements),
    "pivot_rms_simplification_displacement_m": math.sqrt(
      sum(value * value for value in pivot_displacements) / len(pivot_displacements)),
    "pivot_maximum_simplification_displacement_m": max(pivot_displacements),
    "raw_source_length_m": raw_length, "generated_length_m": generated_length,
    "length_change_m": generated_length - raw_length,
    "length_change_percent": 100.0 * (generated_length - raw_length) / raw_length}


def maneuver_summary(segments: list[dict[str, Any]], trajectory: list[dict[str, Any]],
                     pivot_reports: list[dict[str, Any]]) -> dict[str, Any]:
  counts = {mode: sum(segment["mode"] == mode for segment in segments)
            for mode in ("FORWARD", "REVERSE", "PIVOT")}
  lengths = {mode: sum(float(segment["source_polyline_length_m"]) for segment in segments
                       if segment["mode"] == mode) for mode in counts}
  duration = {mode: 0.0 for mode in counts}
  for index in range(len(trajectory) - 1):
    duration[str(trajectory[index]["segment_type"]).upper()] += float(trajectory[index + 1]["t_s"]) - float(trajectory[index]["t_s"])
  valid = [row for row in trajectory if int(row["curvature_valid"]) == 1]
  curvature = [abs(float(row["curvature_ref_1_m"])) for row in valid]
  changes = [abs(float(trajectory[index]["curvature_ref_1_m"]) - float(trajectory[index - 1]["curvature_ref_1_m"]))
             for index in range(1, len(trajectory))
             if int(trajectory[index]["curvature_valid"]) and int(trajectory[index - 1]["curvature_valid"]) and
             int(trajectory[index]["segment_id"]) == int(trajectory[index - 1]["segment_id"])]
  return {"segment_counts": counts, "source_spatial_length_by_mode_m": lengths,
          "generated_duration_by_mode_s": duration, "segments": segments,
          "pivot_reference_reports": pivot_reports,
          "valid_translational_curvature": {
            "maximum_abs_1_m": max(curvature), "median_abs_1_m": statistics.median(curvature),
            "p95_abs_1_m": _percentile(curvature, .95), "p99_abs_1_m": _percentile(curvature, .99),
            "maximum_discrete_change_within_same_segment_1_m": max(changes)}}


def build_map_route_reference(config_path: Path) -> dict[str, Any]:
  config = load_config(config_path)
  canonical_path = REPO_ROOT / config["dynamic_limits_source"]
  canonical = load_reference_config(canonical_path)
  source_path = Path(config["source"]["trajectory_path"])
  source = parse_tum(source_path, float(config["diagnostic_guards"]["quaternion_norm_tolerance"]))
  source_stats = source_diagnostics(source)
  cleaned = greedy_spatial_thinning(source, float(config["cleaning"]["minimum_retained_xy_spacing_m"]))
  uniform = resample_polyline(cleaned, float(config["resampling"]["spacing_m"]))
  smoothed = smooth_open_route(uniform)
  smoothing_stats = smoothing_diagnostics(uniform, smoothed)
  guards = config["diagnostic_guards"]
  guard_results = {
    "rms_displacement": smoothing_stats["rms_point_displacement_m"] <= float(guards["maximum_rms_smoothing_displacement_m"]),
    "maximum_displacement": smoothing_stats["maximum_point_displacement_m"] <= float(guards["maximum_smoothing_displacement_m"]),
    "path_length_change": abs(smoothing_stats["path_length_change_percent"]) <= float(guards["maximum_abs_path_length_change_percent"]),
  }
  if not all(guard_results.values()):
    raise MapRouteError(f"smoothing distortion guard failed: {guard_results}; diagnostics={smoothing_stats}")
  normalized, transform = normalize_start_frame(smoothed)
  geometric = geometric_heading_and_curvature(normalized)
  legacy_path_only_trajectory = parameterize_trajectory(geometric, canonical)
  segments, motion_intervals = segment_maneuvers(source, config)
  trajectory, pivot_reports, maneuver_transform = assemble_maneuver_reference(
    segments, source, config, canonical)
  maneuvers = maneuver_summary(segments, trajectory, pivot_reports)
  geometry_distortion = maneuver_geometry_distortion(trajectory, source, maneuver_transform)
  moving_limit = float(config["maneuver_segmentation"]["moving_interval_min_xy_m"])
  moving_intervals = [row for row in motion_intervals if float(row["ds_xy_m"]) >= moving_limit]
  source_direction = {
    "moving_interval_threshold_m": moving_limit,
    "moving_interval_count": len(moving_intervals),
    "forward_interval_count": sum(float(row["d_forward_m"]) >= 0.0 for row in moving_intervals),
    "reverse_interval_count": sum(float(row["d_forward_m"]) < 0.0 for row in moving_intervals),
    "positive_d_forward_accumulation_m": sum(max(0.0, float(row["d_forward_m"])) for row in moving_intervals),
    "negative_d_forward_magnitude_m": sum(max(0.0, -float(row["d_forward_m"])) for row in moving_intervals)}
  source_direction["forward_interval_fraction"] = source_direction["forward_interval_count"] / len(moving_intervals)
  source_direction["reverse_interval_fraction"] = source_direction["reverse_interval_count"] / len(moving_intervals)
  clean_length = sum(_distances(cleaned))
  raw_uniform = resample_polyline(source, float(config["resampling"]["spacing_m"]))
  raw_normalized, _ = normalize_start_frame(raw_uniform)
  raw_geometry = geometric_heading_and_curvature(raw_normalized)
  alignment = {"FORWARD": [], "REVERSE": []}
  for index in range(len(trajectory) - 1):
    row, following = trajectory[index], trajectory[index + 1]
    mode = str(row["segment_type"]).upper()
    if mode not in alignment or int(row["segment_id"]) != int(following["segment_id"]):
      continue
    dx, dy = float(following["x_m"]) - float(row["x_m"]), float(following["y_m"]) - float(row["y_m"])
    if math.hypot(dx, dy) > 1e-12:
      alignment[mode].append(wrap_to_pi(math.atan2(dy, dx) - float(row["yaw_rad"])))
  reasons = {name: 0 for name in canonical["trajectory_limits"]["limit_reason_priority"]}
  for row in trajectory:
    if str(row["speed_limit_reason"]) in reasons:
      reasons[str(row["speed_limit_reason"])] += 1
  raw_curvature = curvature_statistics(raw_geometry)
  summary = {
    "source_data": {**source_stats, **config["source"],
                    "source_timestamps_used_for_reference_timing": False},
    "processing": {
      "assumptions": {"use_xy_only": True, "preserve_traversal_order": True,
                      "artificial_loop_closure": False,
                      "minimum_retained_xy_spacing_m": config["cleaning"]["minimum_retained_xy_spacing_m"],
                      "resampling_spacing_m": config["resampling"]["spacing_m"],
                      "smoothing": config["smoothing"],
                      "maneuver_segmentation": config["maneuver_segmentation"],
                      "translational_geometry": config["translational_geometry"]},
      "cleaning": {"raw_count": len(source), "retained_count": len(cleaned),
                   "removed_count": len(source) - len(cleaned),
                   "raw_xy_length_m": source_stats["raw_xy_polyline_length_m"],
                   "cleaned_xy_length_m": clean_length,
                   "length_change_m": clean_length - source_stats["raw_xy_polyline_length_m"]},
      "resampling": {"point_count": len(uniform), "pre_smoothing_length_m": sum(_distances(uniform)),
                     "final_remainder_m": float(uniform[-1]["s_m"]) - float(uniform[-2]["s_m"])},
      "smoothing_distortion": {**smoothing_stats, "guards": guard_results},
      "normalization_transform": transform},
    "reference": {
      "point_count": len(trajectory), "processed_path_length_m": trajectory[-1]["s_m"],
      "new_duration_s": trajectory[-1]["t_s"], "original_rc_duration_s": source_stats["duration_s"],
      "normalized_start": {key: trajectory[0][key] for key in ("x_m", "y_m", "yaw_rad")},
      "normalized_endpoint": {key: trajectory[-1][key] for key in ("x_m", "y_m", "yaw_rad")},
      "processed_start_end_distance_m": math.hypot(float(trajectory[-1]["x_m"]), float(trajectory[-1]["y_m"])),
      "heading_minimum_rad": min(float(row["yaw_rad"]) for row in trajectory),
      "heading_maximum_rad": max(float(row["yaw_rad"]) for row in trajectory),
      "maneuvers": maneuvers,
      "source_moving_direction_diagnostic": source_direction,
      "maneuver_normalization_transform": maneuver_transform,
      "maneuver_geometry_distortion": geometry_distortion,
      "path_tangent_vs_body_yaw": {
        mode.lower(): {"sample_count": len(values),
                       "mean_abs_wrapped_difference_rad": sum(abs(value) for value in values) / len(values),
                       "maximum_abs_wrapped_difference_rad": max(abs(value) for value in values),
                       "mean_abs_error_from_expected_rad": sum(
                         abs(value) if mode == "FORWARD" else abs(math.pi - abs(value)) for value in values) / len(values),
                       "maximum_abs_error_from_expected_rad": max(
                         abs(value) if mode == "FORWARD" else abs(math.pi - abs(value)) for value in values)}
        for mode, values in alignment.items()},
      "raw_naive_curvature": raw_curvature,
      "legacy_path_only_curvature": curvature_statistics(legacy_path_only_trajectory),
      "maximum_reference_speed_m_s": max(abs(float(row["v_ref_m_s"])) for row in trajectory),
      "most_negative_reference_speed_m_s": min(float(row["v_ref_m_s"]) for row in trajectory),
      "maximum_positive_reference_speed_m_s": max(float(row["v_ref_m_s"]) for row in trajectory),
      "maximum_positive_acceleration_m_s2": max(float(row["a_ref_m_s2"]) for row in trajectory[:-1]),
      "most_negative_acceleration_m_s2": min(float(row["a_ref_m_s2"]) for row in trajectory[:-1]),
      "maximum_lateral_acceleration_m_s2": max(float(row["v_ref_m_s"]) ** 2 * abs(float(row["curvature_ref_1_m"])) for row in trajectory if int(row["curvature_valid"])),
      "maximum_abs_yaw_rate_rad_s": max(abs(float(row["omega_ref_rad_s"])) for row in trajectory),
      "maximum_abs_left_track_speed_m_s": max(abs(float(row["v_left_ref_m_s"])) for row in trajectory),
      "maximum_abs_right_track_speed_m_s": max(abs(float(row["v_right_ref_m_s"])) for row in trajectory),
      "minimum_nonzero_translational_speed_m_s": min(
        abs(float(row["v_ref_m_s"])) for row in trajectory
        if int(row["motion_direction"]) != 0 and abs(float(row["v_ref_m_s"])) > 1e-12),
      "speed_limit_reason_counts": reasons,
      "speed_limit_reason_fractions": {key: value / sum(reasons.values()) for key, value in reasons.items()},
      "dynamic_limits_source": config["dynamic_limits_source"],
      "projection_diagnostic": projection_diagnostic(trajectory),
      "controller_compatibility_assessment": [
        "Tracking Controller V1 has only been validated on nonnegative forward speed with body yaw aligned to the forward path tangent.",
        "It has not been validated for negative v_ref, forward/reverse zero-speed transitions, or pivot references with omega independent of curvature.",
        "Acceptance of negative numeric inputs does not establish closed-loop correctness; pure kinematic validation is required before Isaac Sim."
      ]},
    "limitations": [
      "The source is the velodyne LiDAR sensor-center trajectory; exact T_base_lidar is unknown.",
      "Stage D uses XY only; source z is not interpreted as terrain height.",
      "No actual loop-closure correction factor was applied and no artificial endpoint closure is introduced.",
      "Original human RC timing is discarded; reference timing is regenerated from canonical uncalibrated simulation assumptions.",
      "Cleaning, segmentation, resampling, one-pass smoothing, and fixed-span geometry settings are uncalibrated numerical preprocessing assumptions.",
      "Pivot XY is a LiDAR sensor-center smoothstep approximation because exact T_base_lidar is unknown."
    ]}
  return {"config": config, "canonical_config": canonical, "source": source,
          "cleaned": cleaned, "uniform": uniform, "smoothed_map": smoothed,
          "raw_geometry": raw_geometry, "legacy_path_only_trajectory": legacy_path_only_trajectory,
          "motion_intervals": motion_intervals, "segments": segments,
          "trajectory": trajectory, "summary": summary}


def write_outputs(result: dict[str, Any], output_dir: Path) -> dict[str, Path]:
  output_dir.mkdir(parents=True, exist_ok=True)
  config = result["config"]; output = config["output"]
  reference_path = output_dir / output["reference_filename"]
  summary_path = output_dir / output["summary_filename"]
  columns = ["index", "t_s", "dt_s", "s_m", "x_m", "y_m", "yaw_rad",
             "segment_id", "motion_direction", "curvature_valid", "u_ref_m_s",
             "v_lateral_reference_m_s",
             "curvature_ref_1_m", "curvature_numeric_1_m", "v_limit_global_m_s",
             "v_limit_curvature_m_s", "v_limit_yaw_rate_m_s", "v_limit_track_m_s",
             "v_limit_m_s", "speed_limit_reason", "v_ref_m_s", "a_ref_m_s2",
             "omega_ref_rad_s", "v_left_ref_m_s", "v_right_ref_m_s", "motion_phase"]
  with reference_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader(); writer.writerows(result["trajectory"])
  with summary_path.open("w", encoding="utf-8") as stream:
    json.dump(result["summary"], stream, indent=2, allow_nan=False); stream.write("\n")
  return {"reference": reference_path, "summary": summary_path}


def plot_outputs(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
  try:
    import matplotlib.pyplot as plt
  except ImportError as error:
    return {"status": f"skipped; matplotlib unavailable: {error}"}
  config = result["config"]; output = config["output"]
  geometry_path = output_dir / output["geometry_plot_filename"]
  profile_path = output_dir / output["profile_plot_filename"]
  figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
  axis.plot([row["x_m"] for row in result["source"]], [row["y_m"] for row in result["source"]],
            color="0.75", linewidth=.8, label="raw GLIM XY")
  axis.plot([row["x_m"] for row in result["uniform"]], [row["y_m"] for row in result["uniform"]],
            color="tab:orange", linewidth=.8, label="cleaned/resampled")
  axis.plot([row["x_m"] for row in result["smoothed_map"]], [row["y_m"] for row in result["smoothed_map"]],
            color="tab:blue", linewidth=1.2, label="smoothed")
  event = result["summary"]["source_data"]["suspicious_largest_step"]
  source_event = result["source"][event["source_index"]]
  axis.scatter([result["source"][0]["x_m"], result["source"][-1]["x_m"], source_event["x_m"]],
               [result["source"][0]["y_m"], result["source"][-1]["y_m"], source_event["y_m"]],
               c=["green", "red", "magenta"], zorder=3, label="start/end/largest step")
  axis.set_aspect("equal", adjustable="box"); axis.grid(True, alpha=.25); axis.legend()
  axis.set_title("GLIM route preprocessing (open traversal)"); axis.set_xlabel("map x [m]"); axis.set_ylabel("map y [m]")
  figure.savefig(geometry_path, dpi=160); plt.close(figure)
  rows = result["trajectory"]
  figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
  axes[0, 0].plot([row["s_m"] for row in result["raw_geometry"]],
                  [row["curvature_ref_1_m"] for row in result["raw_geometry"]], alpha=.45, label="raw naive")
  axes[0, 0].plot([row["s_m"] for row in rows], [row["curvature_ref_1_m"] for row in rows], label="processed")
  axes[0, 0].set_title("Curvature versus progress"); axes[0, 0].legend()
  axes[0, 1].plot([row["s_m"] for row in rows], [row["v_limit_m_s"] for row in rows], label="limit")
  axes[0, 1].plot([row["s_m"] for row in rows], [row["v_ref_m_s"] for row in rows], label="reference")
  axes[0, 1].set_title("Speed versus progress"); axes[0, 1].legend()
  axes[1, 0].plot([row["t_s"] for row in rows], [row["omega_ref_rad_s"] for row in rows])
  axes[1, 0].set_title("Yaw rate versus time")
  axes[1, 1].plot([row["t_s"] for row in rows], [row["a_ref_m_s2"] for row in rows])
  axes[1, 1].set_title("Acceleration versus time")
  for axis in axes.flat:
    axis.grid(True, alpha=.25); axis.set_xlabel("s [m]" if axis in axes[0] else "time [s]")
  figure.savefig(profile_path, dpi=160); plt.close(figure)
  return {"status": "generated", "geometry": str(geometry_path), "profile": str(profile_path)}
