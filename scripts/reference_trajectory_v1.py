#!/usr/bin/env python3
"""Standard-library algorithms for BUNKER Reference Trajectory V1."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


class TrajectoryError(ValueError):
  """Raised when path or trajectory input is invalid."""


def load_config(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    config = json.load(stream)
  validate_config(config)
  return config


def _positive(config: dict[str, Any], key: str) -> float:
  value = float(config[key])
  if not math.isfinite(value) or value <= 0.0:
    raise TrajectoryError(f"{key} must be a positive finite number")
  return value


def validate_config(config: dict[str, Any]) -> None:
  fixed, path, limits = config["measured_fixed"], config["path"], config["trajectory_limits"]
  _positive(fixed, "track_center_distance_m")
  for key in ("dense_sampling_step_m", "resampling_interval_m",
              "duplicate_distance_tolerance_m", "curvature_validation_boundary_margin_m",
              "curvature_near_zero_epsilon_1_m"):
    _positive(path, key)
  for key in ("maximum_body_speed_m_s", "maximum_acceleration_m_s2",
              "maximum_deceleration_m_s2", "maximum_lateral_acceleration_m_s2",
              "maximum_yaw_rate_rad_s", "maximum_track_surface_speed_m_s"):
    _positive(limits, key)
  for key in ("start_speed_m_s", "end_speed_m_s"):
    value = float(limits[key])
    if not math.isfinite(value) or value < 0.0:
      raise TrajectoryError(f"{key} must be non-negative and finite")
  if not path["segments"]:
    raise TrajectoryError("path must contain at least one segment")
  for index, segment in enumerate(path["segments"]):
    kind = segment.get("type")
    if kind == "line":
      _positive(segment, "length_m")
    elif kind == "arc":
      _positive(segment, "radius_m")
      angle = float(segment["angle_deg"])
      if not math.isfinite(angle) or abs(angle) <= 0.0:
        raise TrajectoryError(f"segment {index} arc angle must be nonzero and finite")
    else:
      raise TrajectoryError(f"segment {index} has unsupported type {kind!r}")


def segment_properties(segment: dict[str, Any]) -> tuple[float, float]:
  """Return (length, analytic curvature) for a validated primitive."""
  if segment["type"] == "line":
    return float(segment["length_m"]), 0.0
  radius = float(segment["radius_m"])
  angle = math.radians(float(segment["angle_deg"]))
  return radius * abs(angle), math.copysign(1.0 / radius, angle)


def generate_dense_path(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  path = config["path"]
  pose = path["start_pose"]
  x, y, yaw = float(pose["x_m"]), float(pose["y_m"]), float(pose["yaw_rad"])
  if not all(math.isfinite(value) for value in (x, y, yaw)):
    raise TrajectoryError("start pose must be finite")
  dense_step = float(path["dense_sampling_step_m"])
  rows = [{"dense_index": 0, "x_m": x, "y_m": y, "yaw_analytic_rad": yaw,
           "segment_index": 0, "segment_type": path["segments"][0]["type"],
           "curvature_ref_1_m": segment_properties(path["segments"][0])[1]}]
  boundaries = []
  s_analytic = 0.0
  for segment_index, segment in enumerate(path["segments"]):
    length, curvature = segment_properties(segment)
    x0, y0, yaw0 = x, y, yaw
    steps = max(1, math.ceil(length / dense_step))
    for sample_index in range(1, steps + 1):
      u = length * sample_index / steps
      if segment["type"] == "line":
        x = x0 + u * math.cos(yaw0)
        y = y0 + u * math.sin(yaw0)
        yaw = yaw0
      else:
        yaw = yaw0 + curvature * u
        x = x0 + (math.sin(yaw) - math.sin(yaw0)) / curvature
        y = y0 - (math.cos(yaw) - math.cos(yaw0)) / curvature
      rows.append({"dense_index": len(rows), "x_m": x, "y_m": y,
                   "yaw_analytic_rad": yaw, "segment_index": segment_index,
                   "segment_type": segment["type"], "curvature_ref_1_m": curvature})
    boundaries.append({"segment_index": segment_index, "segment_type": segment["type"],
                       "s_start_m": s_analytic, "s_end_m": s_analytic + length,
                       "length_m": length, "curvature_ref_1_m": curvature,
                       "radius_m": segment.get("radius_m"),
                       "angle_deg": segment.get("angle_deg")})
    s_analytic += length
  return rows, boundaries


def cumulative_arc_length(rows: Iterable[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
  clean: list[dict[str, Any]] = []
  cumulative = 0.0
  for source in rows:
    row = dict(source)
    if clean:
      distance = math.hypot(row["x_m"] - clean[-1]["x_m"], row["y_m"] - clean[-1]["y_m"])
      if distance <= tolerance:
        continue
      cumulative += distance
    row["s_m"] = cumulative
    clean.append(row)
  if len(clean) < 2:
    raise TrajectoryError("path contains fewer than two distinct XY points")
  return clean


def _segment_for_s(s_value: float, boundaries: list[dict[str, Any]]) -> dict[str, Any]:
  tolerance = 1e-10
  for boundary in boundaries[:-1]:
    if s_value < boundary["s_end_m"] - tolerance:
      return boundary
  return boundaries[-1]


def _numeric_heading(rows: list[dict[str, Any]]) -> list[float]:
  headings = []
  for index in range(len(rows)):
    before = max(0, index - 1)
    after = min(len(rows) - 1, index + 1)
    headings.append(math.atan2(rows[after]["y_m"] - rows[before]["y_m"],
                               rows[after]["x_m"] - rows[before]["x_m"]))
  unwrapped = [headings[0]]
  for heading in headings[1:]:
    delta = (heading - unwrapped[-1] + math.pi) % (2.0 * math.pi) - math.pi
    unwrapped.append(unwrapped[-1] + delta)
  return unwrapped


def _differentiate(values: list[float], coordinates: list[float]) -> list[float]:
  result = []
  for index in range(len(values)):
    before = max(0, index - 1)
    after = min(len(values) - 1, index + 1)
    denominator = coordinates[after] - coordinates[before]
    if denominator <= 0.0:
      raise TrajectoryError("differentiation coordinates must increase")
    result.append((values[after] - values[before]) / denominator)
  return result


def resample_path(dense: list[dict[str, Any]], boundaries: list[dict[str, Any]],
                  config: dict[str, Any]) -> list[dict[str, Any]]:
  path = config["path"]
  clean = cumulative_arc_length(dense, float(path["duplicate_distance_tolerance_m"]))
  total = clean[-1]["s_m"]
  interval = float(path["resampling_interval_m"])
  targets = [index * interval for index in range(math.floor(total / interval) + 1)]
  if total - targets[-1] > 1e-12:
    targets.append(total)
  else:
    targets[-1] = total
  rows: list[dict[str, Any]] = []
  source_index = 0
  for index, target in enumerate(targets):
    while source_index + 1 < len(clean) and clean[source_index + 1]["s_m"] < target:
      source_index += 1
    left = clean[source_index]
    right = clean[min(source_index + 1, len(clean) - 1)]
    span = right["s_m"] - left["s_m"]
    fraction = 0.0 if span == 0.0 else (target - left["s_m"]) / span
    boundary = _segment_for_s(target, boundaries)
    rows.append({"index": index, "s_m": target,
                 "x_m": left["x_m"] + fraction * (right["x_m"] - left["x_m"]),
                 "y_m": left["y_m"] + fraction * (right["y_m"] - left["y_m"]),
                 "segment_index": boundary["segment_index"],
                 "segment_type": boundary["segment_type"],
                 "curvature_ref_1_m": boundary["curvature_ref_1_m"]})
  headings = _numeric_heading(rows)
  curvatures = _differentiate(headings, [row["s_m"] for row in rows])
  for row, heading, curvature in zip(rows, headings, curvatures):
    row["yaw_rad"] = heading
    row["curvature_numeric_1_m"] = curvature
  return rows


def _pointwise_limits(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
  limits = config["trajectory_limits"]
  epsilon = float(config["path"]["curvature_near_zero_epsilon_1_m"])
  body_max = float(limits["maximum_body_speed_m_s"])
  lateral_max = float(limits["maximum_lateral_acceleration_m_s2"])
  yaw_max = float(limits["maximum_yaw_rate_rad_s"])
  track_max = float(limits["maximum_track_surface_speed_m_s"])
  track_spacing = float(config["measured_fixed"]["track_center_distance_m"])
  priority = list(limits["limit_reason_priority"])
  tie_tolerance = float(limits["limit_tie_tolerance_m_s"])
  for row in rows:
    curvature = float(row["curvature_ref_1_m"])
    absolute = abs(curvature)
    # A straight's curvature/yaw bounds are mathematically unbounded. Store the
    # global cap as their finite effective value so deterministic CSV output has
    # no infinities; these candidates cannot become more restrictive than global.
    curvature_limit = body_max if absolute < epsilon else math.sqrt(lateral_max / absolute)
    yaw_limit = body_max if absolute < epsilon else yaw_max / absolute
    factors = (abs(1.0 - 0.5 * track_spacing * curvature),
               abs(1.0 + 0.5 * track_spacing * curvature))
    denominator = max(factors)
    track_limit = body_max if denominator < epsilon else track_max / denominator
    candidates = {"global": body_max, "curvature": curvature_limit,
                  "yaw_rate": yaw_limit, "track_speed": track_limit}
    final_limit = min(candidates.values())
    reason = next(name for name in priority if candidates[name] <= final_limit + tie_tolerance)
    row.update({"v_limit_global_m_s": body_max,
                "v_limit_curvature_m_s": curvature_limit,
                "v_limit_yaw_rate_m_s": yaw_limit,
                "v_limit_track_m_s": track_limit, "v_limit_m_s": final_limit,
                "speed_limit_reason": reason})


def _speed_profile(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[float]:
  limits = config["trajectory_limits"]
  acceleration = float(limits["maximum_acceleration_m_s2"])
  deceleration = float(limits["maximum_deceleration_m_s2"])
  speeds = [float(row["v_limit_m_s"]) for row in rows]
  speeds[0] = min(speeds[0], float(limits["start_speed_m_s"]))
  for index in range(1, len(rows)):
    ds = rows[index]["s_m"] - rows[index - 1]["s_m"]
    speeds[index] = min(speeds[index], math.sqrt(max(0.0, speeds[index - 1] ** 2 + 2 * acceleration * ds)))
  speeds[-1] = min(speeds[-1], float(limits["end_speed_m_s"]))
  for index in range(len(rows) - 2, -1, -1):
    ds = rows[index + 1]["s_m"] - rows[index]["s_m"]
    speeds[index] = min(speeds[index], math.sqrt(max(0.0, speeds[index + 1] ** 2 + 2 * deceleration * ds)))
  return speeds


def _phase(acceleration: float, speed: float, limit: float, index: int, last: int,
           config: dict[str, Any]) -> str:
  limits = config["trajectory_limits"]
  if index == 0:
    return "start"
  if index == last:
    return "stop"
  accel_tolerance = float(limits["phase_acceleration_tolerance_m_s2"])
  if acceleration > accel_tolerance:
    return "accel"
  if acceleration < -accel_tolerance:
    return "decel"
  if limit < float(limits["maximum_body_speed_m_s"]) - float(limits["phase_speed_tolerance_m_s"]):
    return "curve_limited"
  return "cruise"


def parameterize_trajectory(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
  _pointwise_limits(rows, config)
  speeds = _speed_profile(rows, config)
  times = [0.0]
  interval_acceleration = []
  for index in range(len(rows) - 1):
    ds = rows[index + 1]["s_m"] - rows[index]["s_m"]
    denominator = speeds[index] + speeds[index + 1]
    if denominator <= 1e-12:
      raise TrajectoryError(f"both endpoint speeds are zero over interval {index}")
    times.append(times[-1] + 2.0 * ds / denominator)
    interval_acceleration.append((speeds[index + 1] ** 2 - speeds[index] ** 2) / (2.0 * ds))
  # Point-aligned acceleration uses the outgoing interval, with the final point
  # retaining the last incoming interval value. This preserves exact kinematic limits.
  point_acceleration = interval_acceleration + [interval_acceleration[-1]]
  track_spacing = float(config["measured_fixed"]["track_center_distance_m"])
  trajectory = []
  for index, source in enumerate(rows):
    row = dict(source)
    speed = speeds[index]
    omega = speed * float(row["curvature_ref_1_m"])
    row.update({"t_s": times[index], "dt_s": 0.0 if index == 0 else times[index] - times[index - 1],
                "v_ref_m_s": speed, "a_ref_m_s2": point_acceleration[index],
                "omega_ref_rad_s": omega,
                "v_left_ref_m_s": speed - 0.5 * track_spacing * omega,
                "v_right_ref_m_s": speed + 0.5 * track_spacing * omega})
    row["motion_phase"] = _phase(point_acceleration[index], speed, row["v_limit_m_s"],
                                  index, len(rows) - 1, config)
    trajectory.append(row)
  return trajectory


def curvature_error_statistics(rows: list[dict[str, Any]], boundaries: list[dict[str, Any]],
                               config: dict[str, Any]) -> dict[str, float | int]:
  margin = float(config["path"]["curvature_validation_boundary_margin_m"])
  errors = []
  for row in rows:
    boundary = boundaries[int(row["segment_index"])]
    if row["s_m"] - boundary["s_start_m"] >= margin and boundary["s_end_m"] - row["s_m"] >= margin:
      errors.append(row["curvature_numeric_1_m"] - row["curvature_ref_1_m"])
  if not errors:
    raise TrajectoryError("curvature validation margin excludes all samples")
  return {"sample_count": len(errors), "maximum_abs_error_1_m": max(abs(value) for value in errors),
          "rms_error_1_m": math.sqrt(sum(value * value for value in errors) / len(errors))}


def _corner_statistics(trajectory: list[dict[str, Any]], boundaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
  stats = []
  interval_accels = [float(row["a_ref_m_s2"]) for row in trajectory[:-1]]
  tolerance = 1e-5
  for boundary in boundaries:
    if boundary["segment_type"] != "arc":
      continue
    inside = [row for row in trajectory
              if boundary["s_start_m"] - 1e-10 <= row["s_m"] <= boundary["s_end_m"] + 1e-10]
    entry_index = min(range(len(trajectory)), key=lambda i: abs(trajectory[i]["s_m"] - boundary["s_start_m"]))
    exit_index = min(range(len(trajectory)), key=lambda i: abs(trajectory[i]["s_m"] - boundary["s_end_m"]))
    decel_start = entry_index
    while decel_start > 0 and interval_accels[decel_start - 1] < -tolerance:
      decel_start -= 1
    accel_resume = None
    for index in range(exit_index, len(interval_accels)):
      if interval_accels[index] > tolerance:
        accel_resume = index
        break
    stats.append({"segment_index": boundary["segment_index"],
                  "arc_entry_s_m": boundary["s_start_m"], "arc_exit_s_m": boundary["s_end_m"],
                  "radius_m": boundary["radius_m"],
                  "curvature_sign": 1 if boundary["curvature_ref_1_m"] > 0 else -1,
                  "minimum_speed_inside_m_s": min(row["v_ref_m_s"] for row in inside),
                  "maximum_speed_inside_m_s": max(row["v_ref_m_s"] for row in inside),
                  "deceleration_began_before_entry_m": max(
                    0.0, boundary["s_start_m"] - trajectory[decel_start]["s_m"]),
                  "acceleration_resumed_after_exit_m": (None if accel_resume is None else max(
                    0.0, trajectory[accel_resume]["s_m"] - boundary["s_end_m"]))})
  return stats


def build_reference_trajectory(config: dict[str, Any]) -> dict[str, Any]:
  dense, boundaries = generate_dense_path(config)
  dense = cumulative_arc_length(dense, float(config["path"]["duplicate_distance_tolerance_m"]))
  resampled = resample_path(dense, boundaries, config)
  trajectory = parameterize_trajectory(resampled, config)
  curvature_stats = curvature_error_statistics(resampled, boundaries, config)
  interval_accels = [row["a_ref_m_s2"] for row in trajectory[:-1]]
  lateral = [row["v_ref_m_s"] ** 2 * abs(row["curvature_ref_1_m"]) for row in trajectory]
  reason_counts = {reason: 0 for reason in config["trajectory_limits"]["limit_reason_priority"]}
  for row in trajectory:
    reason_counts[row["speed_limit_reason"]] += 1
  speed_max = float(config["trajectory_limits"]["maximum_body_speed_m_s"])
  speed_tolerance = float(config["trajectory_limits"]["phase_speed_tolerance_m_s"])
  summary = {
    "path_length_m": trajectory[-1]["s_m"], "total_duration_s": trajectory[-1]["t_s"],
    "dense_sample_count": len(dense), "resampled_sample_count": len(resampled),
    "minimum_curvature_ref_1_m": min(row["curvature_ref_1_m"] for row in trajectory),
    "maximum_curvature_ref_1_m": max(row["curvature_ref_1_m"] for row in trajectory),
    "curvature_error_away_from_boundaries": curvature_stats,
    "minimum_reference_speed_m_s": min(row["v_ref_m_s"] for row in trajectory),
    "maximum_reference_speed_m_s": max(row["v_ref_m_s"] for row in trajectory),
    "maximum_positive_acceleration_m_s2": max(interval_accels),
    "maximum_deceleration_magnitude_m_s2": abs(min(interval_accels)),
    "maximum_lateral_acceleration_m_s2": max(lateral),
    "maximum_abs_yaw_rate_rad_s": max(abs(row["omega_ref_rad_s"]) for row in trajectory),
    "maximum_abs_left_track_speed_m_s": max(abs(row["v_left_ref_m_s"]) for row in trajectory),
    "maximum_abs_right_track_speed_m_s": max(abs(row["v_right_ref_m_s"]) for row in trajectory),
    "speed_limit_reason_counts": reason_counts,
    "v_max_reached": any(abs(row["v_ref_m_s"] - speed_max) <= speed_tolerance for row in trajectory),
    "cruise_plateau_reached": any(row["motion_phase"] == "cruise" for row in trajectory),
    "corner_statistics": _corner_statistics(trajectory, boundaries),
    "segment_boundaries": boundaries, "validation_tolerances": config["validation"],
    "config_used": config,
    "calibration_statement": "Trajectory limits are uncalibrated initial simulation assumptions; no simulator-real equivalence is claimed."
  }
  return {"dense": dense, "resampled": resampled, "trajectory": trajectory,
          "boundaries": boundaries, "summary": summary}


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
  if not rows:
    raise TrajectoryError(f"cannot write empty CSV {path}")
  fieldnames = columns or list(rows[0])
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


TRAJECTORY_COLUMNS = [
  "index", "t_s", "dt_s", "s_m", "x_m", "y_m", "yaw_rad", "segment_index",
  "segment_type", "curvature_ref_1_m", "curvature_numeric_1_m", "v_limit_global_m_s",
  "v_limit_curvature_m_s", "v_limit_yaw_rate_m_s", "v_limit_track_m_s", "v_limit_m_s",
  "speed_limit_reason", "v_ref_m_s", "a_ref_m_s2", "omega_ref_rad_s",
  "v_left_ref_m_s", "v_right_ref_m_s", "motion_phase",
]


def write_outputs(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
  output_dir.mkdir(parents=True, exist_ok=True)
  paths = {"dense_csv": output_dir / "path_dense.csv",
           "resampled_csv": output_dir / "path_resampled.csv",
           "trajectory_csv": output_dir / "reference_trajectory.csv",
           "summary_json": output_dir / "summary.json"}
  write_csv(paths["dense_csv"], result["dense"])
  write_csv(paths["resampled_csv"], result["resampled"])
  write_csv(paths["trajectory_csv"], result["trajectory"], TRAJECTORY_COLUMNS)
  with paths["summary_json"].open("w", encoding="utf-8") as stream:
    json.dump(result["summary"], stream, indent=2, allow_nan=False)
    stream.write("\n")
  return {key: str(value) for key, value in paths.items()}
