#!/usr/bin/env python3
"""Pure geometry, periodic trajectory, metrics, and output helpers for the global-path demo."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from reference_trajectory_v1 import _pointwise_limits
from tracking_controller_v1 import wrap_to_pi


REPO_ROOT = Path(__file__).resolve().parents[1]


class GlobalPathDemoError(ValueError):
  """Raised when a manual closed-loop path or demo configuration is invalid."""


def load_config(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    config = json.load(stream)
  if not config["geometry"]["closed"]:
    raise GlobalPathDemoError("Global Path Demo V1 requires explicit closed geometry")
  if config["geometry"]["corner_smoothing_method"] != "constant_radius_circular_fillet":
    raise GlobalPathDemoError("unsupported corner smoothing method")
  for key in ("corner_radius_m", "resampling_interval_m",
              "minimum_straight_between_fillets_m", "duplicate_distance_tolerance_m"):
    value = float(config["geometry"][key])
    if not math.isfinite(value) or value <= 0.0:
      raise GlobalPathDemoError(f"geometry.{key} must be positive and finite")
  if int(config["execution"]["default_laps"]) < 2:
    raise GlobalPathDemoError("default_laps must demonstrate at least two continuous laps")
  required = {"rounded_loop", "zigzag_loop", "lawnmower_loop"}
  if set(config["presets"]) != required:
    raise GlobalPathDemoError(f"core preset inventory must be exactly {sorted(required)}")
  for name, preset in config["presets"].items():
    points = preset["waypoints_map_xy_m"]
    if not preset["closed"] or len(points) < 4:
      raise GlobalPathDemoError(f"{name} must be an explicit closed path with at least four waypoints")
    if points[0] == points[-1]:
      raise GlobalPathDemoError(f"{name} must not duplicate its first waypoint; closed metadata owns closure")
    if not all(len(point) == 2 and all(math.isfinite(float(value)) for value in point)
               for point in points):
      raise GlobalPathDemoError(f"{name} contains an invalid map-frame waypoint")
  return config


def load_canonical_trajectory_config(config: dict[str, Any]) -> dict[str, Any]:
  path = REPO_ROOT / config["baseline_configs"]["trajectory_limits"]
  with path.open(encoding="utf-8") as stream:
    canonical = json.load(stream)
  if abs(float(canonical["measured_fixed"]["track_center_distance_m"]) - 0.434) > 1e-12:
    raise GlobalPathDemoError("canonical measured track-center spacing changed")
  return canonical


def _unit(first: tuple[float, float], second: tuple[float, float]) -> tuple[float, float, float]:
  dx, dy = second[0] - first[0], second[1] - first[1]
  length = math.hypot(dx, dy)
  if length <= 1e-12:
    raise GlobalPathDemoError("adjacent manual waypoints must be distinct")
  return dx / length, dy / length, length


def _orientation(a: tuple[float, float], b: tuple[float, float],
                 c: tuple[float, float]) -> float:
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_intersection(a: tuple[float, float], b: tuple[float, float],
                         c: tuple[float, float], d: tuple[float, float]) -> bool:
  first = _orientation(a, b, c); second = _orientation(a, b, d)
  third = _orientation(c, d, a); fourth = _orientation(c, d, b)
  tolerance = 1e-12
  return first * second < -tolerance and third * fourth < -tolerance


def raw_self_intersections(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
  count = len(points); crossings = []
  for first in range(count):
    a, b = points[first], points[(first + 1) % count]
    for second in range(first + 1, count):
      if second in (first, (first + 1) % count) or first == (second + 1) % count:
        continue
      c, d = points[second], points[(second + 1) % count]
      if _proper_intersection(a, b, c, d):
        crossings.append((first, second))
  return crossings


def _fillet_corners(points: list[tuple[float, float]], radius: float
                    ) -> list[dict[str, Any]]:
  corners = []
  for index, point in enumerate(points):
    previous, following = points[index - 1], points[(index + 1) % len(points)]
    ux_in, uy_in, incoming_length = _unit(previous, point)
    ux_out, uy_out, outgoing_length = _unit(point, following)
    turn = math.atan2(ux_in * uy_out - uy_in * ux_out,
                      ux_in * ux_out + uy_in * uy_out)
    if abs(abs(turn) - math.pi) < 1e-8:
      raise GlobalPathDemoError(f"waypoint {index} is an unsupported exact reversal")
    offset = 0.0 if abs(turn) < 1e-10 else radius * math.tan(0.5 * abs(turn))
    if offset >= min(incoming_length, outgoing_length) - 1e-10:
      raise GlobalPathDemoError(f"fillet at waypoint {index} consumes an adjacent segment")
    corners.append({
      "waypoint_index": index, "point": point, "incoming_unit": (ux_in, uy_in),
      "outgoing_unit": (ux_out, uy_out), "turn_rad": turn, "offset_m": offset,
      "entry": (point[0] - offset * ux_in, point[1] - offset * uy_in),
      "exit": (point[0] + offset * ux_out, point[1] + offset * uy_out),
    })
  return corners


def _line(start: tuple[float, float], end: tuple[float, float], label: str) -> dict[str, Any]:
  ux, uy, length = _unit(start, end)
  return {"type": "line", "start": start, "end": end, "length_m": length,
          "geometric_yaw_rad": math.atan2(uy, ux), "turn_rad": 0.0, "label": label}


def _arc(corner: dict[str, Any], radius: float) -> dict[str, Any] | None:
  turn = float(corner["turn_rad"])
  if abs(turn) < 1e-10:
    return None
  return {"type": "arc", "start": corner["entry"], "end": corner["exit"],
          "length_m": radius * abs(turn), "turn_rad": turn,
          "geometric_yaw_rad": math.atan2(corner["incoming_unit"][1], corner["incoming_unit"][0]),
          "radius_m": radius, "waypoint_index": corner["waypoint_index"],
          "label": f"corner_{corner['waypoint_index']}"}


def build_fillet_primitives(points: list[tuple[float, float]], geometry: dict[str, Any]
                            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Build a C1 closed rounded polyline and put s=0 at the middle of a long straight."""
  radius = float(geometry["corner_radius_m"])
  corners = _fillet_corners(points, radius)
  lines = []
  for index, corner in enumerate(corners):
    following = corners[(index + 1) % len(corners)]
    segment = _line(corner["exit"], following["entry"], f"straight_{index}")
    if segment["length_m"] < float(geometry["minimum_straight_between_fillets_m"]):
      raise GlobalPathDemoError(f"fillets leave too little straight at waypoint edge {index}")
    lines.append(segment)
  # Split edge zero at its midpoint. This makes both sides of the periodic
  # s=0/L boundary straight, so curvature and tangent are exactly periodic.
  first = lines[0]
  midpoint = (0.5 * (first["start"][0] + first["end"][0]),
              0.5 * (first["start"][1] + first["end"][1]))
  primitives = [_line(midpoint, first["end"], "straight_0_tail")]
  arc = _arc(corners[1], radius)
  if arc: primitives.append(arc)
  for index in range(1, len(points)):
    primitives.append(lines[index])
    following_index = (index + 1) % len(points)
    arc = _arc(corners[following_index], radius)
    if arc: primitives.append(arc)
  primitives.append(_line(first["start"], midpoint, "straight_0_head"))

  current_yaw = float(primitives[0]["geometric_yaw_rad"])
  start_yaw = current_yaw
  cumulative = 0.0
  for segment_id, primitive in enumerate(primitives):
    mismatch = wrap_to_pi(float(primitive["geometric_yaw_rad"]) - current_yaw)
    if abs(mismatch) > 1e-8:
      raise GlobalPathDemoError(f"fillet tangent discontinuity before {primitive['label']}: {mismatch}")
    primitive["segment_id"] = segment_id
    primitive["s_start_m"] = cumulative
    primitive["yaw_start_map_rad"] = current_yaw
    primitive["curvature_ref_1_m"] = (0.0 if primitive["type"] == "line" else
                                       math.copysign(1.0 / radius, primitive["turn_rad"]))
    cumulative += float(primitive["length_m"])
    current_yaw += float(primitive["turn_rad"])
    primitive["s_end_m"] = cumulative
    primitive["yaw_end_map_rad"] = current_yaw
  if abs(abs(current_yaw - start_yaw) - 2.0 * math.pi) > 1e-8:
    raise GlobalPathDemoError("core closed path must have winding number +/-1")
  return primitives, corners


def _sample_primitive(primitive: dict[str, Any], distance: float) -> tuple[float, float, float]:
  distance = max(0.0, min(float(primitive["length_m"]), distance))
  x0, y0 = primitive["start"]
  yaw0 = float(primitive["yaw_start_map_rad"])
  curvature = float(primitive["curvature_ref_1_m"])
  yaw = yaw0 + curvature * distance
  if abs(curvature) <= 1e-15:
    return x0 + distance * math.cos(yaw0), y0 + distance * math.sin(yaw0), yaw
  return (x0 + (math.sin(yaw) - math.sin(yaw0)) / curvature,
          y0 - (math.cos(yaw) - math.cos(yaw0)) / curvature, yaw)


def resample_primitives(primitives: list[dict[str, Any]], geometry: dict[str, Any]
                        ) -> list[dict[str, Any]]:
  length = float(primitives[-1]["s_end_m"])
  interval = float(geometry["resampling_interval_m"])
  targets = [index * interval for index in range(math.floor(length / interval) + 1)]
  if length - targets[-1] > 1e-12: targets.append(length)
  else: targets[-1] = length
  rows = []
  primitive_index = 0
  for index, target in enumerate(targets):
    while (primitive_index + 1 < len(primitives) and
           target >= float(primitives[primitive_index]["s_end_m"]) - 1e-12):
      primitive_index += 1
    primitive = primitives[primitive_index]
    x_m, y_m, yaw = _sample_primitive(primitive, target - float(primitive["s_start_m"]))
    rows.append({"index": index, "s_m": target, "map_x_m": x_m, "map_y_m": y_m,
                 "yaw_map_rad": yaw, "segment_id": int(primitive["segment_id"]),
                 "segment_type": primitive["type"], "segment_label": primitive["label"],
                 "curvature_ref_1_m": float(primitive["curvature_ref_1_m"]),
                 "motion_direction": 1, "curvature_valid": 1})
  # Pin periodic position and tangent exactly rather than retaining roundoff from
  # a full traversal of the analytic primitives.
  rows[-1]["map_x_m"] = rows[0]["map_x_m"]
  rows[-1]["map_y_m"] = rows[0]["map_y_m"]
  rows[-1]["yaw_map_rad"] = rows[0]["yaw_map_rad"] + (
    float(primitives[-1]["yaw_end_map_rad"]) - float(primitives[0]["yaw_start_map_rad"]))
  rows[-1]["curvature_ref_1_m"] = rows[0]["curvature_ref_1_m"]
  return rows


def map_to_local(x_m: float, y_m: float, transform: dict[str, float]) -> tuple[float, float]:
  dx = x_m - transform["map_origin_x_m"]
  dy = y_m - transform["map_origin_y_m"]
  yaw = transform["map_initial_path_yaw_rad"]
  return math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy


def normalize_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
  transform = {"map_origin_x_m": float(rows[0]["map_x_m"]),
               "map_origin_y_m": float(rows[0]["map_y_m"]),
               "map_initial_path_yaw_rad": float(rows[0]["yaw_map_rad"]),
               "map_to_local_rotation_rad": -float(rows[0]["yaw_map_rad"])}
  normalized = []
  for source in rows:
    row = dict(source)
    row["x_m"], row["y_m"] = map_to_local(float(row["map_x_m"]), float(row["map_y_m"]), transform)
    row["yaw_rad"] = float(row["yaw_map_rad"]) - transform["map_initial_path_yaw_rad"]
    normalized.append(row)
  normalized[-1]["x_m"] = normalized[0]["x_m"]
  normalized[-1]["y_m"] = normalized[0]["y_m"]
  numeric = []
  for index in range(len(normalized)):
    before = max(0, index - 1); after = min(len(normalized) - 1, index + 1)
    ds = float(normalized[after]["s_m"]) - float(normalized[before]["s_m"])
    numeric.append((float(normalized[after]["yaw_rad"]) - float(normalized[before]["yaw_rad"])) / ds)
  for row, curvature in zip(normalized, numeric): row["curvature_numeric_1_m"] = curvature
  return normalized, transform


def _periodic_speed_profile(rows: list[dict[str, Any]], canonical: dict[str, Any]) -> list[float]:
  unique = rows[:-1]
  if len(unique) < 3:
    raise GlobalPathDemoError("closed path has too few unique samples")
  acceleration = float(canonical["trajectory_limits"]["maximum_acceleration_m_s2"])
  deceleration = float(canonical["trajectory_limits"]["maximum_deceleration_m_s2"])
  length = float(rows[-1]["s_m"])
  distances = [float(unique[index + 1]["s_m"]) - float(unique[index]["s_m"])
               for index in range(len(unique) - 1)] + [length - float(unique[-1]["s_m"])]
  speeds = [float(row["v_limit_m_s"]) for row in unique]
  for _ in range(4 * len(unique)):
    changed = False
    for index in range(len(unique)):
      following = (index + 1) % len(unique)
      feasible = math.sqrt(max(0.0, speeds[index] ** 2 + 2.0 * acceleration * distances[index]))
      if feasible + 1e-14 < speeds[following]: speeds[following] = feasible; changed = True
    for index in range(len(unique) - 1, -1, -1):
      following = (index + 1) % len(unique)
      feasible = math.sqrt(max(0.0, speeds[following] ** 2 + 2.0 * deceleration * distances[index]))
      if feasible + 1e-14 < speeds[index]: speeds[index] = feasible; changed = True
    if not changed: break
  else:
    raise GlobalPathDemoError("periodic speed propagation did not converge")
  return speeds + [speeds[0]]


def _times_and_accelerations(rows: list[dict[str, Any]], speeds: list[float]
                             ) -> tuple[list[float], list[float]]:
  times = [0.0]; accelerations = []
  for index in range(len(rows) - 1):
    ds = float(rows[index + 1]["s_m"]) - float(rows[index]["s_m"])
    denominator = speeds[index] + speeds[index + 1]
    if denominator <= 1e-12:
      raise GlobalPathDemoError(f"zero speed at both ends of nonzero interval {index}")
    times.append(times[-1] + 2.0 * ds / denominator)
    accelerations.append((speeds[index + 1] ** 2 - speeds[index] ** 2) / (2.0 * ds))
  return times, accelerations + [accelerations[-1]]


def _phase(acceleration: float, row: dict[str, Any], canonical: dict[str, Any]) -> str:
  limits = canonical["trajectory_limits"]
  if acceleration > float(limits["phase_acceleration_tolerance_m_s2"]): return "accel"
  if acceleration < -float(limits["phase_acceleration_tolerance_m_s2"]): return "decel"
  if float(row["v_limit_m_s"]) < float(limits["maximum_body_speed_m_s"]) - float(limits["phase_speed_tolerance_m_s"]):
    return "curve_limited"
  return "cruise"


def parameterize_periodic_lap(rows: list[dict[str, Any]], canonical: dict[str, Any]
                              ) -> list[dict[str, Any]]:
  _pointwise_limits(rows, canonical)
  speeds = _periodic_speed_profile(rows, canonical)
  times, accelerations = _times_and_accelerations(rows, speeds)
  spacing = float(canonical["measured_fixed"]["track_center_distance_m"])
  result = []
  for index, source in enumerate(rows):
    row = dict(source); speed = speeds[index]
    omega = speed * float(row["curvature_ref_1_m"])
    row.update({"t_s": times[index], "dt_s": 0.0 if index == 0 else times[index] - times[index - 1],
                "v_ref_m_s": speed, "v_periodic_m_s": speed,
                "a_ref_m_s2": accelerations[index], "omega_ref_rad_s": omega,
                "v_left_ref_m_s": speed - 0.5 * spacing * omega,
                "v_right_ref_m_s": speed + 0.5 * spacing * omega,
                "motion_phase": "periodic_wrap" if index == len(rows) - 1 else
                                _phase(accelerations[index], row, canonical)})
    result.append(row)
  return result


def build_runtime_reference(periodic: list[dict[str, Any]], laps: int,
                            canonical: dict[str, Any], config: dict[str, Any]
                            ) -> list[dict[str, Any]]:
  if laps < 2: raise GlobalPathDemoError("at least two laps are required for repeat-loop validation")
  lap_length = float(periodic[-1]["s_m"])
  yaw_per_lap = float(periodic[-1]["yaw_rad"]) - float(periodic[0]["yaw_rad"])
  rows = []
  for lap in range(laps):
    for source in periodic[:-1]:
      row = dict(source)
      row.update({"index": len(rows), "lap_index": lap, "s_within_lap_m": float(source["s_m"]),
                  "s_m": lap * lap_length + float(source["s_m"]),
                  "yaw_rad": float(source["yaw_rad"]) + lap * yaw_per_lap,
                  "yaw_map_rad": float(source["yaw_map_rad"]) + lap * yaw_per_lap})
      rows.append(row)
  final = dict(periodic[-1])
  final.update({"index": len(rows), "lap_index": laps - 1, "s_within_lap_m": lap_length,
                "s_m": laps * lap_length,
                "yaw_rad": float(periodic[-1]["yaw_rad"]) + (laps - 1) * yaw_per_lap,
                "yaw_map_rad": float(periodic[-1]["yaw_map_rad"]) + (laps - 1) * yaw_per_lap})
  rows.append(final)

  limits = canonical["trajectory_limits"]
  acceleration = float(limits["maximum_acceleration_m_s2"])
  deceleration = float(limits["maximum_deceleration_m_s2"])
  speeds = [float(row["v_periodic_m_s"]) for row in rows]
  speeds[0] = min(speeds[0], float(config["execution"]["launch_speed_m_s"]))
  for index in range(1, len(rows)):
    ds = float(rows[index]["s_m"]) - float(rows[index - 1]["s_m"])
    speeds[index] = min(speeds[index], math.sqrt(max(0.0, speeds[index - 1] ** 2 + 2.0 * acceleration * ds)))
  speeds[-1] = min(speeds[-1], float(config["execution"]["final_speed_m_s"]))
  for index in range(len(rows) - 2, -1, -1):
    ds = float(rows[index + 1]["s_m"]) - float(rows[index]["s_m"])
    speeds[index] = min(speeds[index], math.sqrt(max(0.0, speeds[index + 1] ** 2 + 2.0 * deceleration * ds)))
  times, accelerations = _times_and_accelerations(rows, speeds)
  spacing = float(canonical["measured_fixed"]["track_center_distance_m"])
  for index, row in enumerate(rows):
    speed = speeds[index]; omega = speed * float(row["curvature_ref_1_m"])
    row.update({"t_s": times[index], "dt_s": 0.0 if index == 0 else times[index] - times[index - 1],
                "v_ref_m_s": speed, "a_ref_m_s2": accelerations[index],
                "omega_ref_rad_s": omega,
                "v_left_ref_m_s": speed - 0.5 * spacing * omega,
                "v_right_ref_m_s": speed + 0.5 * spacing * omega,
                "motion_phase": "launch" if index == 0 else ("final_stop" if index == len(rows) - 1 else
                                _phase(accelerations[index], row, canonical)),
                "lap_progress_fraction": float(row["s_within_lap_m"]) / lap_length})
  return rows


def _constraint_summary(rows: list[dict[str, Any]], canonical: dict[str, Any]) -> dict[str, Any]:
  limits = canonical["trajectory_limits"]
  interval_accels = [float(row["a_ref_m_s2"]) for row in rows[:-1]]
  return {
    "maximum_reference_speed_m_s": max(float(row["v_ref_m_s"]) for row in rows),
    "maximum_positive_acceleration_m_s2": max(interval_accels),
    "maximum_deceleration_magnitude_m_s2": abs(min(interval_accels)),
    "maximum_lateral_acceleration_m_s2": max(float(row["v_ref_m_s"]) ** 2 * abs(float(row["curvature_ref_1_m"])) for row in rows),
    "maximum_abs_yaw_rate_rad_s": max(abs(float(row["omega_ref_rad_s"])) for row in rows),
    "maximum_abs_left_track_speed_m_s": max(abs(float(row["v_left_ref_m_s"])) for row in rows),
    "maximum_abs_right_track_speed_m_s": max(abs(float(row["v_right_ref_m_s"])) for row in rows),
    "limits": {key: limits[key] for key in ("maximum_body_speed_m_s", "maximum_acceleration_m_s2",
      "maximum_deceleration_m_s2", "maximum_lateral_acceleration_m_s2",
      "maximum_yaw_rate_rad_s", "maximum_track_surface_speed_m_s")},
  }


def build_demo_reference(config_path: Path, preset_name: str, laps: int | None = None
                         ) -> dict[str, Any]:
  config = load_config(config_path)
  if preset_name not in config["presets"]:
    raise GlobalPathDemoError(f"unknown path {preset_name!r}; choose one of {sorted(config['presets'])}")
  canonical = load_canonical_trajectory_config(config)
  preset = config["presets"][preset_name]
  points = [(float(point[0]), float(point[1])) for point in preset["waypoints_map_xy_m"]]
  crossings = raw_self_intersections(points)
  if crossings: raise GlobalPathDemoError(f"{preset_name} raw polyline self-intersections: {crossings}")
  primitives, corners = build_fillet_primitives(points, config["geometry"])
  map_rows = resample_primitives(primitives, config["geometry"])
  normalized, transform = normalize_rows(map_rows)
  periodic = parameterize_periodic_lap(normalized, canonical)
  lap_count = int(config["execution"]["default_laps"] if laps is None else laps)
  runtime = build_runtime_reference(periodic, lap_count, canonical, config)
  local_global = [map_to_local(x, y, transform) for x, y in points]
  local_global.append(local_global[0])
  map_global = points + [points[0]]
  lap_length = float(periodic[-1]["s_m"])
  boundary_rows = [runtime[min(range(len(runtime)), key=lambda index: abs(
    float(runtime[index]["s_m"]) - lap * lap_length))] for lap in range(1, lap_count)]
  periodic_start_speed = float(periodic[0]["v_ref_m_s"])
  phase_counts = {phase: sum(row["motion_phase"] == phase for row in periodic[:-1])
                  for phase in ("accel", "cruise", "decel", "curve_limited")}
  summary = {
    "metadata": config["metadata"], "preset": preset_name, "intent": preset["intent"],
    "provider": {"type": "manual_deterministic_preset", "planner_implemented": False,
                 "waypoint_frame": "map", "waypoints_map_xy_m": preset["waypoints_map_xy_m"]},
    "map": config["map"], "map_to_local_transform": transform,
    "geometry": {"closed": True, "explicit_closure_metadata": True,
      "manual_waypoint_count": len(points), "raw_self_intersection_count": len(crossings),
      "smoothing_method": config["geometry"]["corner_smoothing_method"],
      "common_corner_radius_m": config["geometry"]["corner_radius_m"],
      "primitive_count": len(primitives), "filleted_corner_count": sum(abs(float(row["turn_rad"])) > 1e-10 for row in corners),
      "lap_length_m": lap_length, "sample_count_including_periodic_endpoint": len(periodic),
      "position_closure_error_m": math.hypot(float(periodic[-1]["x_m"]) - float(periodic[0]["x_m"]),
                                               float(periodic[-1]["y_m"]) - float(periodic[0]["y_m"])),
      "tangent_closure_error_rad": abs(wrap_to_pi(float(periodic[-1]["yaw_rad"]) - float(periodic[0]["yaw_rad"]))),
      "curvature_closure_error_1_m": abs(float(periodic[-1]["curvature_ref_1_m"]) - float(periodic[0]["curvature_ref_1_m"])),
      "yaw_change_per_lap_rad": float(periodic[-1]["yaw_rad"]) - float(periodic[0]["yaw_rad"]),
      "minimum_curvature_1_m": min(float(row["curvature_ref_1_m"]) for row in periodic),
      "maximum_curvature_1_m": max(float(row["curvature_ref_1_m"]) for row in periodic)},
    "periodic_profile": {"lap_duration_s": float(periodic[-1]["t_s"]),
      "start_speed_m_s": periodic_start_speed, "end_speed_m_s": float(periodic[-1]["v_ref_m_s"]),
      "speed_closure_error_m_s": abs(float(periodic[-1]["v_ref_m_s"]) - periodic_start_speed),
      "omega_closure_error_rad_s": abs(float(periodic[-1]["omega_ref_rad_s"]) - float(periodic[0]["omega_ref_rad_s"])),
      "phase_sample_counts": phase_counts, **_constraint_summary(periodic, canonical)},
    "execution_profile": {"laps": lap_count, "duration_s": float(runtime[-1]["t_s"]),
      "total_progress_m": float(runtime[-1]["s_m"]), "launch_speed_m_s": float(runtime[0]["v_ref_m_s"]),
      "final_speed_m_s": float(runtime[-1]["v_ref_m_s"]),
      "interior_lap_boundary_speeds_m_s": [float(row["v_ref_m_s"]) for row in boundary_rows],
      "interior_lap_boundary_speed_errors_from_periodic_m_s": [abs(float(row["v_ref_m_s"]) - periodic_start_speed) for row in boundary_rows],
      "interior_lap_stop_count": sum(float(row["v_ref_m_s"]) <= 1e-9 for row in boundary_rows),
      **_constraint_summary(runtime, canonical)},
    "limitations": [
      "Manual deterministic path provider only; no global planner or obstacle/collision checker is implemented.",
      "Bag C PLY is XY display/coverage context only; its Z is not used as terrain or collision geometry.",
      "Circular fillets have finite curvature but step changes in curvature at line/arc transitions.",
      "Dynamic limits, controller gains, and Isaac V2 plant remain uncalibrated simulation assumptions."
    ]}
  return {"config": config, "canonical": canonical, "preset": preset,
          "primitives": primitives, "corners": corners,
          "global_path_map": map_global, "global_path_local": local_global,
          "periodic_lap": periodic, "trajectory": runtime, "summary": summary}


def rms(values: Iterable[float]) -> float:
  data = list(values)
  if not data: raise GlobalPathDemoError("cannot compute RMS of an empty selection")
  return math.sqrt(sum(value * value for value in data) / len(data))


def oscillation_diagnostics(rows: list[dict[str, Any]], config: dict[str, Any],
                            curvature_key: str = "reference_curvature_1_m") -> dict[str, Any]:
  evaluation = config["evaluation"]
  selected = [row for row in rows
              if abs(float(row[curvature_key])) <= float(evaluation["straight_curvature_threshold_1_m"])
              and float(row["v_ref_m_s"]) >= float(evaluation["straight_minimum_speed_m_s"])]
  values = [float(row["e_y_m"]) for row in selected]
  deadband = float(evaluation["oscillation_zero_deadband_m"])
  signs = []
  for value in values:
    sign = 1 if value > deadband else (-1 if value < -deadband else 0)
    if sign and (not signs or sign != signs[-1]): signs.append(sign)
  return {"selection": "straight, forward, and above configured minimum reference speed",
          "sample_count": len(selected), "lateral_e_y_rms_m": rms(values) if values else 0.0,
          "lateral_e_y_peak_to_peak_m": max(values) - min(values) if values else 0.0,
          "zero_crossing_count": max(0, len(signs) - 1), "zero_deadband_m": deadband}


def metric_block(rows: list[dict[str, Any]], saturation_tolerance: float,
                 speed_error_key: str) -> dict[str, Any]:
  if not rows: raise GlobalPathDemoError("cannot summarize an empty lap")
  mappings = {"cte_m": "cross_track_error_m", "heading_rad": "heading_error_rad",
              "speed_m_s": speed_error_key, "progress_m": "progress_error_m"}
  result: dict[str, Any] = {"sample_count": len(rows)}
  for label, key in mappings.items():
    values = [float(row[key]) for row in rows]
    result[f"rms_{label}"] = rms(values)
    result[f"maximum_abs_{label}"] = max(abs(value) for value in values)
  updates = [row for row in rows if "control_update" not in row or int(row["control_update"])]
  saturated = sum(float(row["command_scale"]) < 1.0 - saturation_tolerance for row in updates)
  result.update({"control_update_count": len(updates), "saturated_control_update_count": saturated,
                 "saturation_fraction": saturated / len(updates) if updates else 0.0,
                 "minimum_command_scale": min(float(row["command_scale"]) for row in rows)})
  return result


def lap_metrics(rows: list[dict[str, Any]], lap_length: float, laps: int,
                config: dict[str, Any], time_key: str, reference_s_key: str,
                curvature_key: str, speed_error_key: str,
                active_end_time: float) -> dict[str, Any]:
  active = [row for row in rows if float(row[time_key]) <= active_end_time + 1e-12]
  tolerance = float(config["execution"]["saturation_scale_tolerance"])
  blocks = {}
  for lap in range(laps):
    selected = []
    for row in active:
      progress = float(row[reference_s_key])
      row_lap = min(laps - 1, max(0, int(math.floor(progress / lap_length + 1e-12))))
      if row_lap == lap: selected.append(row)
    metrics = metric_block(selected, tolerance, speed_error_key)
    metrics["oscillation"] = oscillation_diagnostics(selected, config, curvature_key)
    blocks[str(lap + 1)] = metrics
  first, second = blocks["1"], blocks["2"]
  floor = float(config["evaluation"]["lap_growth_absolute_floor_m"])
  allowed = float(config["evaluation"]["lap2_maximum_cte_growth_factor"]) * max(
    float(first["rms_cte_m"]), floor)
  return {"laps": blocks, "lap_1_to_2": {
    "cte_rms_growth_factor": float(second["rms_cte_m"]) / max(float(first["rms_cte_m"]), 1e-12),
    "allowed_lap_2_rms_cte_m": allowed,
    "lap_2_nondivergent": float(second["rms_cte_m"]) <= allowed}}


def evaluate_gate(metrics: dict[str, Any], config: dict[str, Any], kind: str) -> dict[str, Any]:
  overall = metrics["overall"]; evaluation = config["evaluation"]
  prefix = "kinematic" if kind == "kinematic" else "isaac"
  checks = {
    "rms_cte": float(overall["rms_cte_m"]) <= float(evaluation[f"{prefix}_maximum_rms_cte_m"]),
    "maximum_cte": float(overall["maximum_abs_cte_m"]) <= float(evaluation[f"{prefix}_maximum_abs_cte_m"]),
    "rms_heading": float(overall["rms_heading_rad"]) <= float(evaluation[f"{prefix}_maximum_rms_heading_rad"]),
    "maximum_heading": float(overall["maximum_abs_heading_rad"]) <= float(evaluation[f"{prefix}_maximum_abs_heading_rad"]),
    "rms_progress": float(overall["rms_progress_m"]) <= float(evaluation[f"{prefix}_maximum_rms_progress_error_m"]),
    "maximum_progress": float(overall["maximum_abs_progress_m"]) <= float(evaluation[f"{prefix}_maximum_abs_progress_error_m"]),
    "lap_2_nondivergent": bool(metrics["lap_metrics"]["lap_1_to_2"]["lap_2_nondivergent"]),
  }
  return {"passed": all(checks.values()), "checks": checks,
          "policy": "Deterministic nominal integration gate; thresholds are demo regression limits, not real-robot claims."}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
