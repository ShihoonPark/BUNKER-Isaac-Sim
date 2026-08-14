#!/usr/bin/env python3
"""Standard-library Tracking Controller V1 and kinematic validation tools."""

from __future__ import annotations

import bisect
import csv
import json
import math
from pathlib import Path
from typing import Any


class ControllerError(ValueError):
  """Raised when controller configuration or numerical input is invalid."""


def load_controller_config(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    config = json.load(stream)
  validate_controller_config(config)
  return config


def validate_controller_config(config: dict[str, Any]) -> None:
  fixed = config["measured_fixed"]
  controller = config["controller"]
  limits = config["command_limits"]
  simulation = config["simulation"]
  values = [float(fixed["track_center_distance_m"]),
            *(float(value) for value in controller.values()),
            float(simulation["time_step_s"]), float(simulation["terminal_hold_s"]),
            float(simulation["near_zero_yaw_rate_rad_s"])]
  if not all(math.isfinite(value) for value in values):
    raise ControllerError("geometry, gains, and simulation values must be finite")
  if float(fixed["track_center_distance_m"]) <= 0.0:
    raise ControllerError("track center distance must be positive")
  for key in ("maximum_abs_body_speed_m_s", "maximum_abs_yaw_rate_rad_s",
              "maximum_abs_track_surface_speed_m_s"):
    value = float(limits[key])
    if not math.isfinite(value) or value <= 0.0:
      raise ControllerError(f"{key} must be positive and finite")
  if float(simulation["time_step_s"]) <= 0.0 or float(simulation["terminal_hold_s"]) < 0.0:
    raise ControllerError("simulation time step must be positive and hold time non-negative")


def wrap_to_pi(angle: float) -> float:
  """Wrap to [-pi, pi), giving deterministic shortest-angle differences."""
  return (angle + math.pi) % (2.0 * math.pi) - math.pi


class ReferenceInterpolator:
  """Deterministic linear time interpolation of continuous-yaw references."""

  FIELDS = ("s_m", "x_m", "y_m", "yaw_rad", "curvature_ref_1_m",
            "v_ref_m_s", "omega_ref_rad_s", "a_ref_m_s2")

  def __init__(self, rows: list[dict[str, Any]]):
    if len(rows) < 2:
      raise ControllerError("reference requires at least two samples")
    self.rows = rows
    self.times = [float(row["t_s"]) for row in rows]
    if not all(self.times[index] > self.times[index - 1] for index in range(1, len(rows))):
      raise ControllerError("reference times must be strictly increasing")

  def sample(self, query_time: float) -> dict[str, float]:
    if not math.isfinite(query_time):
      raise ControllerError("query time must be finite")
    if query_time <= self.times[0]:
      return self._copy(self.rows[0], query_time)
    if query_time >= self.times[-1]:
      result = self._copy(self.rows[-1], query_time)
      result["v_ref_m_s"] = 0.0
      result["omega_ref_rad_s"] = 0.0
      result["a_ref_m_s2"] = 0.0
      return result
    right_index = bisect.bisect_right(self.times, query_time)
    left, right = self.rows[right_index - 1], self.rows[right_index]
    fraction = (query_time - float(left["t_s"])) / (float(right["t_s"]) - float(left["t_s"]))
    result = {field: float(left[field]) + fraction * (float(right[field]) - float(left[field]))
              for field in self.FIELDS}
    result["t_s"] = query_time
    return result

  @classmethod
  def _copy(cls, row: dict[str, Any], query_time: float) -> dict[str, float]:
    result = {field: float(row[field]) for field in cls.FIELDS}
    result["t_s"] = query_time
    return result


def body_frame_errors(actual: tuple[float, float, float], reference: dict[str, float]) -> dict[str, float]:
  x, y, yaw = actual
  dx = reference["x_m"] - x
  dy = reference["y_m"] - y
  return {"e_x_m": math.cos(yaw) * dx + math.sin(yaw) * dy,
          "e_y_m": -math.sin(yaw) * dx + math.cos(yaw) * dy,
          "e_heading_rad": wrap_to_pi(reference["yaw_rad"] - yaw)}


def raw_control(actual: tuple[float, float, float], reference: dict[str, float],
                config: dict[str, Any]) -> dict[str, float]:
  errors = body_frame_errors(actual, reference)
  gains = config["controller"]
  heading = errors["e_heading_rad"]
  v_raw = reference["v_ref_m_s"] * math.cos(heading) + float(gains["k_longitudinal_1_s"]) * errors["e_x_m"]
  omega_raw = (reference["omega_ref_rad_s"]
               + float(gains["k_lateral_rad_s_per_m"]) * errors["e_y_m"]
               + float(gains["k_heading_1_s"]) * math.sin(heading))
  return {**errors, "v_raw_m_s": v_raw, "omega_raw_rad_s": omega_raw}


def differential_track_speeds(v: float, omega: float, track_center_distance: float) -> tuple[float, float]:
  return (v - 0.5 * track_center_distance * omega,
          v + 0.5 * track_center_distance * omega)


def feasible_command(v_raw: float, omega_raw: float, config: dict[str, Any]) -> dict[str, float]:
  if not math.isfinite(v_raw) or not math.isfinite(omega_raw):
    raise ControllerError("raw commands must be finite")
  limits = config["command_limits"]
  spacing = float(config["measured_fixed"]["track_center_distance_m"])
  raw_left, raw_right = differential_track_speeds(v_raw, omega_raw, spacing)
  ratios = [1.0]
  for value, maximum in (
      (v_raw, float(limits["maximum_abs_body_speed_m_s"])),
      (omega_raw, float(limits["maximum_abs_yaw_rate_rad_s"])),
      (raw_left, float(limits["maximum_abs_track_surface_speed_m_s"])),
      (raw_right, float(limits["maximum_abs_track_surface_speed_m_s"]))):
    if abs(value) > 0.0:
      ratios.append(maximum / abs(value))
  scale = min(ratios)
  if scale <= 0.0:
    raise ControllerError("command scale is not positive")
  v_cmd, omega_cmd = scale * v_raw, scale * omega_raw
  left, right = differential_track_speeds(v_cmd, omega_cmd, spacing)
  return {"v_raw_m_s": v_raw, "omega_raw_rad_s": omega_raw,
          "v_cmd_m_s": v_cmd, "omega_cmd_rad_s": omega_cmd,
          "v_left_cmd_m_s": left, "v_right_cmd_m_s": right,
          "command_scale": scale}


def controller_command(actual: tuple[float, float, float], reference: dict[str, float],
                       config: dict[str, Any]) -> dict[str, float]:
  raw = raw_control(actual, reference, config)
  return {**raw, **feasible_command(raw["v_raw_m_s"], raw["omega_raw_rad_s"], config)}


def integrate_unicycle_exact(pose: tuple[float, float, float], v: float, omega: float,
                             dt: float, near_zero: float = 1e-12) -> tuple[float, float, float]:
  x, y, yaw = pose
  if dt <= 0.0 or not all(math.isfinite(value) for value in (x, y, yaw, v, omega, dt)):
    raise ControllerError("pose, command, and positive dt must be finite")
  yaw_next = yaw + omega * dt
  if abs(omega) <= near_zero:
    return x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, yaw_next
  return (x + (v / omega) * (math.sin(yaw_next) - math.sin(yaw)),
          y - (v / omega) * (math.cos(yaw_next) - math.cos(yaw)), yaw_next)


def project_to_polyline(x: float, y: float, reference_rows: list[dict[str, Any]]) -> dict[str, float]:
  if len(reference_rows) < 2:
    raise ControllerError("projection polyline requires at least two points")
  best: dict[str, float] | None = None
  for index in range(len(reference_rows) - 1):
    first, second = reference_rows[index], reference_rows[index + 1]
    dx, dy = second["x_m"] - first["x_m"], second["y_m"] - first["y_m"]
    length_squared = dx * dx + dy * dy
    if length_squared <= 0.0:
      continue
    fraction = max(0.0, min(1.0, ((x - first["x_m"]) * dx + (y - first["y_m"]) * dy) / length_squared))
    projected_x = first["x_m"] + fraction * dx
    projected_y = first["y_m"] + fraction * dy
    offset_x, offset_y = x - projected_x, y - projected_y
    distance_squared = offset_x * offset_x + offset_y * offset_y
    segment_length = math.sqrt(length_squared)
    candidate = {"projected_x_m": projected_x, "projected_y_m": projected_y,
                 "s_projected_m": first["s_m"] + fraction * (second["s_m"] - first["s_m"]),
                 "cross_track_error_m": (dx * offset_y - dy * offset_x) / segment_length,
                 "distance_squared": distance_squared, "projection_segment_index": index}
    if best is None or distance_squared < best["distance_squared"] - 1e-15:
      best = candidate
  if best is None:
    raise ControllerError("projection polyline contains no valid segment")
  best.pop("distance_squared")
  return best


CSV_COLUMNS = [
  "time_s", "reference_x_m", "reference_y_m", "reference_yaw_rad", "actual_x_m",
  "actual_y_m", "actual_yaw_rad", "reference_s_m", "projected_s_m", "projected_x_m",
  "projected_y_m", "e_x_m", "e_y_m", "heading_error_rad", "cross_track_error_m",
  "progress_error_m", "v_ref_m_s", "omega_ref_rad_s", "a_ref_m_s2", "v_raw_m_s",
  "omega_raw_rad_s", "v_cmd_m_s", "omega_cmd_rad_s", "v_actual_m_s",
  "omega_actual_rad_s", "speed_error_m_s", "yaw_rate_error_rad_s", "v_left_cmd_m_s",
  "v_right_cmd_m_s", "command_scale",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
