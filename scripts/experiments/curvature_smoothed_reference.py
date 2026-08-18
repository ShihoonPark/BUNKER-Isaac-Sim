#!/usr/bin/env python3
"""Isolated curvature-continuous reference for one Stage C diagnostic."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from reference_trajectory_v1 import (  # noqa: E402
  load_config as load_canonical_config, parameterize_trajectory,
)


def load_experiment_config(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    config = json.load(stream)
  smoothing = config["smoothing"]
  if smoothing["method"] != "symmetric_linear_curvature_ramp":
    raise ValueError("unsupported smoothing method")
  if not smoothing["preserve_peak_curvature"] or not smoothing["preserve_global_start_goal"]:
    raise ValueError("this diagnostic requires peak-curvature and endpoint preservation")
  heading = float(smoothing["transition_heading_per_ramp_rad"])
  step = float(smoothing["integration_step_m"])
  if not (0.0 < heading < math.pi / 4.0) or step <= 0.0:
    raise ValueError("invalid smoothing heading or integration step")
  return config


def _rk4_step(x: float, y: float, yaw: float, s0: float, h: float,
              curvature: Callable[[float], float]) -> tuple[float, float, float]:
  def derivative(theta: float, s_value: float) -> tuple[float, float, float]:
    return math.cos(theta), math.sin(theta), curvature(s_value)
  k1 = derivative(yaw, s0)
  k2 = derivative(yaw + 0.5 * h * k1[2], s0 + 0.5 * h)
  k3 = derivative(yaw + 0.5 * h * k2[2], s0 + 0.5 * h)
  k4 = derivative(yaw + h * k3[2], s0 + h)
  return (x + h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0,
          y + h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0,
          yaw + h * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0)


def _corner_profile(radius: float, sign: float, ramp_heading: float
                    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
  peak = sign / radius
  ramp = 2.0 * radius * ramp_heading
  constant = radius * (math.pi / 2.0 - 2.0 * ramp_heading)
  parts = [
    {"kind": "ramp_in", "length_m": ramp,
     "curvature": lambda u, length=ramp, value=peak: value * u / length,
     "k_start": 0.0, "k_end": peak},
    {"kind": "constant", "length_m": constant,
     "curvature": lambda _u, value=peak: value,
     "k_start": peak, "k_end": peak},
    {"kind": "ramp_out", "length_m": ramp,
     "curvature": lambda u, length=ramp, value=peak: value * (1.0 - u / length),
     "k_start": peak, "k_end": 0.0},
  ]
  return parts, {"radius_m": radius, "peak_curvature_1_m": peak,
                 "ramp_length_m": ramp, "constant_length_m": constant,
                 "total_length_m": 2.0 * ramp + constant}


def _integrate_displacement(parts: list[dict[str, Any]], integration_step: float
                            ) -> tuple[float, float, float]:
  x = y = yaw = 0.0
  for part in parts:
    length = float(part["length_m"])
    steps = max(1, math.ceil(length / integration_step))
    h = length / steps
    for index in range(steps):
      x, y, yaw = _rk4_step(x, y, yaw, index * h, h, part["curvature"])
  return x, y, yaw


def _append_part(rows: list[dict[str, Any]], boundary_rows: list[dict[str, Any]],
                 x: float, y: float, yaw: float, s_global: float, part: dict[str, Any],
                 dense_step: float, segment_index: int, segment_type: str
                 ) -> tuple[float, float, float, float]:
  length = float(part["length_m"])
  steps = max(1, math.ceil(length / dense_step))
  h = length / steps
  start_s = s_global
  for index in range(steps):
    local_s = index * h
    x, y, yaw = _rk4_step(x, y, yaw, local_s, h, part["curvature"])
    s_global += h
    rows.append({"dense_index": len(rows), "s_m": s_global, "x_m": x, "y_m": y,
                 "yaw_analytic_rad": yaw, "segment_index": segment_index,
                 "segment_type": segment_type,
                 "curvature_ref_1_m": float(part["curvature"]((index + 1) * h))})
  boundary_rows.append({"segment_index": segment_index, "segment_type": segment_type,
                        "profile_part": part["kind"], "s_start_m": start_s,
                        "s_end_m": s_global, "length_m": length,
                        "curvature_start_1_m": float(part["k_start"]),
                        "curvature_end_1_m": float(part["k_end"])})
  return x, y, yaw, s_global


def _line_part(length: float) -> dict[str, Any]:
  return {"kind": "line", "length_m": length, "curvature": lambda _u: 0.0,
          "k_start": 0.0, "k_end": 0.0}


def _resample(dense: list[dict[str, Any]], interval: float) -> list[dict[str, Any]]:
  total = float(dense[-1]["s_m"])
  targets = [index * interval for index in range(math.floor(total / interval) + 1)]
  if total - targets[-1] > 1e-12:
    targets.append(total)
  else:
    targets[-1] = total
  rows: list[dict[str, Any]] = []
  source = 0
  for index, target in enumerate(targets):
    while source + 1 < len(dense) and float(dense[source + 1]["s_m"]) < target:
      source += 1
    left, right = dense[source], dense[min(source + 1, len(dense) - 1)]
    span = float(right["s_m"]) - float(left["s_m"])
    fraction = 0.0 if span == 0.0 else (target - float(left["s_m"])) / span
    rows.append({"index": index, "s_m": target,
                 "x_m": float(left["x_m"]) + fraction * (float(right["x_m"]) - float(left["x_m"])),
                 "y_m": float(left["y_m"]) + fraction * (float(right["y_m"]) - float(left["y_m"])),
                 "yaw_rad": float(left["yaw_analytic_rad"]) + fraction * (
                   float(right["yaw_analytic_rad"]) - float(left["yaw_analytic_rad"])),
                 "curvature_ref_1_m": float(left["curvature_ref_1_m"]) + fraction * (
                   float(right["curvature_ref_1_m"]) - float(left["curvature_ref_1_m"])),
                 "segment_index": int(left["segment_index"]),
                 "segment_type": str(left["segment_type"])})
  headings = []
  for index in range(len(rows)):
    before, after = max(0, index - 1), min(len(rows) - 1, index + 1)
    headings.append(math.atan2(rows[after]["y_m"] - rows[before]["y_m"],
                               rows[after]["x_m"] - rows[before]["x_m"]))
  unwrapped = [headings[0]]
  for heading in headings[1:]:
    unwrapped.append(unwrapped[-1] + (heading - unwrapped[-1] + math.pi) % (2 * math.pi) - math.pi)
  for index, row in enumerate(rows):
    before, after = max(0, index - 1), min(len(rows) - 1, index + 1)
    row["curvature_numeric_1_m"] = ((unwrapped[after] - unwrapped[before]) /
                                      (rows[after]["s_m"] - rows[before]["s_m"]))
  return rows


def build_smoothed_reference(experiment_config_path: Path) -> dict[str, Any]:
  experiment = load_experiment_config(experiment_config_path)
  canonical_path = REPO_ROOT / experiment["canonical_reference_config"]
  canonical = load_canonical_config(canonical_path)
  segments = canonical["path"]["segments"]
  if [segment["type"] for segment in segments] != ["line", "arc", "line", "arc", "line"]:
    raise ValueError("experiment expects the canonical V1 line/arc topology")
  ramp_heading = float(experiment["smoothing"]["transition_heading_per_ramp_rad"])
  integration_step = float(experiment["smoothing"]["integration_step_m"])
  corners = []
  corner_parts = []
  for segment in (segments[1], segments[3]):
    sign = math.copysign(1.0, float(segment["angle_deg"]))
    parts, metadata = _corner_profile(float(segment["radius_m"]), sign, ramp_heading)
    dx, dy, dyaw = _integrate_displacement(parts, integration_step)
    local_outgoing = sign * dy
    if abs(dx - local_outgoing) > 1e-10:
      raise ValueError("smooth 90-degree corner tangent displacements are not symmetric")
    metadata.update({"local_tangent_displacement_m": 0.5 * (dx + local_outgoing),
                     "local_dx_m": dx, "local_dy_m": dy, "heading_change_rad": dyaw,
                     "delta_straight_consumption_m": 0.5 * (dx + local_outgoing) - metadata["radius_m"]})
    corners.append(metadata); corner_parts.append(parts)
  d1, d2 = (corner["delta_straight_consumption_m"] for corner in corners)
  straight_lengths = [float(segments[0]["length_m"]) - d1,
                      float(segments[2]["length_m"]) - d1 - d2,
                      float(segments[4]["length_m"]) - d2]
  if min(straight_lengths) <= 0.0:
    raise ValueError("endpoint preservation consumed an entire adjacent straight")
  profile = [(_line_part(straight_lengths[0]), 0, "line")]
  profile += [(part, 1, f"corner_1_{part['kind']}") for part in corner_parts[0]]
  profile += [(_line_part(straight_lengths[1]), 2, "line")]
  profile += [(part, 3, f"corner_2_{part['kind']}") for part in corner_parts[1]]
  profile += [(_line_part(straight_lengths[2]), 4, "line")]
  start = canonical["path"]["start_pose"]
  x, y, yaw = float(start["x_m"]), float(start["y_m"]), float(start["yaw_rad"])
  rows = [{"dense_index": 0, "s_m": 0.0, "x_m": x, "y_m": y,
           "yaw_analytic_rad": yaw, "segment_index": 0, "segment_type": "line",
           "curvature_ref_1_m": 0.0}]
  boundaries: list[dict[str, Any]] = []
  s_global = 0.0
  for part, segment_index, segment_type in profile:
    x, y, yaw, s_global = _append_part(
      rows, boundaries, x, y, yaw, s_global, part,
      float(canonical["path"]["dense_sampling_step_m"]), segment_index, segment_type)
  resampled = _resample(rows, float(canonical["path"]["resampling_interval_m"]))
  trajectory = parameterize_trajectory(resampled, canonical)
  return {"dense": rows, "resampled": resampled, "trajectory": trajectory,
          "boundaries": boundaries, "corners": corners,
          "adjusted_straight_lengths_m": straight_lengths,
          "canonical_config": canonical, "experiment_config": experiment,
          "summary": {"path_length_m": trajectory[-1]["s_m"],
                      "total_duration_s": trajectory[-1]["t_s"],
                      "final_pose": {"x_m": x, "y_m": y, "yaw_rad": yaw}}}
