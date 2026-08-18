#!/usr/bin/env python3
"""Minimal direction-aware lateral-feedback extension of Tracking Controller V1."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tracking_controller_v1 import body_frame_errors, feasible_command, load_controller_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class DirectionAwareControllerError(ValueError):
  """Raised for invalid V2 configuration or maneuver metadata."""


def load_direction_aware_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
  with path.open(encoding="utf-8") as stream:
    config = json.load(stream)
  policy = config["motion_direction_policy"]
  expected = {"forward": (1, 1), "reverse": (-1, -1), "pivot": (0, 1)}
  for name, values in expected.items():
    row = policy[name]
    actual = int(row["motion_direction"]), int(row["lateral_feedback_factor"])
    if actual != values:
      raise DirectionAwareControllerError(f"invalid {name} motion-direction policy: {actual}")
  baseline_path = REPO_ROOT / config["baseline_controller_config"]
  return config, load_controller_config(baseline_path)


def lateral_feedback_factor(motion_direction: int) -> int:
  """Return the explicit V2 mode policy; pivot deliberately retains V1 behavior."""
  if motion_direction == 1:
    return 1
  if motion_direction == -1:
    return -1
  if motion_direction == 0:
    return 1
  raise DirectionAwareControllerError(
    f"motion_direction must be explicit +1 forward, -1 reverse, or 0 pivot; got {motion_direction}")


def raw_control(actual: tuple[float, float, float], reference: dict[str, float],
                motion_direction: int, controller_config: dict[str, Any]) -> dict[str, float]:
  errors = body_frame_errors(actual, reference)
  gains = controller_config["controller"]
  heading = errors["e_heading_rad"]
  factor = lateral_feedback_factor(motion_direction)
  longitudinal = float(gains["k_longitudinal_1_s"]) * errors["e_x_m"]
  lateral = factor * float(gains["k_lateral_rad_s_per_m"]) * errors["e_y_m"]
  heading_feedback = float(gains["k_heading_1_s"]) * math.sin(heading)
  v_raw = reference["v_ref_m_s"] * math.cos(heading) + longitudinal
  omega_raw = reference["omega_ref_rad_s"] + lateral + heading_feedback
  return {**errors, "motion_direction": motion_direction,
          "lateral_feedback_factor": factor,
          "longitudinal_feedback_m_s": longitudinal,
          "lateral_feedback_rad_s": lateral,
          "heading_feedback_rad_s": heading_feedback,
          "v_raw_m_s": v_raw, "omega_raw_rad_s": omega_raw}


def controller_command(actual: tuple[float, float, float], reference: dict[str, float],
                       motion_direction: int,
                       controller_config: dict[str, Any]) -> dict[str, float]:
  raw = raw_control(actual, reference, motion_direction, controller_config)
  feasible = feasible_command(raw["v_raw_m_s"], raw["omega_raw_rad_s"], controller_config)
  return {**raw, **feasible}
