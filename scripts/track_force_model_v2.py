#!/usr/bin/env python3
"""Pure numerical state and directional force law for tracked-force V2."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


def clamp(value: float, lower: float, upper: float) -> float:
  return max(lower, min(upper, value))


def body_command_to_tracks(v_cmd: float, omega_cmd: float, track_center_distance: float) -> tuple[float, float]:
  return (v_cmd - 0.5 * track_center_distance * omega_cmd,
          v_cmd + 0.5 * track_center_distance * omega_cmd)


@dataclass
class TrackActuator:
  """Uncalibrated virtual-track actuator state with optional nonlinearities."""

  config: dict[str, Any]
  left_speed: float = 0.0
  right_speed: float = 0.0

  def __post_init__(self) -> None:
    self._delay = deque([(0.0, 0.0)])
    self._elapsed = 0.0

  def reset(self) -> None:
    self.left_speed = self.right_speed = self._elapsed = 0.0
    self._delay.clear()
    self._delay.append((0.0, 0.0))

  def update(self, left_cmd: float, right_cmd: float, dt: float) -> tuple[float, float]:
    maximum = float(self.config["maximum_surface_speed_m_s"])
    deadzone = float(self.config["deadzone_m_s"])
    commands = [clamp(left_cmd, -maximum, maximum), clamp(right_cmd, -maximum, maximum)]
    commands = [0.0 if abs(value) < deadzone else value for value in commands]
    delay = float(self.config["command_delay_s"])
    self._elapsed += dt
    self._delay.append((self._elapsed, commands))
    delayed = self._delay[0][1]
    while len(self._delay) > 1 and self._delay[1][0] <= self._elapsed - delay:
      self._delay.popleft()
      delayed = self._delay[0][1]
    tau = float(self.config["actuator_time_constant_s"])
    alpha = 1.0 if tau <= 0.0 else 1.0 - math.exp(-dt / tau)
    limit = float(self.config["acceleration_limit_m_s2"])
    states = [self.left_speed, self.right_speed]
    for index in range(2):
      delta = alpha * (delayed[index] - states[index])
      if limit > 0.0:
        delta = clamp(delta, -limit * dt, limit * dt)
      states[index] += delta
    self.left_speed, self.right_speed = states
    return self.left_speed, self.right_speed


def directional_contact_force(track_speed: float, point_velocity_tangent: np.ndarray,
                              normal_load: float, config: dict[str, Any]) -> np.ndarray:
  """Return [longitudinal, lateral, 0] components in a contact-tangent basis."""
  long_limit = float(config["mu_long"]) * normal_load
  lat_limit = float(config["mu_lat"]) * normal_load
  force_long = clamp(float(config["k_long_n_per_m_s"]) *
                     (track_speed - float(point_velocity_tangent[0])), -long_limit, long_limit)
  force_lat = clamp(-float(config["k_lat_n_per_m_s"]) * float(point_velocity_tangent[1]),
                    -lat_limit, lat_limit)
  maximum = float(config["maximum_track_force_n"])
  result = np.array([force_long, force_lat, 0.0], dtype=float)
  magnitude = float(np.linalg.norm(result[:2]))
  if maximum > 0.0 and magnitude > maximum:
    result *= maximum / magnitude
  return result


def quaternion_wxyz_to_matrix(quaternion: Any) -> np.ndarray:
  w, x, y, z = (float(value) for value in quaternion)
  return np.array([
    [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
    [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
    [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
  ], dtype=float)


def roll_pitch_yaw_wxyz(quaternion: Any) -> tuple[float, float, float]:
  matrix = quaternion_wxyz_to_matrix(quaternion)
  pitch = math.asin(clamp(-matrix[2, 0], -1.0, 1.0))
  roll = math.atan2(matrix[2, 1], matrix[2, 2])
  yaw = math.atan2(matrix[1, 0], matrix[0, 0])
  return roll, pitch, yaw
