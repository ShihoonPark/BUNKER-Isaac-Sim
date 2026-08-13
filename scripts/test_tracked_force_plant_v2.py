#!/usr/bin/env python3
"""Run V2 directional track-force physics tests and order invariance check."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_tracked_force_plant_v2 import (  # noqa: E402
  SUPPORT_MODES, build_stage, load_config, patch_geometry,
)
from track_force_model_v2 import (  # noqa: E402
  TrackActuator, body_command_to_tracks, directional_contact_force,
  quaternion_wxyz_to_matrix, roll_pitch_yaw_wxyz,
)


def as_numpy(value: Any) -> np.ndarray:
  if hasattr(value, "numpy"):
    value = value.numpy()
  return np.asarray(value)


class DirectionalTrackController:
  """Contact-feedback controller called immediately before each physics step."""

  def __init__(self, rigid_view: Any, config: dict[str, Any], detailed: bool = False):
    self.rigid = rigid_view
    self.config = config
    self.fixed = config["measured_fixed"]
    self.tunable = config["tunable_uncalibrated"]
    self.actuator = TrackActuator(self.tunable)
    self.detailed = detailed
    self.v_cmd = self.omega_cmd = 0.0
    self.direct_track_command: tuple[float, float] | None = None
    self.time = 0.0
    self.rows: list[dict[str, Any]] = []
    self._centers = patch_geometry(config)[1]
    self._normal_force_warning_emitted = False
    self.custom_forces_enabled = True
    self._support_loss_steps = {"left": 0, "right": 0}

  def reset(self, command: dict[str, float], custom_forces_enabled: bool = True) -> None:
    self.actuator.reset()
    self.time = 0.0
    self.rows.clear()
    self.custom_forces_enabled = custom_forces_enabled
    self._support_loss_steps = {"left": 0, "right": 0}
    if "v_cmd_m_s" in command:
      self.v_cmd = float(command["v_cmd_m_s"])
      self.omega_cmd = float(command["omega_cmd_rad_s"])
      self.direct_track_command = None
    else:
      self.v_cmd = self.omega_cmd = 0.0
      self.direct_track_command = (
        float(command["v_left_cmd_m_s"]), float(command["v_right_cmd_m_s"]))

  def _commands(self) -> tuple[float, float]:
    if self.direct_track_command is not None:
      return self.direct_track_command
    return body_command_to_tracks(
      self.v_cmd, self.omega_cmd, self.fixed["track_center_distance_b_m"])

  def step(self, dt: float) -> None:
    left_cmd, right_cmd = self._commands()
    left_state, right_state = self.actuator.update(left_cmd, right_cmd, dt)
    positions, orientations = self.rigid.get_world_poses()
    position = as_numpy(positions)[0].astype(float)
    quaternion = as_numpy(orientations)[0].astype(float)
    rotation = quaternion_wxyz_to_matrix(quaternion)
    linear_world = as_numpy(self.rigid.get_linear_velocities())[0].astype(float)
    angular_world = as_numpy(self.rigid.get_angular_velocities())[0].astype(float)
    com_world = position + rotation @ np.asarray(self.fixed["imported_center_of_mass_m"], dtype=float)

    contact_details = []
    side_totals = {
      "left": {"count": 0, "normal": 0.0, "long": 0.0, "lat": 0.0},
      "right": {"count": 0, "normal": 0.0, "long": 0.0, "lat": 0.0},
    }
    yaw_moment = 0.0
    max_abs_force_normal_component = 0.0
    max_abs_legacy_force_normal_component = 0.0
    skipped_tangent_contacts = 0
    contact_data = self.rigid.get_contact_force_data(dt=dt)
    if contact_data is not None:
      normal_forces, points, normals, _distances, counts, starts = map(as_numpy, contact_data)
      count = int(counts[0, 0])
      start = int(starts[0, 0])
      contact_candidates = []
      for contact_index in range(start, start + count):
        point_world = points[contact_index].astype(float)
        normal_load = abs(float(np.asarray(normal_forces[contact_index]).reshape(-1)[0]))
        if normal_load <= 0.0:
          continue
        point_body = rotation.T @ (point_world - position)
        side = "left" if point_body[1] >= 0.0 else "right"
        contact_candidates.append((contact_index, point_world, point_body, side, normal_load))
      contacts_per_side = {
        side: sum(candidate[3] == side for candidate in contact_candidates)
        for side in ("left", "right")}
      for contact_index, point_world, point_body, side, normal_load in contact_candidates:
        totals = side_totals[side]
        totals["count"] += 1
        totals["normal"] += normal_load
        patch_index = min(range(len(self._centers)), key=lambda i: abs(point_body[0] - self._centers[i]))
        if not self.custom_forces_enabled:
          if self.detailed:
            contact_details.append({
              "side": side, "patch_index": patch_index,
              "point_world_m": point_world.tolist(), "point_body_m": point_body.tolist(),
              "normal_world": normals[contact_index].astype(float).tolist(),
              "normal_load_n": normal_load, "custom_force_disabled": True,
            })
          continue
        point_velocity_world = linear_world + np.cross(angular_world, point_world - com_world)
        normal_world = normals[contact_index].astype(float)
        normal_norm = float(np.linalg.norm(normal_world))
        if normal_norm < 1e-12:
          skipped_tangent_contacts += 1
          continue
        normal_world /= normal_norm
        body_forward_world = rotation[:, 0]
        body_left_world = rotation[:, 1]
        tangent_long_raw = (
          body_forward_world - np.dot(body_forward_world, normal_world) * normal_world)
        tangent_long_norm = float(np.linalg.norm(tangent_long_raw))
        if tangent_long_norm < 1e-12:
          skipped_tangent_contacts += 1
          continue
        tangent_long = tangent_long_raw / tangent_long_norm
        tangent_lat = np.cross(normal_world, tangent_long)
        projected_body_left = (
          body_left_world - np.dot(body_left_world, normal_world) * normal_world)
        projected_left_norm = float(np.linalg.norm(projected_body_left))
        if projected_left_norm < 1e-12:
          skipped_tangent_contacts += 1
          continue
        if np.dot(tangent_lat, projected_body_left) < 0.0:
          tangent_lat = -tangent_lat
        velocity_tangent = np.array([
          np.dot(point_velocity_world, tangent_long),
          np.dot(point_velocity_world, tangent_lat), 0.0], dtype=float)
        track_speed = left_state if side == "left" else right_state
        force_config = dict(self.tunable)
        if float(self.tunable["maximum_track_force_n"]) > 0.0:
          force_config["maximum_track_force_n"] = (
            float(self.tunable["maximum_track_force_n"]) / contacts_per_side[side])
        force_tangent = directional_contact_force(
          track_speed, velocity_tangent, normal_load, force_config)
        force_world = force_tangent[0] * tangent_long + force_tangent[1] * tangent_lat
        force_normal_component = float(np.dot(force_world, normal_world))
        legacy_force_normal_component = float(np.dot(rotation @ force_tangent, normal_world))
        max_abs_force_normal_component = max(
          max_abs_force_normal_component, abs(force_normal_component))
        max_abs_legacy_force_normal_component = max(
          max_abs_legacy_force_normal_component, abs(legacy_force_normal_component))
        if abs(force_normal_component) > 1e-7 and not self._normal_force_warning_emitted:
          warnings.warn(
            f"custom contact force has {force_normal_component:.6g} N normal component",
            RuntimeWarning, stacklevel=2)
          self._normal_force_warning_emitted = True
        self.rigid.apply_forces_and_torques_at_pos(
          forces=np.asarray([force_world], dtype=np.float32),
          positions=np.asarray([point_world], dtype=np.float32), is_global=True)
        moment_world = np.cross(point_world - com_world, force_world)
        yaw_moment += float(moment_world[2])
        totals["long"] += float(force_tangent[0])
        totals["lat"] += float(force_tangent[1])
        if self.detailed:
          contact_details.append({
            "side": side, "patch_index": patch_index,
            "point_world_m": point_world.tolist(), "point_body_m": point_body.tolist(),
            "normal_world": normal_world.tolist(),
            "tangent_long_world": tangent_long.tolist(), "tangent_lat_world": tangent_lat.tolist(),
            "normal_load_n": normal_load, "point_velocity_tangent_m_s": velocity_tangent.tolist(),
            "force_tangent_n": force_tangent.tolist(), "force_world_n": force_world.tolist(),
            "force_normal_component_n": force_normal_component,
            "legacy_body_plane_force_normal_component_n": legacy_force_normal_component,
          })

    body_linear = rotation.T @ linear_world
    body_angular = rotation.T @ angular_world
    roll, pitch, yaw = roll_pitch_yaw_wxyz(quaternion)
    threshold = float(self.tunable["support_load_threshold_n"])
    supported = {
      side: side_totals[side]["normal"] > threshold for side in ("left", "right")}
    for side in ("left", "right"):
      self._support_loss_steps[side] = 0 if supported[side] else self._support_loss_steps[side] + 1
    self.time += dt
    self.rows.append({
      "time_s": self.time, "v_cmd_m_s": self.v_cmd, "omega_cmd_rad_s": self.omega_cmd,
      "v_left_cmd_m_s": left_cmd, "v_right_cmd_m_s": right_cmd,
      "v_left_state_m_s": left_state, "v_right_state_m_s": right_state,
      "left_contact_count": side_totals["left"]["count"],
      "right_contact_count": side_totals["right"]["count"],
      "left_normal_load_n": side_totals["left"]["normal"],
      "right_normal_load_n": side_totals["right"]["normal"],
      "total_normal_load_n": side_totals["left"]["normal"] + side_totals["right"]["normal"],
      "left_supported": int(supported["left"]), "right_supported": int(supported["right"]),
      "left_consecutive_support_loss_steps": self._support_loss_steps["left"],
      "right_consecutive_support_loss_steps": self._support_loss_steps["right"],
      "left_longitudinal_force_n": side_totals["left"]["long"],
      "right_longitudinal_force_n": side_totals["right"]["long"],
      "left_lateral_force_n": side_totals["left"]["lat"],
      "right_lateral_force_n": side_totals["right"]["lat"],
      "applied_yaw_moment_n_m": yaw_moment,
      "max_abs_force_normal_component_n": max_abs_force_normal_component,
      "max_abs_legacy_body_plane_force_normal_component_n": max_abs_legacy_force_normal_component,
      "skipped_tangent_contact_count": skipped_tangent_contacts,
      "x_m": position[0], "y_m": position[1], "z_m": position[2],
      "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw,
      "body_vx_m_s": body_linear[0], "body_vy_m_s": body_linear[1], "body_vz_m_s": body_linear[2],
      "world_vz_m_s": linear_world[2],
      "body_wx_rad_s": body_angular[0], "body_wy_rad_s": body_angular[1],
      "body_wz_rad_s": body_angular[2],
      "contacts_json": json.dumps(contact_details, separators=(",", ":")) if self.detailed else "",
    })


def criterion(name: str, dx: float, dy: float, dyaw: float, finite: bool) -> tuple[bool, str]:
  if not finite:
    return False, "finite state required"
  if name == "settle_only":
    return True, "finite passive normal-support simulation"
  if name == "straight":
    return dx > 0 and abs(dy) < 0.05 and abs(dyaw) < 0.05, "forward with |dy|,|dyaw| < 0.05"
  expected = {"left_only": -1, "right_only": 1, "positive_in_place": 1,
              "negative_in_place": -1, "positive_curve": 1, "negative_curve": -1}[name]
  if "in_place" in name:
    return dyaw * expected > 0.30 and math.hypot(dx, dy) < 0.50, "correct yaw > 0.30 rad and limited translation"
  if "curve" in name:
    return dx > 0 and dyaw * expected > 0.05, "forward translation and correct yaw sign"
  return dyaw * expected > 0.01, "one-sided commands require opposite expected yaw signs"


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


def non_negative_float(value: str) -> float:
  number = float(value)
  if not math.isfinite(number) or number < 0.0:
    raise argparse.ArgumentTypeError("must be a non-negative finite number")
  return number


def step_world(world: Any, render: bool, realtime: bool, dt: float) -> None:
  step_started = time.perf_counter()
  world.step(render=render)
  if realtime:
    remaining = dt - (time.perf_counter() - step_started)
    if remaining > 0.0:
      time.sleep(remaining)


def transient_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
  """Summarize vertical attitude and contact/load transients from step logs."""
  times = np.asarray([float(row["time_s"]) for row in rows])
  left_loads = np.asarray([float(row["left_normal_load_n"]) for row in rows])
  right_loads = np.asarray([float(row["right_normal_load_n"]) for row in rows])
  left_counts = np.asarray([int(row["left_contact_count"]) for row in rows])
  right_counts = np.asarray([int(row["right_contact_count"]) for row in rows])
  z_values = np.asarray([float(row["z_m"]) for row in rows])
  roll_values = np.unwrap([float(row["roll_rad"]) for row in rows])
  pitch_values = np.unwrap([float(row["pitch_rad"]) for row in rows])
  vertical_velocities = np.asarray([float(row["world_vz_m_s"]) for row in rows])
  supported = {
    side: np.asarray([bool(int(row[f"{side}_supported"])) for row in rows])
    for side in ("left", "right")}
  total_loads = left_loads + right_loads

  def max_step_change(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.diff(values)))) if len(values) > 1 else 0.0

  def count_changes(values: np.ndarray) -> list[dict[str, Any]]:
    return [
      {"time_s": float(times[index]), "from": int(values[index - 1]), "to": int(values[index])}
      for index in range(1, len(values)) if values[index] != values[index - 1]]

  def longest_loss(values: np.ndarray) -> dict[str, Any]:
    best_start = best_length = current_start = current_length = 0
    for index, value in enumerate(values):
      if not value:
        if current_length == 0:
          current_start = index
        current_length += 1
        if current_length > best_length:
          best_start, best_length = current_start, current_length
      else:
        current_length = 0
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    return {
      "steps": best_length, "duration_s": best_length * dt,
      "start_time_s": float(times[best_start]) if best_length else None,
    }

  yaws = np.unwrap([float(row["yaw_rad"]) for row in rows])
  any_loss = ~(supported["left"] & supported["right"])
  loss_indices = np.flatnonzero(any_loss)
  if len(loss_indices):
    loss_start = int(loss_indices[0])
    reacquired = np.flatnonzero(~any_loss[loss_start + 1:])
    reacquire_index = loss_start + 1 + int(reacquired[0]) if len(reacquired) else len(rows) - 1
    yaw_phases = {
      "support_loss_detected": True,
      "first_loss_time_s": float(times[loss_start]),
      "reacquisition_time_s": float(times[reacquire_index]) if len(reacquired) else None,
      "before_first_loss_rad": float(yaws[loss_start] - yaws[0]),
      "during_loss_rad": float(yaws[reacquire_index] - yaws[loss_start]),
      "after_reacquisition_rad": (
        float(yaws[-1] - yaws[reacquire_index]) if len(reacquired) else 0.0),
    }
  else:
    yaw_phases = {
      "support_loss_detected": False, "first_loss_time_s": None,
      "reacquisition_time_s": None, "before_first_loss_rad": float(yaws[-1] - yaws[0]),
      "during_loss_rad": 0.0, "after_reacquisition_rad": 0.0,
    }

  total_load_change = np.abs(np.diff(total_loads))
  transient_index = int(np.argmax(total_load_change)) + 1 if len(total_load_change) else 0
  return {
    "maximum_abs_custom_force_normal_component_n": max(
      float(row["max_abs_force_normal_component_n"]) for row in rows),
    "maximum_abs_legacy_body_plane_force_normal_component_n": max(
      float(row["max_abs_legacy_body_plane_force_normal_component_n"]) for row in rows),
    "maximum_left_normal_load_n": float(np.max(left_loads)),
    "maximum_right_normal_load_n": float(np.max(right_loads)),
    "maximum_one_step_left_normal_load_change_n": max_step_change(left_loads),
    "maximum_one_step_right_normal_load_change_n": max_step_change(right_loads),
    "maximum_total_normal_load_n": float(np.max(total_loads)),
    "maximum_one_step_total_normal_load_change_n": max_step_change(total_loads),
    "longest_left_support_loss": longest_loss(supported["left"]),
    "longest_right_support_loss": longest_loss(supported["right"]),
    "left_supported_state_changes": count_changes(supported["left"].astype(int)),
    "right_supported_state_changes": count_changes(supported["right"].astype(int)),
    "left_contact_count_range": [int(np.min(left_counts)), int(np.max(left_counts))],
    "right_contact_count_range": [int(np.min(right_counts)), int(np.max(right_counts))],
    "left_contact_count_changes": count_changes(left_counts),
    "right_contact_count_changes": count_changes(right_counts),
    "maximum_one_step_body_z_change_m": max_step_change(z_values),
    "maximum_one_step_roll_change_rad": max_step_change(roll_values),
    "maximum_one_step_pitch_change_rad": max_step_change(pitch_values),
    "maximum_abs_vertical_body_velocity_m_s": float(np.max(np.abs(vertical_velocities))),
    "body_z_drift_m": float(z_values[-1] - z_values[0]),
    "maximum_abs_roll_rad": float(np.max(np.abs(roll_values))),
    "maximum_abs_pitch_rad": float(np.max(np.abs(pitch_values))),
    "final_roll_rad": float(roll_values[-1]),
    "final_pitch_rad": float(pitch_values[-1]),
    "final_z_m": float(z_values[-1]),
    "yaw_accumulation_by_support_phase": yaw_phases,
    "largest_suspected_transient": {
      "basis": "largest one-step change in total reported normal load",
      "time_s": float(times[transient_index]),
      "total_normal_load_change_n": (
        float(total_load_change[transient_index - 1]) if transient_index else 0.0),
    },
  }


def run_case(world: Any, controller: DirectionalTrackController, config: dict[str, Any],
             name: str, output_dir: Path, render: bool = False, realtime: bool = False,
             log_name: str | None = None) -> dict[str, Any]:
  dt = float(config["tunable_uncalibrated"]["physics_dt_s"])
  world.play()
  if not world.is_playing():
    raise RuntimeError("simulation timeline did not enter PLAY state")
  test_config = config["tests"][name]
  custom_forces_enabled = not bool(test_config.get("disable_custom_forces", False))
  if name != "settle_only":
    controller.reset({"v_cmd_m_s": 0.0, "omega_cmd_rad_s": 0.0})
    for _ in range(round(config["tunable_uncalibrated"]["settle_duration_s"] / dt)):
      step_world(world, render, realtime, dt)
  controller.reset(test_config, custom_forces_enabled=custom_forces_enabled)
  duration = float(test_config.get(
    "duration_s", config["tunable_uncalibrated"]["test_duration_s"]))
  for _ in range(round(duration / dt)):
    step_world(world, render, realtime, dt)
  if not controller.rows:
    raise RuntimeError(
      "no physics callback samples were recorded; simulation timeline may not be playing")
  first, last = controller.rows[0], controller.rows[-1]
  unwrapped_yaw = np.unwrap([float(row["yaw_rad"]) for row in controller.rows])
  dyaw = float(unwrapped_yaw[-1] - unwrapped_yaw[0])
  dx = float(last["x_m"] - first["x_m"])
  dy = float(last["y_m"] - first["y_m"])
  finite = all(math.isfinite(float(row[key])) for row in controller.rows for key in (
    "x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad",
    "body_vx_m_s", "body_vy_m_s", "body_vz_m_s"))
  passed, explanation = criterion(name, dx, dy, dyaw, finite)
  csv_path = output_dir / f"{log_name or name}.csv"
  write_rows(csv_path, controller.rows)
  diagnostics = transient_diagnostics(controller.rows)
  return {"name": name, "passed": bool(passed), "criterion": explanation,
          "delta_x_m": dx, "delta_y_m": dy, "delta_yaw_rad": dyaw,
          "final_body_linear_velocity_m_s": [last["body_vx_m_s"], last["body_vy_m_s"], last["body_vz_m_s"]],
          "final_body_angular_velocity_rad_s": [last["body_wx_rad_s"], last["body_wy_rad_s"], last["body_wz_rad_s"]],
          "final_contact_counts": [last["left_contact_count"], last["right_contact_count"]],
          "final_normal_loads_n": [last["left_normal_load_n"], last["right_normal_load_n"]],
          "final_roll_rad": last["roll_rad"], "final_pitch_rad": last["pitch_rad"],
          "final_z_m": last["z_m"],
          "transient_diagnostics": diagnostics,
          "log_csv": str(csv_path)}


def make_world(config: dict[str, Any], usd_path: Path, creation_order: str, detailed: bool,
               support_mode: str):
  import omni.usd
  from isaacsim.core.api import World
  from isaacsim.core.prims import RigidPrim

  World.clear_instance()
  build_info = build_stage(config, usd_path, creation_order, support_mode)
  omni.usd.get_context().open_stage(str(usd_path))
  dt = config["tunable_uncalibrated"]["physics_dt_s"]
  world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0)
  rigid = world.scene.add(RigidPrim(
    prim_paths_expr="/World/BunkerForcePlant", name=f"bunker_force_{creation_order}",
    contact_filter_prim_paths_expr=["/World/Ground"],
    max_contact_count=config["tunable_uncalibrated"]["max_contact_count"]))
  world.reset()
  controller = DirectionalTrackController(rigid, config, detailed)
  world.add_physics_callback(f"track_force_{creation_order}", controller.step)
  world.play()
  if not world.is_playing():
    raise RuntimeError("simulation timeline did not enter PLAY state")
  return world, controller, build_info


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/tracked_force_plant_v2.json")
  parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs/tracked_force_plant_v2")
  parser.add_argument("--case", choices=("straight", "left_only", "right_only", "positive_in_place",
                                         "negative_in_place", "positive_curve", "negative_curve",
                                         "settle_only", "all"), default="all")
  parser.add_argument(
    "--support-mode", choices=SUPPORT_MODES,
    help="physical normal-support geometry (default from config)")
  parser.add_argument("--detailed-contacts", action="store_true")
  parser.add_argument(
    "--skip-order-invariance", action="store_true",
    help="run only the selected case and skip the authoring-order invariance test")
  parser.add_argument("--gui", action="store_true", help="open the Isaac Sim GUI and render simulation steps")
  parser.add_argument("--realtime", action="store_true", help="pace GUI physics steps approximately to wall-clock time")
  parser.add_argument(
    "--hold-open-seconds", type=non_negative_float, default=None, metavar="SECONDS",
    help="keep a paused GUI responsive after the motion test (default with --gui: 3.0)")
  args = parser.parse_args()
  if args.realtime and not args.gui:
    parser.error("--realtime requires --gui")
  if args.skip_order_invariance and args.case == "all":
    parser.error("--skip-order-invariance requires selecting one case with --case")
  hold_open_seconds = 3.0 if args.gui and args.hold_open_seconds is None else (args.hold_open_seconds or 0.0)
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  from isaacsim import SimulationApp
  app = SimulationApp({"headless": not args.gui, "enable_cameras": args.gui})
  try:
    config = load_config(args.config.resolve())
    support_mode = args.support_mode or config["tunable_uncalibrated"]["support_mode"]
    dt = float(config["tunable_uncalibrated"]["physics_dt_s"])
    print(
      f"startup: case={args.case} gui={args.gui} realtime={args.realtime} "
      f"physics_dt_s={dt:.12g} hold_open_seconds={hold_open_seconds:g} "
      f"support_mode={support_mode}")
    names = (tuple(name for name in config["tests"] if name != "settle_only")
             if args.case == "all" else (args.case,))
    world, controller, build_info = make_world(
      config, output_dir / f"tracked_force_plant_v2_{support_mode}_normal.usd",
      "normal", args.detailed_contacts, support_mode)
    results = []
    for index, name in enumerate(names):
      if index:
        world.reset()
      result = run_case(
        world, controller, config, name, output_dir, render=args.gui, realtime=args.realtime)
      results.append(result)
      print(json.dumps(result, indent=2))
    world.remove_physics_callback("track_force_normal")
    if args.gui and hold_open_seconds > 0.0:
      world.pause()
      hold_deadline = time.perf_counter() + hold_open_seconds
      while app.is_running() and time.perf_counter() < hold_deadline:
        update_started = time.perf_counter()
        app.update()
        remaining = dt - (time.perf_counter() - update_started)
        if remaining > 0.0:
          time.sleep(remaining)

    reverse_build = None
    if args.skip_order_invariance:
      invariance_passed = True
      invariance = {
        "skipped": True,
        "reason": "--skip-order-invariance requested; only the selected case was run"}
      print("authoring-order invariance: skipped (--skip-order-invariance)")
    else:
      reverse_world, reverse_controller, reverse_build = make_world(
        config, output_dir / f"tracked_force_plant_v2_{support_mode}_reversed.usd",
        "reversed", args.detailed_contacts, support_mode)
      reverse_result = run_case(
        reverse_world, reverse_controller, config, "straight", output_dir,
        render=args.gui, realtime=args.realtime, log_name="order_reversed_straight")
      normal_straight = next((result for result in results if result["name"] == "straight"), None)
      if normal_straight is None:
        reverse_world.remove_physics_callback("track_force_reversed")
        normal_world, normal_controller, _ = make_world(
          config, output_dir / f"tracked_force_plant_v2_{support_mode}_normal.usd",
          "normal", args.detailed_contacts, support_mode)
        normal_straight = run_case(
          normal_world, normal_controller, config, "straight", output_dir,
          render=args.gui, realtime=args.realtime, log_name="order_normal_straight")
      pose_difference = {
        "delta_x_m": reverse_result["delta_x_m"] - normal_straight["delta_x_m"],
        "delta_y_m": reverse_result["delta_y_m"] - normal_straight["delta_y_m"],
        "delta_yaw_rad": reverse_result["delta_yaw_rad"] - normal_straight["delta_yaw_rad"],
      }
      invariance_passed = all(abs(value) < 0.02 for value in pose_difference.values())
      invariance = {"skipped": False, "passed": invariance_passed, "tolerance": 0.02,
                    "normal": normal_straight, "reversed": reverse_result,
                    "reversed_minus_normal": pose_difference}
    summary = {"contact_data_units": "get_contact_force_data(dt=physics_dt) converts PhysX impulses to newtons",
               "contact_count_interpretation": "PhysX contact-manifold point count, not physical roller count",
               "support_mode": support_mode,
               "normal_build": build_info, "reversed_build": reverse_build,
               "tests": results, "authoring_order_invariance": invariance,
               "all_motion_tests_passed": all(result["passed"] for result in results)}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
      json.dump(summary, stream, indent=2)
      stream.write("\n")
    print(json.dumps({"authoring_order_invariance": invariance}, indent=2))
    return 0 if summary["all_motion_tests_passed"] and invariance_passed else 2
  except BaseException:
    traceback.print_exc()
    raise
  finally:
    app.close()


if __name__ == "__main__":
  raise SystemExit(main())
