#!/usr/bin/env python3
"""Run Stage C1 nominal closed-loop tracking through the Isaac Sim V2 plant."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from reference_trajectory_v1 import build_reference_trajectory, load_config as load_reference_config  # noqa: E402
from test_tracked_force_plant_v2 import as_numpy, load_config as load_plant_config, make_world, step_world  # noqa: E402
from track_force_model_v2 import quaternion_wxyz_to_matrix, roll_pitch_yaw_wxyz  # noqa: E402
from tracking_controller_v1 import (  # noqa: E402
  ReferenceInterpolator, body_frame_errors, controller_command, differential_track_speeds,
  load_controller_config, project_to_polyline, wrap_to_pi,
)


def load_stage_config(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    return json.load(stream)


def world_to_settled_local_xy(x_world: float, y_world: float, x0_world: float,
                              y0_world: float, yaw0_world: float) -> tuple[float, float]:
  dx, dy = x_world - x0_world, y_world - y0_world
  return (math.cos(yaw0_world) * dx + math.sin(yaw0_world) * dy,
          -math.sin(yaw0_world) * dx + math.cos(yaw0_world) * dy)


def continuous_angle_update(previous_unwrapped: float, new_wrapped: float) -> float:
  previous_wrapped = wrap_to_pi(previous_unwrapped)
  return previous_unwrapped + wrap_to_pi(new_wrapped - previous_wrapped)


class DeadlineScheduler:
  """First-grid-time-at-or-after-deadline scheduler with zero-order hold."""

  def __init__(self, target_period_s: float, tolerance_s: float = 1e-12):
    if target_period_s <= 0.0:
      raise ValueError("target control period must be positive")
    self.period = float(target_period_s)
    self.tolerance = float(tolerance_s)
    self.next_target_time = 0.0
    self.update_index = -1
    self.last_actual_update_time: float | None = None

  def maybe_update(self, physics_time_s: float) -> dict[str, float | int] | None:
    if physics_time_s + self.tolerance < self.next_target_time:
      return None
    target = self.next_target_time
    self.update_index += 1
    self.next_target_time += self.period
    lateness = physics_time_s - target
    if lateness < 0.0 and abs(lateness) <= self.tolerance:
      lateness = 0.0
    result = {"control_update_index": self.update_index,
              "control_target_time_s": target,
              "control_actual_update_time_s": physics_time_s,
              "control_update_lateness_s": lateness}
    self.last_actual_update_time = physics_time_s
    return result


def update_plant_body_command(plant_controller: Any, v_cmd: float, omega_cmd: float) -> None:
  if not math.isfinite(v_cmd) or not math.isfinite(omega_cmd):
    raise ValueError("plant body command must be finite")
  plant_controller.v_cmd = float(v_cmd)
  plant_controller.omega_cmd = float(omega_cmd)
  plant_controller.direct_track_command = None


def read_rigid_state(plant_controller: Any) -> dict[str, Any]:
  rigid = plant_controller.rigid
  positions, orientations = rigid.get_world_poses()
  position = as_numpy(positions)[0].astype(float)
  quaternion = as_numpy(orientations)[0].astype(float)
  rotation = quaternion_wxyz_to_matrix(quaternion)
  linear_world = as_numpy(rigid.get_linear_velocities())[0].astype(float)
  angular_world = as_numpy(rigid.get_angular_velocities())[0].astype(float)
  body_linear = rotation.T @ linear_world
  body_angular = rotation.T @ angular_world
  roll, pitch, yaw = roll_pitch_yaw_wxyz(quaternion)
  return {"position": position, "rotation": rotation, "roll_rad": roll,
          "pitch_rad": pitch, "yaw_wrapped_rad": yaw,
          "body_vx_m_s": float(body_linear[0]), "body_vy_m_s": float(body_linear[1]),
          "body_vz_m_s": float(body_linear[2]), "world_vz_m_s": float(linear_world[2]),
          "body_wx_rad_s": float(body_angular[0]), "body_wy_rad_s": float(body_angular[1]),
          "body_wz_rad_s": float(body_angular[2])}


def rms(values: list[float]) -> float:
  return math.sqrt(sum(value * value for value in values) / len(values))


def evaluation_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
  fields = (("cross_track_error_m", "cross_track_error_m"),
            ("heading_error_rad", "heading_error_rad"),
            ("progress_error_m", "progress_error_m"),
            ("speed_error_m_s", "speed_error_m_s"),
            ("yaw_rate_error_rad_s", "yaw_rate_error_rad_s"))
  result: dict[str, Any] = {f"{prefix}_sample_count": len(rows)}
  for source, label in fields:
    values = [float(row[source]) for row in rows]
    result[f"{prefix}_rms_{label}"] = rms(values)
    result[f"{prefix}_maximum_abs_{label}"] = max(abs(value) for value in values)
  return result


def longest_false_run(rows: list[dict[str, Any]], key: str, dt: float) -> dict[str, Any]:
  best = current = total = 0
  for row in rows:
    if not bool(int(row[key])):
      current += 1
      total += 1
      best = max(best, current)
    else:
      current = 0
  return {"unsupported_sample_count": total, "longest_consecutive_steps": best,
          "longest_duration_s": best * dt}


def summarize(rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]],
              configs: dict[str, Any], settled: dict[str, float],
              final_observation: dict[str, Any]) -> dict[str, Any]:
  stage, controller_cfg, plant_cfg = configs["stage"], configs["controller"], configs["plant"]
  reference_end = float(reference_rows[-1]["t_s"])
  active_tolerance = float(stage["runtime"]["active_window_time_tolerance_s"])
  active = [row for row in rows if row["physics_time_s"] <= reference_end + active_tolerance]
  updates = [row for row in rows if int(row["control_update"]) == 1]
  update_times = [float(row["control_actual_update_time_s"]) for row in updates]
  update_intervals = [update_times[index] - update_times[index - 1]
                      for index in range(1, len(update_times))]
  lateness = [float(row["control_update_lateness_s"]) for row in updates]
  saturation_tolerance = float(stage["runtime"]["saturation_scale_tolerance"])
  saturated_updates = sum(float(row["command_scale"]) < 1.0 - saturation_tolerance for row in updates)
  saturated_samples = sum(float(row["command_scale"]) < 1.0 - saturation_tolerance for row in rows)
  goal = reference_rows[-1]
  dt = float(plant_cfg["tunable_uncalibrated"]["physics_dt_s"])
  return {
    "source_config_paths": stage["baseline_configs"],
    "physics_timestep_s": dt,
    "controller_target_period_s": float(controller_cfg["simulation"]["time_step_s"]),
    "reference_duration_s": reference_end,
    "terminal_hold_s": float(controller_cfg["simulation"]["terminal_hold_s"]),
    "last_logged_pre_step_time_s": float(rows[-1]["physics_time_s"]),
    "actual_final_simulation_time_s": float(final_observation["physics_time_s"]),
    "final_post_step_observation": final_observation,
    "settled_world_origin": settled,
    "evaluation_windows": {
      "active": "physics_time_s <= reference_end_time_s + configured tolerance",
      "full_run": "all logged pre-step samples including terminal hold"},
    "observation_semantics": {
      "csv_rows": "Pre-step observations for each logged physics interval.",
      "final_post_step_observation": "Actual rigid-body state after the final logged interval; no extra physics step or callback is added.",
      "v2_callback_time": "v2_callback_legacy_time_s is a legacy callback diagnostic, not the tracking timestamp source."},
    **evaluation_metrics(active, "active"), **evaluation_metrics(rows, "full_run"),
    "final_xy_error_to_reference_goal_m": math.hypot(
      float(final_observation["actual_local_x_m"]) - float(goal["x_m"]),
      float(final_observation["actual_local_y_m"]) - float(goal["y_m"])),
    "final_heading_error_rad": wrap_to_pi(
      float(goal["yaw_rad"]) - float(final_observation["actual_local_yaw_rad"])),
    "final_abs_v_cmd_m_s": abs(float(final_observation["v_cmd_m_s"])),
    "final_abs_omega_cmd_rad_s": abs(float(final_observation["omega_cmd_rad_s"])),
    "final_actual_body_speed_m_s": float(final_observation["v_actual_m_s"]),
    "final_actual_body_yaw_rate_rad_s": float(final_observation["omega_actual_rad_s"]),
    "maximum_abs_v_cmd_m_s": max(abs(float(row["v_cmd_m_s"])) for row in rows),
    "maximum_abs_omega_cmd_rad_s": max(abs(float(row["omega_cmd_rad_s"])) for row in rows),
    "maximum_abs_left_track_command_m_s": max(abs(float(row["v_left_cmd_m_s"])) for row in rows),
    "maximum_abs_right_track_command_m_s": max(abs(float(row["v_right_cmd_m_s"])) for row in rows),
    "maximum_abs_left_actuator_track_state_m_s": max(abs(float(row["v_left_state_m_s"])) for row in rows),
    "maximum_abs_right_actuator_track_state_m_s": max(abs(float(row["v_right_state_m_s"])) for row in rows),
    "minimum_command_scale": min(float(row["command_scale"]) for row in rows),
    "control_timing": {
      "update_count": len(updates),
      "saturated_control_update_count": saturated_updates,
      "saturated_control_update_fraction": saturated_updates / len(updates),
      "saturated_physics_sample_count": saturated_samples,
      "saturated_physics_sample_fraction": saturated_samples / len(rows),
      "minimum_actual_update_interval_s": min(update_intervals),
      "maximum_actual_update_interval_s": max(update_intervals),
      "mean_actual_update_interval_s": sum(update_intervals) / len(update_intervals),
      "maximum_update_lateness_s": max(lateness),
      "rms_update_lateness_s": rms(lateness)},
    "plant": {
      "maximum_abs_roll_rad": max(abs(float(row["roll_rad"])) for row in rows),
      "maximum_abs_pitch_rad": max(abs(float(row["pitch_rad"])) for row in rows),
      "maximum_abs_vertical_velocity_m_s": max(abs(float(row["world_vz_m_s"])) for row in rows),
      "maximum_abs_lateral_body_velocity_m_s": max(abs(float(row["body_vy_m_s"])) for row in rows),
      "maximum_left_normal_load_n": max(float(row["left_normal_load_n"]) for row in rows),
      "maximum_right_normal_load_n": max(float(row["right_normal_load_n"]) for row in rows),
      "maximum_total_normal_load_n": max(float(row["total_normal_load_n"]) for row in rows),
      "left_support_loss": longest_false_run(rows, "left_supported", dt),
      "right_support_loss": longest_false_run(rows, "right_supported", dt),
      "maximum_abs_custom_force_normal_component_n": max(
        abs(float(row["max_abs_custom_force_normal_component_n"])) for row in rows),
      "maximum_abs_speed_tracking_error_m_s": max(abs(float(row["speed_error_m_s"])) for row in rows)},
    "calibration_warning": stage["metadata"]["calibration_warning"],
    "modeling_warning": "body-frame wz is used as a flat-ground yaw-rate feedback proxy; this is not generalized to uneven terrain.",
    "real_robot_equivalence": "No real-robot equivalence is claimed."
  }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def plot_results(rows: list[dict[str, Any]], path: Path) -> tuple[bool, str]:
  try:
    import matplotlib.pyplot as plt
  except ImportError as error:
    return False, f"skipped; matplotlib unavailable: {error}"
  figure, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
  time = [row["physics_time_s"] for row in rows]
  axes[0, 0].plot([row["reference_x_m"] for row in rows], [row["reference_y_m"] for row in rows], "k--", label="reference")
  axes[0, 0].plot([row["actual_local_x_m"] for row in rows], [row["actual_local_y_m"] for row in rows], label="actual")
  axes[0, 0].set_aspect("equal", adjustable="box"); axes[0, 0].legend(); axes[0, 0].set_title("Local XY")
  axes[0, 1].plot(time, [row["cross_track_error_m"] for row in rows], label="cross-track")
  axes[0, 1].plot(time, [row["heading_error_rad"] for row in rows], label="heading")
  axes[0, 1].legend(); axes[0, 1].set_title("Tracking errors")
  axes[1, 0].plot(time, [row["v_ref_m_s"] for row in rows], label="reference")
  axes[1, 0].plot(time, [row["v_cmd_m_s"] for row in rows], label="command")
  axes[1, 0].plot(time, [row["v_actual_m_s"] for row in rows], label="actual")
  axes[1, 0].legend(); axes[1, 0].set_title("Body speed")
  axes[1, 1].plot(time, [row["omega_ref_rad_s"] for row in rows], label="reference")
  axes[1, 1].plot(time, [row["omega_cmd_rad_s"] for row in rows], label="command")
  axes[1, 1].plot(time, [row["omega_actual_rad_s"] for row in rows], label="actual")
  axes[1, 1].legend(); axes[1, 1].set_title("Yaw rate")
  axes[2, 0].plot(time, [row["v_left_cmd_m_s"] for row in rows], label="left cmd")
  axes[2, 0].plot(time, [row["v_left_state_m_s"] for row in rows], label="left state")
  axes[2, 0].plot(time, [row["v_right_cmd_m_s"] for row in rows], label="right cmd")
  axes[2, 0].plot(time, [row["v_right_state_m_s"] for row in rows], label="right state")
  axes[2, 0].legend(); axes[2, 0].set_title("Track command/state")
  axes[2, 1].plot(time, [row["command_scale"] for row in rows], label="command scale")
  axes[2, 1].plot(time, [row["left_supported"] for row in rows], label="left supported")
  axes[2, 1].plot(time, [row["right_supported"] for row in rows], label="right supported")
  axes[2, 1].legend(); axes[2, 1].set_title("Feasibility/support")
  for axis in axes.flat:
    axis.grid(True, alpha=0.25); axis.set_xlabel("time [s]")
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=160); plt.close(figure)
  return True, str(path)


def run_closed_loop(world: Any, plant_controller: Any, reference_rows: list[dict[str, Any]],
                    configs: dict[str, Any], render: bool, realtime: bool
                    ) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
  stage, controller_cfg, plant_cfg = configs["stage"], configs["controller"], configs["plant"]
  dt = float(plant_cfg["tunable_uncalibrated"]["physics_dt_s"])
  control_period = float(controller_cfg["simulation"]["time_step_s"])
  settle_duration = float(plant_cfg["tunable_uncalibrated"]["settle_duration_s"])
  terminal_hold = float(controller_cfg["simulation"]["terminal_hold_s"])
  total_duration = float(reference_rows[-1]["t_s"]) + terminal_hold

  if not world.is_playing():
    world.play()
  if not world.is_playing():
    raise RuntimeError("simulation timeline did not enter PLAY state")
  plant_controller.reset({"v_cmd_m_s": 0.0, "omega_cmd_rad_s": 0.0})
  for _ in range(round(settle_duration / dt)):
    step_world(world, render, realtime, dt)
  settled_state = read_rigid_state(plant_controller)
  settled = {"x0_world_m": float(settled_state["position"][0]),
             "y0_world_m": float(settled_state["position"][1]),
             "z0_world_m": float(settled_state["position"][2]),
             "yaw0_world_rad": float(settled_state["yaw_wrapped_rad"])}
  # Reset actuator state and callback diagnostics exactly once after settling;
  # the settled physical pose is deliberately left untouched.
  plant_controller.reset({"v_cmd_m_s": 0.0, "omega_cmd_rad_s": 0.0})

  scheduler = DeadlineScheduler(control_period, float(stage["runtime"]["scheduler_time_tolerance_s"]))
  interpolator = ReferenceInterpolator(reference_rows)
  step_count = math.ceil(total_duration / dt)
  rows: list[dict[str, Any]] = []
  yaw_unwrapped = settled["yaw0_world_rad"]
  held: dict[str, float] | None = None
  held_context: dict[str, float] | None = None
  for step_index in range(step_count):
    physics_time = step_index * dt
    state = read_rigid_state(plant_controller)
    if step_index:
      yaw_unwrapped = continuous_angle_update(yaw_unwrapped, float(state["yaw_wrapped_rad"]))
    x_local, y_local = world_to_settled_local_xy(
      float(state["position"][0]), float(state["position"][1]),
      settled["x0_world_m"], settled["y0_world_m"], settled["yaw0_world_rad"])
    yaw_local = yaw_unwrapped - settled["yaw0_world_rad"]
    reference = interpolator.sample(physics_time)
    update = scheduler.maybe_update(physics_time)
    if update is not None:
      held = controller_command((x_local, y_local, yaw_local), reference, controller_cfg)
      held_context = {
        "controller_reference_t_s": physics_time,
        "controller_reference_s_m": reference["s_m"],
        "controller_reference_v_ref_m_s": reference["v_ref_m_s"],
        "controller_reference_omega_ref_rad_s": reference["omega_ref_rad_s"],
        "controller_update_e_x_m": held["e_x_m"],
        "controller_update_e_y_m": held["e_y_m"],
        "controller_update_heading_error_rad": held["e_heading_rad"],
      }
      update_plant_body_command(plant_controller, held["v_cmd_m_s"], held["omega_cmd_rad_s"])
    if held is None or held_context is None:
      raise RuntimeError("first controller update did not occur at tracking time zero")
    current_errors = body_frame_errors((x_local, y_local, yaw_local), reference)
    projection = project_to_polyline(x_local, y_local, reference_rows)
    row: dict[str, Any] = {
      "physics_time_s": physics_time, "physics_step_index": step_index,
      "control_update": int(update is not None),
      "control_update_index": held.get("control_update_index", scheduler.update_index),
      "held_command_age_s": physics_time - held_context["controller_reference_t_s"],
      "control_target_time_s": "" if update is None else update["control_target_time_s"],
      "control_actual_update_time_s": "" if update is None else update["control_actual_update_time_s"],
      "control_update_lateness_s": "" if update is None else update["control_update_lateness_s"],
      **held_context,
      "reference_t_s": reference["t_s"], "reference_s_m": reference["s_m"],
      "reference_x_m": reference["x_m"], "reference_y_m": reference["y_m"],
      "reference_yaw_rad": reference["yaw_rad"],
      "reference_curvature_1_m": reference["curvature_ref_1_m"],
      "v_ref_m_s": reference["v_ref_m_s"], "omega_ref_rad_s": reference["omega_ref_rad_s"],
      "actual_local_x_m": x_local, "actual_local_y_m": y_local, "actual_local_yaw_rad": yaw_local,
      "actual_world_x_m": float(state["position"][0]), "actual_world_y_m": float(state["position"][1]),
      "actual_world_z_m": float(state["position"][2]), "roll_rad": state["roll_rad"],
      "pitch_rad": state["pitch_rad"], "actual_world_yaw_wrapped_rad": state["yaw_wrapped_rad"],
      "e_x_m": current_errors["e_x_m"], "e_y_m": current_errors["e_y_m"],
      "heading_error_rad": current_errors["e_heading_rad"],
      "projected_s_m": projection["s_projected_m"],
      "cross_track_error_m": projection["cross_track_error_m"],
      "progress_error_m": reference["s_m"] - projection["s_projected_m"],
      "v_raw_m_s": held["v_raw_m_s"], "omega_raw_rad_s": held["omega_raw_rad_s"],
      "v_cmd_m_s": held["v_cmd_m_s"], "omega_cmd_rad_s": held["omega_cmd_rad_s"],
      "command_scale": held["command_scale"],
      "v_actual_m_s": state["body_vx_m_s"], "omega_actual_rad_s": state["body_wz_rad_s"],
      "body_vy_m_s": state["body_vy_m_s"], "body_vz_m_s": state["body_vz_m_s"],
      "body_wx_rad_s": state["body_wx_rad_s"], "body_wy_rad_s": state["body_wy_rad_s"],
      "world_vz_m_s": state["world_vz_m_s"],
      "v_left_cmd_m_s": held["v_left_cmd_m_s"], "v_right_cmd_m_s": held["v_right_cmd_m_s"],
      "speed_error_m_s": reference["v_ref_m_s"] - state["body_vx_m_s"],
      "yaw_rate_error_rad_s": reference["omega_ref_rad_s"] - state["body_wz_rad_s"],
    }
    callback_count = len(plant_controller.rows)
    step_world(world, render, realtime, dt)
    if len(plant_controller.rows) != callback_count + 1:
      raise RuntimeError("V2 physics callback did not produce exactly one diagnostic row")
    callback = plant_controller.rows[-1]
    row.update({
      "v2_callback_legacy_time_s": callback["time_s"],
      "v_left_state_m_s": callback["v_left_state_m_s"],
      "v_right_state_m_s": callback["v_right_state_m_s"],
      "left_contact_count": callback["left_contact_count"],
      "right_contact_count": callback["right_contact_count"],
      "left_normal_load_n": callback["left_normal_load_n"],
      "right_normal_load_n": callback["right_normal_load_n"],
      "total_normal_load_n": callback["total_normal_load_n"],
      "left_supported": callback["left_supported"], "right_supported": callback["right_supported"],
      "left_consecutive_support_loss_steps": callback["left_consecutive_support_loss_steps"],
      "right_consecutive_support_loss_steps": callback["right_consecutive_support_loss_steps"],
      "left_longitudinal_force_n": callback["left_longitudinal_force_n"],
      "right_longitudinal_force_n": callback["right_longitudinal_force_n"],
      "left_lateral_force_n": callback["left_lateral_force_n"],
      "right_lateral_force_n": callback["right_lateral_force_n"],
      "applied_yaw_moment_n_m": callback["applied_yaw_moment_n_m"],
      "max_abs_custom_force_normal_component_n": callback["max_abs_force_normal_component_n"],
      "skipped_tangent_contact_count": callback["skipped_tangent_contact_count"],
    })
    rows.append(row)
  final_time = step_count * dt
  final_state = read_rigid_state(plant_controller)
  final_yaw_unwrapped = continuous_angle_update(
    yaw_unwrapped, float(final_state["yaw_wrapped_rad"]))
  final_x_local, final_y_local = world_to_settled_local_xy(
    float(final_state["position"][0]), float(final_state["position"][1]),
    settled["x0_world_m"], settled["y0_world_m"], settled["yaw0_world_rad"])
  final_yaw_local = final_yaw_unwrapped - settled["yaw0_world_rad"]
  final_reference = interpolator.sample(final_time)
  final_errors = body_frame_errors(
    (final_x_local, final_y_local, final_yaw_local), final_reference)
  final_projection = project_to_polyline(final_x_local, final_y_local, reference_rows)
  final_observation = {
    "physics_time_s": final_time,
    "reference_t_s": final_reference["t_s"], "reference_s_m": final_reference["s_m"],
    "reference_x_m": final_reference["x_m"], "reference_y_m": final_reference["y_m"],
    "reference_yaw_rad": final_reference["yaw_rad"],
    "v_ref_m_s": final_reference["v_ref_m_s"],
    "omega_ref_rad_s": final_reference["omega_ref_rad_s"],
    "actual_local_x_m": final_x_local, "actual_local_y_m": final_y_local,
    "actual_local_yaw_rad": final_yaw_local,
    "actual_world_x_m": float(final_state["position"][0]),
    "actual_world_y_m": float(final_state["position"][1]),
    "actual_world_z_m": float(final_state["position"][2]),
    "actual_world_yaw_wrapped_rad": final_state["yaw_wrapped_rad"],
    "roll_rad": final_state["roll_rad"], "pitch_rad": final_state["pitch_rad"],
    "v_actual_m_s": final_state["body_vx_m_s"],
    "omega_actual_rad_s": final_state["body_wz_rad_s"],
    "body_vy_m_s": final_state["body_vy_m_s"], "body_vz_m_s": final_state["body_vz_m_s"],
    "body_wx_rad_s": final_state["body_wx_rad_s"],
    "body_wy_rad_s": final_state["body_wy_rad_s"], "world_vz_m_s": final_state["world_vz_m_s"],
    "e_x_m": final_errors["e_x_m"], "e_y_m": final_errors["e_y_m"],
    "heading_error_rad": final_errors["e_heading_rad"],
    "projected_s_m": final_projection["s_projected_m"],
    "cross_track_error_m": final_projection["cross_track_error_m"],
    "progress_error_m": final_reference["s_m"] - final_projection["s_projected_m"],
    "v_cmd_m_s": held["v_cmd_m_s"], "omega_cmd_rad_s": held["omega_cmd_rad_s"],
    "command_scale": held["command_scale"],
  }
  return rows, settled, final_observation


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/isaac_closed_loop_v1.json")
  parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs/isaac_closed_loop_v1")
  parser.add_argument("--gui", action="store_true")
  parser.add_argument("--realtime", action="store_true")
  parser.add_argument("--plot", action="store_true")
  args = parser.parse_args()
  if args.realtime and not args.gui:
    parser.error("--realtime requires --gui")

  from isaacsim import SimulationApp
  app = SimulationApp({"headless": not args.gui, "enable_cameras": args.gui})
  try:
    stage_path = args.config.resolve()
    stage = load_stage_config(stage_path)
    paths = {name: REPO_ROOT / value for name, value in stage["baseline_configs"].items()}
    reference_cfg = load_reference_config(paths["reference_trajectory"])
    controller_cfg = load_controller_config(paths["tracking_controller"])
    plant_cfg = load_plant_config(paths["tracked_force_plant_v2"])
    b_controller = float(controller_cfg["measured_fixed"]["track_center_distance_m"])
    b_plant = float(plant_cfg["measured_fixed"]["track_center_distance_b_m"])
    if b_controller != b_plant or b_controller != 0.434:
      raise RuntimeError(f"track-center distance mismatch: controller={b_controller}, plant={b_plant}")
    support_mode = plant_cfg["tunable_uncalibrated"]["support_mode"]
    if support_mode != stage["runtime"]["support_mode_required"] or support_mode != "flat_track_boxes":
      raise RuntimeError(f"Stage C1 requires flat_track_boxes, got {support_mode}")
    configs = {"stage": stage, "controller": controller_cfg, "plant": plant_cfg}
    reference_rows = build_reference_trajectory(reference_cfg)["trajectory"]
    output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    world, plant_controller, build_info = make_world(
      plant_cfg, output_dir / stage["output"]["stage_filename"],
      stage["runtime"]["creation_order"], bool(stage["runtime"]["detailed_contacts"]), support_mode)
    rows, settled, final_observation = run_closed_loop(
      world, plant_controller, reference_rows, configs, args.gui, args.realtime)
    summary = summarize(rows, reference_rows, configs, settled, final_observation)
    summary["stage_config_path"] = str(stage_path)
    summary["plant_build"] = build_info
    csv_path = output_dir / stage["output"]["csv_filename"]
    summary_path = output_dir / stage["output"]["summary_filename"]
    write_csv(csv_path, rows)
    with summary_path.open("w", encoding="utf-8") as stream:
      json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
    plot_status = "disabled"
    if args.plot:
      _, plot_status = plot_results(rows, output_dir / stage["output"]["plot_filename"])
    timing = summary["control_timing"]
    print(f"physics_dt={summary['physics_timestep_s']:.12g} s, control_target="
          f"{summary['controller_target_period_s']:.12g} s, final_time="
          f"{summary['actual_final_simulation_time_s']:.12g} s")
    print(f"updates={timing['update_count']}, intervals="
          f"{timing['minimum_actual_update_interval_s']:.9f}.."
          f"{timing['maximum_actual_update_interval_s']:.9f} s, "
          f"mean={timing['mean_actual_update_interval_s']:.9f} s, "
          f"max_lateness={timing['maximum_update_lateness_s']:.9f} s")
    print(f"active cross-track RMS={summary['active_rms_cross_track_error_m']:.6f} m, "
          f"heading RMS={summary['active_rms_heading_error_rad']:.6f} rad, "
          f"final goal error={summary['final_xy_error_to_reference_goal_m']:.6f} m")
    print(f"csv={csv_path}\nsummary={summary_path}\nplot={plot_status}")
    return 0
  except BaseException:
    traceback.print_exc(); raise
  finally:
    app.close()


if __name__ == "__main__":
  raise SystemExit(main())
