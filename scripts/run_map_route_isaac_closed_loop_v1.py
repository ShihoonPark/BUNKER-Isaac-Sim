#!/usr/bin/env python3
"""Run the nominal maneuver-aware GLIM map route through the canonical Isaac V2 plant."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from map_route_kinematic_validation_v1 import ordered_project  # noqa: E402
from map_route_reference_v1 import build_map_route_reference  # noqa: E402
from run_isaac_closed_loop_v1 import (  # noqa: E402
  DeadlineScheduler, continuous_angle_update, read_rigid_state,
  update_plant_body_command, world_to_settled_local_xy,
)
from test_tracked_force_plant_v2 import load_config as load_plant_config, make_world, step_world  # noqa: E402
from tracking_controller_v1 import ReferenceInterpolator, body_frame_errors, wrap_to_pi  # noqa: E402
from tracking_controller_v2_direction_aware import (  # noqa: E402
  controller_command, load_direction_aware_config,
)


def load_stage_config(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    return json.load(stream)


def reference_context_index(reference_rows: list[dict[str, Any]], reference_times: list[float],
                            query_time: float) -> int:
  """Select discrete maneuver metadata without interpolation.

  The left sample owns an interval normally. If that sample is the endpoint of
  one maneuver and the following sample starts another, every time strictly
  after the endpoint belongs to the following maneuver. Exact boundary time
  remains with the endpoint row.
  """
  index = max(0, min(len(reference_rows) - 1,
                     bisect.bisect_right(reference_times, min(query_time, reference_times[-1])) - 1))
  if (index + 1 < len(reference_rows) and
      int(reference_rows[index]["segment_id"]) != int(reference_rows[index + 1]["segment_id"]) and
      query_time > reference_times[index] + 1e-12):
    index += 1
  return index


def rms(values: list[float]) -> float:
  return math.sqrt(sum(value * value for value in values) / len(values))


def _saturation_reasons(command: dict[str, float], controller_config: dict[str, Any]) -> str:
  limits = controller_config["command_limits"]
  spacing = float(controller_config["measured_fixed"]["track_center_distance_m"])
  left = float(command["v_raw_m_s"]) - .5 * spacing * float(command["omega_raw_rad_s"])
  right = float(command["v_raw_m_s"]) + .5 * spacing * float(command["omega_raw_rad_s"])
  reasons = []
  if abs(float(command["v_raw_m_s"])) > float(limits["maximum_abs_body_speed_m_s"]): reasons.append("body")
  if abs(float(command["omega_raw_rad_s"])) > float(limits["maximum_abs_yaw_rate_rad_s"]): reasons.append("yaw")
  if abs(left) > float(limits["maximum_abs_track_surface_speed_m_s"]): reasons.append("left_track")
  if abs(right) > float(limits["maximum_abs_track_surface_speed_m_s"]): reasons.append("right_track")
  return "+".join(reasons) if reasons else "none"


def run_closed_loop(world: Any, plant_controller: Any, reference_rows: list[dict[str, Any]],
                    configs: dict[str, Any], render: bool, realtime: bool
                    ) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
  stage, controller_config, plant_config = configs["stage"], configs["controller"], configs["plant"]
  dt = float(plant_config["tunable_uncalibrated"]["physics_dt_s"])
  control_period = float(controller_config["simulation"]["time_step_s"])
  settle_duration = float(plant_config["tunable_uncalibrated"]["settle_duration_s"])
  total_duration = float(reference_rows[-1]["t_s"]) + float(controller_config["simulation"]["terminal_hold_s"])
  reference_times = [float(row["t_s"]) for row in reference_rows]
  projection_window = int(stage["runtime"]["ordered_projection_half_window_segments"])

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
  plant_controller.reset({"v_cmd_m_s": 0.0, "omega_cmd_rad_s": 0.0})

  scheduler = DeadlineScheduler(control_period, float(stage["runtime"]["scheduler_time_tolerance_s"]))
  interpolator = ReferenceInterpolator(reference_rows)
  step_count = math.ceil(total_duration / dt)
  rows = []
  yaw_unwrapped = settled["yaw0_world_rad"]
  held: dict[str, Any] | None = None
  held_context: dict[str, Any] | None = None
  for step_index in range(step_count):
    physics_time = step_index * dt
    state = read_rigid_state(plant_controller)
    if step_index:
      yaw_unwrapped = continuous_angle_update(yaw_unwrapped, float(state["yaw_wrapped_rad"]))
    x_local, y_local = world_to_settled_local_xy(
      float(state["position"][0]), float(state["position"][1]), settled["x0_world_m"],
      settled["y0_world_m"], settled["yaw0_world_rad"])
    yaw_local = yaw_unwrapped - settled["yaw0_world_rad"]
    reference = interpolator.sample(physics_time)
    context_index = reference_context_index(reference_rows, reference_times, physics_time)
    discrete = reference_rows[context_index]
    update = scheduler.maybe_update(physics_time)
    if update is not None:
      held = controller_command((x_local, y_local, yaw_local), reference,
                                int(discrete["motion_direction"]), controller_config)
      held_context = {
        "controller_reference_t_s": physics_time,
        "controller_reference_s_m": reference["s_m"],
        "controller_reference_v_ref_m_s": reference["v_ref_m_s"],
        "controller_reference_omega_ref_rad_s": reference["omega_ref_rad_s"],
        "controller_reference_segment_id": int(discrete["segment_id"]),
        "controller_reference_motion_direction": int(discrete["motion_direction"]),
        "controller_reference_curvature_valid": int(discrete["curvature_valid"]),
        "controller_update_e_x_m": held["e_x_m"],
        "controller_update_e_y_m": held["e_y_m"],
        "controller_update_heading_error_rad": held["e_heading_rad"],
        "controller_lateral_feedback_factor": held["lateral_feedback_factor"],
        "controller_saturation_reasons": _saturation_reasons(held, controller_config),
      }
      update_plant_body_command(plant_controller, held["v_cmd_m_s"], held["omega_cmd_rad_s"])
    if held is None or held_context is None:
      raise RuntimeError("first controller update did not occur at tracking time zero")
    errors = body_frame_errors((x_local, y_local, yaw_local), reference)
    projection = ordered_project(x_local, y_local, reference_rows, context_index, projection_window)
    row = {
      "physics_time_s": physics_time, "physics_step_index": step_index,
      "control_update": int(update is not None), "control_update_index": scheduler.update_index,
      "held_command_age_s": physics_time - float(held_context["controller_reference_t_s"]),
      "control_target_time_s": "" if update is None else update["control_target_time_s"],
      "control_actual_update_time_s": "" if update is None else update["control_actual_update_time_s"],
      "control_update_lateness_s": "" if update is None else update["control_update_lateness_s"],
      **held_context,
      "reference_t_s": reference["t_s"], "reference_s_m": reference["s_m"],
      "reference_x_m": reference["x_m"], "reference_y_m": reference["y_m"],
      "reference_yaw_rad": reference["yaw_rad"], "reference_curvature_1_m": reference["curvature_ref_1_m"],
      "reference_segment_id": int(discrete["segment_id"]),
      "reference_motion_direction": int(discrete["motion_direction"]),
      "reference_curvature_valid": int(discrete["curvature_valid"]),
      "v_ref_m_s": reference["v_ref_m_s"], "omega_ref_rad_s": reference["omega_ref_rad_s"],
      "actual_local_x_m": x_local, "actual_local_y_m": y_local, "actual_local_yaw_rad": yaw_local,
      "actual_world_x_m": float(state["position"][0]), "actual_world_y_m": float(state["position"][1]),
      "actual_world_z_m": float(state["position"][2]), "roll_rad": state["roll_rad"],
      "pitch_rad": state["pitch_rad"], "actual_world_yaw_wrapped_rad": state["yaw_wrapped_rad"],
      "e_x_m": errors["e_x_m"], "e_y_m": errors["e_y_m"], "heading_error_rad": errors["e_heading_rad"],
      "time_aligned_xy_error_m": math.hypot(reference["x_m"] - x_local, reference["y_m"] - y_local),
      "projected_s_m": projection["s_projected_m"], "cross_track_error_m": projection["cross_track_error_m"],
      "progress_error_m": reference["s_m"] - projection["s_projected_m"],
      "projection_segment_index": int(projection["projection_segment_index"]),
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
      "v_left_state_m_s": callback["v_left_state_m_s"], "v_right_state_m_s": callback["v_right_state_m_s"],
      "left_contact_count": callback["left_contact_count"], "right_contact_count": callback["right_contact_count"],
      "left_normal_load_n": callback["left_normal_load_n"], "right_normal_load_n": callback["right_normal_load_n"],
      "total_normal_load_n": callback["total_normal_load_n"],
      "left_supported": callback["left_supported"], "right_supported": callback["right_supported"],
      "left_consecutive_support_loss_steps": callback["left_consecutive_support_loss_steps"],
      "right_consecutive_support_loss_steps": callback["right_consecutive_support_loss_steps"],
      "left_longitudinal_force_n": callback["left_longitudinal_force_n"],
      "right_longitudinal_force_n": callback["right_longitudinal_force_n"],
      "left_lateral_force_n": callback["left_lateral_force_n"], "right_lateral_force_n": callback["right_lateral_force_n"],
      "applied_yaw_moment_n_m": callback["applied_yaw_moment_n_m"],
      "max_abs_custom_force_normal_component_n": callback["max_abs_force_normal_component_n"],
      "skipped_tangent_contact_count": callback["skipped_tangent_contact_count"],
    })
    rows.append(row)

  final_time = step_count * dt
  state = read_rigid_state(plant_controller)
  final_yaw = continuous_angle_update(yaw_unwrapped, float(state["yaw_wrapped_rad"]))
  x_local, y_local = world_to_settled_local_xy(float(state["position"][0]), float(state["position"][1]),
                                                settled["x0_world_m"], settled["y0_world_m"], settled["yaw0_world_rad"])
  yaw_local = final_yaw - settled["yaw0_world_rad"]
  reference = interpolator.sample(final_time)
  context_index = reference_context_index(reference_rows, reference_times, final_time)
  projection = ordered_project(x_local, y_local, reference_rows, context_index, projection_window)
  errors = body_frame_errors((x_local, y_local, yaw_local), reference)
  final_observation = {
    "physics_time_s": final_time, "reference_t_s": reference["t_s"], "reference_s_m": reference["s_m"],
    "reference_x_m": reference["x_m"], "reference_y_m": reference["y_m"], "reference_yaw_rad": reference["yaw_rad"],
    "reference_segment_id": int(reference_rows[context_index]["segment_id"]),
    "reference_motion_direction": int(reference_rows[context_index]["motion_direction"]),
    "v_ref_m_s": reference["v_ref_m_s"], "omega_ref_rad_s": reference["omega_ref_rad_s"],
    "actual_local_x_m": x_local, "actual_local_y_m": y_local, "actual_local_yaw_rad": yaw_local,
    "actual_world_x_m": float(state["position"][0]), "actual_world_y_m": float(state["position"][1]),
    "actual_world_z_m": float(state["position"][2]), "actual_world_yaw_wrapped_rad": state["yaw_wrapped_rad"],
    "roll_rad": state["roll_rad"], "pitch_rad": state["pitch_rad"],
    "v_actual_m_s": state["body_vx_m_s"], "omega_actual_rad_s": state["body_wz_rad_s"],
    "body_vy_m_s": state["body_vy_m_s"], "body_vz_m_s": state["body_vz_m_s"],
    "body_wx_rad_s": state["body_wx_rad_s"], "body_wy_rad_s": state["body_wy_rad_s"],
    "world_vz_m_s": state["world_vz_m_s"], "e_x_m": errors["e_x_m"], "e_y_m": errors["e_y_m"],
    "heading_error_rad": errors["e_heading_rad"], "time_aligned_xy_error_m": math.hypot(reference["x_m"] - x_local, reference["y_m"] - y_local),
    "projected_s_m": projection["s_projected_m"], "cross_track_error_m": projection["cross_track_error_m"],
    "progress_error_m": reference["s_m"] - projection["s_projected_m"],
    "v_cmd_m_s": held["v_cmd_m_s"], "omega_cmd_rad_s": held["omega_cmd_rad_s"],
    "v_left_cmd_m_s": held["v_left_cmd_m_s"], "v_right_cmd_m_s": held["v_right_cmd_m_s"],
    "command_scale": held["command_scale"]}
  return rows, settled, final_observation


def metric_block(rows: list[dict[str, Any]], scale_tolerance: float, physics_dt: float) -> dict[str, Any]:
  result: dict[str, Any] = {"sample_count": len(rows),
                            "sampled_duration_s": len(rows) * physics_dt}
  for field in ("e_x_m", "e_y_m", "heading_error_rad", "time_aligned_xy_error_m",
                "cross_track_error_m", "progress_error_m", "speed_error_m_s",
                "yaw_rate_error_rad_s", "body_vy_m_s"):
    values = [float(row[field]) for row in rows]
    result[f"rms_{field}"] = rms(values); result[f"maximum_abs_{field}"] = max(abs(value) for value in values)
  updates = [row for row in rows if int(row["control_update"]) == 1]
  saturated = sum(float(row["command_scale"]) < 1.0 - scale_tolerance for row in updates)
  result.update({"control_update_count": len(updates), "saturated_control_update_count": saturated,
                 "saturated_control_update_fraction": saturated / len(updates) if updates else 0.0,
                 "minimum_command_scale": min(float(row["command_scale"]) for row in rows)})
  return result


def summarize(rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], configs: dict[str, Any],
              settled: dict[str, float], final: dict[str, Any], kinematic_summary: dict[str, Any]) -> dict[str, Any]:
  stage, controller, plant = configs["stage"], configs["controller"], configs["plant"]
  reference_end = float(reference_rows[-1]["t_s"])
  active = [row for row in rows if float(row["physics_time_s"]) <= reference_end + float(stage["runtime"]["active_window_time_tolerance_s"])]
  scale_tolerance = float(stage["runtime"]["saturation_scale_tolerance"])
  updates = [row for row in rows if int(row["control_update"]) == 1]
  update_times = [float(row["control_actual_update_time_s"]) for row in updates]
  intervals = [right - left for left, right in zip(update_times, update_times[1:])]
  lateness = [float(row["control_update_lateness_s"]) for row in updates]
  modes = {1: "forward", -1: "reverse", 0: "pivot"}
  dt = float(plant["tunable_uncalibrated"]["physics_dt_s"])
  mode_metrics = {name: metric_block([row for row in active if int(row["reference_motion_direction"]) == direction], scale_tolerance, dt)
                  for direction, name in modes.items()}
  segments = {}
  for segment_id in range(23):
    selected = [row for row in active if int(row["reference_segment_id"]) == segment_id]
    metrics = metric_block(selected, scale_tolerance, dt)
    metrics.update({"mode": modes[int(selected[0]["reference_motion_direction"])],
                    "entry_actual": {key: float(selected[0][key]) for key in ("actual_local_x_m", "actual_local_y_m", "actual_local_yaw_rad", "e_y_m", "heading_error_rad")},
                    "exit_actual": {key: float(selected[-1][key]) for key in ("actual_local_x_m", "actual_local_y_m", "actual_local_yaw_rad", "e_y_m", "heading_error_rad")}})
    if metrics["mode"] == "pivot":
      metrics.update({"reference_yaw_change_rad": float(selected[-1]["reference_yaw_rad"]) - float(selected[0]["reference_yaw_rad"]),
                      "actual_yaw_change_rad": float(selected[-1]["actual_local_yaw_rad"]) - float(selected[0]["actual_local_yaw_rad"]),
                      "entry_xy_error_m": float(selected[0]["time_aligned_xy_error_m"]),
                      "exit_xy_error_m": float(selected[-1]["time_aligned_xy_error_m"]),
                      "maximum_abs_v_cmd_m_s": max(abs(float(row["v_cmd_m_s"])) for row in selected),
                      "maximum_abs_v_actual_m_s": max(abs(float(row["v_actual_m_s"])) for row in selected),
                      "maximum_abs_omega_ref_rad_s": max(abs(float(row["omega_ref_rad_s"])) for row in selected),
                      "maximum_abs_omega_cmd_rad_s": max(abs(float(row["omega_cmd_rad_s"])) for row in selected),
                      "maximum_abs_omega_actual_rad_s": max(abs(float(row["omega_actual_rad_s"])) for row in selected),
                      "maximum_abs_track_command_difference_m_s": max(abs(float(row["v_right_cmd_m_s"]) - float(row["v_left_cmd_m_s"])) for row in selected),
                      "maximum_abs_yaw_moment_n_m": max(abs(float(row["applied_yaw_moment_n_m"])) for row in selected)})
    segments[str(segment_id)] = metrics
  kinematic = kinematic_summary["v2_scenarios"]["map_nominal"]
  kinematic_mode = kinematic["mode_metrics"]
  comparison = {"overall": {}, "by_mode": {}}
  isaac_active = metric_block(active, scale_tolerance, dt)
  for label, kinematic_key, isaac_key in (
      ("rms_e_y_m", "rms_e_y", "rms_e_y_m"), ("rms_heading_error_rad", "rms_heading_error", "rms_heading_error_rad"),
      ("rms_time_aligned_xy_m", "rms_time_aligned_xy_error", "rms_time_aligned_xy_error_m"),
      ("rms_cross_track_error_m", "rms_cross_track_error", "rms_cross_track_error_m"),
      ("rms_progress_error_m", "rms_progress_error", "rms_progress_error_m")):
    comparison["overall"][label] = {"kinematic_v2": float(kinematic["active"][kinematic_key]), "isaac": float(isaac_active[isaac_key])}
    for mode in modes.values():
      comparison["by_mode"].setdefault(mode, {})[label] = {
        "kinematic_v2": float(kinematic_mode[mode][kinematic_key]), "isaac": float(mode_metrics[mode][isaac_key])}
  force_guard = float(plant["tunable_uncalibrated"]["maximum_track_force_n"])
  near_force = float(stage["runtime"]["force_guard_near_fraction"]) * force_guard
  actuator_ceiling = float(plant["tunable_uncalibrated"]["maximum_surface_speed_m_s"])
  result = {
    "source_config_paths": stage["baseline_configs"], "reference_duration_s": reference_end,
    "reference_path_length_m": float(reference_rows[-1]["s_m"]), "reference_segment_counts": {"forward": 10, "reverse": 9, "pivot": 4},
    "reference_signed_speed_range_m_s": [min(float(row["v_ref_m_s"]) for row in reference_rows), max(float(row["v_ref_m_s"]) for row in reference_rows)],
    "physics_timestep_s": dt, "controller_target_period_s": float(controller["simulation"]["time_step_s"]),
    "terminal_hold_s": float(controller["simulation"]["terminal_hold_s"]),
    "last_logged_pre_step_time_s": float(rows[-1]["physics_time_s"]), "actual_final_simulation_time_s": float(final["physics_time_s"]),
    "settled_world_origin": settled, "final_post_step_observation": final,
    "observation_semantics": {"csv_rows": "pre-step physics-interval observations",
                              "final_post_step_observation": "true state after the final logged interval",
                              "projection": f"metrics-only nearest projection within +/-{stage['runtime']['ordered_projection_half_window_segments']} segments of time-indexed context",
                              "discrete_mode_boundary": "left endpoint owns exact boundary time; any later time between differing segment rows uses the following segment metadata"},
    "active": isaac_active, "full_run": metric_block(rows, scale_tolerance, dt),
    "by_mode": mode_metrics, "segments": segments, "kinematic_v2_vs_isaac": comparison,
    "final_xy_goal_error_m": math.hypot(float(final["actual_local_x_m"]) - float(reference_rows[-1]["x_m"]), float(final["actual_local_y_m"]) - float(reference_rows[-1]["y_m"])),
    "final_heading_error_rad": wrap_to_pi(float(reference_rows[-1]["yaw_rad"]) - float(final["actual_local_yaw_rad"])),
    "control_timing": {"update_count": len(updates),
      "saturated_update_count": sum(float(row["command_scale"]) < 1.0 - scale_tolerance for row in updates),
      "saturated_update_fraction": sum(float(row["command_scale"]) < 1.0 - scale_tolerance for row in updates) / len(updates),
      "minimum_actual_update_interval_s": min(intervals), "maximum_actual_update_interval_s": max(intervals),
      "mean_actual_update_interval_s": sum(intervals) / len(intervals),
      "maximum_update_lateness_s": max(lateness), "rms_update_lateness_s": rms(lateness),
      "saturation_reason_update_counts": {reason: sum(reason in str(row["controller_saturation_reasons"]).split("+") for row in updates)
                                           for reason in ("body", "yaw", "left_track", "right_track")}},
    "command_and_actuator": {
      "maximum_abs_v_cmd_m_s": max(abs(float(row["v_cmd_m_s"])) for row in rows),
      "maximum_abs_omega_cmd_rad_s": max(abs(float(row["omega_cmd_rad_s"])) for row in rows),
      "maximum_abs_left_track_command_m_s": max(abs(float(row["v_left_cmd_m_s"])) for row in rows),
      "maximum_abs_right_track_command_m_s": max(abs(float(row["v_right_cmd_m_s"])) for row in rows),
      "maximum_abs_left_actuator_state_m_s": max(abs(float(row["v_left_state_m_s"])) for row in rows),
      "maximum_abs_right_actuator_state_m_s": max(abs(float(row["v_right_state_m_s"])) for row in rows),
      "actuator_ceiling_m_s": actuator_ceiling,
      "actuator_ceiling_sample_count": sum(max(abs(float(row["v_left_state_m_s"])), abs(float(row["v_right_state_m_s"]))) >= actuator_ceiling - float(stage["runtime"]["actuator_ceiling_tolerance_m_s"]) for row in rows),
      "minimum_command_scale": min(float(row["command_scale"]) for row in rows)},
    "plant": {
      "maximum_abs_roll_rad": max(abs(float(row["roll_rad"])) for row in rows),
      "maximum_abs_pitch_rad": max(abs(float(row["pitch_rad"])) for row in rows),
      "maximum_abs_vertical_velocity_m_s": max(abs(float(row["world_vz_m_s"])) for row in rows),
      "maximum_abs_body_vy_m_s": max(abs(float(row["body_vy_m_s"])) for row in rows),
      "maximum_left_normal_load_n": max(float(row["left_normal_load_n"]) for row in rows),
      "maximum_right_normal_load_n": max(float(row["right_normal_load_n"]) for row in rows),
      "maximum_total_normal_load_n": max(float(row["total_normal_load_n"]) for row in rows),
      "left_unsupported_sample_count": sum(not int(row["left_supported"]) for row in rows),
      "right_unsupported_sample_count": sum(not int(row["right_supported"]) for row in rows),
      "maximum_abs_custom_force_normal_component_n": max(abs(float(row["max_abs_custom_force_normal_component_n"])) for row in rows),
      "maximum_abs_left_longitudinal_force_n": max(abs(float(row["left_longitudinal_force_n"])) for row in rows),
      "maximum_abs_right_longitudinal_force_n": max(abs(float(row["right_longitudinal_force_n"])) for row in rows),
      "force_guard_n": force_guard,
      "near_force_guard_sample_count": sum(max(abs(float(row["left_longitudinal_force_n"])), abs(float(row["right_longitudinal_force_n"]))) >= near_force for row in rows),
      "maximum_abs_applied_yaw_moment_n_m": max(abs(float(row["applied_yaw_moment_n_m"])) for row in rows)},
    "calibration_warning": stage["metadata"]["calibration_warning"],
    "source_frame_limitation": "The route is LiDAR sensor-center T_map_lidar; exact T_base_lidar is unknown. Isaac tracks a body/base state, so this is integration validation, not exact physical replay.",
    "flat_yaw_rate_limitation": "body-frame wz is used as the flat-ground yaw-rate proxy only."}
  return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def plot_results(rows: list[dict[str, Any]], path: Path) -> str:
  try:
    import matplotlib.pyplot as plt
  except ImportError as error:
    return f"skipped; matplotlib unavailable: {error}"
  figure, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
  time = [float(row["physics_time_s"]) for row in rows]
  axes[0, 0].plot([row["reference_x_m"] for row in rows], [row["reference_y_m"] for row in rows], "k--", label="reference")
  axes[0, 0].plot([row["actual_local_x_m"] for row in rows], [row["actual_local_y_m"] for row in rows], label="Isaac")
  axes[0, 0].set_aspect("equal", adjustable="box"); axes[0, 0].legend(); axes[0, 0].set_title("Map route XY")
  axes[0, 1].plot(time, [row["e_y_m"] for row in rows], label="e_y")
  axes[0, 1].plot(time, [row["heading_error_rad"] for row in rows], label="heading")
  axes[0, 1].legend(); axes[0, 1].set_title("Body errors")
  axes[1, 0].plot(time, [row["v_ref_m_s"] for row in rows], label="ref")
  axes[1, 0].plot(time, [row["v_cmd_m_s"] for row in rows], label="cmd")
  axes[1, 0].plot(time, [row["v_actual_m_s"] for row in rows], label="actual")
  axes[1, 0].legend(); axes[1, 0].set_title("Longitudinal velocity")
  axes[1, 1].plot(time, [row["omega_ref_rad_s"] for row in rows], label="ref")
  axes[1, 1].plot(time, [row["omega_cmd_rad_s"] for row in rows], label="cmd")
  axes[1, 1].plot(time, [row["omega_actual_rad_s"] for row in rows], label="actual")
  axes[1, 1].legend(); axes[1, 1].set_title("Yaw rate")
  axes[2, 0].plot(time, [row["v_left_cmd_m_s"] for row in rows], label="left cmd")
  axes[2, 0].plot(time, [row["v_left_state_m_s"] for row in rows], label="left state")
  axes[2, 0].plot(time, [row["v_right_cmd_m_s"] for row in rows], label="right cmd")
  axes[2, 0].plot(time, [row["v_right_state_m_s"] for row in rows], label="right state")
  axes[2, 0].legend(); axes[2, 0].set_title("Track command/state")
  axes[2, 1].plot(time, [row["body_vy_m_s"] for row in rows], label="body vy")
  axes[2, 1].plot(time, [row["command_scale"] for row in rows], label="command scale")
  axes[2, 1].legend(); axes[2, 1].set_title("Lateral response / saturation")
  for axis in axes.flat: axis.grid(True, alpha=.25); axis.set_xlabel("time [s]")
  figure.savefig(path, dpi=160); plt.close(figure)
  return str(path)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/map_route_isaac_closed_loop_v1.json")
  parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs/map_route_isaac_closed_loop_v1")
  parser.add_argument("--gui", action="store_true"); parser.add_argument("--realtime", action="store_true")
  parser.add_argument("--plot", action="store_true"); args = parser.parse_args()
  if args.realtime and not args.gui: parser.error("--realtime requires --gui")
  from isaacsim import SimulationApp
  app = SimulationApp({"headless": not args.gui, "enable_cameras": args.gui})
  try:
    stage_path = args.config.resolve(); stage = load_stage_config(stage_path)
    reference_result = build_map_route_reference(REPO_ROOT / stage["baseline_configs"]["map_route_reference"])
    _, controller_config = load_direction_aware_config(REPO_ROOT / stage["baseline_configs"]["direction_aware_controller"])
    plant_config = load_plant_config(REPO_ROOT / stage["baseline_configs"]["tracked_force_plant_v2"])
    if float(controller_config["measured_fixed"]["track_center_distance_m"]) != float(plant_config["measured_fixed"]["track_center_distance_b_m"]):
      raise RuntimeError("controller/plant track-center spacing mismatch")
    support = plant_config["tunable_uncalibrated"]["support_mode"]
    if support != stage["runtime"]["support_mode_required"] or support != "flat_track_boxes":
      raise RuntimeError(f"Stage E requires flat_track_boxes, got {support}")
    kinematic_path = REPO_ROOT / stage["comparison"]["kinematic_v2_summary"]
    with kinematic_path.open(encoding="utf-8") as stream: kinematic_summary = json.load(stream)
    output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    world, plant_controller, build_info = make_world(
      plant_config, output_dir / stage["output"]["stage_filename"], stage["runtime"]["creation_order"],
      bool(stage["runtime"]["detailed_contacts"]), support)
    configs = {"stage": stage, "controller": controller_config, "plant": plant_config}
    rows, settled, final = run_closed_loop(world, plant_controller, reference_result["trajectory"], configs, args.gui, args.realtime)
    summary = summarize(rows, reference_result["trajectory"], configs, settled, final, kinematic_summary)
    summary["stage_config_path"] = str(stage_path); summary["plant_build"] = build_info
    csv_path = output_dir / stage["output"]["csv_filename"]
    summary_path = output_dir / stage["output"]["summary_filename"]
    write_csv(csv_path, rows)
    with summary_path.open("w", encoding="utf-8") as stream: json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
    plot_status = "disabled" if not args.plot else plot_results(rows, output_dir / stage["output"]["plot_filename"])
    timing = summary["control_timing"]
    print(f"Stage E nominal complete: physics_dt={summary['physics_timestep_s']:.12g}, control={summary['controller_target_period_s']:.12g}, final={summary['actual_final_simulation_time_s']:.9f}")
    print(f"updates={timing['update_count']}, interval={timing['minimum_actual_update_interval_s']:.9f}..{timing['maximum_actual_update_interval_s']:.9f}, mean={timing['mean_actual_update_interval_s']:.9f}")
    print(f"active XY RMS={summary['active']['rms_time_aligned_xy_error_m']:.6f}, CTE RMS={summary['active']['rms_cross_track_error_m']:.6f}, final goal={summary['final_xy_goal_error_m']:.6f}")
    print(f"csv={csv_path}\nsummary={summary_path}\nplot={plot_status}")
    return 0
  except BaseException:
    traceback.print_exc(); raise
  finally:
    app.close()


if __name__ == "__main__":
  raise SystemExit(main())
