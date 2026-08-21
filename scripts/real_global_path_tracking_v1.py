#!/usr/bin/env python3
"""Pure real-pose shadow adapter for the validated closed-loop tracking core."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from global_path_demo_v1 import build_demo_reference
from tracking_controller_v1 import body_frame_errors, differential_track_speeds, wrap_to_pi
from tracking_controller_v2_direction_aware import controller_command, load_direction_aware_config


REPO_ROOT = Path(__file__).resolve().parents[1]
START_POSE_MODE = "START_POSE_MODE"
NEAREST_PATH_MODE = "NEAREST_PATH_MODE"
START_MODES = (START_POSE_MODE, NEAREST_PATH_MODE)
ROW_SOURCE_POSE_UPDATE = "POSE_UPDATE"
ROW_SOURCE_STATUS_TIMER = "STATUS_TIMER"
ROW_SOURCE_RECORDED_POSE = "RECORDED_POSE"
ROW_SOURCES = (ROW_SOURCE_POSE_UPDATE, ROW_SOURCE_STATUS_TIMER, ROW_SOURCE_RECORDED_POSE)

CSV_COLUMNS = (
  "timestamp", "localization_timestamp", "localization_age_s",
  "actual_map_x", "actual_map_y", "actual_yaw",
  "reference_x", "reference_y", "reference_yaw", "reference_s",
  "reference_v", "reference_omega", "nearest_path_s",
  "e_x", "e_y", "cross_track_error", "heading_error",
  "canonical_v_cmd", "canonical_omega_cmd", "canonical_v_left", "canonical_v_right",
  "real_safe_v_cmd", "real_safe_omega_cmd", "real_safe_v_left", "real_safe_v_right",
  "command_scale", "localization_valid", "safety_reason", "shadow_only", "row_source",
  "preset", "start_mode", "nearest_path_s_within_lap", "nearest_path_yaw",
  "nearest_path_distance", "projection_segment_index", "canonical_command_scale",
  "canonical_candidate_computed", "real_safety_scale", "localization_state",
  "start_xy_distance", "start_heading_error", "pose_jump_m", "pose_jump_yaw_rad",
  "start_alignment_state", "arm_state", "publication_state",
)


class RealTrackingError(ValueError):
  """Raised when the real-shadow configuration or pose data is invalid."""


@dataclass(frozen=True)
class LocalizationSample:
  """ROS-independent representation of one PoseStamped sample."""

  timestamp_s: float
  frame_id: str
  x_m: float
  y_m: float
  z_m: float
  qx: float
  qy: float
  qz: float
  qw: float


def quaternion_from_yaw(yaw_rad: float) -> tuple[float, float, float, float]:
  return 0.0, 0.0, math.sin(0.5 * yaw_rad), math.cos(0.5 * yaw_rad)


def sample_from_xy_yaw(timestamp_s: float, x_m: float, y_m: float, yaw_rad: float,
                       frame_id: str = "map") -> LocalizationSample:
  qx, qy, qz, qw = quaternion_from_yaw(yaw_rad)
  return LocalizationSample(timestamp_s, frame_id, x_m, y_m, 0.0, qx, qy, qz, qw)


def load_real_tracking_config(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    config = json.load(stream)
  runtime = config["runtime"]
  safety = config["initial_safety_envelope_not_calibrated"]
  if (runtime["preset"] != "rounded_loop" or runtime["pose_topic"] != "/localization/pose" or
      runtime["expected_frame_id"] != "map" or runtime["pose_semantics"] != "T_map_lidar"):
    raise RealTrackingError("Real Tracking V1 canonical pose/preset semantics changed")
  if (not runtime["shadow_only"] or runtime["create_command_publisher"] or
      runtime["arm_state"] != "NOT_ARMABLE"):
    raise RealTrackingError("Real Tracking V1 must remain shadow-only and not armable")
  if tuple(runtime["evaluated_start_modes"]) != START_MODES:
    raise RealTrackingError("both start modes must be evaluated in canonical order")
  if safety["status"] != "INITIAL SAFETY CAP / NOT CALIBRATED":
    raise RealTrackingError("real safety limits lost their non-calibrated label")
  for key in ("maximum_real_body_speed_m_s", "maximum_real_yaw_rate_rad_s",
              "maximum_real_track_speed_m_s", "maximum_localization_age_s",
              "maximum_future_timestamp_lead_s", "maximum_pose_jump_m",
              "maximum_pose_jump_yaw_rad", "minimum_quaternion_norm",
              "maximum_quaternion_norm"):
    value = float(safety[key])
    if not math.isfinite(value) or value <= 0.0:
      raise RealTrackingError(f"{key} must be positive and finite")
  if safety["startup_alignment"]["pass_thresholds_defined"]:
    raise RealTrackingError("start-alignment PASS thresholds are prohibited in Shadow V1")
  if config["future_command_layer"]["implemented"]:
    raise RealTrackingError("future command layer must not be implemented in Shadow V1")
  for relative in config["baseline_configs"].values():
    if not (REPO_ROOT / relative).is_file():
      raise RealTrackingError(f"missing baseline file: {relative}")
  return config


def _map_reference(source: dict[str, Any]) -> dict[str, Any]:
  row = dict(source)
  row.update({"x_m": float(source["map_x_m"]), "y_m": float(source["map_y_m"]),
              "yaw_rad": float(source["yaw_map_rad"])})
  return row


def build_shadow_context(config_path: Path) -> dict[str, Any]:
  config = load_real_tracking_config(config_path)
  demo_path = REPO_ROOT / config["baseline_configs"]["global_path_demo"]
  demo = build_demo_reference(demo_path, config["runtime"]["preset"])
  controller_path = REPO_ROOT / config["baseline_configs"]["direction_aware_controller"]
  _, controller_config = load_direction_aware_config(controller_path)
  spacing = float(controller_config["measured_fixed"]["track_center_distance_m"])
  canonical_limits = controller_config["command_limits"]
  safety = config["initial_safety_envelope_not_calibrated"]
  if not (float(safety["maximum_real_body_speed_m_s"]) <
          float(canonical_limits["maximum_abs_body_speed_m_s"]) and
          float(safety["maximum_real_yaw_rate_rad_s"]) <
          float(canonical_limits["maximum_abs_yaw_rate_rad_s"]) and
          float(safety["maximum_real_track_speed_m_s"]) <
          float(canonical_limits["maximum_abs_track_surface_speed_m_s"])):
    raise RealTrackingError("initial real safety caps must be below all simulation limits")
  periodic = [_map_reference(row) for row in demo["periodic_lap"]]
  start = _map_reference(demo["trajectory"][0])
  return {"config": config, "demo": demo, "controller": controller_config,
          "track_center_distance_m": spacing, "periodic_map_reference": periodic,
          "start_map_reference": start,
          "lap_length_m": float(periodic[-1]["s_m"])}


def _progress_near(s_within_lap: float, previous_total_s: float | None,
                   lap_length: float) -> float:
  if previous_total_s is None:
    return s_within_lap
  base_lap = math.floor(previous_total_s / lap_length)
  candidates = [s_within_lap + (base_lap + offset) * lap_length
                for offset in (-1, 0, 1, 2)]
  return min(candidates, key=lambda value: (abs(value - previous_total_s), value))


def project_closed_path(x_m: float, y_m: float, rows: list[dict[str, Any]],
                        previous_total_s: float | None = None) -> dict[str, Any]:
  """Nearest continuous-segment projection with lap-boundary progress unwrapping."""
  if len(rows) < 3:
    raise RealTrackingError("closed reference requires at least two segments and an endpoint")
  lap_length = float(rows[-1]["s_m"])
  best: dict[str, Any] | None = None
  for index, (first, second) in enumerate(zip(rows, rows[1:])):
    dx = float(second["x_m"]) - float(first["x_m"])
    dy = float(second["y_m"]) - float(first["y_m"])
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-20:
      continue
    fraction = max(0.0, min(1.0, ((x_m - float(first["x_m"])) * dx +
                                  (y_m - float(first["y_m"])) * dy) / length_squared))
    projected_x = float(first["x_m"]) + fraction * dx
    projected_y = float(first["y_m"]) + fraction * dy
    offset_x, offset_y = x_m - projected_x, y_m - projected_y
    distance_squared = offset_x * offset_x + offset_y * offset_y
    s_within = float(first["s_m"]) + fraction * (
      float(second["s_m"]) - float(first["s_m"]))
    if abs(s_within - lap_length) <= 1e-10:
      s_within = 0.0
    total_s = _progress_near(s_within, previous_total_s, lap_length)
    yaw = float(first["yaw_rad"]) + fraction * (
      float(second["yaw_rad"]) - float(first["yaw_rad"]))
    if s_within == 0.0 and index == len(rows) - 2:
      yaw = float(rows[0]["yaw_rad"])
    segment_length = math.sqrt(length_squared)
    candidate = {
      "projected_x_m": projected_x, "projected_y_m": projected_y,
      "s_within_lap_m": s_within, "s_total_m": total_s,
      "path_yaw_rad": yaw,
      "cross_track_error_m": (dx * offset_y - dy * offset_x) / segment_length,
      "distance_m": math.sqrt(distance_squared), "distance_squared": distance_squared,
      "projection_segment_index": index,
      "v_ref_m_s": float(first["v_ref_m_s"]) + fraction * (
        float(second["v_ref_m_s"]) - float(first["v_ref_m_s"])),
      "omega_ref_rad_s": float(first["omega_ref_rad_s"]) + fraction * (
        float(second["omega_ref_rad_s"]) - float(first["omega_ref_rad_s"])),
    }
    continuity = abs(total_s - previous_total_s) if previous_total_s is not None else s_within
    if (best is None or distance_squared < float(best["distance_squared"]) - 1e-15 or
        (abs(distance_squared - float(best["distance_squared"])) <= 1e-15 and
         (continuity, index) < (float(best["continuity_score"]),
                                int(best["projection_segment_index"])))):
      candidate["continuity_score"] = continuity
      best = candidate
  if best is None:
    raise RealTrackingError("closed path contains no projectable segment")
  best.pop("distance_squared")
  best.pop("continuity_score")
  return best


def validate_localization(sample: LocalizationSample | None, now_s: float,
                          previous_valid_pose: dict[str, float] | None,
                          config: dict[str, Any]) -> dict[str, Any]:
  safety = config["initial_safety_envelope_not_calibrated"]
  result: dict[str, Any] = {"valid": False, "pose_usable": False,
                            "reasons": [], "age_s": None, "pose": None,
                            "pose_jump_m": None, "pose_jump_yaw_rad": None,
                            "quaternion_norm": None}
  if sample is None:
    result["reasons"].append("NO_LOCALIZATION")
    return result
  numeric = (now_s, sample.timestamp_s, sample.x_m, sample.y_m, sample.z_m,
             sample.qx, sample.qy, sample.qz, sample.qw)
  if not all(math.isfinite(float(value)) for value in numeric):
    result["reasons"].append("NONFINITE_POSE")
    return result
  if sample.frame_id != config["runtime"]["expected_frame_id"]:
    result["reasons"].append("WRONG_FRAME")
  norm = math.sqrt(sample.qx * sample.qx + sample.qy * sample.qy +
                   sample.qz * sample.qz + sample.qw * sample.qw)
  result["quaternion_norm"] = norm
  if not (float(safety["minimum_quaternion_norm"]) <= norm <=
          float(safety["maximum_quaternion_norm"])):
    result["reasons"].append("INVALID_QUATERNION")
    return result
  qx, qy, qz, qw = (sample.qx / norm, sample.qy / norm, sample.qz / norm, sample.qw / norm)
  yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                   1.0 - 2.0 * (qy * qy + qz * qz))
  pose = {"x_m": sample.x_m, "y_m": sample.y_m, "yaw_rad": yaw,
          "timestamp_s": sample.timestamp_s}
  result["pose"] = pose
  result["pose_usable"] = True
  age = now_s - sample.timestamp_s
  result["age_s"] = age
  if age > float(safety["maximum_localization_age_s"]):
    result["reasons"].append("STALE_LOCALIZATION")
  if age < -float(safety["maximum_future_timestamp_lead_s"]):
    result["reasons"].append("FUTURE_LOCALIZATION_TIMESTAMP")
  if previous_valid_pose is not None:
    jump = math.hypot(sample.x_m - previous_valid_pose["x_m"],
                      sample.y_m - previous_valid_pose["y_m"])
    yaw_jump = abs(wrap_to_pi(yaw - previous_valid_pose["yaw_rad"]))
    result["pose_jump_m"] = jump
    result["pose_jump_yaw_rad"] = yaw_jump
    if jump > float(safety["maximum_pose_jump_m"]):
      result["reasons"].append("POSE_JUMP_TRANSLATION")
    if yaw_jump > float(safety["maximum_pose_jump_yaw_rad"]):
      result["reasons"].append("POSE_JUMP_YAW")
  result["valid"] = not result["reasons"]
  return result


def start_alignment_audit(pose: dict[str, float], context: dict[str, Any],
                          previous_total_s: float | None = None) -> dict[str, Any]:
  start = context["start_map_reference"]
  nearest = project_closed_path(pose["x_m"], pose["y_m"],
                                context["periodic_map_reference"], previous_total_s)
  nearest_reference = {"x_m": nearest["projected_x_m"],
                       "y_m": nearest["projected_y_m"],
                       "yaw_rad": nearest["path_yaw_rad"]}
  nearest_errors = body_frame_errors((pose["x_m"], pose["y_m"], pose["yaw_rad"]),
                                     nearest_reference)
  return {
    "current_map_x_m": pose["x_m"], "current_map_y_m": pose["y_m"],
    "current_yaw_rad": pose["yaw_rad"],
    "reference_start_map_x_m": float(start["x_m"]),
    "reference_start_map_y_m": float(start["y_m"]),
    "reference_start_yaw_rad": float(start["yaw_rad"]),
    "start_xy_distance_m": math.hypot(float(start["x_m"]) - pose["x_m"],
                                        float(start["y_m"]) - pose["y_m"]),
    "start_heading_error_rad": wrap_to_pi(float(start["yaw_rad"]) - pose["yaw_rad"]),
    "nearest": nearest, "nearest_e_x_m": nearest_errors["e_x_m"],
    "nearest_e_y_m": nearest_errors["e_y_m"],
    "nearest_heading_error_rad": nearest_errors["e_heading_rad"],
    "alignment_pass_thresholds_defined": False,
    "alignment_state": "START_ALIGNMENT_NOT_CHECKED",
  }


def real_safety_clamp(v_sim: float, omega_sim: float, spacing: float,
                      config: dict[str, Any], localization_valid: bool) -> dict[str, float]:
  if not localization_valid:
    return {"v_safe_m_s": 0.0, "omega_safe_rad_s": 0.0,
            "v_left_safe_m_s": 0.0, "v_right_safe_m_s": 0.0,
            "real_safety_scale": 0.0}
  safety = config["initial_safety_envelope_not_calibrated"]
  left, right = differential_track_speeds(v_sim, omega_sim, spacing)
  ratios = [1.0]
  for value, limit in (
      (v_sim, float(safety["maximum_real_body_speed_m_s"])),
      (omega_sim, float(safety["maximum_real_yaw_rate_rad_s"])),
      (left, float(safety["maximum_real_track_speed_m_s"])),
      (right, float(safety["maximum_real_track_speed_m_s"]))):
    if abs(value) > 0.0:
      ratios.append(limit / abs(value))
  scale = min(ratios)
  v_safe, omega_safe = scale * v_sim, scale * omega_sim
  left_safe, right_safe = differential_track_speeds(v_safe, omega_safe, spacing)
  return {"v_safe_m_s": v_safe, "omega_safe_rad_s": omega_safe,
          "v_left_safe_m_s": left_safe, "v_right_safe_m_s": right_safe,
          "real_safety_scale": scale}


def _zero_canonical_candidate() -> dict[str, Any]:
  return {"v_cmd_m_s": 0.0, "omega_cmd_rad_s": 0.0,
          "v_left_cmd_m_s": 0.0, "v_right_cmd_m_s": 0.0,
          "command_scale": 0.0, "computed": False,
          "e_x_m": None, "e_y_m": None, "e_heading_rad": None}


def _mode_reference(mode: str, context: dict[str, Any],
                    alignment: dict[str, Any]) -> dict[str, Any]:
  if mode == START_POSE_MODE:
    return dict(context["start_map_reference"])
  nearest = alignment["nearest"]
  return {"x_m": nearest["projected_x_m"], "y_m": nearest["projected_y_m"],
          "yaw_rad": nearest["path_yaw_rad"], "s_m": nearest["s_within_lap_m"],
          "v_ref_m_s": nearest["v_ref_m_s"],
          "omega_ref_rad_s": nearest["omega_ref_rad_s"]}


def _shadow_row(mode: str, sample: LocalizationSample | None, now_s: float,
                validation: dict[str, Any], alignment: dict[str, Any] | None,
                context: dict[str, Any], row_source: str) -> dict[str, Any]:
  config = context["config"]
  pose = validation["pose"] if validation["pose_usable"] else None
  if alignment is not None:
    reference = _mode_reference(mode, context, alignment)
    command = controller_command((pose["x_m"], pose["y_m"], pose["yaw_rad"]),
                                 reference, 1, context["controller"])
    command["computed"] = True
    nearest = alignment["nearest"]
    cross_track = nearest["cross_track_error_m"]
  else:
    reference = dict(context["start_map_reference"])
    command = _zero_canonical_candidate()
    nearest = None
    cross_track = None
  safe = real_safety_clamp(float(command["v_cmd_m_s"]),
                           float(command["omega_cmd_rad_s"]),
                           float(context["track_center_distance_m"]), config,
                           bool(validation["valid"] and command["computed"]))
  reasons = validation["reasons"]
  safety_reason = "LOCALIZATION_OK" if not reasons else "+".join(reasons)
  localization_state = ("LOCALIZATION_OK" if validation["valid"] else
                        ("STALE" if "STALE_LOCALIZATION" in reasons else "INVALID"))
  return {
    "timestamp": now_s,
    "localization_timestamp": None if sample is None else sample.timestamp_s,
    "localization_age_s": validation["age_s"],
    "actual_map_x": None if pose is None else pose["x_m"],
    "actual_map_y": None if pose is None else pose["y_m"],
    "actual_yaw": None if pose is None else pose["yaw_rad"],
    "reference_x": float(reference["x_m"]), "reference_y": float(reference["y_m"]),
    "reference_yaw": float(reference["yaw_rad"]), "reference_s": float(reference["s_m"]),
    "reference_v": float(reference["v_ref_m_s"]),
    "reference_omega": float(reference["omega_ref_rad_s"]),
    "nearest_path_s": None if nearest is None else nearest["s_total_m"],
    "e_x": command["e_x_m"], "e_y": command["e_y_m"],
    "cross_track_error": cross_track, "heading_error": command["e_heading_rad"],
    "canonical_v_cmd": command["v_cmd_m_s"],
    "canonical_omega_cmd": command["omega_cmd_rad_s"],
    "canonical_v_left": command["v_left_cmd_m_s"],
    "canonical_v_right": command["v_right_cmd_m_s"],
    "real_safe_v_cmd": safe["v_safe_m_s"],
    "real_safe_omega_cmd": safe["omega_safe_rad_s"],
    "real_safe_v_left": safe["v_left_safe_m_s"],
    "real_safe_v_right": safe["v_right_safe_m_s"],
    "command_scale": safe["real_safety_scale"],
    "localization_valid": int(validation["valid"]), "safety_reason": safety_reason,
    "shadow_only": 1, "row_source": row_source,
    "preset": config["runtime"]["preset"], "start_mode": mode,
    "nearest_path_s_within_lap": None if nearest is None else nearest["s_within_lap_m"],
    "nearest_path_yaw": None if nearest is None else nearest["path_yaw_rad"],
    "nearest_path_distance": None if nearest is None else nearest["distance_m"],
    "projection_segment_index": None if nearest is None else nearest["projection_segment_index"],
    "canonical_command_scale": command["command_scale"],
    "canonical_candidate_computed": int(command["computed"]),
    "real_safety_scale": safe["real_safety_scale"],
    "localization_state": localization_state,
    "start_xy_distance": None if alignment is None else alignment["start_xy_distance_m"],
    "start_heading_error": None if alignment is None else alignment["start_heading_error_rad"],
    "pose_jump_m": validation["pose_jump_m"],
    "pose_jump_yaw_rad": validation["pose_jump_yaw_rad"],
    "start_alignment_state": "START_ALIGNMENT_NOT_CHECKED", "arm_state": "NOT_ARMABLE",
    "publication_state": "NO_CMD_VEL_PUBLISHER",
  }


class ShadowSession:
  """Stateful pose-jump and lap-boundary context around the pure adapter."""

  def __init__(self, config_path: Path):
    self.context = build_shadow_context(config_path)
    self.previous_valid_pose: dict[str, float] | None = None
    self.previous_nearest_total_s: float | None = None

  def evaluate(self, sample: LocalizationSample | None, now_s: float,
               update_history: bool = True,
               row_source: str = ROW_SOURCE_POSE_UPDATE) -> dict[str, Any]:
    if row_source not in ROW_SOURCES:
      raise RealTrackingError(f"invalid shadow row_source: {row_source}")
    validation = validate_localization(sample, now_s, self.previous_valid_pose,
                                       self.context["config"])
    alignment = None
    if validation["pose_usable"]:
      alignment = start_alignment_audit(validation["pose"], self.context,
                                        self.previous_nearest_total_s)
    rows = [_shadow_row(mode, sample, now_s, validation, alignment, self.context, row_source)
            for mode in START_MODES]
    if update_history and validation["valid"] and alignment is not None:
      self.previous_valid_pose = dict(validation["pose"])
      self.previous_nearest_total_s = float(alignment["nearest"]["s_total_m"])
    return {"validation": validation, "alignment": alignment, "rows": rows,
            "shadow_only": True, "arm_state": "NOT_ARMABLE",
            "publication_state": "NO_CMD_VEL_PUBLISHER"}


def write_shadow_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)


def format_shadow_status(result: dict[str, Any]) -> str:
  validation = result["validation"]
  pose = validation["pose"]
  if pose is None:
    localization = f"Localization: {result['rows'][0]['safety_reason']}"
  else:
    age = validation["age_s"]
    localization = (f"Localization: {'fresh' if validation['valid'] else 'unsafe'} "
                    f"x={pose['x_m']:.3f} y={pose['y_m']:.3f} yaw={pose['yaw_rad']:.3f} "
                    f"age={age:.3f}s")
  lines = ["REAL TRACKING V1 — SHADOW ONLY", "NO CMD_VEL PUBLISHER", localization]
  for row in result["rows"]:
    lines.extend([
      f"[{row['start_mode']}] preset={row['preset']} reference="
      f"({row['reference_x']:.3f}, {row['reference_y']:.3f}, {row['reference_yaw']:.3f}) "
      f"nearest_s={row['nearest_path_s']}",
      f"Errors: e_x={row['e_x']} e_y={row['e_y']} heading={row['heading_error']} "
      f"CTE={row['cross_track_error']}",
      f"Canonical: v={row['canonical_v_cmd']:.3f} omega={row['canonical_omega_cmd']:.3f} "
      f"left={row['canonical_v_left']:.3f} right={row['canonical_v_right']:.3f}",
      f"Real-safe: v={row['real_safe_v_cmd']:.3f} omega={row['real_safe_omega_cmd']:.3f} "
      f"left={row['real_safe_v_left']:.3f} right={row['real_safe_v_right']:.3f}",
    ])
  lines.append(f"Safety: {result['rows'][0]['safety_reason']} | START_ALIGNMENT_NOT_CHECKED | "
               "NOT_ARMABLE | SHADOW_ONLY")
  return "\n".join(lines)
