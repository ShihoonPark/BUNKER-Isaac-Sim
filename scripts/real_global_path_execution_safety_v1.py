#!/usr/bin/env python3
"""Pure offline trajectory executor and command-safety supervisor for real tracking."""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from global_path_demo_v1 import _phase, _times_and_accelerations
from real_global_path_tracking_v1 import (
  CSV_COLUMNS as SHADOW_CSV_COLUMNS,
  NEAREST_PATH_MODE,
  REPO_ROOT,
  ROW_SOURCE_POSE_UPDATE,
  ROW_SOURCE_RECORDED_POSE,
  ROW_SOURCE_STATUS_TIMER,
  START_POSE_MODE,
  LocalizationSample,
  build_shadow_context,
  real_safety_clamp,
  start_alignment_audit,
  validate_localization,
)
from tracking_controller_v1 import ReferenceInterpolator, differential_track_speeds, wrap_to_pi
from tracking_controller_v2_direction_aware import controller_command


DISARMED = "DISARMED"
READY = "READY"
RUNNING = "RUNNING"
STOPPING = "STOPPING"
COMPLETE = "COMPLETE"
FAULT = "FAULT"
EXECUTION_STATES = (DISARMED, READY, RUNNING, STOPPING, COMPLETE, FAULT)
ROW_SOURCE_OFFLINE_SYNTHETIC = "OFFLINE_SYNTHETIC"
EXECUTION_ROW_SOURCES = (ROW_SOURCE_POSE_UPDATE, ROW_SOURCE_STATUS_TIMER,
                         ROW_SOURCE_RECORDED_POSE, ROW_SOURCE_OFFLINE_SYNTHETIC)

EXECUTION_ONLY_CSV_COLUMNS = (
  "execution_state", "execution_time_s", "execution_dt_s", "reference_index",
  "reference_next_index", "reference_interpolation_fraction", "reference_progress_m",
  "reference_lap_index", "planned_laps", "reference_motion_phase",
  "launch_envelope_active", "terminal_stop_envelope_active", "stop_reason",
  "fault_reason", "final_candidate_v", "final_candidate_omega",
  "final_candidate_left", "final_candidate_right", "final_candidate_allowed",
  "watchdog_state", "execution_transition_event", "reference_valid",
  "timing_valid", "physical_motion_enabled", "execution_plan_identity",
)
EXECUTION_CSV_COLUMNS = tuple(SHADOW_CSV_COLUMNS) + EXECUTION_ONLY_CSV_COLUMNS


class ExecutionSafetyError(ValueError):
  """Raised when the offline execution contract is inconsistent or misused."""


def _load_json(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    return json.load(stream)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(65536), b""):
      digest.update(block)
  return digest.hexdigest()


def _require_positive_finite(value: Any, name: str) -> float:
  number = float(value)
  if not math.isfinite(number) or number <= 0.0:
    raise ExecutionSafetyError(f"{name} must be positive and finite")
  return number


def load_execution_config(path: Path) -> dict[str, Any]:
  config = _load_json(path)
  metadata = config["metadata"]
  plan = config["execution_plan"]
  start = config["initial_start_alignment_gate_not_calibrated"]
  state = config["state_machine"]
  output = config["command_output"]
  readiness = config["physical_execution_readiness"]
  if metadata["physical_motion_enabled"] or output["physical_motion_enabled"]:
    raise ExecutionSafetyError("physical motion must remain disabled")
  if output["create_command_publisher"] or output["publish_twist"]:
    raise ExecutionSafetyError("command publication is prohibited in this Stage")
  if not output["diagnostic_only"] or output["kind"] != "FINAL COMMAND CANDIDATE":
    raise ExecutionSafetyError("output must remain a diagnostic final candidate")
  if (plan["preset"] != "rounded_loop" or int(plan["requested_laps"]) != 1 or
      int(plan["canonical_builder_laps"]) != 2):
    raise ExecutionSafetyError("offline physical preparation must be rounded_loop, one lap")
  if (plan["default_start_mode"] != START_POSE_MODE or
      not plan["nearest_path_mode_diagnostic_only"] or
      plan["nearest_path_execution_enabled"]):
    raise ExecutionSafetyError("START_POSE_MODE must be the only execution start policy")
  if int(plan["motion_direction"]) != 1:
    raise ExecutionSafetyError("rounded-loop execution must retain explicit forward direction")
  if start["status"] != "INITIAL SAFETY GATE / NOT CALIBRATED":
    raise ExecutionSafetyError("provisional alignment gate lost its non-calibrated label")
  _require_positive_finite(start["maximum_start_xy_error_m"], "maximum_start_xy_error_m")
  _require_positive_finite(start["maximum_start_heading_error_rad"],
                           "maximum_start_heading_error_rad")
  if tuple(state["states"]) != EXECUTION_STATES or state["default_state"] != DISARMED:
    raise ExecutionSafetyError("execution state inventory/default changed")
  if state["ready_is_physical_arm"] or not state["all_states_are_diagnostic_only"]:
    raise ExecutionSafetyError("READY and all other states must remain non-physical")
  if state["arm_state"] != "NOT_ARMABLE":
    raise ExecutionSafetyError("offline executor must remain NOT_ARMABLE")
  _require_positive_finite(config["deterministic_clock"]["nominal_step_s"],
                           "nominal_step_s")
  if config["deterministic_clock"]["status_timer_advances_execution"]:
    raise ExecutionSafetyError("STATUS_TIMER must never advance execution")
  if config["future_publisher_requirements"]["implemented"]:
    raise ExecutionSafetyError("future physical publisher must remain unimplemented")
  if (readiness["reference_timing_ready"] or readiness["future_publisher_ready"] or
      readiness["status"] != "BLOCKED_PENDING_REAL_COMMAND_CALIBRATION"):
    raise ExecutionSafetyError("physical reference timing/publisher readiness must remain blocked")
  if config["output"]["future_tracking_performance_filter"] != "row_source == POSE_UPDATE":
    raise ExecutionSafetyError("tracking-performance row filter changed")
  if set(config["output"]["row_source_semantics"]) != set(EXECUTION_ROW_SOURCES):
    raise ExecutionSafetyError("execution row-source semantics are incomplete")
  for relative in config["baseline_configs"].values():
    if not (REPO_ROOT / relative).is_file():
      raise ExecutionSafetyError(f"missing baseline config: {relative}")
  return config


def _build_one_lap_reference(shadow: dict[str, Any], config: dict[str, Any]
                             ) -> list[dict[str, Any]]:
  """Project canonical two-lap launch/stop envelopes onto one periodic lap."""
  periodic = shadow["periodic_map_reference"]
  runtime = shadow["demo"]["trajectory"]
  unique_count = len(periodic) - 1
  if (int(shadow["demo"]["summary"]["execution_profile"]["laps"]) != 2 or
      len(runtime) != 2 * unique_count + 1):
    raise ExecutionSafetyError("canonical runtime no longer has the expected two-lap structure")

  rows: list[dict[str, Any]] = []
  speeds: list[float] = []
  envelope_sources: list[tuple[float, float, float]] = []
  for index, source in enumerate(periodic):
    periodic_speed = float(source["v_ref_m_s"])
    if index < unique_count:
      launch_speed = float(runtime[index]["v_ref_m_s"])
      stop_speed = float(runtime[unique_count + index]["v_ref_m_s"])
    else:
      launch_speed = periodic_speed
      stop_speed = float(runtime[-1]["v_ref_m_s"])
    speed = min(periodic_speed, launch_speed, stop_speed)
    row = dict(source)
    row.update({
      "index": index,
      "lap_index": 0,
      "s_within_lap_m": float(source["s_m"]),
      "lap_progress_fraction": float(source["s_m"]) / float(periodic[-1]["s_m"]),
    })
    rows.append(row)
    speeds.append(speed)
    envelope_sources.append((periodic_speed, launch_speed, stop_speed))

  times, accelerations = _times_and_accelerations(rows, speeds)
  canonical = shadow["demo"]["canonical"]
  spacing = float(canonical["measured_fixed"]["track_center_distance_m"])
  tolerance = float(canonical["trajectory_limits"]["phase_speed_tolerance_m_s"])
  for index, row in enumerate(rows):
    periodic_speed, launch_speed, stop_speed = envelope_sources[index]
    speed = speeds[index]
    omega = speed * float(row["curvature_ref_1_m"])
    launch_active = launch_speed < periodic_speed - tolerance
    stop_active = stop_speed < min(periodic_speed, launch_speed) - tolerance
    row.update({
      "t_s": times[index],
      "dt_s": 0.0 if index == 0 else times[index] - times[index - 1],
      "v_ref_m_s": speed,
      "a_ref_m_s2": accelerations[index],
      "omega_ref_rad_s": omega,
      "v_left_ref_m_s": speed - 0.5 * spacing * omega,
      "v_right_ref_m_s": speed + 0.5 * spacing * omega,
      "launch_envelope_active": launch_active,
      "terminal_stop_envelope_active": stop_active,
      "canonical_periodic_v_m_s": periodic_speed,
      "canonical_launch_envelope_v_m_s": launch_speed,
      "canonical_terminal_stop_envelope_v_m_s": stop_speed,
      "motion_phase": (
        "launch" if index == 0 else
        "final_stop" if index == len(rows) - 1 else
        _phase(accelerations[index], row, canonical)
      ),
    })
  if abs(float(rows[0]["v_ref_m_s"])) > 1e-15 or abs(float(rows[-1]["v_ref_m_s"])) > 1e-15:
    raise ExecutionSafetyError("one-lap adapter must start and end at zero speed")
  return rows


def physical_reference_timing_diagnostics(
    reference: list[dict[str, Any]], real_caps: dict[str, Any], duration_s: float
    ) -> dict[str, Any]:
  """Quantify clock/cap incompatibility without generating a new trajectory."""
  reference_max_body = max(abs(float(row["v_ref_m_s"])) for row in reference)
  reference_max_yaw = max(abs(float(row["omega_ref_rad_s"])) for row in reference)
  reference_max_track = max(
    abs(float(row[key])) for row in reference
    for key in ("v_left_ref_m_s", "v_right_ref_m_s"))
  cap_body = _require_positive_finite(real_caps["maximum_real_body_speed_m_s"],
                                      "maximum_real_body_speed_m_s")
  cap_yaw = _require_positive_finite(real_caps["maximum_real_yaw_rate_rad_s"],
                                     "maximum_real_yaw_rate_rad_s")
  cap_track = _require_positive_finite(real_caps["maximum_real_track_speed_m_s"],
                                       "maximum_real_track_speed_m_s")
  ratios = {
    "body_speed_ratio": reference_max_body / cap_body,
    "yaw_rate_ratio": reference_max_yaw / cap_yaw,
    "track_speed_ratio": reference_max_track / cap_track,
  }
  lower_bound = max(ratios.values())
  return {
    "canonical_reference_max_body_speed_m_s": reference_max_body,
    "canonical_reference_max_yaw_rate_rad_s": reference_max_yaw,
    "canonical_reference_max_track_speed_m_s": reference_max_track,
    "provisional_real_safe_body_speed_cap_m_s": cap_body,
    "provisional_real_safe_yaw_rate_cap_rad_s": cap_yaw,
    "provisional_real_safe_track_speed_cap_m_s": cap_track,
    **ratios,
    "minimum_required_uniform_time_scale_lower_bound": lower_bound,
    "canonical_one_lap_duration_s": duration_s,
    "uniformly_scaled_duration_lower_bound_s": duration_s * lower_bound,
    "physical_clock_connection_allowed": False,
    "interpretation": (
      "Diagnostic incompatibility lower bound only; not an optimized or validated physical "
      "trajectory timing."),
  }


def assert_physical_reference_timing_ready(readiness: dict[str, Any],
                                           diagnostics: dict[str, Any]) -> None:
  """Guard a future publisher from accepting the current incompatible clock."""
  lower_bound = float(diagnostics["minimum_required_uniform_time_scale_lower_bound"])
  if lower_bound > 1.0 + 1e-12:
    raise ExecutionSafetyError(
      f"canonical reference clock exceeds provisional real-safe caps; "
      f"uniform time-scale lower bound={lower_bound:.6f}")
  if (not readiness["reference_timing_ready"] or
      not readiness["future_publisher_ready"] or
      readiness["status"] != "READY"):
    raise ExecutionSafetyError("physical reference timing/publisher readiness is not approved")


def build_execution_context(config_path: Path) -> dict[str, Any]:
  config_path = config_path.resolve()
  config = load_execution_config(config_path)
  shadow_path = REPO_ROOT / config["baseline_configs"]["real_tracking_shadow"]
  shadow = build_shadow_context(shadow_path)
  shadow_config = shadow["config"]
  if shadow_config["runtime"]["preset"] != config["execution_plan"]["preset"]:
    raise ExecutionSafetyError("Shadow and execution presets differ")
  watchdog = float(shadow_config["initial_safety_envelope_not_calibrated"]
                   ["future_command_timeout_watchdog"]["timeout_s"])
  _require_positive_finite(watchdog, "watchdog_timeout_s")
  caps = config["real_safety_caps_frozen"]
  shadow_caps = shadow_config["initial_safety_envelope_not_calibrated"]
  cap_pairs = (
    ("maximum_real_body_speed_m_s", "maximum_real_body_speed_m_s"),
    ("maximum_real_yaw_rate_rad_s", "maximum_real_yaw_rate_rad_s"),
    ("maximum_real_track_speed_m_s", "maximum_real_track_speed_m_s"),
  )
  if caps["status"] != shadow_caps["status"]:
    raise ExecutionSafetyError("real cap status differs from frozen Shadow V1")
  for execution_key, shadow_key in cap_pairs:
    if abs(float(caps[execution_key]) - float(shadow_caps[shadow_key])) > 1e-15:
      raise ExecutionSafetyError(f"real safety cap changed: {execution_key}")
  if float(config["deterministic_clock"]["nominal_step_s"]) >= watchdog:
    raise ExecutionSafetyError("nominal execution step must remain below watchdog timeout")

  reference = _build_one_lap_reference(shadow, config)
  timing_diagnostics = physical_reference_timing_diagnostics(
    reference, config["real_safety_caps_frozen"], float(reference[-1]["t_s"]))
  readiness = config["physical_execution_readiness"]
  if (readiness["reference_timing_ready"] or readiness["future_publisher_ready"] or
      not timing_diagnostics["minimum_required_uniform_time_scale_lower_bound"] > 1.0):
    raise ExecutionSafetyError("current physical reference timing blocker is not enforced")
  stop_indices = [index for index, row in enumerate(reference)
                  if row["terminal_stop_envelope_active"]]
  if not stop_indices:
    raise ExecutionSafetyError("canonical terminal stop envelope was not found")
  global_path_path = REPO_ROOT / config["baseline_configs"]["global_path_demo"]
  controller_v2_path = REPO_ROOT / config["baseline_configs"]["direction_aware_controller"]
  global_path_config = _load_json(global_path_path)
  controller_v2_config = _load_json(controller_v2_path)
  paths = {
    "execution_config": config_path,
    "shadow_config": shadow_path,
    "shadow_core_source": REPO_ROOT / "scripts/real_global_path_tracking_v1.py",
    "global_path_config": global_path_path,
    "global_path_source": REPO_ROOT / "scripts/global_path_demo_v1.py",
    "canonical_trajectory_config": REPO_ROOT /
      global_path_config["baseline_configs"]["trajectory_limits"],
    "controller_v2_config": controller_v2_path,
    "controller_v2_source": REPO_ROOT / "scripts/tracking_controller_v2_direction_aware.py",
    "controller_v1_config": REPO_ROOT / controller_v2_config["baseline_controller_config"],
    "controller_v1_source": REPO_ROOT / "scripts/tracking_controller_v1.py",
  }
  identities = {name: _sha256(path) for name, path in paths.items()}
  plan_identity = hashlib.sha256("|".join(
    f"{name}:{identities[name]}" for name in sorted(identities)).encode("utf-8")).hexdigest()
  return {
    "config": config,
    "shadow": shadow,
    "reference": reference,
    "duration_s": float(reference[-1]["t_s"]),
    "lap_length_m": float(reference[-1]["s_m"]),
    "stop_start_index": stop_indices[0],
    "stop_start_time_s": float(reference[stop_indices[0]]["t_s"]),
    "watchdog_timeout_s": watchdog,
    "physical_timing_diagnostics": timing_diagnostics,
    "config_sha256": identities,
    "execution_plan_identity": plan_identity,
  }


class ExecutionReferenceInterpolator:
  """Canonical continuous-yaw interpolation plus explicit interval location."""

  def __init__(self, rows: list[dict[str, Any]]):
    self.rows = rows
    self.base = ReferenceInterpolator(rows)
    self.times = [float(row["t_s"]) for row in rows]
    self.duration_s = self.times[-1]

  def sample(self, query_time_s: float) -> dict[str, Any]:
    if not math.isfinite(query_time_s):
      raise ExecutionSafetyError("execution reference time must be finite")
    clamped = max(0.0, min(self.duration_s, query_time_s))
    reference = self.base.sample(clamped)
    if clamped <= self.times[0]:
      left_index, next_index, fraction = 0, 1, 0.0
    elif clamped >= self.times[-1]:
      left_index = next_index = len(self.rows) - 1
      fraction = 0.0
    else:
      next_index = bisect.bisect_right(self.times, clamped)
      left_index = next_index - 1
      fraction = ((clamped - self.times[left_index]) /
                  (self.times[next_index] - self.times[left_index]))
    source = self.rows[left_index]
    reference.update({
      "reference_index": left_index,
      "reference_next_index": next_index,
      "reference_interpolation_fraction": fraction,
      "motion_phase": source["motion_phase"],
      "launch_envelope_active": bool(source["launch_envelope_active"]),
      "terminal_stop_envelope_active": bool(source["terminal_stop_envelope_active"]),
      "lap_index": 0,
      "query_time_s": query_time_s,
      "clamped_to_terminal": query_time_s >= self.duration_s,
    })
    return reference


def evaluate_start_alignment(sample: LocalizationSample | None, now_s: float,
                             context: dict[str, Any]) -> dict[str, Any]:
  validation = validate_localization(sample, now_s, None,
                                     context["shadow"]["config"])
  if not validation["pose_usable"]:
    return {"validation": validation, "audit": None, "start_pose_passed": False,
            "decision": "LOCALIZATION_INVALID"}
  audit = start_alignment_audit(validation["pose"], context["shadow"])
  gate = context["config"]["initial_start_alignment_gate_not_calibrated"]
  xy_passed = audit["start_xy_distance_m"] <= float(gate["maximum_start_xy_error_m"])
  heading_passed = abs(audit["start_heading_error_rad"]) <= float(
    gate["maximum_start_heading_error_rad"])
  passed = bool(validation["valid"] and xy_passed and heading_passed)
  audit.update({
    "alignment_pass_thresholds_defined": True,
    "alignment_gate_status": gate["status"],
    "maximum_start_xy_error_m": float(gate["maximum_start_xy_error_m"]),
    "maximum_start_heading_error_rad": float(gate["maximum_start_heading_error_rad"]),
    "start_xy_passed": xy_passed,
    "start_heading_passed": heading_passed,
    "start_pose_passed": passed,
    "alignment_state": "START_ALIGNMENT_PASS" if passed else "START_ALIGNMENT_FAIL",
  })
  return {"validation": validation, "audit": audit, "start_pose_passed": passed,
          "decision": audit["alignment_state"]}


def _zero_canonical(computed: bool = False) -> dict[str, Any]:
  return {
    "v_cmd_m_s": 0.0, "omega_cmd_rad_s": 0.0,
    "v_left_cmd_m_s": 0.0, "v_right_cmd_m_s": 0.0,
    "command_scale": 0.0, "computed": computed,
    "e_x_m": None, "e_y_m": None, "e_heading_rad": None,
  }


def _zero_safe() -> dict[str, float]:
  return {
    "v_safe_m_s": 0.0, "omega_safe_rad_s": 0.0,
    "v_left_safe_m_s": 0.0, "v_right_safe_m_s": 0.0,
    "real_safety_scale": 0.0,
  }


def zero_final_candidate(reason: str, allowed: bool = False) -> dict[str, Any]:
  return {
    "final_candidate_v_m_s": 0.0,
    "final_candidate_omega_rad_s": 0.0,
    "final_candidate_left_m_s": 0.0,
    "final_candidate_right_m_s": 0.0,
    "command_allowed_in_simulated_execution": allowed,
    "safety_reason": reason,
  }


def supervise_command(execution_state: str, localization_valid: bool,
                      timing_valid: bool, reference_valid: bool,
                      real_safe_candidate: dict[str, float],
                      safety_reason: str = "SAFETY_OK",
                      emergency_stop: bool = False,
                      output_scale: float = 1.0) -> dict[str, Any]:
  """Pure final-candidate policy; it cannot publish or actuate anything."""
  if execution_state not in EXECUTION_STATES:
    return zero_final_candidate("INVALID_EXECUTION_STATE")
  if emergency_stop:
    return zero_final_candidate("EXPLICIT_EMERGENCY_STOP")
  if execution_state in (DISARMED, READY, COMPLETE, FAULT):
    return zero_final_candidate(f"STATE_{execution_state}_ZERO")
  if not localization_valid:
    return zero_final_candidate(safety_reason)
  if not timing_valid:
    return zero_final_candidate(safety_reason)
  if not reference_valid:
    return zero_final_candidate("REFERENCE_INVALID")
  values = [float(real_safe_candidate[key]) for key in (
    "v_safe_m_s", "omega_safe_rad_s", "v_left_safe_m_s", "v_right_safe_m_s")]
  if not all(math.isfinite(value) for value in values):
    return zero_final_candidate("NONFINITE_REAL_SAFE_CANDIDATE")
  if not math.isfinite(output_scale) or not 0.0 <= output_scale <= 1.0:
    return zero_final_candidate("INVALID_ORDERLY_STOP_SCALE")
  return {
    "final_candidate_v_m_s": output_scale * values[0],
    "final_candidate_omega_rad_s": output_scale * values[1],
    "final_candidate_left_m_s": output_scale * values[2],
    "final_candidate_right_m_s": output_scale * values[3],
    "command_allowed_in_simulated_execution": True,
    "safety_reason": ("ORDERLY_STOP_DECELERATION" if output_scale < 1.0 else
                      "SIMULATED_EXECUTION_ALLOWED"),
  }


def command_sign_audit(track_center_distance_m: float = 0.434) -> dict[str, Any]:
  cases = {
    "straight_positive": (0.10, 0.0),
    "positive_yaw": (0.10, 0.20),
    "negative_yaw": (0.10, -0.20),
    "in_place_positive": (0.0, 0.20),
  }
  results: dict[str, Any] = {}
  for name, (v, omega) in cases.items():
    left, right = differential_track_speeds(v, omega, track_center_distance_m)
    results[name] = {"v_m_s": v, "omega_rad_s": omega,
                     "left_m_s": left, "right_m_s": right}
  checks = {
    "straight_both_positive": (results["straight_positive"]["left_m_s"] > 0.0 and
                               results["straight_positive"]["right_m_s"] > 0.0),
    "positive_yaw_right_greater": (results["positive_yaw"]["right_m_s"] >
                                   results["positive_yaw"]["left_m_s"]),
    "negative_yaw_left_greater": (results["negative_yaw"]["left_m_s"] >
                                  results["negative_yaw"]["right_m_s"]),
    "in_place_positive_opposed": (results["in_place_positive"]["left_m_s"] < 0.0 <
                                   results["in_place_positive"]["right_m_s"]),
  }
  return {"track_center_distance_m": track_center_distance_m, "cases": results,
          "checks": checks, "passed": all(checks.values()), "published": False}


def _reference_is_valid(reference: dict[str, Any]) -> bool:
  keys = ("t_s", "s_m", "x_m", "y_m", "yaw_rad", "v_ref_m_s", "omega_ref_rad_s")
  return all(math.isfinite(float(reference[key])) for key in keys)


class ExecutionSafetySession:
  """Explicit diagnostic-only execution state machine; no state is a physical ARM."""

  def __init__(self, config_path: Path):
    self.context = build_execution_context(config_path)
    self.interpolator = ExecutionReferenceInterpolator(self.context["reference"])
    self.state = DISARMED
    self.state_history = [DISARMED]
    self.execution_time_s = 0.0
    self.last_update_timestamp_s: float | None = None
    self.previous_valid_pose: dict[str, float] | None = None
    self.fault_reason = ""
    self.stop_reason = ""
    self.explicit_orderly_stop = False
    self.last_final_candidate = zero_final_candidate("INITIAL_DISARMED")
    self.last_result: dict[str, Any] | None = None

  def _transition(self, target: str) -> str:
    if target == self.state:
      return ""
    event = f"{self.state}->{target}"
    self.state = target
    self.state_history.append(target)
    return event

  def _alignment(self, validation: dict[str, Any]) -> dict[str, Any] | None:
    if not validation["pose_usable"]:
      return None
    result = start_alignment_audit(validation["pose"], self.context["shadow"])
    gate = self.context["config"]["initial_start_alignment_gate_not_calibrated"]
    xy_passed = result["start_xy_distance_m"] <= float(gate["maximum_start_xy_error_m"])
    heading_passed = abs(result["start_heading_error_rad"]) <= float(
      gate["maximum_start_heading_error_rad"])
    result.update({
      "start_xy_passed": xy_passed,
      "start_heading_passed": heading_passed,
      "start_pose_passed": xy_passed and heading_passed and validation["valid"],
      "alignment_state": ("START_ALIGNMENT_PASS" if
                          xy_passed and heading_passed and validation["valid"] else
                          "START_ALIGNMENT_FAIL"),
    })
    return result

  def _watchdog_state(self) -> str:
    if self.state == FAULT:
      return "WATCHDOG_FAULT"
    if self.state in (RUNNING, STOPPING):
      return "WATCHDOG_OK"
    return "WATCHDOG_NOT_ACTIVE"

  def _make_row(self, sample: LocalizationSample | None, now_s: float, dt_s: float,
                row_source: str, validation: dict[str, Any],
                alignment: dict[str, Any] | None, reference: dict[str, Any],
                canonical: dict[str, Any], safe: dict[str, float], final: dict[str, Any],
                timing_valid: bool, transition: str) -> dict[str, Any]:
    pose = validation["pose"] if validation["pose_usable"] else None
    nearest = None if alignment is None else alignment["nearest"]
    reasons = validation["reasons"]
    localization_state = ("LOCALIZATION_OK" if validation["valid"] else
                          ("STALE" if "STALE_LOCALIZATION" in reasons else "INVALID"))
    row = {
      "timestamp": now_s,
      "localization_timestamp": None if sample is None else sample.timestamp_s,
      "localization_age_s": validation["age_s"],
      "actual_map_x": None if pose is None else pose["x_m"],
      "actual_map_y": None if pose is None else pose["y_m"],
      "actual_yaw": None if pose is None else pose["yaw_rad"],
      "reference_x": float(reference["x_m"]),
      "reference_y": float(reference["y_m"]),
      "reference_yaw": float(reference["yaw_rad"]),
      "reference_s": float(reference["s_m"]),
      "reference_v": float(reference["v_ref_m_s"]),
      "reference_omega": float(reference["omega_ref_rad_s"]),
      "nearest_path_s": None if nearest is None else nearest["s_total_m"],
      "e_x": canonical["e_x_m"],
      "e_y": canonical["e_y_m"],
      "cross_track_error": None if nearest is None else nearest["cross_track_error_m"],
      "heading_error": canonical["e_heading_rad"],
      "canonical_v_cmd": canonical["v_cmd_m_s"],
      "canonical_omega_cmd": canonical["omega_cmd_rad_s"],
      "canonical_v_left": canonical["v_left_cmd_m_s"],
      "canonical_v_right": canonical["v_right_cmd_m_s"],
      "real_safe_v_cmd": safe["v_safe_m_s"],
      "real_safe_omega_cmd": safe["omega_safe_rad_s"],
      "real_safe_v_left": safe["v_left_safe_m_s"],
      "real_safe_v_right": safe["v_right_safe_m_s"],
      "command_scale": safe["real_safety_scale"],
      "localization_valid": int(validation["valid"]),
      "safety_reason": final["safety_reason"],
      "shadow_only": 1,
      "row_source": row_source,
      "preset": self.context["config"]["execution_plan"]["preset"],
      "start_mode": START_POSE_MODE,
      "nearest_path_s_within_lap": None if nearest is None else nearest["s_within_lap_m"],
      "nearest_path_yaw": None if nearest is None else nearest["path_yaw_rad"],
      "nearest_path_distance": None if nearest is None else nearest["distance_m"],
      "projection_segment_index": None if nearest is None else nearest["projection_segment_index"],
      "canonical_command_scale": canonical["command_scale"],
      "canonical_candidate_computed": int(canonical["computed"]),
      "real_safety_scale": safe["real_safety_scale"],
      "localization_state": localization_state,
      "start_xy_distance": None if alignment is None else alignment["start_xy_distance_m"],
      "start_heading_error": None if alignment is None else alignment["start_heading_error_rad"],
      "pose_jump_m": validation["pose_jump_m"],
      "pose_jump_yaw_rad": validation["pose_jump_yaw_rad"],
      "start_alignment_state": ("START_ALIGNMENT_NOT_AVAILABLE" if alignment is None else
                                alignment["alignment_state"]),
      "arm_state": "NOT_ARMABLE",
      "publication_state": "NO_CMD_VEL_PUBLISHER",
      "execution_state": self.state,
      "execution_time_s": self.execution_time_s,
      "execution_dt_s": dt_s,
      "reference_index": reference["reference_index"],
      "reference_next_index": reference["reference_next_index"],
      "reference_interpolation_fraction": reference["reference_interpolation_fraction"],
      "reference_progress_m": float(reference["s_m"]),
      "reference_lap_index": 0,
      "planned_laps": 1,
      "reference_motion_phase": reference["motion_phase"],
      "launch_envelope_active": int(reference["launch_envelope_active"]),
      "terminal_stop_envelope_active": int(reference["terminal_stop_envelope_active"]),
      "stop_reason": self.stop_reason,
      "fault_reason": self.fault_reason,
      "final_candidate_v": final["final_candidate_v_m_s"],
      "final_candidate_omega": final["final_candidate_omega_rad_s"],
      "final_candidate_left": final["final_candidate_left_m_s"],
      "final_candidate_right": final["final_candidate_right_m_s"],
      "final_candidate_allowed": int(final["command_allowed_in_simulated_execution"]),
      "watchdog_state": self._watchdog_state(),
      "execution_transition_event": transition,
      "reference_valid": int(_reference_is_valid(reference)),
      "timing_valid": int(timing_valid),
      "physical_motion_enabled": 0,
      "execution_plan_identity": self.context["execution_plan_identity"],
    }
    if set(row) != set(EXECUTION_CSV_COLUMNS):
      missing = sorted(set(EXECUTION_CSV_COLUMNS) - set(row))
      extra = sorted(set(row) - set(EXECUTION_CSV_COLUMNS))
      raise ExecutionSafetyError(f"execution CSV row mismatch; missing={missing}, extra={extra}")
    return row

  def _result(self, sample: LocalizationSample | None, now_s: float, dt_s: float,
              row_source: str, validation: dict[str, Any],
              alignment: dict[str, Any] | None, reference: dict[str, Any],
              timing_valid: bool, transition: str,
              compute_candidate: bool = True, orderly_scale: float = 1.0) -> dict[str, Any]:
    canonical = _zero_canonical()
    safe = _zero_safe()
    reference_valid = _reference_is_valid(reference)
    if (compute_candidate and validation["valid"] and validation["pose_usable"] and
        reference_valid):
      pose = validation["pose"]
      canonical = controller_command(
        (float(pose["x_m"]), float(pose["y_m"]), float(pose["yaw_rad"])),
        reference, int(self.context["config"]["execution_plan"]["motion_direction"]),
        self.context["shadow"]["controller"])
      canonical["computed"] = True
      safe = real_safety_clamp(
        float(canonical["v_cmd_m_s"]), float(canonical["omega_cmd_rad_s"]),
        float(self.context["shadow"]["track_center_distance_m"]),
        self.context["shadow"]["config"], True)
    reason = "LOCALIZATION_OK" if validation["valid"] else "+".join(validation["reasons"])
    final = supervise_command(
      self.state, bool(validation["valid"]), timing_valid, reference_valid, safe,
      reason, output_scale=orderly_scale)
    row = self._make_row(sample, now_s, dt_s, row_source, validation, alignment,
                         reference, canonical, safe, final, timing_valid, transition)
    result = {
      "state": self.state,
      "transition": transition,
      "validation": validation,
      "alignment": alignment,
      "reference": reference,
      "canonical_candidate": canonical,
      "real_safe_candidate": safe,
      "final_candidate": final,
      "row": row,
      "physical_motion_enabled": False,
      "arm_state": "NOT_ARMABLE",
      "publication_state": "NO_CMD_VEL_PUBLISHER",
    }
    self.last_final_candidate = final
    self.last_result = result
    return result

  def _fault(self, reason: str, sample: LocalizationSample | None, now_s: float,
             dt_s: float, row_source: str,
             validation: dict[str, Any] | None = None) -> dict[str, Any]:
    transition = self._transition(FAULT)
    self.fault_reason = reason
    if validation is None:
      validation = validate_localization(sample, now_s, self.previous_valid_pose,
                                         self.context["shadow"]["config"])
    alignment = self._alignment(validation)
    reference = self.interpolator.sample(self.execution_time_s)
    result = self._result(sample, now_s, dt_s, row_source, validation, alignment,
                          reference, False, transition, compute_candidate=False)
    result["final_candidate"] = zero_final_candidate(reason)
    for key, value in (
        ("final_candidate_v", 0.0), ("final_candidate_omega", 0.0),
        ("final_candidate_left", 0.0), ("final_candidate_right", 0.0),
        ("final_candidate_allowed", 0), ("safety_reason", reason),
        ("fault_reason", reason), ("watchdog_state", "WATCHDOG_FAULT")):
      result["row"][key] = value
    self.last_final_candidate = result["final_candidate"]
    self.last_result = result
    return result

  def prepare(self, sample: LocalizationSample | None, now_s: float,
              start_mode: str = START_POSE_MODE,
              row_source: str = ROW_SOURCE_OFFLINE_SYNTHETIC) -> dict[str, Any]:
    if self.state != DISARMED:
      raise ExecutionSafetyError("prepare() is only valid from DISARMED")
    if start_mode == NEAREST_PATH_MODE:
      raise ExecutionSafetyError("NEAREST_PATH_MODE remains diagnostic-only and cannot prepare execution")
    if start_mode != START_POSE_MODE:
      raise ExecutionSafetyError(f"unknown execution start mode: {start_mode}")
    validation = validate_localization(sample, now_s, None,
                                       self.context["shadow"]["config"])
    alignment = self._alignment(validation)
    transition = ""
    if validation["valid"] and alignment is not None and alignment["start_pose_passed"]:
      transition = self._transition(READY)
      self.previous_valid_pose = dict(validation["pose"])
    reference = self.interpolator.sample(0.0)
    return self._result(sample, now_s, 0.0, row_source, validation, alignment,
                        reference, True, transition, compute_candidate=False)

  def start(self, sample: LocalizationSample | None, now_s: float,
            row_source: str = ROW_SOURCE_OFFLINE_SYNTHETIC) -> dict[str, Any]:
    if self.state != READY:
      raise ExecutionSafetyError("simulated operator start is only valid from READY")
    validation = validate_localization(sample, now_s, self.previous_valid_pose,
                                       self.context["shadow"]["config"])
    alignment = self._alignment(validation)
    if not validation["valid"]:
      return self._fault("+".join(validation["reasons"]), sample, now_s, 0.0,
                         row_source, validation)
    if alignment is None or not alignment["start_pose_passed"]:
      return self._fault("START_ALIGNMENT_REJECTED", sample, now_s, 0.0,
                         row_source, validation)
    transition = self._transition(RUNNING)
    self.execution_time_s = 0.0
    self.last_update_timestamp_s = now_s
    self.previous_valid_pose = dict(validation["pose"])
    reference = self.interpolator.sample(0.0)
    return self._result(sample, now_s, 0.0, row_source, validation, alignment,
                        reference, True, transition)

  def step(self, dt_s: float, sample: LocalizationSample | None, now_s: float,
           row_source: str = ROW_SOURCE_OFFLINE_SYNTHETIC) -> dict[str, Any]:
    if row_source not in (ROW_SOURCE_POSE_UPDATE, ROW_SOURCE_OFFLINE_SYNTHETIC):
      raise ExecutionSafetyError("only pose-update/synthetic rows may advance execution")
    if self.state == FAULT:
      return self._fault(self.fault_reason or "FAULT_LATCHED", sample, now_s, dt_s,
                         row_source)
    if self.state not in (RUNNING, STOPPING):
      raise ExecutionSafetyError("step() requires RUNNING or STOPPING")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
      return self._fault("BACKWARD_EXECUTION_TIME", sample, now_s, dt_s, row_source)
    if self.last_update_timestamp_s is None or not math.isfinite(now_s):
      return self._fault("INVALID_EXECUTION_TIMESTAMP", sample, now_s, dt_s, row_source)
    timestamp_gap = now_s - self.last_update_timestamp_s
    if timestamp_gap <= 0.0:
      return self._fault("BACKWARD_EXECUTION_TIMESTAMP", sample, now_s, dt_s, row_source)
    timeout = float(self.context["watchdog_timeout_s"])
    if timestamp_gap > timeout + 1e-12 or dt_s > timeout + 1e-12:
      return self._fault("CONTROL_UPDATE_TIMEOUT", sample, now_s, dt_s, row_source)
    if abs(timestamp_gap - dt_s) > 1e-9:
      return self._fault("EXECUTION_DT_TIMESTAMP_MISMATCH", sample, now_s, dt_s, row_source)

    validation = validate_localization(sample, now_s, self.previous_valid_pose,
                                       self.context["shadow"]["config"])
    if not validation["valid"]:
      return self._fault("+".join(validation["reasons"]), sample, now_s, dt_s,
                         row_source, validation)
    alignment = self._alignment(validation)
    self.last_update_timestamp_s = now_s
    self.previous_valid_pose = dict(validation["pose"])
    self.execution_time_s = min(self.context["duration_s"], self.execution_time_s + dt_s)
    transition = ""
    if (self.state == RUNNING and
        self.execution_time_s >= self.context["stop_start_time_s"] - 1e-12):
      self.stop_reason = "PLANNED_TERMINAL_DECELERATION"
      transition = self._transition(STOPPING)

    reference = self.interpolator.sample(self.execution_time_s)
    orderly_scale = 1.0
    if self.state == STOPPING:
      provisional_validation = validation
      pose = provisional_validation["pose"]
      provisional_command = controller_command(
        (float(pose["x_m"]), float(pose["y_m"]), float(pose["yaw_rad"])),
        reference, 1, self.context["shadow"]["controller"])
      provisional_safe = real_safety_clamp(
        float(provisional_command["v_cmd_m_s"]),
        float(provisional_command["omega_cmd_rad_s"]),
        float(self.context["shadow"]["track_center_distance_m"]),
        self.context["shadow"]["config"], True)
      component_pairs = (
        ("v_safe_m_s", "final_candidate_v_m_s"),
        ("omega_safe_rad_s", "final_candidate_omega_rad_s"),
        ("v_left_safe_m_s", "final_candidate_left_m_s"),
        ("v_right_safe_m_s", "final_candidate_right_m_s"),
      )
      ratios = [1.0]
      for safe_key, final_key in component_pairs:
        candidate_magnitude = abs(float(provisional_safe[safe_key]))
        previous_magnitude = abs(float(self.last_final_candidate[final_key]))
        if candidate_magnitude > previous_magnitude + 1e-15:
          ratios.append(previous_magnitude / candidate_magnitude)
      permitted_speed = None
      if self.explicit_orderly_stop:
        previous_speed = abs(float(self.last_final_candidate["final_candidate_v_m_s"]))
        deceleration = float(self.context["shadow"]["demo"]["canonical"]
                             ["trajectory_limits"]["maximum_deceleration_m_s2"])
        permitted_speed = max(0.0, previous_speed - deceleration * dt_s)
        candidate_speed = abs(float(provisional_safe["v_safe_m_s"]))
        if candidate_speed > permitted_speed + 1e-15:
          ratios.append(permitted_speed / candidate_speed)
      orderly_scale = min(ratios)
      if self.explicit_orderly_stop and permitted_speed is not None and permitted_speed <= 1e-15:
        transition = self._transition(COMPLETE)
        reference = dict(reference)
        reference.update({"v_ref_m_s": 0.0, "omega_ref_rad_s": 0.0,
                          "a_ref_m_s2": 0.0, "motion_phase": "orderly_stop_complete"})

    if self.execution_time_s >= self.context["duration_s"] - 1e-12:
      if self.state == RUNNING:
        self.stop_reason = "PLANNED_TERMINAL_DECELERATION"
        first = self._transition(STOPPING)
        transition = first
      second = self._transition(COMPLETE)
      transition = f"{transition};{second}" if transition else second
    result = self._result(sample, now_s, dt_s, row_source, validation, alignment,
                          reference, True, transition, orderly_scale=orderly_scale)
    return result

  def status_check(self, sample: LocalizationSample | None, now_s: float) -> dict[str, Any]:
    """Evaluate safety/watchdog without advancing execution or pose history."""
    if self.state == FAULT:
      return self._fault(self.fault_reason or "FAULT_LATCHED", sample, now_s, 0.0,
                         ROW_SOURCE_STATUS_TIMER)
    validation = validate_localization(sample, now_s, self.previous_valid_pose,
                                       self.context["shadow"]["config"])
    if self.state in (READY, RUNNING, STOPPING) and not validation["valid"]:
      return self._fault("+".join(validation["reasons"]), sample, now_s, 0.0,
                         ROW_SOURCE_STATUS_TIMER, validation)
    if (self.state in (RUNNING, STOPPING) and self.last_update_timestamp_s is not None and
        now_s - self.last_update_timestamp_s > self.context["watchdog_timeout_s"] + 1e-12):
      return self._fault("CONTROL_UPDATE_TIMEOUT", sample, now_s, 0.0,
                         ROW_SOURCE_STATUS_TIMER, validation)
    alignment = self._alignment(validation)
    reference = self.interpolator.sample(self.execution_time_s)
    return self._result(sample, now_s, 0.0, ROW_SOURCE_STATUS_TIMER,
                        validation, alignment, reference, True, "",
                        compute_candidate=self.state in (RUNNING, STOPPING))

  def request_orderly_stop(self, reason: str = "EXPLICIT_SIMULATED_ORDERLY_STOP") -> str:
    if self.state != RUNNING:
      raise ExecutionSafetyError("orderly stop may only be requested from RUNNING")
    self.stop_reason = reason
    self.explicit_orderly_stop = True
    return self._transition(STOPPING)

  def emergency_stop(self, reason: str = "EXPLICIT_EMERGENCY_STOP") -> dict[str, Any]:
    if self.state not in (READY, RUNNING, STOPPING):
      raise ExecutionSafetyError("emergency stop requires READY, RUNNING, or STOPPING")
    now_s = self.last_update_timestamp_s if self.last_update_timestamp_s is not None else 0.0
    return self._fault(reason, None, now_s, 0.0, ROW_SOURCE_STATUS_TIMER)

  def reset(self) -> str:
    if self.state not in (FAULT, COMPLETE):
      raise ExecutionSafetyError("reset is only valid from FAULT or COMPLETE")
    transition = self._transition(DISARMED)
    self.execution_time_s = 0.0
    self.last_update_timestamp_s = None
    self.previous_valid_pose = None
    self.fault_reason = ""
    self.stop_reason = ""
    self.explicit_orderly_stop = False
    self.last_final_candidate = zero_final_candidate("RESET_DISARMED")
    self.last_result = None
    return transition


def write_execution_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=EXECUTION_CSV_COLUMNS, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
