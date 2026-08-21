#!/usr/bin/env python3
"""Run deterministic one-lap execution and fault gates without ROS or command output."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from real_global_path_execution_safety_v1 import (
  COMPLETE,
  DISARMED,
  FAULT,
  READY,
  RUNNING,
  STOPPING,
  REPO_ROOT,
  ROW_SOURCE_OFFLINE_SYNTHETIC,
  ExecutionSafetyError,
  ExecutionSafetySession,
  assert_physical_reference_timing_ready,
  command_sign_audit,
  write_execution_csv,
)
from real_global_path_tracking_v1 import LocalizationSample, sample_from_xy_yaw
from tracking_controller_v1 import wrap_to_pi


def _zero_final(result: dict[str, Any]) -> bool:
  final = result["final_candidate"]
  return all(abs(float(final[key])) <= 1e-15 for key in (
    "final_candidate_v_m_s", "final_candidate_omega_rad_s",
    "final_candidate_left_m_s", "final_candidate_right_m_s"))


def _exact_sample(reference: dict[str, Any], timestamp_s: float) -> LocalizationSample:
  return sample_from_xy_yaw(timestamp_s, float(reference["x_m"]),
                            float(reference["y_m"]), float(reference["yaw_rad"]))


def _running_session(config_path: Path, start_time_s: float = 10.0
                     ) -> tuple[ExecutionSafetySession, LocalizationSample]:
  session = ExecutionSafetySession(config_path)
  reference = session.interpolator.sample(0.0)
  sample = _exact_sample(reference, start_time_s)
  prepared = session.prepare(sample, start_time_s)
  if prepared["state"] != READY:
    raise RuntimeError("exact canonical start did not prepare")
  started = session.start(sample, start_time_s)
  if started["state"] != RUNNING:
    raise RuntimeError("simulated operator start did not enter RUNNING")
  return session, sample


def run_fault_injections(config_path: Path) -> dict[str, Any]:
  cases: dict[str, dict[str, Any]] = {}

  def capture(name: str, injector: Any) -> None:
    session, _ = _running_session(config_path)
    result = injector(session)
    original_reason = str(result["row"]["fault_reason"])
    next_reference = session.interpolator.sample(min(
      session.context["duration_s"], session.execution_time_s + 0.02))
    recovery_sample = _exact_sample(next_reference, 10.02)
    latched = session.step(0.02, recovery_sample, 10.02)
    latched_zero = latched["state"] == FAULT and _zero_final(latched)
    reset_transition = session.reset()
    cases[name] = {
      "fault_reason": original_reason,
      "entered_fault": result["state"] == FAULT,
      "final_candidate_zero": _zero_final(result),
      "latched_until_reset": latched_zero,
      "reset_transition": reset_transition,
      "reset_state": session.state,
    }

  def reference_at(session: ExecutionSafetySession, dt_s: float = 0.02) -> dict[str, Any]:
    return session.interpolator.sample(session.execution_time_s + dt_s)

  def stale(session: ExecutionSafetySession) -> dict[str, Any]:
    now_s = 10.02
    reference = reference_at(session)
    age = float(session.context["shadow"]["config"]
                ["initial_safety_envelope_not_calibrated"]["maximum_localization_age_s"])
    return session.step(0.02, _exact_sample(reference, now_s - age - 0.01), now_s)

  def wrong_frame(session: ExecutionSafetySession) -> dict[str, Any]:
    sample = replace(_exact_sample(reference_at(session), 10.02), frame_id="odom")
    return session.step(0.02, sample, 10.02)

  def nonfinite(session: ExecutionSafetySession) -> dict[str, Any]:
    sample = replace(_exact_sample(reference_at(session), 10.02), x_m=math.nan)
    return session.step(0.02, sample, 10.02)

  def invalid_quaternion(session: ExecutionSafetySession) -> dict[str, Any]:
    reference = reference_at(session)
    return session.step(0.02, LocalizationSample(
      10.02, "map", float(reference["x_m"]), float(reference["y_m"]), 0.0,
      0.0, 0.0, 0.0, 0.0), 10.02)

  def translation_jump(session: ExecutionSafetySession) -> dict[str, Any]:
    reference = reference_at(session)
    limit = float(session.context["shadow"]["config"]
                  ["initial_safety_envelope_not_calibrated"]["maximum_pose_jump_m"])
    sample = _exact_sample(reference, 10.02)
    return session.step(0.02, replace(sample, x_m=sample.x_m + limit + 0.1), 10.02)

  def yaw_jump(session: ExecutionSafetySession) -> dict[str, Any]:
    reference = reference_at(session)
    limit = float(session.context["shadow"]["config"]
                  ["initial_safety_envelope_not_calibrated"]["maximum_pose_jump_yaw_rad"])
    sample = _exact_sample(reference, 10.02)
    jumped = sample_from_xy_yaw(10.02, sample.x_m, sample.y_m,
                                float(reference["yaw_rad"]) + limit + 0.1)
    return session.step(0.02, jumped, 10.02)

  def timeout(session: ExecutionSafetySession) -> dict[str, Any]:
    dt_s = float(session.context["watchdog_timeout_s"]) + 0.01
    now_s = 10.0 + dt_s
    return session.step(dt_s, _exact_sample(reference_at(session, dt_s), now_s), now_s)

  def backward_timestamp(session: ExecutionSafetySession) -> dict[str, Any]:
    return session.step(0.02, _exact_sample(reference_at(session), 9.99), 9.99)

  def emergency_stop(session: ExecutionSafetySession) -> dict[str, Any]:
    return session.emergency_stop()

  def missing_localization(session: ExecutionSafetySession) -> dict[str, Any]:
    return session.step(0.02, None, 10.02)

  injectors = {
    "stale_localization": stale,
    "wrong_frame": wrong_frame,
    "nan_pose": nonfinite,
    "invalid_quaternion": invalid_quaternion,
    "translation_jump": translation_jump,
    "yaw_jump": yaw_jump,
    "control_update_timeout": timeout,
    "backward_execution_timestamp": backward_timestamp,
    "explicit_emergency_stop": emergency_stop,
    "missing_localization": missing_localization,
  }
  for name, injector in injectors.items():
    capture(name, injector)
  passed_count = sum(
    item["entered_fault"] and item["final_candidate_zero"] and
    item["latched_until_reset"] and item["reset_transition"] == "FAULT->DISARMED" and
    item["reset_state"] == DISARMED
    for item in cases.values())
  return {"case_count": len(cases), "passed_count": passed_count,
          "passed": passed_count == len(cases), "cases": cases}


def run_nominal(config_path: Path) -> tuple[ExecutionSafetySession, list[dict[str, Any]],
                                           dict[str, Any]]:
  session = ExecutionSafetySession(config_path)
  start_time_s = 100.0
  start_reference = session.interpolator.sample(0.0)
  start_sample = _exact_sample(start_reference, start_time_s)
  prepared = session.prepare(start_sample, start_time_s)
  started = session.start(start_sample, start_time_s)
  rows = [prepared["row"], started["row"]]
  now_s = start_time_s
  nominal_step = float(session.context["config"]["deterministic_clock"]["nominal_step_s"])
  while session.state != COMPLETE:
    remaining = session.context["duration_s"] - session.execution_time_s
    dt_s = min(nominal_step, remaining)
    if dt_s <= 1e-12:
      raise RuntimeError("nominal execution reached a non-terminal zero remainder")
    next_time = session.execution_time_s + dt_s
    reference = session.interpolator.sample(next_time)
    now_s += dt_s
    result = session.step(dt_s, _exact_sample(reference, now_s), now_s,
                          ROW_SOURCE_OFFLINE_SYNTHETIC)
    rows.append(result["row"])
    if result["state"] == FAULT:
      raise RuntimeError(f"nominal execution faulted: {result['row']['fault_reason']}")

  reference_rows = session.context["reference"]
  active_rows = [row for row in rows if row["execution_state"] in (RUNNING, STOPPING)]
  moving_rows = [row for row in active_rows if abs(float(row["reference_v"])) > 1e-12]
  straight_speeds = [float(row["v_ref_m_s"]) for row in reference_rows
                     if abs(float(row["curvature_ref_1_m"])) <= 1e-9]
  corner_speeds = [float(row["v_ref_m_s"]) for row in reference_rows
                   if abs(float(row["curvature_ref_1_m"])) > 1e-9 and
                   float(row["v_ref_m_s"]) > 1e-12]
  safety = session.context["shadow"]["config"]["initial_safety_envelope_not_calibrated"]
  canonical_limits = session.context["shadow"]["demo"]["canonical"]["trajectory_limits"]
  reference_accelerations = [float(row["a_ref_m_s2"]) for row in reference_rows[:-1]]
  transitions = [row["execution_transition_event"] for row in rows
                 if row["execution_transition_event"]]
  first_stopping_index = next(index for index, row in enumerate(rows)
                              if row["execution_state"] == STOPPING)
  stopping_window = rows[first_stopping_index - 1:]
  checks = {
    "explicit_state_sequence": session.state_history ==
      [DISARMED, READY, RUNNING, STOPPING, COMPLETE],
    "starts_at_zero": all(abs(float(started["row"][key])) <= 1e-15 for key in (
      "reference_v", "final_candidate_v", "final_candidate_omega",
      "final_candidate_left", "final_candidate_right")),
    "launch_acceleration_present": any(
      float(row["a_ref_m_s2"]) > 1e-9 and row["launch_envelope_active"]
      for row in reference_rows),
    "periodic_section_reached": any(
      not row["launch_envelope_active"] and not row["terminal_stop_envelope_active"] and
      float(row["v_ref_m_s"]) > 0.0 for row in reference_rows),
    "corner_slowdown_present": bool(straight_speeds and corner_speeds and
                                    min(corner_speeds) < max(straight_speeds) - 1e-6),
    "terminal_deceleration_present": any(
      float(row["a_ref_m_s2"]) < -1e-9 and row["terminal_stop_envelope_active"]
      for row in reference_rows),
    "canonical_dynamic_limits_unchanged": (
      max(float(row["v_ref_m_s"]) for row in reference_rows) <=
      float(canonical_limits["maximum_body_speed_m_s"]) + 1e-12 and
      max(reference_accelerations) <=
      float(canonical_limits["maximum_acceleration_m_s2"]) + 1e-12 and
      abs(min(reference_accelerations)) <=
      float(canonical_limits["maximum_deceleration_m_s2"]) + 1e-12 and
      max(abs(float(row["omega_ref_rad_s"])) for row in reference_rows) <=
      float(canonical_limits["maximum_yaw_rate_rad_s"]) + 1e-12 and
      max(abs(float(row[key])) for row in reference_rows
          for key in ("v_left_ref_m_s", "v_right_ref_m_s")) <=
      float(canonical_limits["maximum_track_surface_speed_m_s"]) + 1e-12),
    "terminal_reference_and_candidate_zero": (
      abs(float(rows[-1]["reference_v"])) <= 1e-15 and
      all(abs(float(rows[-1][key])) <= 1e-15 for key in (
        "final_candidate_v", "final_candidate_omega",
        "final_candidate_left", "final_candidate_right"))),
    "stopping_candidate_components_nonincreasing": all(
      abs(float(right[key])) <= abs(float(left[key])) + 1e-12
      for left, right in zip(stopping_window, stopping_window[1:])
      for key in ("final_candidate_v", "final_candidate_omega",
                  "final_candidate_left", "final_candidate_right")),
    "reference_progress_monotonic": all(
      float(right["reference_progress_m"]) >= float(left["reference_progress_m"]) - 1e-12
      for left, right in zip(rows, rows[1:])),
    "execution_time_monotonic": all(
      float(right["execution_time_s"]) >= float(left["execution_time_s"]) - 1e-12
      for left, right in zip(rows, rows[1:])),
    "closed_position_continuity": math.hypot(
      float(reference_rows[-1]["x_m"]) - float(reference_rows[0]["x_m"]),
      float(reference_rows[-1]["y_m"]) - float(reference_rows[0]["y_m"])) <= 1e-12,
    "closed_yaw_continuity": abs(wrap_to_pi(
      float(reference_rows[-1]["yaw_rad"]) - float(reference_rows[0]["yaw_rad"]))) <= 1e-12,
    "canonical_and_real_safe_separate": any(
      abs(float(row["canonical_v_cmd"])) > abs(float(row["real_safe_v_cmd"])) + 1e-9
      for row in moving_rows),
    "real_body_cap": all(abs(float(row["real_safe_v_cmd"])) <=
                         float(safety["maximum_real_body_speed_m_s"]) + 1e-12
                         for row in rows),
    "real_yaw_cap": all(abs(float(row["real_safe_omega_cmd"])) <=
                        float(safety["maximum_real_yaw_rate_rad_s"]) + 1e-12
                        for row in rows),
    "real_track_cap": all(
      abs(float(row[key])) <= float(safety["maximum_real_track_speed_m_s"]) + 1e-12
      for row in rows for key in ("real_safe_v_left", "real_safe_v_right")),
    "all_rows_synthetic_provenance": all(
      row["row_source"] == ROW_SOURCE_OFFLINE_SYNTHETIC for row in rows),
    "no_physical_motion_state": all(
      not int(row["physical_motion_enabled"]) and row["arm_state"] == "NOT_ARMABLE"
      for row in rows),
  }
  metrics = {
    "row_count": len(rows),
    "state_row_counts": dict(sorted(Counter(row["execution_state"] for row in rows).items())),
    "transitions": transitions,
    "maximum_abs_canonical_v_m_s": max(abs(float(row["canonical_v_cmd"])) for row in rows),
    "maximum_abs_canonical_omega_rad_s": max(
      abs(float(row["canonical_omega_cmd"])) for row in rows),
    "maximum_abs_real_safe_v_m_s": max(abs(float(row["real_safe_v_cmd"])) for row in rows),
    "maximum_abs_real_safe_omega_rad_s": max(
      abs(float(row["real_safe_omega_cmd"])) for row in rows),
    "maximum_abs_real_safe_track_m_s": max(
      abs(float(row[key])) for row in rows
      for key in ("real_safe_v_left", "real_safe_v_right")),
    "real_safety_clamped_row_count": sum(
      int(row["canonical_candidate_computed"]) and
      float(row["real_safety_scale"]) < 1.0 - 1e-12 for row in rows),
    "maximum_reference_acceleration_m_s2": max(reference_accelerations),
    "maximum_reference_deceleration_magnitude_m_s2": abs(min(reference_accelerations)),
    "final_execution_time_s": float(rows[-1]["execution_time_s"]),
    "final_reference_progress_m": float(rows[-1]["reference_progress_m"]),
  }
  return session, rows, {"passed": all(checks.values()), "checks": checks,
                         "metrics": metrics}


def run_offline_gate(config_path: Path, output_dir: Path) -> dict[str, Any]:
  session, rows, nominal = run_nominal(config_path)
  faults = run_fault_injections(config_path)
  signs = command_sign_audit(float(session.context["shadow"]["track_center_distance_m"]))
  config = session.context["config"]
  readiness = config["physical_execution_readiness"]
  timing_diagnostics = session.context["physical_timing_diagnostics"]
  try:
    assert_physical_reference_timing_ready(readiness, timing_diagnostics)
    timing_guard_rejected = False
  except ExecutionSafetyError:
    timing_guard_rejected = True
  output_dir.mkdir(parents=True, exist_ok=True)
  csv_path = output_dir / config["output"]["offline_csv_filename"]
  write_execution_csv(csv_path, rows)
  reference = session.context["reference"]
  checks = {
    "nominal_one_lap": nominal["passed"],
    "ten_fault_injections": faults["passed"] and faults["case_count"] == 10,
    "command_sign_audit": signs["passed"],
    "diagnostic_only": all(not int(row["physical_motion_enabled"]) for row in rows),
    "physical_timing_correctly_blocked": (
      not readiness["reference_timing_ready"] and
      not readiness["future_publisher_ready"] and
      timing_diagnostics["minimum_required_uniform_time_scale_lower_bound"] > 1.0 and
      timing_guard_rejected),
  }
  summary = {
    "stage": "Real Global Path Execution & Safety V1 — offline Gate",
    "gate": {"passed": all(checks.values()), "checks": checks},
    "architecture": [
      "accepted localization sample",
      "frozen Real Tracking Core",
      "pure execution state machine",
      "Direction-Aware Controller V2",
      "frozen real safety clamp",
      "pure command safety supervisor",
      "FINAL COMMAND CANDIDATE — STOP HERE",
    ],
    "reference": {
      "preset": config["execution_plan"]["preset"],
      "requested_laps": 1,
      "canonical_builder_laps": 2,
      "one_lap_adapter": config["execution_plan"]["one_lap_adapter"],
      "lap_length_m": session.context["lap_length_m"],
      "duration_s": session.context["duration_s"],
      "sample_count": len(reference),
      "stop_start_index": session.context["stop_start_index"],
      "stop_start_time_s": session.context["stop_start_time_s"],
      "position_closure_error_m": math.hypot(
        float(reference[-1]["x_m"]) - float(reference[0]["x_m"]),
        float(reference[-1]["y_m"]) - float(reference[0]["y_m"])),
      "wrapped_yaw_closure_error_rad": abs(wrap_to_pi(
        float(reference[-1]["yaw_rad"]) - float(reference[0]["yaw_rad"]))),
      "config_sha256": session.context["config_sha256"],
      "execution_plan_identity": session.context["execution_plan_identity"],
    },
    "state_machine": {
      "states": list(config["state_machine"]["states"]),
      "nominal_history": session.state_history,
      "no_state_is_physical_arm": True,
      "arm_state": "NOT_ARMABLE",
    },
    "physical_execution_readiness": {
      **readiness,
      "current_canonical_clock_connection_allowed": False,
      "guard_rejects_current_clock": timing_guard_rejected,
      "diagnostics": timing_diagnostics,
    },
    "nominal": nominal,
    "fault_injections": faults,
    "command_sign_audit": signs,
    "row_source_semantics": config["output"]["row_source_semantics"],
    "future_tracking_performance_filter": config["output"]["future_tracking_performance_filter"],
    "physical_command_publication": False,
    "performance_claim": "None; exact synthetic feedback proves state-machine and safety semantics only, not physical tracking feasibility.",
    "output_csv": str(csv_path),
  }
  summary_path = output_dir / config["output"]["offline_summary_filename"]
  with summary_path.open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, allow_nan=False)
    stream.write("\n")
  print(f"offline one-lap rows={len(rows)}, duration={session.context['duration_s']:.3f}s, "
        f"length={session.context['lap_length_m']:.3f}m")
  print(f"state sequence={' -> '.join(session.state_history)}")
  print(f"fault injections={faults['passed_count']}/{faults['case_count']}, "
        f"command signs={'PASS' if signs['passed'] else 'FAIL'}")
  print(f"physical timing={readiness['status']}, uniform-scale-lower-bound="
        f"{timing_diagnostics['minimum_required_uniform_time_scale_lower_bound']:.3f}x")
  print(f"gate={'PASS' if summary['gate']['passed'] else 'FAIL'}")
  print("physical publication=NO; output is FINAL COMMAND CANDIDATE only")
  print(f"output={output_dir}")
  return summary


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path,
                      default=REPO_ROOT / "config/real_global_path_execution_safety_v1.json")
  parser.add_argument("--output-dir", type=Path)
  args = parser.parse_args()
  config_path = args.config.resolve()
  config = json.loads(config_path.read_text(encoding="utf-8"))
  default = REPO_ROOT / config["output"]["default_directory"] / "offline_one_lap"
  summary = run_offline_gate(config_path, (args.output_dir or default).resolve())
  return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
