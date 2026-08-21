#!/usr/bin/env python3
"""Deterministic tests for offline real-path execution and command-safety preparation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable

from real_global_path_execution_safety_v1 import (
  COMPLETE,
  DISARMED,
  EXECUTION_CSV_COLUMNS,
  FAULT,
  READY,
  RUNNING,
  STOPPING,
  REPO_ROOT,
  ROW_SOURCE_OFFLINE_SYNTHETIC,
  ExecutionReferenceInterpolator,
  ExecutionSafetyError,
  ExecutionSafetySession,
  assert_physical_reference_timing_ready,
  build_execution_context,
  command_sign_audit,
  evaluate_start_alignment,
  supervise_command,
  zero_final_candidate,
)
from real_global_path_tracking_v1 import (
  CSV_COLUMNS as SHADOW_CSV_COLUMNS,
  NEAREST_PATH_MODE,
  ROW_SOURCE_POSE_UPDATE,
  ROW_SOURCE_STATUS_TIMER,
  sample_from_xy_yaw,
)
from run_real_global_path_execution_offline_v1 import (
  run_fault_injections,
  run_nominal,
)
from tracking_controller_v1 import wrap_to_pi


def require(condition: bool, message: str) -> None:
  if not condition:
    raise AssertionError(message)


def require_raises(exception: type[BaseException], function: Callable[[], Any],
                   message: str) -> None:
  try:
    function()
  except exception:
    return
  raise AssertionError(message)


def exact_sample(reference: dict[str, Any], timestamp_s: float):
  return sample_from_xy_yaw(timestamp_s, float(reference["x_m"]),
                            float(reference["y_m"]), float(reference["yaw_rad"]))


def final_is_zero(result: dict[str, Any]) -> bool:
  final = result["final_candidate"]
  return all(abs(float(final[key])) <= 1e-15 for key in (
    "final_candidate_v_m_s", "final_candidate_omega_rad_s",
    "final_candidate_left_m_s", "final_candidate_right_m_s"))


def running_session(config_path: Path, advance_s: float = 0.0) -> ExecutionSafetySession:
  session = ExecutionSafetySession(config_path)
  start = session.interpolator.sample(0.0)
  sample = exact_sample(start, 10.0)
  session.prepare(sample, 10.0)
  session.start(sample, 10.0)
  now_s = 10.0
  while session.execution_time_s < advance_s - 1e-12:
    dt_s = min(0.02, advance_s - session.execution_time_s)
    now_s += dt_s
    reference = session.interpolator.sample(session.execution_time_s + dt_s)
    result = session.step(dt_s, exact_sample(reference, now_s), now_s)
    require(result["state"] != FAULT, "setup execution faulted")
  return session


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path,
                      default=REPO_ROOT / "config/real_global_path_execution_safety_v1.json")
  parser.add_argument("--require-offline-output", action="store_true")
  args = parser.parse_args()
  config_path = args.config.resolve()
  context = build_execution_context(config_path)
  config = context["config"]
  reference = context["reference"]
  shadow = context["shadow"]
  tests: list[str] = []

  def check(name: str, function: Callable[[], None]) -> None:
    function()
    tests.append(name)
    print(f"PASS: {name}")

  def architecture_lock() -> None:
    readiness = config["physical_execution_readiness"]
    require(not config["metadata"]["physical_motion_enabled"] and
            not config["command_output"]["physical_motion_enabled"] and
            not config["command_output"]["create_command_publisher"] and
            not config["command_output"]["publish_twist"] and
            config["state_machine"]["arm_state"] == "NOT_ARMABLE" and
            not config["future_publisher_requirements"]["implemented"] and
            not readiness["reference_timing_ready"] and
            not readiness["future_publisher_ready"] and
            readiness["status"] == "BLOCKED_PENDING_REAL_COMMAND_CALIBRATION",
            "offline physical/publication lock changed")
  check("offline architecture and physical lock", architecture_lock)

  def one_lap_adapter() -> None:
    require(int(config["execution_plan"]["requested_laps"]) == 1 and
            int(config["execution_plan"]["canonical_builder_laps"]) == 2 and
            len(reference) == len(shadow["periodic_map_reference"]) and
            len(shadow["demo"]["trajectory"]) == 2 * (len(reference) - 1) + 1 and
            abs(float(reference[0]["v_ref_m_s"])) <= 1e-15 and
            abs(float(reference[-1]["v_ref_m_s"])) <= 1e-15 and
            any(row["launch_envelope_active"] for row in reference) and
            any(row["terminal_stop_envelope_active"] for row in reference),
            "canonical two-lap envelope was not projected to one lap")
  check("canonical one-lap adapter", one_lap_adapter)

  def geometry_unchanged() -> None:
    periodic = shadow["periodic_map_reference"]
    require(all(abs(float(row["x_m"]) - float(source["x_m"])) <= 1e-15 and
                abs(float(row["y_m"]) - float(source["y_m"])) <= 1e-15 and
                abs(float(row["yaw_rad"]) - float(source["yaw_rad"])) <= 1e-15 and
                abs(float(row["curvature_ref_1_m"]) -
                    float(source["curvature_ref_1_m"])) <= 1e-15
                for row, source in zip(reference, periodic)),
            "one-lap adapter changed canonical geometry/curvature")
  check("rounded geometry and curvature reused exactly", geometry_unchanged)

  def identity_and_caps() -> None:
    caps = config["real_safety_caps_frozen"]
    require(len(context["execution_plan_identity"]) == 64 and
            all(len(value) == 64 for value in context["config_sha256"].values()) and
            float(caps["maximum_real_body_speed_m_s"]) == 0.10 and
            float(caps["maximum_real_yaw_rate_rad_s"]) == 0.20 and
            float(caps["maximum_real_track_speed_m_s"]) == 0.12 and
            abs(float(shadow["track_center_distance_m"]) - 0.434) <= 1e-15,
            "identity or frozen safety geometry/caps mismatch")
  check("reproducible identity and frozen real caps", identity_and_caps)

  def physical_timing_blocker() -> None:
    diagnostics = context["physical_timing_diagnostics"]
    readiness = config["physical_execution_readiness"]
    forced_ready = {**readiness, "reference_timing_ready": True,
                    "future_publisher_ready": True, "status": "READY"}
    require(abs(float(diagnostics["canonical_reference_max_body_speed_m_s"]) - 0.6) <=
            1e-12 and
            abs(float(diagnostics["canonical_reference_max_yaw_rate_rad_s"]) - 0.7) <=
            1e-12 and
            abs(float(diagnostics["canonical_reference_max_track_speed_m_s"]) - 0.6) <=
            1e-12 and
            abs(float(diagnostics["provisional_real_safe_body_speed_cap_m_s"]) - 0.1) <=
            1e-12 and
            abs(float(diagnostics["provisional_real_safe_yaw_rate_cap_rad_s"]) - 0.2) <=
            1e-12 and
            abs(float(diagnostics["provisional_real_safe_track_speed_cap_m_s"]) - 0.12) <=
            1e-12 and
            abs(float(diagnostics["body_speed_ratio"]) - 6.0) <= 1e-12 and
            abs(float(diagnostics["yaw_rate_ratio"]) - 3.5) <= 1e-12 and
            abs(float(diagnostics["track_speed_ratio"]) - 5.0) <= 1e-12 and
            abs(float(diagnostics["minimum_required_uniform_time_scale_lower_bound"]) -
                6.0) <= 1e-12 and
            not diagnostics["physical_clock_connection_allowed"] and
            not readiness["reference_timing_ready"] and
            not readiness["future_publisher_ready"],
            "canonical-clock versus provisional-cap diagnostic mismatch")
    require_raises(
      ExecutionSafetyError,
      lambda: assert_physical_reference_timing_ready(forced_ready, diagnostics),
      "forced readiness bypassed incompatible canonical-clock guard")
  check("physical reference timing blocker and publisher guard", physical_timing_blocker)

  interpolator = ExecutionReferenceInterpolator(reference)

  def interpolation_initial_terminal() -> None:
    initial = interpolator.sample(0.0)
    before = interpolator.sample(-1.0)
    terminal = interpolator.sample(context["duration_s"])
    after = interpolator.sample(context["duration_s"] + 5.0)
    require(initial["reference_index"] == 0 and
            before["s_m"] == initial["s_m"] and
            terminal["reference_index"] == len(reference) - 1 and
            after["s_m"] == terminal["s_m"] and
            abs(float(after["v_ref_m_s"])) <= 1e-15 and
            abs(float(after["omega_ref_rad_s"])) <= 1e-15,
            "initial/terminal interpolation clamp mismatch")
  check("exact t=0 and terminal interpolation clamp", interpolation_initial_terminal)

  def interpolation_midpoint() -> None:
    index = len(reference) // 3
    left, right = reference[index], reference[index + 1]
    query = 0.5 * (float(left["t_s"]) + float(right["t_s"]))
    sample = interpolator.sample(query)
    require(sample["reference_index"] == index and
            sample["reference_next_index"] == index + 1 and
            abs(float(sample["reference_interpolation_fraction"]) - 0.5) <= 1e-12 and
            abs(float(sample["s_m"]) -
                0.5 * (float(left["s_m"]) + float(right["s_m"]))) <= 1e-12 and
            abs(float(sample["yaw_rad"]) -
                0.5 * (float(left["yaw_rad"]) + float(right["yaw_rad"]))) <= 1e-12,
            "reference midpoint interpolation mismatch")
  check("reference interval and continuous-yaw interpolation", interpolation_midpoint)

  def yaw_closure() -> None:
    yaw_deltas = [float(right["yaw_rad"]) - float(left["yaw_rad"])
                  for left, right in zip(reference, reference[1:])]
    require(max(abs(value) for value in yaw_deltas) < 0.1 and
            abs(wrap_to_pi(float(reference[-1]["yaw_rad"]) -
                           float(reference[0]["yaw_rad"]))) <= 1e-12 and
            math.hypot(float(reference[-1]["x_m"]) - float(reference[0]["x_m"]),
                       float(reference[-1]["y_m"]) - float(reference[0]["y_m"])) <= 1e-12,
            "closed reference wrapped or failed geometric closure")
  check("unwrapped yaw and closed-boundary continuity", yaw_closure)

  def explicit_start_transitions() -> None:
    session = ExecutionSafetySession(config_path)
    start = session.interpolator.sample(0.0)
    sample = exact_sample(start, 1.0)
    require_raises(ExecutionSafetyError, lambda: session.start(sample, 1.0),
                   "DISARMED automatically entered RUNNING")
    prepared = session.prepare(sample, 1.0)
    require(prepared["state"] == READY and final_is_zero(prepared),
            "prepare did not enter zero-candidate READY")
    started = session.start(sample, 1.0)
    require(started["state"] == RUNNING and session.state_history ==
            [DISARMED, READY, RUNNING] and
            abs(float(started["reference"]["t_s"])) <= 1e-15,
            "explicit simulated start transition/t=0 mismatch")
  check("explicit prepare and simulated operator start", explicit_start_transitions)

  def nearest_execution_disabled() -> None:
    session = ExecutionSafetySession(config_path)
    start = session.interpolator.sample(0.0)
    sample = exact_sample(start, 1.0)
    require(config["execution_plan"]["default_start_mode"] == "START_POSE_MODE" and
            not config["execution_plan"]["nearest_path_execution_enabled"],
            "physical-default start policy changed")
    require_raises(ExecutionSafetyError,
                   lambda: session.prepare(sample, 1.0, NEAREST_PATH_MODE),
                   "NEAREST_PATH_MODE prepared physical execution")
  check("START_POSE default and nearest execution disabled", nearest_execution_disabled)

  def alignment_matrix() -> None:
    start = reference[0]
    now_s = 5.0
    exact = evaluate_start_alignment(exact_sample(start, now_s), now_s, context)
    small_xy = evaluate_start_alignment(sample_from_xy_yaw(
      now_s, float(start["x_m"]) + 0.05, float(start["y_m"]),
      float(start["yaw_rad"])), now_s, context)
    heading = evaluate_start_alignment(sample_from_xy_yaw(
      now_s, float(start["x_m"]), float(start["y_m"]),
      float(start["yaw_rad"]) + math.radians(5.0)), now_s, context)
    far = evaluate_start_alignment(sample_from_xy_yaw(
      now_s, float(start["x_m"]) + 2.0, float(start["y_m"]) + 2.0,
      float(start["yaw_rad"])), now_s, context)
    other = shadow["periodic_map_reference"][len(reference) // 2]
    near_path_far_start = evaluate_start_alignment(exact_sample(other, now_s), now_s, context)
    endpoint = evaluate_start_alignment(exact_sample(reference[-1], now_s), now_s, context)
    require(exact["start_pose_passed"] and small_xy["start_pose_passed"] and
            heading["start_pose_passed"] and not far["start_pose_passed"] and
            not near_path_far_start["start_pose_passed"] and
            float(near_path_far_start["audit"]["nearest"]["distance_m"]) <= 1e-10 and
            endpoint["start_pose_passed"] and
            float(endpoint["audit"]["nearest"]["s_within_lap_m"]) <= 1e-10 and
            all(item["audit"] is not None for item in
                (exact, small_xy, heading, far, near_path_far_start, endpoint)),
            "START_POSE alignment matrix or raw diagnostics mismatch")
  check("six-case provisional start-alignment matrix", alignment_matrix)

  def uncalibrated_alignment_label() -> None:
    gate = config["initial_start_alignment_gate_not_calibrated"]
    require(gate["status"] == "INITIAL SAFETY GATE / NOT CALIBRATED" and
            "not measured" in gate["rationale"].lower() and
            float(gate["maximum_start_xy_error_m"]) > 0.0 and
            float(gate["maximum_start_heading_error_rad"]) > 0.0,
            "alignment thresholds look calibrated or are not separated")
  check("alignment values explicitly provisional", uncalibrated_alignment_label)

  def supervisor_zero_invariants() -> None:
    safe = {"v_safe_m_s": 0.1, "omega_safe_rad_s": 0.1,
            "v_left_safe_m_s": 0.08, "v_right_safe_m_s": 0.12,
            "real_safety_scale": 0.5}
    state_zero = all(not supervise_command(state, True, True, True, safe)
                     ["command_allowed_in_simulated_execution"] and
                     all(abs(float(supervise_command(state, True, True, True, safe)[key])) <= 1e-15
                         for key in ("final_candidate_v_m_s", "final_candidate_omega_rad_s",
                                     "final_candidate_left_m_s", "final_candidate_right_m_s"))
                     for state in (DISARMED, READY, COMPLETE, FAULT))
    unsafe = (
      supervise_command(RUNNING, False, True, True, safe, "NO_LOCALIZATION"),
      supervise_command(RUNNING, True, False, True, safe, "TIMING_INVALID"),
      supervise_command(RUNNING, True, True, False, safe),
      supervise_command(RUNNING, True, True, True, safe, emergency_stop=True),
    )
    require(state_zero and all(not item["command_allowed_in_simulated_execution"] and
                               all(abs(float(item[key])) <= 1e-15 for key in (
                                 "final_candidate_v_m_s", "final_candidate_omega_rad_s",
                                 "final_candidate_left_m_s", "final_candidate_right_m_s"))
                               for item in unsafe),
            "command supervisor violated a zero-candidate invariant")
  check("pure supervisor zero-candidate invariants", supervisor_zero_invariants)

  def supervisor_allows_only_execution() -> None:
    safe = {"v_safe_m_s": 0.1, "omega_safe_rad_s": 0.1,
            "v_left_safe_m_s": 0.08, "v_right_safe_m_s": 0.12,
            "real_safety_scale": 0.5}
    for state in (RUNNING, STOPPING):
      result = supervise_command(state, True, True, True, safe)
      require(result["command_allowed_in_simulated_execution"] and
              abs(float(result["final_candidate_v_m_s"]) - 0.1) <= 1e-15,
              f"valid {state} candidate was not preserved")
  check("RUNNING and STOPPING supervisor pass-through", supervisor_allows_only_execution)

  def status_timer_and_watchdog() -> None:
    session = running_session(config_path, 0.1)
    before = session.execution_time_s
    now_s = float(session.last_update_timestamp_s) + 0.1
    reference_now = session.interpolator.sample(before)
    status = session.status_check(exact_sample(reference_now, now_s), now_s)
    require(status["row"]["row_source"] == ROW_SOURCE_STATUS_TIMER and
            abs(session.execution_time_s - before) <= 1e-15 and
            status["state"] == RUNNING,
            "STATUS_TIMER advanced execution or changed nominal state")
    timeout_now = float(session.last_update_timestamp_s) + context["watchdog_timeout_s"] + 0.01
    timeout = session.status_check(exact_sample(reference_now, timeout_now), timeout_now)
    require(timeout["state"] == FAULT and final_is_zero(timeout) and
            timeout["row"]["fault_reason"] == "CONTROL_UPDATE_TIMEOUT",
            "pure status watchdog did not latch zero-candidate FAULT")
  check("non-progressing STATUS_TIMER and pure watchdog", status_timer_and_watchdog)

  def backward_and_gap_timing() -> None:
    backward = running_session(config_path)
    ref = backward.interpolator.sample(0.02)
    result = backward.step(0.02, exact_sample(ref, 9.99), 9.99)
    gap = running_session(config_path)
    dt_s = context["watchdog_timeout_s"] + 0.01
    gap_ref = gap.interpolator.sample(dt_s)
    timeout = gap.step(dt_s, exact_sample(gap_ref, 10.0 + dt_s), 10.0 + dt_s)
    require(result["state"] == FAULT and final_is_zero(result) and
            result["row"]["fault_reason"] == "BACKWARD_EXECUTION_TIMESTAMP" and
            timeout["state"] == FAULT and final_is_zero(timeout) and
            timeout["row"]["fault_reason"] == "CONTROL_UPDATE_TIMEOUT",
            "backward/large-gap execution timing was accepted")
  check("backward clock and large-gap rejection", backward_and_gap_timing)

  def fault_matrix() -> None:
    result = run_fault_injections(config_path)
    expected = {"stale_localization", "wrong_frame", "nan_pose", "invalid_quaternion",
                "translation_jump", "yaw_jump", "control_update_timeout",
                "backward_execution_timestamp", "explicit_emergency_stop",
                "missing_localization"}
    require(result["passed"] and result["case_count"] == result["passed_count"] == 10 and
            set(result["cases"]) == expected and
            all(case["entered_fault"] and case["final_candidate_zero"] and
                case["latched_until_reset"] and case["reset_state"] == DISARMED
                for case in result["cases"].values()),
            "ten-case fault/zero/latch/reset matrix failed")
  check("ten safety-critical fault injections", fault_matrix)

  def orderly_stop() -> None:
    session = running_session(config_path, 1.0)
    require(abs(float(session.last_final_candidate["final_candidate_v_m_s"])) > 0.0,
            "orderly-stop setup never produced a moving candidate")
    require(session.request_orderly_stop() == "RUNNING->STOPPING",
            "explicit orderly stop did not enter STOPPING")
    speeds = [abs(float(session.last_final_candidate["final_candidate_v_m_s"]))]
    now_s = float(session.last_update_timestamp_s)
    while session.state != COMPLETE:
      now_s += 0.02
      ref = session.interpolator.sample(session.execution_time_s + 0.02)
      result = session.step(0.02, exact_sample(ref, now_s), now_s)
      speeds.append(abs(float(result["final_candidate"]["final_candidate_v_m_s"])))
      require(result["state"] != FAULT, "orderly stop faulted")
    require(all(right <= left + 1e-12 for left, right in zip(speeds, speeds[1:])) and
            len(speeds) > 2 and speeds[-1] <= 1e-15 and
            session.stop_reason == "EXPLICIT_SIMULATED_ORDERLY_STOP" and
            session.state_history[-2:] == [STOPPING, COMPLETE],
            "orderly stop was abrupt, increased speed, or failed to complete")
  check("explicit orderly-stop monotonic deceleration", orderly_stop)

  def nominal_gate() -> None:
    session, rows, result = run_nominal(config_path)
    require(result["passed"] and
            session.state_history == [DISARMED, READY, RUNNING, STOPPING, COMPLETE] and
            result["metrics"]["final_execution_time_s"] == context["duration_s"] and
            result["metrics"]["final_reference_progress_m"] == context["lap_length_m"] and
            all(row["row_source"] == ROW_SOURCE_OFFLINE_SYNTHETIC for row in rows),
            "nominal one-lap offline execution Gate failed")
  check("nominal one-lap execution Gate", nominal_gate)

  def canonical_safe_separation() -> None:
    _, rows, result = run_nominal(config_path)
    metrics = result["metrics"]
    require(metrics["maximum_abs_canonical_v_m_s"] >
            metrics["maximum_abs_real_safe_v_m_s"] and
            metrics["maximum_abs_canonical_omega_rad_s"] >
            metrics["maximum_abs_real_safe_omega_rad_s"] and
            metrics["maximum_abs_real_safe_v_m_s"] <= 0.10 + 1e-12 and
            metrics["maximum_abs_real_safe_omega_rad_s"] <= 0.20 + 1e-12 and
            metrics["maximum_abs_real_safe_track_m_s"] <= 0.12 + 1e-12 and
            metrics["real_safety_clamped_row_count"] > 0,
            "canonical/real-safe candidates were overwritten or caps failed")
  check("canonical versus real-safe candidate separation", canonical_safe_separation)

  def sign_audit() -> None:
    result = command_sign_audit()
    cases = result["cases"]
    require(result["passed"] and not result["published"] and
            cases["straight_positive"]["left_m_s"] > 0.0 and
            cases["straight_positive"]["right_m_s"] > 0.0 and
            cases["positive_yaw"]["right_m_s"] > cases["positive_yaw"]["left_m_s"] and
            cases["negative_yaw"]["left_m_s"] > cases["negative_yaw"]["right_m_s"] and
            cases["in_place_positive"]["left_m_s"] < 0.0 <
            cases["in_place_positive"]["right_m_s"],
            "B=0.434 command-sign expectations failed")
  check("four-case differential command-sign audit", sign_audit)

  def csv_schema() -> None:
    required = {"execution_state", "execution_time_s", "execution_dt_s",
      "reference_index", "reference_next_index", "reference_interpolation_fraction",
      "reference_progress_m", "reference_lap_index", "planned_laps", "stop_reason",
      "fault_reason", "final_candidate_v", "final_candidate_omega",
      "final_candidate_left", "final_candidate_right", "final_candidate_allowed",
      "watchdog_state", "row_source", "execution_plan_identity"}
    require(set(SHADOW_CSV_COLUMNS) <= set(EXECUTION_CSV_COLUMNS) and
            required <= set(EXECUTION_CSV_COLUMNS) and
            len(EXECUTION_CSV_COLUMNS) == len(set(EXECUTION_CSV_COLUMNS)),
            "execution CSV does not add cleanly to the Shadow schema")
  check("additive execution CSV schema", csv_schema)

  def row_source_semantics() -> None:
    semantics = config["output"]["row_source_semantics"]
    require(config["output"]["future_tracking_performance_filter"] ==
            "row_source == POSE_UPDATE" and
            ROW_SOURCE_POSE_UPDATE in semantics and ROW_SOURCE_STATUS_TIMER in semantics and
            ROW_SOURCE_OFFLINE_SYNTHETIC in semantics and
            "only this source" in semantics[ROW_SOURCE_POSE_UPDATE] and
            "exclude" in semantics[ROW_SOURCE_STATUS_TIMER],
            "row provenance could duplicate-weight localization samples")
  check("row-source performance semantics", row_source_semantics)

  def frozen_baselines() -> None:
    paths = [
      "config/real_global_path_tracking_v1.json",
      "scripts/real_global_path_tracking_v1.py",
      "scripts/run_real_global_path_tracking_shadow_v1.py",
      "scripts/run_real_global_path_tracking_recorded_v1.py",
      "scripts/test_real_global_path_tracking_v1.py",
      "config/global_path_demo_v1.json",
      "scripts/global_path_demo_v1.py",
      "config/tracking_controller_v1.json",
      "scripts/tracking_controller_v1.py",
      "config/tracking_controller_v2_direction_aware.json",
      "scripts/tracking_controller_v2_direction_aware.py",
    ]
    completed = subprocess.run(["git", "diff", "--exit-code", "--", *paths],
                               cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    require(completed.returncode == 0 and not completed.stdout,
            "frozen Shadow/reference/controller baseline changed")
  check("Shadow and canonical baselines frozen", frozen_baselines)

  def no_command_runtime() -> None:
    paths = [
      REPO_ROOT / "scripts/real_global_path_execution_safety_v1.py",
      REPO_ROOT / "scripts/run_real_global_path_execution_offline_v1.py",
      REPO_ROOT / "scripts/run_real_global_path_tracking_shadow_v1.py",
    ]
    forbidden = ("create_publisher", ".publish(", "rclpy.publisher", "geometry_msgs.msg.Twist")
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    require(not any(token in source for token in forbidden) and
            "create_subscription" in paths[-1].read_text(encoding="utf-8"),
            "real execution/shadow code contains a command-publisher path")
  check("static no-command-publisher proof", no_command_runtime)

  if args.require_offline_output:
    def offline_output() -> None:
      directory = REPO_ROOT / config["output"]["default_directory"] / "offline_one_lap"
      csv_path = directory / config["output"]["offline_csv_filename"]
      summary_path = directory / config["output"]["offline_summary_filename"]
      require(csv_path.is_file() and summary_path.is_file(), "offline Gate output missing")
      summary = json.loads(summary_path.read_text(encoding="utf-8"))
      with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
      require(summary["gate"]["passed"] and
              summary["fault_injections"]["passed_count"] == 10 and
              summary["fault_injections"]["case_count"] == 10 and
              summary["nominal"]["metrics"]["row_count"] == len(rows) and
              fields == EXECUTION_CSV_COLUMNS and
              all(row["row_source"] == ROW_SOURCE_OFFLINE_SYNTHETIC for row in rows) and
              not summary["physical_execution_readiness"]["reference_timing_ready"] and
              not summary["physical_execution_readiness"]["future_publisher_ready"] and
              summary["physical_execution_readiness"]["guard_rejects_current_clock"] and
              summary["physical_execution_readiness"]["diagnostics"]
              ["minimum_required_uniform_time_scale_lower_bound"] > 1.0 and
              "semantics only" in summary["performance_claim"] and
              not summary["physical_command_publication"],
              "saved offline Gate/schema/provenance mismatch")
    check("saved offline one-lap output Gate", offline_output)

  print(f"result: {len(tests)}/{len(tests)} checks passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
