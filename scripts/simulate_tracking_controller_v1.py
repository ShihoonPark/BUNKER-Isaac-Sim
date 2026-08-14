#!/usr/bin/env python3
"""Simulate Tracking Controller V1 against an exact kinematic unicycle."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from reference_trajectory_v1 import build_reference_trajectory, load_config as load_reference_config  # noqa: E402
from tracking_controller_v1 import (  # noqa: E402
  ReferenceInterpolator, controller_command, integrate_unicycle_exact,
  load_controller_config, project_to_polyline, wrap_to_pi, write_csv,
)


def scenario_pose(scenario: dict[str, Any]) -> tuple[float, float, float]:
  yaw = float(scenario.get("yaw_rad", math.radians(float(scenario.get("yaw_deg", 0.0)))))
  return float(scenario["x_m"]), float(scenario["y_m"]), yaw


def simulate_scenario(name: str, initial_pose: tuple[float, float, float],
                      reference_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
  dt = float(config["simulation"]["time_step_s"])
  end_time = float(reference_rows[-1]["t_s"]) + float(config["simulation"]["terminal_hold_s"])
  step_count = math.ceil(end_time / dt)
  interpolator = ReferenceInterpolator(reference_rows)
  pose = initial_pose
  rows = []
  for step in range(step_count + 1):
    time_s = min(step * dt, end_time)
    reference = interpolator.sample(time_s)
    command = controller_command(pose, reference, config)
    projection = project_to_polyline(pose[0], pose[1], reference_rows)
    rows.append({
      "time_s": time_s,
      "reference_x_m": reference["x_m"], "reference_y_m": reference["y_m"],
      "reference_yaw_rad": reference["yaw_rad"],
      "actual_x_m": pose[0], "actual_y_m": pose[1], "actual_yaw_rad": pose[2],
      "reference_s_m": reference["s_m"], "projected_s_m": projection["s_projected_m"],
      "projected_x_m": projection["projected_x_m"], "projected_y_m": projection["projected_y_m"],
      "e_x_m": command["e_x_m"], "e_y_m": command["e_y_m"],
      "heading_error_rad": command["e_heading_rad"],
      "cross_track_error_m": projection["cross_track_error_m"],
      "progress_error_m": reference["s_m"] - projection["s_projected_m"],
      "v_ref_m_s": reference["v_ref_m_s"], "omega_ref_rad_s": reference["omega_ref_rad_s"],
      "a_ref_m_s2": reference["a_ref_m_s2"],
      "v_raw_m_s": command["v_raw_m_s"], "omega_raw_rad_s": command["omega_raw_rad_s"],
      "v_cmd_m_s": command["v_cmd_m_s"], "omega_cmd_rad_s": command["omega_cmd_rad_s"],
      "v_actual_m_s": command["v_cmd_m_s"], "omega_actual_rad_s": command["omega_cmd_rad_s"],
      "speed_error_m_s": reference["v_ref_m_s"] - command["v_cmd_m_s"],
      "yaw_rate_error_rad_s": reference["omega_ref_rad_s"] - command["omega_cmd_rad_s"],
      "v_left_cmd_m_s": command["v_left_cmd_m_s"],
      "v_right_cmd_m_s": command["v_right_cmd_m_s"], "command_scale": command["command_scale"],
    })
    if step < step_count:
      integration_dt = min(dt, end_time - time_s)
      pose = integrate_unicycle_exact(
        pose, command["v_cmd_m_s"], command["omega_cmd_rad_s"], integration_dt,
        float(config["simulation"]["near_zero_yaw_rate_rad_s"]))
  return rows


def rms(values: list[float]) -> float:
  return math.sqrt(sum(value * value for value in values) / len(values))


def evaluation_window_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
  cross = [float(row["cross_track_error_m"]) for row in rows]
  heading = [float(row["heading_error_rad"]) for row in rows]
  progress = [float(row["progress_error_m"]) for row in rows]
  speed = [float(row["speed_error_m_s"]) for row in rows]
  saturated = sum(float(row["command_scale"]) < 1.0 - 1e-12 for row in rows)
  return {
    f"{prefix}_sample_count": len(rows),
    f"{prefix}_rms_cross_track_error_m": rms(cross),
    f"{prefix}_maximum_abs_cross_track_error_m": max(abs(value) for value in cross),
    f"{prefix}_rms_heading_error_rad": rms(heading),
    f"{prefix}_maximum_abs_heading_error_rad": max(abs(value) for value in heading),
    f"{prefix}_rms_progress_error_m": rms(progress),
    f"{prefix}_maximum_abs_progress_error_m": max(abs(value) for value in progress),
    f"{prefix}_rms_speed_error_m_s": rms(speed),
    f"{prefix}_saturated_sample_count": saturated,
    f"{prefix}_saturated_sample_fraction": saturated / len(rows),
  }


def summarize_scenario(rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]) -> dict[str, Any]:
  final = rows[-1]
  goal = reference_rows[-1]
  reference_end_time = float(reference_rows[-1]["t_s"])
  active_rows = [row for row in rows if float(row["time_s"]) <= reference_end_time + 1e-12]
  if not active_rows:
    raise ValueError("active tracking window contains no samples")
  return {
    "reference_end_time_s": reference_end_time,
    "active_window_last_time_s": float(active_rows[-1]["time_s"]),
    **evaluation_window_metrics(active_rows, "active"),
    **evaluation_window_metrics(rows, "full_run"),
    "final_xy_position_error_to_goal_m": math.hypot(
      final["actual_x_m"] - goal["x_m"], final["actual_y_m"] - goal["y_m"]),
    "final_heading_error_rad": wrap_to_pi(goal["yaw_rad"] - final["actual_yaw_rad"]),
    "final_abs_v_cmd_m_s": abs(final["v_cmd_m_s"]),
    "final_abs_omega_cmd_rad_s": abs(final["omega_cmd_rad_s"]),
    "maximum_abs_v_cmd_m_s": max(abs(row["v_cmd_m_s"]) for row in rows),
    "maximum_abs_omega_cmd_rad_s": max(abs(row["omega_cmd_rad_s"]) for row in rows),
    "maximum_abs_left_track_command_m_s": max(abs(row["v_left_cmd_m_s"]) for row in rows),
    "maximum_abs_right_track_command_m_s": max(abs(row["v_right_cmd_m_s"]) for row in rows),
    "minimum_command_scale": min(row["command_scale"] for row in rows),
    "initial_cross_track_error_m": float(rows[0]["cross_track_error_m"]),
    "final_cross_track_error_m": float(rows[-1]["cross_track_error_m"]),
    "initial_heading_error_rad": float(rows[0]["heading_error_rad"]),
    "final_controller_heading_error_rad": float(rows[-1]["heading_error_rad"]),
  }


def run_all(config: dict[str, Any], reference_config_path: Path,
            output_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
  reference = build_reference_trajectory(load_reference_config(reference_config_path))["trajectory"]
  output_dir.mkdir(parents=True, exist_ok=True)
  scenarios = {}
  summaries = {}
  for name, initial in config["simulation"]["scenarios"].items():
    rows = simulate_scenario(name, scenario_pose(initial), reference, config)
    scenarios[name] = rows
    summaries[name] = summarize_scenario(rows, reference)
    write_csv(output_dir / f"{name}.csv", rows)
  summary = {
    "calibration_warning": config["metadata"]["calibration_status"],
    "modeling_scope": "Exact kinematic unicycle validation plant; not Isaac Sim or a real actuator model.",
    "controller_config": config, "reference_config_path": str(reference_config_path),
    "reference_duration_s": reference[-1]["t_s"],
    "terminal_hold_s": config["simulation"]["terminal_hold_s"],
    "evaluation_windows": {
      "active": "ACTIVE TRACKING WINDOW: samples with time_s <= reference_end_time_s + 1e-12.",
      "full_run": "FULL RUN INCLUDING TERMINAL HOLD: every simulation sample.",
    },
    "scenarios": summaries,
  }
  with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, allow_nan=False)
    stream.write("\n")
  return scenarios, summary


def plot_results(scenarios: dict[str, list[dict[str, Any]]], output_path: Path) -> tuple[bool, str]:
  try:
    import matplotlib.pyplot as plt
  except ImportError as error:
    return False, f"skipped; matplotlib unavailable: {error}"
  figure, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
  for row_index, (name, rows) in enumerate(scenarios.items()):
    times = [row["time_s"] for row in rows]
    axes[row_index, 0].plot([row["reference_x_m"] for row in rows],
                            [row["reference_y_m"] for row in rows], "k--", label="reference")
    axes[row_index, 0].plot([row["actual_x_m"] for row in rows],
                            [row["actual_y_m"] for row in rows], label="actual")
    axes[row_index, 0].set_aspect("equal", adjustable="box")
    axes[row_index, 0].set_title(f"{name}: XY")
    axes[row_index, 0].legend()
    axes[row_index, 1].plot(times, [row["cross_track_error_m"] for row in rows], label="cross-track")
    axes[row_index, 1].plot(times, [row["heading_error_rad"] for row in rows], label="heading")
    axes[row_index, 1].set_title(f"{name}: errors")
    axes[row_index, 1].legend()
    axes[row_index, 2].plot(times, [row["v_cmd_m_s"] for row in rows], label="v")
    axes[row_index, 2].plot(times, [row["omega_cmd_rad_s"] for row in rows], label="omega")
    axes[row_index, 2].set_title(f"{name}: commands")
    axes[row_index, 2].legend()
    for axis in axes[row_index]:
      axis.grid(True, alpha=0.25)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, dpi=160)
  plt.close(figure)
  return True, str(output_path)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/tracking_controller_v1.json")
  parser.add_argument("--reference-config", type=Path, default=None)
  parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs/tracking_controller_v1")
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--plot", dest="plot", action="store_true")
  group.add_argument("--no-plot", dest="plot", action="store_false")
  parser.set_defaults(plot=False)
  args = parser.parse_args()
  config = load_controller_config(args.config.resolve())
  reference_path = (args.reference_config.resolve() if args.reference_config else
                    REPO_ROOT / config["reference"]["config_path"])
  scenarios, summary = run_all(config, reference_path, args.output_dir.resolve())
  plot_status = "disabled"
  if args.plot:
    plotted, plot_status = plot_results(
      scenarios, args.output_dir.resolve() / config["output"]["plot_filename"])
    if not plotted:
      print(f"plot: {plot_status}")
  print(f"reference duration: {summary['reference_duration_s']:.6f} s; "
        f"terminal hold: {summary['terminal_hold_s']:.3f} s")
  for name, metrics in summary["scenarios"].items():
    print(f"{name}: active_rms_cross={metrics['active_rms_cross_track_error_m']:.6f} m, "
          f"full_rms_cross={metrics['full_run_rms_cross_track_error_m']:.6f} m, "
          f"active_rms_heading={metrics['active_rms_heading_error_rad']:.6f} rad, "
          f"final_xy={metrics['final_xy_position_error_to_goal_m']:.6f} m, "
          f"final_heading={metrics['final_heading_error_rad']:.6f} rad, "
          f"active_saturated={metrics['active_saturated_sample_count']}/"
          f"{metrics['active_sample_count']}, full_saturated="
          f"{metrics['full_run_saturated_sample_count']}/{metrics['full_run_sample_count']}")
  if args.plot and plot_status != "disabled":
    print(f"plot: {plot_status}")
  print(f"outputs: {args.output_dir.resolve()}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
