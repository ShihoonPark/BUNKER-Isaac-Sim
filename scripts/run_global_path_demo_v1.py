#!/usr/bin/env python3
"""Run one selected manual closed-loop global path in exact kinematics or canonical Isaac V2."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from global_path_demo_v1 import (  # noqa: E402
  build_demo_reference, evaluate_gate, lap_metrics, metric_block, write_csv, write_json,
)
from map_route_kinematic_validation_v1 import load_validation_config  # noqa: E402
from map_route_visualization_v1 import load_ply  # noqa: E402
from simulate_map_route_kinematic_validation_v2 import simulate_v2  # noqa: E402
from tracking_controller_v2_direction_aware import load_direction_aware_config  # noqa: E402


def _active(rows: list[dict[str, Any]], time_key: str, end_time: float) -> list[dict[str, Any]]:
  return [row for row in rows if float(row[time_key]) <= end_time + 1e-12]


def _projection_diagnostics(rows: list[dict[str, Any]], time_key: str,
                            end_time: float, backward_tolerance: float) -> dict[str, Any]:
  active = _active(rows, time_key, end_time)
  progress = [float(row["projected_s_m"]) for row in active]
  decreases = [right - left for left, right in zip(progress, progress[1:])
               if right < left - backward_tolerance]
  return {"metrics_only": True,
          "method": "time-indexed local route neighborhood over total lap-unwrapped progress",
          "backward_numerical_tolerance_m": backward_tolerance,
          "nonmonotonic_step_count": len(decreases),
          "largest_backward_step_m": min(decreases, default=0.0),
          "initial_projected_progress_m": progress[0], "final_projected_progress_m": progress[-1]}


def _common_metrics(rows: list[dict[str, Any]], result: dict[str, Any], kind: str,
                    time_key: str, reference_s_key: str, curvature_key: str,
                    speed_error_key: str) -> dict[str, Any]:
  config = result["config"]; trajectory = result["trajectory"]
  end_time = float(trajectory[-1]["t_s"]); laps = int(result["summary"]["execution_profile"]["laps"])
  lap_length = float(result["summary"]["geometry"]["lap_length_m"])
  active = _active(rows, time_key, end_time)
  overall = metric_block(active, float(config["execution"]["saturation_scale_tolerance"]), speed_error_key)
  per_lap = lap_metrics(rows, lap_length, laps, config, time_key, reference_s_key,
                        curvature_key, speed_error_key, end_time)
  metrics = {"overall": overall, "lap_metrics": per_lap,
             "projection": _projection_diagnostics(
               rows, time_key, end_time,
               float(config["evaluation"]["projection_backward_numerical_tolerance_m"]))}
  metrics["gate"] = evaluate_gate(metrics, config, kind)
  return metrics


def _annotate_kinematic(rows: list[dict[str, Any]], trajectory: list[dict[str, Any]]) -> None:
  for row in rows:
    source = trajectory[int(row["reference_index"])]
    row["reference_curvature_1_m"] = float(source["curvature_ref_1_m"])
    row["reference_lap_index"] = int(source["lap_index"])
    row["reference_s_within_lap_m"] = float(source["s_within_lap_m"])
    row["speed_error_m_s"] = float(row["v_ref_m_s"]) - float(row["v_cmd_m_s"])


def run_kinematic(result: dict[str, Any], output_dir: Path,
                  plot_enabled: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  config = result["config"]; trajectory = result["trajectory"]
  _, controller = load_direction_aware_config(
    REPO_ROOT / config["baseline_configs"]["direction_aware_controller"])
  validation = load_validation_config(
    REPO_ROOT / config["baseline_configs"]["v1_kinematic_validation"])
  expected_window = int(config["execution"]["projection_half_window_segments"])
  if int(validation["evaluation"]["ordered_projection_half_window_segments"]) != expected_window:
    raise RuntimeError("kinematic and demo projection windows differ")
  initial = (float(trajectory[0]["x_m"]), float(trajectory[0]["y_m"]), float(trajectory[0]["yaw_rad"]))
  rows = simulate_v2(trajectory, initial, controller, validation, result["summary"]["preset"])
  _annotate_kinematic(rows, trajectory)
  metrics = _common_metrics(rows, result, "kinematic", "time_s", "reference_s_m",
                            "reference_curvature_1_m", "speed_error_m_s")
  summary = {"stage": "Multiple Closed-Loop Global Paths V1", "run_kind": "exact_kinematic_nominal",
             "reference": result["summary"], **metrics,
             "controller": {"name": "Direction-Aware Controller V2", "gains": controller["controller"],
                            "command_limits": controller["command_limits"]},
             "run_policy": "one deterministic nominal run; no gain, plant, or path-specific parameter sweep",
             "limitations": ["Exact unicycle kinematics only; no actuator, force, contact, slip, delay, or noise."]}
  output_dir.mkdir(parents=True, exist_ok=True)
  write_csv(output_dir / config["output"]["reference_csv_filename"], trajectory)
  write_csv(output_dir / config["output"]["kinematic_csv_filename"], rows)
  write_json(output_dir / config["output"]["kinematic_summary_filename"], summary)
  if plot_enabled:
    plot_results(result, rows, summary, output_dir / config["output"]["summary_plot_filename"], "kinematic")
  return summary, rows


def _isaac_stage_config(config: dict[str, Any]) -> dict[str, Any]:
  return {"metadata": config["metadata"], "baseline_configs": config["baseline_configs"],
          "runtime": {**config["execution"],
                      "ordered_projection_half_window_segments": config["execution"]["projection_half_window_segments"]}}


def run_isaac(result: dict[str, Any], output_dir: Path, gui: bool, realtime: bool,
              plot_enabled: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  config = result["config"]; trajectory = result["trajectory"]
  from global_path_demo_visualization_v1 import GlobalPathDemoObserver
  from run_map_route_isaac_closed_loop_v1 import run_closed_loop
  from test_tracked_force_plant_v2 import load_config as load_plant_config, make_world
  _, controller = load_direction_aware_config(
    REPO_ROOT / config["baseline_configs"]["direction_aware_controller"])
  plant = load_plant_config(REPO_ROOT / config["baseline_configs"]["tracked_force_plant_v2"])
  if float(controller["measured_fixed"]["track_center_distance_m"]) != float(
      plant["measured_fixed"]["track_center_distance_b_m"]):
    raise RuntimeError("controller/plant track-center spacing mismatch")
  support = plant["tunable_uncalibrated"]["support_mode"]
  if support != config["execution"]["support_mode_required"] or support != "flat_track_boxes":
    raise RuntimeError(f"Global Path Demo V1 requires flat_track_boxes, got {support}")
  output_dir.mkdir(parents=True, exist_ok=True)
  usd_path = output_dir / config["output"]["stage_filename"]
  world, plant_controller, build_info = make_world(
    plant, usd_path, config["execution"]["creation_order"],
    bool(config["execution"]["detailed_contacts"]), support)
  observer = None
  if gui:
    ply = load_ply(Path(config["map"]["point_cloud_path"]))
    if ply["vertex_count"] != int(config["map"]["expected_raw_point_count"]):
      raise RuntimeError("Bag C PLY identity changed")
    observer = GlobalPathDemoObserver(config, result, ply,
                                      int(result["summary"]["execution_profile"]["laps"]))
  stage = _isaac_stage_config(config)
  configs = {"stage": stage, "controller": controller, "plant": plant}
  rows, settled, final = run_closed_loop(world, plant_controller, trajectory, configs,
                                          render=gui, realtime=realtime, observer=observer)
  metrics = _common_metrics(rows, result, "isaac", "physics_time_s", "reference_s_m",
                            "reference_curvature_1_m", "speed_error_m_s")
  normal_tolerance = float(config["execution"]["custom_force_normal_tolerance_n"])
  plant_metrics = {
    "left_unsupported_sample_count": sum(not int(row["left_supported"]) for row in rows),
    "right_unsupported_sample_count": sum(not int(row["right_supported"]) for row in rows),
    "maximum_abs_custom_force_normal_component_n": max(abs(float(row["max_abs_custom_force_normal_component_n"])) for row in rows),
    "maximum_abs_roll_rad": max(abs(float(row["roll_rad"])) for row in rows),
    "maximum_abs_pitch_rad": max(abs(float(row["pitch_rad"])) for row in rows),
    "maximum_abs_vertical_velocity_m_s": max(abs(float(row["world_vz_m_s"])) for row in rows),
  }
  metrics["gate"]["checks"].update({
    "left_support_continuous": plant_metrics["left_unsupported_sample_count"] == 0,
    "right_support_continuous": plant_metrics["right_unsupported_sample_count"] == 0,
    "custom_force_tangent_only": plant_metrics["maximum_abs_custom_force_normal_component_n"] <= normal_tolerance,
    "controlled_final_stop": abs(float(final["v_actual_m_s"])) <= 0.05 and abs(float(final["omega_actual_rad_s"])) <= 0.08,
  })
  metrics["gate"]["passed"] = all(metrics["gate"]["checks"].values())
  summary = {"stage": "Multiple Closed-Loop Global Paths V1", "run_kind": "canonical_isaac_v2_nominal",
             "reference": result["summary"], **metrics, "plant": plant_metrics,
             "settled_world_origin": settled, "final_post_step_observation": final,
             "plant_build": build_info,
             "visualization": None if observer is None else observer.diagnostics(),
             "controller": {"name": "Direction-Aware Controller V2", "gains": controller["controller"],
                            "command_limits": controller["command_limits"]},
             "run_policy": "one deterministic nominal run; canonical plant/controller values unchanged",
             "limitations": ["Isaac V2 is a directional-force tracked-plant simplification, not articulated belts.",
                             "Contact, actuator, friction, and controller parameters remain uncalibrated against the real BUNKER."]}
  write_csv(output_dir / config["output"]["reference_csv_filename"], trajectory)
  write_csv(output_dir / config["output"]["isaac_csv_filename"], rows)
  write_json(output_dir / config["output"]["isaac_summary_filename"], summary)
  if plot_enabled:
    plot_results(result, rows, summary, output_dir / config["output"]["summary_plot_filename"], "isaac")
  import omni.usd
  omni.usd.get_context().save_as_stage(str(usd_path), None)
  return summary, rows


def plot_results(result: dict[str, Any], rows: list[dict[str, Any]], summary: dict[str, Any],
                 path: Path, kind: str) -> str:
  os.environ.setdefault("MPLCONFIGDIR", "/tmp/global_path_demo_v1_matplotlib")
  try:
    import matplotlib.pyplot as plt
  except ImportError as error:
    return f"skipped; matplotlib unavailable: {error}"
  time_key = "time_s" if kind == "kinematic" else "physics_time_s"
  actual_x = "actual_x_m" if kind == "kinematic" else "actual_local_x_m"
  actual_y = "actual_y_m" if kind == "kinematic" else "actual_local_y_m"
  end_time = float(result["trajectory"][-1]["t_s"])
  active = _active(rows, time_key, end_time)
  figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
  global_xy = result["global_path_local"]
  axes[0, 0].plot([point[0] for point in global_xy], [point[1] for point in global_xy],
                  color="tab:cyan", linewidth=1.0, label="manual global path")
  lap = result["periodic_lap"]
  axes[0, 0].plot([row["x_m"] for row in lap], [row["y_m"] for row in lap],
                  color="tab:green", linewidth=2.0, label="velocity-aware reference")
  axes[0, 0].plot([row[actual_x] for row in active], [row[actual_y] for row in active],
                  color="tab:orange", linewidth=1.0, label=f"{kind} actual")
  axes[0, 0].scatter([0.0], [0.0], color="green", s=35, label="start")
  axes[0, 0].set_aspect("equal", adjustable="box"); axes[0, 0].set_title(result["summary"]["preset"])
  axes[0, 0].legend(fontsize=8)
  trajectory = result["trajectory"]
  axes[0, 1].plot([row["s_m"] for row in trajectory], [row["v_ref_m_s"] for row in trajectory],
                  label="v_ref")
  axes[0, 1].plot([row["s_m"] for row in trajectory], [row["v_periodic_m_s"] for row in trajectory],
                  "--", alpha=.7, label="steady periodic cap")
  for lap_index in range(1, int(result["summary"]["execution_profile"]["laps"])):
    axes[0, 1].axvline(lap_index * float(result["summary"]["geometry"]["lap_length_m"]), color="k", alpha=.2)
  axes[0, 1].set_title("Launch, periodic laps, and final stop"); axes[0, 1].set_xlabel("total s [m]"); axes[0, 1].legend()
  axes[1, 0].plot([row[time_key] for row in active], [row["cross_track_error_m"] for row in active], label="CTE")
  axes[1, 0].plot([row[time_key] for row in active], [row["e_y_m"] for row in active], alpha=.75, label="e_y")
  axes[1, 0].set_title("Lateral tracking / oscillation context"); axes[1, 0].set_xlabel("time [s]"); axes[1, 0].legend()
  laps = summary["lap_metrics"]["laps"]
  labels = list(laps); positions = list(range(len(labels)))
  axes[1, 1].bar([value - .18 for value in positions], [laps[label]["rms_cte_m"] for label in labels], .36, label="CTE RMS")
  axes[1, 1].bar([value + .18 for value in positions], [laps[label]["rms_heading_rad"] for label in labels], .36, label="heading RMS")
  axes[1, 1].set_xticks(positions, [f"lap {label}" for label in labels]); axes[1, 1].set_title("Per-lap tracking"); axes[1, 1].legend()
  for axis in axes.flat: axis.grid(True, alpha=.25)
  figure.savefig(path, dpi=160); plt.close(figure)
  return str(path)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/global_path_demo_v1.json")
  parser.add_argument("--path", choices=("rounded_loop", "zigzag_loop", "lawnmower_loop"), required=True)
  parser.add_argument("--laps", type=int)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument("--kinematic", action="store_true", help="run the exact-unicycle nominal gate instead of Isaac V2")
  parser.add_argument("--gui", action="store_true"); parser.add_argument("--realtime", action="store_true")
  parser.add_argument("--plot", action="store_true"); args = parser.parse_args()
  if args.realtime and not args.gui: parser.error("--realtime requires --gui")
  if args.kinematic and (args.gui or args.realtime): parser.error("--kinematic does not create an Isaac GUI")
  result = build_demo_reference(args.config.resolve(), args.path, args.laps)
  base = REPO_ROOT / result["config"]["output"]["default_directory"] / args.path
  output_dir = (args.output_dir or base).resolve()
  if args.kinematic:
    summary, _ = run_kinematic(result, output_dir, args.plot)
  else:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not args.gui, "enable_cameras": args.gui})
    try: summary, _ = run_isaac(result, output_dir, args.gui, args.realtime, args.plot)
    except BaseException:
      traceback.print_exc(); raise
    finally: app.close()
  overall = summary["overall"]; gate = summary["gate"]
  print(f"{args.path} {summary['run_kind']} complete: gate={'PASS' if gate['passed'] else 'FAIL'}")
  print(f"laps={result['summary']['execution_profile']['laps']}, duration={result['summary']['execution_profile']['duration_s']:.3f}s, "
        f"CTE RMS/max={overall['rms_cte_m']:.4f}/{overall['maximum_abs_cte_m']:.4f}m, "
        f"heading RMS/max={overall['rms_heading_rad']:.4f}/{overall['maximum_abs_heading_rad']:.4f}rad")
  print(f"output={output_dir}")
  return 0 if gate["passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
