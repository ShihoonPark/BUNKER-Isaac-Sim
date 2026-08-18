#!/usr/bin/env python3
"""Run one Stage C command-headroom case with a smoothed reference."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from curvature_smoothed_reference import build_smoothed_reference  # noqa: E402
from run_isaac_closed_loop_v1 import (  # noqa: E402
  load_stage_config, plot_results, run_closed_loop, summarize, write_csv,
)
from test_tracked_force_plant_v2 import load_config as load_plant_config, make_world  # noqa: E402
from tracking_controller_v1 import load_controller_config  # noqa: E402


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
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
    reference_result = build_smoothed_reference(paths["reference_trajectory"])
    controller_cfg = load_controller_config(paths["tracking_controller"])
    plant_cfg = load_plant_config(paths["tracked_force_plant_v2"])
    if float(controller_cfg["measured_fixed"]["track_center_distance_m"]) != 0.434 or float(
        plant_cfg["measured_fixed"]["track_center_distance_b_m"]) != 0.434:
      raise RuntimeError("track-center distance mismatch")
    if plant_cfg["tunable_uncalibrated"]["support_mode"] != "flat_track_boxes":
      raise RuntimeError("experiment requires canonical flat_track_boxes support")
    configs = {"stage": stage, "controller": controller_cfg, "plant": plant_cfg}
    output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    world, plant_controller, build_info = make_world(
      plant_cfg, output_dir / stage["output"]["stage_filename"],
      stage["runtime"]["creation_order"], bool(stage["runtime"]["detailed_contacts"]),
      stage["runtime"]["support_mode_required"])
    reference_rows = reference_result["trajectory"]
    rows, settled, final_observation = run_closed_loop(
      world, plant_controller, reference_rows, configs, args.gui, args.realtime)
    summary = summarize(rows, reference_rows, configs, settled, final_observation)
    summary["stage_config_path"] = str(stage_path)
    summary["plant_build"] = build_info
    summary["experimental_reference"] = {
      "method": "symmetric_linear_curvature_ramp",
      "transition_heading_per_ramp_rad": 0.10,
      "path_length_m": reference_result["summary"]["path_length_m"],
      "total_duration_s": reference_result["summary"]["total_duration_s"],
      "corners": reference_result["corners"],
      "adjusted_straight_lengths_m": reference_result["adjusted_straight_lengths_m"],
    }
    csv_path = output_dir / stage["output"]["csv_filename"]
    summary_path = output_dir / stage["output"]["summary_filename"]
    write_csv(csv_path, rows)
    with summary_path.open("w", encoding="utf-8") as stream:
      json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
    plot_status = "disabled"
    if args.plot:
      _, plot_status = plot_results(rows, output_dir / stage["output"]["plot_filename"])
    print(f"reference_length={reference_rows[-1]['s_m']:.9f} m, "
          f"duration={reference_rows[-1]['t_s']:.9f} s")
    print(f"active CTE RMS={summary['active_rms_cross_track_error_m']:.6f} m, "
          f"heading RMS={summary['active_rms_heading_error_rad']:.6f} rad")
    print(f"csv={csv_path}\nsummary={summary_path}\nplot={plot_status}")
    return 0
  except BaseException:
    traceback.print_exc(); raise
  finally:
    app.close()


if __name__ == "__main__":
  raise SystemExit(main())
