#!/usr/bin/env python3
"""Run one GUI Stage-E case with observer-only GLIM/map-route overlays."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from map_route_reference_v1 import build_map_route_reference, parse_tum  # noqa: E402
from map_route_visualization_v1 import (  # noqa: E402
  MapRouteVisualizationObserver, alignment_diagnostics, forward_wobble_diagnostics, load_ply,
  transform_map_points,
)
from run_map_route_isaac_closed_loop_v1 import (  # noqa: E402
  load_stage_config, plot_results, run_closed_loop, summarize, write_csv,
)
from test_tracked_force_plant_v2 import load_config as load_plant_config, make_world  # noqa: E402
from tracking_controller_v2_direction_aware import load_direction_aware_config  # noqa: E402


def load_visualization_config(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    config = json.load(stream)
  if not config["metadata"]["all_parameters_are_visualization_only"]:
    raise ValueError("visualization config is not marked display-only")
  display = config["display"]
  for key in ("point_size_m", "raw_route_width_m", "reference_width_m", "actual_trail_width_m",
              "pivot_marker_radius_m", "endpoint_marker_radius_m", "actual_trail_update_period_s"):
    if float(display[key]) <= 0.0:
      raise ValueError(f"display.{key} must be positive")
  return config


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/map_route_visualization_v1.json")
  parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs/map_route_visualization_v1")
  parser.add_argument("--gui", action="store_true"); parser.add_argument("--realtime", action="store_true")
  parser.add_argument("--plot", action="store_true"); args = parser.parse_args()
  if not args.gui or not args.realtime:
    parser.error("Map Route Visualization V1 requires --gui --realtime")
  from isaacsim import SimulationApp
  app = SimulationApp({"headless": False, "enable_cameras": True})
  try:
    visual_path = args.config.resolve(); visual = load_visualization_config(visual_path)
    stage_path = (REPO_ROOT / visual["stage_e_config"]).resolve(); stage = load_stage_config(stage_path)
    reference_result = build_map_route_reference(REPO_ROOT / stage["baseline_configs"]["map_route_reference"])
    raw_rows = parse_tum(Path(visual["raw_trajectory_path"]),
                         float(reference_result["config"]["diagnostic_guards"]["quaternion_norm_tolerance"]))
    ply = load_ply(Path(visual["point_cloud_path"]))
    _, controller = load_direction_aware_config(REPO_ROOT / stage["baseline_configs"]["direction_aware_controller"])
    plant = load_plant_config(REPO_ROOT / stage["baseline_configs"]["tracked_force_plant_v2"])
    with (REPO_ROOT / stage["comparison"]["kinematic_v2_summary"]).open(encoding="utf-8") as stream:
      kinematic_summary = json.load(stream)
    output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    usd_path = output_dir / visual["output"]["stage_filename"]
    support = plant["tunable_uncalibrated"]["support_mode"]
    world, plant_controller, build_info = make_world(
      plant, usd_path, stage["runtime"]["creation_order"], bool(stage["runtime"]["detailed_contacts"]), support)
    observer = MapRouteVisualizationObserver(visual, reference_result, raw_rows, ply)
    configs = {"stage": stage, "controller": controller, "plant": plant}
    rows, settled, final = run_closed_loop(world, plant_controller, reference_result["trajectory"], configs,
                                            render=True, realtime=True, observer=observer)
    summary = summarize(rows, reference_result["trajectory"], configs, settled, final, kinematic_summary)
    transform = reference_result["summary"]["reference"]["maneuver_normalization_transform"]
    map_points = transform_map_points(ply["points"], transform, float(visual["display"]["map_visual_z_offset_m"]))
    summary["visualization"] = {
      "config_path": str(visual_path), "stage_e_config_path": str(stage_path),
      "ply": {key: ply[key] for key in ("format", "vertex_count", "fields", "bounds_xyz")},
      "coordinate_alignment": alignment_diagnostics(raw_rows, reference_result["trajectory"], map_points, transform),
      "stage_d_transform": transform, "settled_isaac_transform": settled,
      "z_policy": {"map_visual_z_offset_m": visual["display"]["map_visual_z_offset_m"],
                   "meaning": "display-only; source Z retained; no physical terrain or LiDAR/base Z registration"},
      "observer": observer.diagnostics(),
      "forward_wobble": forward_wobble_diagnostics(
        rows, float(reference_result["trajectory"][-1]["t_s"])),
      "visual_run_role": "visual verification only; canonical Stage-E headless result remains quantitative baseline",
      "limitations": ["GLIM cloud and T_map_lidar route share the map frame.",
                      "Reference is LiDAR sensor-center-derived while Isaac tracks body/base; exact T_base_lidar is unknown.",
                      "PLY has no collision role and its Z is not a validated terrain model."]}
    dataset_source = visual.get("metadata", {}).get("dataset_source")
    if dataset_source:
      with (REPO_ROOT / dataset_source).open(encoding="utf-8") as stream:
        summary["dataset_provenance"] = json.load(stream)
    summary["plant_build"] = build_info
    write_csv(output_dir / visual["output"]["csv_filename"], rows)
    with (output_dir / visual["output"]["summary_filename"]).open("w", encoding="utf-8") as stream:
      json.dump(summary, stream, indent=2, allow_nan=False); stream.write("\n")
    plot_status = "disabled" if not args.plot else plot_results(rows, output_dir / visual["output"]["plot_filename"])
    import omni.usd
    omni.usd.get_context().save_as_stage(str(usd_path), None)
    print(f"Map Route Visualization V1 complete: points={ply['vertex_count']}, display trail={len(observer.sampler.points)}")
    print(f"usd={usd_path}\nsummary={output_dir / visual['output']['summary_filename']}\nplot={plot_status}")
    return 0
  except BaseException:
    traceback.print_exc(); raise
  finally:
    app.close()


if __name__ == "__main__":
  raise SystemExit(main())
