#!/usr/bin/env python3
"""Generate Stage-D map-route Reference Trajectory V1 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from map_route_reference_v1 import build_map_route_reference, plot_outputs, write_outputs

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/map_route_reference_v1.json")
  parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs/map_route_reference_v1")
  plot_group = parser.add_mutually_exclusive_group()
  plot_group.add_argument("--plot", action="store_true")
  plot_group.add_argument("--no-plot", action="store_true")
  args = parser.parse_args()
  result = build_map_route_reference(args.config.resolve())
  paths = write_outputs(result, args.output_dir.resolve())
  plot = plot_outputs(result, args.output_dir.resolve()) if args.plot and not args.no_plot else {"status": "disabled"}
  source, processing, reference = (result["summary"][key] for key in ("source_data", "processing", "reference"))
  print(f"source poses={source['pose_count']}, duration={source['duration_s']:.6f} s, "
        f"raw_xy_length={source['raw_xy_polyline_length_m']:.6f} m")
  print(f"cleaned={processing['cleaning']['retained_count']} poses, resampled={processing['resampling']['point_count']}, "
        f"processed_length={reference['processed_path_length_m']:.6f} m")
  print(f"new duration={reference['new_duration_s']:.6f} s, max_speed={reference['maximum_reference_speed_m_s']:.6f} m/s, "
        f"max_yaw_rate={reference['maximum_abs_yaw_rate_rad_s']:.6f} rad/s")
  print(f"reference={paths['reference']}\nsummary={paths['summary']}\nplots={json.dumps(plot)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
