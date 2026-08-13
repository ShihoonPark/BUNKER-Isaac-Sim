#!/usr/bin/env python3
"""Generate Reference Trajectory V1 CSV/JSON outputs and an optional plot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from reference_trajectory_v1 import build_reference_trajectory, load_config, write_outputs  # noqa: E402


def plot_result(result: dict[str, Any], output_path: Path) -> tuple[bool, str]:
  try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
  except ImportError as error:
    return False, f"plotting skipped because matplotlib is unavailable: {error}"

  trajectory = result["trajectory"]
  boundaries = result["boundaries"]
  x = [row["x_m"] for row in trajectory]
  y = [row["y_m"] for row in trajectory]
  s = [row["s_m"] for row in trajectory]
  time = [row["t_s"] for row in trajectory]
  speed = [row["v_ref_m_s"] for row in trajectory]
  segments = [[(x[index], y[index]), (x[index + 1], y[index + 1])]
              for index in range(len(x) - 1)]

  figure, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
  collection = LineCollection(segments, cmap="viridis")
  collection.set_array(speed[:-1])
  collection.set_linewidth(3)
  axes[0, 0].add_collection(collection)
  axes[0, 0].autoscale()
  axes[0, 0].set_aspect("equal", adjustable="box")
  axes[0, 0].plot(x[0], y[0], "go", label="start")
  axes[0, 0].plot(x[-1], y[-1], "rx", label="goal")
  axes[0, 0].set(title="XY path colored by reference speed", xlabel="X [m]", ylabel="Y [m]")
  axes[0, 0].legend()
  figure.colorbar(collection, ax=axes[0, 0], label="v_ref [m/s]")

  axes[0, 1].plot(s, [row["v_limit_m_s"] for row in trajectory], label="v_limit")
  axes[0, 1].plot(s, speed, label="v_ref")
  axes[0, 1].set(title="Speed profile", xlabel="s [m]", ylabel="speed [m/s]")
  axes[0, 1].legend()

  axes[1, 0].plot(s, [row["curvature_ref_1_m"] for row in trajectory], label="analytic")
  axes[1, 0].plot(s, [row["curvature_numeric_1_m"] for row in trajectory], label="numeric", alpha=0.75)
  axes[1, 0].set(title="Curvature", xlabel="s [m]", ylabel="curvature [1/m]")
  axes[1, 0].legend()

  axes[1, 1].plot(time, [row["a_ref_m_s2"] for row in trajectory])
  axes[1, 1].set(title="Longitudinal acceleration", xlabel="time [s]", ylabel="a_ref [m/s²]")
  axes[2, 0].plot(time, [row["omega_ref_rad_s"] for row in trajectory])
  axes[2, 0].set(title="Yaw-rate reference", xlabel="time [s]", ylabel="omega_ref [rad/s]")
  axes[2, 1].plot(time, [row["v_left_ref_m_s"] for row in trajectory], label="left")
  axes[2, 1].plot(time, [row["v_right_ref_m_s"] for row in trajectory], label="right")
  axes[2, 1].set(title="Track surface-speed references", xlabel="time [s]", ylabel="speed [m/s]")
  axes[2, 1].legend()

  for axis in (axes[0, 1], axes[1, 0]):
    for boundary in boundaries[:-1]:
      axis.axvline(boundary["s_end_m"], color="0.7", linewidth=0.7)
    for boundary in boundaries:
      if boundary["segment_type"] == "arc":
        axis.axvspan(boundary["s_start_m"], boundary["s_end_m"], color="orange", alpha=0.08)
  for axis in axes.flat:
    axis.grid(True, alpha=0.25)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, dpi=160)
  plt.close(figure)
  return True, str(output_path)


def print_report(summary: dict[str, Any], paths: dict[str, str], plot_status: str) -> None:
  print(f"path length: {summary['path_length_m']:.6f} m")
  print(f"trajectory time: {summary['total_duration_s']:.6f} s")
  print(f"speed min/max: {summary['minimum_reference_speed_m_s']:.6f} / "
        f"{summary['maximum_reference_speed_m_s']:.6f} m/s")
  print(f"acceleration max/deceleration max: {summary['maximum_positive_acceleration_m_s2']:.6f} / "
        f"{summary['maximum_deceleration_magnitude_m_s2']:.6f} m/s^2")
  print(f"maximum lateral acceleration: {summary['maximum_lateral_acceleration_m_s2']:.6f} m/s^2")
  print(f"maximum absolute yaw rate: {summary['maximum_abs_yaw_rate_rad_s']:.6f} rad/s")
  print(f"maximum absolute left/right track speed: "
        f"{summary['maximum_abs_left_track_speed_m_s']:.6f} / "
        f"{summary['maximum_abs_right_track_speed_m_s']:.6f} m/s")
  print(f"speed-limit reasons: {json.dumps(summary['speed_limit_reason_counts'], sort_keys=True)}")
  for corner in summary["corner_statistics"]:
    print(f"corner segment {corner['segment_index']}: radius={corner['radius_m']:.3f} m, "
          f"speed={corner['minimum_speed_inside_m_s']:.6f}.."
          f"{corner['maximum_speed_inside_m_s']:.6f} m/s")
  for name, path in paths.items():
    print(f"{name}: {path}")
  if "plot" not in paths:
    print(f"plot: {plot_status}")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/reference_trajectory_v1.json")
  parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs/reference_trajectory_v1")
  plot_group = parser.add_mutually_exclusive_group()
  plot_group.add_argument("--plot", dest="plot", action="store_true", help="generate the diagnostic PNG")
  plot_group.add_argument("--no-plot", dest="plot", action="store_false", help="skip plotting")
  parser.set_defaults(plot=False)
  args = parser.parse_args()

  config = load_config(args.config.resolve())
  result = build_reference_trajectory(config)
  output_dir = args.output_dir.resolve()
  paths = write_outputs(result, output_dir)
  if args.plot:
    plotted, plot_status = plot_result(result, output_dir / config["output"]["plot_filename"])
    if plotted:
      paths["plot"] = plot_status
  else:
    plot_status = "disabled (--no-plot or default)"
  print_report(result["summary"], paths, plot_status)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
