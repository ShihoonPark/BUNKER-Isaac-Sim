#!/usr/bin/env python3
"""Run pure kinematic Controller V1 compatibility validation; never launches Isaac Sim."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from map_route_kinematic_validation_v1 import REPO_ROOT, run_validation


def plot_results(scenarios, output_path: Path) -> str:
  try:
    import matplotlib.pyplot as plt
  except ImportError as error:
    return f"skipped; matplotlib unavailable: {error}"
  names = ("forward_lateral", "reverse_lateral_small", "reverse_curve_lateral",
           "transition_lateral", "pivot_xy", "map_nominal", "map_perturbed")
  figure, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
  map_rows = scenarios["map_perturbed"]
  axes[0, 0].plot([row["reference_x_m"] for row in map_rows],
                  [row["reference_y_m"] for row in map_rows], "k--", label="reference")
  axes[0, 0].plot([row["actual_x_m"] for row in map_rows],
                  [row["actual_y_m"] for row in map_rows], label="actual")
  axes[0, 0].set_aspect("equal", adjustable="box"); axes[0, 0].set_title("Map perturbed XY"); axes[0, 0].legend()
  for name in names[:4]:
    rows = scenarios[name]
    axes[0, 1].plot([row["time_s"] for row in rows],
                    [math.hypot(row["e_y_m"], row["heading_error_rad"]) for row in rows], label=name)
  axes[0, 1].set_title("Synthetic lateral/heading error norm"); axes[0, 1].legend(fontsize=8)
  for name, axis in (("reverse_lateral_small", axes[1, 0]), ("reverse_curve_lateral", axes[1, 1]),
                     ("map_nominal", axes[2, 0]), ("map_perturbed", axes[2, 1])):
    rows = scenarios[name]; times = [row["time_s"] for row in rows]
    axis.plot(times, [row["e_y_m"] for row in rows], label="e_y m")
    axis.plot(times, [row["heading_error_rad"] for row in rows], label="heading rad")
    axis.plot(times, [row["command_scale"] for row in rows], label="command scale", alpha=.6)
    axis.set_title(name); axis.legend(fontsize=8)
  for axis in axes.flat:
    axis.grid(True, alpha=.25)
  figure.savefig(output_path, dpi=160); plt.close(figure)
  return str(output_path)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path,
                      default=REPO_ROOT / "config/map_route_kinematic_validation_v1.json")
  parser.add_argument("--output-dir", type=Path,
                      default=REPO_ROOT / "logs/map_route_kinematic_validation_v1")
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--plot", dest="plot", action="store_true")
  group.add_argument("--no-plot", dest="plot", action="store_false")
  parser.set_defaults(plot=False)
  args = parser.parse_args()
  summary, scenarios = run_validation(args.config.resolve(), args.output_dir.resolve())
  print("pure kinematic validation complete (no Isaac Sim)")
  for mode in ("forward", "reverse"):
    stability = summary["local_stability"][mode]
    print(f"{mode} eigenvalues: {stability['eigenvalues_1_s']}")
  for name in ("forward_lateral", "reverse_lateral_small", "reverse_curve_lateral",
               "transition_lateral", "pivot_xy", "map_nominal", "map_perturbed"):
    values = summary["scenarios"][name]
    print(f"{name}: early={values['early_error_norm_rms']:.6f}, "
          f"late={values['late_active_error_norm_rms']:.6f}, "
          f"active_xy_rms={values['active']['rms_time_aligned_xy_error']:.6f}, "
          f"sat={values['active']['saturated_sample_count']}/{values['active']['sample_count']}")
  if args.plot:
    print(f"plot: {plot_results(scenarios, args.output_dir.resolve() / 'map_route_kinematic_validation_v1.png')}")
  print(f"outputs: {args.output_dir.resolve()}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
