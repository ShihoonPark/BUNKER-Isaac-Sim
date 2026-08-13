#!/usr/bin/env python3
"""Analyze the binary STL geometry of the AgileX BUNKER MINI 2.0.

This tool intentionally uses only the Python standard library so that the
measurements can be reproduced in the Isaac Sim workspace without installing
geometry packages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_STL = Path("assets/ugv_gazebo_sim/bunker/bunker_mini/meshes/base_link.STL")
DEFAULT_SLICE_MM = (5.0, 10.0, 15.0, 20.0, 30.0)


class GeometryError(ValueError):
  """Raised when an STL or requested analysis cannot produce valid geometry."""


def positive_float(value: str) -> float:
  """Return a positive finite float for argparse."""
  number = float(value)
  if not math.isfinite(number) or number <= 0.0:
    raise argparse.ArgumentTypeError("must be a positive finite number")
  return number


def positive_int(value: str) -> int:
  """Return a positive integer for argparse."""
  number = int(value)
  if number <= 0:
    raise argparse.ArgumentTypeError("must be a positive integer")
  return number


def parse_binary_stl(path: Path) -> tuple[bytes, int, list[tuple[float, float, float]]]:
  """Validate *path* as binary STL and return its header, count, and vertices."""
  try:
    file_size = path.stat().st_size
    with path.open("rb") as stl_file:
      header = stl_file.read(80)
      count_bytes = stl_file.read(4)
      if len(header) != 80 or len(count_bytes) != 4:
        raise GeometryError("file is too short to contain an 84-byte binary STL header")

      triangle_count = struct.unpack("<I", count_bytes)[0]
      expected_size = 84 + 50 * triangle_count
      if file_size != expected_size:
        raise GeometryError(
          f"invalid binary STL size: file has {file_size} bytes, but triangle "
          f"count {triangle_count} requires {expected_size} bytes"
        )

      vertices: list[tuple[float, float, float]] = []
      record_struct = struct.Struct("<12fH")
      for triangle_index in range(triangle_count):
        record = stl_file.read(record_struct.size)
        if len(record) != record_struct.size:
          raise GeometryError(f"unexpected EOF in triangle {triangle_index}")
        values = record_struct.unpack(record)
        for offset in (3, 6, 9):
          vertex = (values[offset], values[offset + 1], values[offset + 2])
          if not all(math.isfinite(coordinate) for coordinate in vertex):
            raise GeometryError(f"non-finite vertex coordinate in triangle {triangle_index}")
          vertices.append(vertex)
  except OSError as error:
    raise GeometryError(f"cannot read STL {path}: {error}") from error

  if triangle_count == 0:
    raise GeometryError("binary STL contains no triangles")
  return header, triangle_count, vertices


def bounds(vertices: Sequence[tuple[float, float, float]]) -> dict[str, dict[str, float]]:
  """Calculate axis-aligned bounds for vertices."""
  result: dict[str, dict[str, float]] = {}
  for index, axis in enumerate("xyz"):
    minimum = min(vertex[index] for vertex in vertices)
    maximum = max(vertex[index] for vertex in vertices)
    result[axis] = {"min": minimum, "max": maximum, "size": maximum - minimum}
  return result


def track_metrics(vertices: Sequence[tuple[float, float, float]]) -> dict[str, float]:
  """Return X extent and Y geometry for one track's selected vertices."""
  if not vertices:
    raise GeometryError("no vertices found for a track in this bottom slice")
  x_values = [vertex[0] for vertex in vertices]
  y_values = [vertex[1] for vertex in vertices]
  x_min, x_max = min(x_values), max(x_values)
  y_min, y_max = min(y_values), max(y_values)
  return {
    "x_min": x_min,
    "x_max": x_max,
    "contact_length": x_max - x_min,
    "y_min": y_min,
    "y_max": y_max,
    "track_width": y_max - y_min,
    "track_center_y": (y_min + y_max) / 2.0,
  }


def analyze_slices(
  vertices: Sequence[tuple[float, float, float]],
  global_min_z: float,
  slice_mm: Sequence[float],
  track_y_threshold: float,
) -> list[dict[str, Any]]:
  """Measure left/right track extents at each height above global minimum Z.

  Slice-derived contact length changes with height because a curved/chamfered
  track end reaches progressively farther in X as the slice includes higher
  vertices. It is therefore a geometric section measurement, not contact
  mechanics.

  Track width is each track's Y extent. Center distance B instead spans between
  the two track centerlines, so these quantities must not be conflated.
  """
  results = []
  for height_mm in slice_mm:
    ceiling = global_min_z + height_mm / 1000.0
    bottom_vertices = [vertex for vertex in vertices if vertex[2] <= ceiling]
    left_vertices = [vertex for vertex in bottom_vertices if vertex[1] >= track_y_threshold]
    right_vertices = [vertex for vertex in bottom_vertices if vertex[1] <= -track_y_threshold]
    try:
      left = track_metrics(left_vertices)
      right = track_metrics(right_vertices)
    except GeometryError as error:
      raise GeometryError(f"at {height_mm:g} mm: {error}") from error
    results.append({
      "height_mm": height_mm,
      "left": left,
      "right": right,
      "track_center_distance_b": abs(left["track_center_y"] - right["track_center_y"]),
    })
  return results


def detect_track_geometry(
  vertices: Sequence[tuple[float, float, float]],
  global_min_z: float,
  detection_mm: float,
  track_y_threshold: float,
) -> dict[str, Any]:
  """Detect near-ground track bounds independently of report slice heights.

  Keeping this detection height separate from ``--slice-mm`` makes the bottom
  profile reproducible when callers request a different set of slice reports.
  """
  ceiling = global_min_z + detection_mm / 1000.0
  detection_vertices = [vertex for vertex in vertices if vertex[2] <= ceiling]
  left_vertices = [
    vertex for vertex in detection_vertices if vertex[1] >= track_y_threshold
  ]
  right_vertices = [
    vertex for vertex in detection_vertices if vertex[1] <= -track_y_threshold
  ]
  try:
    left = track_metrics(left_vertices)
    right = track_metrics(right_vertices)
  except GeometryError as error:
    raise GeometryError(f"at track detection height {detection_mm:g} mm: {error}") from error
  return {
    "height_mm": detection_mm,
    "left": left,
    "right": right,
    "track_center_distance_b": abs(left["track_center_y"] - right["track_center_y"]),
  }


def bottom_profile(
  vertices: Sequence[tuple[float, float, float]],
  left_y_min: float,
  left_y_max: float,
  x_min: float,
  x_max: float,
  bins: int,
  global_min_z: float,
) -> list[dict[str, float | None]]:
  """Bin minimum Z in X using the detected left-track Y interval."""
  bin_width = (x_max - x_min) / bins
  minima: list[float | None] = [None] * bins
  for x, y, z in vertices:
    if not left_y_min <= y <= left_y_max or not x_min <= x <= x_max:
      continue
    index = min(int((x - x_min) / bin_width), bins - 1)
    current = minima[index]
    if current is None or z < current:
      minima[index] = z

  profile = []
  for index, minimum_z in enumerate(minima):
    profile.append({
      "x_center": x_min + (index + 0.5) * bin_width,
      "minimum_z": minimum_z,
      "delta_from_global_min_mm": (
        None if minimum_z is None else (minimum_z - global_min_z) * 1000.0
      ),
    })
  return profile


def flat_contact_estimate(
  profile: Sequence[dict[str, float | None]],
  x_min: float,
  x_max: float,
  tolerance_mm: float,
) -> dict[str, float] | None:
  """Find the longest contiguous run of bins within the geometric tolerance.

  This is only a geometric flat-contact estimate from rigid mesh vertices. It
  does not model flexible-track deformation, tread compliance, terrain shape,
  loading, or contact physics and must not be interpreted as physical contact.
  """
  bin_width = (x_max - x_min) / len(profile)
  best_start = best_end = run_start = None
  for index, row in enumerate(profile):
    delta = row["delta_from_global_min_mm"]
    qualifies = delta is not None and delta <= tolerance_mm
    if qualifies and run_start is None:
      run_start = index
    if run_start is not None and (not qualifies or index == len(profile) - 1):
      run_end = index if qualifies else index - 1
      if best_start is None or run_end - run_start > best_end - best_start:
        best_start, best_end = run_start, run_end
      run_start = None
  if best_start is None or best_end is None:
    return None
  estimate_x_min = x_min + best_start * bin_width
  estimate_x_max = x_min + (best_end + 1) * bin_width
  return {
    "x_min": estimate_x_min,
    "x_max": estimate_x_max,
    "length": estimate_x_max - estimate_x_min,
    "tolerance_mm": tolerance_mm,
  }


def write_json(path: Path, result: dict[str, Any]) -> None:
  """Write the analysis summary as JSON, creating parent directories."""
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
      json.dump(result, output, indent=2)
      output.write("\n")
  except OSError as error:
    raise GeometryError(f"cannot write JSON {path}: {error}") from error


def write_profile_csv(path: Path, profile: Iterable[dict[str, float | None]]) -> None:
  """Write bottom-profile rows as CSV, creating parent directories."""
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
      writer = csv.DictWriter(
        output, fieldnames=("x_center", "minimum_z", "delta_from_global_min_mm")
      )
      writer.writeheader()
      writer.writerows(profile)
  except OSError as error:
    raise GeometryError(f"cannot write profile CSV {path}: {error}") from error


def print_report(result: dict[str, Any]) -> None:
  """Print a readable analysis report."""
  print(f"BUNKER geometry analysis: {result['stl']}")
  print(f"Triangle count: {result['triangle_count']:,}")
  print("Bounding box (m):")
  for axis in "xyz":
    values = result["bounds"][axis]
    print(
      f"  {axis.upper()}: min={values['min']:.6f}  max={values['max']:.6f}  "
      f"size={values['size']:.6f}"
    )
  print(f"Global minimum Z: {result['global_min_z']:.6f} m")
  print(f"Track Y-separation threshold: {result['track_y_threshold']:.6f} m")
  print("\nBottom slices:")
  for section in result["slices"]:
    print(f"  {section['height_mm']:g} mm above global minimum Z")
    for side in ("left", "right"):
      track = section[side]
      print(
        f"    {side:5}: x=[{track['x_min']:.6f}, {track['x_max']:.6f}]  "
        f"contact_length={track['contact_length']:.6f} m  "
        f"y=[{track['y_min']:.6f}, {track['y_max']:.6f}]  "
        f"width={track['track_width']:.6f} m  center_y={track['track_center_y']:.6f} m"
      )
    print(f"    track center distance B={section['track_center_distance_b']:.6f} m")

  detection = result["track_detection"]
  left_region = result["profile_left_track_region"]
  print(f"\nProfile track detection: {detection['height_mm']:g} mm above global minimum Z")
  print(
    "  detected left-track region: "
    f"x=[{left_region['x_min']:.6f}, {left_region['x_max']:.6f}] m  "
    f"y=[{left_region['y_min']:.6f}, {left_region['y_max']:.6f}] m"
  )
  print(f"\nLeft-track bottom profile ({result['bins']} X bins):")
  print("  x_center_m     minimum_z_m    delta_from_global_min_mm")
  for row in result["bottom_profile"]:
    if row["minimum_z"] is None:
      print(f"  {row['x_center']: .6f}      no vertices")
    else:
      print(
        f"  {row['x_center']: .6f}      {row['minimum_z']: .6f}"
        f"        {row['delta_from_global_min_mm']: .3f}"
      )
  estimate = result["geometric_flat_contact_estimate"]
  if estimate is None:
    print("\nGeometric flat-contact estimate: no qualifying contiguous region")
  else:
    print(
      "\nGeometric flat-contact estimate: "
      f"x=[{estimate['x_min']:.6f}, {estimate['x_max']:.6f}] m, "
      f"length={estimate['length']:.6f} m "
      f"(tolerance={estimate['tolerance_mm']:g} mm)"
    )
  print("  Geometric mesh estimate only; not flexible-track/terrain physical contact length.")


def argument_parser() -> argparse.ArgumentParser:
  """Create the command-line parser."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stl", type=Path, default=DEFAULT_STL, help="binary STL input path")
  parser.add_argument(
    "--slice-mm", type=positive_float, nargs="+", default=DEFAULT_SLICE_MM,
    metavar="MM", help="bottom slice heights in mm (default: 5 10 15 20 30)",
  )
  parser.add_argument("--bins", type=positive_int, default=40, help="X profile bin count")
  parser.add_argument(
    "--track-y-threshold", type=positive_float, default=0.15, metavar="M",
    help="absolute Y threshold separating track regions in metres (default: 0.15)",
  )
  parser.add_argument(
    "--track-detection-mm", type=positive_float, default=30.0, metavar="MM",
    help=(
      "independent near-ground height used to detect the track bounds for the "
      "bottom profile (default: 30)"
    ),
  )
  parser.add_argument(
    "--flat-tolerance-mm", type=positive_float, default=5.0, metavar="MM",
    help="flat-profile tolerance above global minimum Z (default: 5)",
  )
  parser.add_argument("--json", type=Path, metavar="PATH", help="optional JSON output path")
  parser.add_argument(
    "--profile-csv", type=Path, metavar="PATH", help="optional bottom-profile CSV path"
  )
  return parser


def analyze(args: argparse.Namespace) -> dict[str, Any]:
  """Run the complete STL analysis from parsed CLI arguments."""
  _header, triangle_count, vertices = parse_binary_stl(args.stl)
  full_bounds = bounds(vertices)
  global_min_z = full_bounds["z"]["min"]
  slices = analyze_slices(vertices, global_min_z, args.slice_mm, args.track_y_threshold)

  detection = detect_track_geometry(
    vertices, global_min_z, args.track_detection_mm, args.track_y_threshold
  )
  left = detection["left"]
  profile = bottom_profile(
    vertices, left["y_min"], left["y_max"], left["x_min"], left["x_max"],
    args.bins, global_min_z,
  )
  estimate = flat_contact_estimate(
    profile, left["x_min"], left["x_max"], args.flat_tolerance_mm
  )
  return {
    "stl": str(args.stl),
    "triangle_count": triangle_count,
    "bounds": full_bounds,
    "global_min_z": global_min_z,
    "track_y_threshold": args.track_y_threshold,
    "slice_heights_mm": list(args.slice_mm),
    "slices": slices,
    "track_detection": detection,
    "profile_left_track_region": {
      "x_min": left["x_min"],
      "x_max": left["x_max"],
      "y_min": left["y_min"],
      "y_max": left["y_max"],
    },
    "bins": args.bins,
    "bottom_profile": profile,
    "geometric_flat_contact_estimate": estimate,
  }


def main() -> int:
  """CLI entry point."""
  args = argument_parser().parse_args()
  try:
    result = analyze(args)
    print_report(result)
    if args.json:
      write_json(args.json, result)
    if args.profile_csv:
      write_profile_csv(args.profile_csv, result["bottom_profile"])
  except GeometryError as error:
    print(f"error: {error}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
