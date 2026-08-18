#!/usr/bin/env python3
"""Pure data preparation and Isaac USD authoring for Map Route Visualization V1."""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any


PLY_SCALAR_FORMATS = {
  "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
  "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
  "int": "i", "int32": "i", "uint": "I", "uint32": "I",
  "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def load_ply(path: Path) -> dict[str, Any]:
  """Read scalar vertex properties from an ASCII or binary PLY deterministically."""
  with path.open("rb") as stream:
    header_lines = []
    while True:
      line = stream.readline()
      if not line:
        raise ValueError("PLY header has no end_header")
      try:
        decoded = line.decode("ascii").strip()
      except UnicodeDecodeError as error:
        raise ValueError("PLY header is not ASCII") from error
      header_lines.append(decoded)
      if decoded == "end_header":
        break
    if header_lines[0] != "ply":
      raise ValueError("not a PLY file")
    format_line = next((line for line in header_lines if line.startswith("format ")), "")
    ply_format = format_line.split()[1] if format_line else ""
    vertex_count = None
    properties: list[tuple[str, str]] = []
    in_vertices = False
    for line in header_lines:
      fields = line.split()
      if fields[:2] == ["element", "vertex"]:
        vertex_count = int(fields[2]); in_vertices = True
      elif fields and fields[0] == "element":
        in_vertices = False
      elif in_vertices and fields and fields[0] == "property":
        if fields[1] == "list":
          raise ValueError("list-valued vertex PLY properties are unsupported")
        if fields[1] not in PLY_SCALAR_FORMATS:
          raise ValueError(f"unsupported PLY scalar type: {fields[1]}")
        properties.append((fields[1], fields[2]))
    if vertex_count is None or vertex_count <= 0:
      raise ValueError("PLY has no positive vertex count")
    names = [name for _, name in properties]
    if not {"x", "y", "z"} <= set(names):
      raise ValueError("PLY vertices lack x/y/z")
    records: list[dict[str, float]] = []
    if ply_format == "ascii":
      for _ in range(vertex_count):
        values = stream.readline().decode("ascii").split()
        if len(values) != len(properties):
          raise ValueError("malformed ASCII PLY vertex")
        records.append({name: float(value) for (_, name), value in zip(properties, values)})
    elif ply_format in ("binary_little_endian", "binary_big_endian"):
      prefix = "<" if ply_format == "binary_little_endian" else ">"
      record_struct = struct.Struct(prefix + "".join(PLY_SCALAR_FORMATS[kind] for kind, _ in properties))
      for _ in range(vertex_count):
        data = stream.read(record_struct.size)
        if len(data) != record_struct.size:
          raise ValueError("truncated binary PLY vertex data")
        records.append({name: float(value) for (_, name), value in zip(properties, record_struct.unpack(data))})
    else:
      raise ValueError(f"unsupported PLY format: {ply_format}")
  points = [(row["x"], row["y"], row["z"]) for row in records]
  if not all(math.isfinite(value) for point in points for value in point):
    raise ValueError("PLY contains non-finite XYZ")
  color_names = ("red", "green", "blue")
  colors = None
  if set(color_names) <= set(names):
    colors = [tuple(max(0.0, min(1.0, row[name] / 255.0)) for name in color_names) for row in records]
  return {"format": ply_format, "vertex_count": vertex_count, "fields": names,
          "points": points, "colors": colors,
          "bounds_xyz": [[min(point[axis] for point in points), max(point[axis] for point in points)]
                         for axis in range(3)]}


def map_xy_to_reference(x_m: float, y_m: float, transform: dict[str, Any]) -> tuple[float, float]:
  """Apply the exact rigid map-to-local convention emitted by Stage D."""
  dx = x_m - float(transform["map_origin_x_m"])
  dy = y_m - float(transform["map_origin_y_m"])
  yaw = float(transform["map_initial_body_yaw_rad"])
  return math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy


def transform_map_points(points: list[tuple[float, float, float]], transform: dict[str, Any],
                         z_offset_m: float) -> list[tuple[float, float, float]]:
  return [(*map_xy_to_reference(x, y, transform), z + z_offset_m) for x, y, z in points]


class ActualTrailSampler:
  """Display-only time/spacing downsampler; input rows are never mutated."""
  def __init__(self, period_s: float, minimum_spacing_m: float, z_m: float):
    self.period_s = period_s; self.minimum_spacing_m = minimum_spacing_m; self.z_m = z_m
    self.last_time = -math.inf
    self.points: list[tuple[float, float, float]] = []

  def consider(self, row: dict[str, Any], force: bool = False) -> bool:
    point = (float(row["actual_local_x_m"]), float(row["actual_local_y_m"]), self.z_m)
    elapsed = float(row["physics_time_s"]) - self.last_time
    moved = not self.points or math.hypot(point[0] - self.points[-1][0], point[1] - self.points[-1][1]) >= self.minimum_spacing_m
    if force or (elapsed + 1e-12 >= self.period_s and moved):
      self.points.append(point); self.last_time = float(row["physics_time_s"]); return True
    return False


def alignment_diagnostics(raw_rows: list[dict[str, Any]], reference: list[dict[str, Any]],
                          map_points: list[tuple[float, float, float]], transform: dict[str, Any]) -> dict[str, Any]:
  raw_xy = [map_xy_to_reference(float(row["x_m"]), float(row["y_m"]), transform) for row in raw_rows]
  reference_xy = [(float(row["x_m"]), float(row["y_m"])) for row in reference]
  map_xy = [(point[0], point[1]) for point in map_points]
  bounds = lambda values: {"x_m": [min(x for x, _ in values), max(x for x, _ in values)],
                           "y_m": [min(y for _, y in values), max(y for _, y in values)]}
  return {
    "transformed_raw_start_xy_m": list(raw_xy[0]), "processed_reference_start_xy_m": list(reference_xy[0]),
    "start_difference_m": math.dist(raw_xy[0], reference_xy[0]),
    "transformed_raw_endpoint_xy_m": list(raw_xy[-1]), "processed_reference_endpoint_xy_m": list(reference_xy[-1]),
    "endpoint_difference_m": math.dist(raw_xy[-1], reference_xy[-1]),
    "raw_start_end_distance_m": math.dist(raw_xy[0], raw_xy[-1]),
    "reference_start_end_distance_m": math.dist(reference_xy[0], reference_xy[-1]),
    "map_xy_bounds_m": bounds(map_xy), "reference_xy_bounds_m": bounds(reference_xy),
  }


def forward_wobble_diagnostics(rows: list[dict[str, Any]], active_end_s: float | None = None) -> dict[str, Any]:
  rows = [row for row in rows
          if active_end_s is None or float(row["physics_time_s"]) <= active_end_s + 1e-12]
  segments = {}
  for segment_id in sorted({int(row["reference_segment_id"]) for row in rows
                            if int(row["reference_motion_direction"]) == 1}):
    selected = [row for row in rows if int(row["reference_segment_id"]) == segment_id]
    rms = lambda key: math.sqrt(sum(float(row[key]) ** 2 for row in selected) / len(selected))
    maximum = lambda key: max(abs(float(row[key])) for row in selected)
    updates = [row for row in selected if int(row["control_update"])]
    segments[str(segment_id)] = {
      "rms_cte_m": rms("cross_track_error_m"), "maximum_abs_cte_m": maximum("cross_track_error_m"),
      "rms_e_y_m": rms("e_y_m"), "maximum_abs_e_y_m": maximum("e_y_m"),
      "rms_heading_rad": rms("heading_error_rad"), "maximum_abs_heading_rad": maximum("heading_error_rad"),
      "rms_progress_error_m": rms("progress_error_m"), "maximum_abs_progress_error_m": maximum("progress_error_m"),
      "rms_body_vy_m_s": rms("body_vy_m_s"), "maximum_abs_body_vy_m_s": maximum("body_vy_m_s"),
      "saturated_update_fraction": sum(float(row["command_scale"]) < 1.0 - 1e-12 for row in updates) / len(updates),
    }
  return {"segments": segments,
          "three_worst_by_rms_cte": sorted(segments, key=lambda key: segments[key]["rms_cte_m"], reverse=True)[:3],
          "three_worst_by_maximum_cte": sorted(segments, key=lambda key: segments[key]["maximum_abs_cte_m"], reverse=True)[:3]}


class MapRouteVisualizationObserver:
  """Observer-only USD layer; it never writes robot, control, plant, or reference state."""
  ROOT_PATH = "/World/MapRouteVisualization"

  def __init__(self, config: dict[str, Any], reference_result: dict[str, Any], raw_rows: list[dict[str, Any]],
               ply: dict[str, Any]):
    self.config = config; self.reference_result = reference_result; self.raw_rows = raw_rows; self.ply = ply
    display = config["display"]
    self.sampler = ActualTrailSampler(float(display["actual_trail_update_period_s"]),
                                      float(display["actual_trail_minimum_spacing_m"]),
                                      float(display["actual_visual_z_m"]))
    self.settled: dict[str, float] | None = None
    self.prim_paths: list[str] = []

  @staticmethod
  def _curve(stage: Any, path: str, points: list[tuple[float, float, float]], width: float,
             color: tuple[float, float, float]) -> Any:
    from pxr import Gf, UsdGeom, Vt
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr("linear"); curve.CreateBasisAttr("bezier"); curve.CreateWrapAttr("nonperiodic")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in points]))
    curve.CreateWidthsAttr([width]); curve.SetWidthsInterpolation("constant")
    curve.CreateDisplayColorAttr([Gf.Vec3f(*color)]); return curve

  def _author_static(self, world: Any, settled: dict[str, float], reference_rows: list[dict[str, Any]]) -> None:
    from pxr import Gf, UsdGeom, Vt
    stage = world.stage
    root = UsdGeom.Xform.Define(stage, self.ROOT_PATH)
    root.AddTranslateOp().Set(Gf.Vec3d(settled["x0_world_m"], settled["y0_world_m"], 0.0))
    root.AddRotateZOp().Set(math.degrees(settled["yaw0_world_rad"]))
    self.prim_paths.append(self.ROOT_PATH)
    transform = self.reference_result["summary"]["reference"]["maneuver_normalization_transform"]
    layers, display = self.config["layers"], self.config["display"]
    map_points = transform_map_points(self.ply["points"], transform, float(display["map_visual_z_offset_m"]))
    if layers["show_point_cloud"]:
      path = self.ROOT_PATH + "/ClassroomPointCloud"; points = UsdGeom.Points.Define(stage, path)
      points.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in map_points]))
      points.CreateWidthsAttr([float(display["point_size_m"])])
      colors = self.ply["colors"]
      points.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color) for color in colors]) if colors else [Gf.Vec3f(.55, .58, .62)])
      self.prim_paths.append(path)
    raw_points = [(*map_xy_to_reference(float(row["x_m"]), float(row["y_m"]), transform),
                   float(display["reference_visual_z_m"]) * .65) for row in self.raw_rows]
    if layers["show_raw_glim_route"]:
      path = self.ROOT_PATH + "/RawGLIMTraversal"; self._curve(stage, path, raw_points, float(display["raw_route_width_m"]), (.30, .72, .78)); self.prim_paths.append(path)
    if layers["show_reference"]:
      colors = {1: (.15, .85, .20), -1: (1.0, .48, .05), 0: (.90, .10, .85)}
      for segment_id in range(23):
        selected = [row for row in reference_rows if int(row["segment_id"]) == segment_id]
        points = [(float(row["x_m"]), float(row["y_m"]), float(display["reference_visual_z_m"])) for row in selected]
        path = self.ROOT_PATH + f"/Reference/Segment_{segment_id:02d}"
        self._curve(stage, path, points, float(display["reference_width_m"]), colors[int(selected[0]["motion_direction"])]); self.prim_paths.append(path)
    if layers["show_pivot_markers"]:
      for segment_id in range(23):
        selected = [row for row in reference_rows if int(row["segment_id"]) == segment_id and int(row["motion_direction"]) == 0]
        if not selected: continue
        middle = selected[len(selected) // 2]; path = self.ROOT_PATH + f"/PivotMarkers/Pivot_{segment_id:02d}"
        sphere = UsdGeom.Sphere.Define(stage, path); sphere.CreateRadiusAttr(float(display["pivot_marker_radius_m"]))
        sphere.AddTranslateOp().Set(Gf.Vec3d(float(middle["x_m"]), float(middle["y_m"]), float(display["reference_visual_z_m"])))
        sphere.CreateDisplayColorAttr([Gf.Vec3f(.90, .10, .85)]); self.prim_paths.append(path)
    if layers["show_start_end"]:
      for name, row, color in (("Start", reference_rows[0], (.1, 1., .1)), ("End", reference_rows[-1], (1., .1, .1))):
        path = self.ROOT_PATH + "/Markers/" + name; sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr(float(display["endpoint_marker_radius_m"]))
        sphere.AddTranslateOp().Set(Gf.Vec3d(float(row["x_m"]), float(row["y_m"]), float(display["reference_visual_z_m"])))
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)]); self.prim_paths.append(path)
    self._author_camera(stage, settled, map_points, reference_rows)

  def _author_camera(self, stage: Any, settled: dict[str, float], map_points: list[tuple[float, float, float]],
                     reference_rows: list[dict[str, Any]]) -> None:
    if not self.config["camera"]["overview_camera_enabled"]: return
    from pxr import Gf, UsdGeom
    yaw = settled["yaw0_world_rad"]; cosine, sine = math.cos(yaw), math.sin(yaw)
    local = [(point[0], point[1]) for point in map_points] + [(float(row["x_m"]), float(row["y_m"])) for row in reference_rows]
    world_xy = [(settled["x0_world_m"] + cosine*x - sine*y, settled["y0_world_m"] + sine*x + cosine*y) for x, y in local]
    x0, x1 = min(x for x, _ in world_xy), max(x for x, _ in world_xy); y0, y1 = min(y for _, y in world_xy), max(y for _, y in world_xy)
    span = max(x1 - x0, y1 - y0); center = Gf.Vec3d(.5*(x0+x1), .5*(y0+y1), 0.0)
    eye = Gf.Vec3d(center[0], center[1], max(5.0, float(self.config["camera"]["height_scale"]) * span))
    camera_path = self.ROOT_PATH + "/OverviewCamera"; camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, center, Gf.Vec3d(0, 1, 0)).GetInverse())
    camera.CreateFocalLengthAttr(24.0); self.prim_paths.append(camera_path)
    try:
      from omni.kit.viewport.utility import get_active_viewport
      viewport = get_active_viewport()
      if viewport is not None: viewport.set_active_camera(camera_path)
    except (ImportError, AttributeError):
      pass

  def _update_actual_curve(self, world: Any) -> None:
    if not self.config["layers"]["show_actual_trail"] or len(self.sampler.points) < 2: return
    self._curve(world.stage, self.ROOT_PATH + "/ActualTrajectory", self.sampler.points,
                float(self.config["display"]["actual_trail_width_m"]), (1.0, .92, .05))
    if self.ROOT_PATH + "/ActualTrajectory" not in self.prim_paths: self.prim_paths.append(self.ROOT_PATH + "/ActualTrajectory")

  def after_settle(self, world: Any, settled: dict[str, float], reference_rows: list[dict[str, Any]]) -> None:
    self.settled = settled; self._author_static(world, settled, reference_rows)
    print("Visualization legend: point cloud=gray, raw GLIM=cyan, forward=green, reverse=orange, pivot=magenta, actual=yellow")

  def after_logged_row(self, world: Any, row: dict[str, Any]) -> None:
    if self.sampler.consider(row): self._update_actual_curve(world)

  def after_run(self, world: Any, rows: list[dict[str, Any]], final_observation: dict[str, Any]) -> None:
    final_row = {**final_observation, "physics_time_s": final_observation["physics_time_s"]}
    self.sampler.consider(final_row, force=True); self._update_actual_curve(world)

  def diagnostics(self) -> dict[str, Any]:
    return {"root_prim": self.ROOT_PATH, "authored_prim_paths": self.prim_paths,
            "point_count": self.ply["vertex_count"], "actual_trail_display_point_count": len(self.sampler.points),
            "actual_trail_sampling": {"period_s": self.sampler.period_s, "minimum_spacing_m": self.sampler.minimum_spacing_m},
            "observer_policy": "read current state; author display prims only; never write robot/controller/plant/reference state",
            "physics_apis_authored": []}
