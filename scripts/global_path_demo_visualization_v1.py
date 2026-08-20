#!/usr/bin/env python3
"""Observer-only Isaac visualization for Multiple Closed-Loop Global Paths V1."""

from __future__ import annotations

import math
from typing import Any

from global_path_demo_v1 import map_to_local
from map_route_visualization_v1 import ActualTrailSampler


class GlobalPathDemoObserver:
  """Author display prims and lap UI without writing robot, controller, plant, or reference state."""

  ROOT_PATH = "/World/GlobalPathDemo"

  def __init__(self, config: dict[str, Any], result: dict[str, Any], ply: dict[str, Any],
               total_laps: int):
    self.config = config; self.result = result; self.ply = ply; self.total_laps = total_laps
    display = config["visualization"]
    self.sampler = ActualTrailSampler(float(display["actual_trail_update_period_s"]),
                                      float(display["actual_trail_minimum_spacing_m"]),
                                      float(display["actual_visual_z_m"]))
    self.prim_paths: list[str] = []
    self.current_lap = 1; self.completed_laps = 0
    self.status_window = None; self.status_label = None

  @staticmethod
  def _curve(stage: Any, path: str, points: list[tuple[float, float, float]], width: float,
             color: tuple[float, float, float]) -> Any:
    from pxr import Gf, UsdGeom, Vt
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr("linear"); curve.CreateBasisAttr("bezier"); curve.CreateWrapAttr("nonperiodic")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in points]))
    curve.CreateWidthsAttr([width]); curve.SetWidthsInterpolation("constant")
    curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return curve

  def _status_text(self, complete: bool = False) -> str:
    if complete:
      return f"{self.result['summary']['preset']}  |  complete {self.total_laps}/{self.total_laps} laps"
    return (f"{self.result['summary']['preset']}  |  current lap {self.current_lap}/{self.total_laps}"
            f"  |  complete {self.completed_laps}/{self.total_laps}")

  def _create_status_ui(self) -> None:
    try:
      import omni.ui as ui
      self.status_window = ui.Window("Global Path Demo Status", width=430, height=80)
      with self.status_window.frame:
        self.status_label = ui.Label(self._status_text(), style={"font_size": 20})
    except (ImportError, AttributeError, RuntimeError) as error:
      print(f"Lap status UI unavailable; console/prim status remains active: {error}")

  def _update_status(self, stage: Any, complete: bool = False) -> None:
    from pxr import Sdf, UsdGeom
    status = UsdGeom.Xform.Define(stage, self.ROOT_PATH + "/LapStatus").GetPrim()
    status.CreateAttribute("currentLap", Sdf.ValueTypeNames.Int).Set(self.current_lap)
    status.CreateAttribute("completedLaps", Sdf.ValueTypeNames.Int).Set(self.completed_laps)
    status.CreateAttribute("totalLaps", Sdf.ValueTypeNames.Int).Set(self.total_laps)
    status.CreateAttribute("statusText", Sdf.ValueTypeNames.String).Set(self._status_text(complete))
    if self.status_label is not None:
      try: self.status_label.text = self._status_text(complete)
      except (AttributeError, RuntimeError): pass

  def _author_static(self, world: Any, settled: dict[str, float]) -> None:
    from pxr import Gf, UsdGeom, Vt
    stage = world.stage; display = self.config["visualization"]
    root = UsdGeom.Xform.Define(stage, self.ROOT_PATH)
    root.AddTranslateOp().Set(Gf.Vec3d(settled["x0_world_m"], settled["y0_world_m"], 0.0))
    root.AddRotateZOp().Set(math.degrees(settled["yaw0_world_rad"]))
    self.prim_paths.append(self.ROOT_PATH)
    transform = self.result["summary"]["map_to_local_transform"]
    map_points = [(*map_to_local(x, y, transform), z) for x, y, z in self.ply["points"]]
    point_path = self.ROOT_PATH + "/BagCPointCloud"
    points = UsdGeom.Points.Define(stage, point_path)
    points.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in map_points]))
    points.CreateWidthsAttr([float(display["point_size_m"])])
    points.SetWidthsInterpolation("constant")
    points.CreateDisplayColorAttr([Gf.Vec3f(.52, .55, .60)])
    self.prim_paths.append(point_path)
    global_points = [(x, y, float(display["reference_visual_z_m"]) * .65)
                     for x, y in self.result["global_path_local"]]
    global_path = self.ROOT_PATH + "/ManualGlobalPath"
    self._curve(stage, global_path, global_points, float(display["global_path_width_m"]), (.15, .75, 1.0))
    self.prim_paths.append(global_path)
    reference_points = [(float(row["x_m"]), float(row["y_m"]), float(display["reference_visual_z_m"]))
                        for row in self.result["periodic_lap"]]
    reference_path = self.ROOT_PATH + "/VelocityAwareReference"
    self._curve(stage, reference_path, reference_points, float(display["reference_width_m"]), (.15, .95, .25))
    self.prim_paths.append(reference_path)
    marker_path = self.ROOT_PATH + "/StartMarker"
    marker = UsdGeom.Sphere.Define(stage, marker_path)
    marker.CreateRadiusAttr(float(display["start_marker_radius_m"]))
    marker.AddTranslateOp().Set(Gf.Vec3d(reference_points[0][0], reference_points[0][1],
                                         float(display["reference_visual_z_m"])))
    marker.CreateDisplayColorAttr([Gf.Vec3f(.1, 1.0, .1)])
    self.prim_paths.append(marker_path)
    self._author_camera(stage, settled, map_points, reference_points)
    self._create_status_ui(); self._update_status(stage)

  def _author_camera(self, stage: Any, settled: dict[str, float],
                     map_points: list[tuple[float, float, float]],
                     reference_points: list[tuple[float, float, float]]) -> None:
    from pxr import Gf, UsdGeom
    yaw = settled["yaw0_world_rad"]; cosine, sine = math.cos(yaw), math.sin(yaw)
    local = [(point[0], point[1]) for point in map_points] + [(point[0], point[1]) for point in reference_points]
    world_xy = [(settled["x0_world_m"] + cosine*x - sine*y,
                 settled["y0_world_m"] + sine*x + cosine*y) for x, y in local]
    x0, x1 = min(x for x, _ in world_xy), max(x for x, _ in world_xy)
    y0, y1 = min(y for _, y in world_xy), max(y for _, y in world_xy)
    span = max(x1 - x0, y1 - y0); center = Gf.Vec3d(.5*(x0+x1), .5*(y0+y1), 0.0)
    eye = Gf.Vec3d(center[0], center[1], max(5.0, float(self.config["visualization"]["camera_height_scale"]) * span))
    camera_path = self.ROOT_PATH + "/OverviewCamera"; camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, center, Gf.Vec3d(0, 1, 0)).GetInverse())
    camera.CreateFocalLengthAttr(24.0); self.prim_paths.append(camera_path)
    try:
      from omni.kit.viewport.utility import get_active_viewport
      viewport = get_active_viewport()
      if viewport is not None: viewport.set_active_camera(camera_path)
    except (ImportError, AttributeError): pass

  def _update_actual_curve(self, world: Any) -> None:
    if len(self.sampler.points) < 2: return
    path = self.ROOT_PATH + "/IsaacActualTrail"
    self._curve(world.stage, path, self.sampler.points,
                float(self.config["visualization"]["actual_trail_width_m"]), (1.0, .90, .05))
    if path not in self.prim_paths: self.prim_paths.append(path)

  def after_settle(self, world: Any, settled: dict[str, float],
                   reference_rows: list[dict[str, Any]]) -> None:
    self._author_static(world, settled)
    print("Visualization legend: Bag C=gray, manual global path=cyan, velocity-aware reference=green, Isaac actual=yellow")
    print(self._status_text())

  def after_logged_row(self, world: Any, row: dict[str, Any]) -> None:
    lap_length = float(self.result["summary"]["geometry"]["lap_length_m"])
    progress = min(self.total_laps * lap_length, max(0.0, float(row["reference_s_m"])))
    completed = min(self.total_laps, int(math.floor(progress / lap_length + 1e-12)))
    current = min(self.total_laps, completed + 1)
    if completed != self.completed_laps or current != self.current_lap:
      self.completed_laps = completed; self.current_lap = current
      self._update_status(world.stage); print(self._status_text())
    if self.sampler.consider(row): self._update_actual_curve(world)

  def after_run(self, world: Any, rows: list[dict[str, Any]], final_observation: dict[str, Any]) -> None:
    final_row = {**final_observation, "physics_time_s": final_observation["physics_time_s"]}
    self.sampler.consider(final_row, force=True); self._update_actual_curve(world)
    self.completed_laps = self.total_laps; self.current_lap = self.total_laps
    self._update_status(world.stage, complete=True); print(self._status_text(complete=True))

  def diagnostics(self) -> dict[str, Any]:
    return {"root_prim": self.ROOT_PATH, "authored_prim_paths": self.prim_paths,
            "point_count": self.ply["vertex_count"],
            "actual_trail_display_point_count": len(self.sampler.points),
            "lap_status": {"current_lap": self.current_lap, "completed_laps": self.completed_laps,
                           "total_laps": self.total_laps},
            "observer_policy": "read current state; author display/UI only; never write robot/controller/plant/reference state",
            "physics_apis_authored": []}
