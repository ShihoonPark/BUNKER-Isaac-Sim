#!/usr/bin/env python3
"""Build the V2 single-body compound-contact BUNKER plant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config/tracked_force_plant_v2.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs/tracked_force_plant_v2/tracked_force_plant_v2.usd"
IMPORTED_ASSET = REPO_ROOT / "assets/bunker_mini_imported/bunker_mini/bunker_mini.usd"
SUPPORT_MODES = ("flat_track_boxes", "cylinders_legacy")


def load_config(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    return json.load(stream)


def patch_geometry(config: dict[str, Any]) -> tuple[float, list[float]]:
  fixed, tunable = config["measured_fixed"], config["tunable_uncalibrated"]
  count = int(tunable["contact_patch_count_per_side"])
  length = float(fixed["flat_contact_length_m"])
  x_min, x_max = fixed["flat_contact_interval_x_m"]
  if abs((x_max - x_min) - length) > 1e-6:
    raise ValueError("flat contact interval and length disagree")
  if tunable["patch_radius_method"] != "tile_flat_interval":
    raise ValueError("V2 supports only patch_radius_method=tile_flat_interval")
  radius = 0.5 * length / count * float(tunable["patch_radius_scale"])
  spacing = length / count
  return radius, [x_min + (index + 0.5) * spacing for index in range(count)]


def build_stage(config: dict[str, Any], output_path: Path, creation_order: str = "normal",
                support_mode: str | None = None) -> dict[str, Any]:
  from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

  fixed, tunable = config["measured_fixed"], config["tunable_uncalibrated"]
  radius, centers = patch_geometry(config)
  support_mode = support_mode or tunable["support_mode"]
  if support_mode not in SUPPORT_MODES:
    raise ValueError(f"support_mode must be one of {SUPPORT_MODES}")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  if output_path.exists():
    output_path.unlink()
  stage = Usd.Stage.CreateNew(str(output_path))
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
  stage.SetDefaultPrim(world)
  scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
  scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
  scene.CreateGravityMagnitudeAttr(9.81)

  def physics_material(path: str, static: float, dynamic: float):
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(tunable["native_restitution"])
    return material

  patch_material = physics_material("/World/Materials/LowFrictionPatch",
                                    tunable["native_static_friction"], tunable["native_dynamic_friction"])
  ground_material = physics_material("/World/Materials/Ground", 0.5, 0.5)
  ground = UsdGeom.Cube.Define(stage, "/World/Ground")
  ground.CreateSizeAttr(1.0)
  ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
  ground.AddScaleOp().Set(Gf.Vec3f(20, 20, 0.1))
  UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
  UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(
    ground_material, UsdShade.Tokens.weakerThanDescendants, "physics")

  radius_clearance = radius + float(tunable["spawn_clearance_m"])
  spawn_z = radius_clearance - float(fixed["stl_global_min_z_m"])
  base = UsdGeom.Xform.Define(stage, "/World/BunkerForcePlant")
  base.AddTranslateOp().Set(Gf.Vec3d(0, 0, spawn_z))
  base_prim = base.GetPrim()
  UsdPhysics.RigidBodyAPI.Apply(base_prim)
  mass = UsdPhysics.MassAPI.Apply(base_prim)
  mass.CreateMassAttr(fixed["imported_mass_kg"])
  mass.CreateCenterOfMassAttr(Gf.Vec3f(*fixed["imported_center_of_mass_m"]))
  mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*fixed["imported_diagonal_inertia_kg_m2"]))
  mass.CreatePrincipalAxesAttr(Gf.Quatf(*fixed["imported_principal_axes_wxyz"]))
  PhysxSchema.PhysxRigidBodyAPI.Apply(base_prim).CreateDisableGravityAttr(False)

  visual = UsdGeom.Xform.Define(stage, "/World/BunkerForcePlant/visual_model")
  visual.GetPrim().GetReferences().AddReference(str(IMPORTED_ASSET), "/bunker_mini")
  imported_base = stage.OverridePrim("/World/BunkerForcePlant/visual_model/base_link")
  imported_base.RemoveAPI(UsdPhysics.RigidBodyAPI)
  imported_base.RemoveAPI(UsdPhysics.MassAPI)
  UsdPhysics.RigidBodyAPI(imported_base).CreateRigidBodyEnabledAttr(False)
  imported_collision_root = stage.OverridePrim(
    "/World/BunkerForcePlant/visual_model/base_link/collisions")
  imported_collision_root.SetInstanceable(False)
  imported_collision = stage.OverridePrim(
    "/World/BunkerForcePlant/visual_model/base_link/collisions/base_link/node_STL_BINARY_")
  imported_collision.RemoveAPI(UsdPhysics.CollisionAPI)
  imported_collision.RemoveAPI(UsdPhysics.MeshCollisionAPI)
  UsdPhysics.CollisionAPI(imported_collision).CreateCollisionEnabledAttr(False)

  chassis_cfg = tunable["chassis_collider"]
  chassis = UsdGeom.Cube.Define(stage, "/World/BunkerForcePlant/chassis_collider")
  chassis.CreateSizeAttr(1.0)
  chassis.AddTranslateOp().Set(Gf.Vec3d(*chassis_cfg["center_in_base_frame_m"]))
  chassis.AddScaleOp().Set(Gf.Vec3f(*chassis_cfg["size_m"]))
  UsdPhysics.CollisionAPI.Apply(chassis.GetPrim())
  UsdGeom.Imageable(chassis.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

  entries = ([(index, side) for index in range(len(centers)) for side in ("left", "right")]
             if support_mode == "cylinders_legacy" else [(None, side) for side in ("left", "right")])
  if creation_order == "reversed":
    entries.reverse()
  elif creation_order != "normal":
    raise ValueError("creation_order must be normal or reversed")
  patch_paths = []
  support_bottom_z = float(fixed["stl_global_min_z_m"]) - radius
  box_thickness = float(tunable["flat_track_box_thickness_m"])
  if box_thickness <= 0.0:
    raise ValueError("flat_track_box_thickness_m must be positive")
  x_min, x_max = (float(value) for value in fixed["flat_contact_interval_x_m"])
  box_center_x = 0.5 * (x_min + x_max)
  for index, side in entries:
    y = fixed[f"{side}_track_center_y_m"]
    path = (f"/World/BunkerForcePlant/contact_support/{side}_{index}"
            if support_mode == "cylinders_legacy"
            else f"/World/BunkerForcePlant/contact_support/{side}_flat_box")
    if support_mode == "cylinders_legacy":
      patch = UsdGeom.Cylinder.Define(stage, path)
      patch.CreateAxisAttr(UsdGeom.Tokens.y)
      patch.CreateRadiusAttr(radius)
      patch.CreateHeightAttr(tunable["patch_width_m"])
      patch.AddTranslateOp().Set(Gf.Vec3d(centers[index], y, fixed["stl_global_min_z_m"]))
    else:
      patch = UsdGeom.Cube.Define(stage, path)
      patch.CreateSizeAttr(1.0)
      patch.AddTranslateOp().Set(Gf.Vec3d(
        box_center_x, y, support_bottom_z + 0.5 * box_thickness))
      patch.AddScaleOp().Set(Gf.Vec3f(
        float(fixed["flat_contact_length_m"]), float(fixed["track_width_m"]), box_thickness))
    UsdPhysics.CollisionAPI.Apply(patch.GetPrim())
    collision = PhysxSchema.PhysxCollisionAPI.Apply(patch.GetPrim())
    collision.CreateContactOffsetAttr(tunable["contact_offset_m"])
    collision.CreateRestOffsetAttr(tunable["rest_offset_m"])
    UsdShade.MaterialBindingAPI.Apply(patch.GetPrim()).Bind(
      patch_material, UsdShade.Tokens.weakerThanDescendants, "physics")
    UsdGeom.Imageable(patch.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    patch_paths.append(path)

  stage.GetRootLayer().Save()
  return {"output_usd": str(output_path), "creation_order": creation_order,
          "support_mode": support_mode, "support_bottom_z_in_base_frame_m": support_bottom_z,
          "flat_box_size_m": ([float(fixed["flat_contact_length_m"]),
                               float(fixed["track_width_m"]), box_thickness]
                              if support_mode == "flat_track_boxes" else None),
          "flat_box_center_m": ([box_center_x, 0.0, support_bottom_z + 0.5 * box_thickness]
                                if support_mode == "flat_track_boxes" else None),
          "patch_radius_m": radius, "patch_x_centers_m": centers,
          "patch_paths": patch_paths, "mass_kg": fixed["imported_mass_kg"]}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--creation-order", choices=("normal", "reversed"), default="normal")
  parser.add_argument("--support-mode", choices=SUPPORT_MODES)
  args = parser.parse_args()
  print(json.dumps(build_stage(
    load_config(args.config), args.output, args.creation_order, args.support_mode), indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
