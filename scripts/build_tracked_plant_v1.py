#!/usr/bin/env python3
"""Build the BUNKER V1 multi-contact tracked surrogate as a derived USD stage.

Run with Isaac Sim 5.1's ``python.sh``. The cylinders are numerical contact
elements, not a representation of the real BUNKER road-wheel or belt layout.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config/tracked_plant_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs/tracked_plant_v1/tracked_plant_v1.usd"
IMPORTED_ASSET = REPO_ROOT / "assets/bunker_mini_imported/bunker_mini/bunker_mini.usd"


def load_config(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    return json.load(stream)


def roller_geometry(config: dict[str, Any]) -> tuple[float, list[float]]:
  measured = config["measured_fixed"]
  tunable = config["tunable_assumptions"]
  count = int(tunable["roller_count_per_side"])
  if count < 2:
    raise ValueError("roller_count_per_side must be at least 2")
  x_min, x_max = measured["flat_contact_interval_x_m"]
  length = float(measured["flat_contact_length_m"])
  if not math.isclose(x_max - x_min, length, abs_tol=1e-6):
    raise ValueError("flat-contact interval and length disagree")
  if tunable["roller_radius_method"] != "tile_flat_interval":
    raise ValueError("V1 supports only roller_radius_method=tile_flat_interval")
  radius = 0.5 * length / count * float(tunable["roller_radius_scale"])
  spacing = length / count
  centers = [x_min + (index + 0.5) * spacing for index in range(count)]
  return radius, centers


def command_mapping(config: dict[str, Any], v_cmd: float, omega_cmd: float) -> dict[str, float]:
  """Map body command to surface and roller speeds; angular speeds are rad/s."""
  b = float(config["measured_fixed"]["track_center_distance_b_m"])
  radius, _ = roller_geometry(config)
  v_left = v_cmd - 0.5 * b * omega_cmd
  v_right = v_cmd + 0.5 * b * omega_cmd
  return {
    "v_cmd_m_s": v_cmd,
    "omega_cmd_rad_s": omega_cmd,
    "v_left_m_s": v_left,
    "v_right_m_s": v_right,
    "roller_radius_m": radius,
    **surface_speed_mapping(config, v_left, v_right),
  }


def surface_speed_mapping(config: dict[str, Any], v_left: float, v_right: float) -> dict[str, float]:
  """Map requested track surface speeds to signed authored joint targets."""
  radius, _ = roller_geometry(config)
  signs = config["tunable_assumptions"]["drive_sign_by_side"]
  return {
    "v_left_m_s": v_left, "v_right_m_s": v_right,
    "roller_radius_m": radius,
    "omega_left_rad_s": float(signs["left"]) * v_left / radius,
    "omega_right_rad_s": float(signs["right"]) * v_right / radius,
  }


def build_stage(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
  from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

  measured = config["measured_fixed"]
  symmetry = config["derived_symmetry_overrides"]
  tunable = config["tunable_assumptions"]
  radius, x_centers = roller_geometry(config)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  # CreateNew requires a non-existent layer. This is a generated, gitignored
  # inspection artifact at the caller-selected path, so replace it reproducibly.
  if output_path.exists():
    output_path.unlink()
  stage = Usd.Stage.CreateNew(str(output_path))
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
  stage.SetDefaultPrim(world)

  scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
  scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
  scene.CreateGravityMagnitudeAttr(9.81)
  physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
  physx_scene.CreateEnableCCDAttr(True)

  def material(path: str, static: float, dynamic: float, restitution: float):
    prim = UsdShade.Material.Define(stage, path).GetPrim()
    api = UsdPhysics.MaterialAPI.Apply(prim)
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(restitution)
    return UsdShade.Material(prim)

  roller_material = material(
    "/World/Materials/Roller", tunable["roller_static_friction"],
    tunable["roller_dynamic_friction"], tunable["roller_restitution"])
  ground_material = material(
    "/World/Materials/Ground", tunable["ground_static_friction"],
    tunable["ground_dynamic_friction"], tunable["ground_restitution"])

  ground = UsdGeom.Cube.Define(stage, "/World/Ground")
  ground.CreateSizeAttr(1.0)
  ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.05))
  ground.AddScaleOp().Set(Gf.Vec3f(20.0, 20.0, 0.1))
  UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
  UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(
    ground_material, UsdShade.Tokens.weakerThanDescendants, "physics")

  plant = UsdGeom.Xform.Define(stage, "/World/BunkerPlant").GetPrim()
  UsdPhysics.ArticulationRootAPI.Apply(plant)
  PhysxSchema.PhysxArticulationAPI.Apply(plant).CreateEnabledSelfCollisionsAttr(False)
  links = UsdGeom.Scope.Define(stage, "/World/BunkerPlant/Links")
  joints = UsdGeom.Scope.Define(stage, "/World/BunkerPlant/Joints")

  spawn_z = radius - float(measured["stl_bounds_m"]["global_min_z"]) + tunable["spawn_clearance_m"]
  base = UsdGeom.Xform.Define(stage, "/World/BunkerPlant/Links/base_link")
  base.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, spawn_z))
  base_prim = base.GetPrim()
  UsdPhysics.RigidBodyAPI.Apply(base_prim)
  mass_api = UsdPhysics.MassAPI.Apply(base_prim)
  total_roller_mass = 2 * len(x_centers) * tunable["roller_mass_kg_each"]
  base_mass = measured["imported_mass_kg"] - total_roller_mass
  if base_mass <= 0.0:
    raise ValueError("roller masses leave no positive base mass")
  mass_api.CreateMassAttr(base_mass)
  mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*symmetry["base_center_of_mass_m"]))
  mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*measured["imported_diagonal_inertia_kg_m2"]))
  mass_api.CreatePrincipalAxesAttr(Gf.Quatf(*symmetry["base_principal_axes_wxyz"]))
  physx_base = PhysxSchema.PhysxRigidBodyAPI.Apply(base_prim)
  physx_base.CreateSolverPositionIterationCountAttr(tunable["solver_position_iterations"])
  physx_base.CreateSolverVelocityIterationCountAttr(tunable["solver_velocity_iterations"])

  # Reference the complete imported robot namespace so its absolute material
  # relationships remain in scope. Override its physics only in this derived
  # stage; the protected importer output is never edited.
  visual = UsdGeom.Xform.Define(stage, "/World/BunkerPlant/Links/base_link/visual_model")
  visual.GetPrim().GetReferences().AddReference(str(IMPORTED_ASSET), "/bunker_mini")
  imported_base = stage.OverridePrim(
    "/World/BunkerPlant/Links/base_link/visual_model/base_link")
  imported_base.RemoveAPI(UsdPhysics.RigidBodyAPI)
  imported_base.RemoveAPI(UsdPhysics.MassAPI)
  UsdPhysics.RigidBodyAPI(imported_base).CreateRigidBodyEnabledAttr(False)
  imported_collisions_root = stage.OverridePrim(
    "/World/BunkerPlant/Links/base_link/visual_model/base_link/collisions")
  imported_collisions_root.SetInstanceable(False)
  imported_collision = stage.OverridePrim(
    "/World/BunkerPlant/Links/base_link/visual_model/base_link/collisions/base_link/node_STL_BINARY_")
  imported_collision.RemoveAPI(UsdPhysics.CollisionAPI)
  imported_collision.RemoveAPI(UsdPhysics.MeshCollisionAPI)
  UsdPhysics.CollisionAPI(imported_collision).CreateCollisionEnabledAttr(False)

  collider_cfg = tunable["chassis_collider"]
  chassis = UsdGeom.Cube.Define(stage, "/World/BunkerPlant/Links/base_link/chassis_collider")
  chassis.CreateSizeAttr(1.0)
  chassis.AddTranslateOp().Set(Gf.Vec3d(*collider_cfg["center_in_base_frame_m"]))
  chassis.AddScaleOp().Set(Gf.Vec3f(*collider_cfg["size_m"]))
  UsdPhysics.CollisionAPI.Apply(chassis.GetPrim())
  UsdGeom.Imageable(chassis.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

  roller_paths: dict[str, list[str]] = {"left": [], "right": []}
  # Author mirrored pairs consecutively. This avoids a left-first constraint
  # insertion bias in PhysX's highly constrained multi-contact solve.
  for index, x_center in enumerate(x_centers):
    side_order = (("left", measured["left_track_center_y_m"]),
                  ("right", measured["right_track_center_y_m"]))
    for side, center_y in side_order:
      name = f"{side}_roller_{index}"
      roller_path = f"/World/BunkerPlant/Links/{name}"
      roller = UsdGeom.Cylinder.Define(stage, roller_path)
      roller.CreateAxisAttr(UsdGeom.Tokens.y)
      roller.CreateRadiusAttr(radius)
      roller.CreateHeightAttr(tunable["roller_width_m"])
      roller.AddTranslateOp().Set(Gf.Vec3d(x_center, center_y, radius + tunable["spawn_clearance_m"]))
      UsdGeom.Imageable(roller.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
      roller_prim = roller.GetPrim()
      UsdPhysics.CollisionAPI.Apply(roller_prim)
      UsdPhysics.RigidBodyAPI.Apply(roller_prim)
      UsdPhysics.MassAPI.Apply(roller_prim).CreateMassAttr(tunable["roller_mass_kg_each"])
      collision = PhysxSchema.PhysxCollisionAPI.Apply(roller_prim)
      collision.CreateContactOffsetAttr(tunable["contact_offset_m"])
      collision.CreateRestOffsetAttr(tunable["rest_offset_m"])
      UsdShade.MaterialBindingAPI.Apply(roller_prim).Bind(
        roller_material, UsdShade.Tokens.weakerThanDescendants, "physics")

      joint = UsdPhysics.RevoluteJoint.Define(stage, f"/World/BunkerPlant/Joints/{name}_joint")
      joint.CreateAxisAttr(UsdPhysics.Tokens.y)
      joint.CreateBody0Rel().SetTargets([base_prim.GetPath()])
      joint.CreateBody1Rel().SetTargets([roller_prim.GetPath()])
      joint.CreateLocalPos0Attr(Gf.Vec3f(x_center, center_y, radius + tunable["spawn_clearance_m"] - spawn_z))
      joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
      joint.CreateCollisionEnabledAttr(False)
      drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
      drive.CreateTypeAttr("force")
      drive.CreateStiffnessAttr(0.0)
      drive.CreateDampingAttr(tunable["drive_damping_n_m_s_per_rad"])
      drive.CreateMaxForceAttr(tunable["max_drive_force_n_m"])
      drive.CreateTargetVelocityAttr(0.0)
      roller_paths[side].append(roller_path)

  stage.GetRootLayer().Save()
  return {
    "output_usd": str(output_path), "roller_radius_m": radius,
    "roller_x_centers_m": x_centers, "base_mass_kg": base_mass,
    "total_authored_mass_kg": base_mass + total_roller_mass,
    "roller_paths": roller_paths,
    "joint_axis_base_frame": [0.0, 1.0, 0.0],
    "positive_joint_speed_bottom_surface_direction": [-1.0, 0.0, 0.0],
    "positive_joint_speed_vehicle_reaction_direction": [1.0, 0.0, 0.0],
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()
  config = load_config(args.config.resolve())
  result = build_stage(config, args.output.resolve())
  print(json.dumps(result, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
