#!/usr/bin/env python3
"""Run finite headless physics tests for the BUNKER V1 tracked surrogate.

This script must be launched by Isaac Sim 5.1's ``python.sh``. It never writes
body poses during a test: all motion comes from revolute-joint velocity drives
and PhysX contact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_tracked_plant_v1 import (  # noqa: E402
  build_stage, command_mapping, load_config, surface_speed_mapping,
)


def yaw_from_wxyz(quaternion: Any) -> float:
  w, x, y, z = (float(value) for value in quaternion)
  return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def set_drive_targets(stage: Any, mapping: dict[str, float], count: int) -> dict[str, float]:
  from pxr import UsdPhysics

  # USD angular-drive targetVelocity is degrees/s; the plant interface and logs
  # remain SI rad/s, so conversion is explicit here.
  for side in ("left", "right"):
    target_deg_s = math.degrees(mapping[f"omega_{side}_rad_s"])
    for index in range(count):
      prim = stage.GetPrimAtPath(f"/World/BunkerPlant/Joints/{side}_roller_{index}_joint")
      UsdPhysics.DriveAPI.Get(prim, "angular").GetTargetVelocityAttr().Set(target_deg_s)
  return {
    "authored_left_target_deg_s": math.degrees(mapping["omega_left_rad_s"]),
    "authored_right_target_deg_s": math.degrees(mapping["omega_right_rad_s"]),
  }


def case_mapping(config: dict[str, Any], command: dict[str, float]) -> dict[str, float]:
  if "v_cmd_m_s" in command:
    return command_mapping(config, command["v_cmd_m_s"], command["omega_cmd_rad_s"])
  return surface_speed_mapping(config, command["v_left_m_s"], command["v_right_m_s"])


def mean_joint_velocities(articulation: Any) -> dict[str, float]:
  names = list(articulation.dof_names)
  velocities = articulation.get_joint_velocities()[0]
  result = {}
  for side in ("left", "right"):
    values = [float(velocities[index]) for index, name in enumerate(names)
              if name.startswith(f"{side}_roller_")]
    result[f"actual_{side}_joint_mean_rad_s"] = sum(values) / len(values)
    result[f"actual_{side}_joint_min_rad_s"] = min(values)
    result[f"actual_{side}_joint_max_rad_s"] = max(values)
  return result


def symmetry_audit(stage: Any, config: dict[str, Any]) -> dict[str, Any]:
  from pxr import UsdGeom, UsdPhysics, UsdShade

  count = int(config["tunable_assumptions"]["roller_count_per_side"])
  mismatches = []
  pairs = []
  for index in range(count):
    side_data = {}
    for side in ("left", "right"):
      roller_path = f"/World/BunkerPlant/Links/{side}_roller_{index}"
      joint_path = f"/World/BunkerPlant/Joints/{side}_roller_{index}_joint"
      roller = stage.GetPrimAtPath(roller_path)
      joint_prim = stage.GetPrimAtPath(joint_path)
      transform = UsdGeom.Xformable(roller).ComputeLocalToWorldTransform(0.0)
      drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
      side_data[side] = {
        "position_m": list(transform.ExtractTranslation()),
        "radius_m": float(UsdGeom.Cylinder(roller).GetRadiusAttr().Get()),
        "width_m": float(UsdGeom.Cylinder(roller).GetHeightAttr().Get()),
        "mass_kg": float(UsdPhysics.MassAPI(roller).GetMassAttr().Get()),
        "joint_axis": str(UsdPhysics.RevoluteJoint(joint_prim).GetAxisAttr().Get()),
        "joint_local_pos_base_m": list(UsdPhysics.RevoluteJoint(joint_prim).GetLocalPos0Attr().Get()),
        "material_target": str(
          UsdShade.MaterialBindingAPI(roller).GetDirectBinding("physics").GetMaterialPath()),
        "drive_damping": float(drive.GetDampingAttr().Get()),
        "drive_max_force": float(drive.GetMaxForceAttr().Get()),
        "drive_stiffness": float(drive.GetStiffnessAttr().Get()),
        "drive_type": str(drive.GetTypeAttr().Get()),
      }
    left, right = side_data["left"], side_data["right"]
    comparisons = {
      "x": (left["position_m"][0], right["position_m"][0]),
      "abs_y": (abs(left["position_m"][1]), abs(right["position_m"][1])),
      "z": (left["position_m"][2], right["position_m"][2]),
      "radius": (left["radius_m"], right["radius_m"]),
      "width": (left["width_m"], right["width_m"]),
      "mass": (left["mass_kg"], right["mass_kg"]),
      "axis": (left["joint_axis"], right["joint_axis"]),
      "local_x": (left["joint_local_pos_base_m"][0], right["joint_local_pos_base_m"][0]),
      "abs_local_y": (abs(left["joint_local_pos_base_m"][1]), abs(right["joint_local_pos_base_m"][1])),
      "material": (left["material_target"], right["material_target"]),
      "damping": (left["drive_damping"], right["drive_damping"]),
      "max_force": (left["drive_max_force"], right["drive_max_force"]),
      "stiffness": (left["drive_stiffness"], right["drive_stiffness"]),
      "drive_type": (left["drive_type"], right["drive_type"]),
    }
    for field, values in comparisons.items():
      if isinstance(values[0], float):
        differs = not math.isclose(values[0], values[1], abs_tol=1e-6)
      else:
        differs = values[0] != values[1]
      if differs:
        mismatches.append({"roller_index": index, "field": field, "left_right": values})
    pairs.append(side_data)
  return {"mismatches": mismatches, "pairs": pairs,
          "world_joint_axis_both_sides": [0.0, 1.0, 0.0]}


def classify_result(name: str, dx: float, dy: float, yaw: float, finite: bool) -> tuple[bool, str]:
  if not finite:
    return False, "non-finite pose/velocity (simulation explosion)"
  if name in ("straight", "both_positive"):
    passed = dx > 0.05 and abs(dy) < max(0.10, abs(dx)) and abs(yaw) < 0.5
    return passed, "requires forward-dominant finite motion"
  expected_sign = {
    "left_only": -1.0, "right_only": 1.0, "opposite": 1.0,
    "opposite_reverse": -1.0, "positive_in_place": 1.0,
    "negative_in_place": -1.0, "positive_curve": 1.0,
  }.get(name)
  passed = expected_sign is not None and yaw * expected_sign > 0.001
  return passed, "requires finite yaw with the expected coordinate-convention sign"


def run_case(world: Any, articulation: Any, stage: Any, config: dict[str, Any],
             name: str, command: dict[str, float], output_dir: Path) -> dict[str, Any]:
  tunable = config["tunable_assumptions"]
  dt = float(tunable["physics_dt_s"])
  settle_steps = round(float(tunable["settle_duration_s"]) / dt)
  test_steps = round(float(tunable["test_duration_s"]) / dt)
  count = int(tunable["roller_count_per_side"])
  mapping = case_mapping(config, command)

  set_drive_targets(stage, command_mapping(config, 0.0, 0.0), count)
  for _ in range(settle_steps):
    world.step(render=False)
  start_position, start_orientation = articulation.get_world_poses()
  start = [float(value) for value in start_position[0]]
  start_yaw = yaw_from_wxyz(start_orientation[0])
  authored_targets = set_drive_targets(stage, mapping, count)

  rows = []
  sample_stride = max(1, round(0.1 / dt))
  for step in range(test_steps):
    world.step(render=False)
    if step % sample_stride == 0 or step == test_steps - 1:
      position, orientation = articulation.get_world_poses()
      linear = articulation.get_linear_velocities()[0]
      angular = articulation.get_angular_velocities()[0]
      actual_joints = mean_joint_velocities(articulation)
      rows.append({
        "time_s": (step + 1) * dt,
        **mapping,
        **authored_targets, **actual_joints,
        "x_m": float(position[0][0]), "y_m": float(position[0][1]),
        "z_m": float(position[0][2]), "yaw_rad": yaw_from_wxyz(orientation[0]),
        "vx_m_s": float(linear[0]), "vy_m_s": float(linear[1]),
        "vz_m_s": float(linear[2]), "yaw_rate_rad_s": float(angular[2]),
      })

  set_drive_targets(stage, command_mapping(config, 0.0, 0.0), count)
  final = rows[-1]
  dx, dy = final["x_m"] - start[0], final["y_m"] - start[1]
  yaw_delta = math.atan2(math.sin(final["yaw_rad"] - start_yaw),
                         math.cos(final["yaw_rad"] - start_yaw))
  finite = all(math.isfinite(float(value)) for row in rows for value in row.values())
  passed, criterion = classify_result(name, dx, dy, yaw_delta, finite)
  csv_path = output_dir / f"{name}.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
  return {
    "name": name, "passed": passed, "criterion": criterion,
    "command_mapping": mapping, "start_position_m": start,
    "authored_drive_targets": authored_targets,
    "final_actual_joint_velocities": {
      key: final[key] for key in final if key.startswith("actual_")},
    "delta_x_m": dx, "delta_y_m": dy, "delta_yaw_rad": yaw_delta,
    "final_linear_velocity_m_s": [final["vx_m_s"], final["vy_m_s"], final["vz_m_s"]],
    "final_yaw_rate_rad_s": final["yaw_rate_rad_s"], "log_csv": str(csv_path),
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/tracked_plant_v1.json")
  parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs/tracked_plant_v1")
  diagnostic_names = (
    "straight", "both_positive", "left_only", "right_only", "opposite",
    "opposite_reverse", "positive_in_place", "negative_in_place", "positive_curve")
  parser.add_argument("--case", choices=(*diagnostic_names, "all"), default="all")
  args = parser.parse_args()
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  from isaacsim import SimulationApp
  simulation_app = SimulationApp({"headless": True, "enable_cameras": False})
  try:
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation

    config = load_config(args.config.resolve())
    usd_path = output_dir / "tracked_plant_v1.usd"
    build_info = build_stage(config, usd_path)
    omni.usd.get_context().open_stage(str(usd_path))
    world = World(physics_dt=config["tunable_assumptions"]["physics_dt_s"],
                  rendering_dt=config["tunable_assumptions"]["physics_dt_s"],
                  stage_units_in_meters=1.0)
    articulation = world.scene.add(Articulation("/World/BunkerPlant", name="bunker_plant"))
    world.reset()
    stage = omni.usd.get_context().get_stage()
    audit = symmetry_audit(stage, config)
    names = diagnostic_names if args.case == "all" else (args.case,)
    results = []
    for index, name in enumerate(names):
      if index:
        # Reset restores the authored known pose; no pose is written during drive.
        world.reset()
      results.append(run_case(world, articulation, stage, config, name, config["tests"][name], output_dir))
      print(json.dumps(results[-1], indent=2))
    summary = {"build": build_info, "symmetry_audit": audit, "tests": results,
               "all_passed": all(r["passed"] for r in results)}
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
      json.dump(summary, stream, indent=2)
      stream.write("\n")
    print(f"summary: {summary_path}")
    return 0 if summary["all_passed"] else 2
  except BaseException:
    traceback.print_exc()
    raise
  finally:
    simulation_app.close()


if __name__ == "__main__":
  raise SystemExit(main())
