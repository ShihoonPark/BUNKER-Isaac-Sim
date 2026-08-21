#!/usr/bin/env python3
"""ROS 2 PoseStamped subscriber for Real Tracking V1 shadow diagnostics only."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from real_global_path_tracking_v1 import (
  CSV_COLUMNS, REPO_ROOT, ROW_SOURCE_POSE_UPDATE, ROW_SOURCE_STATUS_TIMER,
  LocalizationSample, ShadowSession, format_shadow_status,
)


def _stamp_seconds(stamp: Any) -> float:
  return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path,
                      default=REPO_ROOT / "config/real_global_path_tracking_v1.json")
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument("--duration-s", type=float, default=0.0,
                      help="stop after this wall duration; zero runs until Ctrl+C")
  parser.add_argument("--status-period-s", type=float, default=1.0)
  args = parser.parse_args()
  if args.duration_s < 0.0 or args.status_period_s <= 0.0:
    parser.error("duration must be non-negative and status period positive")

  try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
  except ImportError as error:
    raise RuntimeError("ROS 2 Humble rclpy and geometry_msgs are required only for live shadow") from error

  session = ShadowSession(args.config.resolve())
  config = session.context["config"]
  output_dir = (args.output_dir or
                (REPO_ROOT / config["output"]["default_directory"] / "live_shadow")).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  csv_path = output_dir / config["output"]["live_shadow_csv_filename"]
  stream = csv_path.open("w", newline="", encoding="utf-8")
  writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="raise")
  writer.writeheader()

  rclpy.init()
  node = Node("real_global_path_tracking_shadow_v1")
  last_sample: list[LocalizationSample | None] = [None]
  start_wall = time.monotonic()

  def ros_now_s() -> float:
    return 1e-9 * float(node.get_clock().now().nanoseconds)

  def save(result: dict[str, Any]) -> None:
    writer.writerows(result["rows"])
    stream.flush()

  def pose_callback(message: Any) -> None:
    pose = message.pose
    sample = LocalizationSample(
      _stamp_seconds(message.header.stamp), message.header.frame_id,
      float(pose.position.x), float(pose.position.y), float(pose.position.z),
      float(pose.orientation.x), float(pose.orientation.y),
      float(pose.orientation.z), float(pose.orientation.w))
    last_sample[0] = sample
    save(session.evaluate(sample, ros_now_s(), update_history=True,
                          row_source=ROW_SOURCE_POSE_UPDATE))

  node.create_subscription(PoseStamped, config["runtime"]["pose_topic"],
                           pose_callback, qos_profile_sensor_data)

  def status_callback() -> None:
    # Timer rows remain in the safety/watchdog log but must not be weighted as
    # independent localization samples by future tracking-performance metrics.
    result = session.evaluate(last_sample[0], ros_now_s(), update_history=False,
                              row_source=ROW_SOURCE_STATUS_TIMER)
    save(result)
    print(format_shadow_status(result), flush=True)
    if args.duration_s > 0.0 and time.monotonic() - start_wall >= args.duration_s:
      rclpy.shutdown()

  node.create_timer(args.status_period_s, status_callback)
  print(format_shadow_status(session.evaluate(
    None, ros_now_s(), update_history=False,
    row_source=ROW_SOURCE_STATUS_TIMER)), flush=True)
  print(f"CSV diagnostics: {csv_path}", flush=True)
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    stream.close()
    node.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
