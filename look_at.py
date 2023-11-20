#!/usr/bin/env python3
# encoding: utf-8
# import time
import argparse
from ainex_kinematics.motion_manager import MotionManager

argparser = argparse.ArgumentParser()
argparser.add_argument("--command", type=str, required=True, help="command name")
args = argparser.parse_args()


def look_at(command: str) -> str:
    print(f"Look_at {command}")
    motion_manager = MotionManager(
        "/home/ubuntu/software/ainex_controller/ActionGroups"
    )
    print("MotionManager initialized.")
    assert command in [
        "forward",
        "left",
        "right",
        "up",
        "down",
    ], f"Unknown look_at command: {command}"
    if command == "forward":
        motion_manager.set_servos_position(500, [[23, 500], [24, 500]])
    if command == "left":
        motion_manager.set_servos_position(500, [[23, 500], [24, 1000]])
    if command == "right":
        motion_manager.set_servos_position(500, [[23, 1000], [24, 500]])
    if command == "up":
        motion_manager.set_servos_position(500, [[23, 500], [24, 500]])
    if command == "down":
        motion_manager.set_servos_position(500, [[23, 500], [24, 500]])
    print(f"Look_at {command} completed.")


if __name__ == "__main__":
    look_at(args.command)
