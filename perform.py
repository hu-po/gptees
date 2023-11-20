#!/usr/bin/env python3
# encoding: utf-8
# import time
import argparse
from ainex_kinematics.motion_manager import MotionManager

argparser = argparse.ArgumentParser()
argparser.add_argument('--action', type=str, required=True, help='action name')
args = argparser.parse_args()

def perform(action_name: str) -> str:
    print(f"Performing action: {action_name}")
    motion_manager = MotionManager('/home/ubuntu/software/ainex_controller/ActionGroups')
    print("MotionManager initialized.")
    assert action_name in [
        'left_shot',
        'right_shot',
        'stand',
        'walk_ready',
        'twist',
        'three',
        'four',
        'hand_back',
        'greet',
    ], f"Unknown action: {action_name}"
    motion_manager.run_action(action_name)
    print(f"Action {action_name} completed.")

if __name__ == '__main__':
    perform(args.action)