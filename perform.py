#!/usr/bin/env python3
# encoding: utf-8
import time
import argparse
from ainex_kinematics.motion_manager import MotionManager

argparser = argparse.ArgumentParser()
argparser.add_argument('--action', type=str, required=True, help='action name')
args = argparser.parse_args()

def perform(action_name: str) -> str:
    print(f"Performing action: {action_name}")
    motion_manager = MotionManager('/home/ubuntu/software/ainex_controller/ActionGroups')
    print("MotionManager initialized.")
    
    motion_manager.set_servos_position(500, [[23, 300]])
    print("Set servos position to 300.")
    time.sleep(0.5) 

    motion_manager.set_servos_position(500, [[23, 500], [24, 500]])
    print("Set servos position to 500.")
    time.sleep(0.5)
    
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