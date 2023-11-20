#!/usr/bin/env python3
# encoding: utf-8
# import time
import argparse
import rospy
from ainex_kinematics.gait_manager import GaitManager

argparser = argparse.ArgumentParser()
argparser.add_argument('--move', type=str, required=True, help='move name')
args = argparser.parse_args()

def move(move_name: str) -> str:
    gait_manager = GaitManager()
    rospy.sleep(0.2)
    print("GaitManager initialized.")
    
# param step_velocity: Speed selection has three levels: 1, 2, 3, and 4, with speed decreasing from fast to slow.
# param x_amplitude: Step stride in the x direction (meters).
# param y_amplitude: Step stride in the y direction (meters).
# param rotation_angle: Rotation extent (degrees).
# param arm_swap: Arm swing extent (degrees), default is 30. When it is 0, no commands will be sent to the arms.
# step_velocity: int = 1, # 1, 2, 3, 4
# x_amplitude: float = 0.0, # 0.01, -0.01, 0.02, -0.02
# y_amplitude: float = 0.9, # 0.01, -0.01

    if "fast" in move_name:
        speed = 3
        arm_swing_degree = 30
    else:
        speed = 1
        arm_swing_degree = 0
    print(f"Speed set to {speed} in range [1, 4]")
    print(f"Arm swing degree set to {arm_swing_degree} in range [0, 30]")

    if "forward" in move_name:
        x_amplitude = 0.01
        y_amplitude = 0.0
        rotation_angle = 0.0
    elif "backward" in move_name:
        x_amplitude = -0.01
        y_amplitude = 0.0
        rotation_angle = 0.0
    elif "left" in move_name:
        x_amplitude = 0.0
        y_amplitude = 0.01
        rotation_angle = 0.0
    elif "right" in move_name:
        x_amplitude = 0.0
        y_amplitude = -0.01
        rotation_angle = 0.0
    elif "rotate left" in move_name:
        x_amplitude = 0.0
        y_amplitude = 0.0
        rotation_angle = 5
    elif "rotate right" in move_name:
        x_amplitude = 0.0
        y_amplitude = 0.0
        rotation_angle = -5
    else:
        raise ValueError(f"Unknown move name: {move_name}")
    print(f"x_amplitude set to {x_amplitude}")
    print(f"y_amplitude set to {y_amplitude}")
    print(f"rotation_angle set to {rotation_angle}")

    num_steps = 3
    print(f"num_steps set to {num_steps}")
    
    print(f"Moving {move_name}...")
    gait_manager.move(speed, x_amplitude, y_amplitude, rotation_angle, arm_swing_degree, step_num=num_steps)  # 控制行走步数
    print(f"Movement completed.")
    rospy.sleep(0.2)
    print(f"Stopping GaitManager...")
    gait_manager.stop()
    print(f"GaitManager stopped.")

if __name__ == '__main__':
    move(args.move)