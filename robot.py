robot_tools = [
    {
        "type": "function",
        "function": {
            "name": "move_to",
            "description": "Move the robot in a specified direction with a rotation",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "The direction to move in, e.g., forward, backward, left, right",
                    },
                    "rotation": {
                        "type": "number",
                        "description": "The rotation angle in degrees",
                    },
                },
                "required": ["direction", "rotation"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "look_at",
            "description": "Orient the robot's camera to a specified pan and tilt",
            "parameters": {
                "type": "object",
                "properties": {
                    "pan": {
                        "type": "number",
                        "description": "Pan angle in degrees",
                    },
                    "tilt": {
                        "type": "number",
                        "description": "Tilt angle in degrees",
                    },
                },
                "required": ["pan", "tilt"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_image",
            "description": "Capture an image using the robot's camera",
            "parameters": {
                "type": "object",
                "properties": {}
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform",
            "description": "Perform a specified action",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_name": {
                        "type": "string",
                        "description": "The name of the action to perform",
                    },
                },
                "required": ["action_name"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_up",
            "description": "Command the robot to get up",
            "parameters": {
                "type": "object",
                "properties": {}
            },
        }
    },
]

def move_to(direction: str, rotation: float) -> None:
    """
    Move the robot in a specified direction with a rotation.
    
    Args:
        direction (str): The direction to move in, e.g., forward, backward, left, right.
        rotation (float): The rotation angle in degrees.
    """
    pass  # Implementation goes here

def look_at(pan: float, tilt: float) -> None:
    """
    Orient the robot's camera to a specified pan and tilt.
    
    Args:
        pan (float): Pan angle in degrees.
        tilt (float): Tilt angle in degrees.
    """
    pass  # Implementation goes here

def take_image() -> None:
    """
    Capture an image using the robot's camera.
    """
    pass  # Implementation goes here

def perform(action_name: str) -> None:
    """
    Perform a specified action.
    
    Args:
        action_name (str): The name of the action to perform.
    """
    pass  # Implementation goes here

def get_up() -> None:
    """
    Command the robot to get up.
    """
    pass  # Implementation goes here
