from openai import OpenAI
import json

client = OpenAI()

SYSTEM_PROMPT: str = ". ".join(
    [
        "You are the master node in a robot control system",
        "The user is the robot vision module",
        "As the master node, you decide what tools to use",
        "The robot's goals are to explore and understand the environment",
        "If a human is visible, perform the wave action",
        "If the robot is looking at the ceiling, use the get_up tool",
        "When in doubt, move around",
    ]
)
MODEL: str = "gpt-4-1106-preview"
MAX_TOKENS: int = 32
TEMPERATURE: float = 0.0
TOOLS = [
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
        },
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
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_image",
            "description": "Capture an image using the robot's camera",
            "parameters": {"type": "object", "properties": {}},
        },
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
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_up",
            "description": "Command the robot to get up",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
FUNCTIONS = [tool["function"] for tool in TOOLS]


def move_to(direction: str, rotation: float) -> None:
    print(f"Moving to {direction} with rotation {rotation}")
    pass  # Implementation goes here


def look_at(pan: float, tilt: float) -> None:
    print(f"Looking at pan {pan} and tilt {tilt}")
    pass  # Implementation goes here


def take_image() -> None:
    print("Taking image")
    pass  # Implementation goes here


def perform(action_name: str) -> None:
    print(f"Performing action {action_name}")
    pass  # Implementation goes here


def get_up() -> None:
    print("Getting up")
    pass  # Implementation goes here


TOOLS_DICT = {
    "move_to": move_to,
    "look_at": look_at,
    "take_image": take_image,
    "perform": perform,
    "get_up": get_up,
}


def choose_tool(
    prompt: str,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    system: str = SYSTEM_PROMPT,
    functions: list = FUNCTIONS,
    tools_dict: dict = TOOLS_DICT,
) -> str:
    print(f"Choosing tool for prompt: {prompt}")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        functions=functions,
        max_tokens=max_tokens,
    )
    print(f"Model response {response.choices[0].message.function_call}")
    if response.choices[0].finish_reason == "function_call":
        function_name = response.choices[0].message.function_call.name
        function_args = json.loads(response.choices[0].message.function_call.arguments)
        function_callable = tools_dict.get(function_name)
        if function_callable:
            return function_callable(**function_args)
        else:
            print(f"Could not find tool for function name: {function_name}")
    else:
        return "No tool chosen."


if __name__ == "__main__":
    choose_tool("I am on the ground in a room")
    choose_tool("I see a human")
    choose_tool("I see a featureless white wall")
