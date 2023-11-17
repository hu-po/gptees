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
        "If the robot is looking at the ceiling, perform the get_up action",
        "When in doubt, move around",
        "Try to be random in your movements",
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
            "description": "Move the robot using a specified direction",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": [
                            "forward",
                            "backward",
                            "left",
                            "right",
                            "rotate_left",
                            "rotate_right",
                        ],
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look_at",
            "description": "Orient the robot's head camera (pan and tilt)",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": [
                            "look_up",
                            "look_down",
                            "look_left",
                            "look_right",
                        ],
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "perform",
            "description": "Perform a specified named action",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": [
                            "wave",
                            "get_up",
                        ],
                    },
                },
            },
            "required": ["action_name"],
        },
    },
]
FUNCTIONS = [tool["function"] for tool in TOOLS]


def move_to(direction: str) -> None:
    print(f"Moving to {direction}")
    pass  # Implementation goes here


def look_at(direction: str) -> None:
    print(f"Looking at {direction}")
    pass  # Implementation goes here


def perform(action_name: str) -> None:
    print(f"Performing action {action_name}")
    pass  # Implementation goes here


TOOLS_DICT = {
    "move_to": move_to,
    "look_at": look_at,
    "perform": perform,
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
