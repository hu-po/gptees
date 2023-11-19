import base64
import io
import json
import os
import shutil
import subprocess

import cv2
import requests
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd
from scipy.io.wavfile import write


CLIENT: OpenAI = OpenAI()

# Vision model turns single images into text descriptions consumed by the system model
VISION_MODEL: str = "gpt-4-vision-preview"
VISION_PROMPT: str = ". ".join(
    [
        "You are the robot vision module",
        "Describe what you see in brevity",
        "If there are humans mention them and their location",
        "If there are objects mention them and their location",
        "You might be staring at the ceiling",
        "If you don't know what you see, say so",
    ]
)
MAX_TOKENS_VISION: int = 32  # max tokens for reply
VISION_DEVICE_PATH: str = "/dev/video0"  # Camera device path
# VISION_DEVICE_PATH: str = "/dev/usb_cam"  # Camera device path

# Audio models
TTS_MODEL: str = "tts-1"  # Text-to-speech model
STT_MODEL: str = "whisper-1"  # Speech-to-text model
VOICE: str = "echo"  # (alloy, echo, fable, onyx, nova, and shimmer)
GREETING: str = "hello there"  # Greeting is spoken on start
AUDIO_RECORD_SECONDS: int = 6  # Duration for audio recording
AUDIO_SAMPLE_RATE: int = 22100  # Sample rate for audio recording
AUDIO_CHANNELS: int = 1  # mono
AUDIO_OUTPUT_PATH: str = "/tmp/audio.wav"  # audio is constantly overwritten

# System model chooses tools and actions to perform based on vision
SYSTEM_MODEL: str = "gpt-4-1106-preview"
SYSTEM_PROMPT: str = ". ".join(
    [
        "You are the master node in a robot control system",
        "The user is the robot vision module",
        "As the master node, you decide what tools to use",
        "The robot's goals are to explore and understand the environment",
        "If a human is visible, perform the greet action",
        # "If the robot is looking at the ceiling, perform the get_up action",
        "When in doubt, pick a random action!",
        "Try to be random in your movements",
    ]
)
SYSTEM_MAX_TOKENS: int = 32
SYSTEM_TEMPERATURE: float = 0.0
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "explore",
            "description": "Explore the world in a specified direction",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": [
                            "move_forward",
                            "move_backward",
                            "move_left",
                            "move_right",
                            "rotate_left",
                            "rotate_right",
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
            "name": "look",
            "description": "Take an image from the camera and use the robot vision module to describe it",
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
                    "action_name": {
                        "type": "string",
                        "enum": [
                            "left_shot",
                            "right_shot",
                            "stand",
                            "walk_ready",
                            "twist",
                            "three",
                            "four",
                            "hand_back",
                            "greet",
                        ],
                    },
                },
            },
            "required": ["action_name"],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "listen",
            "description": "Listen for a specified duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["duration"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak",
            "description": "Speak a specified text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                    },
                },
                "required": ["text"],
            },
        },
    },
]
FUNCTIONS = [tool["function"] for tool in TOOLS]


def look(
    device: str = VISION_DEVICE_PATH,
    prompt: str = VISION_PROMPT,
    vision_model: str = VISION_MODEL,
    max_tokens: int = MAX_TOKENS_VISION,
) -> str:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        return f"Cannot open webcam at {device}"
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        return f"Could not capture an image from the webcam at {device}"
    cap.release()  # Release the webcam
    _, buffer = cv2.imencode(".jpg", frame)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    }
    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response.json()["choices"][0]["message"]["content"]


def listen(
    duration: int = AUDIO_RECORD_SECONDS,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    channels: int = AUDIO_CHANNELS,
    output_path: str = AUDIO_OUTPUT_PATH,
) -> str:
    print(f"Recording for {duration} seconds.")
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
    )
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    write(output_path, sample_rate, audio_data)  # Save as WAV file
    with open(output_path, "rb") as audio_file:
        transcript = CLIENT.audio.transcriptions.create(
            model=STT_MODEL, file=audio_file, response_format="text"
        )
    print(f"Transcript: {transcript}")
    return f"Listened for {duration} seconds, heard __{transcript}__"


def speak(
    text: str,
    model: str = TTS_MODEL,
    voice: str = VOICE,
    save_to_file=True,
) -> str:
    # Check if the file already exists
    file_name = f"/tmp/test.{text[:10]}.mp3"
    if not os.path.exists(file_name):
        response = CLIENT.audio.speech.create(model=model, voice=voice, input=text)
        byte_stream = io.BytesIO(response.content)
        audio = AudioSegment.from_file(byte_stream, format="mp3")
        if save_to_file:
            audio.export(file_name, format="mp3")
            print(f"Saved audio to {file_name}")
    else:
        print(f"Audio already exists at {file_name}")
        audio = AudioSegment.from_file(file_name, format="mp3")
    print(f"Playing audio: {text}")
    play(audio)
    return f"Speaking text: {text}"


def perform(action_name: str) -> str:
    # Copy over local file to ros directory
    shutil.copy(
        "/home/ubuntu/gptees/perform.py",
        "/home/ubuntu/ros_ws/src/ainex_example/scripts/perform.py",
    )
    cmd = [
        "python3",
        "/home/ubuntu/ros_ws/src/ainex_example/scripts/perform/perform.py",
        action_name,
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = proc.communicate()
    except Exception as e:
        return f"Error executing action {action_name}: {e}"
    if proc.returncode != 0:
        return f"Action {action_name} failed with error: {stderr}"
    else:
        return f"Action {action_name} completed. Output: {stdout}"


def explore(direction: str) -> str:
    # cmd = ["rosrun", "robot", "explore.py", direction]
    # proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    return f"I explored by moving in {direction}"


# def perform(action_name: str) -> str:
#     cmd = ["roslaunch", "ainex_example"]
#     if action_name == "climb_stairs":
#         cmd.extend(["climb_stairs_node.launch", "state:=climb_stairs"])
#     elif action_name == "declimb_stairs":
#         cmd.extend(["climb_stairs_node.launch", "state:=declimb_stairs"])
#     elif action_name == "crawl_under":
#         cmd.append("crawl_under_node.launch")
#     elif action_name == "hurdles":
#         cmd.append("hurdles_node.launch")
#     elif action_name == "kick_ball":
#         cmd.append("kick_ball_node.launch")
#     elif action_name == "visual_patrol":
#         cmd.append("visual_patrol_node.launch")
#     else:
#         return f"Unknown action {action_name}"


TOOLS_DICT = {
    "explore": explore,
    "look": look,
    "perform": perform,
    "listen": listen,
    "speak": speak,
}


def choose_tool(
    prompt: str,
    model: str = SYSTEM_MODEL,
    max_tokens: int = SYSTEM_MAX_TOKENS,
    temperature: float = SYSTEM_TEMPERATURE,
    system: str = SYSTEM_PROMPT,
    functions: list = FUNCTIONS,
    tools_dict: dict = TOOLS_DICT,
) -> str:
    print(f"Choosing tool for prompt: {prompt}")
    response = CLIENT.chat.completions.create(
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
            print(f"Calling {function_name} with {function_args}")
            return function_callable(**function_args)
        else:
            return f"Unknown tool {function_name}"
    else:
        return "No tool chosen"


if __name__ == "__main__":
    speak(GREETING)
    what_i_see = look()
    speak(what_i_see)
    what_i_hear = listen()
    speak(what_i_hear)
    what_i_did = choose_tool(f"{what_i_see}. {what_i_hear}")
    speak(what_i_did)
