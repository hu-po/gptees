import base64
import hashlib
import io
import json
import os
import subprocess

import cv2
import requests
from openai import OpenAI
import pyaudio
from pydub import AudioSegment
from pydub.utils import make_chunks
import sounddevice as sd
from scipy.io.wavfile import write


CLIENT: OpenAI = OpenAI()

# Vision model turns single images into text descriptions consumed by the system model
VISION_MODEL: str = "gpt-4-vision-preview"
VISION_PROMPT: str = ". ".join(
    [
        "Describe the scene and objects",
        "You are a robot vision module",
        "You are small and only 20cm off the ground",
        "Focus on the most important things",
        "If there are humans mention them and their relative position",
        "do not mention the image, save tokens by directly describing the scene",
        "Your response will be sent over JSON, make sure to return a valid JSON string",
        # "You might be staring at the ceiling",
    ]
)
MAX_TOKENS_VISION: int = 16  # max tokens for reply
IMAGE_WIDTH: int = 512  # width of image in pixels
IMAGE_HEIGHT: int = 512  # height of image in pixels
# VISION_DEVICE_PATH: str = "/dev/video0"  # Camera device path on igigi
VISION_DEVICE_PATH: str = "/dev/usb_cam"  # Camera device path on humanoid
IMAGE_OUTPUT_FILENAME: str = "image.jpg"  # Image is constantly overwritten

# Audio models
TTS_MODEL: str = "tts-1"  # Text-to-speech model
STT_MODEL: str = "whisper-1"  # Speech-to-text model
VOICE: str = "echo"  # (alloy, echo, fable, onyx, nova, and shimmer)
GREETING: str = "hello there"  # Greeting is spoken on start
AUDIO_RECORD_SECONDS: int = 4  # Duration for audio recording
# AUDIO_SAMPLE_RATE: int = 44100  # Sample rate for quality audio recording
AUDIO_SAMPLE_RATE: int = 16000  # Sample rate for speedy audio recording
AUDIO_CHANNELS: int = 1  # mono
AUDIO_DEVICE: int = 1  # audio device index
AUDIO_OUTPUT_PATH: str = "/tmp/audio.wav"  # recorded audio is constantly overwritten

# System model chooses functions based on logs
SYSTEM_MODEL: str = "gpt-4-1106-preview"
SYSTEM_PROMPT: str = ". ".join(
    [
        "You are the function master node in a robot control system",
        "You monitor the robot log and decide when to run functions",
        "The robot's goals are to explore and understand the environment",
        "The robot can observe the world through sight and sound",
        "Make sure to often listen and look",
        "If a human is visible, perform the greet action or speak to them",
        "If you hear a human, respond to them by speaking",
        "Try to move towards interesting things",
        # "A good default is to listen",
        "Always pick a function to run, the other robot nodes depend on you",
    ]
)
SYSTEM_MAX_TOKENS: int = 16
SYSTEM_TEMPERATURE: float = 0.3
SYSTEM_LOG_LENGTH: int = 4  # Number of lines to keep in the log
FUNCTIONS = [
    {
        "name": "move",
        "description": "Explore the world by moving in a specified direction",
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
                        "rotate left",
                        "rotate right",
                    ],
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "look",
        "description": "Look in the specified direction, and use the robot vision module to describe the scene",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": [
                        "forward",
                        "left",
                        "right",
                        "up",
                        "down",
                    ],
                },
            },
            "required": ["direction"],
        },
    },
    {
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
    {
        "name": "listen",
        "description": "Listen for a specified duration in seconds",
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
    {
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
]
DEFAULT_FUNCTION: str = "listen"
DEFAULT_ACTION_NAME: str = "greet"
DEFAULT_MOVE_DIRECTION: str = "forward"
DEFAULT_LOOK_DIRECTION: str = "forward"

def listen(
    duration: int = AUDIO_RECORD_SECONDS,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    channels: int = AUDIO_CHANNELS,
    output_path: str = AUDIO_OUTPUT_PATH,
) -> str:
    speak(f"listening for {duration} seconds")
    print(f"Listening for {duration} seconds")
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
    )
    sd.wait()  # Wait until recording is finished
    print(f"Recording finished, saving to {output_path}")
    write(output_path, sample_rate, audio_data)  # Save as WAV file
    with open(output_path, "rb") as audio_file:
        transcript = CLIENT.audio.transcriptions.create(
            model=STT_MODEL, file=audio_file, response_format="text"
        )
    print(f"Transcript: {transcript}")
    speak(f"{transcript}?")
    return f"Listened for {duration} seconds and heard {transcript}"


def speak(
    text: str,
    model: str = TTS_MODEL,
    voice: str = VOICE,
    device: str = AUDIO_DEVICE,
    save_to_file=True,
) -> str:
    file_name = f"/tmp/tmp{hashlib.sha256(text.encode()).hexdigest()[:10]}.mp3"
    if not os.path.exists(file_name):
        response = CLIENT.audio.speech.create(model=model, voice=voice, input=text)
        byte_stream = io.BytesIO(response.content)
        seg = AudioSegment.from_file(byte_stream, format="mp3")
        if save_to_file:
            seg.export(file_name, format="mp3")
            print(f"Saved audio to {file_name}")
    else:
        print(f"Audio already exists at {file_name}")
        seg = AudioSegment.from_file(file_name, format="mp3")
    # print(f"Playing audio: {text}")
    # p = pyaudio.PyAudio()
    # stream = p.open(
    #     format=p.get_format_from_width(seg.sample_width),
    #     channels=seg.channels,
    #     rate=seg.frame_rate,
    #     output=True,
    #     output_device_index=device,
    # )

    # # Just in case there were any exceptions/interrupts, we release the resource
    # # So as not to raise OSError: Device Unavailable should play() be used again
    # try:
    #     # break audio into half-second chunks (to allows keyboard interrupts)
    #     for chunk in make_chunks(seg, 500):
    #         stream.write(chunk._data)
    # finally:
    #     stream.stop_stream()
    #     stream.close()

    #     p.terminate()
    return f"Spoke {text}"

def robot_command(command:str, filename:str, logstr:str):
    _path = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    cmd = ["python3", _path, "--command", command]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = proc.communicate()
    except Exception as e:
        print(f"Exception on robot command {command}, {e}")
        return f"Error on robot command {command}"
    if proc.returncode != 0:
        print(f"Robot command {command} failed with error: {stderr}")
        return f"Error on robot command {command}"
    else:
        print(f"Robot command {command} sucessfully. Output: {stdout}")
        return f"{logstr} {command}"
    
def perform(action_name: str = DEFAULT_ACTION_NAME) -> str:
    return robot_command(action_name, "perform.py", "Performed action")

def move(direction: str = DEFAULT_MOVE_DIRECTION) -> str:
    return robot_command(direction, "move.py", "Moved")

def look_at(direction: str = DEFAULT_LOOK_DIRECTION) -> str:
    return robot_command(direction, "look_at.py", "Looked at ")

def image() -> str:
    return robot_command(IMAGE_OUTPUT_FILENAME, "image.py", "Captured image")

def look(
    direction: str = DEFAULT_LOOK_DIRECTION,
    device: str = VISION_DEVICE_PATH,
    prompt: str = VISION_PROMPT,
    vision_model: str = VISION_MODEL,
    max_tokens: int = MAX_TOKENS_VISION,
    image_path: str = IMAGE_OUTPUT_FILENAME,
) -> str:
    speak(look_at(direction))
    print(f"Looking at {device}")
    frame = cv2.imread(image_path)
    if frame is None:
        return f"Could not read the image from {image_path}"
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
    content = response.json()["choices"][0]["message"]["content"]
    print(f"Vision response: {content}")
    speak(content)
    return f"Looked {direction} and saw {content}"


REPERTOIRE = {
    "move": move,
    "look": look,
    "perform": perform,
    "listen": listen,
    "speak": speak,
}


def do(
    prompt: str,
    model: str = SYSTEM_MODEL,
    max_tokens: int = SYSTEM_MAX_TOKENS,
    temperature: float = SYSTEM_TEMPERATURE,
    system: str = SYSTEM_PROMPT,
    functions: list = FUNCTIONS,
    repertoire: dict = REPERTOIRE,
    default_function: str = DEFAULT_FUNCTION,
) -> str:
    print(f"Prompt for do: {prompt}")
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
    if response.choices[0].message.function_call is None:
        print(f"Defaulting to {default_function}")
        return repertoire.get(default_function)()
    else:
        function_name = response.choices[0].message.function_call.name
        print(f"Function name: {function_name}")
        try:
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            print(f"Function args: {function_args}")
        except json.JSONDecodeError as e:
            print(f"Could not decode function args: {e}")
            function_args = {}
        print(f"Function args: {function_args}")
        function_callable = repertoire.get(function_name)
        if not function_callable:
            return f"Unknown function {function_name}"
        return function_callable(**function_args)


if __name__ == "__main__":
    log = []
    speak(GREETING)
    log.append(look())
    log.append(listen())
    while True:
        if len(log) > SYSTEM_LOG_LENGTH:
            log = log[-SYSTEM_LOG_LENGTH:]
        log.append(do("\n".join(log)))
