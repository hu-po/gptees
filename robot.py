import os
import base64
import requests
import io
import json
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play


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
VISION_OUTPUT_PATH: str = "/tmp/test.png"  # default image for test behavior

# Audio models
TTS_MODEL: str = "tts-1"  # Text-to-speech model
STT_MODEL: str = "whisper-1"  # Speech-to-text model
VOICE: str = "echo"  # (alloy, echo, fable, onyx, nova, and shimmer)
GREETING: str = "hello there" # Greeting is spoken on start
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
        "If a human is visible, perform the wave action",
        "If the robot is looking at the ceiling, perform the get_up action",
        "When in doubt, move around",
        "Try to be random in your movements",
    ]
)
SYSTEM_MAX_TOKENS: int = 32
SYSTEM_TEMPERATURE: float = 0.0
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
                    "action_name": {
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


def encode_image_from_path(image_path: str = VISION_OUTPUT_PATH) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def vision(
    base64_image: str,
    prompt: str = VISION_PROMPT,
    vision_model: str = VISION_MODEL,
    max_tokens: int = MAX_TOKENS_VISION,
) -> str:
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
    save_to_file = True,
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


def move_to(direction: str) -> str:
    return f"Moving to {direction}"


def look_at(direction: str) -> str:
    return f"Looking at {direction}"


def perform(action_name: str) -> str:
    return f"Performing action {action_name}"


TOOLS_DICT = {
    "move_to": move_to,
    "look_at": look_at,
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
    base64_image = encode_image_from_path()
    what_i_see = vision(base64_image)
    what_i_hear = listen()
    what_i_did = choose_tool(f"{what_i_see}. {what_i_hear}")
    speak(what_i_did)
