"""
https://platform.openai.com/docs/guides/vision
"""
import os
import base64
import cv2
import requests
import io
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play

client = OpenAI()
cap = cv2.VideoCapture('/dev/video4')  # Using the device path

def capture_image_from_webcam():
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    ret, frame = cap.read()  # Capture frame-by-frame
    cap.release()  # When everything done, release the capture
    cv2.destroyAllWindows()
    if ret:
        return frame
    else:
        raise ValueError("Could not capture an image from the webcam")


def encode_image_from_webcam():
    img = capture_image_from_webcam()

    # Convert the image to base64
    _, buffer = cv2.imencode(".jpg", img)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")

    return jpg_as_text


def encode_image_from_path(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def vision(
    image_path: str = None,
    api_key: str = os.environ["OPENAI_API_KEY"],
    prompt: str = "Describe what you see",
    max_tokens: int = 24,
):
    if image_path is None:
        base64_image = encode_image_from_webcam()
    else:
        base64_image = encode_image_from_path(image_path)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
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


def stream_and_play(text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )

    # Convert the binary response content to a byte stream
    byte_stream = io.BytesIO(response.content)

    # Read the audio data from the byte stream
    audio = AudioSegment.from_file(byte_stream, format="mp3")

    # Play the audio
    play(audio)


if __name__ == "__main__":
    reply = vision()
    stream_and_play(reply)
