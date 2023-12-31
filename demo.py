import os
import base64
import cv2
import requests
import io
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play

API_KEY = os.environ["OPENAI_API_KEY"]  # Put it in your bashrc
VISION_MODEL = "gpt-4-vision-preview"  # Vision model
TTS_MODEL = "tts-1"  # Text-to-speech model
STT_MODEL = "whisper-1"  # Speech-to-text model
VOICE = "echo"  # (alloy, echo, fable, onyx, nova, and shimmer)
VIDEO_DEVICE_PATH = "/dev/video0"  # Camera device path
CAMERA_WAIT_ON = False  # Pause camera to show image
CAMERA_WAIT_MS = 2056  # How long to show image before
MAX_TOKENS_VISION = 32  # max tokens for reply
AUDIO_RECORD_SECONDS = 6  # Duration for audio recording
AUDIO_SAMPLE_RATE = 22100  # Sample rate for audio recording
AUDIO_CHANNELS = 1  # mono
AUDIO_OUTPUT_PATH = "/tmp/gpt_audio.wav"

client = OpenAI()


def capture(show_image=CAMERA_WAIT_ON):
    cap = cv2.VideoCapture(VIDEO_DEVICE_PATH)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        raise ValueError("Could not capture an image from the webcam")
    cap.release()  # Release the webcam
    if show_image:
        cv2.destroyAllWindows()
        cv2.imshow("Captured Image", frame)
        cv2.waitKey(1)  # Display the image for a short moment to render the window
        print("Press any key to close the image and start audio recording.")
        cv2.waitKey(CAMERA_WAIT_MS)
        cv2.destroyAllWindows()
    return frame


def encode_image_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    return jpg_as_text


def vision(prompt, base64_image):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": VISION_MODEL,
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
        "max_tokens": MAX_TOKENS_VISION,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response.json()["choices"][0]["message"]["content"]


def record_audio():
    print(f"Recording for {AUDIO_RECORD_SECONDS} seconds.")
    audio_data = sd.rec(
        int(AUDIO_RECORD_SECONDS * AUDIO_SAMPLE_RATE),
        samplerate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
    )
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    write(AUDIO_OUTPUT_PATH, AUDIO_SAMPLE_RATE, audio_data)  # Save as WAV file


def transcribe_audio(audio_path: str = AUDIO_OUTPUT_PATH):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=STT_MODEL, file=audio_file, response_format="text"
        )
    return transcript


def text2speech(text, save_to_file=False, file_name="output.mp3"):
    # Check if the file already exists and the save_to_file flag is True
    if save_to_file and os.path.exists(file_name):
        audio = AudioSegment.from_file(file_name, format="mp3")
    else:
        # If the file doesn't exist, create the audio and save it if required
        response = client.audio.speech.create(model=TTS_MODEL, voice=VOICE, input=text)
        byte_stream = io.BytesIO(response.content)
        audio = AudioSegment.from_file(byte_stream, format="mp3")

        # Save the file if save_to_file is True
        if save_to_file:
            audio.export(file_name, format="mp3")

    # Play the audio
    play(audio)


if __name__ == "__main__":
    while True:
        text2speech(
            "taking image in 5, 4, 3, 2, ... 1",
            save_to_file=True,
            file_name="/tmp/countdown.mp3",
        )
        frame = capture()
        text2speech(
            "speak your question",
            save_to_file=True,
            file_name="/tmp/speak_question.mp3",
        )
        record_audio()
        prompt = transcribe_audio()
        print(f"Transcribed prompt: {prompt}")
        print(f"Sending to vision model with size {frame.shape}")
        base64_image = encode_image_to_base64(frame)
        reply = vision(prompt, base64_image)
        print(f"Vision model reply: {reply}")
        print("Playing audio.")
        text2speech(reply)
