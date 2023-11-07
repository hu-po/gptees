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

# Constants for easy editing
API_KEY = os.environ["OPENAI_API_KEY"]
VISION_MODEL = "gpt-4-vision-preview"
TTS_MODEL = "tts-1"
STT_MODEL = "whisper-1"
VOICE = "alloy"
VIDEO_DEVICE_PATH = "/dev/video4"
AUDIO_RECORD_SECONDS = 3  # Duration for audio recording
AUDIO_SAMPLE_RATE = 44100  # Sample rate for audio recording
AUDIO_CHANNELS = 1  # mono
AUDIO_OUTPUT_PATH = "/tmp/gpt_audio.wav"

# Initialize OpenAI client
client = OpenAI()

# Initialize the webcam
cap = cv2.VideoCapture(VIDEO_DEVICE_PATH)
if not cap.isOpened():
    raise IOError("Cannot open webcam")


def capture_and_show_image():
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        raise ValueError("Could not capture an image from the webcam")
    return frame


def encode_image_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    return jpg_as_text


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
        "max_tokens": 24,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response.json()["choices"][0]["message"]["content"]


def stream_and_play(text):
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=VOICE,
        input=text,
    )
    byte_stream = io.BytesIO(response.content)
    audio = AudioSegment.from_file(byte_stream, format="mp3")
    play(audio)


def main():
    try:
        while True:
            # Take an image and show it to the user
            frame = capture_and_show_image()
            cv2.imshow("Captured Image", frame)
            cv2.waitKey(1)  # Display the image for a short moment to render the window
            print("Press any key to close the image and start audio recording.")
            cv2.waitKey()  # Wait for a key press
            cv2.destroyAllWindows()
            record_audio()
            prompt = transcribe_audio()
            print(f"Transcribed prompt: {prompt}")
            print(f"Sending to vision model with size {frame.shape}")
            base64_image = encode_image_to_base64(frame)
            reply = vision(prompt, base64_image)
            print(f"Vision model reply: {reply}")
            print("Playing audio.")
            #HACK: Audio beginnign clips, so add some nonsense to the start
            reply = "mhm mhm mhm " + reply
            stream_and_play(reply)
    except KeyboardInterrupt:
        print("Exiting program.")
    finally:
        cap.release()  # Release the webcam
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()