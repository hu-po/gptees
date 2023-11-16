import os
import io
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play

TTS_MODEL = "tts-1"  # Text-to-speech model
STT_MODEL = "whisper-1"  # Speech-to-text model
VOICE = "echo"  # (alloy, echo, fable, onyx, nova, and shimmer)
AUDIO_RECORD_SECONDS = 6  # Duration for audio recording
AUDIO_SAMPLE_RATE = 22100  # Sample rate for audio recording
AUDIO_CHANNELS = 1  # mono
AUDIO_OUTPUT_PATH = "/tmp/gpt_audio.wav"

client = OpenAI()

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

