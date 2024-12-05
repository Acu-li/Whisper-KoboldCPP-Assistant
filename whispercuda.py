import os
import json
import time
import sounddevice as sd
import numpy as np
import whisper
import pygame
from dotenv import load_dotenv
import requests
import tempfile
from datetime import datetime
import pytz
import warnings
from sentence_transformers import SentenceTransformer, util
import torch

load_dotenv()
LOCALHOST_ENDPOINT = os.getenv('LOCALHOST_ENDPOINT')  # URL des Koboldcpp-Servers
XTTS_ENDPOINT = os.getenv('XTTS_ENDPOINT')  # URL des XTTS-Servers

warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

bot_name = "Sophie"
max_context_length = 6000
max_length = 256
repetition_penalty = 1.1
temperature = 0.5
snip_words = ["User:", "Bot:", "You:", "Me:", "{}:".format(bot_name)]

MODEL_TYPE = "small"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(MODEL_TYPE, device=device)

KEYWORDS = ["hey sophie", "hey, sophie", "sophie"]
RESET_KEYWORDS = [
    "reset your memorys", "reset your memories", "reset", "reset memorys", "reset memory", "reset memories"
]
RECORDING_DURATION = 7

chat_history = []
reset_active = False

# Pygame initialisieren für Audio
pygame.mixer.init()

def play_sound(filename="bling.mp3"):
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

def play_tts(text):
    url = XTTS_ENDPOINT
    data = {
        "text": text,
        "speaker_wav": "1",
        "language": "en"
    }
    headers = {
        "Content-Type": "application/json",
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print("TTS-Request successful")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(response.content)
            temp_audio_file_path = temp_audio_file.name
            print(f"Saved as {temp_audio_file_path}")
        
        # WAV-Datei mit pygame abspielen
        pygame.mixer.music.load(temp_audio_file_path)
        pygame.mixer.music.play()
        
        # Auf das Ende der Wiedergabe warten
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        # Temporäre Datei löschen
        os.remove(temp_audio_file_path)
        print("Sound file deleted")
    else:
        print(f"TTS-Error: {response.status_code} - {response.text}")

def load_database(file_path):
    database = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('|', 1)
            database[key.strip()] = value.strip()
    return database

def search_database(prompt, database):
    relevant_results = []
    for key, value in database.items():
        if is_relevant(prompt, key):  # Überprüfe nur die Schlüssel
            relevant_results.append(f"{key}: {value}")  # Schlüssel und Wert zusammenführen
    
    return " | ".join(relevant_results) if relevant_results else ""  # Ergebnisse zusammenfügen

# Laden des SentenceTransformer-Modells
st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

def is_relevant(prompt, key):
    # Nutze SentenceTransformer, um Ähnlichkeiten zu prüfen
    embeddings = st_model.encode([prompt, key], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item() > 0.7  # Relevanz basierend auf einem Schwellenwert

def choose_microphone():
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{idx}: {device['name']}")
    mic_index = int(input("Choose a Microphone (Number): "))
    return mic_index

def get_berlin_time():
    berlin_tz = pytz.timezone("Europe/Berlin")
    berlin_time = datetime.now(berlin_tz).strftime("%H:%M")  # 24-Stunden-Format ohne Sekunden
    return berlin_time

def send_prompt(user, prompt):
    current_time = get_berlin_time()
    database = load_database("db.txt")
    db_result = search_database(prompt, database)

    if db_result:
        authors_note = (
            f"[Authors Note: The current time is {current_time}. Database results: {db_result}. "
            "This is only information for you, it is not coming from the user. Please use the 24H clock.]"
        )
    else:
        authors_note = f"[Authors Note: The current time is {current_time}. Please use the 24H clock.]"

    post_url = f'{LOCALHOST_ENDPOINT}/api/v1/generate'
    instruct_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are " + bot_name + ", the user’s closest childhood friend, always there for them no matter what. You’re warm, funny, and quick-witted, with a knack for making people feel comfortable. You love chatting, cracking jokes, and being supportive whenever they need someone to talk to. This is roleplay, and you are their trusted friend, not an AI. Never break character and always be their favorite person to talk to. Keep your responses concise but maintain the warmth and humor that makes you their favorite person."

    for interaction in chat_history:
        instruct_prompt += f"<|eot_id|><|start_header_id|>user<|end_header_id|>{interaction['user_name']}: {interaction['user_prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{bot_name}: {interaction['bot_response']}"

    append_prompt = f"<|eot_id|><|start_header_id|>user<|end_header_id|>{user}: {prompt}{authors_note}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{bot_name}:"
    payload = {
        "max_context_length": max_context_length,
        "max_length": max_length,
        "prompt": f"{instruct_prompt} {append_prompt}",
        "rep_pen": repetition_penalty,
        "temperature": temperature,
        "stopping_strings": json.dumps(snip_words)
    }
    response = requests.post(post_url, json=payload)
    return response.text

def record_audio(duration):
    print("Recording...")
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    print("Recording stopped")
    return np.squeeze(audio)

def transcribe_audio(audio):
    print("Transcribing....")
    result = model.transcribe(audio, fp16=False)
    return result['text']

def reset_chat_history():
    global chat_history
    chat_history = []
    print("Context was reset")

def listen_for_keyword(mic_index):
    print("Waiting for Keyword...")
    while True:
        audio = record_audio(3)
        transcription = transcribe_audio(audio).lower()
        print(f"Recognised Text: {transcription}")

        for keyword in KEYWORDS:
            if keyword in transcription:
                print(f"Keyword '{keyword}' recognised!")
                play_sound()
                return "sophie"

        for reset_keyword in RESET_KEYWORDS:
            if reset_keyword in transcription:
                print(f"Reset-Keyword recognised: {reset_keyword}")
                reset_chat_history()
                play_sound()
                return "reset"

def main():
    print("Keyword Spotting is now actively listening...")
    mic_index = choose_microphone()

    while True:
        keyword_transcription = listen_for_keyword(mic_index)

        if keyword_transcription == "sophie":
            print("Recording prompt...")
            audio = record_audio(RECORDING_DURATION)
            text = transcribe_audio(audio)
            print(f"Transcribed Text: {text}")

            print("Sending to LLM...")
            response = send_prompt("User", f"{keyword_transcription} {text}")

            try:
                response_text = json.loads(response)['results'][0]['text']
            except (KeyError, IndexError, json.JSONDecodeError):
                response_text = "Error. Idk what happened"

            print(f"{bot_name}: {response_text}")
            play_tts(response_text)

            chat_history.append({"user_name": "User", "user_prompt": text, "bot_response": response_text})

if __name__ == '__main__':
    main()
