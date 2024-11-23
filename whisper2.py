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
import winsound
from sentence_transformers import SentenceTransformer, util

load_dotenv()
LOCALHOST_ENDPOINT = os.getenv('LOCALHOST_ENDPOINT') # URL des Koboldcpp-Servers
XTTS_ENDPOINT = os.getenv('XTTS_ENDPOINT') # URL des XTTS-Servers

warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

bot_name = "Sophie"
max_context_length = 6000
max_length = 256
repetition_penalty = 1.1
temperature = 0.5
snip_words = ["User:", "Bot:", "You:", "Me:", "{}:".format(bot_name)]

MODEL_TYPE = "base"
model = whisper.load_model(MODEL_TYPE)

KEYWORDS = ["hey sophie", "hey, sophie", "sophie"]
RESET_KEYWORDS = [
    "reset your memorys", "reset your memories", "reset", "reset memorys", "reset memory", "reset memories"
]
RECORDING_DURATION = 7

chat_history = []
reset_active = False

def play_sound(filename="bling.mp3"):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

def load_database(file_path):
    database = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('|', 1)
            database[key.strip()] = value.strip()  # Leerzeichen entfernen
    return database

def search_database(prompt, database):
    relevant_results = []
    for key, value in database.items():
        if is_relevant(prompt, key):  # Überprüfe nur die Schlüssel
            relevant_results.append(f"{key}: {value}")  # Schlüssel und Wert zusammenführen
    
    return " | ".join(relevant_results) if relevant_results else ""  # Ergebnisse zusammenfügen


def is_relevant(prompt, key):
    prompt_lower = prompt.lower()
    key_lower = key.lower()
    return key_lower in prompt_lower  # Wenn der Schlüssel im Prompt vorkommt, ist er relevant



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
    current_time = get_berlin_time()  # Deutsche Zeit abrufen
    
    database = load_database("db.txt")  # Lädt die Datenbank
    db_result = search_database(prompt, database)  # Sucht relevante Informationen zum aktuellen Prompt

    
    # Authors Note mit Bedingung für Datenbank-Ergebnisse
    if db_result:
        authors_note = f"[Authors Note: Database results: {db_result}. This is only information for you, it is not coming from the user.]"
    else:
        authors_note = ""  # Keine Authors Note, wenn keine Ergebnisse gefunden wurden

    # Prompt erweitern und an die API senden
    post_url = f'{LOCALHOST_ENDPOINT}/api/v1/generate'
    instruct_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are " + bot_name + ", the user’s closest childhood friend, always there for them no matter what. You’re warm, funny, and quick-witted, with a knack for making people feel comfortable. You love chatting, cracking jokes, and being supportive whenever they need someone to talk to. This is roleplay, and you are their trusted friend, not an AI. Never break character and always be their favorite person to talk to. Keep your responses concise but maintain the warmth and humor that makes you their favorite person."
    
    # Historie einfügen
    for interaction in chat_history:
        instruct_prompt += f"<|eot_id|><|start_header_id|>user<|end_header_id|>{interaction['user_name']}: {interaction['user_prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{bot_name}: {interaction['bot_response']}"
    
    # User-Prompt einfügen (authors_note bleibt separater Zusatz)
    append_prompt = f"<|eot_id|><|start_header_id|>user<|end_header_id|>{user}: {prompt}{authors_note}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{bot_name}:"
    payload = {
        "max_context_length": max_context_length,
        "max_length": max_length,
        "prompt": f"{instruct_prompt} {append_prompt}",
        "rep_pen": repetition_penalty,
        "temperature": temperature,
        "stopping_strings": json.dumps(snip_words)
    }
    
    # Anfrage senden
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
                if keyword == "sophie":
                    print("Keyword 'Sophie' recognised...")
                    return "sophie"
                else:
                    print("Waiting for 'Sophie'...")
                    continue

        for reset_keyword in RESET_KEYWORDS:
            if reset_keyword in transcription:
                print(f"Reset-Keyword recognised: {reset_keyword}")
                reset_chat_history()
                play_sound()
                print("Waiting for a new keyword...")
                return "reset"

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
        winsound.PlaySound(temp_audio_file_path, winsound.SND_FILENAME)
        os.remove(temp_audio_file_path)
        print("Sound file deleted")
    else:
        print(f"TTS-Error: {response.status_code} - {response.text}")

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

        elif keyword_transcription == "reset":
            pass



if __name__ == '__main__':
    main()
