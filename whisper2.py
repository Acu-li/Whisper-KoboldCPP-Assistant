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
import winsound
from datetime import datetime
import pytz

load_dotenv()
LOCALHOST_ENDPOINT = os.getenv('LOCALHOST_ENDPOINT') # URL des Koboldcpp-Servers
XTTS_ENDPOINT = os.getenv('XTTS_ENDPOINT') # URL des XTTS-Servers

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
RECORDING_DURATION = 3

chat_history = []
reset_active = False

def play_sound(filename="bling.mp3"):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

def choose_microphone():
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{idx}: {device['name']}")
    mic_index = int(input("Wähle ein Mikrofon (Nummer): "))
    return mic_index

def get_berlin_time():
    berlin_tz = pytz.timezone("Europe/Berlin")
    berlin_time = datetime.now(berlin_tz).strftime("%H:%M")  # 24-Stunden-Format ohne Sekunden
    return berlin_time

def send_prompt(user, prompt):
    current_time = get_berlin_time()  # Deutsche Zeit abrufen
    authors_note = f"[Authors Note: It is currently {current_time} in berlin,Germany]"
    augmented_prompt = f"{prompt} {authors_note}"
    
    post_url = f'{LOCALHOST_ENDPOINT}/api/v1/generate'
    instruct_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are " + bot_name + " the user’s closest childhood friend, always there for them no matter what. You’re warm, funny, and quick-witted, with a knack for making people feel comfortable. You love chatting, cracking jokes, and being supportive whenever they need someone to talk to. This is roleplay, and you are their trusted friend, not an AI. Never break character and always be their favorite person to talk to."
    for interaction in chat_history:
        instruct_prompt += f"<|eot_id|><|start_header_id|>user<|end_header_id|>{interaction['user_name']}: {interaction['user_prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{bot_name}: {interaction['bot_response']}"
    append_prompt = f"<|eot_id|><|start_header_id|>user<|end_header_id|>{user}: {augmented_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{bot_name}: "
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
    print("Aufnahme läuft...")
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    print("Aufnahme beendet.")
    return np.squeeze(audio)

def transcribe_audio(audio):
    print("Audio wird transkribiert...")
    result = model.transcribe(audio, fp16=False)
    return result['text']

def reset_chat_history():
    global chat_history
    chat_history = []
    print("Die Konversationshistorie wurde zurückgesetzt.")

def listen_for_keyword(mic_index):
    print("Warte auf das Keyword...")
    while True:
        audio = record_audio(5)
        transcription = transcribe_audio(audio).lower()
        print(f"Erkannter Text: {transcription}")

        for keyword in KEYWORDS:
            if keyword in transcription:
                print(f"Keyword '{keyword}' erkannt!")
                play_sound()
                if keyword == "sophie":
                    print("Keyword 'Sophie' erkannt! Starte Aufnahme...")
                    return "sophie"
                else:
                    print("Warte weiter auf 'Sophie'...")
                    continue

        for reset_keyword in RESET_KEYWORDS:
            if reset_keyword in transcription:
                print(f"Reset-Keyword erkannt: {reset_keyword}")
                reset_chat_history()
                play_sound()
                print("Warte nach Reset weiter auf das Keyword...")
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
        print("TTS-Anfrage erfolgreich gesendet!")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(response.content)
            temp_audio_file_path = temp_audio_file.name
            print(f"Audiodatei gespeichert als {temp_audio_file_path}")
        winsound.PlaySound(temp_audio_file_path, winsound.SND_FILENAME)
        os.remove(temp_audio_file_path)
        print("Audiodatei gelöscht.")
    else:
        print(f"TTS-Fehler: {response.status_code} - {response.text}")

def main():
    print("Kontinuierliches Keyword-Spotting läuft...")
    mic_index = choose_microphone()
    
    while True:
        keyword_transcription = listen_for_keyword(mic_index)
        
        if keyword_transcription == "sophie":
            print("Aufnahme der nächsten Äußerung...")
            audio = record_audio(RECORDING_DURATION)
            text = transcribe_audio(audio)
            print(f"Transkribierter Text: {text}")
            
            print("Sende an das LLM...")
            response = send_prompt("User", f"{keyword_transcription} {text}")
            
            try:
                response_text = json.loads(response)['results'][0]['text']
            except (KeyError, IndexError, json.JSONDecodeError):
                response_text = "Fehler beim Abrufen der Antwort."
            
            print(f"{bot_name}: {response_text}")
            
            play_tts(response_text)
            
            chat_history.append({"user_name": "User", "user_prompt": text, "bot_response": response_text})

        elif keyword_transcription == "reset":
            pass



if __name__ == '__main__':
    main()
