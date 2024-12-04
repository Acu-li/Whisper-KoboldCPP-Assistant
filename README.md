# Whisper-KoboldCPP-Assistant.

A STT/TTS assistant using KoboldCPP as its llm backend.

## Features

- **Keyword Detection**: Listens for specific keywords such as "Sophie" to activate the assistant.
- **Speech-to-Text**: Records and transcribes spoken user input using OpenAI's Whisper model.
- **Conversational AI**: Sends the transcribed text to a local LLM endpoint for response generation.
- **Text-to-Speech**: Converts the AI's response into speech and plays it back using the XTTS server.
- **Chat History Management**: Maintains a history of interactions, with the ability to reset it using specific keywords like "reset."
- **Database**: If a specific word is in the prompt, a database will be searched, and Information within that database will be added to the context.

## Requirements

- Python 3.x
- Libraries:
  - `requests`
  - `os`
  - `tempfile`
  - `sounddevice`
  - `numpy`
  - `whisper`
  - `pygame`
  - `dotenv`
- KoboldCPP Instance (https://github.com/LostRuins/koboldcpp)
- XTTS-API-SERVER (https://github.com/daswer123/xtts-api-server)

## How to Use

Here is a Video on how to set it up!
https://www.youtube.com/watch?v=ifzhvX6HnIc

1. **Setup the Environment**:
   - Ensure all dependencies are installed. Use `pip install -r requirements.txt` for any missing packages.
   - Whisper may also require PyTorch to be installed. Use the following command to install it: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`.
   - Edit the `env.template.txt` and rename it to `.env`.
   - Edit the `db.txt` to include everything your LLM needs to know. If the corresponding word is in the prompt, the corresponding information will be sent with the author's note.

2. **Run the Program**:
   - Start the script using Python:  
     ```bash
     python whispercuda.py
     ```
   - Choose a microphone by entering its corresponding number.

3. **Interact with Sophie**:
   - Speak the activation keyword (e.g., "Sophie").
   - Speak your Question after the `beep` tone.
   - Wait.

4. **Reset Chat History**: (Optional)
   - Speak a reset keyword (e.g., "reset") to clear the chat history.

## Planned for the future

- Function calling, so it can turn on lights, or play music. Kinda Siri Like.
- Better Keyword Spotting, its pretty shitty currently.
- LLM can add stuff to the database itself, kinda like long term memory.
