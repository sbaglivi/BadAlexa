import json
import httpx
import threading
import tts_chunker

BASE_PROMPT = """You are a concise and friendly voice assistant.
Avoid markdown usage.
Answer the user clearly and naturally, as if speaking aloud.

Conversation:
"""

def wrapper(prev: list[tuple[str,str]], event: threading.Event, add_to_q):
    chunker = tts_chunker.ChunkStreamer(add_to_q)
    full_r = ""
    for part in invoke(prev):
        if event.is_set():
            print("llm: event is set!")
            break

        chunker.handle_part(part)
        full_r += part
    chunker.finalize()
    print(full_r)
        
def message_to_text(msg: tuple[str,str]) -> str:
    author, content = msg
    return f"{author.capitalize()}: {content}"

def invoke(prev_context: list[tuple[str,str]]):
    data = {
        "model": "openchat:7b",
        "prompt": BASE_PROMPT + "\n".join(message_to_text(msg) for msg in prev_context) + "\nAssistant:"
    }
    with httpx.stream("POST", "http://localhost:11434/api/generate", json=data, timeout=20) as r:
        for raw_part in r.iter_bytes():
            part = json.loads(raw_part)
            yield part["response"]