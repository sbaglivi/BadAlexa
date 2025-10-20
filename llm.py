import json
import httpx
import threading
import tts_chunker
import queue

import utils

class LLMClient:
    BASE_PROMPT = """You are a concise and friendly voice assistant.
    Avoid markdown usage.
    Answer the user clearly and naturally, as if speaking aloud.

    Conversation:
    """

    def __init__(self, prev_context: list[utils.Message], event: threading.Event, vocal_q: queue.Queue):
        self.client = httpx.Client(base_url="http://localhost:11434", timeout=20)
        self.stop_event = event
        self.prev_context = prev_context
        self.chunker = tts_chunker.ChunkStreamer(vocal_q)
        self.thread: threading.Thread | None = None

    def add_message(self, author: str, content: str):
        assert author in ["user", "assistant"], "author should be user or assistant"
        self.prev_context.append((author, content))
    
    def reset_context(self):
        self.prev_context = []

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def invoke_llm(self):
        data = {
            "model": "openchat:7b",
            "prompt": self.BASE_PROMPT + "\n".join(message_to_text(msg) for msg in self.prev_context) + "\nAssistant: "
        }
        with self.client.stream("POST", "/api/generate", json=data) as r:
            for raw_part in r.iter_bytes():
                if self.stop_event.is_set():
                    return

                part = json.loads(raw_part)
                yield part["response"]

    def run(self):
        full_r = ""
        for part in self.invoke_llm():
            if self.stop_event.is_set():
                print("llm: event is set!")
                break

            self.chunker.handle_part(part)
            full_r += part
        self.chunker.finalize()
        print(full_r)
            
def message_to_text(msg: tuple[str,str]) -> str:
    author, content = msg
    return f"{author.capitalize()}: {content}"
