import json
import httpx
import threading
import tts_chunker
import queue

BASE_PROMPT = """You are a concise and friendly voice assistant.
Answer the user clearly and naturally, as if speaking aloud.
Do not include explanations about what you're doing or how.
Just say the response itself, nothing else.

Conversation:
"""

def parse_follow_up(raw: str) -> bool:
    print("raw", raw)
    try:
        follow_up = json.loads(raw)["follow_up"]
    except json.JSONDecodeError:
        print(f"failed to parse follow up: {raw}")
        return False

    if not isinstance(follow_up, bool):
        raise TypeError(f"follow up was not bool: {follow_up}")

    print(f"parsed follow_up: {follow_up}")
    return follow_up

def wrapper(prev: list[tuple[str,str]], event: threading.Event, result_q: queue.Queue, add_to_q):
    chunker = tts_chunker.ChunkStreamer(add_to_q)
    full_response = "" # if barge in, this is not a good source, we need to know what was vocalized
    # that kinda means we never need this since we should probably always rely on what was vocalized?
    # this might just be good for logging purposes
    # when things get vocalized, they need to be accumulated somewhere, until generation stops, then those things
    # get saved somewhere else (e.g. conversation context) and then we start clean again when we reach generation again
    # if we reach idle state we wipe out conversation context as well
    for part in invoke(prev):
        if event.is_set():
            print("event is set!")
            # NB: follow up is useless if we get interrupted
            break

        # full_response += part
        # if follow_up is None:
        #     if '\n' not in full_response:
        #         continue

        #     follow_up_raw, full_response = full_response.split("\n", 1)
        #     part = full_response
        #     follow_up = parse_follow_up(follow_up_raw)

        chunker.handle_part(part)
    chunker.finalize()
    print(full_response)
    result_q.put(False)
        
def message_to_text(msg: tuple[str,str]) -> str:
    author, content = msg
    return f"{author.capitalize()}: {content}"

def invoke(prev_context: list[tuple[str,str]]):
    # fake = [
    #     "this is a random response.\n",
    #     "from an llm called John\n",
    #     "I hope this was helpful\n",
    #     "have a nice day\n",
    # ]
    # for p in fake:
    #     yield p
    data = {
        "model": "llama3:8b",
        "prompt": BASE_PROMPT + "\n".join(message_to_text(msg) for msg in prev_context) + "\nAssistant:"
    }
    with httpx.stream("POST", "http://localhost:11434/api/generate", json=data, timeout=20) as r:
        for raw_part in r.iter_bytes():
            part = json.loads(raw_part)
            yield part["response"] # part["done"] tells us if it's the last part