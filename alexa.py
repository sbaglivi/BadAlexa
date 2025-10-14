import queue, time, collections, tempfile, os, wave
import re
import threading
import subprocess
import httpx
import json
import numpy as np
import sounddevice as sd
import webrtcvad
from scipy.signal import butter, lfilter
from faster_whisper import WhisperModel
from openwakeword.model import Model as OWWModel

RATE = 16000
CHANNELS = 1
BLOCKSIZE = 320        # 20 ms @ 16 kHz; one queue item = 320 samples
OWW_CHUNK = 1280       # OWW wants 1280 samples (~80 ms)
PREROLL_MS = 200       # keep a bit before wake fires
TRAIL_SIL_MS = 900     # stop after this much silence
MAX_UTTER_SEC = 15
VAD_AGGR = 2           # 0..3 (higher = more aggressive)
RESPONSE_AWAIT_MS = 3000

def float32_to_int16(x):
    return np.clip(x, -1.0, 1.0).astype(np.float32).reshape(-1)  # ensure 1D
def to_int16(pcm_f32):
    return (pcm_f32 * 32767).astype(np.int16)

def write_wav(path, int16_pcm, rate=RATE):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes(int16_pcm.tobytes())

audio_q = queue.Queue()
def audio_callback(indata, frames, time_info, status):
    # indata: float32 [-1,1], shape (frames, channels)
    pcm_f32 = indata[:,0] if indata.ndim > 1 else indata
    audio_q.put(pcm_f32.copy())

stream = sd.InputStream(
    samplerate=RATE, blocksize=BLOCKSIZE, channels=CHANNELS,
    dtype="float32", callback=audio_callback
)
stream.start()

models_dir = "./venv/lib/python3.11/site-packages/openwakeword/resources/models/"
oww = OWWModel(inference_framework="onnx", wakeword_models=[models_dir + "hey_mycroft_v0.1.onnx"])   # ONNX backend on macOS
vad = webrtcvad.Vad(VAD_AGGR)
# size can be base, small, medium etc
whisper = WhisperModel("small.en")               # pick size you like

# --- state ---
STATE_IDLE = 0
STATE_RECORDING = 1
state = STATE_IDLE
oww_buf = np.array([], dtype=np.float32)     # accumulate for 1280-sample OWW windows
preroll = collections.deque(maxlen=int((PREROLL_MS/1000)*RATE))
utter = []                                   # buffers after wake
last_voiced_ts = None
utt_t0 = None

def vad_has_speech(int16_block_20ms):
    # webrtcvad wants 10/20/30ms @ 16k, 16-bit
    return vad.is_speech(int16_block_20ms.tobytes(), RATE)

print("Listening…")

tts_q = queue.Queue()
def tts_worker():
    while True:
        chunk = tts_q.get()
        if chunk is None:
            break
        try:
            speak(chunk)  
        finally:
            tts_q.task_done()

threading.Thread(target=tts_worker, daemon=True).start()
SENTENCE_RE = re.compile(r'[\.!\?…]+[\"\')\]]*\s')   # sentence enders
SOFT_RE     = re.compile(r'[,;:\n]\s')               # softer pauses

MIN_SENT_LEN   = 60       
MIN_SOFT_LEN   = 80
MAX_CHARS      = 280
FLUSH_SECONDS  = 1.6     

def maybe_flush(buffer, last_flush_time, force=False):
    text = "".join(buffer)

    # hard sentence boundary
    if len(text) >= MIN_SENT_LEN:
        m = None
        for m in SENTENCE_RE.finditer(text):
            pass
        if m:
            cut = m.end()
            chunk, rest = text[:cut], text[cut:]
            tts_q.put(chunk.strip())
            return ([rest], time.monotonic())

    # soft boundary if we already have enough text
    if len(text) >= MIN_SOFT_LEN:
        m = None
        for m in SOFT_RE.finditer(text):
            pass
        if m:
            cut = m.end()
            chunk, rest = text[:cut], text[cut:]
            tts_q.put(chunk.strip())
            return ([rest], time.monotonic())

    # time-based flush
    if force or (time.monotonic() - last_flush_time) >= FLUSH_SECONDS:
        if len(text) > 0:
            # prefer to cut at last space to avoid mid-word
            if len(text) > MAX_CHARS:
                cut = text.rfind(" ", 0, MAX_CHARS)
                if cut == -1:
                    cut = MAX_CHARS
                chunk, rest = text[:cut], text[cut:]
            else:
                chunk, rest = text, ""
            tts_q.put(chunk.strip())
            return ([rest], time.monotonic())

    return (buffer, last_flush_time)

def stream_example(prompt, prev_context) -> tuple[bool, str]:
    base = """
    [SYSTEM]:
    You are a voice assistant.
    Always respond in exactly two parts:
    Line 1: A single JSON object, no extra text, in the form:
    {"follow_up": true|false}

    Rules:
    - follow_up=true if you ask the user a question, need clarification, or want them to keep talking.
    - follow_up=false if your answer is complete and you expect no immediate reply.
    - Use lowercase true/false literals.
    - Do not include anything else on line 1.

    Line 2 and onward: Your natural spoken reply (conversational, no lists or markdown).
    Never output JSON again after line 1.
    Never add explanations or commentary outside this format.
    Examples

    User: "ask me a question"
    Assistant:
    {"follow_up": true}
    What’s the first thing you think about when you hear the word adventure?

    User: "set a timer for 5 minutes"
    Assistant:
    {"follow_up": false}
    Okay, setting a timer for 5 minutes.
    """
    if prev_context:
        base += prev_context
    data = {
        "model": "llama3:8b",
        "prompt": base + "[USER]: " + prompt
    }
    out = []
    last_flush = time.monotonic()
    follow_up = None
    follow_up_acc: str = ""
    full_response = []
    with httpx.stream("POST", "http://localhost:11434/api/generate", json=data, timeout=20) as r:
        for raw_part in r.iter_bytes():
            part = json.loads(raw_part)
            tok = part["response"]
            if follow_up is None:
                follow_up_acc += tok
                if '\n' not in follow_up_acc:
                    continue

                raw, rest = follow_up_acc.split("\n", maxsplit=1)
                try:
                    print(raw)
                    follow_up = json.loads(raw)["follow_up"]
                    if not isinstance(follow_up, bool):
                        raise ValueError(f"follow up was not bool: {follow_up}")

                    print(f"parsed follow_up: {follow_up}")
                    tok = rest
                except Exception as e:
                    print("acc:")
                    print(follow_up_acc)
                    raise e

            print(tok, end="", flush=True)
            out.append(tok)
            full_response.append(tok)
            out, last_flush = maybe_flush(out, last_flush)
            if part.get("done"):
                out, last_flush = maybe_flush(out, last_flush, force=True)
    
    tts_q.join()
    print(flush=True)
    return follow_up, "".join(full_response)

MODEL = "en_US-amy-medium.onnx"  # or .onnx.gz
SAMPLE_RATE = 22050                 # set to your model's rate (e.g., 16000, 22050, 24000)

def speak(text: str):
    cmd = [
        "piper",
        "--model", MODEL,
        "--output-raw",
        text,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    done = threading.Event()

    with sd.RawOutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=0,
        finished_callback=done.set,
    ) as stream:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            stream.write(chunk)

    proc.wait()
    done.wait()               # playback drained
    sd.sleep(150)             # short guard before opening the mic


def highpass_iir(x, sr=16000, cutoff=100.0, order=4):
    b, a = butter(order, cutoff/(sr/2.0), btype='highpass')
    return lfilter(b, a, x)

def rms_normalize(x, target_dbfs=-23.0, eps=1e-7):
    rms = np.sqrt(np.mean(np.square(x)) + eps)
    target = 10**(target_dbfs/20.0)
    gain = target / max(rms, eps)
    y = x * gain
    return np.clip(y, -1.0, 1.0)

def empty(q: queue.Queue):
    try:
        while True:
            q.get(False)
    except queue.Empty:
        pass

response_await_start = None
prev_context = ""
while True:
    block_f32 = audio_q.get()                    # 20 ms, 320 float32 samples
    preroll.extend(block_f32)                    # always keep preroll

    # ---- FEED OWW every 1280 samples (80 ms) ----
    oww_buf = np.concatenate([oww_buf, block_f32])
    while len(oww_buf) >= OWW_CHUNK:
        chunk80_f32 = oww_buf[:OWW_CHUNK]; oww_buf = oww_buf[OWW_CHUNK:]
        chunk80_i16 = to_int16(chunk80_f32)
        probs = oww.predict(chunk80_i16)          # returns probability score
        prob = next(iter(probs.values()))
        if prob > 0.7 and state == STATE_IDLE:   # tune threshold
            print("Wakeword detected!")
            state = STATE_RECORDING
            utt_t0 = time.time()
            last_voiced_ts = time.time()         # initialize as now
            # seed utterance with preroll so we don't clip the first word
            if len(preroll):
                utter.append(np.array(preroll, dtype=np.float32))
                preroll.clear()

    # ---- RECORDING STATE: accumulate + VAD endpointing ----
    if state == STATE_RECORDING:
        utter.append(block_f32)

        # 20ms VAD on the same block
        block_i16 = to_int16(block_f32)
        voiced = vad_has_speech(block_i16)
        if voiced:
            last_voiced_ts = time.time()
            if not utt_t0:
                utt_t0 = time.time()
            response_await_start = None
        elif response_await_start:
            if time.time() - response_await_start >= RESPONSE_AWAIT_MS:
                speak("thanks for nothing")
                response_await_start = None
                state = STATE_IDLE
            continue

        # stop conditions: trailing silence OR max length
        if (time.time() - last_voiced_ts) >= (TRAIL_SIL_MS/1000) or \
           (time.time() - utt_t0) >= MAX_UTTER_SEC:
            # finalize utterance
            pcm_f32 = np.concatenate(utter) if utter else np.zeros(1, np.float32)
            pcm_f32 = highpass_iir(pcm_f32, RATE, cutoff=100.)
            pcm_f32 = rms_normalize(pcm_f32, -23.)
            pcm_i16 = to_int16(pcm_f32)
            # save temp WAV for Whisper
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            write_wav(tmp.name, pcm_i16, RATE)

            # transcribe
            segs, info = whisper.transcribe(tmp.name, language=None)
            text = "".join(s.text for s in segs).strip()
            os.unlink(tmp.name)

            print("User said:", text if text else "[no speech]")
            follow_up, response = stream_example(text, prev_context)
            # reset state
            if follow_up:
                print("expecting a follow up!")
                response_await_start = time.time()
                prev_context += f"[USER]: {text}\n[YOU]: {response}\n"
            else:
                state = STATE_IDLE
                prev_context = ""
            empty(audio_q)
            utter.clear()
            last_voiced_ts = None
            utt_t0 = None
