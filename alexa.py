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

import transcribe
RATE = 16000
CHANNELS = 1
BLOCKSIZE = 320        # 20 ms @ 16 kHz; one queue item = 320 samples
OWW_CHUNK = 1280       # OWW wants 1280 samples (~80 ms)
PREROLL_MS = 200       # keep a bit before wake fires
TRAIL_SIL_MS = 900     # stop after this much silence
MAX_UTTER_SEC = 15
VAD_AGGR = 2           # 0..3 (higher = more aggressive)
RESPONSE_AWAIT_MS = 3000

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
            text = transcribe.transcribe(utter)
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
