import sounddevice as sd
import webrtcvad
import numpy as np
import threading
import queue
import time
from scipy.io import wavfile
from scipy.signal import resample

import utils
import transcribe

RATE = 16000
CHANNELS = 1
BLOCKSIZE = 320        # 20 ms
VAD = webrtcvad.Vad(3)

# === configurable ===
PLAYBACK_FILE = "test.wav"   # path to your test audio
COOLDOWN_AFTER_STOP = 0.5          # seconds
ECHO_SUPPRESS_MULT = 2.1


def compute_rms(x):
    return np.sqrt(np.mean(x ** 2))


i = 0
# def is_user_speaking(block_f32, last_tts_end = None, tts_rms = None):
def is_user_speaking(block_f32, rms_holder):
    last_tts_end = rms_holder["last_tts_end"]
    tts_rms = rms_holder["tts_rms"]
    ts = time.time()

    recent_tts = last_tts_end and ts - last_tts_end < 1.
    if recent_tts:
        energy = compute_rms(block_f32)
        ratio = energy / max(tts_rms, 1e-2)
        if ratio < ECHO_SUPPRESS_MULT:
            return False

    return VAD.is_speech(utils.to_int16(block_f32).tobytes(), RATE)

# def is_user_speaking(block_f32, rms_holder):
#     tts_active = rms_holder["tts_active"]
#     tts_rms = rms_holder["tts_rms"]
#     last_tts_end_ts = rms_holder["last_tts_end"]
#     ts = time.time()
#     global i 
#     i = (i+1) % 100

#     # Stage 1: cooldown after playback
#     if (not tts_active) and last_tts_end_ts and (ts - last_tts_end_ts) < COOLDOWN_AFTER_STOP:
#         print("cooldown")
#         return False

#     energy = compute_rms(block_f32)

#     ratio = energy / (tts_rms + 1e-5)
#     if tts_active and ratio < ECHO_SUPPRESS_MULT:
#         return False
#     # Stage 2: active playback gating
#     # if tts_active and tts_rms and energy < tts_rms * ECHO_SUPPRESS_MULT:
#     #     if i == 0:
#     #         print("low energy", energy, tts_rms * ECHO_SUPPRESS_MULT)
#     #     return False

#     return VAD.is_speech(utils.to_int16(block_f32).tobytes(), RATE)


def play_audio(path, stop_event, rms_holder):
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]  # mono

    # --- optional resample if not 16kHz ---
    if sr != RATE:
        from scipy.signal import resample
        num_samples = int(len(data) * RATE / sr)
        data = resample(data, num_samples).astype(np.int16)
        print(f"Resampled from {sr} to {RATE} Hz")

    rms_holder["tts_active"] = True
    rms_holder["last_tts_end"] = None

    rms_smooth = 0.0
    MIN_RMS = 0.002
    MAX_RMS = 0.02
    SMOOTH = 0.9  # higher = slower changes

    with sd.RawOutputStream(samplerate=RATE, channels=1, dtype="int16") as stream:
        pos = 0
        block_len = 2048
        while pos < len(data):
            if stop_event.is_set():
                print("Playback interrupted by user speech!")
                break
            end = pos + block_len
            chunk = data[pos:end]
            stream.write(chunk.tobytes())

            new_rms = np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2))
            rms_smooth = SMOOTH * rms_smooth + (1 - SMOOTH) * new_rms
            rms_holder["tts_rms"] = float(np.clip(rms_smooth, MIN_RMS, MAX_RMS))
            rms_holder["last_tts_end"] = time.time()

            pos = end
        rms_holder["tts_active"] = False
    print("Playback finished.")


def main():
    stop_event = threading.Event()
    rms_holder = {"tts_rms": 0.0, "tts_active": False, "last_tts_end": None}

    # Start playback in background
    t = threading.Thread(target=play_audio, args=(PLAYBACK_FILE, stop_event, rms_holder), daemon=True)
    t.start()

    # Set up mic input
    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        pcm_f32 = indata[:, 0]
        audio_q.put(pcm_f32.copy())

    with sd.InputStream(samplerate=RATE, blocksize=BLOCKSIZE, channels=CHANNELS,
                        dtype="float32", callback=audio_callback):
        print("Listening for barge-in... Speak to interrupt playback!")

        while not stop_event.is_set():
            user_buf = []
            preroll = queue.deque(maxlen=int(1.2 * RATE))
            tts_stop_ts = None
            last_voiced_ts = None
            TRAIL_SIL_MS = 1000
            MAX_UTTER_S = 10.0
            start_ts = time.time()

            while True:
                block = audio_q.get()
                ts = time.time()
                preroll.extend(block)

                user_speaking = is_user_speaking(block, rms_holder)

                # ---- during playback ----
                if not stop_event.is_set():
                    if user_speaking:
                        print("Detected user speech! Stopping playback.")
                        stop_event.set()
                        tts_stop_ts = ts
                        # include a bit of preroll so we don't cut off early speech
                        user_buf.extend(preroll)
                        user_buf.extend(block)
                    continue

                # ---- listening mode ----
                # skip the first 200 ms after playback stop to avoid echo tail
                if tts_stop_ts and (ts - tts_stop_ts) < 0.2:
                    continue

                if user_speaking:
                    last_voiced_ts = ts
                    user_buf.extend(block)
                    continue

                if last_voiced_ts and (ts - last_voiced_ts) > TRAIL_SIL_MS / 1000:
                    print("Silence detected — stopping listening.")
                    break

                if (ts - start_ts) > MAX_UTTER_S:
                    print("Max utterance length reached.")
                    break
    audio_np = np.array(user_buf, dtype=np.float32)
    text = transcribe.transcribe(audio_np)
    print("\nYou said:", text)

main()