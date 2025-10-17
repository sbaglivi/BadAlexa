
import time
import queue
import sounddevice as sd
from collections import deque
import numpy as np
import webrtcvad
import threading

import utils
import wakeword
import state
import transcribe
import vocalize
import llm

RATE = 16000
CHANNELS = 1
BLOCKSIZE = 320        # 20 ms @ 16 kHz; one queue item = 320 samples
PREROLL_MS = 500
RESPONSE_AWAIT_S = 2.5
TRAIL_SIL_MS = 1000     # stop after this much silence
MAX_UTTER_S = 20.
OWW_CHUNK = 1280       # OWW wants 1280 samples (~80 ms)


vad = webrtcvad.Vad(3)
def is_user_speaking(block_f32, tts_rms=None, last_tts_end_ts=None):
    ts = time.time()
    pcm = utils.to_int16(block_f32).tobytes()
    vad_result = vad.is_speech(pcm, RATE)

    # No recent TTS → just return VAD
    if not (last_tts_end_ts and ts - last_tts_end_ts < 4.5 and tts_rms is not None):
        return vad_result

    # --- During or right after TTS ---
    energy = utils.compute_rms(block_f32) * 32768
    tts_rms = max(1e-3, min(tts_rms, 2e-2))
    ratio = energy / tts_rms

    # Less strict threshold
    if ratio < 1.7:   # instead of 1.9
        return False

    return vad_result

class App:
    def __init__(self):
        # State and queues
        self.state = state.Idle
        self.audio_q = queue.Queue()
        def audio_callback(indata, frames, time_info, status):
            pcm_f32 = indata[:,0]
            self.audio_q.put((time.time(), pcm_f32.copy()))

        # Audio input stream
        self.stream = sd.InputStream(
            samplerate=RATE, blocksize=BLOCKSIZE, channels=CHANNELS,
            dtype="float32", callback=audio_callback
        )
        self.stream.start()

        # Long-lived components
        self.vocalizer = vocalize.Vocalizer()
        self.vocalizer.start()

        # Audio buffers and state
        self.user_speech_counter = 0
        self.wake_buf = np.array([], dtype=np.float32)
        self.utter_buf = deque(maxlen=int(MAX_UTTER_S * RATE))
        self.stop_event = threading.Event()
        self.last_audio = time.time()
        self.i = 0

    
    def get_audio(self):
        self.i = (self.i + 1) % 100
        if self.i == 0:
            ts = time.time()
            print(f"since last audio: {ts - self.last_audio:.2f}")
            self.last_audio = ts
        ts = time.time()
        bts, block_f32 = self.audio_q.get()                    # 20 ms, 320 float32 samples
        while ts - bts > .5:
            bts, block_f32 = self.audio_q.get()                    # 20 ms, 320 float32 samples
        self.last_block = block_f32
        if self.state == state.Idle:
            self.wake_buf = np.concatenate([self.wake_buf, block_f32])
        else:
            self.utter_buf.extend(block_f32)                    # always keep preroll

    def main_loop(self):
        while True:
            self.get_audio()
            match self.state:
                case state.Idle:
                    while len(self.wake_buf) >= OWW_CHUNK:
                        chunk80_f32, self.wake_buf = self.wake_buf[:OWW_CHUNK], self.wake_buf[OWW_CHUNK:]
                        if not wakeword.predict(utils.to_int16(chunk80_f32)):
                            continue

                        wakeword.predict(np.zeros(int(RATE * 1.0), dtype=np.int16))
                        print("Found!")
                        tail_len = int(0.150 * RATE)
                        self.utter_buf.extend(self.wake_buf[-tail_len:])
                        self.wake_buf = np.array([], dtype=np.float32)
                        ts = time.time()
                        self.state = state.Listening(prev_turns=[], last_voiced_ts=ts, start_ts=ts)
                        break

                case state.Listening(prev_turns, last_voiced_ts, start_ts):
                    tts_rms, last_end = 0., None
                    if self.vocalizer is not None:
                        tts_rms, last_end = self.vocalizer.tts_rms, self.vocalizer.last_tts_end
                    user_speaking = is_user_speaking(self.last_block, tts_rms, last_end)
                    # user_speaking = vad.is_speech(utils.to_int16(self.last_block).tobytes(), RATE)
                    ts = time.time()
                    if user_speaking:
                        self.state.last_voiced_ts = ts
                        continue

                    if self.state.last_voiced_ts is None and (ts - self.state.start_ts) >= RESPONSE_AWAIT_S:
                        self.state = state.Idle
                        self.utter_buf.clear()
                        continue

                    if self.state.last_voiced_ts is not None and (ts - self.state.last_voiced_ts) < TRAIL_SIL_MS/1000:
                        continue
                    
                    if len(self.utter_buf) < int(0.3 * RATE):
                        self.utter_buf.clear()
                        continue

                    buffer = np.array(list(self.utter_buf), dtype=np.float32)
                    result = transcribe.transcribe(buffer)
                    self.utter_buf.clear()
                    self.vocalizer.reset()  # Reset state for new conversation
                    self.state = state.Generating(prev_turns + [("user", result)])
                    self.generation_thread = threading.Thread(target=llm.wrapper, args=(self.state.prev_turns, self.stop_event, self.vocalizer.tts_q))
                    self.generation_thread.start()
                    print("started generating and vocalizing!")
                case state.Generating(prev_turns):
                    tts_rms, last_end = 0., None
                    if self.vocalizer is not None:
                        tts_rms, last_end = self.vocalizer.tts_rms, self.vocalizer.last_tts_end
                    user_speaking = is_user_speaking(self.last_block, tts_rms, last_end)
                    if user_speaking:
                        self.user_speech_counter += 1
                    else:
                        self.user_speech_counter = 0

                    if self.user_speech_counter > 3:  # ~60 ms of continuous speech
                        preroll_samples = int(RATE * 0.1) # 100ms
                        print("STOP EVENT user speech detected while generating")
                        # Signal all components to stop
                        self.stop_event.set()
                        self.generation_thread.join()
                        
                        # Get actually vocalized text before reset
                        vocalized = self.vocalizer.get_last_vocalized_text()
                        self.vocalizer.reset()  # Clean up current utterance
                        
                        # Prepare for next utterance
                        ts = time.time()
                        self.stop_event.clear()
                        utils.trim_deque(self.utter_buf, preroll_samples)
                        self.state = state.Listening(prev_turns=prev_turns + [("assistant", vocalized)], last_voiced_ts=ts, start_ts=ts)
                    else:
                        if self.generation_thread.is_alive() or self.vocalizer.is_speaking():
                            continue

                        vocalized = self.vocalizer.get_last_vocalized_text()
                        ts = time.time()
                        self.state = state.Listening(prev_turns=prev_turns + [("assistant", vocalized)], last_voiced_ts=None, start_ts=ts)
                        self.utter_buf.clear()
                        self.stop_event.clear()
                        print("no user speech detected and gen done, now back to listening")
                        
            
if __name__ == "__main__":
    app = App()
    app.main_loop()