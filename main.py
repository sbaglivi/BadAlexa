
import time
import queue
import sounddevice as sd
from collections import deque
import numpy as np
import threading

import utils
import vad
import wakeword
import state
import transcribe
import vocalize
import llm

CHANNELS = 1
BLOCKSIZE = 320        # 20 ms @ 16 kHz; one queue item = 320 samples
PREROLL_MS = 500
RESPONSE_AWAIT_S = 2.5
TRAIL_SIL_MS = 1000     # stop after this much silence
MAX_UTTER_S = 20.
OWW_CHUNK = 1280       # OWW wants 1280 samples (~80 ms)


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
            samplerate=utils.RATE, blocksize=BLOCKSIZE, channels=CHANNELS,
            dtype="float32", callback=audio_callback
        )
        self.stream.start()

        # Audio buffers and state
        self.user_speech_counter = 0
        self.wake_buf = np.array([], dtype=np.float32)
        self.utter_buf = deque(maxlen=int(MAX_UTTER_S * utils.RATE))
        self.stop_event = threading.Event()
        self.last_audio = time.time()
        self.i = 0

        # Long-lived components
        self.tts_q = queue.Queue()
        self.llm_client = llm.LLMClient([], self.stop_event, self.tts_q)
        self.vocalizer = vocalize.Vocalizer(self.tts_q, self.stop_event)

    
    def get_audio(self):
        self.i = (self.i + 1) % 100
        if self.i == 0:
            ts = time.time()
            print(f"since last audio: {ts - self.last_audio:.2f}")
            self.last_audio = ts
        ts = time.time()
        bts, block_f32 = self.audio_q.get()                    # 20 ms, 320 float32 samples
        while ts - bts > .2:
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

                        wakeword.predict(np.zeros(int(utils.RATE * 1.0), dtype=np.int16))
                        print("Found!")
                        tail_len = int(0.150 * utils.RATE)
                        self.utter_buf.extend(self.wake_buf[-tail_len:])
                        self.wake_buf = np.array([], dtype=np.float32)
                        ts = time.time()
                        self.state = state.Listening(last_voiced_ts=ts, start_ts=ts)
                        break

                case state.Listening(last_voiced_ts, start_ts):
                    tts_rms, last_end = self.vocalizer.get_rms_and_last_ts()
                    user_speaking = vad.is_user_speaking(self.last_block, tts_rms, last_end)
                    # user_speaking = vad.is_speech(utils.to_int16(self.last_block).tobytes(), RATE)
                    ts = time.time()
                    if user_speaking:
                        self.state.last_voiced_ts = ts
                        continue

                    if last_voiced_ts is None and (ts - start_ts) >= RESPONSE_AWAIT_S:
                        print("back to idle")
                        self.state = state.Idle
                        self.llm_client.reset_context()
                        self.utter_buf.clear()
                        continue

                    if last_voiced_ts is not None and (ts - last_voiced_ts) < TRAIL_SIL_MS/1000:
                        continue
                    
                    if len(self.utter_buf) < int(0.3 * utils.RATE):
                        self.utter_buf.clear()
                        continue

                    buffer = np.array(list(self.utter_buf), dtype=np.float32)
                    result = transcribe.transcribe(buffer)
                    # empty q, reset stop event, clear audio accumulated
                    utils.empty(self.tts_q)
                    self.stop_event.clear()
                    self.utter_buf.clear()
                    self.vocalizer.start()
                    self.llm_client.add_message("user", result)
                    self.llm_client.start()
                    self.state = state.Generating
                    print("started generating and vocalizing!")

                case state.Generating:
                    if vad.is_user_speaking(self.last_block, *(self.vocalizer.get_rms_and_last_ts()), debug=True):
                        self.user_speech_counter += 1

                    if self.user_speech_counter > 4:  # ~20 ms of continuous speech each
                        self.user_speech_counter = 0
                        preroll_samples = int(utils.RATE * 0.5) # 0.1 = 100ms
                        print("STOP EVENT user speech detected while generating")
                        # Signal all components to stop
                        self.stop_event.set()
                        utils.empty(self.tts_q)
                        # Get actually vocalized text before reset
                        vocalized = self.vocalizer.get_last_vocalized_text()
                        
                        # Prepare for next utterance
                        ts = time.time()
                        utils.trim_deque(self.utter_buf, preroll_samples)
                        self.llm_client.add_message("assistant", vocalized)
                        self.state = state.Listening(last_voiced_ts=ts, start_ts=ts)
                    else:
                        if self.vocalizer.is_speaking():
                            continue

                        self.user_speech_counter = 0
                        vocalized = self.vocalizer.get_last_vocalized_text()
                        ts = time.time()
                        self.llm_client.add_message("assistant", vocalized)
                        self.state = state.Listening(last_voiced_ts=None, start_ts=ts)
                        self.utter_buf.clear()
                        self.stop_event.clear()
                        print("no user speech detected and gen done, now back to listening")
                        
            
if __name__ == "__main__":
    app = App()
    app.main_loop()