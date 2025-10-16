
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
RESPONSE_AWAIT_MS = 2500
TRAIL_SIL_MS = 900     # stop after this much silence
MAX_UTTER_S = 8.
OWW_CHUNK = 1280       # OWW wants 1280 samples (~80 ms)

silence_window=True
input_too_long=True
follow_up = True
wakeword_detected=True

vad = webrtcvad.Vad(3)

class App:
    def __init__(self):
        self.state = state.Idle
        self.audio_q = queue.Queue()
        def audio_callback(indata, frames, time_info, status):
            pcm_f32 = indata[:,0]
            self.audio_q.put(pcm_f32.copy())

        self.stream = sd.InputStream(
            samplerate=RATE, blocksize=BLOCKSIZE, channels=CHANNELS,
            dtype="float32", callback=audio_callback
        )
        self.stream.start()
        self.wake_buf = np.array([], dtype=np.float32)
        # self.wake_buf = deque(maxlen=int((PREROLL_MS/1000)*RATE))
        self.utter_buf = deque(maxlen=int(MAX_UTTER_S * RATE))
        self.stop_event = threading.Event()
        self.i = 0

    
    def get_audio(self):
        self.i = (self.i + 1) % 80
        if self.i == 0:
            print("qsize", self.audio_q.qsize())
        block_f32 = self.audio_q.get()                    # 20 ms, 320 float32 samples
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
                    user_speaking = vad.is_speech(utils.to_int16(self.last_block).tobytes(), RATE)
                    ts = time.time()
                    if user_speaking:
                        self.state.last_voiced_ts = ts
                        continue

                    if self.state.last_voiced_ts is None and (ts - self.state.start_ts) >= RESPONSE_AWAIT_MS:
                        self.state = state.Idle
                        self.utter_buf.clear()
                        continue

                    if (ts - self.state.last_voiced_ts) < TRAIL_SIL_MS/1000:
                        continue

                    buffer = np.array(list(self.utter_buf), dtype=np.float32)
                    result = transcribe.transcribe(buffer)
                    self.vocalizer = vocalize.Vocalizer(self.stop_event)
                    self.state = state.Generating(prev_turns + [("user", result)])
                    self.result_q = queue.Queue()
                    self.generation_thread = threading.Thread(target=llm.wrapper, args=(self.state.prev_turns, self.stop_event, self.result_q, self.vocalizer.tts_q))
                    self.generation_thread.start()
                    print("started generating and vocalizing!")
                case state.Generating(prev_turns):
                    user_speaking = vad.is_speech(utils.to_int16(self.last_block).tobytes(), RATE)
                    if user_speaking:
                        print("STOP EVENT user speech detected while generating")
                        self.stop_event.set()
                        self.generation_thread.join()
                        ts = time.time()
                        self.stop_event.clear()
                        vocalized = "".join(self.vocalizer.spoken)
                        self.state = state.Listening(prev_turns=prev_turns + [("assistant", vocalized)], last_voiced_ts=ts, start_ts=ts)
                    else:
                        try:
                            follow_up = self.result_q.get(timeout=.5)
                        except queue.Empty:
                            print("empty!")
                            continue
                        # if generation finishes
                        print(f"received follow up: {follow_up}")
                        self.generation_thread.join()
                        self.vocalizer.thread.join()
                        if follow_up:
                            vocalized = "".join(self.vocalizer.spoken)
                            ts = time.time()
                            self.state = state.Listening(prev_turns=prev_turns + [("assistant", vocalized)], last_voiced_ts=None, start_ts=ts)
                            self.stop_event.clear()
                        else:
                            self.state = state.Idle
                        
            
if __name__ == "__main__":
    # event = threading.Event()
    # v = vocalize.Vocalizer(event)
    # v.add_to_q("can you say something?")
    # v.thread.join()
    app = App()
    app.main_loop()