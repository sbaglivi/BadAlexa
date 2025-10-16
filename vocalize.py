import time
import queue
from piper import PiperVoice
import threading
import sounddevice as sd

import utils


MODEL = "en_US-amy-medium.onnx"  # or .onnx.gz
SAMPLE_RATE = 22050                 # set to your model's rate (e.g., 16000, 22050, 24000)


class Vocalizer:
    def __init__(self, stop_event: threading.Event):
        self.tts_q: queue.Queue[str] = queue.Queue()
        self.voice = PiperVoice.load("./" + MODEL)
        self.max_chunk_length = 12000
        self.spoken = []
        self.tts_active = False
        self.tts_rms = None
        self.last_tts_end = None

        self.stop_event = stop_event
        self.thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.thread.start()
        self.stop_playback = False

    def tts_worker(self):
        while not self.stop_event.is_set():
            try:
                chunk = self.tts_q.get(timeout=1.)
            except queue.Empty:
                print("worker q empty")
                continue

            if chunk == "":
                self.tts_q.task_done()
                break

            self.speak(chunk)  
            print("spoken", chunk)
            self.spoken.append(chunk)
            self.tts_q.task_done()

        utils.empty(self.tts_q)

    def add_to_q(self, text: str):
        self.tts_q.put(text)

    def speak(self, text: str):
        total, spoken = 0, 0
        print("vocalizing", text)
        self.tts_active = True

        # Create a queue for audio chunks
        audio_q = queue.Queue()

        def writer():
            with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
                while True:
                    try:
                        chunk = audio_q.get(timeout=0.1)
                    except queue.Empty:
                        if self.stop_playback:
                            break
                        continue
                    if chunk is None or self.stop_playback:
                        break
                    self.last_tts_end = time.time()
                    self.tts_rms = utils.compute_rms(chunk)
                    stream.write(chunk)
                    self.last_tts_end = time.time()

        # Start writer thread
        writer_thread = threading.Thread(target=writer, daemon=True)
        writer_thread.start()

        # Feed chunks from synthesis
        for large_chunk in self.voice.synthesize(text):
            arr = large_chunk.audio_int16_array
            for chunk in utils.chunk_by_len(arr, self.max_chunk_length):
                if self.stop_event.is_set():
                    self.stop_playback = True
                    break
                audio_q.put(chunk)
                spoken += len(chunk)
            total += len(arr)

        audio_q.put(None)
        writer_thread.join()  # short-lived thread
        predicted = int(len(text) * spoken / total)
        print(spoken, total, f"{spoken/total:.1%}", text[:predicted])
