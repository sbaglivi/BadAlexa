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

        self.stop_event = stop_event
        self.thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.thread.start()

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

        self.stop()

    def add_to_q(self, text: str):
        self.tts_q.put(text)

    def speak(self, text: str):
        # depends on voice, text and stop event, should be able to return its result to someone
        total, spoken = 0, 0
        print("vocalizing", text)
        with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
            for large_chunk in self.voice.synthesize(text):
                arr = large_chunk.audio_int16_array
                for chunk in utils.chunk_by_len(arr, self.max_chunk_length):
                    if self.stop_event.is_set():
                        break
                    
                    stream.write(chunk)
                    spoken += len(chunk)
                total += len(arr)

        predicted = int(len(text) * spoken / total)
        print(spoken, total, f"{spoken/total:.1%}", text[:predicted])
    
    def stop(self):
        utils.empty(self.tts_q)




    # v = Vocalizer()
    # v.add_to_q("a violet fox was roaming in a lush garden, full of flowers")
    # v.add_to_q("when a blue caterpillar poked his head up, and asked her: \"what's for dinner\"")
    # v.stop()
    # print("main done")

# def get_input():
#     while True:
#         print("waiting for input: ", end="")
#         input()
#         print("setting event")
#         stop_event.set()
#         print("event set")

# if __name__ == "__main__":
#     talk = threading.Thread(target=main)
#     interr = threading.Thread(target=get_input, daemon=True)
#     talk.start()
#     interr.start()
#     talk.join()