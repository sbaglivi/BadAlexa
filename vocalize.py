import time
import queue
from piper import PiperVoice
import threading
import sounddevice as sd

import utils


MODEL = "en_US-amy-medium.onnx"  # or .onnx.gz
SAMPLE_RATE = 22050                 # set to your model's rate (e.g., 16000, 22050, 24000)


class Vocalizer:
    def __init__(self):
        # TTS model and queue
        self.voice = PiperVoice.load("./" + MODEL)
        self.tts_q: queue.Queue[str] = queue.Queue()
        self.max_chunk_length = 12000

        # State tracking
        self.vocalized_text = ""  # What was actually played to the user
        self.tts_rms = None
        self.last_tts_end = None
        
        # Thread control
        self._stop_event = threading.Event()
        self._worker_thread = None

    def start(self):
        """Start the TTS worker thread."""
        if self._worker_thread is not None:
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """Stop the TTS worker thread and cleanup."""
        if self._worker_thread is None:
            return
        self._stop_event.set()
        self.stop_playback = True
        utils.empty(self.tts_q)  # Clear pending texts
        self._worker_thread.join()
        self._worker_thread = None

    def reset(self):
        """Reset state between conversations while keeping thread alive."""
        self.vocalized_text = ""
        self.tts_rms = None
        self.last_tts_end = None
        utils.empty(self.tts_q)
        
    def is_speaking(self):
        """Public interface to check if vocalization is in progress."""
        return self._worker_thread is not None and self._worker_thread.is_alive()
        
    def get_last_vocalized_text(self):
        """Get the text that was actually played to the user."""
        return self.vocalized_text

    def _tts_worker(self):
        """Long-lived worker that processes text chunks from tts_q."""
        while not self._stop_event.is_set():
            try:
                chunk = self.tts_q.get(timeout=1.)
            except queue.Empty:
                continue

            if chunk == "":  # Signal to finish current conversation
                self.tts_q.task_done()
                continue

            self.speak(chunk)
            self.spoken.append(chunk)
            self.tts_q.task_done()

    def add_to_q(self, text: str):
        """Add text to the TTS queue."""
        self.tts_q.put(text)

    def speak(self, text: str):
        """Synthesize and play a single text chunk."""
        audio_q = queue.Queue()
        total_size = 0
        played_size = 0

        def writer():
            """Short-lived worker that handles audio output for one utterance."""
            nonlocal played_size
            with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
                while True:
                    try:
                        chunk = audio_q.get(timeout=0.1)
                    except queue.Empty:
                        if self._stop_event.is_set():
                            break
                        continue
                    
                    if chunk is None or self._stop_event.is_set():
                        break

                    self.last_tts_end = time.time()
                    self.tts_rms = utils.compute_rms(chunk)
                    stream.write(chunk)
                    played_size += len(chunk)
                    self.last_tts_end = time.time()

        # Start writer thread for this utterance
        writer_thread = threading.Thread(target=writer, daemon=True)
        writer_thread.start()

        # Feed chunks from synthesis and track total size
        for large_chunk in self.voice.synthesize(text):
            arr = large_chunk.audio_int16_array
            for chunk in utils.chunk_by_len(arr, self.max_chunk_length):
                if self._stop_event.is_set():
                    break
                total_size += len(chunk)
                audio_q.put(chunk)

        audio_q.put(None)  # Signal writer to finish
        writer_thread.join()  # Clean up the writer thread
        
        # Update vocalized text based on how much was actually played
        if total_size > 0:
            vocalized_portion = played_size / total_size
            vocalized_chars = int(len(text) * vocalized_portion)
            self.vocalized_text = text[:vocalized_chars]
