import sounddevice as sd
import numpy as np
from collections import deque
from openwakeword import Model
import queue
import sys

RATE = 16000
BLOCK = 320
CHUNK = 1280

audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    # Always flatten to 1D float32
    audio_q.put(indata[:, 0].copy())

stream = sd.InputStream(
    samplerate=RATE,
    blocksize=BLOCK,
    channels=1,
    dtype="float32",
    callback=audio_callback,
)
stream.start()

oww = Model(inference_framework="onnx")
model_names = list(oww.models.keys())
n_models = len(model_names)

print("\n\n")
print("#" * 80)
print("Listening for wakewords (deque-based)...")
print("#" * 80)
print("\n" * (n_models + 2))

buffer = deque(maxlen=CHUNK)

while True:
    v = audio_q.get()
    buffer.extend(v)

    if len(buffer) < CHUNK:
        continue

    # Create a contiguous float32 array from the deque
    chunk = [buffer.popleft() for _ in range(CHUNK)]
    chunk80_f32 = np.array(chunk, dtype=np.float32)
    # chunk80_f32 = np.fromiter(buffer, dtype=np.float32, count=CHUNK)
    chunk80_i16 = np.clip(chunk80_f32 * 32767, -32768, 32767).astype(np.int16)
    # print("RMS:", np.sqrt(np.mean(chunk80_f32**2)))


    # Run inference
    oww.predict(chunk80_i16)

    n_spaces = 16
    output_string_header = """
        Model Name         | Score | Wakeword Status
        --------------------------------------
        """

    for mdl in oww.prediction_buffer.keys():
        # Add scores in formatted table
        scores = list(oww.prediction_buffer[mdl])
        if scores[-1] >= 0.5:
            print(mdl)
    #     curr_score = format(scores[-1], '.20f').replace("-", "")

    #     output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Wakeword Detected!"}
    #     """

    # # Print results table
    # print("\033[F"*(4*n_models+1))
    # print(output_string_header, "                             ", end='\r')