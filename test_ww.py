# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Imports
# import pyaudio
import sounddevice as sd
import numpy as np
import openwakeword
from openwakeword.model import Model
import argparse
import queue
from collections import deque

openwakeword.utils.download_models()
# Parse input arguments
parser=argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to predict on at once",
    type=int,
    default=1280,
    required=False
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default='onnx',
    required=False
)

args=parser.parse_args()

# Get microphone stream
# FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size
# audio = pyaudio.PyAudio()
# mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

audio_q = queue.Queue()
def audio_callback(indata, frames, time_info, status):
    pcm_f32 = indata[:,0]
    audio_q.put(pcm_f32.copy())

stream = sd.InputStream(
    samplerate=RATE, blocksize=320, channels=CHANNELS,
    dtype="float32", callback=audio_callback
        )
stream.start()

# Load pre-trained openwakeword models
# if args.model_path != "":
# models_dir = "./venv/lib/python3.12/site-packages/openwakeword/resources/models/"
# owwModel = Model(wakeword_models=[models_dir], inference_framework=args.inference_framework)
# else:
owwModel = Model(inference_framework=args.inference_framework)

n_models = len(owwModel.models.keys())

# Run capture loop continuosly, checking for wakewords
import sys
if __name__ == "__main__":
    # v = audio_q.get()
    # print(type(v))
    # sys.exit(0)
    # Generate output string header
    print("\n\n")
    print("#"*100)
    print("Listening for wakewords...")
    print("#"*100)
    print("\n"*(n_models*3))

    buffer = deque([], maxlen=8000)
    # buffer = np.array([], dtype=np.float32)
    count = 0
    while True:
        v = audio_q.get()
        buffer.extend(v.flatten())
        count = (count + 1 ) % 2
        # buffer = np.concatenate((buffer, v))
        # Get audio
        # audio = pyaudio.PyAudio()
        if len(buffer) > CHUNK:

            chunk = [buffer.popleft() for _ in range(CHUNK)]
            chunk80_f32 = np.array(chunk, dtype=np.float32)
            # buffer = buffer[CHUNK:]
            chunk80_i16 = (chunk80_f32 * 32767).astype(np.int16)
            # mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

            # audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

            # Feed to openWakeWord model
            prediction = owwModel.predict(chunk80_i16)

            # Column titles
            n_spaces = 16
            output_string_header = """
                Model Name         | Score | Wakeword Status
                --------------------------------------
                """

            for mdl in owwModel.prediction_buffer.keys():
                # Add scores in formatted table
                scores = list(owwModel.prediction_buffer[mdl])
                curr_score = format(scores[-1], '.20f').replace("-", "")

                output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Wakeword Detected!"}
                """

            # Print results table
            print("\033[F"*(4*n_models+1))
            print(output_string_header, "                             ", end='\r')