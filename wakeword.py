from openwakeword.model import Model as OWWModel
from openwakeword.utils import download_models

download_models()
models_dir = "./venv/lib/python3.12/site-packages/openwakeword/resources/models/"
oww = OWWModel(inference_framework="onnx", wakeword_models=[models_dir + "hey_mycroft_v0.1.onnx"])   # ONNX backend on macOS

def predict(audio_chunk) -> bool:
    # 80ms of audio, 1280 samples
    probs = oww.predict(audio_chunk)
    prob = next(iter(probs.values()))
    return prob > 0.85
