import os
import wave
import numpy as np
import tempfile
from scipy.signal import butter, lfilter
from faster_whisper import WhisperModel
import utils

RATE = 16000
whisper = WhisperModel("small.en")               # pick size you like

def highpass_iir(x, sr=16000, cutoff=100.0, order=4):
    b, a = butter(order, cutoff/(sr/2.0), btype='highpass')
    return lfilter(b, a, x)

def rms_normalize(x, target_dbfs=-23.0, eps=1e-7):
    # root mean square amplitude of waveform
    # try to mostly equalize the loudness of the audio clip
    rms = np.sqrt(np.mean(np.square(x)) + eps)
    target = 10**(target_dbfs/20.0)
    gain = target / max(rms, eps)
    y = x * gain
    return np.clip(y, -1.0, 1.0)

def write_wav(path, int16_pcm, rate=RATE):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(int16_pcm.tobytes())

def transcribe(audio) -> str:
    # Pulse-Code Modulation: most common way to digitally represent analog audio signals.
    # signals sampled at regular intervals from continuous waveform
    # can be in int16 range of as a float [-1,1]
    
    # preprocess signals
    pcm_f32 = highpass_iir(audio, RATE, cutoff=100.)
    pcm_f32 = rms_normalize(pcm_f32, -23.)
    pcm_i16 = utils.to_int16(pcm_f32)

    # save temp WAV for Whisper
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    write_wav(tmp.name, pcm_i16, RATE)

    # transcribe
    segs, _info = whisper.transcribe(tmp.name, language=None)
    text = "".join(s.text for s in segs).strip()
    print("tr", text)
    os.unlink(tmp.name)
    return text