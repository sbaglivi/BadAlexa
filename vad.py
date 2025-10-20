import time
import webrtcvad

import utils

VAD = webrtcvad.Vad(3)


def is_user_speaking(block_f32, tts_rms, last_tts_end_ts, debug=False):
    ts = time.time()
    pcm = utils.to_int16(block_f32).tobytes()
    vad_result = VAD.is_speech(pcm, utils.RATE)
    log = debug and vad_result

    # No recent TTS → just return VAD
    if not (last_tts_end_ts and ts - last_tts_end_ts < 4.5 and tts_rms is not None):
        if log: print("user speaking, no recent tts")
        return vad_result

    # --- During or right after TTS ---
    energy = utils.compute_rms(block_f32) * 32768
    tts_rms = max(1e-2, min(tts_rms, 2e-2))
    ratio = energy / tts_rms

    # Less strict threshold
    if ratio < 1.7:   # instead of 1.9
        return False

    if log: print(f"user speaking, recent tts: ration {ratio} | voice {energy} | tts {tts_rms}")
    return vad_result
