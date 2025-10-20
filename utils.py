import queue
import numpy as np
from collections import deque
from datetime import datetime
from pathlib import Path

RATE = 16000
Message= tuple[str, str]

def chunk_by_len(arr, cl):
    for i in range(0, len(arr), cl):
        yield arr[i:i+cl]


def empty(q: queue.Queue):
    try:
        while True:
            q.get(False)
    except queue.Empty:
        pass


def last_iter_value(i):
    v = None
    for v in i:
        pass
    return v


def to_int16(pcm_f32):
    # sd outputs values in [-1,1] range
    # whisper expects them as int16 [-32768, 32767], we go by 1 less to avoid overflow
    return (pcm_f32 * 32767).astype(np.int16)

def compute_rms(int16_chunk: np.ndarray) -> float:
    # Convert int16 PCM to float in [-1, 1] and compute RMS
    f32 = int16_chunk.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(f32 ** 2))) if f32.size else 0.0

def trim_deque(dq: deque, keep: int):
    # Remove items from the left until length <= keep
    while len(dq) > keep:
        dq.popleft()

start = datetime.now().strftime("%m-%d_%H:%M:%S")
log_path = Path(f"./history/{start}")
log_path.mkdir(0o755, exist_ok=True)