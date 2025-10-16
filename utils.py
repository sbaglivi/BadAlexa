import queue
import numpy as np


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
