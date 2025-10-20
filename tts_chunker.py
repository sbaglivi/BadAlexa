import re
import utils
import queue

SENTENCE_RE = re.compile(r'[\.!\?…]+[\"\')\]]*\s')   # sentence enders
SOFT_RE     = re.compile(r'[,;:\n]\s')               # softer pauses

class ChunkStreamer:
    def __init__(self, vocal_q: queue.Queue, min_sent_len: int = 80):
        self.vocal_q = vocal_q
        self.buffer = ""
        self.min_sent_len = min_sent_len

    def handle_part(self, text: str):
        self.buffer += text
        if len(self.buffer) >= self.min_sent_len:
            self._flush()

    def _flush(self, force: bool = False):
        self.buffer = self.buffer.strip()
        if not self.buffer:
            return

        if force:
            # self.add_to_q(self.buffer)
            self.vocal_q.put(self.buffer)
            self.buffer = ""
            return

        if (m := utils.last_iter_value(SENTENCE_RE.finditer(self.buffer))):
            cut = m.end()
            v, self.buffer = self.buffer[:cut], self.buffer[cut:]
            self.vocal_q.put(v)

        # soft boundary if we already have enough text
        if (m := utils.last_iter_value(SOFT_RE.finditer(self.buffer))):
            cut = m.end()
            v, self.buffer = self.buffer[:cut], self.buffer[cut:]
            self.vocal_q.put(v)

    def finalize(self):
        self._flush(force=True)
        self.vocal_q.put("")
