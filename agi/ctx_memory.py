
from collections import deque

class RingContext:
    def __init__(self, max_len=262144, decay=0.9997):
        self.buf = deque(maxlen=max_len)
        self.decay = float(decay)
        self.weights = deque(maxlen=max_len)

    def append(self, tok:int):
        self.buf.append(tok)
        self.weights.append(1.0)

    def extend(self, toks):
        for t in toks:
            self.append(t)

    def tokens(self):
        return list(self.buf)

    def decay_step(self):
        if not self.weights:
            return
        self.weights = deque([w*self.decay for w in self.weights], maxlen=self.weights.maxlen)
