"""
Decoding & sampling utilities for the AGI project.

This file provides:
 - decode_with_tokenizer(tok, ids)         : preferred decode (uses tokenizer.decode)
 - decode_tokens(tok, ids)                 : compatibility alias for older imports
 - apply_repetition_controls(logits, recent_ids, ...) : penalties
 - top_k_top_p_filter(logits, top_k, top_p, temperature)
 - sample_from_distribution(dist)
"""

import math
import random
import re
from typing import List, Dict, Tuple

# NOTE: We import tokenizers only when needed to avoid heavy imports at package import time.
# Users should call load_tokenizer from agi.common to get a Tokenizer instance.


def decode_with_tokenizer(tokenizer, ids: List[int]) -> str:
    """
    Decode ids to text using the tokenizer if possible.
    Prefer skipping special tokens and avoiding cleanup that might
    collapse significant spaces. Also sanitize any leftover byte-level
    word-start markers (e.g., 'Ġ' or '▁') if they appear due to
    accidental token-string concatenation upstream.
    """
    # Try HF-style decode with kwargs; fall back gracefully.
    try:
        try:
            text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except TypeError:
            text = tokenizer.decode(ids)
        # Cosmetic sanitization if any markers slipped in
        if 'Ġ' in text or '▁' in text:
            text = text.replace('Ġ', ' ')
            text = text.replace('▁', ' ')
            # Tighten spaces around punctuation
            text = re.sub(r"\s+([,\.\!\?;:\)\]\}])", r"\1", text)
            text = re.sub(r"([\(\[\{])\s+", r"\1", text)
            text = re.sub(r"\s{2,}", " ", text).strip()
        return text
    except Exception:
        # Fallback: attempt to reconstruct from vocab (best-effort)
        vocab = {}
        try:
            v = tokenizer.get_vocab()
            vocab = {i: t for t, i in v.items()}  # invert mapping
        except Exception:
            vocab = {}
        punct = set(list(".,!?;:)]}…"))
        out = []
        for i, tid in enumerate(ids):
            t = vocab.get(tid, "")
            # strip common markers if present
            if t.startswith('Ġ'):
                t = t[1:]
            if t.startswith('▁'):
                t = t[1:]
            if not out:
                out.append(t)
            elif t in punct:
                out.append(t)
            else:
                out.append(" " + t)
        return "".join(out)

# --- Compatibility wrapper older code expects ---
def decode_tokens(tokenizer, ids: List[int]) -> str:
    """
    Backwards-compatible name used by older modules:
       decode_tokens(tokenizer, ids) -> str
    Maps to decode_with_tokenizer.
    """
    return decode_with_tokenizer(tokenizer, ids)

# --- Repetition / frequency controls -----------------------------------------
def apply_repetition_controls(logits: Dict[int, float], recent_ids: List[int],
                              repetition_penalty: float = 1.15,
                              presence_penalty: float = 0.2,
                              frequency_penalty: float = 0.5) -> Dict[int, float]:
    """
    Adjust logits in-place (returns a new dict) by penalizing tokens seen in recent_ids.
    - repetition_penalty: divides the logit for repeated tokens
    - presence_penalty: subtractive penalty for presence
    - frequency_penalty: subtractive penalty scaled by counts
    """
    counts = {}
    for tid in recent_ids:
        counts[tid] = counts.get(tid, 0) + 1
    adj = {}
    for tid, logit in logits.items():
        x = float(logit)
        if tid in counts:
            x = x / float(max(1e-8, repetition_penalty))
            x = x - float(presence_penalty) - float(frequency_penalty) * float(counts[tid])
        adj[tid] = x
    return adj

def top_k_top_p_filter(logits: Dict[int, float], top_k: int = 40, top_p: float = 0.95, temperature: float = 0.85) -> List[Tuple[int, float]]:
    """
    From a mapping {token_id: logit}, produce a list [(token_id, prob), ...] after:
      - selecting top_k by logit
      - applying temperature-scaled softmax
      - truncating by cumulative top_p mass
    Returns a normalized probability list suitable for sampling.
    """
    if not logits:
        return []

    items = sorted(logits.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        items = items[:max(1, int(top_k))]

    mx = max(v for _, v in items) if items else 0.0
    probs = []
    s = 0.0
    for tid, v in items:
        # stabilized softmax numerator with temperature
        p = math.exp((v - mx) / max(1e-8, float(temperature)))
        probs.append((tid, p))
        s += p

    if s <= 0:
        # uniform fallback
        n = len(probs)
        if n == 0:
            return []
        return [(tid, 1.0 / n) for tid, _ in probs]

    probs = [(tid, p / s) for tid, p in probs]

    # top-p truncation
    probs.sort(key=lambda kv: kv[1], reverse=True)
    cum = 0.0
    filtered = []
    for tid, p in probs:
        if cum >= top_p and len(filtered) >= 1:
            break
        filtered.append((tid, p))
        cum += p

    # renormalize
    s2 = sum(p for _, p in filtered)
    if s2 <= 0:
        n = len(filtered)
        if n == 0:
            return []
        return [(tid, 1.0 / n) for tid, _ in filtered]
    return [(tid, p / s2) for tid, p in filtered]

def sample_from_distribution(dist: List[Tuple[int, float]]) -> int:
    """
    Sample a token id from a list of (id, prob) pairs.
    Returns the sampled id or None if dist empty.
    """
    if not dist:
        return None
    r = random.random()
    acc = 0.0
    for tid, p in dist:
        acc += p
        if r <= acc:
            return tid
    # guard
    return dist[-1][0]

# Optional alias names for compatibility with other modules that might import them:
# Older code sometimes used `decode_tokens`, `sample_from_dist`, or `top_k_top_p`.
sample_from_dist = sample_from_distribution
top_k_top_p = top_k_top_p_filter
