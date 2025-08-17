
import os
from typing import Any, Dict, Optional, List

SPECIALS_DEFAULT = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[NL]", "[URL]", "[EMAIL]", "[NUM]"]

class _EncObj:
    """Mimic tokenizers.Tokenizer.encode(text).ids interface."""
    def __init__(self, ids: List[int]):
        self.ids = ids

class HFTokenizerWrapper:
    """
    Wrap a HuggingFace fast tokenizer and expose a minimal interface used across the repo:
      - encode(text, add_special_tokens=False) -> _EncObj with .ids
      - decode(ids, skip_special_tokens=True) -> str
      - token_to_id(token) / id_to_token(idx)
      - vocab_size property
      - .hf attribute for access to underlying tokenizer (if needed)
    Also ensures PAD exists (defaults to EOS) and that project extras exist.
    """
    EXTRA_SPECIALS = ["[NL]", "[URL]", "[EMAIL]", "[NUM]"]

    def __init__(self, hf_tok, auto_add_extras: bool = True, pad_to_eos: bool = True):
        self.hf = hf_tok
        if pad_to_eos and getattr(self.hf, "pad_token", None) is None and getattr(self.hf, "eos_token", None) is not None:
            self.hf.pad_token = self.hf.eos_token
        if auto_add_extras:
            missing = [t for t in self.EXTRA_SPECIALS if self.hf.convert_tokens_to_ids(t) in (None, -1)]
            if missing:
                self.hf.add_special_tokens({"additional_special_tokens": missing})

    def encode(self, text: str, add_special_tokens: bool = False) -> _EncObj:
        ids = self.hf.encode(text, add_special_tokens=add_special_tokens)
        return _EncObj(ids)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.hf.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)

    def token_to_id(self, token: str) -> Optional[int]:
        tid = self.hf.convert_tokens_to_ids(token)
        if (tid is None or tid == -1) and token in self.EXTRA_SPECIALS:
            self.hf.add_special_tokens({"additional_special_tokens": [token]})
            tid = self.hf.convert_tokens_to_ids(token)
        return None if tid in (-1, None) else int(tid)

    def id_to_token(self, idx: int) -> str:
        return self.hf.convert_ids_to_tokens([idx])[0]

    @property
    def vocab_size(self) -> int:
        return len(self.hf)

def load_tokenizer(tok_arg: str):
    """
    Accepts:
      - Hugging Face repo id (e.g., 'meta-llama/Llama-3.1-8B')
      - Local directory containing tokenizer assets
      - Path to a single tokenizer.json (fast tokenizer)
    Returns an HFTokenizerWrapper.
    """
    try:
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
    except Exception as e:
        raise RuntimeError("transformers not installed. pip install -U transformers huggingface_hub") from e

    if os.path.isdir(tok_arg):
        hf_tok = AutoTokenizer.from_pretrained(tok_arg, use_fast=True)
        return HFTokenizerWrapper(hf_tok)

    if os.path.isfile(tok_arg) and tok_arg.endswith(".json"):
        # Load a fast tokenizer directly
        hf_tok = PreTrainedTokenizerFast(tokenizer_file=tok_arg)
        return HFTokenizerWrapper(hf_tok)

    # Assume repo id
    hf_tok = AutoTokenizer.from_pretrained(tok_arg, use_fast=True)
    return HFTokenizerWrapper(hf_tok)

def get_special_ids(tokenizer: Any) -> Dict[str, Optional[int]]:
    """
    Return a dict of special token ids for keys in SPECIALS_DEFAULT or known HF specials.
    Works with our HFTokenizerWrapper or a raw HF tokenizer (if provided).
    """
    out: Dict[str, Optional[int]] = {}
    # Support wrapper or raw
    hf = getattr(tokenizer, "hf", tokenizer)
    # HF canonical specials
    for key, attr in (("[BOS]", "bos_token_id"), ("[EOS]", "eos_token_id"), ("[PAD]", "pad_token_id"), ("[UNK]", "unk_token_id")):
        out[key] = getattr(hf, attr, None)
    # Project extras
    for s in ["[NL]", "[URL]", "[EMAIL]", "[NUM]"]:
        try:
            tid = hf.convert_tokens_to_ids(s)
            out[s] = None if tid in (None, -1) else int(tid)
        except Exception:
            out[s] = None
    return out

def vocab_opt(Nnv, d_model, tied=False, gamma=0.84, a=2.66):
    """
    Compute optimal vocabulary size given non-vocab parameter count and d_model.
    If tied=True, embeddings are tied and divisor is d_model instead of 2*d_model.
    """
    N_v_opt = a * (float(Nnv) ** float(gamma))
    divisor = (float(d_model) if tied else (2.0 * float(d_model)))
    return int(round(N_v_opt / divisor))
