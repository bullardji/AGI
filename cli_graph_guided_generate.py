#!/usr/bin/env python3
from __future__ import annotations
import argparse, time, random
from contextlib import nullcontext
from typing import List, Set

import torch
from transformers import PreTrainedTokenizerBase
from agi.common import load_tokenizer, get_special_ids
from agi.nn.microformer import MicroTransformer, MicroConfig
from agi.graph.connector import RelGraphView


def resolve_device(arg: str) -> torch.device:
    arg = (arg or "auto").lower()
    if arg == "auto":
        if torch.backends.mps.is_available(): return torch.device("mps")
        if torch.cuda.is_available(): return torch.device("cuda")
        return torch.device("cpu")
    if arg == "mps": return torch.device("mps")
    if arg == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")

def resolve_dtype_for_amp(arg: str, device: torch.device):
    arg = (arg or "off").lower()
    if arg == "off": return None
    if arg == "auto":
        if device.type == "mps": return torch.bfloat16
        if device.type == "cuda": return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return None
    if arg == "bf16": return torch.bfloat16
    if arg == "fp16": return torch.float16
    raise ValueError("--amp must be off|fp16|bf16|auto")

def encode_ids(tok: PreTrainedTokenizerBase, text: str, add_special=False) -> List[int]:
    out = tok.encode(text, add_special_tokens=add_special)
    return out.ids if hasattr(out, "ids") else out

def decode_ids(tok: PreTrainedTokenizerBase, ids: List[int]) -> str:
    return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

def eos_id(tok: PreTrainedTokenizerBase, fallback=0) -> int:
    try:
        sp = get_special_ids(tok)
        if sp and "eos" in sp and sp["eos"] is not None:
            return int(sp["eos"])
    except Exception:
        pass
    return getattr(tok, "eos_token_id", None) or fallback

def load_microformer(snapshot_path: str, device: torch.device) -> MicroTransformer:
    ckpt = torch.load(snapshot_path, map_location="cpu")
    cfgd = ckpt["config"]
    cfg = MicroConfig(**cfgd)
    model = MicroTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().to(device)
    return model

def select_seed_tokens(prompt_ids: List[int], special: Set[int], max_seeds=16) -> List[int]:
    seen, seeds = set(), []
    for tid in prompt_ids:
        if tid in special or tid < 0: continue
        if tid not in seen:
            seen.add(tid); seeds.append(tid)
        if len(seeds) >= max_seeds: break
    return seeds

def build_logit_bias(vocab_size: int, guided_ids: List[int], weights: List[float], alpha: float, device) -> torch.Tensor:
    bias = torch.zeros(vocab_size, device=device)
    for tid, w in zip(guided_ids, weights):
        if 0 <= tid < vocab_size:
            bias[tid] += float(w) * alpha
    return bias

def topk_topp_sample(logits_row: torch.Tensor, top_k: int, top_p: float, temperature: float) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits_row).item())

    logits = logits_row / max(1e-6, temperature)

    if top_k > 0 and top_k < logits.size(-1):
        vals, idxs = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask[idxs] = vals
        logits = mask

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        keep = cum <= top_p
        if not torch.any(keep): keep[0] = True
        mask = torch.full_like(sorted_logits, float("-inf"))
        mask[keep] = sorted_logits[keep]
        back = torch.full_like(logits, float("-inf"))
        back[sorted_idx] = mask
        logits = back

    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return int(next_id)

def main():
    ap = argparse.ArgumentParser("Graph-guided generation with Microformer")
    ap.add_argument("--graph", required=True, help="Path to graph snapshot (e.g., snapshots/relgraph_*.pt)")
    ap.add_argument("--model", required=True, help="Microformer snapshot (.pt) saved by cli_train_microformer")
    ap.add_argument("--tokenizer", required=True, help="HF repo id or local tokenizer dir")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", dest="max_new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", dest="top_k", type=int, default=40)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="auto", help="off|fp16|bf16|auto")
    ap.add_argument("--k-neigh", dest="k_neigh", type=int, default=24, help="neighbors per seed (aggregated)")
    ap.add_argument("--alpha", type=float, default=3.0, help="logit-bias strength for neighbor tokens")
    ap.add_argument("--window", type=int, default=8, help="use recent N tokens to refresh neighbors")
    ap.add_argument("--inject-prefix", action="store_true",
                    help="Also inject a short textual graph context before the prompt")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype_for_amp(args.amp, device)

    tok: PreTrainedTokenizerBase = load_tokenizer(args.tokenizer)
    graph = RelGraphView.from_checkpoint(args.graph)
    model = load_microformer(args.model, device)
    vocab_size = model.lm_head.weight.size(0)

    special_ids = set()
    try:
        sp = get_special_ids(tok) or {}
        for k in ("bos", "eos", "pad", "unk"):
            v = sp.get(k, None)
            if v is not None:
                special_ids.add(int(v))
    except Exception:
        pass

    prompt_ids = encode_ids(tok, args.prompt, add_special=False)
    seeds = select_seed_tokens(prompt_ids, special_ids, max_seeds=16)

    base_neighbors = graph.neighbors_multi(seeds, k=args.k_neigh, exclude=set(seeds))
    base_ids = [nid for (nid, _w) in base_neighbors]
    if len(base_neighbors) > 0:
        wmax = max(w for _, w in base_neighbors) or 1.0
        base_weights = [float(w) / wmax for (_, w) in base_neighbors]
    else:
        base_weights = []

    prefix_ids: List[int] = []
    if args.inject_prefix and base_ids:
        keep = base_ids[: min(12, len(base_ids))]
        txt = " [GRAPH] " + ", ".join(decode_ids(tok, [tid]) for tid in keep if tid not in special_ids) + " [/GRAPH] "
        prefix_ids = encode_ids(tok, txt, add_special=False)

    seq: List[int] = prefix_ids + prompt_ids
    max_seq_len = model.cfg.max_seq_len

    printed_up_to = 0
    if prefix_ids:
        printed_up_to = len(prefix_ids)
    start_len = len(seq)
    t0 = time.time()

    cast_ctx = torch.autocast(device_type=device.type, dtype=dtype) if dtype in (torch.float16, torch.bfloat16) else nullcontext()

    for step in range(args.max_new):
        inp = seq[-max_seq_len:]
        x = torch.tensor([inp], dtype=torch.long, device=device)

        with cast_ctx:
            logits = model(x)                 # [1,T,V]
            last = logits[0, -1, :]           # [V]

        recent = seq[-args.window:] if args.window > 0 else []
        dyn_neighbors = graph.neighbors_multi(recent, k=args.k_neigh, exclude=set(seq))
        dyn_ids = [nid for (nid, _w) in dyn_neighbors]
        if len(dyn_neighbors) > 0:
            dwmax = max(w for _, w in dyn_neighbors) or 1.0
            dyn_weights = [float(w) / dwmax for (_, w) in dyn_neighbors]
        else:
            dyn_weights = []

        guided_ids = base_ids + dyn_ids
        guided_wts = base_weights + dyn_weights
        if guided_ids:
            bias = build_logit_bias(vocab_size, guided_ids, guided_wts, alpha=args.alpha, device=device)
            last = last + bias

        next_id = topk_topp_sample(last, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)
        seq.append(next_id)

        chunk = decode_ids(tok, seq[printed_up_to:])
        if chunk:
            print(chunk, end="", flush=True)
            printed_up_to = len(seq)

        if next_id == eos_id(tok):
            break

    dt = time.time() - t0
    gen_len = max(0, len(seq) - start_len)
    print(f"\n---\nGenerated {gen_len} tokens in {dt:.2f}s ({gen_len/max(1e-9, dt):.1f} tok/s), device={device.type}, amp={dtype}")


if __name__ == "__main__":
    main()
