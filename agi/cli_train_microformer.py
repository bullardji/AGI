#!/usr/bin/env python3
import argparse, time, json
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from agi.common import load_tokenizer, get_special_ids
from agi.nn.microformer import MicroTransformer, MicroConfig
from contextlib import nullcontext


# ----------------------------
# Device + AMP helpers
# ----------------------------
def resolve_device(arg: str) -> torch.device:
    arg = (arg or "auto").lower()
    if arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if arg == "mps":
        return torch.device("mps")
    if arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")

def resolve_dtype_for_amp(arg: str, device: torch.device):
    arg = (arg or "off").lower()
    if arg == "off":
        return None
    if arg == "auto":
        if device.type == "mps":
            return torch.bfloat16
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return None
    if arg == "bf16":
        return torch.bfloat16
    if arg == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported --amp {arg!r}. Use off|fp16|bf16|auto")

# ----------------------------
# Tokenization utilities
# ----------------------------
def encode_ids(tok: PreTrainedTokenizerBase, text: str, add_special=False):
    out = tok.encode(text, add_special_tokens=add_special)
    return out.ids if hasattr(out, "ids") else out

def eos_id_or_default(tok: PreTrainedTokenizerBase, default_id: int = 0) -> int:
    try:
        sp = get_special_ids(tok)
        if sp and "eos" in sp and sp["eos"] is not None:
            return int(sp["eos"])
    except Exception:
        pass
    eid = getattr(tok, "eos_token_id", None)
    if isinstance(eid, int) and eid >= 0:
        return eid
    for name in ("<|eos|>", "[EOS]", "</s>"):
        try:
            tid = tok.convert_tokens_to_ids(name) if hasattr(tok, "convert_tokens_to_ids") else None
            if isinstance(tid, int) and tid >= 0:
                return tid
            if hasattr(tok, "token_to_id"):
                tid = tok.token_to_id(name)
                if isinstance(tid, int) and tid >= 0:
                    return tid
        except Exception:
            pass
    return default_id

# ----------------------------
# Data streaming
# ----------------------------
def stream_batches(tok, dataset_iterable, seq_len=256, batch=16, limit_steps=None):
    buf = []
    nsteps = 0
    eos_id = eos_id_or_default(tok, default_id=0)

    for row in dataset_iterable:
        text = (row.get("text")
                or row.get("content")
                or row.get("article")
                or row.get("content_text")
                or "")
        if not isinstance(text, str) or not text:
            continue

        ids = encode_ids(tok, text, add_special=False)
        buf.extend(ids)
        buf.append(eos_id)

        need = (seq_len + 1) * batch
        while len(buf) >= need:
            x_list, y_list = [], []
            for _ in range(batch):
                chunk = buf[:seq_len + 1]
                del buf[:seq_len]
                x_list.append(chunk[:-1])
                y_list.append(chunk[1:])
            x = torch.tensor(x_list, dtype=torch.long)
            y = torch.tensor(y_list, dtype=torch.long)
            yield x, y
            nsteps += 1
            if limit_steps and nsteps >= limit_steps:
                return

# ----------------------------
# Chunked CE to cap memory
# ----------------------------
def cross_entropy_chunked(logits: torch.Tensor, targets: torch.Tensor, chunk_tokens: int) -> torch.Tensor:
    """
    Computes CE over [B,T,V] logits and [B,T] targets by slicing token dimension.
    Reduces peak memory on large vocabularies.
    """
    B, T, V = logits.shape
    logits2 = logits.view(B * T, V)
    targets2 = targets.view(B * T)
    total = 0.0
    n = 0
    losses = []
    for start in range(0, logits2.size(0), chunk_tokens):
        end = min(start + chunk_tokens, logits2.size(0))
        # sum reduction to combine properly; we normalize at the end
        loss_part = F.cross_entropy(logits2[start:end], targets2[start:end], reduction="sum")
        losses.append(loss_part)
        n += (end - start)
    loss = torch.stack(losses).sum() / max(1, n)
    return loss

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Train Microformer on an HF dataset (streaming).")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--mix", default=None,
                    help="JSON list of {path, split, weight, [config]}")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--tie-embeddings", action="store_true", default=True)
    ap.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    ap.add_argument("--amp", default="auto", help="off|fp16|bf16|auto")
    ap.add_argument("--loss-chunk-toks", type=int, default=2048,
                    help="Tokens per CE slice to limit peak memory. Reduce if you still OOM.")
    ap.add_argument("--snapshot", required=True)
    args = ap.parse_args()

    # Tokenizer
    tok = load_tokenizer(args.tokenizer)
    if getattr(tok, "pad_token_id", None) is None:
        try:
            tok.pad_token = tok.eos_token
        except Exception:
            pass

    # Dataset iterable (streaming)
    if args.mix:
        spec = json.loads(args.mix)
        if not isinstance(spec, list) or not spec:
            raise ValueError("--mix must be a non-empty JSON list")
        iters, weights = [], []
        for s in spec:
            path = s["path"]
            split = s.get("split", args.split)
            cfg = s.get("config", None)
            ds_iter = load_dataset(path, cfg, split=split, streaming=True) if cfg \
                      else load_dataset(path, split=split, streaming=True)
            iters.append(iter(ds_iter))
            w = float(s.get("weight", 1.0))
            weights.append(max(1, int(round(w * 10))))
        schedule = []
        for i, w in enumerate(weights):
            schedule.extend([i] * w)
        def mixed_iter():
            idx = 0
            while True:
                src = schedule[idx % len(schedule)]
                idx += 1
                try:
                    yield next(iters[src])
                except StopIteration:
                    s = spec[src]
                    ds_iter = load_dataset(s["path"], s.get("config"), split=s.get("split", args.split), streaming=True) \
                              if s.get("config") else load_dataset(s["path"], split=s.get("split", args.split), streaming=True)
                    iters[src] = iter(ds_iter)
                    yield next(iters[src])
        dataset_iterable = mixed_iter()
    else:
        if not args.dataset:
            raise ValueError("Provide --dataset or use --mix.")
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True) \
             if args.config else load_dataset(args.dataset, split=args.split, streaming=True)
        dataset_iterable = iter(ds)

    # Model
    vocab_size = getattr(tok, "vocab_size", None)
    if vocab_size is None and hasattr(tok, "get_vocab"):
        vocab_size = len(tok.get_vocab())
    cfg = MicroConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        tie_embeddings=args.tie_embeddings,
    )
    model = MicroTransformer(cfg)

    # Device + AMP
    device = resolve_device(args.device)
    dtype = resolve_dtype_for_amp(args.amp, device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # New GradScaler API to avoid deprecation warning
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and device.type == "cuda"))
    use_autocast = (dtype in (torch.float16, torch.bfloat16))

    t0 = time.time()
    step, running_loss = 0, 0.0

    for x, y in stream_batches(tok, dataset_iterable, seq_len=args.seq_len, batch=args.batch, limit_steps=args.steps):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        cast_ctx = torch.autocast(device_type=device.type, dtype=dtype) if use_autocast else nullcontext()
        with cast_ctx:
            logits = model(x)  # [B,T,V]
            # Memory-capped loss:
            loss = cross_entropy_chunked(logits, y, chunk_tokens=args.loss_chunk_toks)

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        step += 1
        running_loss += loss.item()

        if step % 100 == 0 or step == 1:
            dt = time.time() - t0
            avg = running_loss / (100 if step % 100 == 0 else step)
            toks_per_step = args.batch * args.seq_len
            toks_per_s = (100 * toks_per_step) / dt if step % 100 == 0 else (step * toks_per_step) / dt
            print(f"[{step}/{args.steps}] loss={avg:.4f} tok/s={toks_per_s:,.0f} device={device.type} amp={dtype}")
            if step % 100 == 0:
                running_loss, t0 = 0.0, time.time()

        if step >= args.steps:
            break

    torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, args.snapshot)
    print(f"Saved microformer to: {args.snapshot}")


if __name__ == "__main__":
    main()
