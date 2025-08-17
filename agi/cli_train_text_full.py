
"""
Larger text training with streaming datasets and periodic reflection/snapshots.
Keeps RAM bounded (O(V*K*fanout)), suitable for CPU.

Example:
python -m agi_graph_snn_pro_ctx256k_cli_v6_full.cli_train_text_full \
  --tokenizer snapshots/bpe_256k.json \
  --dataset HuggingFaceFW/fineweb-edu --split train \
  --pairs 2000000 --Krel 18 --fanout 48 \
  --reflect-every 250000 --reflect-steps 128 \
  --snapshot snapshots/relgraph_text_fw.pt
"""
import argparse, time, os
from agi.common import load_tokenizer
from datasets import load_dataset
import torch
from .lite_graph import RelationalGraph, GraphConfig

def _mem_gb():
    try:
        import psutil, os as _os
        p = psutil.Process(_os.getpid())
        return p.memory_info().rss / (1024**3)
    except Exception:
        return float('nan')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--dataset', required=True, help='HF dataset path')
    ap.add_argument('--split', default='train')
    ap.add_argument('--pairs', type=int, default=500000)
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda','mps'])
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--Krel', type=int, default=12)
    ap.add_argument('--fanout', type=int, default=32)
    ap.add_argument('--reflect-every', type=int, default=100000)
    ap.add_argument('--reflect-steps', type=int, default=128)
    ap.add_argument('--progress-every', type=int, default=10000)
    ap.add_argument('--snapshot', required=True)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    V = tok.vocab_size
    dtype = "float16" if args.fp16 else "float32"
    cfg = GraphConfig(vocab_size=V, K_rel=args.Krel, fanout_cap=args.fanout, dtype_weights=dtype,
                      w_max=1.0, w_prune=0.01, decay=0.995, A_plus=0.05, A_minus=0.03,
                      tau_plus=30.0, tau_minus=30.0, vote_gain=0.03, vote_wta=0.98)
    g = RelationalGraph(cfg)

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    learned = 0
    t0 = time.time()
    for ex in ds:
        if learned >= args.pairs: break
        text = ex.get("text", None)
        if not text: continue
        ids = tok.encode(text).ids
        last = None
        for t in ids:
            if last is not None and last != t:
                g.learn_pair(int(last), int(t))
                learned += 1
                if learned % args.progress_every == 0:
                    dt = time.time() - t0
                    rate = learned / max(1e-6, dt)
                    print(f"  learned {learned}/{args.pairs} pairs | rss~{_mem_gb():.2f} GB | {rate:.1f} pairs/s")
                if learned % args.reflect_every == 0:
                    print(f"Reflecting ({args.reflect_steps} steps)...")
                    g.reflect(steps=args.reflect_steps)
                if learned >= args.pairs: break
            last = t

    g.reflect(steps=args.reflect_steps)
    os.makedirs(os.path.dirname(args.snapshot) or ".", exist_ok=True)
    torch.save(g.to_state(), args.snapshot)
    print(f"Saved: {args.snapshot}")

if __name__ == '__main__':
    main()
