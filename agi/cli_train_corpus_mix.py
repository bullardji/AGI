
import argparse, json, random, math, time
from datasets import load_dataset
from agi.common import load_tokenizer, get_special_ids
from agi.lite_graph import RelGraph

def iter_weighted_mix(mix_spec, tokenizer, limit_pairs=None, seed=1234):
    """
    mix_spec: list of dicts {path, split, weight, config?}
    Yields token sequences; ensures approximate weighting.
    """
    rnd = random.Random(seed)
    # Normalize weights
    total_w = sum(item["weight"] for item in mix_spec)
    streams = []
    for item in mix_spec:
        w = item["weight"] / total_w if total_w > 0 else 1.0/len(mix_spec)
        ds = (load_dataset(item["path"], item.get("config"), split=item["split"], streaming=True)
              if item.get("config") else
              load_dataset(item["path"], split=item["split"], streaming=True))
        streams.append((w, iter(ds)))
    # Round-robin with probability proportional to weight
    seen = 0
    specials = {"[NL]": tokenizer.token_to_id("[NL]")}
    while True:
        # pick a stream
        r = rnd.random()
        acc = 0.0
        picked = None
        for w, it in streams:
            acc += w
            if r <= acc:
                picked = it
                break
        if picked is None:
            picked = streams[-1][1]
        try:
            row = next(picked)
        except StopIteration:
            # restart stream if ended (for streaming datasets this is rare)
            continue
        text = row.get("text") or row.get("content") or row.get("article") or row.get("content_text") or ""
        if isinstance(text, str):
            text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\n', ' [NL] ')
        if not isinstance(text, str) or not text.strip():
            continue
        # The tokenizer itself contains mappings for [NL], etc. if present in text.
        ids = tokenizer.encode(text).ids
        if not ids:
            continue
        yield ids
        seen += len(ids)
        if limit_pairs and seen >= limit_pairs:
            break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--pairs", type=int, default=2_000_000)
    ap.add_argument("--Krel", type=int, default=18)
    ap.add_argument("--fanout", type=int, default=64)
    ap.add_argument("--mix", type=str, required=True, help="JSON list of dataset specs with weights")
    ap.add_argument("--reflect-every", type=int, default=250_000)
    ap.add_argument("--reflect-steps", type=int, default=128)
    ap.add_argument("--snapshot", type=str, required=True)
    ap.add_argument("--and_k", type=int, default=3)
    ap.add_argument("--eta_base", type=float, default=0.05)
    ap.add_argument("--eta_conj", type=float, default=0.08)
    ap.add_argument("--self_loop_penalty", type=float, default=0.2)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    spec_ids = get_special_ids(tok)

    g = RelGraph(vocab_size=tok.vocab_size, K_rel=args.Krel, fanout=args.fanout, decay=0.997,
                 and_k=args.and_k, eta_base=args.eta_base, eta_conj=args.eta_conj,
                 self_loop_penalty=args.self_loop_penalty)

    mix = json.loads(args.mix)
    seen_pairs = 0
    last_reflect = 0
    t0 = time.time()

    BOS = spec_ids.get("[BOS]")
    EOS = spec_ids.get("[EOS]")

    for ids in iter_weighted_mix(mix, tok, limit_pairs=args.pairs):
        # anchors
        if BOS is not None and len(ids) > 0:
            g.learn_pair(BOS, ids[0], delta=0.5)
        # bigrams + simple conjunction updates
        for i in range(len(ids) - 1):
            u, v = ids[i], ids[i+1]
            g.learn_pair(u, v, delta=1.0)
            ctx_tail = ids[max(0, i-args.and_k+1):i+1]
            if ctx_tail:
                g.learn_conj(ctx_tail, v)
        if EOS is not None and len(ids) > 0:
            g.learn_pair(ids[-1], EOS, delta=0.5)

        seen_pairs += max(0, len(ids) - 1)
        if seen_pairs - last_reflect >= args.reflect_every:
            g.reflect(args.reflect_steps)
            last_reflect = seen_pairs
            # graph statistics and top edges
            total_nodes = len(g.adj)
            total_edges = sum(len(rd) for rels in g.adj.values() for rd in rels.values())
            tops = g.top_edges(5)
            print(f"[reflect] seen_pairs={seen_pairs} nodes={total_nodes} edges={total_edges} tops={tops} dt={time.time()-t0:.1f}s")

    g.save(args.snapshot)
    print(f"Saved snapshot to: {args.snapshot}")

if __name__ == "__main__":
    main()
