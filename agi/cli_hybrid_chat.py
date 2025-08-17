
import argparse, sys, torch
from agi.common import load_tokenizer, get_special_ids
from agi.lite_graph import RelGraph
from agi.decoding_utils import apply_repetition_controls, top_k_top_p_filter, sample_from_distribution, decode_with_tokenizer
from agi.nn.microformer import MicroTransformer, MicroConfig

def load_microformer(path, vocab_size):
    ckpt = torch.load(path, map_location="cpu")
    cfgd = ckpt.get("config", {})
    cfgd["vocab_size"] = vocab_size  # ensure match
    cfg = MicroConfig(**cfgd)
    model = MicroTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="RelGraph snapshot")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--nn", required=True, help="microformer .pt path")
    ap.add_argument("--alpha", type=float, default=1.0, help="weight for graph score")
    ap.add_argument("--beta", type=float, default=0.5, help="weight for nn logits")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--min-new", type=int, default=32)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    g = RelGraph.load(args.snapshot)
    nn_model = load_microformer(args.nn, tok.vocab_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nn_model.to(device)

    sp = get_special_ids(tok)
    BOS = sp.get("[BOS]"); EOS = sp.get("[EOS]")

    def generate(prompt_ids, min_new):
        ctx = list(prompt_ids)
        recent = list(prompt_ids)[-64:]
        for _ in range(min_new):
            # Graph proposal
            scored = g.score_next(ctx, top_k=max(8, args.top_k*2))
            logits = {tid: float(score) for tid, score in scored}
            # NN residual
            x = torch.tensor([ctx[-nn_model.cfg.max_seq_len:]], dtype=torch.long, device=device)
            with torch.no_grad():
                nn_logits = nn_model(x)[0, -1]  # (V,)
            # Combine
            for tid in range(nn_logits.size(0)):
                if tid in logits:
                    logits[tid] = args.alpha * logits[tid] + args.beta * float(nn_logits[tid])
                else:
                    # allow NN to propose new tokens weakly
                    logits[tid] = args.beta * float(nn_logits[tid]) * 0.1

            # repetition controls
            logits = apply_repetition_controls(logits, recent)
            # sample
            dist = top_k_top_p_filter(logits, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)
            if not dist:
                break
            tid = sample_from_distribution(dist)
            if tid is None or (EOS is not None and tid == EOS):
                break
            ctx.append(tid)
            recent.append(tid); recent = recent[-128:]
        return ctx

    if args.interactive:
        print("Hybrid chat ready. Type '/help' for commands. Ctrl+C to quit.")
        while True:
            try:
                line = input("> ")
            except KeyboardInterrupt:
                print("\nBye.")
                return
            if not line.strip():
                continue
            if line.strip() == "/help":
                print("Enter text and press Enter. '/quit' to exit.")
                continue
            if line.strip() == "/quit":
                print("Bye.")
                return
            prompt_ids = tok.encode(line).ids
            if BOS is not None:
                prompt_ids = [BOS] + prompt_ids
            out_ids = generate(prompt_ids, args.min_new)
            text = decode_with_tokenizer(tok, out_ids)
            print(text)
    else:
        data = sys.stdin.read()
        prompt_ids = tok.encode(data).ids
        if BOS is not None:
            prompt_ids = [BOS] + prompt_ids
        out_ids = generate(prompt_ids, args.min_new)
        sys.stdout.write(decode_with_tokenizer(tok, out_ids))

if __name__ == "__main__":
    main()
