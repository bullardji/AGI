
import argparse, sys
from agi.common import load_tokenizer, get_special_ids
from agi.lite_graph import RelGraph
from agi.decoding_utils import apply_repetition_controls, top_k_top_p_filter, sample_from_distribution, decode_with_tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--min-new", type=int, default=32)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    g = RelGraph.load(args.snapshot)
    sp = get_special_ids(tok)

    BOS = sp.get("[BOS]")
    EOS = sp.get("[EOS]")

    def generate(prompt_ids, min_new):
        ctx = list(prompt_ids)
        recent = list(prompt_ids)[-64:]
        for _ in range(min_new):
            # get graph scores
            scored = g.score_next(ctx, top_k=max(8, args.top_k*2))
            logits = {tid: float(score) for tid, score in scored}
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
            recent.append(tid)
            recent = recent[-128:]
        return ctx

    if args.interactive:
        print("Chat ready. Type '/help' for commands. Ctrl+C to quit.")
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
