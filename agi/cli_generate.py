
import argparse
from .lite_graph import RelGraph
from .common import load_tokenizer
from .decoding_utils import decode_tokens, decode_with_tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    g = RelGraph.load(args.snapshot)
    ids = tok.encode(args.prompt).ids
    out = decode_tokens(g, tok, ids, max_new_tokens=args.max_new,
                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print(out if isinstance(out, str) else decode_with_tokenizer(tok, out))

if __name__ == "__main__":
    main()
