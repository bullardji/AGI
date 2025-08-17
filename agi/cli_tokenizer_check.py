
import argparse
from agi.common import load_tokenizer, get_special_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)

    print("vocab size:", tok.vocab_size)
    specials = get_special_ids(tok)
    print("special ids:", specials)

    sample = "Hello [URL] world [NL] Done."
    ids = tok.encode(sample, add_special_tokens=False).ids
    dec = tok.decode(ids, skip_special_tokens=True)
    print("encode length:", len(ids))
    print("round-trip ok:", dec == sample)

if __name__ == "__main__":
    main()
