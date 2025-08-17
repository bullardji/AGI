import argparse, os, re, sys, shutil
from datasets import load_dataset

# Hugging Face tokenizer loader
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

# Kept for the legacy "train from scratch" path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders

URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
LONGNUM_RE = re.compile(r'\d{4,}')

def normalize_text(s: str):
    # Light, task-agnostic mapping (used only for legacy training)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = URL_RE.sub("[URL]", s)
    s = EMAIL_RE.sub("[EMAIL]", s)
    s = LONGNUM_RE.sub("[NUM]", s)
    s = s.replace("\n", " [NL] ")
    return s

def export_pretrained(args):
    if AutoTokenizer is None:
        sys.exit("transformers not installed. Run: pip install -U transformers huggingface_hub")

    tok = AutoTokenizer.from_pretrained(
        args.from_pretrained,
        use_fast=True,
        trust_remote_code=args.trust_remote_code
    )

    # Optional conveniences
    if args.pad_eos and tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.add_extras:
        extras = ["[NL]", "[URL]", "[EMAIL]", "[NUM]"]
        tok.add_special_tokens({"additional_special_tokens": extras})

    # Save either to a directory (preferred) or a single tokenizer.json
    out = args.out
    if out.endswith(".json"):
        # Try saving the fast tokenizer core directly to a JSON path
        saved_json = False
        for attr in ("backend_tokenizer", "_tokenizer"):
            core = getattr(tok, attr, None)
            if core is not None:
                try:
                    core.save(out)
                    saved_json = True
                    break
                except Exception:
                    pass

        if not saved_json:
            # Fallback: save to a temp dir then copy tokenizer.json out
            tmpdir = out + ".tmp_tok"
            if os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir)
            os.makedirs(tmpdir, exist_ok=True)
            tok.save_pretrained(tmpdir)
            src_json = os.path.join(tmpdir, "tokenizer.json")
            if not os.path.exists(src_json):
                shutil.rmtree(tmpdir, ignore_errors=True)
                sys.exit("Could not find tokenizer.json in saved assets.")
            shutil.copyfile(src_json, out)
            shutil.rmtree(tmpdir, ignore_errors=True)

        print(f"Saved tokenizer JSON to: {out}")
        print("(Tip: pass a directory path to --out if you also want special_tokens_map.json & tokenizer_config.json)")
    else:
        os.makedirs(out, exist_ok=True)
        tok.save_pretrained(out)
        print(f"Saved tokenizer to directory: {out}")
        j = os.path.join(out, "tokenizer.json")
        if os.path.exists(j):
            print(f"  tokenizer.json: {j}")

    print(f"vocab size: {len(tok)}")
    print(f"specials: bos={tok.bos_token_id} eos={tok.eos_token_id} pad={tok.pad_token_id}")

def train_from_scratch(args):
    tok = Tokenizer(models.BPE(unk_token="[UNK]", byte_fallback=True))
    tok.normalizer = normalizers.NFC()

    if args.space_style == "byte":
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel(add_prefix_space=False)
        initial_alpha = pre_tokenizers.ByteLevel.alphabet()
    else:
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁")
        tok.decoder = decoders.Metaspace(replacement="▁")
        initial_alpha = None

    trainer_kwargs = {
        "vocab_size": int(args.vocab),
        "min_frequency": int(args.min_frequency),
        "show_progress": True,
        "special_tokens": ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[NL]", "[URL]", "[EMAIL]", "[NUM]"],
    }
    if initial_alpha is not None:
        trainer_kwargs["initial_alphabet"] = initial_alpha
    trainer = trainers.BpeTrainer(**trainer_kwargs)

    def stream():
        ds = (load_dataset(args.dataset, args.config, split=args.split, streaming=True)
              if args.config else
              load_dataset(args.dataset, split=args.split, streaming=True))
        n = 0
        for rec in ds:
            txt = rec.get("text") or rec.get("content") or rec.get("article") or rec.get("content_text") or ""
            if isinstance(txt, str) and txt.strip():
                yield normalize_text(txt)
                n += 1
                if args.limit and n >= args.limit:
                    break

    tok.train_from_iterator(stream(), trainer=trainer, length=args.limit)

    # Save: if --out endswith .json, write single JSON; else treat as directory
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if out.endswith(".json"):
        tok.save(out)
        print(f"Saved tokenizer to: {out}")
    else:
        os.makedirs(out, exist_ok=True)
        # Save as a Hugging Face-style directory (tokenizer.json)
        # Convert to HF fast tokenizer JSON
        tmp_json = os.path.join(out, "tokenizer.json")
        tok.save(tmp_json)
        print(f"Saved tokenizer directory to: {out}")
        print(f"  tokenizer.json: {tmp_json}")

def main():
    ap = argparse.ArgumentParser()
    # NEW: pretrained path
    ap.add_argument("--from-pretrained", type=str,
                    help="HF repo id or local tokenizer dir to export (e.g. meta-llama/Llama-3.1-8B). If set, training args are ignored.")
    ap.add_argument("--pad-eos", action="store_true",
                    help="When exporting a pretrained tokenizer, set pad_token = eos_token if PAD is undefined.")
    ap.add_argument("--add-extras", action="store_true",
                    help="When exporting a pretrained tokenizer, add [NL],[URL],[EMAIL],[NUM] as additional specials.")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="Pass through to AutoTokenizer.from_pretrained for custom tokenizers.")

    # Legacy training args (still supported)
    ap.add_argument("--dataset")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--vocab", type=int, default=79_000)
    ap.add_argument("--space-style", choices=["byte", "meta"], default="meta",
                   help="Whitespace handling for *training*: byte-level (Ġ) or metaspace (▁).")
    ap.add_argument("--min-frequency", type=int, default=2)

    # Output (required for both modes)
    ap.add_argument("--out", required=True,
                    help="Output path. If ends with .json, writes a single tokenizer.json file; "
                         "otherwise saves a directory with tokenizer assets.")
    args = ap.parse_args()

    if args.from_pretrained:
        export_pretrained(args)
    else:
        # Backwards-compat: require dataset only if we're actually training
        if not args.dataset:
            sys.exit("Either set --from-pretrained, or provide --dataset to train a tokenizer.")
        train_from_scratch(args)

if __name__ == "__main__":
    main()
