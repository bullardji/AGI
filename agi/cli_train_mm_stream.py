
"""
Multimodal streaming trainer (text + images + audio), RAM bounded.
Default: CIFAR-10 (images), Google Speech Commands (audio).

NOTE: If you have a file named 'speech_commands.py' in your current directory,
datasets will treat it as a local dataset script and error:
"Dataset scripts are no longer supported, but found speech_commands.py".
Rename or remove that local file, or change --audio-ds to something else.
"""
import argparse, os, random, time
from datasets import load_dataset
from agi.common import load_tokenizer
import torch
from PIL import Image

from .lite_graph import RelationalGraph, GraphConfig
from .mm_utils import read_wav_simple, audio_to_codes, image_to_codes

def _mem_gb():
    try:
        import psutil, os as _os
        p = psutil.Process(_os.getpid())
        return p.memory_info().rss / (1024**3)
    except Exception:
        return float('nan')

def stream_text(ds):
    for ex in ds:
        t = ex.get("text", None)
        if t: yield t

def stream_images(ds):
    for ex in ds:
        img = ex.get("img") or ex.get("image") or ex.get("image_file_path")
        if img is None: continue
        if hasattr(img, "convert"):
            yield img
        else:
            try:
                yield Image.open(img)
            except Exception:
                continue

def stream_audio_paths(ds):
    for ex in ds:
        aud = ex.get("audio")
        if isinstance(aud, dict):
            path = aud.get("path") or aud.get("filepath") or aud.get("file")
            if path: yield path
        for k in ["file", "filename", "path", "audio_file"]:
            if k in ex:
                yield ex[k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--pairs', type=int, default=500000)
    ap.add_argument('--Krel', type=int, default=16)
    ap.add_argument('--fanout', type=int, default=40)
    ap.add_argument('--fp16', action='store_true')
    # text
    ap.add_argument('--text-ds', default='roneneldan/TinyStories')
    ap.add_argument('--text-split', default='train')
    ap.add_argument('--text-weight', type=float, default=0.6)
    # images
    ap.add_argument('--img-ds', default='cifar10')
    ap.add_argument('--img-split', default='train')
    ap.add_argument('--img-weight', type=float, default=0.25)
    ap.add_argument('--img-codes', type=int, default=512)
    # audio
    ap.add_argument('--audio-ds', default='speech_commands')
    ap.add_argument('--audio-split', default='train')
    ap.add_argument('--audio-weight', type=float, default=0.15)
    ap.add_argument('--aud-codes', type=int, default=512)
    # bridge
    ap.add_argument('--bridge-prob', type=float, default=0.3)
    # reflect
    ap.add_argument('--reflect-every', type=int, default=100000)
    ap.add_argument('--reflect-steps', type=int, default=128)
    ap.add_argument('--progress-every', type=int, default=10000)
    ap.add_argument('--snapshot', required=True)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    V_text = tok.vocab_size
    dtype = "float16" if args.fp16 else "float32"
    g = RelationalGraph(GraphConfig(vocab_size=V_text + args.img_codes + args.aud_codes, K_rel=args.Krel, fanout_cap=args.fanout, dtype_weights=dtype))

    # offsets
    OFF_IMG = V_text
    OFF_AUD = V_text + args.img_codes

    # load streams (streaming=True)
    txt = stream_text(load_dataset(args.text_ds, split=args.text_split, streaming=True))
    img = stream_images(load_dataset(args.img_ds, split=args.img_split, streaming=True))
    try:
        aud_paths = stream_audio_paths(load_dataset(args.audio_ds, split=args.audio_split, streaming=True))
    except Exception as e:
        print("Audio dataset error:", e)
        print("If you have a local 'speech_commands.py', please rename/move it. Fallback: no audio.")
        aud_paths = iter(())

    learned = 0
    t0 = time.time()
    it_txt = iter(txt); it_img = iter(img); it_aud = iter(aud_paths)
    weights = [args.text_weight, args.img_weight, args.audio_weight]
    total_w = sum(w for w in weights if w>0)
    norm_w = [w/total_w for w in weights]

    def learn_ids(ids):
        nonlocal learned
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
                if learned >= args.pairs:
                    return True
            last = t
        return False

    while learned < args.pairs:
        r = random.random()
        cum = 0.0; pick = 0
        for i,w in enumerate(norm_w):
            cum += w
            if r <= cum:
                pick = i; break
        if pick == 0:  # text
            try:
                t = next(it_txt)
            except StopIteration:
                norm_w[0] = 0.0; total_w = sum(norm_w); norm_w = [w/total_w if total_w>0 else 0 for w in norm_w]; 
                if total_w <= 0: break
                continue
            ids = tok.encode(t).ids
            if learn_ids(ids): break
        elif pick == 1:  # image
            try:
                im = next(it_img)
            except StopIteration:
                norm_w[1] = 0.0; total_w = sum(norm_w); norm_w = [w/total_w if total_w>0 else 0 for w in norm_w]; 
                if total_w <= 0: break
                continue
            codes = image_to_codes(im, codebook_size=args.img_codes)
            ids = [OFF_IMG + c for c in codes]
            if learn_ids(ids): break
        else:  # audio
            try:
                p = next(it_aud)
            except StopIteration:
                norm_w[2] = 0.0; total_w = sum(norm_w); norm_w = [w/total_w if total_w>0 else 0 for w in norm_w]; 
                if total_w <= 0: break
                continue
            try:
                wav, sr = read_wav_simple(p)
                codes = audio_to_codes(wav, sr, codebook_size=args.aud_codes)
            except Exception:
                continue
            ids = [OFF_AUD + c for c in codes]
            if learn_ids(ids): break

    g.reflect(steps=args.reflect_steps)
    os.makedirs(os.path.dirname(args.snapshot) or ".", exist_ok=True)
    torch.save(g.to_state(), args.snapshot)
    print(f"Saved: {args.snapshot}")

if __name__ == '__main__':
    main()
