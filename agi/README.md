
# AGI v2 — Relation Graph with Conjunctive Decoder

**What's new:**
- Higher-order **conjunctive nodes** (hashed n-gram state) -> better local coherence
- BOS/EOS handling to avoid cross-doc junk
- Self-loop penalties and stronger anti-repeat sampling
- **Resume training** support; streaming, RAM-bounded

## Train BPE (256k)
```bash
PYTHONPATH="$PWD/agi_v2" python -m agi_v2.bpe_train \
  --dataset roneneldan/TinyStories --split train \
  --limit 200000 --vocab 262144 \
  --out snapshots/bpe_256k.json
```

## Train on mixture (fixed & improved)
```bash
PYTHONPATH="$PWD/agi_v2" python -m agi_v2.cli_train_corpus_mix \
  --tokenizer snapshots/bpe_256k.json \
  --pairs 2000000 --Krel 18 --fanout 48 \
  --and-k 3 --conj-cap 500000 --eta-base 0.05 --eta-conj 0.08 \
  --mix '[
      {"path":"HuggingFaceFW/fineweb-edu","split":"train","weight":0.6},
      {"path":"wikimedia/wikipedia","config":"20231101.en","split":"train","weight":0.2},
      {"path":"Geralt-Targaryen/openwebtext2","split":"train","weight":0.2}
  ]' \
  --reflect-every 250000 --reflect-steps 128 \
  --snapshot snapshots/relgraph_corpus_mix.pt
```

Resume later:
```bash
--resume snapshots/relgraph_corpus_mix.pt --pairs 20000000
```

## Chat
```bash
PYTHONPATH="$PWD/agi_v2" python -m agi_v2.cli_chat \
  --snapshot snapshots/relgraph_corpus_mix.pt \
  --tokenizer snapshots/bpe_256k.json \
  --interactive --top-k 24 --min-new 48 --and-k 3 --alpha-conj 1.2
```
