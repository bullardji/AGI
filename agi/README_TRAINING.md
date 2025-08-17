
# Training & Usage Cheatsheet

## 1) 256k BPE
```bash
python -m agi_graph_snn_pro_ctx256k_cli_v6_full.bpe_train \
  --dataset roneneldan/TinyStories --split train \
  --limit 200000 \
  --vocab 262144 \
  --out snapshots/bpe_256k.json
```

## 2) Single dataset text training
```bash
python -m agi_graph_snn_pro_ctx256k_cli_v6_full.cli_train_text_full \
  --tokenizer snapshots/bpe_256k.json \
  --dataset HuggingFaceFW/fineweb-edu --split train \
  --pairs 2000000 --Krel 18 --fanout 48 \
  --reflect-every 250000 --reflect-steps 128 \
  --snapshot snapshots/relgraph_text_fw.pt
```

## 3) Multi-corpus mix (default: FineWeb-EDU + Wikipedia + OWT)
```bash
python -m agi_graph_snn_pro_ctx256k_cli_v6_full.cli_train_corpus_mix \
  --tokenizer snapshots/bpe_256k.json \
  --pairs 2000000 --Krel 18 --fanout 48 \
  --mix '[
      {"path":"HuggingFaceFW/fineweb-edu","split":"train","weight":0.6},
      {"path":"wikimedia/wikipedia","split":"20231101.en","weight":0.2},
      {"path":"Skylion007/openwebtext","split":"train","weight":0.2}
  ]' \
  --reflect-every 250000 --reflect-steps 128 \
  --snapshot snapshots/relgraph_corpus_mix.pt
```

## 4) Multimodal (text + CIFAR10 + Speech Commands)
```bash
python -m agi_graph_snn_pro_ctx256k_cli_v6_full.cli_train_mm_stream \
  --tokenizer snapshots/bpe_256k.json \
  --text-ds HuggingFaceFW/fineweb-edu --text-split train --text-weight 0.6 \
  --img-ds cifar10 --img-split train --img-weight 0.25 \
  --audio-ds speech_commands --audio-split train --audio-weight 0.15 \
  --pairs 500000 --Krel 16 --fanout 40 \
  --img-codes 512 --aud-codes 512 \
  --bridge-prob 0.3 \
  --reflect-every 100000 --reflect-steps 128 \
  --snapshot snapshots/relgraph_mm.pt
```
**Note:** if a local `speech_commands.py` file exists in your working directory, `datasets` will refuse to load. Rename or move it.

## 5) Chat (256k context + anti-repeat + RoPE-like bias)
```bash
python -m agi_graph_snn_pro_ctx256k_cli_v6_full.cli_chat \
  --snapshot snapshots/relgraph_text_fw.pt \
  --tokenizer snapshots/bpe_256k.json \
  --device cpu --interactive \
  --system "You are a helpful, concise assistant." \
  --top-k 16 --min-new 16 \
  --ctx-len 262144 --ctx-alpha 0.7 --ctx-decay 0.9997 --ctx-fanout 32 \
  --rope-base 10000 --rope-scale 1.0 \
  --no-repeat-ngram 4 --max-run 6 --freq-penalty 0.2 --self-loop-penalty 0.7
```

## 6) One-shot generation
```bash
python -m agi_graph_snn_pro_ctx256k_cli_v6_full.cli_generate \
  --snapshot snapshots/relgraph_text_fw.pt \
  --tokenizer snapshots/bpe_256k.json \
  --device cpu \
  --prompt "Write a short story about a friendly dog." \
  --max-new 200 --top-k 16 --temp 1.0 --selfteach 1 \
  --ctx-len 262144 --ctx-alpha 0.7 --ctx-fanout 32 \
  --rope-base 10000 --rope-scale 1.0 \
  --no-repeat-ngram 4 --max-run 6 --freq-penalty 0.2 --self-loop-penalty 0.7 --min-new 16
```
