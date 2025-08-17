
from __future__ import annotations
import numpy as np, io, wave
from PIL import Image

def read_wav_simple(path):
    """Pure-Python PCM WAV reader to avoid external libs. Returns float32 mono @ native sr."""
    with wave.open(path, 'rb') as wf:
        nchan = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sampwidth != 2:
        dtype = np.int16
    else:
        dtype = np.int16
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if nchan > 1:
        data = data.reshape(-1, nchan).mean(axis=1)
    data /= 32768.0
    return data, fr

def audio_to_codes(x, sr, n_mels=40, frame_ms=25, hop_ms=10, codebook_size=512, seed=13):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return []
    n_fft = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if n_fft < 16: n_fft = 256
    if hop < 8: hop = n_fft // 2
    win = np.hanning(n_fft).astype(np.float32)
    frames = []
    for i in range(0, len(x)-n_fft, hop):
        seg = x[i:i+n_fft] * win
        fft = np.fft.rfft(seg, n=n_fft)
        mag = (np.abs(fft) + 1e-6).astype(np.float32)
        frames.append(mag)
    if not frames:
        return []
    S = np.stack(frames, axis=0)  # [T, F]
    F = S.shape[1]
    m = n_mels
    mel = np.zeros((S.shape[0], m), dtype=np.float32)
    bins = np.linspace(0, F-1, m+1).astype(int)
    for i in range(m):
        mel[:, i] = S[:, bins[i]:bins[i+1]].mean(axis=1)
    mel = np.log1p(mel)
    rng = np.random.RandomState(seed)
    W = rng.randn(m, int(np.ceil(np.log2(codebook_size)))).astype(np.float32)
    proj = (mel @ W) > 0.0
    codes = []
    for row in proj:
        v = 0
        for bit in row.astype(np.uint8):
            v = (v << 1) | int(bit)
        codes.append(int(v % codebook_size))
    return codes

def image_to_codes(pil_img, side=64, patch=8, codebook_size=512, seed=7):
    img = pil_img.convert("L").resize((side, side))
    x = np.array(img, dtype=np.float32) / 255.0  # [H,W]
    H, W = x.shape
    ph = pw = patch
    patches = []
    for y in range(0, H, ph):
        for x0 in range(0, W, pw):
            p = img.crop((x0, y, x0+pw, y+ph))
            patches.append(np.array(p, dtype=np.float32).reshape(-1) / 255.0)
    P = np.stack(patches, axis=0)  # [N, ph*pw]
    rng = np.random.RandomState(seed)
    W = rng.randn(P.shape[1], int(np.ceil(np.log2(codebook_size)))).astype(np.float32)
    proj = (P @ W) > 0.0
    codes = []
    for row in proj:
        v = 0
        for bit in row.astype(np.uint8):
            v = (v << 1) | int(bit)
        codes.append(int(v % codebook_size))
    return codes
