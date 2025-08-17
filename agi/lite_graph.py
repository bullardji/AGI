
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import math
import pickle
import heapq

def _dd(v=float):
    return defaultdict(v)

class RelGraph:
    """
    Lightweight relational graph with K_rel distance buckets.
    Stores edges as adj[src][rel][dst] = weight.
    """
    def __init__(
        self,
        vocab_size: int,
        K_rel: int = 12,
        fanout: int = 64,
        decay: float = 0.997,
        and_k: int = 3,
        conj_cap: int = 0,
        eta_base: float = 1.0,
        eta_conj: float = 1.0,
        self_loop_penalty: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.V = int(vocab_size)
        self.K = int(K_rel)
        self.fanout = int(fanout)
        self.decay = float(decay)
        self.device = device

        # compatibility/knobs
        self.and_k = int(and_k)
        self.conj_cap = int(conj_cap)
        self.eta_base = float(eta_base)
        self.eta_conj = float(eta_conj)
        self.self_loop_penalty = float(self_loop_penalty)

        # data
        self.adj: Dict[int, Dict[int, Dict[int, float]]] = {}
        self.usage = [0.0] * self.V
        self.unigram = defaultdict(float)
        
        # sqrt cache for performance
        self._sqrt_cache = {}

    def _cached_usage_sqrt(self, token_id: int) -> float:
        """Get cached 1/sqrt(1 + usage) for frequent tokens."""
        usage_val = self.usage[token_id]
        if usage_val in self._sqrt_cache:
            return self._sqrt_cache[usage_val]
        
        result = 1.0 / math.sqrt(1.0 + usage_val)
        # Only cache for frequently used tokens to avoid memory bloat
        if usage_val > 10.0:  # Cache values for tokens used >10 times
            self._sqrt_cache[usage_val] = result
        return result

    def observe(self, u: int, v: int, back: int = 1, eta: float = 1.0) -> None:
        """Update edge (u --rel=back--> v) by eta; apply basic penalties, cap fanout, row-normalize."""
        if u < 0 or v < 0 or u >= self.V or v >= self.V:
            return
        rel = max(1, min(int(back), self.K)) - 1  # 0..K-1
        if u == v and self.self_loop_penalty:
            eta *= max(0.0, 1.0 - float(self.self_loop_penalty))

        row = self.adj.setdefault(u, {}).setdefault(rel, {})
        row[v] = row.get(v, 0.0) + float(eta)

        # Cap fanout and renormalize
        if len(row) > self.fanout:
            # keep top-k
            top = sorted(row.items(), key=lambda kv: kv[1], reverse=True)[: self.fanout]
            self.adj[u][rel] = row = dict(top)
        s = sum(row.values())
        if s > 0.0:
            for d in row:
                row[d] = row[d] / s

        # usage / unigram
        self.usage[u] += 1.0
        self.unigram[v] += 1.0

    # --- Compatibility helpers for corpus_mix trainer ---
    def learn_pair(self, u: int, v: int, delta: float = 1.0) -> None:
        """Map a bigram update to observe(back=1) with frequency-aware scaling."""
        eta = float(self.eta_base) * float(delta)
        # down-weight very frequent tokens on the fly (using cached sqrt)
        su = self._cached_usage_sqrt(u)
        sv = self._cached_usage_sqrt(v)
        eta *= (su * sv)
        if u == v and self.self_loop_penalty:
            eta *= max(0.0, 1.0 - float(self.self_loop_penalty))
        self.observe(int(u), int(v), back=1, eta=eta)

    def learn_conj(self, ctx_tail: List[int], next_id: int) -> None:
        """Record edges from trailing context tokens to the next token.
        Optimized implementation with cached sqrt and reduced function calls.
        """
        if not ctx_tail:
            return
        L = min(int(self.and_k or 0) or 0, len(ctx_tail))
        if L <= 0:
            L = len(ctx_tail)
            
        # Pre-compute values outside the loop
        next_id_int = int(next_id)
        base_eta = float(self.eta_conj)
        self_loop_penalty = float(self.self_loop_penalty) if self.self_loop_penalty else 0.0
        sv = self._cached_usage_sqrt(next_id_int)
        
        for back in range(1, L + 1):
            src = int(ctx_tail[-back])
            
            # Compute eta with optimizations
            if src == next_id_int and self_loop_penalty:
                eta = base_eta * max(0.0, 1.0 - self_loop_penalty)
            else:
                eta = base_eta
                
            # Use cached sqrt and pre-computed sv
            su = self._cached_usage_sqrt(src)
            self.observe(src, next_id_int, back=back, eta=eta * su * sv)

    def top_edges(self, k: int = 5) -> List[Tuple[int, int, int, float]]:
        """Return top-k edges by weight as (src, rel, dst, weight)."""
        items: List[Tuple[int, int, int, float]] = []
        for s, rels in self.adj.items():
            for r, rd in rels.items():
                for d, w in rd.items():
                    items.append((int(s), int(r), int(d), float(w)))
        items.sort(key=lambda t: t[3], reverse=True)
        return items[: int(k)]

    def reflect(self, steps: int = 1) -> None:
        """Decay and optional diffusion; prune tiny edges for stability."""
        for _ in range(int(steps)):
            # decay usage & weights
            self.usage = [u * self.decay for u in self.usage]
            for u, rels in list(self.adj.items()):
                for r, row in list(rels.items()):
                    # decay
                    for d in list(row.keys()):
                        row[d] *= self.decay
                        # prune tiny
                        if row[d] < 1e-8:
                            del row[d]
                    if not row:
                        del rels[r]
                        continue
                    # renormalize row-stochastic
                    s = sum(row.values())
                    if s > 0.0:
                        for d in row:
                            row[d] = row[d] / s
                if not rels:
                    del self.adj[u]

    def score_next(self, ctx: List[int], top_k: int = 32) -> List[Tuple[int,float]]:
        """Simple backoff across relation buckets to score next-token candidates."""
        cand = defaultdict(float)
        L = len(ctx)
        if L == 0:
            return []
        # accumulate from up to K buckets
        for back in range(1, min(self.K, L) + 1):
            u = ctx[-back]
            rel = back - 1
            row = self.adj.get(u, {}).get(rel, None)
            if not row:
                continue
            # inverse distance bias
            bias = 1.0 / (1.0 + 0.25 * (back - 1))
            for d, w in row.items():
                cand[d] += bias * w
        # top-k
        items = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return items

    def save(self, path: str) -> None:
        state = {
            "V": self.V, "K": self.K, "fanout": self.fanout, "decay": self.decay,
            "adj": self.adj, "usage": self.usage, "unigram": dict(self.unigram),
            "and_k": self.and_k, "eta_base": self.eta_base,
            "eta_conj": self.eta_conj, "self_loop_penalty": self.self_loop_penalty,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(path: str) -> "RelGraph":
        with open(path, "rb") as f:
            state = pickle.load(f)
        g = RelGraph(
            vocab_size=state.get("V", 0),
            K_rel=state.get("K", 12),
            fanout=state.get("fanout", 64),
            decay=state.get("decay", 0.997),
        )
        g.adj = state.get("adj", {})
        g.usage = state.get("usage", [0.0] * g.V)
        ug2 = state.get("unigram", {})
        g.unigram = _dd(float)
        for k, v in ug2.items():
            g.unigram[k] = float(v)

        # restore optional attrs
        g.and_k = int(state.get("and_k", getattr(g, "and_k", 3)))
        g.eta_base = float(state.get("eta_base", getattr(g, "eta_base", 1.0)))
        g.eta_conj = float(state.get("eta_conj", getattr(g, "eta_conj", 1.0)))
        g.self_loop_penalty = float(state.get("self_loop_penalty", getattr(g, "self_loop_penalty", 0.0)))
        return g
