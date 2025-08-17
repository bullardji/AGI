# agi/graph/connector.py
from __future__ import annotations
import os
import torch
from typing import Dict, List, Tuple, Optional, Iterable, Set


class RelGraphView:
    """
    Adapter over your saved graph snapshot.
    Tries a few common checkpoint layouts:
      - {'adj': {int: [(int, float), ...]}}
      - {'edges': [(src, dst, weight)]} or (src, rel, dst, weight)
      - {'edge_index': LongTensor[2,E], 'edge_weight': FloatTensor[E]}
      - {'graph': <one of the above>} possibly inside {'state_dict':..., 'graph':...}
    Assumes nodes are token IDs (ints). If not, methods return empty results gracefully.
    """

    def __init__(self, adj: Dict[int, List[Tuple[int, float]]], num_nodes: Optional[int] = None):
        self.adj = adj
        self.num_nodes = num_nodes or (max(adj.keys()) + 1 if adj else None)
        for k in self.adj:
            self.adj[k].sort(key=lambda t: t[1], reverse=True)

    @staticmethod
    def from_checkpoint(path: str) -> "RelGraphView":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Graph snapshot not found: {path}")
        ckpt = torch.load(path, map_location="cpu")

        def _normalize_adj(adj_like) -> Dict[int, List[Tuple[int, float]]]:
            adj: Dict[int, List[Tuple[int, float]]] = {}
            for k, lst in adj_like.items():
                kk = int(k)
                out = []
                for t in lst:
                    if isinstance(t, (list, tuple)) and len(t) >= 2:
                        dst = int(t[0])
                        w = float(t[1]) if t[1] is not None else 1.0
                        out.append((dst, w))
                adj[kk] = out
            return adj

        gblob = ckpt.get("graph", ckpt) if isinstance(ckpt, dict) else ckpt

        if isinstance(gblob, dict) and "adj" in gblob and isinstance(gblob["adj"], dict):
            return RelGraphView(_normalize_adj(gblob["adj"]))

        if isinstance(gblob, dict) and "edges" in gblob:
            edges = gblob["edges"]
            adj: Dict[int, List[Tuple[int, float]]] = {}
            for e in edges:
                if len(e) == 3:
                    src, dst, w = e
                elif len(e) >= 4:
                    src, _rel, dst, w = e[0], e[1], e[2], e[3]
                else:
                    continue
                src, dst = int(src), int(dst)
                w = float(w) if w is not None else 1.0
                adj.setdefault(src, []).append((dst, w))
            return RelGraphView(adj)

        if isinstance(gblob, dict) and "edge_index" in gblob:
            ei = gblob["edge_index"]
            ew = gblob.get("edge_weight", None)
            if torch.is_tensor(ei): ei = ei.long().cpu()
            if torch.is_tensor(ew): ew = ew.float().cpu()
            adj: Dict[int, List[Tuple[int, float]]] = {}
            E = ei.shape[1]
            for j in range(E):
                src = int(ei[0, j]); dst = int(ei[1, j])
                w = float(ew[j].item()) if ew is not None else 1.0
                adj.setdefault(src, []).append((dst, w))
            return RelGraphView(adj)

        return RelGraphView(adj={})

    def neighbors(self, node_id: int, k: int = 16) -> List[Tuple[int, float]]:
        return self.adj.get(int(node_id), [])[:k]

    def neighbors_multi(self, nodes: Iterable[int], k: int = 32, exclude: Optional[Set[int]] = None) -> List[Tuple[int, float]]:
        exclude = exclude or set()
        scores: Dict[int, float] = {}
        for nid in nodes:
            for dst, w in self.adj.get(int(nid), []):
                if dst in exclude:
                    continue
                scores[dst] = scores.get(dst, 0.0) + float(w)
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return items[:k]
