
import argparse
from agi.lite_graph import RelGraph

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    g = RelGraph.load(args.snapshot)
    tops = g.top_edges(args.top)
    print("Top edges (src,rel,dst,weight):")
    for t in tops:
        print(t)
    print("K_rel:", g.K, "fanout:", g.fanout, "decay:", g.decay)

if __name__ == "__main__":
    main()
