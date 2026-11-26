"""
Print one table of all finished student runs, averaged over seeds (mean ± std).

Usage
  python print_results.py                       # CUB, final phase
  python print_results.py --dataset aircraft
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

COLS = [("top1", "top-1"), ("mean_per_class", "mean/cls"), ("inheritance", "inherit"),
        ("flagged_pair_inheritance", "flagged"), ("acc_teacher_correct", "t-correct"), ("ece", "ECE")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cub")
    ap.add_argument("--phase", default="final")
    ap.add_argument("--root", default="runs")
    args = ap.parse_args()

    groups = defaultdict(list)
    for f in glob.glob(os.path.join(args.root, args.dataset, args.phase, "students", "*", "results.json")):
        name = os.path.basename(os.path.dirname(f))
        groups[re.sub(r"_s\d+$", "", name)].append(json.load(open(f)))
    if not groups:
        print("No finished runs found.")
        return

    order = ["ce", "kd", "dkd", "wsl", "ka", "pergate", "class_trust", "cpt", "cpt_dkd"]
    def key(g):
        m = groups[g][0]["method"]
        return (order.index(m) if m in order else 99, g)

    print(f"{'run':40s} {'seeds':>5s} " + " ".join(f"{h:>12s}" for _, h in COLS))
    for g in sorted(groups, key=key):
        rs = groups[g]
        cells = []
        for k, _ in COLS:
            v = np.array([r[k] for r in rs], dtype=float)
            cells.append(f"{v.mean():6.2f}" + (f" ±{v.std(ddof=1):4.1f}" if len(v) > 1 else "      "))
        print(f"{g:40s} {len(rs):5d} " + " ".join(f"{c:>12s}" for c in cells))
    print("\ninherit = teacher-error inheritance (lower is better); flagged = same on low-trust pairs only;"
          "\nt-correct = accuracy on images the teacher gets right; ± = std over seeds.")


if __name__ == "__main__":
    main()
