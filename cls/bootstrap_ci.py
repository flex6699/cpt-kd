"""
Paired bootstrap confidence intervals: CPT-KD minus vanilla KD (and other methods minus KD), from saved test logits.

For each bootstrap sample we resample the TEST IMAGES with replacement (the same images for both methods and all
seeds), recompute each metric for each seed, take the seed mean, and record the difference to KD.
Metrics: accuracy (top-1 for CUB, mean per-class for Aircraft), teacher-error inheritance, flagged-pair inheritance.
Reported: mean difference, 95% percentile interval, and the share of bootstrap samples where the sign flips (p-like).

Usage: python cls/bootstrap_ci.py --runs <ml_research>/runs --out report/tables/tab_bootstrap.tex [--B 2000]
"""
import argparse
import json
import os

import numpy as np

SETUPS = [("cub", "resnet18", "CUB, R18"), ("cub", "mobilenet_v2", "CUB, MNv2"),
          ("aircraft", "resnet18", "Aircraft, R18"), ("aircraft", "mobilenet_v2", "Aircraft, MNv2")]
METHODS = [("cpt_{s}_keept", "CPT-KD"), ("wsl_{s}", "WSL"), ("class_trust_{s}_keept", "Class trust")]


def load(path):
    d = np.load(path)
    o = np.argsort(d["idx"])
    return d["logits"][o].argmax(1), d["labels"][o]


def metrics(ps, y, pt, flagged, C, per_class, w):
    """w: bootstrap counts per test image (resampling weights)."""
    corr = (ps == y) * w
    if per_class:
        acc = np.mean([corr[y == c].sum() / max(w[y == c].sum(), 1) for c in range(C)])
    else:
        acc = corr.sum() / w.sum()
    tw = (pt != y)
    same = (ps == pt)
    inh = (same * tw * w).sum() / max((tw * w).sum(), 1)
    fl = (same * flagged * w).sum() / max((flagged * w).sum(), 1)
    return 100 * np.array([acc, inh, fl])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="report/tables/tab_bootstrap.tex")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    res = {}
    lines = ["\\begin{tabular}{ll ccc}", "\\toprule",
             "Setup & Method $-$ KD & $\\Delta$Acc. & $\\Delta$Inh. & $\\Delta$Flag. \\\\", "\\midrule"]
    for ds, st, name in SETUPS:
        oof = os.path.join(a.runs, ds, "final", "oof")
        pt, y = load(os.path.join(oof, "teacher_full_test.npz"))
        trust = np.load(os.path.join(oof, "trust_T1.npy"))
        C = trust.shape[0]
        flagged = (pt != y) & (trust[y, pt] < 0.5)
        sd = os.path.join(a.runs, ds, "final", "students")
        kd = [load(os.path.join(sd, f"kd_{st}_s{s}", "test_logits.npz"))[0] for s in range(3)]
        W = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=a.B).astype(np.float64)
        per_class = ds == "aircraft"
        for key, mname in METHODS:
            other = [load(os.path.join(sd, key.format(s=st) + f"_s{s}", "test_logits.npz"))[0] for s in range(3)]
            ones = np.ones(len(y))
            point = np.mean([metrics(o, y, pt, flagged, C, per_class, ones) - metrics(k, y, pt, flagged, C, per_class, ones)
                             for o, k in zip(other, kd)], 0)
            boot = np.zeros((a.B, 3))
            for b in range(a.B):
                w = W[b]
                boot[b] = np.mean([metrics(o, y, pt, flagged, C, per_class, w) - metrics(k, y, pt, flagged, C, per_class, w)
                                   for o, k in zip(other, kd)], 0)
            lo, hi = np.percentile(boot, [2.5, 97.5], 0)
            p = np.minimum((boot >= 0).mean(0), (boot <= 0).mean(0)) * 2
            res[f"{name}/{mname}"] = dict(diff=point.tolist(), lo=lo.tolist(), hi=hi.tolist(), p_two_sided=p.tolist(),
                                          n_test=int(len(y)), n_flagged=int(flagged.sum()))
            cell = lambda i: "%+.1f [%+.1f, %+.1f]%s" % (point[i], lo[i], hi[i], "$^*$" if (lo[i] > 0 or hi[i] < 0) else "")
            lines.append("%s & %s & %s & %s & %s \\\\" % (name if mname == "CPT-KD" else "", mname, cell(0), cell(1), cell(2)))
            print("%-15s %-12s dAcc %s | dInh %s | dFlag %s" % (name, mname, cell(0), cell(1), cell(2)))
        lines.append("\\midrule" if (ds, st) != SETUPS[-1][:2] else "")
    lines = [l for l in lines if l] + ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    json.dump(res, open(a.out.replace(".tex", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
