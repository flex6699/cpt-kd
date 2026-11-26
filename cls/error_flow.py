"""
Where do the removed mistakes go? And do CPT-KD students help an ensemble? (from saved test logits, 3 seeds)

On flagged test images (teacher wrong on a low-trust pair): % correct, % copying the teacher, % another wrong class.
Two-student ensembles (equal-weight softmax average, different seeds): KD+KD vs KD+CPT-KD vs CPT-KD+CPT-KD.

Usage: python cls/error_flow.py --runs <ml_research>/runs --out report/tables/tab_errorflow.tex
"""
import argparse
import itertools
import json

import numpy as np


def load(path):
    d = np.load(path)
    o = np.argsort(d["idx"])
    z = d["logits"][o]
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True), d["labels"][o]


def acc(p, y, per_class):
    pr = p.argmax(1)
    return 100 * (np.mean([np.mean(pr[y == c] == c) for c in np.unique(y)]) if per_class else np.mean(pr == y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="report/tables/tab_errorflow.tex")
    a = ap.parse_args()
    res = {}
    lines = ["\\begin{tabular}{l ccc ccc ccc}", "\\toprule",
             " & \\multicolumn{3}{c}{Flagged: correct (\\%)} & \\multicolumn{3}{c}{Flagged: other wrong (\\%)} & \\multicolumn{3}{c}{2-student ensemble acc.} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
             "Setup & alone & KD & CPT & alone & KD & CPT & KD+KD & KD+CPT & CPT+CPT \\\\", "\\midrule"]
    for ds, st, name in (("cub", "resnet18", "CUB, R18"), ("cub", "mobilenet_v2", "CUB, MNv2"),
                         ("aircraft", "resnet18", "Aircraft, R18"), ("aircraft", "mobilenet_v2", "Aircraft, MNv2")):
        pc = ds == "aircraft"
        oof = f"{a.runs}/{ds}/final/oof"
        Pt, y = load(f"{oof}/teacher_full_test.npz")
        pt = Pt.argmax(1)
        trust = np.load(f"{oof}/trust_T1.npy")
        fl = (pt != y) & (trust[y, pt] < 0.5)
        S = {m: [load(f"{a.runs}/{ds}/final/students/{k}_s{s}/test_logits.npz")[0] for s in range(3)]
             for m, k in (("alone", f"ce_{st}"), ("KD", f"kd_{st}"), ("CPT", f"cpt_{st}_keept"))}
        corr = {m: np.mean([np.mean(p.argmax(1)[fl] == y[fl]) for p in S[m]]) * 100 for m in S}
        other = {m: np.mean([np.mean((p.argmax(1)[fl] != y[fl]) & (p.argmax(1)[fl] != pt[fl])) for p in S[m]]) * 100 for m in S}
        pairs = list(itertools.combinations(range(3), 2))
        cross = [(i, j) for i in range(3) for j in range(3) if i != j]
        ens = {"KD+KD": np.mean([acc((S["KD"][i] + S["KD"][j]) / 2, y, pc) for i, j in pairs]),
               "KD+CPT": np.mean([acc((S["KD"][i] + S["CPT"][j]) / 2, y, pc) for i, j in cross]),
               "CPT+CPT": np.mean([acc((S["CPT"][i] + S["CPT"][j]) / 2, y, pc) for i, j in pairs])}
        res[name] = dict(n_flagged=int(fl.sum()), n_test=int(len(y)), correct=corr, other_wrong=other, ensemble=ens)
        lines.append("%s & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.2f & %.2f & %.2f \\\\" % (
            name, corr["alone"], corr["KD"], corr["CPT"], other["alone"], other["KD"], other["CPT"],
            ens["KD+KD"], ens["KD+CPT"], ens["CPT+CPT"]))
        print(name, res[name])
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(a.out, "w").write("\n".join(lines) + "\n")
    json.dump(res, open(a.out.replace(".tex", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
