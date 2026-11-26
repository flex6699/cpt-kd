"""
Final analysis: tables and figures for the report / final poster.

Reads every runs/<dataset>/final/students/*/results.json and writes to results/:
  summary.csv                         one row per (dataset, student, method version): mean and std over seeds
  table_<dataset>_<student>.md        Markdown table (paste into the report)
  paired_vs_kd_<dataset>_<student>.md each method minus KD on the SAME seeds (mean ± std of the differences)
  fig_inheritance_<dataset>_<student>.png   inheritance (all teacher errors and flagged pairs), per method
  fig_acc_vs_inh_<dataset>_<student>.png    accuracy vs inheritance (top-left is best)

The headline accuracy is top-1 for CUB and mean per-class accuracy for FGVC-Aircraft (official metric).

Usage
  python analyze.py
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

NICE = {"ce": "Student alone", "kd": "KD", "dkd": "DKD", "wsl": "Weighted soft labels", "ka": "Knowledge Adjustment",
        "pergate": "Per-image gate", "class_trust": "Class-level trust", "cpt": "CPT-KD (ours)",
        "cpt_dkd": "CPT-KD + DKD (ours)"}
ORDER = ["ce", "kd", "dkd", "wsl", "ka", "pergate", "class_trust", "cpt", "cpt_dkd"]
METRICS = ["top1", "mean_per_class", "inheritance", "flagged_pair_inheritance", "acc_teacher_correct",
           "agreement", "ece", "worst10_class"]


def version_label(group, method, student):
    """'cpt_resnet18_pow2_keept' -> 'CPT-KD (ours) [pow2, keept]'."""
    rest = group.split(f"{method}_{student}", 1)[-1].strip("_")
    return NICE.get(method, method) + (f" [{rest.replace('_', ', ')}]" if rest else "")


def load(root):
    data = defaultdict(lambda: defaultdict(dict))      # (ds, student) -> group -> seed -> result
    for f in glob.glob(os.path.join(root, "*", "final", "students", "*", "results.json")):
        ds = f.split(os.sep)[-5]
        r = json.load(open(f))
        group = re.sub(r"_s\d+$", "", os.path.basename(os.path.dirname(f)))
        data[(ds, r["student"])][group][r["seed"]] = r
    return data


def fmt(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.2f}" + (f" ± {v.std(ddof=1):.2f}" if len(v) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = load(args.root)
    if not data:
        print("No finished runs found.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_rows = []
    for (ds, st), groups in sorted(data.items()):
        acc_key = "mean_per_class" if ds == "aircraft" else "top1"
        acc_name = "Mean per-class acc." if ds == "aircraft" else "Top-1 acc."
        items = []
        for g, seeds in groups.items():
            m = next(iter(seeds.values()))["method"]
            items.append((ORDER.index(m) if m in ORDER else 99, g, m, seeds))
        items.sort()

        # ---------- main table
        hdr = ["Method", "Seeds", acc_name, "Inheritance ↓", "Flagged-pair inh. ↓", "Acc. teacher-correct",
               "Agreement", "ECE ↓"]
        lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for _, g, m, seeds in items:
            rs = list(seeds.values())
            vals = {k: [r[k] for r in rs] for k in METRICS}
            lines.append("| " + " | ".join([version_label(g, m, st), str(len(rs)), fmt(vals[acc_key]),
                                            fmt(vals["inheritance"]), fmt(vals["flagged_pair_inheritance"]),
                                            fmt(vals["acc_teacher_correct"]), fmt(vals["agreement"]),
                                            fmt(vals["ece"])]) + " |")
            row = dict(dataset=ds, student=st, method=m, version=g, seeds=len(rs))
            for k in METRICS:
                v = np.array(vals[k], dtype=float)
                row[f"{k}_mean"], row[f"{k}_std"] = round(v.mean(), 3), round(v.std(ddof=1), 3) if len(v) > 1 else ""
            csv_rows.append(row)
        title = f"## {ds.upper()}, student {st}\n\nMean ± std over seeds. Inheritance: of test images the teacher " \
                f"gets wrong, % where the student makes the same mistake.\n\n"
        with open(os.path.join(args.out, f"table_{ds}_{st}.md"), "w") as f:
            f.write(title + "\n".join(lines) + "\n")
        print("\n" + title + "\n".join(lines))

        # ---------- paired differences vs KD (same seeds)
        kd = next((s for _, g, m, s in items if m == "kd" and g == f"kd_{st}"), None)
        if kd:
            pl = ["| Method − KD | Seeds | " + acc_name + " | Inheritance | Flagged-pair inh. |", "|---|---|---|---|---|"]
            for _, g, m, seeds in items:
                if m == "kd":
                    continue
                common = sorted(set(seeds) & set(kd))
                if not common:
                    continue
                d = {k: [seeds[s][k] - kd[s][k] for s in common]
                     for k in (acc_key, "inheritance", "flagged_pair_inheritance")}
                pl.append(f"| {version_label(g, m, st)} | {len(common)} | {fmt(d[acc_key])} | "
                          f"{fmt(d['inheritance'])} | {fmt(d['flagged_pair_inheritance'])} |")
            with open(os.path.join(args.out, f"paired_vs_kd_{ds}_{st}.md"), "w") as f:
                f.write(f"## {ds.upper()}, {st}: difference to KD on the same seeds (negative inheritance = better)\n\n"
                        + "\n".join(pl) + "\n")
            print("\nDifference to KD on the same seeds:\n" + "\n".join(pl))

        # ---------- figures
        labels = [version_label(g, m, st) for _, g, m, _ in items]
        def stat(k):
            v = [np.array([r[k] for r in s.values()], float) for *_, s in items]
            return np.array([x.mean() for x in v]), np.array([x.std(ddof=1) if len(x) > 1 else 0 for x in v])
        inh, inh_e = stat("inheritance")
        fl, fl_e = stat("flagged_pair_inheritance")
        acc, acc_e = stat(acc_key)
        ours = ["ours" in l for l in labels]

        fig, ax = plt.subplots(figsize=(10, 4.2))
        x = np.arange(len(labels))
        ax.bar(x - 0.2, inh, 0.4, yerr=inh_e, capsize=3, label="all teacher errors",
               color=["#2F3A8F" if o else "#9AA3C9" for o in ours])
        ax.bar(x + 0.2, fl, 0.4, yerr=fl_e, capsize=3, label="flagged look-alike pairs",
               color=["#C23B2E" if o else "#E9A9A2" for o in ours])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("teacher-error inheritance (%)")
        ax.set_title(f"{ds.upper()}, {st}: how often the student copies the teacher's mistakes (lower is better)",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"fig_inheritance_{ds}_{st}.png"), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for i, l in enumerate(labels):
            ax.errorbar(inh[i], acc[i], xerr=inh_e[i], yerr=acc_e[i], fmt="o", capsize=3,
                        color="#C23B2E" if ours[i] else "#2F3A8F")
            ax.annotate(l, (inh[i], acc[i]), textcoords="offset points", xytext=(5, 4), fontsize=7)
        ax.set_xlabel("teacher-error inheritance (%)  ← better")
        ax.set_ylabel(f"{acc_name.lower()} (%)  ↑ better")
        ax.set_title(f"{ds.upper()}, {st}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"fig_acc_vs_inh_{ds}_{st}.png"), dpi=200)
        plt.close(fig)

    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nWrote tables, paired comparisons, figures and summary.csv to {args.out}/")


if __name__ == "__main__":
    main()
