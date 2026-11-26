"""
LaTeX tables for the report, computed directly from runs/<dataset>/final/students/*/results.json.

  tab_main_<dataset>.tex   all methods, both students: accuracy, inheritance, flagged-pair inheritance (mean ± std, 3 seeds)
  tab_extra.tex            accuracy on teacher-correct images, agreement, ECE, worst-10-class accuracy (ResNet-18)
  tab_ablation.tex         CPT-KD variants on CUB / ResNet-18, as differences to KD on the same seeds
  numbers.json             every number used in the text

Usage:  python cls/make_tables.py --runs <ml_research>/runs --out report/tables
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROWS = [("ce", "Student alone"), ("kd", "Vanilla KD~\\cite{hinton2015distilling}"),
        ("dkd", "DKD~\\cite{zhao2022dkd}"), ("wsl", "Weighted soft labels~\\cite{zhou2021wsl}"),
        ("ka", "Knowledge Adjustment~\\cite{wen2021preparing}"),
        ("pergate_{s}_oofteacher", "Per-image gate~\\cite{mishra2026correctness}"),
        ("class_trust_{s}_keept", "Class-level trust~\\cite{lukasik2022teacherspet}"),
        ("cpt_{s}_keept", "\\textbf{CPT-KD (ours)}"), ("cpt_dkd_{s}_keept", "\\textbf{CPT-KD + DKD (ours)}"),
        ("cpt_wsl_{s}_keept", "\\textbf{CPT-KD + WSL (ours)}"), ("cpt_pair_{s}_keept", "\\textbf{CPT-KD + pair penalty (ours)}")]
STUDENTS = [("resnet18", "ResNet-18"), ("mobilenet_v2", "MobileNetV2")]


def load(runs, ds):
    g = defaultdict(dict)
    for f in glob.glob(os.path.join(runs, ds, "final", "students", "*", "results.json")):
        name = os.path.basename(os.path.dirname(f))
        m = re.match(r"(.*)_s(\d+)$", name)
        g[m.group(1)][int(m.group(2))] = json.load(open(f))
    return g


def group(key, s):
    return key.format(s=s) if "{s}" in key else f"{key}_{s}"


def ms(v):
    return "%.1f{\\scriptsize$\\pm$%.1f}" % (np.mean(v), np.std(v, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="report/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    numbers = {}
    for ds, dsn, acc_key in (("cub", "CUB-200-2011", "top1"), ("aircraft", "FGVC-Aircraft", "mean_per_class")):
        g = load(a.runs, ds)
        lines = ["\\begin{tabular}{l ccc ccc}", "\\toprule",
                 " & \\multicolumn{3}{c}{ResNet-50 $\\to$ ResNet-18} & \\multicolumn{3}{c}{ResNet-50 $\\to$ MobileNetV2} \\\\",
                 "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                 "Method & %s$\\uparrow$ & Inh.$\\downarrow$ & Flag.$\\downarrow$ & %s$\\uparrow$ & Inh.$\\downarrow$ & Flag.$\\downarrow$ \\\\"
                 % (("Top-1", "Top-1") if ds == "cub" else ("mAcc", "mAcc")), "\\midrule"]
        for key, label in ROWS:
            cells = []
            for s, _ in STUDENTS:
                runs = g.get(group(key, s), {})
                if len(runs) == 0:
                    cells += ["--"] * 3
                    continue
                v = {k: [r[k] for r in runs.values()] for k in (acc_key, "inheritance", "flagged_pair_inheritance",
                                                                 "acc_teacher_correct", "agreement", "ece", "worst10_class")}
                numbers[f"{ds}/{group(key, s)}"] = {k: [float(np.mean(x)), float(np.std(x, ddof=1)), len(x)] for k, x in v.items()}
                cells += [ms(v[acc_key]), ms(v["inheritance"]), ms(v["flagged_pair_inheritance"])]
            if key == "cpt_{s}_keept":
                lines.append("\\midrule")
            lines.append(label + " & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(a.out, f"tab_main_{ds}.tex"), "w").write("\n".join(lines) + "\n")

    # extra metrics, ResNet-18, both datasets
    lines = ["\\begin{tabular}{l cccc cccc}", "\\toprule",
             " & \\multicolumn{4}{c}{CUB-200-2011} & \\multicolumn{4}{c}{FGVC-Aircraft} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
             "Method & T-ok$\\uparrow$ & Agr. & ECE$\\downarrow$ & W10$\\uparrow$ & T-ok$\\uparrow$ & Agr. & ECE$\\downarrow$ & W10$\\uparrow$ \\\\",
             "\\midrule"]
    for key, label in ROWS:
        cells = []
        for ds in ("cub", "aircraft"):
            n = numbers.get(f"{ds}/{group(key, 'resnet18')}")
            if n is None:
                cells += ["--"] * 4
                continue
            cells += ["%.1f" % n[k][0] for k in ("acc_teacher_correct", "agreement", "ece", "worst10_class")]
        lines.append(label.split("~")[0] + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(a.out, "tab_extra.tex"), "w").write("\n".join(lines) + "\n")

    # ablation: CUB ResNet-18, paired difference to KD on the same seeds
    g = load(a.runs, "cub")
    kd = g["kd_resnet18"]
    variants = [("cpt_resnet18", "Rescale all classes (true class absorbs mass)"),
                ("cpt_resnet18_keept", "Keep true-class prob. (\\textbf{default})"),
                ("cpt_resnet18_pow2", "trust$^2$, rescale all"),
                ("cpt_resnet18_pow2_keept", "trust$^2$, keep true class"),
                ("cpt_resnet18_trustT4", "Trust estimated at $\\tau{=}4$ (rescale all)"),
                ("class_trust_resnet18_keept", "Per-class trust (keep true class)"),
                ("cpt_dkd_resnet18_keept", "CPT-KD + DKD")]
    lines = ["\\begin{tabular}{l c ccc}", "\\toprule",
             "Variant (CUB, ResNet-18) & Seeds & $\\Delta$Top-1 & $\\Delta$Inh. & $\\Delta$Flag. \\\\", "\\midrule"]
    for key, label in variants:
        if key not in g:
            continue
        seeds = sorted(set(g[key]) & set(kd))
        d = {m: [g[key][s][m] - kd[s][m] for s in seeds] for m in ("top1", "inheritance", "flagged_pair_inheritance")}
        numbers[f"ablation/{key}"] = {m: [float(np.mean(v)), float(np.std(v, ddof=1)), len(v)] for m, v in d.items()}
        f = (lambda v: "%+.2f{\\scriptsize$\\pm$%.2f}" % (np.mean(v), np.std(v, ddof=1))) if len(seeds) > 1 else (lambda v: "%+.2f" % v[0])
        lines.append("%s & %d & %s & %s & %s \\\\" % (label, len(seeds), f(d["top1"]), f(d["inheritance"]),
                                                     f(d["flagged_pair_inheritance"])))
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(a.out, "tab_ablation.tex"), "w").write("\n".join(lines) + "\n")
    # label-noise robustness (20% symmetric noise on the training labels), ResNet-18
    lines = ["\\begin{tabular}{l ccc ccc}", "\\toprule",
             " & \\multicolumn{3}{c}{CUB, 20\\% noise} & \\multicolumn{3}{c}{Aircraft, 20\\% noise} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Method & Top-1$\\uparrow$ & Inh.$\\downarrow$ & Flag.$\\downarrow$ & mAcc$\\uparrow$ & Inh.$\\downarrow$ & Flag.$\\downarrow$ \\\\", "\\midrule"]
    have_noise = False
    for key, label in (("ce", "Student alone"), ("kd", "Vanilla KD"), ("cpt_{s}_keept", "\\textbf{CPT-KD (ours)}")):
        cells = []
        for ds, acc_key in (("cub_noise20", "top1"), ("aircraft_noise20", "mean_per_class")):
            if not os.path.isdir(os.path.join(a.runs, ds)):
                cells += ["--"] * 3
                continue
            g = load(a.runs, ds)
            runs = g.get(group(key, "resnet18"), {})
            if not runs:
                cells += ["--"] * 3
                continue
            have_noise = True
            v = {k: [r[k] for r in runs.values()] for k in (acc_key, "inheritance", "flagged_pair_inheritance")}
            numbers[f"{ds}/{group(key, 'resnet18')}"] = {k: [float(np.mean(x)), float(np.std(x, ddof=1)), len(x)] for k, x in v.items()}
            cells += [ms(v[acc_key]), ms(v["inheritance"]), ms(v["flagged_pair_inheritance"])]
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if have_noise:
        open(os.path.join(a.out, "tab_noise.tex"), "w").write("\n".join(lines) + "\n")
    json.dump(numbers, open(os.path.join(a.out, "numbers.json"), "w"), indent=1)
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
