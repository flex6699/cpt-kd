"""
ACDC table of the report (tables/tab_acdc.tex) from cpt_pilot.py output and the training logs.

Usage: python seg/make_acdc_table.py --pilot <runs>/cpt_final/cpt_pilot.json --runs <runs> --out report/tables/tab_acdc.tex
"""
import argparse
import json
import os
import re

import numpy as np

GROUPS = [("Student alone", ["student", "student_s1234"]),
          ("KD ($\\tau{=}1$)", ["kis_kl_t1", "kis_kl_t1_s1234"]),
          ("CPT-KD ($\\tau{=}1$)", ["kis_cpt_t1", "kis_cpt_t1_s1234"]),
          ("KD ($\\tau{=}2$)", ["kis_kl_t2", "kis_kl_t2_s1234"]),
          ("CPT-KD ($\\tau{=}2$)", ["kis_cpt_t2", "kis_cpt_t2_s1234"])]


def vol_dice(runs, exp):
    for f in (os.path.join(runs, exp, "log.txt"), os.path.join(runs, exp + ".out")):
        if os.path.exists(f):
            m = re.findall(r"TEST \(best-val checkpoint\) \| dice ([0-9.]+)", open(f).read())
            if m:
                return float(m[-1]) * 100
    return None


def fmt(v, d=1):
    v = [x for x in v if x is not None]
    if not v:
        return "--"
    if len(v) == 1:
        return ("%." + str(d) + "f") % v[0]
    return ("%." + str(d) + "f{\\scriptsize$\\pm$%." + str(d) + "f}") % (np.mean(v), np.std(v, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = json.load(open(a.pilot))
    t = P["test_teacher"]
    lines = ["\\begin{tabular}{l c ccccc}", "\\toprule",
             "Method & seeds & Dice$\\uparrow$ & vol.\\ Dice$\\uparrow$ & Inh.$\\downarrow$ & Flag.$\\downarrow$ & T-ok$\\uparrow$ \\\\",
             "\\midrule",
             "Teacher (SAM ViT-L) & 1 & %.1f & %s & -- & -- & -- \\\\" % (100 * np.mean(t["dice"]), fmt([vol_dice(a.runs, "teacher")]))]
    summary = {}
    for name, exps in GROUPS:
        have = [e for e in exps if e in P]
        if not have:
            continue
        d = [100 * np.mean(P[e]["dice"]) for e in have]
        inh = [100 * P[e]["inherit"] for e in have]
        fl = [100 * P[e]["inherit_flagged"] for e in have]
        ok = [100 * P[e]["acc_teacher_right"] for e in have]
        vd = [vol_dice(a.runs, e) for e in have]
        summary[name] = dict(seeds=len(have), dice=d, vol_dice=vd, inherit=inh, flagged=fl, t_ok=ok)
        lines.append("%s & %d & %s & %s & %s & %s & %s \\\\" % (name, len(have), fmt(d), fmt(vd), fmt(inh), fmt(fl),
                                                              fmt(ok, 2)))
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(a.out, "w").write("\n".join(lines) + "\n")
    json.dump(summary, open(a.out.replace(".tex", ".json"), "w"), indent=1)
    for k, v in summary.items():
        print("%-22s n=%d Dice %.2f vol %s inh %.2f flag %.2f" % (k, v["seeds"], np.mean(v["dice"]),
              v["vol_dice"], np.mean(v["inherit"]), np.mean(v["flagged"])))


if __name__ == "__main__":
    main()
