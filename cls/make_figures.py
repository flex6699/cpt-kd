"""
Report / slide figures from the FINAL runs (seed 0, ResNet-50 -> ResNet-18).

  fig_problem_<dataset>.png  real test images where the teacher is wrong on a look-alike class, the student
                             trained alone is right, vanilla KD copies the teacher's mistake and CPT-KD does not
  fig_problem.png / fig_problem_kd_only.png   2 CUB + 2 Aircraft examples, with / without the CPT-KD bars
  fig_c1_inheritance.png     teacher-error inheritance: student alone vs KD vs CPT-KD, both datasets, both students
  fig_trust_<dataset>.png    pair-trust heatmap over the classes of the lowest-trust look-alike pairs

Usage (from the repository root, after the experiments):
  python cls/make_figures.py --runs <ml_research>/runs --data <ml_research>/data/tea_kd --out report/figs
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

C_T, C_S, C_KD, C_CPT = "#7f7f7f", "#2ca02c", "#d62728", "#1f77b4"


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load_logits(path):
    d = np.load(path)
    o = np.argsort(d["idx"])
    return d["logits"][o], d["labels"][o], d["idx"][o]


def nice(name):
    return name.split(".", 1)[-1].replace("_", " ")


def problem_figure(ds, runs, data, out, n=4, seed=0, student="resnet18"):
    meta = list(csv.DictReader(open(os.path.join(data, ds, "meta.csv"), encoding="utf-8")))
    names = {}
    for r in meta:
        names[int(r["label"])] = r["class_name"]
    oof = os.path.join(runs, ds, "final", "oof")
    trust = np.load(os.path.join(oof, "trust_T1.npy"))
    Lt, y, idx = load_logits(os.path.join(oof, "teacher_full_test.npz"))
    st = os.path.join(runs, ds, "final", "students")
    Ls = load_logits(os.path.join(st, f"ce_{student}_s{seed}", "test_logits.npz"))[0]
    Lk = load_logits(os.path.join(st, f"kd_{student}_s{seed}", "test_logits.npz"))[0]
    Lc = load_logits(os.path.join(st, f"cpt_{student}_keept_s{seed}", "test_logits.npz"))[0]
    Pt, Ps, Pk, Pc = (softmax(L) for L in (Lt, Ls, Lk, Lc))
    pt, ps, pk, pc = (P.argmax(1) for P in (Pt, Ps, Pk, Pc))
    cand = np.where((pt != y) & (ps == y) & (pk == pt) & (pc == y))[0]
    tr = trust[y[cand], pt[cand]]
    # most confident teacher mistakes on low-trust (look-alike) pairs first
    cand = cand[tr < 0.7]          # only look-alike pairs that CPT-KD actually turns down
    order = cand[np.argsort(-Pt[cand, pt[cand]])]
    chosen, seen = [], set()
    for i in order:
        key = (int(y[i]), int(pt[i]))
        if key in seen:
            continue
        seen.add(key)
        chosen.append(i)
        if len(chosen) == n:
            break
    rows = {int(r["idx"]): r for r in meta}
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 5.6), gridspec_kw=dict(height_ratios=[1.25, 1]))
    info = []
    for j, i in enumerate(chosen):
        r = rows[int(idx[i])]
        img = Image.open(os.path.join(data, ds, r["file"])).convert("RGB")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((256, 256))
        ax = axes[0, j]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("label: %s" % nice(names[int(y[i])]), fontsize=10, color=C_S)
        ax = axes[1, j]
        models = ["Teacher\n(R-50)", "Student\nalone", "KD", "CPT-KD\n(ours)"]
        P = [Pt[i], Ps[i], Pk[i], Pc[i]]
        x = np.arange(4)
        a = [p[y[i]] for p in P]
        b = [p[pt[i]] for p in P]
        ax.bar(x - 0.2, a, 0.4, color=C_S, label="true: " + nice(names[int(y[i])]))
        ax.bar(x + 0.2, b, 0.4, color=C_KD, label="look-alike: " + nice(names[int(pt[i])]))
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
        ax.set_title("teacher says: %s\ntrust = %.2f" % (nice(names[int(pt[i])]), trust[y[i], pt[i]]),
                     fontsize=9, color=C_KD)
        if j == 0:
            ax.set_ylabel("probability")
        info.append(dict(ds=ds, file=r["file"], label=names[int(y[i])], teacher=names[int(pt[i])],
                         p=[[float(v) for v in a], [float(v) for v in b]], trust=float(trust[y[i], pt[i]])))
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_problem_{ds}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    json.dump(dict(n_candidates=int(len(cand)), examples=info),
              open(os.path.join(out, f"fig_problem_{ds}.json"), "w"), indent=1)
    print(ds, "candidates:", len(cand), "->", [(e["label"], e["teacher"]) for e in info])


def c1_figure(runs, out):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=False)
    k = 0
    for ds, dsn in (("cub", "CUB-200"), ("aircraft", "FGVC-Aircraft")):
        for st, stn in (("resnet18", "ResNet-18"), ("mobilenet_v2", "MobileNetV2")):
            ax = axes[k]; k += 1
            vals = []
            for m, lab, col in ((f"ce_{st}", "Student\nalone", C_S), (f"kd_{st}", "KD", C_KD),
                                (f"cpt_{st}_keept", "CPT-KD", C_CPT)):
                for metric, off in (("inheritance", -0.2), ("flagged_pair_inheritance", 0.2)):
                    v = [json.load(open(os.path.join(runs, ds, "final", "students", f"{m}_s{s}", "results.json")))[metric]
                         for s in range(3)]
                    vals.append((lab, metric, np.mean(v), np.std(v), col))
            labs = ["Student\nalone", "KD", "CPT-KD"]
            for li, lab in enumerate(labs):
                for metric, off, hatch in (("inheritance", -0.2, ""), ("flagged_pair_inheritance", 0.2, "//")):
                    m_, s_, col = [(v[2], v[3], v[4]) for v in vals if v[0] == lab and v[1] == metric][0]
                    ax.bar(li + off, m_, 0.4, yerr=s_, color=col, hatch=hatch, edgecolor="white", capsize=3,
                           alpha=1.0 if not hatch else 0.65)
            ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=8)
            ax.set_title(f"{dsn}, {stn}", fontsize=9)
            if k == 1:
                ax.set_ylabel("teacher-error inheritance (%)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#999"), Patch(facecolor="#999", hatch="//", alpha=0.65, edgecolor="white")],
               labels=["all teacher errors", "flagged (low-trust) pairs"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.07), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_c1_inheritance.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def trust_figure(ds, runs, data, out, k=12):
    """Heatmap of the pair-trust table restricted to the classes of the lowest-trust look-alike pairs."""
    meta = list(csv.DictReader(open(os.path.join(data, ds, "meta.csv"), encoding="utf-8")))
    names = {int(r["label"]): r["class_name"] for r in meta}
    inv = {v: k_ for k_, v in names.items()}
    oof = os.path.join(runs, ds, "final", "oof")
    trust = np.load(os.path.join(oof, "trust_T1.npy"))
    rows = list(csv.DictReader(open(os.path.join(oof, "lowest_trust_pairs.csv"))))
    cls = []
    for r in rows:
        for nm in (r["true_class"], r["confused_with"]):
            c = inv[nm] if nm in inv else [k_ for k_, v in names.items() if nice(v) == nice(nm)][0]
            if c not in cls:
                cls.append(c)
        if len(cls) >= k:
            break
    cls = cls[:k]
    M = trust[np.ix_(cls, cls)]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1)
    lab = [nice(names[c])[:22] for c in cls]
    ax.set_xticks(range(len(cls))); ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(cls))); ax.set_yticklabels(lab, fontsize=7)
    for i in range(len(cls)):
        for j in range(len(cls)):
            if i != j and M[i, j] < 0.95:
                ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=6)
    ax.set_xlabel("wrong class c"); ax.set_ylabel("true class y")
    fig.colorbar(im, ax=ax, fraction=0.046, label="trust(y, c)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_trust_{ds}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def combined_problem_figure(data, out, per_ds=2, with_cpt=True):
    """Fig. 1: 2 CUB + 2 Aircraft examples from the per-dataset JSON files."""
    ex = []
    for ds in ("cub", "aircraft"):
        ex += json.load(open(os.path.join(out, f"fig_problem_{ds}.json")))["examples"][:per_ds]
    n = len(ex)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5.9), gridspec_kw=dict(height_ratios=[1.15, 1]))
    for j, e in enumerate(ex):
        img = Image.open(os.path.join(data, e["ds"], e["file"])).convert("RGB")
        w, h = img.size
        if e["ds"] == "aircraft":
            img = img.resize((256, int(256 * h / w)))
        else:
            s_ = min(w, h)
            img = img.crop(((w - s_) // 2, (h - s_) // 2, (w + s_) // 2, (h + s_) // 2)).resize((256, 256))
        ax = axes[0, j]
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("true: %s" % nice(e["label"]), fontsize=10, color=C_S)
        ax = axes[1, j]
        k = 4 if with_cpt else 3
        x = np.arange(k)
        ax.bar(x - 0.2, e["p"][0][:k], 0.4, color=C_S, label="true class")
        ax.bar(x + 0.2, e["p"][1][:k], 0.4, color=C_KD, label="teacher's wrong class")
        ax.set_xticks(x); ax.set_xticklabels(["Teacher", "Student\nalone", "KD", "CPT-KD\n(ours)"][:k], fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(("teacher: %s\ntrust = %.2f" % (nice(e["teacher"]), e["trust"])) if with_cpt
                     else "teacher says: %s" % nice(e["teacher"]), fontsize=9, color=C_KD)
        if j == 0:
            ax.set_ylabel("probability")
    axes[1, 0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.28), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_problem.png" if with_cpt else "fig_problem_kd_only.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="report/figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for ds in ("cub", "aircraft"):
        problem_figure(ds, a.runs, a.data, a.out)
    combined_problem_figure(a.data, a.out)
    combined_problem_figure(a.data, a.out, with_cpt=False)
    c1_figure(a.runs, a.out)
    trust_figure("cub", a.runs, a.data, a.out)
    trust_figure("aircraft", a.runs, a.data, a.out)


if __name__ == "__main__":
    main()
