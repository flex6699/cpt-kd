"""
Qualitative ACDC figure: does the distilled student copy the SAM teacher's segmentation mistakes?

For every test slice, "copied pixels" = pixels where the teacher is WRONG, the student trained alone is RIGHT
and the KD student predicts the teacher's wrong class. The slices with the most copied pixels are shown:
  MRI | ground truth | teacher (SAM ViT-L) | student alone | KD | CPT-KD (if given)
Copied pixels are outlined in yellow on the KD / CPT-KD panels.

Usage (from the IAD repo root, which provides the model code):
  PYTHONPATH=.:ACDC_Tool python <repo>/seg/make_seg_figure.py --root ~/cvpr2027/data/ACDC --runs ~/cvpr2027/runs \
      --ckpt ~/cvpr2027/checkpoints/sam_vit_l_0b3195.pth --alone student --kd kis_kl_t1 --cpt kis_cpt_t1 \
      --out <repo>/report/figs/fig_problem_acdc.png
"""
import os
import sys
import glob
import json
import argparse
from importlib import import_module

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CMAP = ListedColormap([(0, 0, 0, 0), (0.85, 0.2, 0.2, 0.55), (0.2, 0.75, 0.25, 0.55), (0.2, 0.45, 0.95, 0.55)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alone", default="student")
    ap.add_argument("--kd", default="kis_kl_t1")
    ap.add_argument("--cpt", default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from scipy.ndimage import zoom, binary_dilation
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    from segment_anything_Stu import sam_model_registry
    from segment_anything_Tea import sam_model_registry_tea
    sys.argv = _argv
    S, runs = 256, os.path.expanduser(a.runs)
    kw = dict(image_size=S, num_classes=3, checkpoint=os.path.expanduser(a.ckpt), pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])

    X, Y = [], []
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(a.root), "test", "*.npz"))):
        d = np.load(f)
        for z in range(d["img"].shape[0]):
            img, lab = d["img"][z], d["label"][z]
            h, w = img.shape
            X.append(zoom(img, (S / h, S / w), order=3).astype(np.float32))
            Y.append(zoom(lab, (S / h, S / w), order=0).astype(np.int64))
    X, Y = np.stack(X), np.stack(Y)

    @torch.no_grad()
    def pred(net):
        out = []
        for i in range(0, len(X), 24):
            o = net(torch.from_numpy(X[i:i + 24]).unsqueeze(1).cuda())
            o = o[0] if isinstance(o, (tuple, list)) else o
            out.append(o.argmax(1).cpu().numpy())
        return np.concatenate(out)

    m, _ = sam_model_registry_tea["vit_l"](**kw)
    tea = import_module("sam_lora_image_encoder_tea").LoRA_Sam(m, 8).cuda().eval()
    tea.load_lora_parameters(os.path.join(runs, "teacher", "best.pth"))
    P = {"Teacher (SAM ViT-L)": pred(tea)}
    del tea
    names = [("Student alone", a.alone), ("KD", a.kd)] + ([("CPT-KD (ours)", a.cpt)] if a.cpt else [])
    for nm, exp in names:
        m, _ = sam_model_registry["vit_l"](**kw)
        net = import_module("sam_lora_image_encoder_stu").LoRA_Sam(m, 8).cuda().eval()
        net.load_state_dict(torch.load(os.path.join(runs, exp, "best.pth"), map_location="cuda"))
        P[nm] = pred(net)
        del net
        torch.cuda.empty_cache()

    T, A, K = P["Teacher (SAM ViT-L)"], P["Student alone"], P["KD"]
    copied = (T != Y) & (A == Y) & (K == T)
    score = copied.sum((1, 2))
    pick = np.argsort(-score)[:a.n]
    cols = ["MRI", "Ground truth"] + list(P.keys())
    fig, axes = plt.subplots(a.n, len(cols), figsize=(2.15 * len(cols), 2.25 * a.n))
    stats = []
    for r, i in enumerate(pick):
        lo, hi = np.percentile(X[i], [1, 99])
        for c, col in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(X[i], cmap="gray", vmin=lo, vmax=hi)
            if col != "MRI":
                ax.imshow(Y[i] if col == "Ground truth" else P[col][i], cmap=CMAP, vmin=0, vmax=3,
                          interpolation="nearest")
            if col in ("KD", "CPT-KD (ours)"):
                still = copied[i] & (P[col][i] == T[i]) if col != "KD" else copied[i]
                edge = binary_dilation(still, iterations=1) & ~still
                ax.imshow(np.ma.masked_where(~edge, edge), cmap=ListedColormap(["yellow"]), interpolation="nearest")
                ax.text(0.03, 0.04, "%d copied px" % still.sum(), color="yellow", fontsize=7, transform=ax.transAxes)
            # crop to the heart
            ys, xs = np.where((Y[i] > 0) | (T[i] > 0))
            if len(ys):
                cy, cx = int(ys.mean()), int(xs.mean())
                ax.set_xlim(max(0, cx - 64), min(S, cx + 64)); ax.set_ylim(min(S, cy + 64), max(0, cy - 64))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col, fontsize=9)
        stats.append(dict(slice=int(i), copied_kd=int(copied[i].sum())))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=CMAP(1)), Patch(color=CMAP(2)), Patch(color=CMAP(3)), Patch(color="yellow")],
               labels=["RV", "Myocardium", "LV", "teacher error copied by student"], loc="lower center", ncol=4,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(wspace=0.03, hspace=0.03, bottom=0.05)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    tot = {nm: int(((T != Y) & (A == Y) & (P[nm] == T)).sum()) for nm in P if nm not in ("Teacher (SAM ViT-L)", "Student alone")}
    json.dump(dict(slices=stats, total_copied_pixels=tot, teacher_wrong_alone_right=int(((T != Y) & (A == Y)).sum())),
              open(a.out.replace(".png", ".json"), "w"), indent=1)
    print("total copied pixels (teacher wrong, alone right, model = teacher):", tot,
          "of", int(((T != Y) & (A == Y)).sum()))


if __name__ == "__main__":
    main()
