"""
Mahalanobis check (no training, ~5 minutes on one GPU).

Question: can we tell, WITHOUT the label, when the teacher is wrong?
          If Mahalanobis distance in the teacher's feature space detects teacher errors better than
          plain softmax confidence, it is a good label-free trust signal for distillation.

Uses the teacher trained by quick_evidence.py (runs/quick_evidence/teacher.pt) and the
student-alone test predictions saved there.

Steps
  1. Extract the teacher's penultimate features (input of the final fc layer) for all
     training and test images (no augmentation).
  2. Reduce to --pca dimensions (default 256), then fit one Gaussian per class with a shared,
     Ledoit-Wolf-shrunk covariance on the TRAINING features (Lee et al., NeurIPS 2018).
  3. For every test image compute "suspicion" scores (higher = more likely the teacher is wrong):
       msp            1 - max softmax probability              (standard confidence baseline)
       entropy        entropy of the softmax
       logit_margin   -(top-1 logit - top-2 logit)
       maha_pred      Mahalanobis distance to the teacher's predicted class
       maha_margin    -(2nd-closest - closest class distance)  (look-alike ambiguity)
       rel_maha_pred  relative Mahalanobis (Ren et al., 2021): class distance minus distance
                      to one Gaussian fitted on all training features
  4. Report, for each score:
       AUROC_errors   how well it separates teacher-wrong from teacher-right test images
       AUROC_S        how well it separates S (teacher wrong, student-alone right) from teacher-right
       catch@20%      share of teacher errors among the 20% most suspicious images (recall)
     AUROC: 0.5 = chance, 1.0 = perfect.

Outputs in --out (default: runs/quick_evidence/mahalanobis):
  results.json, scores_test.npz, hist_maha_pred.png

Usage
  python mahalanobis_check.py --data data/tea_kd/cub --run runs/quick_evidence --gpu 1
"""
import argparse
import os

import numpy as np
import torch

from common import (RAMImageDataset, autocast, build_model, build_transforms, get_device,
                    load_meta, make_loader, save_json, select_rows)


# ============================================================================ features
@torch.no_grad()
def extract(model, loader, device):
    """Return (features, logits, labels, idx), sorted by idx. Features = input of the last fc."""
    feats = []
    head = model.fc if hasattr(model, "fc") else model.classifier[-1]
    hook = head.register_forward_hook(lambda m, inp, out: feats.append(inp[0].float().cpu()))
    model.eval()
    logits, labels, idx = [], [], []
    for x, y, i in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with autocast(device):
            out = model(x)
        logits.append(out.float().cpu())
        labels.append(y)
        idx.append(i)
    hook.remove()
    F, L = torch.cat(feats).numpy(), torch.cat(logits).numpy()
    Y, I = torch.cat(labels).numpy(), torch.cat(idx).numpy()
    o = np.argsort(I)
    return F[o], L[o], Y[o], I[o]


# ============================================================================ Gaussian model
def ledoit_wolf(X):
    """Shrunk covariance of already-centred data X (n x p), same formula as sklearn."""
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    delta = ((S - mu * np.eye(p)) ** 2).sum() / p
    X2 = X ** 2
    beta = ((X2.T @ X2) / n - S ** 2).sum() / (p * n)
    shrink = 0.0 if delta == 0 else min(beta, delta) / delta
    return (1 - shrink) * S + shrink * mu * np.eye(p), float(shrink)


def whitener(cov):
    """Matrix W with W cov W^T = I, so Mahalanobis distance = Euclidean distance after W."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-8, None)
    return (vecs / np.sqrt(vals)).T


def fit_gaussians(F_train, y_train, n_classes, pca_dim):
    mean = F_train.mean(0)
    Xc = F_train - mean
    # PCA (top pca_dim directions of the training features)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:pca_dim].T if pca_dim and pca_dim < Xc.shape[1] else np.eye(Xc.shape[1])
    Z = Xc @ P
    mus = np.stack([Z[y_train == c].mean(0) for c in range(n_classes)])
    within = Z - mus[y_train]
    cov, shrink = ledoit_wolf(within)
    W = whitener(cov)
    # background Gaussian for relative Mahalanobis
    bg_mu = Z.mean(0)
    bg_cov, _ = ledoit_wolf(Z - bg_mu)
    W_bg = whitener(bg_cov)
    return dict(mean=mean, P=P, mus_w=mus @ W.T, W=W, bg_mu_w=bg_mu @ W_bg.T, W_bg=W_bg, shrink=shrink)


def class_distances(G, F):
    """Squared Mahalanobis distance of each row of F to each class mean -> (n, C)."""
    Zw = ((F - G["mean"]) @ G["P"]) @ G["W"].T
    d = (Zw ** 2).sum(1, keepdims=True) - 2 * Zw @ G["mus_w"].T + (G["mus_w"] ** 2).sum(1)[None]
    Zb = ((F - G["mean"]) @ G["P"]) @ G["W_bg"].T
    d_bg = ((Zb - G["bg_mu_w"]) ** 2).sum(1)
    return np.maximum(d, 0), d_bg


# ============================================================================ metrics
def auroc(score, positive):
    """Probability that a random positive gets a higher score than a random negative (ties = 0.5)."""
    positive = positive.astype(bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s_sorted = score[order]
    i = 0
    while i < len(score):                       # average ranks for ties
        j = i
        while j + 1 < len(score) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def catch_at(score, positive, frac=0.2):
    k = max(1, int(round(frac * len(score))))
    top = np.argsort(-score)[:k]
    return float(positive[top].sum() / max(1, positive.sum()) * 100)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", default="runs/quick_evidence", help="folder written by quick_evidence.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--teacher-arch", default="resnet50")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--pca", type=int, default=256, help="0 = no PCA")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, "mahalanobis")
    os.makedirs(out, exist_ok=True)

    device = get_device(args.gpu)
    rows, info = load_meta(args.data)
    C = info["num_classes"]
    teacher = build_model(args.teacher_arch, C, pretrained=False).to(device)
    teacher.load_state_dict(torch.load(os.path.join(args.run, "teacher.pt"), map_location=device))

    tf = build_transforms(train=False)
    loaders = {}
    for part, sel in [("train", select_rows(rows, "final", "train")), ("test", select_rows(rows, "final", "eval"))]:
        loaders[part] = make_loader(RAMImageDataset(args.data, sel, tf), 128, train=False, workers=args.workers)

    print("Extracting teacher features ...")
    Ftr, Ltr, ytr, _ = extract(teacher, loaders["train"], device)
    Fte, Lte, yte, ite = extract(teacher, loaders["test"], device)
    print(f"  train {Ftr.shape}, test {Fte.shape}, teacher train acc {(Ltr.argmax(1) == ytr).mean() * 100:.1f}%, "
          f"test acc {(Lte.argmax(1) == yte).mean() * 100:.1f}%")

    G = fit_gaussians(Ftr.astype(np.float64), ytr, C, args.pca)
    d, d_bg = class_distances(G, Fte.astype(np.float64))
    print(f"  Gaussian fit: PCA {G['P'].shape[1]} dims, Ledoit-Wolf shrinkage {G['shrink']:.3f}")

    P = softmax(Lte)
    pred = Lte.argmax(1)
    srt = np.sort(Lte, 1)
    dsrt = np.sort(d, 1)
    rows_n = np.arange(len(yte))
    scores = {
        "msp": 1 - P.max(1),
        "entropy": -(P * np.log(P + 1e-12)).sum(1),
        "logit_margin": -(srt[:, -1] - srt[:, -2]),
        "maha_pred": d[rows_n, pred],
        "maha_margin": -(dsrt[:, 1] - dsrt[:, 0]),
        "rel_maha_pred": d[rows_n, pred] - d_bg,
    }

    t_wrong = pred != yte
    # S = teacher wrong, student-alone right (from quick_evidence.py)
    a = np.load(os.path.join(args.run, "student_alone_s0_test.npz"))
    assert (a["idx"] == ite).all(), "test order mismatch with quick_evidence outputs"
    S = t_wrong & (a["logits"].argmax(1) == yte)
    mask_S = S | ~t_wrong                          # S vs teacher-correct

    maha_acc = float((d.argmin(1) == yte).mean() * 100)
    res = dict(n_test=int(len(yte)), teacher_wrong=int(t_wrong.sum()), S=int(S.sum()),
               pca_dims=int(G["P"].shape[1]), shrinkage=G["shrink"],
               mahalanobis_nearest_class_accuracy=maha_acc, scores={})
    print(f"\nTeacher wrong on {t_wrong.sum()} / {len(yte)} test images; S (teacher wrong, student right) = {S.sum()}")
    print(f"Nearest-class-mean (Mahalanobis) accuracy: {maha_acc:.1f}%  (sanity check, should be near teacher accuracy)\n")
    print(f"{'score':16s} {'AUROC errors':>13s} {'AUROC S':>9s} {'catch@20%':>10s}")
    for name, s in scores.items():
        r = dict(auroc_errors=auroc(s, t_wrong), auroc_S=auroc(s[mask_S], S[mask_S]), catch_at_20=catch_at(s, t_wrong))
        res["scores"][name] = r
        print(f"{name:16s} {r['auroc_errors']:13.3f} {r['auroc_S']:9.3f} {r['catch_at_20']:9.1f}%")

    best = max(res["scores"], key=lambda k: res["scores"][k]["auroc_errors"])
    base = res["scores"]["msp"]["auroc_errors"]
    maha_best = max(("maha_pred", "maha_margin", "rel_maha_pred"), key=lambda k: res["scores"][k]["auroc_errors"])
    gain = res["scores"][maha_best]["auroc_errors"] - base
    verdict = (f"Best Mahalanobis score ({maha_best}) {'BEATS' if gain > 0.01 else 'does NOT clearly beat'} "
               f"softmax confidence: AUROC {res['scores'][maha_best]['auroc_errors']:.3f} vs {base:.3f} "
               f"({gain:+.3f}). Best overall: {best}.")
    res["verdict"] = verdict
    print("\n" + verdict)

    save_json(res, os.path.join(out, "results.json"))
    np.savez(os.path.join(out, "scores_test.npz"), idx=ite, labels=yte, teacher_pred=pred, S=S, **scores)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = scores["maha_pred"]
    bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 50)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(s[~t_wrong], bins=bins, alpha=0.6, density=True, label="teacher correct", color="#2F3A8F")
    ax.hist(s[t_wrong], bins=bins, alpha=0.6, density=True, label="teacher wrong", color="#C23B2E")
    ax.set_xlabel("Mahalanobis distance to the teacher's predicted class")
    ax.set_ylabel("density")
    ax.set_title(f"{info['dataset'].upper()} test set: AUROC {res['scores']['maha_pred']['auroc_errors']:.3f} "
                 f"(softmax confidence {base:.3f})", fontsize=10)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "hist_maha_pred.png"), dpi=250)
    print(f"\nSaved results.json, scores_test.npz and hist_maha_pred.png in {out}")


if __name__ == "__main__":
    main()
