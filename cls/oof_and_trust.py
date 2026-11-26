"""
Out-of-fold predictions + class-pair trust table for CPT-KD.

What it does
  1. For each of the 5 folds of the training pool, trains
       - a teacher (ResNet-50) on the other 4 folds, and
       - a student-alone (ResNet-18) on the other 4 folds,
     and saves each model's predictions on the fold it never saw ("out-of-fold", cross-fitting).
  2. Also trains one normal teacher on the whole training pool (the teacher used later for KD).
  3. Builds the trust table from the out-of-fold predictions:
       T[y, c] = teacher's average probability on class c, over training images of true class y
       S[y, c] = same for the student trained alone
       trust[y, c] = min(1, (S + eps) / (T + eps))   for c != y,   trust[y, y] = 1
     trust near 1: the teacher is no more confused than the student on this pair (keep it)
     trust near 0: a confusion only the teacher has (CPT-KD will turn it down)
     Built at temperature 1 and at the KD temperature (default 4), saved as trust_T1.npy / trust_T4.npy.
     Also saves a per-class trust vector (the class-level baseline).

Resumable: every finished model is saved and skipped on the next run.

Using both GPUs (run these two at the same time, e.g. in two tmux windows):
  python oof_and_trust.py --data data/tea_kd/cub --models teacher --gpu 1
  python oof_and_trust.py --data data/tea_kd/cub --models student --gpu 2
Whichever finishes last builds the trust table automatically. To rebuild it later:
  python oof_and_trust.py --data data/tea_kd/cub --models none

Outputs in runs/<dataset>/<phase>/oof/:
  oof_teacher_fold{k}.npz, oof_student_fold{k}.npz   out-of-fold logits (idx, logits, labels)
  teacher_full.pt, teacher_full_test.npz             the normal teacher for KD (+ its test logits)
  trust_T1.npy, trust_T4.npy                         C x C trust tables
  class_trust_T1.npy, class_trust_T4.npy             per-class trust (baseline)
  lowest_trust_pairs.csv                             the 40 most suspicious class pairs (sanity check)
  summary.json, lowest_trust_pairs.png
"""
import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import Subset

from common import (RAMImageDataset, accuracy, build_transforms, get_device, load_meta, make_loader,
                    predict, save_json, select_rows)
from quick_evidence import train_model


# ============================================================================ training
def run_fold_models(kind, arch, args, rows, info, device, out):
    """Train one model per fold on the other folds; save logits on the held-out fold."""
    fold_key = f"fold_{args.phase}"
    pool = select_rows(rows, args.phase, "train")
    folds = sorted({r[fold_key] for r in pool})
    train_ds = RAMImageDataset(args.data, pool, build_transforms(train=True))
    eval_ds = RAMImageDataset(args.data, pool, build_transforms(train=False))
    pos = {r["idx"]: i for i, r in enumerate(pool)}
    epochs = args.teacher_epochs if kind == "teacher" else args.student_epochs
    for k in folds:
        path = os.path.join(out, f"oof_{kind}_fold{k}.npz")
        if os.path.exists(path):
            print(f"[{kind} fold {k}] already done, skipping")
            continue
        tr = [pos[r["idx"]] for r in pool if r[fold_key] != k]
        ho = [pos[r["idx"]] for r in pool if r[fold_key] == k]
        seed = k if kind == "teacher" else 100 + k
        model = train_model(arch, Subset(train_ds, tr), info["num_classes"], device, args, seed, epochs,
                            tag=f"{kind} fold {k}")
        logits, labels, idx = predict(model, make_loader(Subset(eval_ds, ho), 128, False, args.workers), device)
        np.savez(path, idx=idx, logits=logits, labels=labels)
        print(f"[{kind} fold {k}] accuracy on its unseen fold: {accuracy(logits, labels):.2f}%")
        del model
        torch.cuda.empty_cache()


def run_full_teacher(args, rows, info, device, out):
    ckpt, npz = os.path.join(out, "teacher_full.pt"), os.path.join(out, "teacher_full_test.npz")
    if os.path.exists(ckpt) and os.path.exists(npz):
        print("[full teacher] already done, skipping")
        return
    pool = select_rows(rows, args.phase, "train")
    ev = select_rows(rows, args.phase, "eval")
    model = train_model(args.teacher_arch, RAMImageDataset(args.data, pool, build_transforms(train=True)),
                        info["num_classes"], device, args, seed=0, epochs=args.teacher_epochs, tag="full teacher")
    logits, labels, idx = predict(model, make_loader(RAMImageDataset(args.data, ev, build_transforms(False)),
                                                     128, False, args.workers), device)
    torch.save(model.state_dict(), ckpt)
    np.savez(npz, idx=idx, logits=logits, labels=labels)
    split = "val" if args.phase == "dev" else "test"
    print(f"[full teacher] {split} accuracy {accuracy(logits, labels):.2f}%")


# ============================================================================ trust table
def load_oof(out, kind, n_folds):
    parts = [np.load(os.path.join(out, f"oof_{kind}_fold{k}.npz")) for k in range(n_folds)]
    idx = np.concatenate([p["idx"] for p in parts])
    o = np.argsort(idx)
    return (idx[o], np.concatenate([p["logits"] for p in parts])[o],
            np.concatenate([p["labels"] for p in parts])[o])


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def soft_confusion(P, y, C):
    """M[y, c] = average probability on class c over images whose true class is y."""
    M = np.zeros((C, C))
    for c in range(C):
        M[c] = P[y == c].mean(0)
    return M


def build_trust(args, info, out):
    C, names = info["num_classes"], info["class_names"]
    n_folds = len([f for f in os.listdir(out) if f.startswith("oof_teacher_fold")])
    ti, tl, y = load_oof(out, "teacher", n_folds)
    si, sl, y2 = load_oof(out, "student", n_folds)
    assert (ti == si).all() and (y == y2).all(), "teacher and student out-of-fold sets differ"
    eps = args.eps if args.eps is not None else 0.1 / C
    off = ~np.eye(C, dtype=bool)
    summary = dict(n_train=int(len(y)), n_folds=n_folds, eps=eps,
                   oof_teacher_acc=accuracy(tl, y), oof_student_acc=accuracy(sl, y))
    print(f"\nOut-of-fold accuracy on training images: teacher {summary['oof_teacher_acc']:.1f}%, "
          f"student alone {summary['oof_student_acc']:.1f}%")

    for T in args.trust_temps:
        Tm = soft_confusion(softmax(tl, T), y, C)
        Sm = soft_confusion(softmax(sl, T), y, C)
        R = np.minimum(1.0, (Sm + eps) / (Tm + eps))
        np.fill_diagonal(R, 1.0)
        # class-level baseline: one trust value per true class (all wrong classes together)
        Rc = np.minimum(1.0, ((Sm * off).sum(1) + eps) / ((Tm * off).sum(1) + eps))
        tag = f"T{T:g}"
        np.save(os.path.join(out, f"trust_{tag}.npy"), R.astype(np.float32))
        np.save(os.path.join(out, f"class_trust_{tag}.npy"), Rc.astype(np.float32))
        np.save(os.path.join(out, f"soft_conf_teacher_{tag}.npy"), Tm.astype(np.float32))
        np.save(os.path.join(out, f"soft_conf_student_{tag}.npy"), Sm.astype(np.float32))
        r_off = R[off]
        summary[tag] = dict(mean_trust=float(r_off.mean()),
                            pairs_below_0_5=int((r_off < 0.5).sum()),
                            pairs_below_0_9=int((r_off < 0.9).sum()),
                            total_pairs=int(off.sum()))
        print(f"trust {tag}: mean {r_off.mean():.3f}, pairs < 0.5: {summary[tag]['pairs_below_0_5']}, "
              f"pairs < 0.9: {summary[tag]['pairs_below_0_9']} (of {off.sum()})")

    # sanity check table at the KD temperature: the most suspicious pairs, with hard-confusion counts
    T = args.trust_temps[-1]
    tag = f"T{T:g}"
    R = np.load(os.path.join(out, f"trust_{tag}.npy"))
    Tm = np.load(os.path.join(out, f"soft_conf_teacher_{tag}.npy"))
    Sm = np.load(os.path.join(out, f"soft_conf_student_{tag}.npy"))
    tp, sp = tl.argmax(1), sl.argmax(1)
    # rank by how much probability mass trust removes (big confusion x low trust)
    removed = Tm * (1 - R)
    order = np.dstack(np.unravel_index(np.argsort(-removed, axis=None), removed.shape))[0][:40]
    with open(os.path.join(out, "lowest_trust_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_class", "confused_with", f"teacher_avg_prob_{tag}", f"student_avg_prob_{tag}",
                    "trust", "teacher_oof_mistakes", "student_oof_mistakes", "images_of_true_class"])
        for a, c in order:
            w.writerow([names[a], names[c], f"{Tm[a, c]:.4f}", f"{Sm[a, c]:.4f}", f"{R[a, c]:.3f}",
                        int(((y == a) & (tp == c)).sum()), int(((y == a) & (sp == c)).sum()), int((y == a).sum())])
    print(f"\nMost suspicious pairs ({tag}); these should look like real look-alike classes:")
    for a, c in order[:12]:
        print(f"  {names[a]:<32s} -> {names[c]:<32s} trust {R[a, c]:.2f}   teacher mistakes "
              f"{((y == a) & (tp == c)).sum()}, student mistakes {((y == a) & (sp == c)).sum()}")
    save_json(summary, os.path.join(out, "summary.json"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = order[:15][::-1]
    labels = [f"{names[a]} → {names[c]}".replace("_", " ") for a, c in top]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(range(len(top)), [Tm[a, c] for a, c in top], color="#C23B2E", label="teacher (out-of-fold)")
    ax.barh(range(len(top)), [Sm[a, c] for a, c in top], color="#0F7D5A", alpha=0.85, label="student alone (out-of-fold)")
    for i, (a, c) in enumerate(top):
        ax.text(Tm[a, c], i, f"  trust {R[a, c]:.2f}", va="center", fontsize=8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"average probability on the confused class (temperature {T:g})")
    ax.set_title("Class pairs where the teacher is more confused than the student", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "lowest_trust_pairs.png"), dpi=200)
    print(f"\nSaved trust tables, lowest_trust_pairs.csv/.png and summary.json in {out}")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None, help="default: runs/<dataset>/<phase>/oof")
    ap.add_argument("--phase", default="final", choices=["dev", "final"])
    ap.add_argument("--models", default="both", choices=["both", "teacher", "student", "none"],
                    help="'teacher' = 5 fold teachers + full teacher; 'student' = 5 fold students; 'none' = only build trust")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--teacher-arch", default="resnet50")
    ap.add_argument("--student-arch", default="resnet18")
    ap.add_argument("--teacher-epochs", type=int, default=40)
    ap.add_argument("--student-epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--trust-temps", type=float, nargs="+", default=[1.0, 4.0],
                    help="temperatures for the trust table; the LAST one is used for the sanity-check table")
    ap.add_argument("--eps", type=float, default=None, help="smoothing; default 0.1 / number of classes")
    ap.add_argument("--no-pretrained", action="store_true", help="debug only")
    args = ap.parse_args()
    args.alpha, args.T = 0.0, 1.0            # not used here (no KD), required by train_model

    rows, info = load_meta(args.data)
    out = args.out or os.path.join("runs", info["dataset"], args.phase, "oof")
    os.makedirs(out, exist_ok=True)
    save_json(vars(args), os.path.join(out, f"args_{args.models}.json"))

    if args.models != "none":
        device = get_device(args.gpu)
        if args.models in ("both", "teacher"):
            run_fold_models("teacher", args.teacher_arch, args, rows, info, device, out)
            run_full_teacher(args, rows, info, device, out)
        if args.models in ("both", "student"):
            run_fold_models("student", args.student_arch, args, rows, info, device, out)

    n_t = len([f for f in os.listdir(out) if f.startswith("oof_teacher_fold")])
    n_s = len([f for f in os.listdir(out) if f.startswith("oof_student_fold")])
    n_folds = len({r[f"fold_{args.phase}"] for r in select_rows(rows, args.phase, "train")})
    if n_t == n_s == n_folds:
        build_trust(args, info, out)
    else:
        print(f"\nTrust table not built yet: {n_t}/{n_folds} teacher folds and {n_s}/{n_folds} student folds done. "
              f"It is built automatically when both are complete (or run with --models none).")


if __name__ == "__main__":
    main()
