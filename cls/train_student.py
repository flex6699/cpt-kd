"""
Train ONE student with one method and evaluate it.

Needs the outputs of oof_and_trust.py (runs/<dataset>/<phase>/oof/):
  teacher_full.pt, teacher_full_test.npz, trust_T1.npy / trust_T4.npy, class_trust_T*.npy,
  and (for --teacher-source oof) oof_teacher_fold*.npz

Examples
  python train_student.py --data data/tea_kd/cub --method kd  --gpu 1
  python train_student.py --data data/tea_kd/cub --method cpt --gpu 2
  python train_student.py --data data/tea_kd/cub --method cpt --trust-temp 4 --gpu 1     # ablation
  python train_student.py --data data/tea_kd/cub --method pergate --teacher-source oof   # per-image baseline
                                                                                          # with out-of-fold teachers
Each run is skipped if its results.json already exists.

Outputs in runs/<dataset>/<phase>/students/<method>_<student>[_tag]_s<seed>/:
  results.json   accuracy, mean per-class accuracy, teacher-error inheritance, flagged-pair inheritance,
                 accuracy on teacher-correct images, student-teacher agreement, ECE
  test_logits.npz
"""
import argparse
import os
import time

import numpy as np
import torch

from common import (RAMImageDataset, autocast, build_model, build_optimizer, build_transforms,
                    get_device, load_meta, make_loader, predict, save_json, seed_everything,
                    select_rows, set_lr)
from losses import METHODS, NEEDS_TEACHER, compute_loss


# ============================================================================ metrics
def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(logits, y, bins=15):
    P = softmax_np(logits)
    conf, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total * 100)


def evaluate(s_logits, y, t_logits, trust, flag_below=0.5):
    """All poster metrics for one student, given the teacher's test logits and the trust table."""
    ps, pt = s_logits.argmax(1), t_logits.argmax(1)
    C = s_logits.shape[1]
    t_wrong = pt != y
    per_class = [np.mean(ps[y == c] == c) for c in range(C) if (y == c).any()]
    flagged = t_wrong & (trust[y, pt] < flag_below)       # teacher mistakes on low-trust pairs
    res = dict(
        top1=float((ps == y).mean() * 100),
        mean_per_class=float(np.mean(per_class) * 100),
        worst10_class=float(np.mean(np.sort(per_class)[:10]) * 100),
        inheritance=float((ps[t_wrong] == pt[t_wrong]).mean() * 100) if t_wrong.any() else float("nan"),
        flagged_pair_inheritance=float((ps[flagged] == pt[flagged]).mean() * 100) if flagged.any() else float("nan"),
        n_flagged_test=int(flagged.sum()),
        acc_teacher_correct=float((ps[~t_wrong] == y[~t_wrong]).mean() * 100),
        agreement=float((ps == pt).mean() * 100),
        ece=ece(s_logits, y),
    )
    return res


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--phase", default="final", choices=["dev", "final"])
    ap.add_argument("--student-arch", default="resnet18")
    ap.add_argument("--teacher-arch", default="resnet50")
    ap.add_argument("--teacher-source", default="full", choices=["full", "oof"],
                    help="full = normal teacher (online); oof = cached out-of-fold teacher logits")
    ap.add_argument("--oof-dir", default=None, help="default: runs/<dataset>/<phase>/oof")
    ap.add_argument("--trust-temp", default="1", choices=["1", "4"], help="which trust table CPT-KD uses")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="KD weight: CE + alpha * KD")
    ap.add_argument("--T", type=float, default=4.0, help="KD temperature")
    ap.add_argument("--dkd-alpha", type=float, default=1.0)
    ap.add_argument("--dkd-beta", type=float, default=8.0)
    ap.add_argument("--dkd-warmup", type=int, default=5, help="epochs")
    ap.add_argument("--cpt-power", type=float, default=1.0, help="CPT-KD: use trust^power")
    ap.add_argument("--cpt-keep-target", action="store_true",
                    help="CPT-KD: keep the true-class probability; spread removed mass over wrong classes")
    ap.add_argument("--pair-lambda", type=float, default=1.0, help="cpt_pair: weight of the pair penalty")
    ap.add_argument("--tag", default="", help="extra name for ablation runs")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="debug only")
    args = ap.parse_args()

    rows, info = load_meta(args.data)
    C, ds_name = info["num_classes"], info["dataset"]
    oof_dir = args.oof_dir or os.path.join("runs", ds_name, args.phase, "oof")
    name = f"{args.method}_{args.student_arch}"
    if args.method in ("cpt", "cpt_dkd", "cpt_wsl", "cpt_pair", "class_trust"):
        if args.trust_temp != "1":
            name += f"_trustT{args.trust_temp}"
        if args.cpt_power != 1.0:
            name += f"_pow{args.cpt_power:g}"
        if args.cpt_keep_target:
            name += "_keept"
    if args.teacher_source == "oof":
        name += "_oofteacher"
    if args.tag:
        name += f"_{args.tag}"
    name += f"_s{args.seed}"
    out = os.path.join(args.out_root or os.path.join("runs", ds_name, args.phase, "students"), name)
    if os.path.exists(os.path.join(out, "results.json")):
        print(f"{name}: already done, skipping")
        return
    os.makedirs(out, exist_ok=True)
    save_json(vars(args), os.path.join(out, "args.json"))

    device = get_device(args.gpu)
    seed_everything(args.seed)
    trust_np = np.load(os.path.join(oof_dir, f"trust_T{args.trust_temp}.npy"))
    trust_eval = np.load(os.path.join(oof_dir, "trust_T1.npy"))          # flagged pairs always from T1
    trust = torch.tensor(trust_np, device=device)
    class_trust = torch.tensor(np.load(os.path.join(oof_dir, f"class_trust_T{args.trust_temp}.npy")), device=device)
    cfg = dict(alpha=args.alpha, T=args.T, dkd_alpha=args.dkd_alpha, dkd_beta=args.dkd_beta,
               dkd_warmup=args.dkd_warmup, cpt_power=args.cpt_power, cpt_keep_target=args.cpt_keep_target, pair_lambda=args.pair_lambda)

    train_rows = select_rows(rows, args.phase, "train")
    eval_rows = select_rows(rows, args.phase, "eval")
    train_ds = RAMImageDataset(args.data, train_rows, build_transforms(train=True))
    eval_loader = make_loader(RAMImageDataset(args.data, eval_rows, build_transforms(train=False)),
                              128, False, args.workers)
    loader = make_loader(train_ds, args.batch_size, True, args.workers)

    teacher, oof_logits = None, None
    if args.method in NEEDS_TEACHER:
        if args.teacher_source == "full":
            teacher = build_model(args.teacher_arch, C, pretrained=False).to(device)
            teacher.load_state_dict(torch.load(os.path.join(oof_dir, "teacher_full.pt"), map_location=device))
            teacher.eval()
        else:
            n_folds = len([f for f in os.listdir(oof_dir) if f.startswith("oof_teacher_fold")])
            oof_logits = torch.zeros(len(rows), C)
            for k in range(n_folds):
                d = np.load(os.path.join(oof_dir, f"oof_teacher_fold{k}.npz"))
                oof_logits[torch.from_numpy(d["idx"])] = torch.from_numpy(d["logits"])
            oof_logits = oof_logits.to(device)

    student = build_model(args.student_arch, C, pretrained=not args.no_pretrained).to(device)
    opt = build_optimizer(student, lr=args.lr, weight_decay=args.wd)
    total, warmup, step = args.epochs * len(loader), len(loader), 0
    print(f"\n[{name}] {len(train_ds)} training images, {args.epochs} epochs")
    t_start = time.time()
    for ep in range(args.epochs):
        student.train()
        t0, n, correct, loss_sum, extra = time.time(), 0, 0, 0.0, {}
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            set_lr(opt, step, total, warmup)
            t_logits = None
            with autocast(device):
                s_logits = student(x)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
            if oof_logits is not None:
                t_logits = oof_logits[idx.to(device)]
            loss, inf = compute_loss(args.method, s_logits.float(), y,
                                     None if t_logits is None else t_logits.float(),
                                     cfg, trust, class_trust, epoch=ep)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss became {loss.item()} at epoch {ep + 1}; try a lower --lr")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            n += y.numel()
            correct += (s_logits.argmax(1) == y).sum().item()
            loss_sum += loss.item() * y.numel()
            for k, v in inf.items():
                extra[k] = extra.get(k, 0.0) + v * y.numel()
        if ep == 0 or (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
            ex = "".join(f"  {k} {v / n:.3f}" for k, v in extra.items())
            print(f"  epoch {ep + 1:3d}/{args.epochs}  loss {loss_sum / n:.3f}  train acc {100 * correct / n:5.1f}%"
                  f"{ex}  ({time.time() - t0:.0f}s/epoch)")

    s_logits, y, idx = predict(student, eval_loader, device)
    np.savez(os.path.join(out, "test_logits.npz"), logits=s_logits, labels=y, idx=idx)
    t = np.load(os.path.join(oof_dir, "teacher_full_test.npz"))
    assert (t["idx"] == idx).all(), "teacher and student evaluation sets differ"
    res = evaluate(s_logits, y, t["logits"], trust_eval)
    res.update(method=args.method, seed=args.seed, student=args.student_arch, trust_temp=args.trust_temp,
               cpt_power=args.cpt_power, cpt_keep_target=args.cpt_keep_target, dataset=ds_name,
               teacher_source=args.teacher_source, minutes=round((time.time() - t_start) / 60, 1),
               eval_split="val" if args.phase == "dev" else "test")
    save_json(res, os.path.join(out, "results.json"))
    print(f"\n[{name}] top-1 {res['top1']:.2f}%  mean/class {res['mean_per_class']:.2f}%  "
          f"inheritance {res['inheritance']:.1f}%  flagged-pair inheritance {res['flagged_pair_inheritance']:.1f}% "
          f"(n={res['n_flagged_test']})  acc on teacher-correct {res['acc_teacher_correct']:.2f}%  "
          f"ECE {res['ece']:.2f}  ({res['minutes']} min)")


if __name__ == "__main__":
    main()
