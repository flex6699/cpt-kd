"""
Quick preliminary evidence for the poster (CUB-200-2011, 1 seed, ~1 hour on one A5000).

Question: when the teacher is wrong but the student could be right, does standard KD
          make the student copy the teacher's mistake?

Runs (each is skipped if its results already exist, so the script can be re-started):
  1. teacher        ResNet-50, fine-tuned with cross-entropy
  2. alone          ResNet-18 trained alone (no teacher), seed 0
  3. alone_control  ResNet-18 trained alone, seed 1  -> how much predictions change by chance
  4. kd             ResNet-18 trained with standard KD from the teacher, seed 0

Then, on the test set:
  * S = images where the teacher is WRONG and the student-alone is RIGHT
  * on S, how often the KD student switches to the teacher's wrong class,
    compared with the control student (a different seed with no teacher)
  * teacher-error inheritance: of all test images the teacher gets wrong,
    % where the student predicts the same wrong class (alone vs. KD)

Outputs in --out:
  results.json        all numbers
  poster_text.txt     a ready-to-paste sentence for the poster
  examples.csv        the clearest real examples (image, label, teacher / student predictions)
  example_*.png       poster-ready figures of those examples (image + probability bars)

Usage
  conda activate teakd
  python quick_evidence.py --data /path/to/tea_kd_data/cub --out runs/quick_evidence --gpu 1
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (RAMImageDataset, accuracy, autocast, build_model, build_optimizer,
                    build_transforms, get_device, load_meta, make_loader, predict,
                    save_json, seed_everything, select_rows, set_lr)


# ============================================================================ training
def train_model(arch, train_ds, num_classes, device, args, seed, epochs, teacher=None, tag=""):
    """Fine-tune one model. If `teacher` is given, add the standard KD term."""
    seed_everything(seed)
    model = build_model(arch, num_classes, pretrained=not args.no_pretrained).to(device)
    opt = build_optimizer(model, lr=args.lr, weight_decay=args.wd)
    loader = make_loader(train_ds, args.batch_size, train=True, workers=args.workers)
    total, warmup, step = epochs * len(loader), len(loader), 0
    if teacher is not None:
        teacher.eval()
    print(f"\n[{tag}] {arch}, {epochs} epochs, {len(train_ds)} images, seed {seed}"
          + (f", KD alpha={args.alpha} T={args.T}" if teacher is not None else ""))
    for ep in range(epochs):
        model.train()
        t0, n, correct, loss_sum = time.time(), 0, 0, 0.0
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            set_lr(opt, step, total, warmup)
            with autocast(device):
                logits = model(x)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
            logits = logits.float()
            loss = F.cross_entropy(logits, y)
            if teacher is not None:
                T = args.T
                kd = F.kl_div(F.log_softmax(logits / T, 1), F.softmax(t_logits.float() / T, 1),
                              reduction="batchmean") * T * T
                loss = loss + args.alpha * kd
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"[{tag}] loss became {loss.item()} at epoch {ep + 1}; lower --lr")
            loss.backward()
            opt.step()
            step += 1
            n += y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            loss_sum += loss.item() * y.numel()
        if ep == 0 or (ep + 1) % 5 == 0 or ep + 1 == epochs:
            print(f"  epoch {ep + 1:3d}/{epochs}  loss {loss_sum / n:.3f}  train acc {100 * correct / n:5.1f}%"
                  f"  ({time.time() - t0:.0f}s/epoch)")
    return model


def get_run(name, arch, train_ds, test_loader, num_classes, device, args, seed, epochs, teacher=None):
    """Train (or load) a run; returns (model, test_logits, test_labels, test_idx)."""
    ckpt = os.path.join(args.out, f"{name}.pt")
    npz = os.path.join(args.out, f"{name}_test.npz")
    model = build_model(arch, num_classes, pretrained=False).to(device)
    if os.path.exists(ckpt) and os.path.exists(npz):
        print(f"[{name}] found existing results, skipping training")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        d = np.load(npz)
        return model, d["logits"], d["labels"], d["idx"]
    model = train_model(arch, train_ds, num_classes, device, args, seed, epochs, teacher, tag=name)
    logits, labels, idx = predict(model, test_loader, device)
    torch.save(model.state_dict(), ckpt)
    np.savez(npz, logits=logits, labels=labels, idx=idx)
    print(f"[{name}] test accuracy {accuracy(logits, labels):.2f}%")
    return model, logits, labels, idx


# ============================================================================ analysis
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def analyse(t, a, c, k, y):
    """t/a/c/k = test logits of teacher / alone / alone_control / kd student; y = labels."""
    pt, pa, pc, pk = (x.argmax(1) for x in (t, a, c, k))
    t_wrong = pt != y
    S = t_wrong & (pa == y)                       # teacher wrong, student-alone right
    nS = int(S.sum())

    def on_S(pred):
        return dict(copies_teacher=float((pred[S] == pt[S]).mean() * 100) if nS else 0.0,
                    stays_correct=float((pred[S] == y[S]).mean() * 100) if nS else 0.0,
                    other_wrong=float(((pred[S] != y[S]) & (pred[S] != pt[S])).mean() * 100) if nS else 0.0)

    def inheritance(pred):                        # teacher-error inheritance on all teacher-wrong images
        return float((pred[t_wrong] == pt[t_wrong]).mean() * 100) if t_wrong.any() else 0.0

    return dict(
        n_test=int(len(y)),
        acc=dict(teacher=accuracy(t, y), student_alone=accuracy(a, y),
                 student_alone_control=accuracy(c, y), student_kd=accuracy(k, y)),
        teacher_wrong_pct=float(t_wrong.mean() * 100),
        teacher_wrong_student_right=dict(count=nS, pct_of_test=float(S.mean() * 100),
                                         pct_of_teacher_errors=float(nS / max(1, t_wrong.sum()) * 100)),
        on_S_kd_student=on_S(pk),
        on_S_control_student=on_S(pc),
        teacher_error_inheritance=dict(student_alone=inheritance(pa), student_alone_control=inheritance(pc),
                                       student_kd=inheritance(pk)),
    )


def pick_examples(t, a, k, y, n):
    """Clearest cases: teacher confidently wrong, alone-student right, KD student copies teacher."""
    Pt, Pa, Pk = softmax(t), softmax(a), softmax(k)
    pt, pa, pk = t.argmax(1), a.argmax(1), k.argmax(1)
    cand = np.where((pt != y) & (pa == y) & (pk == pt))[0]
    score = Pt[cand, pt[cand]] * Pa[cand, y[cand]]
    return cand[np.argsort(-score)][:n], Pt, Pa, Pk


def plot_example(path, img, label_name, names, Pt, Pa, Pk, y, t_arch="resnet50", s_arch="resnet18"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = list(np.argsort(-Pt)[:4])
    if y not in top:
        top[-1] = y
    nice = [names[i].replace("_", " ") for i in top]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), gridspec_kw=dict(width_ratios=[1, 1.3, 1.3, 1.3]))
    axes[0].imshow(img)
    axes[0].set_title(f"Label: {label_name.replace('_', ' ')}", fontsize=11)
    axes[0].axis("off")
    for ax, P, title, base in [(axes[1], Pt, f"Teacher ({t_arch})", "#2F3A8F"),
                               (axes[2], Pa, f"Student alone ({s_arch})", "#0F7D5A"),
                               (axes[3], Pk, f"Student after KD ({s_arch})", "#7A4FA3")]:
        vals = [P[i] for i in top]
        pred = int(P.argmax())
        colors = ["#C23B2E" if (i == pred and i != y) else base for i in top]
        ax.barh(range(len(top))[::-1], vals, color=colors)
        ax.set_yticks(range(len(top))[::-1])
        ax.set_yticklabels(nice, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=11)
        for j, v in enumerate(vals):
            ax.text(min(v + 0.02, 0.85), len(top) - 1 - j, f"{v:.2f}", va="center", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="output folder of prepare_data.py for CUB")
    ap.add_argument("--out", default="runs/quick_evidence")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--teacher-arch", default="resnet50")
    ap.add_argument("--student-arch", default="resnet18")
    ap.add_argument("--teacher-epochs", type=int, default=40)
    ap.add_argument("--student-epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01, help="backbone lr; new head uses 10x")
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="KD weight: CE + alpha*T^2*KL")
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--no-pretrained", action="store_true", help="debug only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    save_json(vars(args), os.path.join(args.out, "args.json"))

    device = get_device(args.gpu)
    rows, info = load_meta(args.data)
    C, names = info["num_classes"], info["class_names"]
    train_rows = select_rows(rows, "final", "train")      # official train set (train + held-out val)
    test_rows = select_rows(rows, "final", "eval")        # official test set
    print(f"train {len(train_rows)} images, test {len(test_rows)} images, {C} classes")
    train_ds = RAMImageDataset(args.data, train_rows, build_transforms(train=True))
    test_ds = RAMImageDataset(args.data, test_rows, build_transforms(train=False))
    test_loader = make_loader(test_ds, 128, train=False, workers=args.workers)

    t0 = time.time()
    common = (train_ds, test_loader, C, device, args)
    teacher, t, y, idx = get_run("teacher", args.teacher_arch, *common, seed=0, epochs=args.teacher_epochs)
    _, a, _, _ = get_run("student_alone_s0", args.student_arch, *common, seed=0, epochs=args.student_epochs)
    _, c, _, _ = get_run("student_alone_s1", args.student_arch, *common, seed=1, epochs=args.student_epochs)
    _, k, _, _ = get_run("student_kd_s0", args.student_arch, *common, seed=0, epochs=args.student_epochs, teacher=teacher)
    print(f"\nAll runs done in {(time.time() - t0) / 60:.1f} min")

    res = analyse(t, a, c, k, y)
    save_json(res, os.path.join(args.out, "results.json"))

    S = res["teacher_wrong_student_right"]
    kdS, ctS = res["on_S_kd_student"], res["on_S_control_student"]
    inh = res["teacher_error_inheritance"]
    report = f"""
================ PRELIMINARY RESULT (CUB-200-2011, 1 seed) ================
Test accuracy:  teacher {res['acc']['teacher']:.1f}%  |  student alone {res['acc']['student_alone']:.1f}%
                student alone (seed 1) {res['acc']['student_alone_control']:.1f}%  |  student with KD {res['acc']['student_kd']:.1f}%

Teacher wrong on {res['teacher_wrong_pct']:.1f}% of test images.
Teacher WRONG but student-alone RIGHT: {S['count']} images ({S['pct_of_test']:.1f}% of test,
  {S['pct_of_teacher_errors']:.1f}% of the teacher's errors).

On those {S['count']} images:
                         copies teacher's wrong class   stays correct   other error
  student with KD              {kdS['copies_teacher']:5.1f}%                  {kdS['stays_correct']:5.1f}%        {kdS['other_wrong']:5.1f}%
  control (alone, seed 1)      {ctS['copies_teacher']:5.1f}%                  {ctS['stays_correct']:5.1f}%        {ctS['other_wrong']:5.1f}%

Teacher-error inheritance (of all teacher-wrong test images, % with the SAME wrong class):
  student alone {inh['student_alone']:.1f}%  |  control {inh['student_alone_control']:.1f}%  |  student with KD {inh['student_kd']:.1f}%
==========================================================================="""
    print(report)

    sentence = (f"Preliminary result (CUB-200-2011, 1 seed): on {S['pct_of_test']:.1f}% of test images the "
                f"teacher ({args.teacher_arch}) is wrong while a {args.student_arch} trained alone is right. After standard KD, the "
                f"student copies the teacher's wrong class on {kdS['copies_teacher']:.0f}% of these images "
                f"(vs. {ctS['copies_teacher']:.0f}% for a second student trained alone), and repeats the teacher's "
                f"exact mistake on {inh['student_kd']:.0f}% of all teacher errors (vs. {inh['student_alone']:.0f}% "
                f"without KD).")
    with open(os.path.join(args.out, "poster_text.txt"), "w") as f:
        f.write(report + "\n\nPoster sentence:\n" + sentence + "\n")
    print("\nPoster sentence:\n" + sentence)

    # ---- real examples for the poster figure
    ex, Pt, Pa, Pk = pick_examples(t, a, k, y, args.n_examples)
    by_idx = {r["idx"]: r for r in test_rows}
    eval_tf = build_transforms(train=False)
    from PIL import Image
    with open(os.path.join(args.out, "examples.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "file", "label", "teacher_pred", "p_teacher_pred", "alone_pred",
                    "p_alone_label", "kd_pred", "p_kd_pred", "figure"])
        for r_i, j in enumerate(ex):
            row = by_idx[int(idx[j])]
            img = Image.open(os.path.join(args.data, row["file"])).convert("RGB")
            img = eval_tf.transforms[1](eval_tf.transforms[0](img))     # resize + center crop, as the models saw it
            fig_path = os.path.join(args.out, f"example_{r_i + 1}.png")
            plot_example(fig_path, img, names[y[j]], names, Pt[j], Pa[j], Pk[j], int(y[j]),
                         args.teacher_arch, args.student_arch)
            w.writerow([r_i + 1, row["file"], names[y[j]], names[t[j].argmax()], f"{Pt[j].max():.3f}",
                        names[a[j].argmax()], f"{Pa[j, y[j]]:.3f}", names[k[j].argmax()],
                        f"{Pk[j].max():.3f}", os.path.basename(fig_path)])
    print(f"\nSaved {len(ex)} example figures, examples.csv, results.json and poster_text.txt in {args.out}")
    if len(ex) == 0:
        print("No image matched all three conditions; see results.json for the overall numbers.")


if __name__ == "__main__":
    main()
