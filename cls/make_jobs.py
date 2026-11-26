"""
Write the job file for the full experiment, once you have chosen the final CPT-KD version.

All methods x 3 seeds x 2 students x 2 datasets (~108 runs; already finished runs are skipped
automatically by train_student.py, so CUB ResNet-18 runs you already have are not repeated).

Usage
  python make_jobs.py --keep-target                 # the chosen version (CPT-KD keep-target)
  python make_jobs.py                               # plain CPT-KD
  python make_jobs.py --power 2 --keep-target
  (advanced: --cpt-flags="--cpt-keep-target", note the = sign)
  -> writes jobs_full.txt, then:
  python run_queue.py jobs_full.txt --gpus 1 2 --per-gpu 2

Before running Aircraft, the out-of-fold step must be done for it (oof_and_trust.py), which you already did.
"""
import argparse

METHODS = [
    ("ce", ""),
    ("kd", ""),
    ("dkd", ""),
    ("wsl", ""),
    ("ka", ""),
    ("pergate", "--teacher-source oof"),
    ("class_trust", "CPT"),
    ("cpt", "CPT"),
    ("cpt_dkd", "CPT"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-target", action="store_true", help="CPT-KD keep-target version")
    ap.add_argument("--power", type=float, default=1.0, help="CPT-KD trust power")
    ap.add_argument("--cpt-flags", default="", help='raw extra CPT-KD flags; write as --cpt-flags="..."')
    ap.add_argument("--datasets", nargs="+", default=["cub", "aircraft"])
    ap.add_argument("--students", nargs="+", default=["resnet18", "mobilenet_v2"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data-root", default="data/tea_kd")
    ap.add_argument("--out", default="jobs_full.txt")
    args = ap.parse_args()
    flags = [args.cpt_flags.strip()] if args.cpt_flags.strip() else []
    if args.keep_target and "--cpt-keep-target" not in args.cpt_flags:
        flags.append("--cpt-keep-target")
    if args.power != 1.0:
        flags.append(f"--cpt-power {args.power:g}")
    args.cpt_flags = " ".join(flags)

    lines = [f"# Full experiment. CPT-KD flags: '{args.cpt_flags}'",
             "# Order: CPT-KD and KD first (the main comparison), then the other methods."]
    priority = {"cpt": 0, "kd": 1, "ce": 2}
    ordered = sorted(METHODS, key=lambda m: priority.get(m[0], 3))
    n = 0
    for ds in args.datasets:
        for st in args.students:
            for method, extra in ordered:
                extra = args.cpt_flags if extra == "CPT" else extra
                for seed in args.seeds:
                    lines.append(f"python train_student.py --data {args.data_root}/{ds} --method {method} "
                                 f"--student-arch {st} --seed {seed} {extra}".rstrip())
                    n += 1
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {n} jobs to {args.out}")


if __name__ == "__main__":
    main()
