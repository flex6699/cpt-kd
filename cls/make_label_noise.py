"""
Label-noise robustness study: make a copy of a prepared dataset in which a fraction of the TRAINING-pool labels
are flipped to a uniformly random other class (symmetric noise). Test labels stay clean. Images are symlinked.

CPT-KD estimates class-pair confusions from the labels, so noisy labels can corrupt its trust table.

Usage (inside cls/):
  python make_label_noise.py --src data/tea_kd/cub --rate 0.2 --seed 0      # -> data/tea_kd/cub_noise20
  python oof_and_trust.py --data data/tea_kd/cub_noise20 --models teacher --gpu 0 &
  python oof_and_trust.py --data data/tea_kd/cub_noise20 --models student --gpu 1
  python train_student.py --data data/tea_kd/cub_noise20 --method kd  --seed 0
  python train_student.py --data data/tea_kd/cub_noise20 --method cpt --seed 0 --cpt-keep-target
"""
import argparse
import csv
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--rate", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = a.src.rstrip("/")
    info = json.load(open(os.path.join(src, "info.json")))
    tag = "%s_noise%d" % (info["dataset"], round(a.rate * 100))
    out = a.out or os.path.join(os.path.dirname(src), tag)
    os.makedirs(out, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(src, "meta.csv"), encoding="utf-8")))
    C = info["num_classes"]
    rng = np.random.default_rng(a.seed)
    pool = [i for i, r in enumerate(rows) if int(r["fold_final"]) >= 0 or int(r["fold_dev"]) >= 0]
    flip = rng.choice(pool, int(round(a.rate * len(pool))), replace=False)
    names = info.get("class_names")
    for i in flip:
        y = int(rows[i]["label"])
        z = int(rng.integers(C - 1))
        z = z + (z >= y)
        rows[i]["label"] = str(z)
        if names:
            rows[i]["class_name"] = names[z]
    with open(os.path.join(out, "meta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    info = dict(info, dataset=tag, label_noise=dict(rate=a.rate, seed=a.seed, flipped=int(len(flip)),
                                                    training_pool=len(pool), source=os.path.abspath(src)))
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=1)
    link = os.path.join(out, "images")
    if not os.path.exists(link):
        os.symlink(os.path.abspath(os.path.join(src, "images")), link)
    test_changed = sum(1 for i in flip if rows[i]["split"] == "test")
    print("%s: flipped %d of %d training-pool labels (test labels changed: %d)" % (out, len(flip), len(pool), test_changed))


if __name__ == "__main__":
    main()
