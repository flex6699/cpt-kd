"""
Slide figure "what CPT-KD does right": real test images where the teacher makes a look-alike mistake
(pair trust < 0.5), the KD student copies it in >= 2 of 3 seeds, and the CPT-KD student is correct in >= 2 of 3 seeds.
Also counts the reverse case (KD correct, CPT-KD copies) so the picture is not cherry-picked silently.

Writes square crops of the test image and of one training image of the teacher's wrong class, plus a JSON.
Usage: python cls/pick_win_examples.py --runs <ml_research>/runs --data <ml_research>/data/tea_kd --out <dir>
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image


def load(p):
    d = np.load(p)
    o = np.argsort(d["idx"])
    return d["logits"][o].argmax(1), d["labels"][o], d["idx"][o]


def nice(n):
    return n.split(".", 1)[-1].replace("_", " ")


def crop(path, out, aircraft):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if aircraft:                       # drop the copyright banner at the bottom
        img = img.crop((0, 0, w, h - 20)); h -= 20
    s = min(w, h)
    img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((360, 360)).save(out, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = dict(examples=[], counts={})
    for ds, st, k in (("cub", "resnet18", 3), ("cub", "mobilenet_v2", 0), ("aircraft", "resnet18", 0),
                      ("aircraft", "mobilenet_v2", 1)):
        R = f"{a.runs}/{ds}/final"
        pt, y, idx = load(f"{R}/oof/teacher_full_test.npz")
        tr = np.load(f"{R}/oof/trust_T1.npy")
        fl = (pt != y) & (tr[y, pt] < 0.5)
        K = np.stack([load(f"{R}/students/kd_{st}_s{s}/test_logits.npz")[0] for s in range(3)])
        C = np.stack([load(f"{R}/students/cpt_{st}_keept_s{s}/test_logits.npz")[0] for s in range(3)])
        win = fl & ((K == pt).sum(0) >= 2) & ((C == y).sum(0) >= 2)
        rev = fl & ((K == y).sum(0) >= 2) & ((C == pt).sum(0) >= 2)
        res["counts"][f"{ds}_{st}"] = dict(flagged=int(fl.sum()), wins=int(win.sum()), reverse=int(rev.sum()),
                                           copies_kd=float((K[:, fl] == pt[fl]).sum(1).mean()),
                                           copies_cpt=float((C[:, fl] == pt[fl]).sum(1).mean()))
        meta = list(csv.DictReader(open(f"{a.data}/{ds}/meta.csv", encoding="utf-8")))
        rows = {int(r["idx"]): r for r in meta}
        names = {int(r["label"]): r["class_name"] for r in meta}
        cand = np.where(win)[0]
        cand = cand[np.lexsort((tr[y[cand], pt[cand]], -(C[:, cand] == y[cand]).sum(0) - (K[:, cand] == pt[cand]).sum(0)))]
        seen = set()
        for i in cand:
            if len([e for e in res["examples"] if e["setup"] == f"{ds}_{st}"]) >= k:
                break
            if (y[i], pt[i]) in seen:
                continue
            seen.add((y[i], pt[i]))
            ref = next(r for r in meta if int(r["label"]) == pt[i] and r["split"] != "test")
            tag = f"{ds}_{int(idx[i])}"
            crop(f"{a.data}/{ds}/{rows[int(idx[i])]['file']}", f"{a.out}/{tag}_test.jpg", ds == "aircraft")
            crop(f"{a.data}/{ds}/{ref['file']}", f"{a.out}/{tag}_ref.jpg", ds == "aircraft")
            res["examples"].append(dict(setup=f"{ds}_{st}", tag=tag, true=nice(names[int(y[i])]),
                                        teacher=nice(names[int(pt[i])]), trust=round(float(tr[y[i], pt[i]]), 2),
                                        kd_copies=int((K[:, i] == pt[i]).sum()), cpt_right=int((C[:, i] == y[i]).sum())))
    json.dump(res, open(f"{a.out}/win_examples.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
