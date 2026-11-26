"""
Step 1 - Prepare CUB-200-2011 and FGVC-Aircraft for TEA-KD experiments.

What it does
  1. Reads the official annotation files of each dataset.
  2. Resizes every image once so its shorter side is 256 px (huge data-loading speed-up).
     For FGVC-Aircraft it first removes the 20-pixel copyright banner at the bottom.
  3. Writes one metadata table, meta.csv, with one row per image:

     idx          contiguous integer id (0..N-1), used to index per-image memory in TEA-KD
     file         image path relative to the output folder
     label        class id (0-based)
     class_name   readable class name
     split        'train' | 'val' | 'test'
                    CUB:      'val' = 3 images per class held out from the official train set
                    Aircraft: official train / val / test
     fold_dev     0..K-1 for images in the DEV training pool (split == 'train'), else -1
     fold_final   0..K-1 for images in the FINAL training pool (split in {train, val}), else -1
     family, manufacturer   Aircraft only (used for error analysis), empty for CUB

  Two experiment phases use these columns:
     dev   : train on split=='train',          evaluate on split=='val'   (tuning, 1 seed)
     final : train on split in {train,val},    evaluate on split=='test'  (reported results)
  Out-of-fold teacher k is trained on the pool minus fold k, and teaches only fold k.

Expected input layouts (as downloaded and extracted)
  CUB      --src .../CUB_200_2011
             images/, images.txt, image_class_labels.txt, train_test_split.txt, classes.txt
  Aircraft --src .../fgvc-aircraft-2013b/data
             images/, variants.txt, images_variant_{train,val,test}.txt,
             images_family_{train,val,test}.txt, images_manufacturer_{train,val,test}.txt

  Folder layout (GitHub copies, e.g. cyizhuo/CUB-200-2011-dataset, cyizhuo/FGVC_Aircraft_dataset)
           --src .../<repo>-main   containing  train/<class>/*.jpg  and  test/<class>/*.jpg
           Detected automatically. CUB: official train/test split. Aircraft: official test set;
           official train+val are merged, so 33 images per class are held out as 'val'
           (same size as the official val set). Family/manufacturer labels are not available.

Usage
  python prepare_data.py --dataset cub      --src /data/raw/CUB_200_2011            --out /data/tea_kd/cub
  python prepare_data.py --dataset aircraft --src /data/raw/fgvc-aircraft-2013b/data --out /data/tea_kd/aircraft
  python prepare_data.py --dataset cub      --src data/raw/CUB_200_2011_dataset-main  --out data/tea_kd/cub
  python prepare_data.py --dataset aircraft --src data/raw/FGVC_Aircraft_dataset-main --out data/tea_kd/aircraft
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from multiprocessing import Pool

from PIL import Image

AIRCRAFT_BANNER_PX = 20


# ----------------------------------------------------------------------------- readers
def _read_pairs(path):
    """Read 'key value...' lines; the value may contain spaces (e.g. 'BAE 146-200')."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                k, v = line.split(" ", 1)
                out.append((k, v))
    return out


def read_cub(src):
    images = dict(_read_pairs(os.path.join(src, "images.txt")))
    labels = {k: int(v) - 1 for k, v in _read_pairs(os.path.join(src, "image_class_labels.txt"))}
    is_train = {k: v == "1" for k, v in _read_pairs(os.path.join(src, "train_test_split.txt"))}
    classes = {int(k) - 1: v for k, v in _read_pairs(os.path.join(src, "classes.txt"))}
    rows = []
    for img_id in sorted(images, key=int):
        rows.append(dict(
            src_path=os.path.join(src, "images", images[img_id]),
            out_name=images[img_id],                       # keeps the class sub-folder
            label=labels[img_id],
            class_name=classes[labels[img_id]].split(".", 1)[-1],
            split="train" if is_train[img_id] else "test",
            family="", manufacturer="", crop_banner=False,
        ))
    return rows, [classes[i].split(".", 1)[-1] for i in range(len(classes))]


def read_aircraft(src):
    with open(os.path.join(src, "variants.txt"), encoding="utf-8") as f:
        variants = [l.strip() for l in f if l.strip()]
    vid = {v: i for i, v in enumerate(variants)}
    rows = []
    for split in ("train", "val", "test"):
        var = _read_pairs(os.path.join(src, f"images_variant_{split}.txt"))
        fam = dict(_read_pairs(os.path.join(src, f"images_family_{split}.txt")))
        man = dict(_read_pairs(os.path.join(src, f"images_manufacturer_{split}.txt")))
        for img_id, v in var:
            rows.append(dict(
                src_path=os.path.join(src, "images", f"{img_id}.jpg"),
                out_name=f"{img_id}.jpg",
                label=vid[v], class_name=v, split=split,
                family=fam.get(img_id, ""), manufacturer=man.get(img_id, ""),
                crop_banner=True,
            ))
    return rows, variants


def is_folder_layout(src):
    return os.path.isdir(os.path.join(src, "train")) and os.path.isdir(os.path.join(src, "test"))


def read_folders(src, dataset):
    """train/<class>/*.jpg + test/<class>/*.jpg; classes sorted by folder name."""
    classes = sorted(d for d in os.listdir(os.path.join(src, "train"))
                     if os.path.isdir(os.path.join(src, "train", d)))
    test_classes = sorted(d for d in os.listdir(os.path.join(src, "test"))
                          if os.path.isdir(os.path.join(src, "test", d)))
    assert classes == test_classes, "train/ and test/ must contain the same class folders"
    names = [c.split(".", 1)[-1] if dataset == "cub" else c for c in classes]
    rows = []
    for split in ("train", "test"):
        for label, c in enumerate(classes):
            folder = os.path.join(src, split, c)
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    rows.append(dict(
                        src_path=os.path.join(folder, fn),
                        out_name=f"{c}/{fn}" if dataset == "cub" else fn,
                        label=label, class_name=names[label], split=split,
                        family="", manufacturer="", crop_banner=dataset == "aircraft",
                    ))
    return rows, names


# ----------------------------------------------------------------------------- splits
def hold_out_per_class(rows, n_per_class, rng):
    """CUB has no official val split: move n images per class from train to 'val'."""
    by_class = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            by_class[r["label"]].append(r)
    for items in by_class.values():
        rng.shuffle(items)
        for r in items[:n_per_class]:
            r["split"] = "val"


def assign_folds(rows, in_pool, key, k, rng):
    """Stratified K-fold: each class is spread evenly over the folds."""
    by_class = defaultdict(list)
    for r in rows:
        r[key] = -1
        if in_pool(r):
            by_class[r["label"]].append(r)
    for items in by_class.values():
        rng.shuffle(items)
        offset = rng.randrange(k)            # avoid fold 0 always getting the extra image
        for i, r in enumerate(items):
            r[key] = (i + offset) % k


# ----------------------------------------------------------------------------- resizing
def _resize_one(job):
    src_path, dst_path, short, crop_banner = job
    if os.path.exists(dst_path):
        return None
    try:
        im = Image.open(src_path).convert("RGB")
        if crop_banner:
            im = im.crop((0, 0, im.width, im.height - AIRCRAFT_BANNER_PX))
        w, h = im.size
        s = short / min(w, h)
        if s < 1:                              # only shrink, never upscale
            im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        im.save(dst_path, quality=95)
        return None
    except Exception as e:                      # report, don't crash the whole pool
        return f"{src_path}: {e}"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cub", "aircraft"])
    ap.add_argument("--src", required=True, help="extracted dataset folder (see docstring)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--short", type=int, default=256, help="shorter side after resizing")
    ap.add_argument("--val-per-class", type=int, default=None,
                    help="images per class held out as 'val' (default: CUB 3; Aircraft folder layout 33)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if is_folder_layout(args.src):
        print("Detected train/ + test/ class-folder layout")
        rows, class_names = read_folders(args.src, args.dataset)
        hold_out_per_class(rows, args.val_per_class or (3 if args.dataset == "cub" else 33), rng)
    elif args.dataset == "cub":
        rows, class_names = read_cub(args.src)
        hold_out_per_class(rows, args.val_per_class or 3, rng)
    else:
        rows, class_names = read_aircraft(args.src)

    assign_folds(rows, lambda r: r["split"] == "train", "fold_dev", args.folds, rng)
    assign_folds(rows, lambda r: r["split"] in ("train", "val"), "fold_final", args.folds, rng)

    # resize
    img_dir = os.path.join(args.out, "images")
    jobs = [(r["src_path"], os.path.join(img_dir, r["out_name"]), args.short, r["crop_banner"]) for r in rows]
    with Pool(args.workers) as pool:
        errors = [e for e in pool.imap_unordered(_resize_one, jobs, chunksize=32) if e]
    if errors:
        print(f"WARNING: {len(errors)} images failed, e.g.:", *errors[:5], sep="\n  ")

    # metadata
    os.makedirs(args.out, exist_ok=True)
    cols = ["idx", "file", "label", "class_name", "split", "fold_dev", "fold_final", "family", "manufacturer"]
    with open(os.path.join(args.out, "meta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow(dict(idx=i, file=os.path.join("images", r["out_name"]), label=r["label"],
                            class_name=r["class_name"], split=r["split"], fold_dev=r["fold_dev"],
                            fold_final=r["fold_final"], family=r["family"], manufacturer=r["manufacturer"]))
    with open(os.path.join(args.out, "info.json"), "w") as f:
        json.dump(dict(dataset=args.dataset, num_classes=len(class_names), class_names=class_names,
                       folds=args.folds, seed=args.seed, short_side=args.short,
                       layout="folders" if is_folder_layout(args.src) else "official",
                       has_family=any(r["family"] for r in rows)), f, indent=1)

    # summary
    split_counts = Counter(r["split"] for r in rows)
    print(f"{args.dataset}: {len(rows)} images, {len(class_names)} classes, splits {dict(split_counts)}")
    for key in ("fold_dev", "fold_final"):
        c = Counter(r[key] for r in rows if r[key] >= 0)
        print(f"  {key}: " + ", ".join(f"fold {k}={c[k]}" for k in sorted(c)))
    print(f"  wrote {os.path.join(args.out, 'meta.csv')}")


if __name__ == "__main__":
    main()
