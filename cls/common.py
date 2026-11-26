"""
Step 2 - Shared code used by every training / analysis script.

  * GPU selection that matches nvidia-smi numbering (PCI bus order)
  * Reading meta.csv and selecting rows for a phase / fold
  * RAM-cached dataset: JPEG bytes are read once, decoded on the fly (fast, low memory)
  * Train / eval transforms
  * Model builder (ImageNet-pretrained torchvision models with a new classifier head)
  * Optimizer with a higher learning rate for the new head, cosine schedule with warm-up
  * Evaluation that returns logits ordered by image idx (needed for all poster metrics)

Import it from the other scripts:  from common import *
"""
import os

# Make CUDA device ids match nvidia-smi ids. Must be set before CUDA is initialised.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import csv
import io
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================ device / seeds
def get_device(gpu: int, min_free_gb: float = 8.0) -> torch.device:
    """Pick a GPU by its nvidia-smi index and warn if another process is using much of it."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU (only useful for debugging).")
        return torch.device("cpu")
    dev = torch.device(f"cuda:{gpu}")
    free, total = torch.cuda.mem_get_info(dev)
    name = torch.cuda.get_device_name(dev)
    print(f"Using GPU {gpu}: {name}, {free / 2**30:.1f} / {total / 2**30:.1f} GB free")
    if free / 2**30 < min_free_gb:
        print(f"WARNING: less than {min_free_gb} GB free on GPU {gpu}; other jobs are using it.")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return dev


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(device):
    """bf16 mixed precision on Ampere GPUs (A5000/A6000); no-op on CPU."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


# ============================================================================ metadata
def load_meta(data_root: str):
    with open(os.path.join(data_root, "info.json")) as f:
        info = json.load(f)
    with open(os.path.join(data_root, "meta.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("idx", "label", "fold_dev", "fold_final"):
            r[k] = int(r[k])
    assert [r["idx"] for r in rows] == list(range(len(rows))), "meta.csv idx must be 0..N-1"
    return rows, info


def select_rows(rows, phase: str, part: str, train_fold=None, only_fold=None):
    """
    phase: 'dev'   -> train pool = split 'train',          eval = split 'val'
           'final' -> train pool = split 'train' + 'val',  eval = split 'test'
    part:  'train' or 'eval'
    train_fold: if given, drop this fold from the training pool  (training out-of-fold teacher k)
    only_fold:  if given, keep only this fold of the training pool (teacher k predicting its fold)
    """
    assert phase in ("dev", "final") and part in ("train", "eval")
    fold_key = f"fold_{phase}"
    if part == "eval":
        want = "val" if phase == "dev" else "test"
        return [r for r in rows if r["split"] == want]
    pool = [r for r in rows if r[fold_key] >= 0]
    if train_fold is not None:
        pool = [r for r in pool if r[fold_key] != train_fold]
    if only_fold is not None:
        pool = [r for r in pool if r[fold_key] == only_fold]
    return pool


# ============================================================================ dataset
class RAMImageDataset(Dataset):
    """Keeps the (already resized) JPEG bytes in RAM; returns (image, label, idx)."""

    def __init__(self, data_root, rows, transform):
        self.rows = rows
        self.transform = transform
        t0 = time.time()
        self.blobs = []
        for r in rows:
            with open(os.path.join(data_root, r["file"]), "rb") as f:
                self.blobs.append(f.read())
        self.labels = torch.tensor([r["label"] for r in rows])
        self.idx = torch.tensor([r["idx"] for r in rows])
        mb = sum(len(b) for b in self.blobs) / 2**20
        print(f"  cached {len(rows)} images ({mb:.0f} MB) in {time.time() - t0:.1f}s")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = Image.open(io.BytesIO(self.blobs[i])).convert("RGB")
        return self.transform(img), self.labels[i], self.idx[i]


def build_transforms(train: bool, img_size: int = 224, resize: int = 256):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm,
        ])
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(), norm,
    ])


def make_loader(dataset, batch_size, train, workers=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train,
                      num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0)


# ============================================================================ models
ARCHS = {
    "resnet18": (models.resnet18, "fc"),
    "resnet34": (models.resnet34, "fc"),
    "resnet50": (models.resnet50, "fc"),
    "resnet101": (models.resnet101, "fc"),
    "mobilenet_v2": (models.mobilenet_v2, "classifier"),
}


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """torchvision model with ImageNet weights ('DEFAULT' = best available) and a new head."""
    fn, head = ARCHS[arch]
    model = fn(weights="DEFAULT" if pretrained else None)
    if head == "fc":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model.to(memory_format=torch.channels_last)


def head_parameters(model):
    return list((model.fc if hasattr(model, "fc") else model.classifier[-1]).parameters())


def build_optimizer(model, lr, weight_decay=5e-4, momentum=0.9, head_lr_mult=10.0):
    """SGD; the freshly initialised classifier head learns head_lr_mult times faster."""
    head_ids = {id(p) for p in head_parameters(model)}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    groups = [dict(params=backbone, lr=lr), dict(params=head_parameters(model), lr=lr * head_lr_mult)]
    for g in groups:
        g["base_lr"] = g["lr"]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)


def set_lr(optimizer, step, total_steps, warmup_steps):
    """Linear warm-up, then cosine decay to 0. Call once per iteration."""
    if step < warmup_steps:
        f = (step + 1) / warmup_steps
    else:
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        f = 0.5 * (1 + math.cos(math.pi * p))
    for g in optimizer.param_groups:
        g["lr"] = g["base_lr"] * f


# ============================================================================ evaluation
@torch.no_grad()
def predict(model, loader, device):
    """Return logits (N, C) float32, labels (N,), idx (N,) as numpy, sorted by idx."""
    model.eval()
    out_logits, out_labels, out_idx = [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with autocast(device):
            logits = model(x)
        out_logits.append(logits.float().cpu())
        out_labels.append(y)
        out_idx.append(idx)
    logits = torch.cat(out_logits).numpy()
    labels = torch.cat(out_labels).numpy()
    idx = torch.cat(out_idx).numpy()
    order = np.argsort(idx)
    return logits[order], labels[order], idx[order]


def accuracy(logits, labels):
    return float((logits.argmax(1) == labels).mean() * 100)


# ============================================================================ files
def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def run_name(**parts):
    """Readable, unique folder name, e.g. cub_final_resnet50_full_s0."""
    return "_".join(str(v) for v in parts.values() if v is not None and v != "")
