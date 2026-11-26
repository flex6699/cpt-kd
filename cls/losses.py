"""
All distillation methods compared in the project.

Every method follows the same pattern:
    1. build the teacher's soft label q (temperature T), possibly "cleaned"
    2. loss = cross-entropy with the label + a distillation term between q and the student

Methods ("--method" in train_student.py)
  ce            student alone: cross-entropy only
  kd            vanilla KD (Hinton et al. 2015)
  dkd           Decoupled KD (Zhao et al. 2022): target part (TCKD) + non-target part (NCKD)
  wsl           Weighted soft labels (Zhou et al. 2021): per-image weight 1 - exp(-CE_student / CE_teacher)
  ka            Knowledge Adjustment (Wen et al. 2021): if the teacher is wrong, swap its top-1 and
                true-class probabilities
  pergate       per-image correctness gate (in the spirit of Mishra et al. 2026): no KD on images
                where the teacher is wrong
  class_trust   class-level trust: one trust value per true class scales ALL wrong-class probabilities
  cpt           CPT-KD (ours): pair-level trust scales each wrong-class probability separately
  cpt_dkd       CPT-KD cleaning, then DKD

CPT-KD options (also used by class_trust):
  cpt_power        apply trust^power (2 = turn flagged pairs down more strongly)
  cpt_keep_target  keep the teacher's true-class probability unchanged and spread the removed mass
                   over the other wrong classes (avoids an over-confident student)
"""
import torch
import torch.nn.functional as F

METHODS = ["ce", "kd", "dkd", "wsl", "ka", "pergate", "class_trust", "cpt", "cpt_dkd"]
NEEDS_TEACHER = set(METHODS) - {"ce"}


# ============================================================================ building the soft label
def clean_with_trust(q, trust_rows, y=None, power=1.0, keep_target=False):
    """Multiply each class probability by its trust value (trust_rows: (B, C), 1 on the true class).
    Default: rescale to sum to 1, so removed mass returns to the true class and the other classes.
    keep_target: the true-class probability stays as it was; only the wrong classes are rescaled."""
    if power != 1.0:
        trust_rows = trust_rows ** power
    if not keep_target:
        q = q * trust_rows
        return q / q.sum(1, keepdim=True)
    gt = F.one_hot(y, q.size(1)).bool()
    q_gt = q[gt].unsqueeze(1)
    q_nt = (q * trust_rows).masked_fill(gt, 0)
    q_nt = q_nt / q_nt.sum(1, keepdim=True).clamp_min(1e-12) * (1 - q_gt)
    return q_nt + gt * q_gt


def knowledge_adjustment(q, t_logits, y):
    """Swap the teacher's top-1 and true-class probabilities where the teacher is wrong."""
    q = q.clone()
    pred = t_logits.argmax(1)
    wrong = pred != y
    if wrong.any():
        rows = wrong.nonzero(as_tuple=True)[0]
        p_true = q[rows, y[rows]].clone()
        q[rows, y[rows]] = q[rows, pred[rows]]
        q[rows, pred[rows]] = p_true
    return q


# ============================================================================ distillation terms
def kd_per_image(s_logits, q, T):
    """T^2 * KL(q || softmax(s/T)) for each image."""
    logp = F.log_softmax(s_logits / T, 1)
    return (q * (torch.log(q.clamp_min(1e-12)) - logp)).sum(1) * T * T


def dkd_loss(s_logits, q, y, T, alpha, beta):
    """Decoupled KD written for a (possibly cleaned) teacher distribution q."""
    gt = F.one_hot(y, s_logits.size(1)).bool()
    p_s = F.softmax(s_logits / T, 1)
    # target part: binary (true class vs the rest)
    qt = q[gt].clamp(1e-12, 1 - 1e-12)
    st = p_s[gt].clamp(1e-12, 1 - 1e-12)
    tckd = (qt * torch.log(qt / st) + (1 - qt) * torch.log((1 - qt) / (1 - st))) * T * T
    # non-target part: distribution over the wrong classes only
    q_nt = q.masked_fill(gt, 0)
    q_nt = q_nt / q_nt.sum(1, keepdim=True).clamp_min(1e-12)
    logp_nt = F.log_softmax(s_logits / T - 1000.0 * gt, 1)
    nckd = (q_nt * (torch.log(q_nt.clamp_min(1e-12)) - logp_nt)).sum(1) * T * T
    return (alpha * tckd + beta * nckd).mean()


# ============================================================================ main entry
def compute_loss(method, s_logits, y, t_logits=None, cfg=None, trust=None, class_trust=None, epoch=0):
    """
    s_logits, t_logits: (B, C) float32; y: (B,)
    cfg: dict with alpha, T, dkd_alpha, dkd_beta, dkd_warmup
    trust: (C, C) tensor (pair trust); class_trust: (C,) tensor
    Returns (loss, info dict).
    """
    ce = F.cross_entropy(s_logits, y)
    if method == "ce":
        return ce, {}
    a, T = cfg["alpha"], cfg["T"]
    q = F.softmax(t_logits / T, 1)
    info = {}

    opts = dict(y=y, power=cfg.get("cpt_power", 1.0), keep_target=cfg.get("cpt_keep_target", False))
    if method in ("cpt", "cpt_dkd"):
        q = clean_with_trust(q, trust[y], **opts)
    elif method == "class_trust":
        C = q.size(1)
        rows = class_trust[y].unsqueeze(1).expand(-1, C).clone()
        rows[torch.arange(len(y), device=y.device), y] = 1.0
        q = clean_with_trust(q, rows, **opts)
    elif method == "ka":
        q = knowledge_adjustment(q, t_logits, y)

    if method in ("dkd", "cpt_dkd"):
        warm = min(1.0, (epoch + 1) / max(1, cfg["dkd_warmup"]))
        return ce + warm * dkd_loss(s_logits, q, y, T, cfg["dkd_alpha"], cfg["dkd_beta"]), info

    per_img = kd_per_image(s_logits, q, T)
    if method == "wsl":
        with torch.no_grad():
            ce_s = F.cross_entropy(s_logits, y, reduction="none")
            ce_t = F.cross_entropy(t_logits, y, reduction="none")
            w = 1 - torch.exp(-ce_s / ce_t.clamp_min(1e-7))
        per_img = w * per_img
        info["mean_weight"] = w.mean().item()
    elif method == "pergate":
        w = (t_logits.argmax(1) == y).float()
        per_img = w * per_img
        info["teacher_wrong_frac"] = 1 - w.mean().item()
    return ce + a * per_img.mean(), info
