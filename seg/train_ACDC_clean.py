"""
Clean ACDC training script for reproducing IAD (CVPR 2026).

Modes
  teacher : SAM-L + LoRA teacher, segmentation loss only
  student : EfficientNet-B1 student, segmentation loss only (baseline)
  kis     : IAD stage 1 = seg + soft-label L2 (L_sld) + class-prototype L2 (L_cpd)   [needs --tea_ckpt]
  kas     : IAD stage 2 = seg + reverse feature constraint (L_rfc) + contrastive (L_css) [needs --stu_ckpt]

Fixes relative to the released train_ACDC.py
  1. Loss weights: the released code multiplied every distillation term by a weight that was
     re-created as 0 each iteration, so KIS/KAS reduced to plain supervised training.
     Here: fixed weights (paper Eq. 7/8) or proper uncertainty weighting 0.5*exp(-s)*L + 0.5*s.
  2. KAS projection convs are created once and trained (released code re-created them randomly
     every iteration).
  3. Checkpoint selection uses the validation set only; the test set is evaluated once at the end.
  4. All paths are arguments; numeric arguments have proper types.

Run from the repo root:
  PYTHONPATH=.:ACDC_Tool python ACDC_Tool/train_ACDC_clean.py --mode student ...
"""
import os
import sys
import random
import logging
import argparse
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from medpy.metric import dc
from scipy.ndimage import zoom

# ----------------------------------------------------------------------------- args
p = argparse.ArgumentParser()
p.add_argument('--mode', required=True, choices=['teacher', 'student', 'kis', 'kas'])
p.add_argument('--exp', default=None, help='run name (default: the mode)')
p.add_argument('--save_path', required=True)
p.add_argument('--root_dir', required=True)
p.add_argument('--list_dir', required=True)
p.add_argument('--volume_path', required=True)
p.add_argument('--ckpt', required=True, help='SAM checkpoint, e.g. sam_vit_l_0b3195.pth')
p.add_argument('--tea_ckpt', default=None, help='trained teacher LoRA checkpoint (kis)')
p.add_argument('--stu_ckpt', default=None, help='stage-1 (kis) student checkpoint (kas)')
p.add_argument('--vit_name', default='vit_l')
p.add_argument('--tea_module', default='sam_lora_image_encoder_tea')
p.add_argument('--stu_module', default='sam_lora_image_encoder_stu')
p.add_argument('--rank', type=int, default=8)
p.add_argument('--stu_train_encoder', type=int, default=1,
               help='unfreeze the student encoder (LoRA_Sam freezes it, and EfficientNet has no attention layers for LoRA)')
p.add_argument('--batch_size', type=int, default=6)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--max_epochs', type=int, default=200)
p.add_argument('--img_size', type=int, default=256)
p.add_argument('--num_classes', type=int, default=3, help='foreground classes (output has +1)')
p.add_argument('--z_spacing', type=int, default=10)
p.add_argument('--n_gpu', type=int, default=1)
p.add_argument('--num_workers', type=int, default=4)
p.add_argument('--seed', type=int, default=2345)
p.add_argument('--deterministic', type=int, default=1)
# loss weights (paper Eq. 7: alpha*L_kis + beta*L_seg ; Eq. 8: gamma*L_kas + delta*L_seg)
p.add_argument('--weighting', default='fixed', choices=['fixed', 'uncertainty'])
p.add_argument('--alpha', type=float, default=0.2)
p.add_argument('--beta', type=float, default=0.8)
p.add_argument('--gamma', type=float, default=0.1)
p.add_argument('--delta', type=float, default=1.0)
p.add_argument('--kas_in_ch', type=int, default=256, help='channels of the student global feature')
p.add_argument('--trust', default=None, help='cpt: KxK pair-trust matrix (.npy, rows = true class) from cpt_pilot.py')
p.add_argument('--kd_loss', default='l2', choices=['l2', 'kl', 'cpt'],
               help='KIS soft-label loss: l2 on logits (paper) or KL on softened probabilities')
p.add_argument('--kd_tau', type=float, default=1.0, help='temperature for --kd_loss kl')
# debugging
p.add_argument('--max_train_iters', type=int, default=0, help='stop each epoch after N iters (0 = off)')
# speed
p.add_argument('--amp', type=int, default=1, help='bf16 mixed precision for the training forward pass')
p.add_argument('--val_every', type=int, default=1, help='validate every N epochs (last epoch always)')
# kept only because test_ACDC.inference may read them
p.add_argument('--encoder', default='PVT')
p.add_argument('--skip_aggregation', default='additive')
p.add_argument('--checkpoint', default=None)
p.add_argument('--test_save_dir', default='pre')
p.add_argument('--module', default='sam_lora_image_encoder_tea')
p.add_argument('--warmup', action='store_true')
p.add_argument('--warmup_period', type=int, default=250)
p.add_argument('--AdamW', action='store_true')
args = p.parse_args()

# Project imports. test_ACDC may parse sys.argv at import time, so hide our arguments from it.
_argv, sys.argv = sys.argv, [sys.argv[0]]
from ACDC_Tool.utils import DiceLoss
from datasets.dataset_ACDC import ACDCdataset, RandomGenerator
from test_ACDC import inference
from segment_anything_Stu import sam_model_registry
from segment_anything_Tea import sam_model_registry_tea
from distillation_loss import comp_class_pro, contrastive_loss
sys.argv = _argv

# ----------------------------------------------------------------------------- setup
cudnn.benchmark = not args.deterministic
cudnn.deterministic = bool(args.deterministic)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

snapshot_path = os.path.join(args.save_path, args.exp or args.mode)
os.makedirs(snapshot_path, exist_ok=True)
args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
os.makedirs(args.test_save_dir, exist_ok=True)
args.module = args.tea_module

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S',
                    handlers=[logging.FileHandler(os.path.join(snapshot_path, 'log.txt')),
                              logging.StreamHandler(sys.stdout)])
logging.info(str(args))

N_OUT = args.num_classes + 1  # background + foreground classes


# ----------------------------------------------------------------------------- models
class TeacherWrap(nn.Module):
    """The teacher returns a single logits tensor; the student returns (logits, local, global).
    Wrap the teacher so every call site (training, validation, test_ACDC) can unpack 3 values."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m(x)
        if isinstance(o, (tuple, list)):
            return (o[0], None, None) if len(o) != 3 else tuple(o)
        return o, None, None

    def save_lora_parameters(self, path):
        self.m.save_lora_parameters(path)

    def load_lora_parameters(self, path):
        self.m.load_lora_parameters(path)


def build_teacher():
    sam_tea, _ = sam_model_registry_tea[args.vit_name](
        image_size=args.img_size, num_classes=args.num_classes, checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
    return TeacherWrap(import_module(args.tea_module).LoRA_Sam(sam_tea, args.rank)).cuda()


def build_student():
    sam, _ = sam_model_registry[args.vit_name](
        image_size=args.img_size, num_classes=args.num_classes, checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
    net_s = import_module(args.stu_module).LoRA_Sam(sam, args.rank)
    if args.stu_train_encoder:
        for q in net_s.sam.image_encoder.parameters():
            q.requires_grad_(True)
    return net_s.cuda()


net_tea = None
if args.mode == 'teacher':
    net = build_teacher()
else:
    net = build_student()
    if args.mode == 'kis':
        assert args.tea_ckpt, '--tea_ckpt is required for kis'
        net_tea = build_teacher()
        net_tea.load_lora_parameters(args.tea_ckpt)
        net_tea.eval()
        for q in net_tea.parameters():
            q.requires_grad_(False)
        logging.info('teacher loaded from %s' % args.tea_ckpt)
        if args.kd_loss == 'cpt':
            assert args.trust, '--trust is required for --kd_loss cpt'
            TRUST = torch.from_numpy(np.load(args.trust)).float().cuda()
            logging.info('CPT-KD pair trust (rows = true class):\n%s' % np.array2string(TRUST.cpu().numpy(), precision=3))
    if args.mode == 'kas':
        assert args.stu_ckpt, '--stu_ckpt is required for kas'
        net.load_state_dict(torch.load(args.stu_ckpt, map_location='cuda'))
        logging.info('stage-1 student loaded from %s' % args.stu_ckpt)

if args.n_gpu > 1:
    net = nn.DataParallel(net)
net = net.cuda()


class AuxWeights(nn.Module):
    """fixed: w * sum(L_i).  uncertainty: sum(0.5*exp(-s_i)*L_i + 0.5*s_i), s_i learnable."""

    def __init__(self, n_terms, mode, w):
        super().__init__()
        self.mode, self.w = mode, w
        if mode == 'uncertainty':
            self.s = nn.Parameter(torch.zeros(n_terms))

    def forward(self, losses):
        if self.mode == 'fixed':
            return self.w * sum(losses)
        return sum(0.5 * torch.exp(-self.s[i]) * l + 0.5 * self.s[i] for i, l in enumerate(losses))


class KASHead(nn.Module):
    """Projection of the encoder feature (F+) and of the negative sample (F-), trained with the student."""

    def __init__(self, in_ch, n_out):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, n_out, kernel_size=1)
        self.neg = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)


extra_modules = nn.ModuleList()
aux = None
if args.mode == 'kis':
    aux = AuxWeights(2, args.weighting, args.alpha).cuda()
    extra_modules.append(aux)
if args.mode == 'kas':
    aux = AuxWeights(3, args.weighting, args.gamma).cuda()
    kas_head = KASHead(args.kas_in_ch, N_OUT).cuda()
    extra_modules.extend([aux, kas_head])

seg_w = 1.0
if args.weighting == 'fixed':
    seg_w = args.beta if args.mode == 'kis' else (args.delta if args.mode == 'kas' else 1.0)

params = [q for q in net.parameters() if q.requires_grad] + list(extra_modules.parameters())
optimizer = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.1)
logging.info('trainable parameters: %d' % sum(q.numel() for q in params))

ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(N_OUT)
css_loss = contrastive_loss()


def seg_loss(out, label):
    return 0.1 * ce_loss(out, label.long()) + 0.9 * dice_loss(out, label, softmax=True)


# ----------------------------------------------------------------------------- data
def worker_init_fn(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


train_set = ACDCdataset(args.root_dir, args.list_dir, split='train', transform=transforms.Compose(
    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, worker_init_fn=worker_init_fn)
val_loader = DataLoader(ACDCdataset(args.root_dir, args.list_dir, split='valid'), batch_size=1, shuffle=False)
test_loader = DataLoader(ACDCdataset(args.volume_path, args.list_dir, split='test'), batch_size=1, shuffle=False)
logging.info('train slices: %d | val slices: %d | test volumes: %d'
             % (len(train_set), len(val_loader), len(test_loader)))


# ----------------------------------------------------------------------------- helpers
def save(path):
    m = net.module if isinstance(net, nn.DataParallel) else net
    if args.mode == 'teacher' and hasattr(m, 'save_lora_parameters'):
        m.save_lora_parameters(path)  # same format the KIS stage loads with load_lora_parameters
    else:
        torch.save(m.state_dict(), path)


def load(path):
    m = net.module if isinstance(net, nn.DataParallel) else net
    if args.mode == 'teacher' and hasattr(m, 'load_lora_parameters'):
        m.load_lora_parameters(path)
    else:
        m.load_state_dict(torch.load(path, map_location='cuda'))


@torch.no_grad()
def validate():
    """Mean per-class foreground Dice over validation slices."""
    net.eval()
    S = args.img_size
    scores = []
    for batch in val_loader:
        img = batch['image'].squeeze(0).numpy()
        lab = batch['label'].squeeze(0).numpy()
        x, y = img.shape[0], img.shape[1]
        img_r = zoom(img, (S / x, S / y), order=3) if (x, y) != (S, S) else img
        out, _, _ = net(torch.from_numpy(img_r).unsqueeze(0).unsqueeze(0).float().cuda())
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
        if (x, y) != (S, S):
            pred = zoom(pred, (x / S, y / S), order=0)
        per = [dc(pred == c, lab == c) for c in range(1, N_OUT) if (lab == c).any() or (pred == c).any()]
        if per:
            scores.append(np.mean(per))
    net.train()
    return float(np.mean(scores)) if scores else 0.0


# ----------------------------------------------------------------------------- train
best_val = -1.0
iter_num = 0
for epoch in tqdm(range(args.max_epochs), ncols=70):
    net.train()
    extra_modules.train()
    for i_batch, batch in enumerate(train_loader):
        if args.max_train_iters and i_batch >= args.max_train_iters:
            break
        image = batch['image'].float().cuda()
        label = batch['label'].float().cuda()

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(args.amp)):
            out, stu_local, stu_global = net(image)
            L_seg = seg_loss(out, label)
            logs = {'seg': L_seg.item()}

            if args.mode in ('teacher', 'student'):
                loss = L_seg

            elif args.mode == 'kis':
                with torch.no_grad():
                    t_out, _, _ = net_tea(image)
                if t_out.shape[-2:] != out.shape[-2:]:
                    t_out = F.interpolate(t_out, size=out.shape[-2:], mode='bilinear', align_corners=False)
                if args.kd_loss in ('kl', 'cpt'):
                    tau = args.kd_tau
                    zt, zs = t_out.float() / tau, out.float() / tau
                    q = torch.softmax(zt, 1)
                    if args.kd_loss == 'cpt':
                        # CPT-KD: scale each wrong-class prob by trust[y, c], rescale wrong classes to their
                        # original total; true-class prob unchanged.
                        y1 = F.one_hot(label.long(), N_OUT).permute(0, 3, 1, 2).float()
                        tr = TRUST[label.long()].permute(0, 3, 1, 2)
                        wrong = q * (1 - y1)
                        scaled = wrong * tr
                        scaled = scaled * (wrong.sum(1, keepdim=True) / scaled.sum(1, keepdim=True).clamp_min(1e-12))
                        q = (q * y1 + scaled).detach()
                        logs['moved'] = 0.5 * (q - torch.softmax(zt, 1)).abs().sum(1).mean().item()
                    L_sld = (torch.xlogy(q, q) - q * torch.log_softmax(zs, 1)).sum(1).mean() * tau ** 2
                    P_t = comp_class_pro(label, torch.softmax(t_out.float(), 1), num_classes=N_OUT)
                    P_s = comp_class_pro(label, torch.softmax(out.float(), 1), num_classes=N_OUT)
                else:
                    L_sld = F.mse_loss(out, t_out)                               # Eq. 1
                    P_t = comp_class_pro(label, t_out, num_classes=N_OUT)
                    P_s = comp_class_pro(label, out, num_classes=N_OUT)
                L_cpd = F.mse_loss(P_s, P_t)                                     # Eq. 2 (w_n = 1 on ACDC)
                loss = seg_w * L_seg + aux([L_sld, L_cpd])                       # Eq. 7
                logs.update(sld=L_sld.item(), cpd=L_cpd.item())

            else:  # kas
                X = F.interpolate(kas_head.proj(stu_global), size=out.shape[-2:],
                                  mode='bilinear', align_corners=False)          # Re(Conv(F_l))
                L_l2 = F.mse_loss(X, out)
                Xs, Os = torch.softmax(X, dim=1), torch.softmax(out, dim=1)
                inter = (Xs * Os).sum(dim=(2, 3))
                union = (Xs + Os).sum(dim=(2, 3))
                L_dice = 1 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()       # Eq. 4 = L_l2 + L_dice
                F_neg = kas_head.neg((1 - Os) + (1 - Xs))
                L_css = css_loss(out, X, F_neg)                                  # Eq. 5
                loss = seg_w * L_seg + aux([L_dice, L_css, L_l2])                # Eq. 8
                logs.update(l2=L_l2.item(), dice=L_dice.item(), css=L_css.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_num += 1
        if iter_num % 50 == 0:
            logging.info('iter %d | loss %.4f | %s' % (
                iter_num, loss.item(), ' '.join('%s %.4f' % kv for kv in logs.items())))

    save(os.path.join(snapshot_path, 'last.pth'))
    if (epoch + 1) % args.val_every == 0 or epoch == args.max_epochs - 1:
        val_dice = validate()
        if val_dice > best_val:
            best_val = val_dice
            save(os.path.join(snapshot_path, 'best.pth'))
        logging.info('epoch %d | val dice %.4f | best %.4f' % (epoch, val_dice, best_val))

# ----------------------------------------------------------------------------- test (once, best-val model)
load(os.path.join(snapshot_path, 'best.pth'))
net.eval()
with torch.no_grad():
    # test_ACDC.inference expects num_classes to INCLUDE background (its own default is 4)
    test_args = argparse.Namespace(**{**vars(args), 'num_classes': N_OUT})
    test_dice, test_hd, test_jac, test_asd = inference(test_args, net, test_loader, args.test_save_dir)
logging.info('TEST (best-val checkpoint) | dice %.4f | hd95 %.4f | jaccard %.4f | asd %.4f'
             % (test_dice, test_hd, test_jac, test_asd))
