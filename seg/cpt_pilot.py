"""
CPT-KD pilot for segmentation (no training): Class-Pair Trust KD carried over from fine-grained
classification (CSL7620 project) to SAM -> small-student distillation on ACDC.

1. Pair trust (estimated on the held-out VALID slices, unseen by teacher and student-alone):
     P_m[y, c] = mean softmax probability model m gives class c on pixels whose true class is y  (T = 1)
     trust[y, c] = min(1, (P_student_alone[y, c] + eps) / (P_teacher[y, c] + eps)),  trust[y, y] = 1
   Also reported on TRAIN slices (in-sample) to check whether the teacher has memorised its training set.
2. On the TEST volumes (all slices, 256x256), for every student checkpoint:
     Dice per class | teacher-error inheritance | flagged-pair inheritance | accuracy on teacher-correct
     pixels | student-teacher agreement | ECE
   inheritance = among pixels the TEACHER gets wrong, % where the student predicts the SAME wrong class.

Usage (from the IAD repo root):
  PYTHONPATH=.:ACDC_Tool CUDA_VISIBLE_DEVICES=0 python -u cpt_pilot.py \
      --root ~/cvpr2027/data/ACDC --runs ~/cvpr2027/runs --ckpt ~/cvpr2027/checkpoints/sam_vit_l_0b3195.pth \
      --students student kis kis_kl_t1 kis_kl_t2 --out ~/cvpr2027/runs/cpt_pilot
"""
import os
import sys
import json
import glob
import argparse
from importlib import import_module

import numpy as np

CLS = ['BG', 'RV', 'Myo', 'LV']
EPS = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tea_exp', default='teacher')
    ap.add_argument('--ref_exp', default='student', help='student-alone used for the trust scores')
    ap.add_argument('--students', nargs='+', default=['student', 'kis', 'kis_kl_t1', 'kis_kl_t2'])
    ap.add_argument('--flag_thr', type=float, default=0.5, help='same threshold as the classification experiments')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--trust_only', type=int, default=0)
    args = ap.parse_args()

    import torch
    from scipy.ndimage import zoom
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    from segment_anything_Stu import sam_model_registry
    from segment_anything_Tea import sam_model_registry_tea
    sys.argv = _argv

    root, runs, S, K = os.path.expanduser(args.root), os.path.expanduser(args.runs), 256, 4
    kw = dict(image_size=S, num_classes=K - 1, checkpoint=os.path.expanduser(args.ckpt),
              pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
    os.makedirs(os.path.expanduser(args.out), exist_ok=True)

    def teacher():
        m, _ = sam_model_registry_tea['vit_l'](**kw)
        net = import_module('sam_lora_image_encoder_tea').LoRA_Sam(m, 8).cuda().eval()
        net.load_lora_parameters(os.path.join(runs, args.tea_exp, 'best.pth'))
        return net

    def student(exp):
        m, _ = sam_model_registry['vit_l'](**kw)
        net = import_module('sam_lora_image_encoder_stu').LoRA_Sam(m, 8).cuda().eval()
        net.load_state_dict(torch.load(os.path.join(runs, exp, 'best.pth'), map_location='cuda'))
        return net

    # ------------------------------------------------------------------ data (all at 256x256)
    def slices(split):
        if split == 'test':
            for f in sorted(glob.glob(os.path.join(root, 'test', '*.npz'))):
                d = np.load(f)
                for z in range(d['img'].shape[0]):
                    yield d['img'][z], d['label'][z]
        else:
            names = [l.strip() for l in open(os.path.join(root, 'lists_ACDC', split + '.txt')) if l.strip()]
            for n in names:
                d = np.load(os.path.join(root, split, n))
                yield d['img'], d['label']

    def load(split):
        X, Y = [], []
        for img, lab in slices(split):
            h, w = img.shape
            if (h, w) != (S, S):
                img = zoom(img, (S / h, S / w), order=3)
                lab = zoom(lab, (S / h, S / w), order=0)
            X.append(img.astype(np.float32))
            Y.append(lab.astype(np.int64))
        return np.stack(X), np.stack(Y)

    data = {s: load(s) for s in (['train', 'valid'] if args.trust_only else ['train', 'valid', 'test'])}
    for s, (x, y) in data.items():
        print('%s: %d slices' % (s, len(x)))

    @torch.no_grad()
    def probs(net, X):
        out = []
        for i in range(0, len(X), args.batch):
            x = torch.from_numpy(X[i:i + args.batch]).unsqueeze(1).cuda()
            o = net(x)
            o = o[0] if isinstance(o, (tuple, list)) else o
            out.append(torch.softmax(o.float(), 1).half().cpu())
        return torch.cat(out).numpy()  # (N,K,H,W) float16

    def pair_probs(P, Y):
        """M[y, c] = mean prob of class c over pixels of true class y."""
        M = np.zeros((K, K))
        for y in range(K):
            m = Y == y
            for c in range(K):
                M[y, c] = P[:, c][m].astype(np.float64).mean()
        return M

    def confusion(pred, Y):
        C = np.zeros((K, K), np.int64)
        np.add.at(C, (Y.ravel(), pred.ravel()), 1)
        return C

    res = {}
    # ------------------------------------------------------------------ teacher + reference student
    tea = teacher()
    Pt = {s: probs(tea, data[s][0]) for s in data}
    del tea
    torch.cuda.empty_cache()
    ref = student(args.ref_exp)
    Pr = {s: probs(ref, data[s][0]) for s in ('train', 'valid')}
    del ref
    torch.cuda.empty_cache()

    for s in ('train', 'valid'):
        Y = data[s][1]
        Mt, Ms = pair_probs(Pt[s], Y), pair_probs(Pr[s], Y)
        T = np.minimum(1.0, (Ms + EPS) / (Mt + EPS))
        np.fill_diagonal(T, 1.0)
        acc_t = float((Pt[s].argmax(1) == Y).mean())
        acc_r = float((Pr[s].argmax(1) == Y).mean())
        res['trust_' + s] = {'teacher_pair_prob': Mt.tolist(), 'student_pair_prob': Ms.tolist(), 'trust': T.tolist(),
                             'teacher_pixel_acc': acc_t, 'student_pixel_acc': acc_r,
                             'teacher_conf': confusion(Pt[s].argmax(1), Y).tolist()}
        np.save(os.path.join(os.path.expanduser(args.out), 'trust_%s.npy' % s), T.astype(np.float32))
        print('\n=== %s slices: teacher pixel acc %.4f | student-alone %.4f' % (s, acc_t, acc_r))
        print('pair prob  teacher | student-alone | trust   (rows = true class, cols = wrong class)')
        for y in range(K):
            print('  %-4s ' % CLS[y] + ' '.join('%.4f' % v for v in Mt[y]) + ' | '
                  + ' '.join('%.4f' % v for v in Ms[y]) + ' | ' + ' '.join('%.2f' % v for v in T[y]))

    if args.trust_only:
        json.dump(res, open(os.path.join(os.path.expanduser(args.out), 'cpt_pilot.json'), 'w'), indent=2)
        return

    # ------------------------------------------------------------------ test-set analysis
    T = np.array(res['trust_valid']['trust'])
    flagged = [(y, c) for y in range(K) for c in range(K) if y != c and T[y, c] < args.flag_thr]
    print('\nflagged pairs (trust < %.2f on valid): %s' % (args.flag_thr,
          ', '.join('%s->%s (%.2f)' % (CLS[y], CLS[c], T[y, c]) for y, c in flagged) or 'none'))
    Y = data['test'][1]
    at = Pt['test'].argmax(1)
    t_wrong = at != Y
    t_right = ~t_wrong
    fl = np.zeros_like(t_wrong)
    for y, c in flagged:
        fl |= (Y == y) & (at == c)
    res['test_teacher'] = {'pixel_acc': float(t_right.mean()), 'n_teacher_wrong': int(t_wrong.sum()),
                           'n_flagged_wrong': int(fl.sum()), 'conf': confusion(at, Y).tolist(),
                           'dice': dice(at, Y, K)}
    res['flagged'] = [[CLS[y], CLS[c], float(T[y, c])] for y, c in flagged]

    def ece(P, Y, mask=None, bins=15):
        conf, pred = P.max(1).astype(np.float64), P.argmax(1)
        corr = (pred == Y)
        if mask is not None:
            conf, corr = conf[mask], corr[mask]
        conf, corr = conf.ravel(), corr.ravel()
        e, edges = 0.0, np.linspace(0, 1, bins + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                e += m.mean() * abs(conf[m].mean() - corr[m].mean())
        return float(e)

    rows = []
    for exp in args.students:
        if not os.path.exists(os.path.join(runs, exp, 'best.pth')):
            print('skip %s (no best.pth)' % exp)
            continue
        net = student(exp)
        P = probs(net, data['test'][0])
        del net
        torch.cuda.empty_cache()
        a = P.argmax(1)
        fg = (Y > 0) | (a > 0) | (at > 0)
        r = {'dice': dice(a, Y, K),
             'inherit': float((a[t_wrong] == at[t_wrong]).mean()),
             'inherit_flagged': float((a[fl] == at[fl]).mean()) if fl.any() else float('nan'),
             'acc_teacher_right': float((a[t_right] == Y[t_right]).mean()),
             'agree': float((a == at).mean()), 'agree_fg': float((a[fg] == at[fg]).mean()),
             'ece': ece(P, Y), 'ece_fg': ece(P, Y, fg),
             'pair_prob_test': pair_probs(P, Y).tolist()}
        res[exp] = r
        rows.append((exp, r))
        print('%s done' % exp)

    line = '-' * 112
    print('\n' + line)
    print('TEST (%d slices, 256x256)   teacher Dice RV/Myo/LV %s  mean %.4f | teacher-wrong px %d, flagged %d' % (
        len(Y), ' '.join('%.4f' % d for d in res['test_teacher']['dice']), np.mean(res['test_teacher']['dice']),
        t_wrong.sum(), fl.sum()))
    print(line)
    print('%-14s %-22s %7s %9s %11s %9s %8s %9s %7s %7s' % ('model', 'Dice RV/Myo/LV', 'mean', 'inherit',
          'inh_flag', 'acc_Tok', 'agree', 'agree_fg', 'ECE', 'ECEfg'))
    for exp, r in rows:
        print('%-14s %-22s %7.4f %8.2f%% %10.2f%% %8.2f%% %7.2f%% %8.2f%% %7.4f %7.4f' % (
            exp, ' '.join('%.4f' % d for d in r['dice']), np.mean(r['dice']), 100 * r['inherit'],
            100 * r['inherit_flagged'], 100 * r['acc_teacher_right'], 100 * r['agree'], 100 * r['agree_fg'],
            r['ece'], r['ece_fg']))
    print(line)
    json.dump(res, open(os.path.join(os.path.expanduser(args.out), 'cpt_pilot.json'), 'w'), indent=2)
    print('saved to %s' % os.path.expanduser(args.out))


def dice(a, Y, K):
    out = []
    for c in range(1, K):
        s = (a == c).sum() + (Y == c).sum()
        out.append(float(2 * ((a == c) & (Y == c)).sum() / s) if s else 1.0)
    return out


if __name__ == '__main__':
    main()
