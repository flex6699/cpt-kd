#!/usr/bin/env bash
# All ACDC commands of the segmentation extension (run from the IAD repository root, see README.md).
# Paths: edit the four variables below.
set -euo pipefail
R=${RUNS:-$HOME/cvpr2027/runs}                          # output folder
D=${ACDC:-$HOME/cvpr2027/data/ACDC}                     # ACDC (EMCAD preprocessing): train/ valid/ test/ lists_ACDC/
SAM=${SAM_CKPT:-$HOME/cvpr2027/checkpoints/sam_vit_l_0b3195.pth}
export PYTHONPATH=.:ACDC_Tool
COMMON="--save_path $R --root_dir $D --list_dir $D/lists_ACDC --volume_path $D/test --ckpt $SAM"
FAST="--amp 1 --val_every 5"

# 1) teacher (SAM ViT-L + LoRA) and student alone (EfficientNet-B1), seed 2345 (IAD default)
python ACDC_Tool/train_ACDC_clean.py --mode teacher $COMMON
python ACDC_Tool/train_ACDC_clean.py --mode student $COMMON
python ACDC_Tool/train_ACDC_clean.py --mode student --exp student_s1234 --seed 1234 $COMMON $FAST

# 2) vanilla KD (pixel-wise KL + IAD class-prototype term), tau = 1 and 2
KIS="--mode kis --tea_ckpt $R/teacher/best.pth --alpha 1.0 --beta 1.0"
python ACDC_Tool/train_ACDC_clean.py $KIS --exp kis_kl_t1 --kd_loss kl --kd_tau 1.0 $COMMON $FAST
python ACDC_Tool/train_ACDC_clean.py $KIS --exp kis_kl_t2 --kd_loss kl --kd_tau 2.0 $COMMON $FAST

# 3) pair-trust table on the held-out validation slices + pixel-level inheritance of the existing models
python cpt_pilot.py --root $D --runs $R --ckpt $SAM --students student kis_kl_t1 kis_kl_t2 --out $R/cpt_pilot

# 4) CPT-KD (pixel-wise cleaning of the teacher soft label with the trust table), tau = 1 and 2, two seeds
for s in 2345 1234; do
  sfx=$([ "$s" = 2345 ] && echo "" || echo "_s$s")
  python ACDC_Tool/train_ACDC_clean.py $KIS --exp kis_cpt_t1$sfx --kd_loss cpt --kd_tau 1.0 --trust $R/cpt_pilot/trust_valid.npy --seed $s $COMMON $FAST
  python ACDC_Tool/train_ACDC_clean.py $KIS --exp kis_cpt_t2$sfx --kd_loss cpt --kd_tau 2.0 --trust $R/cpt_pilot/trust_valid.npy --seed $s $COMMON $FAST
done
python ACDC_Tool/train_ACDC_clean.py $KIS --exp kis_kl_t1_s1234 --kd_loss kl --kd_tau 1.0 --seed 1234 $COMMON $FAST
python ACDC_Tool/train_ACDC_clean.py $KIS --exp kis_kl_t2_s1234 --kd_loss kl --kd_tau 2.0 --seed 1234 $COMMON $FAST

# 5) final metrics of all models and the qualitative figure
python cpt_pilot.py --root $D --runs $R --ckpt $SAM --out $R/cpt_final \
    --students student student_s1234 kis_kl_t1 kis_kl_t1_s1234 kis_kl_t2 kis_kl_t2_s1234 \
               kis_cpt_t1 kis_cpt_t1_s1234 kis_cpt_t2 kis_cpt_t2_s1234
python make_seg_figure.py --root $D --runs $R --ckpt $SAM --alone student --kd kis_kl_t1 --cpt kis_cpt_t1 --n 2 \
    --out fig_problem_acdc.png
