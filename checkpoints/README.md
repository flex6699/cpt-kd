# Checkpoints

## Why the weights are not in git

| File | Size | Count | Total |
|---|---|---|---|
| ResNet-50 teacher (`teacher_full.pt`), per dataset | 96 MB | 2 | 192 MB |
| Classification student test logits (`test_logits.npz`), all runs | 1–5 MB | 120 | 372 MB |
| ACDC teacher (SAM ViT-L LoRA + decoder, `teacher/best.pth`) | 19 MB | 1 | 19 MB |
| ACDC students (`student*/kis*/best.pth`) | 49 MB | 9 | 443 MB |
| SAM ViT-L base weights (`sam_vit_l_0b3195.pth`, Meta, Apache-2.0) | 1.25 GB | 1 | 1.25 GB |

Together this is about 2.3 GB, well above GitHub's 100 MB per-file / ~1 GB per-repository guidance. The weights
are therefore hosted separately:

**Download:** `CHECKPOINT_URL` (Hugging Face model repository; the folder layout matches the paths below).

The SAM base weights come from the official source:
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

Already **in this repository** (small): all out-of-fold predictions and the teacher's test logits
(`results/cls/runs/<dataset>/final/oof/*.npz`), the trust tables (`trust_T1.npy`, `trust_T4.npy`), every
student's `results.json` / `args.json`, and the ACDC trust tables (`results/seg/trust_*.npy`). These are enough
to recompute every table of the report and the trust scores without any weights.

`cls/train_student.py` saves each classification student's **test logits, not its weights**. This keeps the
120 runs small, and the logits are all that any reported metric needs. A student's weights can be regenerated
with the single command in its `args.json` (5–11 minutes).

## SHA-256 of the files used

See [`sha256.txt`](sha256.txt). Verify with `sha256sum -c sha256.txt` from the folder that holds the downloaded files.

## Regenerating instead of downloading

| Checkpoint | Command |
|---|---|
| CUB / Aircraft teacher + out-of-fold models | `python cls/oof_and_trust.py --data data/tea_kd/<dataset> --models teacher` (and `--models student`) |
| any classification student | `python cls/train_student.py` with the arguments in its `results/cls/runs/.../args.json` |
| ACDC teacher / students | the commands in `seg/run_acdc.sh` (arguments also at the top of each `log.txt`) |

## Packaging for upload

`bash checkpoints/package_checkpoints.sh <ml_research_dir> <cvpr2027_runs_dir> <out_dir>` collects all files
into `<out_dir>` with the layout above. Then run
`huggingface-cli upload <user>/cpt-kd-checkpoints <out_dir> .`
