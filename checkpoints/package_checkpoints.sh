#!/usr/bin/env bash
# Collect all checkpoints / logits into one folder for upload (e.g. to Hugging Face) and write sha256.txt.
# Usage: bash checkpoints/package_checkpoints.sh <ml_research_dir> <cvpr2027_runs_dir> <out_dir>
set -euo pipefail
ML="$1"; SEG="$2"; OUT="$3"
mkdir -p "$OUT"
for ds in cub aircraft; do
  mkdir -p "$OUT/cls/$ds"
  cp "$ML/runs/$ds/final/oof/teacher_full.pt" "$OUT/cls/$ds/teacher_full.pt"
  (cd "$ML/runs/$ds/final/students" && tar -czf "$OUT/cls/$ds/student_test_logits.tar.gz" */test_logits.npz)
done
for r in teacher student student_s1234 kis kis_kl_t1 kis_kl_t2 kis_kl_t1_s1234 kis_kl_t2_s1234 \
         kis_cpt_t1 kis_cpt_t2 kis_cpt_t1_s1234 kis_cpt_t2_s1234; do
  if [ -f "$SEG/$r/best.pth" ]; then mkdir -p "$OUT/seg/acdc/$r"; cp "$SEG/$r/best.pth" "$OUT/seg/acdc/$r/best.pth"; fi
done
(cd "$OUT" && find . -type f ! -name sha256.txt | sort | xargs sha256sum > sha256.txt)
echo "packaged into $OUT"; du -sh "$OUT"
