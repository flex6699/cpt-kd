## CUB, student mobilenet_v2

Mean ± std over seeds. Inheritance: of test images the teacher gets wrong, % where the student makes the same mistake.

| Method | Seeds | Top-1 acc. | Inheritance ↓ | Flagged-pair inh. ↓ | Acc. teacher-correct | Agreement | ECE ↓ |
|---|---|---|---|---|---|---|---|
| Student alone | 3 | 78.75 ± 0.15 | 40.34 ± 0.85 | 32.47 ± 3.26 | 90.01 ± 0.04 | 81.33 ± 0.12 | 1.62 ± 0.16 |
| KD | 3 | 81.72 ± 0.03 | 47.52 ± 0.86 | 37.07 ± 4.80 | 93.66 ± 0.13 | 85.59 ± 0.16 | 3.71 ± 0.13 |
| DKD | 3 | 81.53 ± 0.10 | 49.59 ± 0.25 | 38.51 ± 3.89 | 93.88 ± 0.13 | 86.14 ± 0.08 | 2.27 ± 0.21 |
| Weighted soft labels | 3 | 81.79 ± 0.27 | 46.17 ± 0.54 | 37.36 ± 2.77 | 93.62 ± 0.16 | 85.32 ± 0.07 | 4.31 ± 0.12 |
| Knowledge Adjustment | 3 | 81.73 ± 0.09 | 46.53 ± 0.23 | 35.92 ± 2.77 | 93.67 ± 0.10 | 85.43 ± 0.10 | 3.61 ± 0.14 |
| Per-image gate [oofteacher] | 3 | 80.68 ± 0.39 | 45.54 ± 0.85 | 37.36 ± 2.17 | 92.47 ± 0.36 | 84.27 ± 0.30 | 1.61 ± 0.19 |
| Class-level trust [keept] | 3 | 81.68 ± 0.06 | 47.58 ± 0.17 | 35.63 ± 3.26 | 93.80 ± 0.18 | 85.72 ± 0.13 | 3.63 ± 0.11 |
| CPT-KD (ours) [keept] | 3 | 81.21 ± 0.04 | 45.87 ± 0.60 | 31.03 ± 3.11 | 93.14 ± 0.22 | 84.88 ± 0.28 | 4.44 ± 0.10 |
| CPT-KD + DKD (ours) [keept] | 3 | 81.09 ± 0.15 | 47.55 ± 1.04 | 33.33 ± 3.26 | 93.24 ± 0.17 | 85.25 ± 0.28 | 3.57 ± 0.17 |
