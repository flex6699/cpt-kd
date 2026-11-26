## AIRCRAFT, student resnet18

Mean ± std over seeds. Inheritance: of test images the teacher gets wrong, % where the student makes the same mistake.

| Method | Seeds | Mean per-class acc. | Inheritance ↓ | Flagged-pair inh. ↓ | Acc. teacher-correct | Agreement | ECE ↓ |
|---|---|---|---|---|---|---|---|
| Student alone | 3 | 78.98 ± 0.44 | 31.52 ± 2.09 | 21.15 ± 3.85 | 89.71 ± 0.32 | 79.15 ± 0.21 | 8.18 ± 0.13 |
| KD | 3 | 82.84 ± 0.19 | 36.58 ± 0.85 | 22.44 ± 4.00 | 93.87 ± 0.28 | 83.47 ± 0.35 | 6.91 ± 0.32 |
| DKD | 3 | 78.91 ± 4.73 | 38.62 ± 3.72 | 23.72 ± 4.44 | 90.49 ± 4.91 | 81.08 ± 4.69 | 6.60 ± 0.49 |
| Weighted soft labels | 3 | 82.77 ± 0.39 | 34.60 ± 1.44 | 22.44 ± 7.77 | 93.51 ± 0.45 | 82.82 ± 0.51 | 7.62 ± 0.27 |
| Knowledge Adjustment | 3 | 82.84 ± 0.31 | 36.58 ± 1.62 | 21.15 ± 5.77 | 93.91 ± 0.13 | 83.51 ± 0.19 | 7.02 ± 0.38 |
| Per-image gate [oofteacher] | 3 | 79.09 ± 0.22 | 37.13 ± 1.81 | 28.85 ± 9.62 | 90.98 ± 0.38 | 81.21 ± 0.60 | 4.88 ± 0.23 |
| Class-level trust [keept] | 3 | 82.97 ± 0.41 | 35.54 ± 1.43 | 25.00 ± 1.92 | 93.98 ± 0.31 | 83.37 ± 0.22 | 6.80 ± 0.36 |
| CPT-KD (ours) [keept] | 3 | 82.68 ± 0.13 | 34.99 ± 1.38 | 18.59 ± 1.11 | 93.79 ± 0.15 | 83.12 ± 0.31 | 7.33 ± 0.16 |
| CPT-KD + DKD (ours) [keept] | 3 | 81.72 ± 0.31 | 38.73 ± 0.63 | 19.87 ± 4.84 | 93.46 ± 0.26 | 83.53 ± 0.24 | 7.08 ± 0.30 |
