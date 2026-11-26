## AIRCRAFT, student mobilenet_v2

Mean ± std over seeds. Inheritance: of test images the teacher gets wrong, % where the student makes the same mistake.

| Method | Seeds | Mean per-class acc. | Inheritance ↓ | Flagged-pair inh. ↓ | Acc. teacher-correct | Agreement | ECE ↓ |
|---|---|---|---|---|---|---|---|
| Student alone | 3 | 79.82 ± 0.63 | 32.62 ± 1.92 | 25.64 ± 2.22 | 90.76 ± 0.44 | 80.21 ± 0.27 | 5.56 ± 0.07 |
| KD | 3 | 84.13 ± 0.25 | 37.58 ± 0.48 | 30.13 ± 2.94 | 95.31 ± 0.10 | 84.83 ± 0.10 | 6.18 ± 0.28 |
| DKD | 3 | 83.20 ± 0.48 | 41.87 ± 0.94 | 32.69 ± 0.00 | 95.05 ± 0.35 | 85.40 ± 0.18 | 5.61 ± 0.53 |
| Weighted soft labels | 3 | 84.73 ± 0.44 | 34.99 ± 0.67 | 28.21 ± 2.94 | 95.59 ± 0.34 | 84.59 ± 0.17 | 6.78 ± 0.40 |
| Knowledge Adjustment | 3 | 84.43 ± 0.65 | 37.52 ± 1.91 | 28.85 ± 6.93 | 95.36 ± 0.56 | 84.86 ± 0.18 | 6.03 ± 0.68 |
| Per-image gate [oofteacher] | 3 | 80.22 ± 0.29 | 38.90 ± 1.88 | 32.05 ± 2.94 | 91.76 ± 0.57 | 82.17 ± 0.55 | 4.39 ± 0.39 |
| Class-level trust [keept] | 3 | 84.07 ± 0.28 | 38.13 ± 1.28 | 29.49 ± 4.00 | 95.10 ± 0.12 | 84.76 ± 0.30 | 6.41 ± 0.39 |
| CPT-KD (ours) [keept] | 3 | 84.01 ± 0.20 | 36.42 ± 0.10 | 21.15 ± 3.33 | 95.03 ± 0.18 | 84.39 ± 0.14 | 6.43 ± 0.17 |
| CPT-KD + DKD (ours) [keept] | 3 | 83.75 ± 0.09 | 38.79 ± 1.01 | 17.95 ± 5.88 | 95.06 ± 0.18 | 84.85 ± 0.16 | 5.98 ± 0.23 |
