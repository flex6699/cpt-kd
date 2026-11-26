## CUB, resnet18: difference to KD on the same seeds (negative inheritance = better)

| Method − KD | Seeds | Top-1 acc. | Inheritance | Flagged-pair inh. |
|---|---|---|---|---|
| Student alone | 3 | -4.07 ± 0.66 | -8.72 ± 1.82 | -6.32 ± 5.05 |
| DKD | 3 | -0.16 ± 0.28 | 2.47 ± 2.03 | 3.45 ± 3.11 |
| Weighted soft labels | 3 | -0.02 ± 0.04 | -1.71 ± 1.60 | -3.45 ± 3.95 |
| Knowledge Adjustment | 3 | 0.15 ± 0.11 | -0.49 ± 0.94 | -1.44 ± 3.59 |
| Per-image gate [oofteacher] | 3 | -1.58 ± 0.27 | -2.24 ± 0.92 | -1.72 ± 4.48 |
| Class-level trust | 1 | 0.10 | 1.97 | 0.00 |
| Class-level trust [keept] | 3 | 0.21 ± 0.18 | -0.63 ± 2.37 | -1.15 ± 2.17 |
| CPT-KD (ours) | 3 | -0.45 ± 0.42 | -1.65 ± 1.92 | -8.91 ± 2.49 |
| CPT-KD (ours) [keept] | 3 | -0.24 ± 0.32 | -2.14 ± 1.25 | -9.77 ± 1.32 |
| CPT-KD (ours) [pow2] | 3 | -1.01 ± 0.35 | -3.55 ± 1.94 | -9.77 ± 1.32 |
| CPT-KD (ours) [pow2, keept] | 3 | -0.97 ± 0.20 | -2.63 ± 1.14 | -10.34 ± 5.24 |
| CPT-KD (ours) [trustT4] | 1 | 0.36 | 0.20 | -6.03 |
| CPT-KD + DKD (ours) | 1 | -0.85 | -0.99 | -15.52 |
| CPT-KD + DKD (ours) [keept] | 3 | -0.43 ± 0.33 | -1.22 ± 1.81 | -10.06 ± 4.07 |
