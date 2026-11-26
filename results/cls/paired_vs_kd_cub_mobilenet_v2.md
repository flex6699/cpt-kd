## CUB, mobilenet_v2: difference to KD on the same seeds (negative inheritance = better)

| Method − KD | Seeds | Top-1 acc. | Inheritance | Flagged-pair inh. |
|---|---|---|---|---|
| Student alone | 3 | -2.97 ± 0.18 | -7.17 ± 1.68 | -4.60 ± 4.98 |
| DKD | 3 | -0.19 ± 0.14 | 2.07 ± 1.10 | 1.44 ± 8.64 |
| Weighted soft labels | 3 | 0.07 ± 0.28 | -1.35 ± 1.16 | 0.29 ± 7.53 |
| Knowledge Adjustment | 3 | 0.01 ± 0.07 | -0.99 ± 1.04 | -1.15 ± 2.77 |
| Per-image gate [oofteacher] | 3 | -1.04 ± 0.41 | -1.97 ± 1.28 | 0.29 ± 4.07 |
| Class-level trust [keept] | 3 | -0.04 ± 0.03 | 0.07 ± 0.75 | -1.44 ± 6.53 |
| CPT-KD (ours) [keept] | 3 | -0.51 ± 0.03 | -1.65 ± 0.74 | -6.03 ± 4.48 |
| CPT-KD + DKD (ours) [keept] | 3 | -0.63 ± 0.16 | 0.03 ± 1.76 | -3.74 ± 7.82 |
