## AIRCRAFT, mobilenet_v2: difference to KD on the same seeds (negative inheritance = better)

| Method − KD | Seeds | Mean per-class acc. | Inheritance | Flagged-pair inh. |
|---|---|---|---|---|
| Student alone | 3 | -4.31 ± 0.87 | -4.96 ± 2.31 | -4.49 ± 4.84 |
| DKD | 3 | -0.93 ± 0.32 | 4.30 ± 0.50 | 2.56 ± 2.94 |
| Weighted soft labels | 3 | 0.60 ± 0.59 | -2.59 ± 0.42 | -1.92 ± 5.09 |
| Knowledge Adjustment | 3 | 0.31 ± 0.89 | -0.06 ± 2.00 | -1.28 ± 4.84 |
| Per-image gate [oofteacher] | 3 | -3.91 ± 0.25 | 1.32 ± 2.34 | 1.92 ± 1.92 |
| Class-level trust [keept] | 3 | -0.06 ± 0.50 | 0.55 ± 1.48 | -0.64 ± 6.75 |
| CPT-KD (ours) [keept] | 3 | -0.12 ± 0.39 | -1.16 ± 0.57 | -8.97 ± 2.22 |
| CPT-KD + DKD (ours) [keept] | 3 | -0.38 ± 0.21 | 1.21 ± 0.58 | -12.18 ± 8.67 |
