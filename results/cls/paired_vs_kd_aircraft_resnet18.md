## AIRCRAFT, resnet18: difference to KD on the same seeds (negative inheritance = better)

| Method − KD | Seeds | Mean per-class acc. | Inheritance | Flagged-pair inh. |
|---|---|---|---|---|
| Student alone | 3 | -3.86 ± 0.27 | -5.07 ± 2.68 | -1.28 ± 4.84 |
| DKD | 3 | -3.94 ± 4.56 | 2.04 ± 3.27 | 1.28 ± 1.11 |
| Weighted soft labels | 3 | -0.07 ± 0.54 | -1.98 ± 2.01 | -0.00 ± 5.09 |
| Knowledge Adjustment | 3 | -0.01 ± 0.13 | 0.00 ± 2.45 | -1.28 ± 7.77 |
| Per-image gate [oofteacher] | 3 | -3.75 ± 0.29 | 0.55 ± 1.16 | 6.41 ± 5.88 |
| Class-level trust [keept] | 3 | 0.13 ± 0.51 | -1.05 ± 1.00 | 2.56 ± 5.88 |
| CPT-KD (ours) [keept] | 3 | -0.17 ± 0.32 | -1.60 ± 1.61 | -3.85 ± 3.33 |
| CPT-KD + DKD (ours) [keept] | 3 | -1.12 ± 0.29 | 2.15 ± 0.86 | -2.56 ± 7.28 |
