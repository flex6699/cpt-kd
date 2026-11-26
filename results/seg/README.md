# ACDC (segmentation extension) results

- `cpt_final/cpt_pilot.json`: final metrics of all ACDC models (Dice, pixel inheritance, flagged-pair inheritance, agreement, ECE), from `seg/cpt_pilot.py`
- `trust_*.npy`: pair-trust tables (rows = true class BG/RV/Myo/LV) from the validation (used) and training slices
- `logs/`: full training logs (arguments on the first line, test Dice at the end)
- Runs `kis_cpt_t2`, `kis_kl_t1_s1234`, `kis_cpt_t1_s1234`, `kis_kl_t2_s1234` and `kis_cpt_t2_s1234` were stopped early
  (shared GPUs, submission deadline); their partial logs are kept for transparency and they are NOT reported.
