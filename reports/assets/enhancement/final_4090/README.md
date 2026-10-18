# Canon official test split — final 4090 evaluation

The 76 predictions were generated blind from low-light RAW with the official
SIED EMA model, 20 DDIM steps, three deterministic ensemble members, and the
1500-step exposure/color refiner. The ground truth was loaded only after each
prediction was complete. `summary.json` and `metrics.csv` describe the refined
outputs; the corresponding unrefined evaluation is stored in `base_evaluation/`.

The full 4K prediction set is intentionally omitted from Git. `comparison.png`
contains six representative rows selected without changing the reported test
metrics.
