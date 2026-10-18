# GeoCorrection RTX 4090 final results

`predicted/` contains the held-out 22-image automatic pipeline driven by
PageMaskUNet predictions. `oracle/` repeats the same image keys with dataset
masks to measure the remaining mask-quality gap. The two input-boundary values
use different masks and must only be compared with the corresponding corrected
value inside the same setting.

The full predicted masks and corrected images are reproducible from the code,
the fixed `split.json`, and the checkpoint documented in the final report; they
are omitted from Git to keep the repository small.
