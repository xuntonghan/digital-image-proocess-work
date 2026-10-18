# PageMaskUNet training record

The split is stratified by the seven filename-derived distortion families. The
156/22/22 train/validation/test keys are stored in `split.json`; `summary.json`
contains every epoch plus the final held-out test metrics. The 90 MB checkpoint
is excluded from Git and stored locally at
`checkpoints/geocorrection/page_mask_unet/best.pt`.

Checkpoint SHA256:
`250789695f3c92b3acaa2c373fb6a51083cf0dfa54093ed3acd3c2cd084eccc6`.
