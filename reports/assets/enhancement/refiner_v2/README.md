# Canon SIED refiner v2 training record

The lightweight refiner was initialized from the previous validated checkpoint
and fine-tuned for 1500 steps on 832 cached blind SIED predictions. Model
selection used all 92 validation images every 25 steps. The 76-image official
test manifest was not used during training or checkpoint selection.

`best.pt` is included here because it is under 200 KB; the larger official SIED
EMA checkpoint remains excluded and must be obtained separately.
