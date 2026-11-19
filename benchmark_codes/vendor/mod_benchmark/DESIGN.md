# SW0250: the two-dataset transfer benchmark for SW0240

Historical design note for the harness in this directory, kept for provenance. It describes
the SW0250 transfer study on model 240, not the KAIROS (variant 253) protocol the repository
reports; see the repository root `README.md` for that.

Approved scope: continue training from the three existing independent seeds of model 240.
`benchmark/` and `benchmark.md` stay frozen at the f9301ed originals, and all code and data
adapters are written in this folder only. The SHAs of the 11 protected files are checked
against `BENCHMARK_LOCK.json` before and after every run.

The data is the correct `multi_dsprites` (colored_on_colored, HWE) and `tetrominoes` (EHW)
family under `/work/USERS/tkim1/mod`. The `mdsprites` family is not used. The original
manifest's TRAIN 0..5999, validation 6000..6999 and TEST 7000..7999 are kept. Nothing is
downloaded separately and no split is redrawn at random.

Each dataset gets zero_shot and finetune x seed 0/1/2, so 12 jobs. They start from the
portable SW0240 endpoint `SW0249_matched35000_terminal`. zero_shot uses that checkpoint with
no further training. finetune trains on TRAIN, 6,000 images x 10 passes (60,000 exposures,
batch 16, 3,750 actual Adam updates), with a fresh Adam, LR 3e-5 / 3e-4 / 3e-4 and gradient
clip 1, updating only the whole encoder and the actual sensory projection. The graph, core,
integration, readout and the feature statistics stay fixed. The existing frozen teacher's
frequency-denoising and gamma-anchor mask-supervised objective is kept. The teacher's encoder
and frequency computations run on the same GPU as the student under `no_grad`, to keep the CPU
from becoming the bottleneck. The checkpoint's encoder is never replaced by a new encoder or by
precomputed gamma.

Only the fixed ten-pass terminal checkpoint is evaluated: no validation-based selection, no
early stopping, no loss search. TEST masks are never used in training. The prediction protocol
is completed for all 1,000 TEST images of each dataset before scoring reads the pixel truth.
The original `preflight.py` is run unmodified at the start of every real evaluation.

Inputs are prepared as nearest-neighbour 128x128 RGB by the original `benchmark/make128.py`.
The native model stays at the real 64x64 SW0240 for both datasets. Multi-dSprites is evaluated
on the native64 prediction directly. For Tetrominoes no grid64 ground truth is built, because
the original 35-pixel cells would leave empty cells; during training the existing grid32 TRAIN
partition is repeated 2x2 to match the model's 64x64 frequency, and at evaluation the native64
prediction is reduced to 32x32 by the modal rule of the original
`benchmark/make_targets_grid.py`, then that grid32 is expanded to the same function's 35-pixel
cell edges. This fixed readout adapter is applied identically to zero_shot and finetune. No
separate experiment rebuilds the core at native32 or converts node parameters.

Pixel FG-ARI, foreground IoU and Hungarian-matched object IoU are taken by calling the original
functions in the protected `benchmark/evaluation.py` directly. Pixel MBO (kept distinct from
one-to-one Hungarian matching), pred_fg, true_fg and the native output group count may be
reported alongside. The oracle grid ceiling of the actual TEST split is measured with the
original modal/expand/evaluation functions; it is not to be confused with the 0.9409 / 1.0000
measured on the original validation300. Three-seed means and the sample standard deviation are
reported. For the models that receive no further training, the fact that the source CLEVR
training was mask-supervised is stated explicitly. No claim of beating a published model is
made from these results alone, without reproducing that paper's baseline.

Before running, the Linux unit checks, the real two-dataset CPU gradient check and a real CUDA
forward/readout smoke on TRAIN image 0 must pass. Each job keeps its immutable binding, the
resumable optimizer/encoder/projection boundary, a per-row prediction SHA, and the terminal
model SHA with the dataset and file SHAs. Original data, original checkpoints and other users'
processes are never modified. Jobs share GPUs with at least 8 GiB free and are not stopped
automatically just because another user appears. Structure and loss exploration does not
continue after completion.
