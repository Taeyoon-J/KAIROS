# How every number in this repository was scored

Copy `benchmark_codes/score_any.py`, point it at your predictions, and you get the same three
numbers by the same code path. Two reference values at the end let you confirm your setup
before you trust it.

## The whole thing is one function call

```python
from mod_benchmark import common as c
metric = c.benchmark_module('evaluation')        # -> vendor/benchmark/evaluation.py
target = torch.from_numpy(Dataset(ds, output).labels(ids, split))
s = metric.evaluate_patch_masks(torch.from_numpy(pred), target)   # pred, target: [N, 64, 64]
fg_ari  = float(s['mean']['fg_ari'])
fg_iou  = float(s['mean']['foreground_iou'])
obj_iou = float(s['mean']['matched_object_iou'])
```

`evaluate_patch_masks(prediction, target, background_id=0)` is at
`benchmark_codes/vendor/benchmark/evaluation.py:194`. Its docstring is the contract:

> Evaluate three metrics on aligned `[B, grid_h, grid_w]` patch labels. Return `per_image`,
> `mean` (scalar image-wise average), and `valid_count`. Undefined scores are excluded per
> metric. **No combined weighted score is formed.** Each patch must have exactly one ID:
> resolve overlapping masks before calling, **without using ground truth**.

That directory is hash-locked: `mod_benchmark.common.immutable()` compares every file under it
against `vendor/mod_benchmark/BENCHMARK_LOCK.json` on each call and raises if anything differs.

## Run it

```bash
cd benchmark_codes
python score_any.py --code . --config configs/frontier.json --run <run dir> \
  --pred mypreds.npz --dataset clevr --split fixed320
```

`mypreds.npz` holds exactly two arrays:

| array | dtype | shape | meaning |
|---|---|---|---|
| `pred` | int64 | `[N, G, G]` | patch labels, **background = 0**, one id per patch |
| `ids` | int64 | `[N]` | image ids, **same order as `pred`** |

## The protocol, in four points

**1. Both sides live on the patch grid.** The ground truth is quantised to the grid and the
prediction is compared there. A prediction is **never expanded to pixels**. That is why the
ceiling is **1.0** — and why these numbers are **not comparable with pixel-protocol numbers
from papers**, whose ceiling is below 1.0 because a grid prediction can never recover pixel
boundaries.

**2. The grid is 64.** A coarser prediction is lifted by an exact `np.repeat` (64/G must be a
whole number); a finer one is reduced by patch majority vote, ties to the smaller label id
(`to_grid64.py`). No interpolation, so no new information is invented.

**3. Background is id 0, and it matters.** The metrics are foreground-only in the *truth's*
sense — a patch the truth calls background is excluded from `fg_ari`. So which of your labels
you map to 0 moves the score. Decide it by a rule fixed **in advance**, on **validation**,
never on the split you report. `bg_any.py` does this from four rules fixed up front
(`global_largest`, `image_largest`, `border_largest`, `border_union`), choosing on validation by
FG-ARI with foreground IoU as the tie-break.

**4. The three metrics are reported together and never combined.** No single score.

## What each metric can and cannot see

- **Foreground ARI** restricts both label maps to the patches whose *ground truth* is
  foreground. Predicting background on a true object is charged; **labelling background as
  object is invisible to it**.
- **Foreground IoU** merges every non-background identity on each side and compares the two
  sets over their **union**. It is the metric that charges false foreground — the error FG-ARI
  cannot see.
- **Matched-object IoU** solves a one-to-one Hungarian assignment, sums the matched IoUs and
  divides by the number of *true* objects. Extra predicted objects carry no direct penalty
  beyond going unmatched, so over-segmentation is **not** charged here; the object count is
  reported separately in `results/min_region_size_rule.csv`.

## Three traps, each of which cost us something

**A readout that designates no background.** If nothing in your prediction is 0, then
`prediction != 0` is the whole image and `foreground_iou` collapses to the truth's foreground
fraction — a **constant**, identical for every model and every checkpoint. One of our variants
ran to completion reading `foreground_iou = 0.2021225` at all five checkpoints, bitwise
identical, and it was read as a plateau rather than as the bug it was. **If a metric is constant
to seven decimals across checkpoints, stop the run.**

**`mean` is a nanmean.** Images where a metric is undefined are dropped **per metric**, so the
three means can rest on different image counts. Always confirm `valid_count == N` before
quoting a mean.

**Label ids are arbitrary.** Only the partition and which id is 0 carry meaning. Do not expect
your ids to match the truth's, and do not renumber using the truth.

## Confirm your setup with these two

Both are reproduced by `score_any.py` as shipped, on 320 images:

| what | command inputs | expected |
|---|---|---|
| the core's own partition, CLEVR | `--dataset clevr --split fixed320`, grid 64 | **0.8421 / 0.7545 / 0.6410** |
| a grid-32 core on Tetrominoes, lifted | `--dataset tetrominoes --split test`, grid 32 | **0.7812 / 0.7015 / 0.7431** |

The first exercises the native-grid path, the second the lift. `score_cache.py` scores the core
straight from a `cache_any.py` directory with no head at all, and reproduces CLEVR
0.8421 / 0.7545 / 0.6410 and Tetrominoes 0.8580 / 0.7885 / 0.8173.

## Splits

Fixed by the manifests in `benchmark_codes/contracts/`; do not redraw them. CLEVR validation is
ids 1000–1319 and the reserved evaluation split `fixed320` is 1320–1639. Tetrominoes and
Multi-dSprites have 1,000 validation and 1,000 test ids, and the rows published here use the
**first 320** of test.

No hyperparameter, threshold or rule of any model in `results/table1_main.csv` was selected on
those reserved images. They were read once, after every choice was frozen.
