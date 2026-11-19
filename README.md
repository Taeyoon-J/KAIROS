# KAIROS

**K**uramoto **A**ssemblies **I**nferred from **R**hythmic **O**scillatory **S**ynchrony.

KAIROS groups the patches of a static image into object masks using *when* its units fire
rather than what they encode. Image patches become coupled Kuramoto oscillators; their phases
gate a spiking core; patches whose emitted spike traces rise and fall together, and which the
image-derived graph also connects, are merged into one object. The network trains without
ground-truth masks and needs no object count at inference.

Trained on 10,000 CLEVR images for 10 epochs (100,000 image exposures, one seed), it reaches
**0.841 foreground ARI on CLEVR** against 0.633 for the strongest baseline measured under the
same protocol, and scores higher on Tetrominoes and Multi-dSprites **without training on
either**.

## Results

All models trained on the same 10,000 CLEVR images for 10 epochs and evaluated on 320 held-out
images per dataset by the same scoring function. Tetrominoes and Multi-dSprites are zero-shot.
Metrics are foreground ARI / foreground IoU / matched-object IoU on the 64x64 patch grid, every
model with the minimum-region-size rule at *f* = 0.01. One training seed.

| Model | CLEVR (in-domain) | Tetrominoes (zero-shot) | Multi-dSprites (zero-shot) |
|---|---|---|---|
| **KAIROS (ours)** | **.8408 / .7528 / .6387** | **.8568 / .7950 / .8170** | **.6966 / .8190 / .7625** |
| AKOrN (attn) + up-tiling | .6326 / .3468 / .3636 | .5484 / .7230 / .4167 | .5505 / .6539 / .6348 |
| AKOrN (conv) | .6046 / .4219 / .2769 | .7202 / .7540 / .5913 | .4055 / .6834 / .5434 |
| AKOrN (attn) | .6027 / .3327 / .3750 | .5676 / .7383 / .4687 | .4343 / .6019 / .5751 |
| ItrSA | .5794 / .2787 / .3853 | .6319 / .7137 / .5687 | .4817 / .2489 / .6093 |
| Slot Attention | .2872 / .2220 / .0801 | .1739 / .2192 / .1071 | .1930 / .0876 / .0680 |
| CAE++ | .2838 / .2193 / .0882 | .2715 / .2452 / .1901 | .2648 / .1048 / .1125 |
| CtCAE | .2357 / .2148 / .0964 | .2681 / .2420 / .2034 | .3079 / .1158 / .1609 |
| CAE | .1091 / .2008 / .0634 | .1526 / .2305 / .1192 | .1200 / .1038 / .0630 |
| *Best baseline, per cell* | .6326 / .4267 / .3853 | .7202 / .7540 / .5913 | .5505 / .6834 / .6348 |
| **Our margin** | **+.2082 / +.3261 / +.2534** | **+.1366 / +.0410 / +.2257** | **+.1461 / +.1356 / +.1277** |

The *best baseline* row takes the best baseline in each cell separately, so it is a composite of
three architectures that no single baseline attains.

Three things to read with the table:

- **Scores are not comparable with published pixel-protocol numbers.** Ground truth is
  quantised to the patch grid and scored there; a prediction is never expanded to pixels. The
  ceiling is 1.0 by construction. See [docs/evaluation_protocol.md](docs/evaluation_protocol.md).
- **The baselines are tuned on the target, and KAIROS is not.** Their cluster count and
  background mapping are selected on each target dataset's own validation split; every free
  choice of ours is fixed on CLEVR validation and transferred unchanged.
- **One training seed.** No error bars are claimed. Reseeding one baseline moved its CLEVR
  FG-ARI by up to 0.134, which is evidence about that baseline, not about KAIROS.

Every table in [`results/`](results/) carries its own conditions in a header comment:
headline table, the minimum-region-size rule, rollout length, data scaling, released weights,
per-baseline training budget, and the timing ablation below.

### What carries the grouping

Measured on the same rollouts for the first 40 held-out CLEVR images
([`results/timing_ablation.json`](results/timing_ablation.json)):

| | FG-ARI | FG-IoU | Obj-IoU | regions/image |
|---|---|---|---|---|
| control | .8648 | .8028 | .6925 | 7.8 |
| each patch's traces circularly shifted | **.0735** | .7358 | **.0325** | 562.3 |
| the oscillatory gate fed to the same readout | .8648 | .8028 | .6925 | 7.8 |

Shifting each patch's traces by its own offset preserves every patch's firing rate and waveform
and destroys only the alignment *between* patches: the partition collapses. Foreground IoU
barely moves, because it compares only the union of predicted foreground, which shattering a
region leaves nearly unchanged. Substituting the oscillatory gate for the spike traces
reproduces every mask on all 40 images — the firing indicator is 1 in 99.99% of settled steps,
so on this evidence the timing that matters is the gate's, transmitted by the spiking layer
without loss, and no separate grouping information can be attributed to the firing threshold.

## Install and use

```bash
git clone https://github.com/Taeyoon-J/KAIROS.git && cd KAIROS
pip install -r benchmark_codes/requirements.txt    # torch 2.13, numpy, scipy, h5py, Pillow
```

```python
import kairos

model = kairos.build(seed=0)                       # fresh weights
model = kairos.load('/path/to/native64_core_253')  # the trained core
out = model(rgb_uint8, inference_seed=0)           # rgb_uint8: [B, 128, 128, 3] uint8
out['q']                                           # patch affinity, the readout's input
out['trace']['component_spikes']                   # the emitted spike traces, per component
out['trace']['component_gates']                    # the oscillatory gates
out['trace']['theta']                              # the Kuramoto phases
```

`kairos.load` needs both `resume.pt` and `initial.pt` from the checkpoint directory; the feature
normalization lives in `initial.pt` and the model cannot be built without it.
[`docs/checkpoints.md`](docs/checkpoints.md) has the locations and SHA-256 hashes.

A full rollout is 1024 steps over 4096 patches and costs about 10.8 s per frame on one GPU;
`total_steps=256` keeps most of the score at 8.8 s ([`results/rollout_length.csv`](results/rollout_length.csv)).

## Repository map

| path | what it is |
|---|---|
| `kairos/` | the public entry point: `build`, `load`, and the variant constants |
| `benchmark_codes/vendor/snn_kuramoto_bidirectional/` | the core: Kuramoto layer, sinusoidal gating, dendrite, membrane, spike readout |
| `benchmark_codes/vendor/snn_kuramoto_bidirectional/sw240/` | the integration that gates the dendritic and membrane updates, the rollout, and the correlation readout |
| `benchmark_codes/vendor/full_benchmark/` | the model constructor (`model.py`, variant 253 = KAIROS), training engine, loss, geodesic distance, data |
| `benchmark_codes/vendor/benchmark/` | the frozen scorer. Hash-locked; do not edit — see below |
| `benchmark_codes/*.py` | training (`run.py`, `recipe.py`), caching (`cache_any.py`), scoring (`score_any.py`, `score_cache.py`), readout studies (`spike_timing.py`, `spike_readout.py`, `objcount.py`, `minsize*.py`), baselines (`sa_*.py`, `score_sa.py`) |
| `benchmark_codes/contracts/` | split manifests: which image ids are train, validation and reserved, by dataset |
| `results/` | the tables above, as CSV, each with its conditions |
| `docs/` | the model's equations against the code, the evaluation protocol, checkpoints |

`benchmark_codes/vendor/benchmark/` is verified on every call:
`mod_benchmark.common.immutable()` hashes each file under it against
`vendor/mod_benchmark/BENCHMARK_LOCK.json` and refuses to load the scorer if anything moved or
changed. That is why the research package names survive inside a repository called KAIROS —
renaming them would break the one check that guarantees these numbers came from unmodified
evaluation code. `kairos/` is the KAIROS-named surface over that tree.

## Reproducing

```bash
cd benchmark_codes
export PYTHONPATH=vendor

# train the core (variant 253 = KAIROS), 10,000 CLEVR images x 10 epochs
python run.py --output run train --dataset clevr --model 253 --seed 0

# cache the rollout activity, then score the core's own partition
python cache_any.py --run run --dataset clevr --split fixed320 --min-group 8
python score_cache.py --cache <cache dir> --dataset clevr --split fixed320 \
    --code . --config configs/frontier.json --run run
```

Two reference values confirm a setup before any new number is trusted: the 253 core's own
partition on CLEVR `fixed320` at grid 64 scores **0.8421 / 0.7545 / 0.6410**, and a grid-32 core
on Tetrominoes `test`, lifted to the grid, scores **0.7812 / 0.7015 / 0.7431**. The first
exercises the native-grid path, the second the lift.
[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) is the full contract, including how
to score predictions from a different model by the same function.

`--min-group 8` is the component floor, selected on CLEVR validation; every number published
here uses it.

## Limitations

- **Counting is weaker than grouping.** With the size rule KAIROS predicts 5.15 objects per
  CLEVR image against a true 6.36, exact on 25.0%. On Multi-dSprites 2.64 against 2.49 (53.1%
  exact) and on Tetrominoes 5.28 against exactly 3 (9.4%). Most of the error is single-patch
  fragments: without the rule the predictions are 8.04, 6.62 and 21.07.
- **The firing indicator is saturated** over the readout window, so these results support
  oscillatory timing and do not isolate a contribution from the threshold nonlinearity.
- **One seed**, and the held-out split is reserved from the training corpus rather than an
  independent benchmark test set.
- **No reconstruction objective**, unlike every baseline in the table, which is one reason
  KAIROS still splits some objects into parts.

## Provenance and what is not here

- The trained weights are 336 MB and 201 MB and are **not in git**; the locations and hashes are
  in [`docs/checkpoints.md`](docs/checkpoints.md).
- Datasets are not redistributed. `benchmark_codes/contracts/` fixes the image ids so a rebuilt
  dataset can be checked against the one that was used.
- `results/checkpoint/` holds the trained core's own records: `complete.json`,
  `training_report.json` and `effective_code.json` (6,250 updates, 100,000 images seen, variant
  253, seed 0, `diagnostic_only: false`).
- Some scripts carry absolute cluster paths from the machine they ran on. They are left as they
  ran rather than rewritten, so the commands in the experiment log stay readable against them.
- Everything here is in English. Three documents were originally Korean and were translated:
  `benchmark_codes/README.md` (rewritten for this repository), `vendor/mod_benchmark/DESIGN.md`
  (was `DESIGN_KO.md`), and one quoted sentence in the hash-locked `vendor/benchmark.md`, whose
  lock entry was recomputed for that file alone — the ten code hashes under `vendor/benchmark/`
  are unchanged.
- `benchmark_codes/vendor/benchmark/build_gamma_grid.py` imports a helper
  (`snn_kuramoto_bidirectional.gamma_ordering`) that is not in this tree. It is a frozen
  one-off grid builder from an earlier generation of the code, is not on the model's path, and
  is shipped only because the hash-locked directory must be complete.

## License

[CC BY-NC 4.0](LICENSE) — free to share and adapt with attribution, non-commercial use only.
