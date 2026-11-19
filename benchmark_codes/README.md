# benchmark_codes

The tree that produced every number in [`../results/`](../results/). It is the research bundle
as it ran on the cluster, copied here unchanged except where this file says otherwise, so the
commands recorded in the experiment log still read against it.

KAIROS is **variant 253** throughout this code. [`../kairos/`](../kairos/) is the named entry
point over it, and [`../docs/model.md`](../docs/model.md) maps the equations to the files.

## Layout

| path | what it is |
|---|---|
| `vendor/snn_kuramoto_bidirectional/` | the core: graph, Kuramoto layer, sinusoidal gating, dendrite, membrane, spike readout |
| `vendor/snn_kuramoto_bidirectional/sw240/` | the integration that gates the dendritic and membrane updates, the rollout, the correlation readout and the partition |
| `vendor/full_benchmark/` | `model.py` (the constructor; variant 253 = KAIROS), the training engine, the loss, the geodesic distance, the data layer |
| `vendor/benchmark/` + `vendor/benchmark.md` | the frozen scorer and its specification. **Hash-locked** — see below |
| `vendor/mod_benchmark/` | the harness that loads the frozen scorer and holds `BENCHMARK_LOCK.json` |
| `vendor/neutral_readout/`, `vendor/readout_benchmark/` | mask heads. Variant 253 has `head=None` and uses neither; they are here because the constructor imports them |
| `configs/` | the run configurations, including the one the published rollouts used |
| `contracts/` | the split manifests: which image ids are train, validation and reserved, per dataset |
| `*.py` | training (`run.py`, `recipe.py`), caching (`cache_any.py`), scoring (`score_any.py`, `score_cache.py`, `bg_any.py`, `to_grid64.py`), studies (`spike_timing.py`, `spike_readout.py`, `objcount.py`, `minsize*.py`), baselines (`sa_*.py`, `score_sa.py`) |

## Running it

```bash
export PYTHONPATH=vendor
pip install -r requirements.txt      # torch 2.13, numpy, scipy, h5py, Pillow

# train the core
python run.py --output run train --dataset clevr --model 253 --seed 0

# cache the rollout activity, then score the core's own partition
python cache_any.py --run run --dataset clevr --split fixed320 --min-group 8
python score_cache.py --cache <cache dir> --dataset clevr --split fixed320 \
    --code . --config configs/frontier.json --run run
```

`--min-group 8` is the component floor selected on CLEVR validation; every published number
uses it. [`../docs/evaluation_protocol.md`](../docs/evaluation_protocol.md) is the full
scoring contract, including the two reference values that confirm a setup.

## The frozen scorer

`mod_benchmark.common.immutable()` runs on every call that loads the scorer. It hashes
`vendor/benchmark.md` and every file under `vendor/benchmark/` and compares them against
`vendor/mod_benchmark/BENCHMARK_LOCK.json`, raising if anything differs. That is what
guarantees the published numbers came from unmodified evaluation code, and it is why the
research package names are kept inside a repository called KAIROS: renaming them would break
the check.

**One prose line in the lock was changed for this release.** `vendor/benchmark.md` quoted a
sentence from another README in Korean; it is translated to English here, and the
`benchmark.md` entry of `BENCHMARK_LOCK.json` was recomputed for that one file. The ten code
entries under `vendor/benchmark/` are **unchanged**, so `evaluation.py` and every other scorer
file still hash to exactly what the frozen f9301ed snapshot recorded.

## What is not in this copy

The bundle on the cluster also carried `results/`, `validation/`, `snapshot/` and
`materialize_assets.py` — run outputs, per-job validation records, controller state, and a
fetcher that needs cluster credentials. They are left out, so references to them elsewhere in
these files do not resolve here. The published results live in [`../results/`](../results/)
instead, and the checkpoint locations are in [`../docs/checkpoints.md`](../docs/checkpoints.md).

`vendor/mod_benchmark/DESIGN_KO.md` was translated and is now `DESIGN.md`. The snapshot
manifests `FILES_SHA256.json` and `contracts/source_snapshot.json` still list it under the old
name, because they record the server tree as it was read, not this copy.

This bundle was originally written for an earlier four-model study (252, 253, 254, 240) on a
3,500-image protocol; `METHODS.md` and `DATA_AND_CHECKPOINTS.md` still describe that study.
KAIROS as published is variant 253 trained on **10,000 CLEVR images for 10 epochs**, which is
6,250 updates at batch 16 and 100,000 image exposures — the record is in
[`../results/checkpoint/`](../results/checkpoint/).

Weights and datasets are not in git. `contracts/` fixes the image ids so a rebuilt dataset can
be checked against the one that was used.
