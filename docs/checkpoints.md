# Checkpoints

The trained weights are 336 MB and 201 MB and are not in git. They live on the UNC ACM cluster,
readable by anyone in group `acm`:

```
/work/USERS/tkim1/share/native64_core_253/
    resume.pt     336,219,420 B   sha256 e56b1a4219ab606b35952dab99ce0623220977e11ad87c60aba94cb43a9d11ca
    initial.pt    201,584,309 B   sha256 11981e0eee128336e6f93de4d18aecea0ebf0acfabdfd2ab0cb217677702b490
    complete.json  training_report.json  effective_code.json  binding.json
```

The three small JSON records are copied into [`../results/checkpoint/`](../results/checkpoint/)
so the training record travels with this repository.

## Both files are required

`initial.pt` is not optional. The encoder's feature normalization (`normalization.mean`,
`normalization.std`, and its provenance string) lives there, and the model cannot be built
without it. `kairos.load(directory)` reads both:

```python
import kairos
model = kairos.load('/work/USERS/tkim1/share/native64_core_253')
model.normalization_provenance   # 'own_random_encoder_exact_selected_TRAIN_no_masks'
```

`resume.pt` holds `model`, `optimizer`, `updates`, `images_seen` and the RNG state;
`initial.pt` holds `model` at initialization plus `normalization`.

## What this checkpoint is

From `complete.json`:

| field | value |
|---|---|
| variant | 253 (KAIROS) |
| seed | 0 |
| dataset | clevr |
| updates | 6,250 |
| images seen | 100,000 (10,000 images x 10 epochs, batch 16) |
| diagnostic_only | false |
| evaluation count | 320 |

## Do not load the look-alike

```
/work/USERS/tkim1/sw0255b/run/jobs/model253/clevr/seed0/resume.pt
```

is an **aborted run at 8 updates**, and its file size is **identical** (336,219,420 B) to the
real checkpoint's, so size does not distinguish them. Check the sha256 above.

## Rebuilding instead

```bash
cd benchmark_codes
export PYTHONPATH=vendor
python run.py --output run train --dataset clevr --model 253 --seed 0
```

The rollout is deterministic given the seed and the inference seed, so a rebuilt core can be
compared against the published numbers directly with `score_cache.py`.
