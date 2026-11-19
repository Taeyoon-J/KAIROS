# Results

Every file carries its conditions in a header comment. Nothing here is a combined score: the
three metrics are always reported together and never averaged into one number.

| file | what it holds |
|---|---|
| `table1_main.csv` | the headline table: nine models x three datasets x three metrics, every model with the minimum-region-size rule at f = 0.01, one training seed |
| `min_region_size_rule.csv` | the same core and the same masks with and without the size rule, plus the object-count statistics, on all three datasets |
| `timing_ablation.json` | control / shifted traces / oscillatory gate, 40 held-out CLEVR images, and the within-object, across-object and object-background affinity means |
| `rollout_length.csv` | CLEVR score against rollout length T, with seconds per frame |
| `data_scaling.csv` | CLEVR score against the number of unique training images |
| `released_weights.csv` | Slot Attention's released checkpoint (~32M exposures) against the matched-budget retraining and against KAIROS, all without the size rule |
| `training_budget.csv` | each baseline at our budget against its own published figure, by **its own** metric and evaluation code |
| `checkpoint/` | the trained core's own records: `complete.json`, `training_report.json`, `effective_code.json` |

## Reading the headline table

**The scores are patch-grid scores.** Ground truth is quantised to the 64x64 grid and the
prediction is compared there, never expanded to pixels, so the ceiling is 1.0 and the numbers
are not comparable with published pixel-protocol figures. `training_budget.csv` is the one file
that does use each model's own metric, which is why it is kept separate.

**The baselines see target labels and KAIROS does not.** Their cluster count and background
mapping are selected on each target dataset's own validation split. Every free choice of ours —
the Q >= 0.5 link threshold, the 8-patch component floor, the largest-component background
heuristic, the rollout length, the readout window, and f = 0.01 — is fixed on CLEVR validation
and transferred unchanged.

**One cell differs between two files by design.** AKOrN (conv) CLEVR foreground IoU reads
`.4219` in its own row and `.4267` in the *best baseline* row: the minimum-region-size rule was
offered to every baseline on the same terms, and that is the only baseline cell it moved. The
best-baseline row takes the better of the two.

**The oracle was removed from the CAE family.** The reference implementations of CAE, CAE++,
CtCAE and SynCx set the cluster count from the per-image ground-truth object count. Here it is
a per-dataset constant chosen on that dataset's validation split instead, which is stricter than
their published protocol and weaker than ours.
