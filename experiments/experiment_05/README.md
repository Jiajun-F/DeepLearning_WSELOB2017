# Experiment 5: DeepLOB with OFN inputs

Purpose: compare OFN inputs with experiment 3's price-level inputs while keeping
the model, labels, chronological splits, and training settings fixed. Train one
model per stock on Quartz. Training is approved for KGHM, PKNORLEN, PKOBP, PZU,
and PEKAO after their full-year OFN checks passed. This is one run per stock
with seed 42.

Status checked on November 21, 2026 (America/Indianapolis):

| Stock | Quartz job | Status |
| --- | --- | --- |
| KGHM | 12976541 | Training and clean test finished; results collected locally |
| PKNORLEN | 12982557 | Training and clean test finished; results collected locally |
| PKOBP | 12982569 | Training and clean test finished; results collected locally |
| PZU | 12990810 | Training and clean test finished; results collected locally |
| PEKAO | 12990738 | Training and clean test finished; results collected locally |

All stocks passed full-year local OFN checks and remote input verification.
For PKNORLEN, PKOBP, PZU, and PEKAO, all 245 feature files and 245 label files
per stock have matching local and remote SHA-256 checksums.
Each job uses `h100-single`, one H100, one CPU, 16 GB memory, and a 48-hour limit.
Keep each active console output at
`experiments/experiment_05_<STOCK>-<JOB_ID>.log`.

## Clean test results

All five jobs exited with code 0. Each stopped after 20 epochs without
higher validation accuracy and evaluated its selected checkpoint on all 62 test
days. Scores are percentages except for loss.

| Stock | Epochs trained | Best epoch | Validation accuracy | Test accuracy | Test weighted F1 | Test loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KGHM | 32 | 12 | 68.2011 | 67.5721 | 67.8740 | 0.731907 |
| PKNORLEN | 43 | 23 | 70.0662 | 66.0433 | 66.3838 | 0.770409 |
| PKOBP | 32 | 12 | 67.5453 | 66.1163 | 66.5260 | 0.759190 |
| PEKAO | 31 | 11 | 65.0681 | 64.9389 | 65.2285 | 0.786013 |
| PZU | 30 | 10 | 67.2734 | 65.9359 | 66.3374 | 0.762905 |

Each completed stock has the same eight local files as experiment 3:
`best_model.pt`, `history.csv`, `test_results.txt`, `test_metrics.npz`,
`training_curves.png`, `confusion_matrix.png`, `test_metrics.png`, and
`console.log`. All files match their remote SHA-256 checksums. The original
remote console paths are retained. These are single-seed clean-test results.
Perturbed-test comparisons are reported below.

## Data and inputs

- Read `data/wselob_2017_prepared/<STOCK>/features_ofn/`; reuse that stock's
  `labels/k20/`.
- Use January-August for training, September for validation, and October-December
  for testing. Windows contain 50 snapshots, use stride 1, and stay within a day.
- Keep k=20 and the existing label formula. Use alpha=0.0000263 for KGHM,
  0.000022 for PKNORLEN, 0.0000207 for PKOBP, 0.000029 for PZU, and 0.00005
  for PEKAO.
- Use all reconstructed orders within mid-price +/-10%, after the existing
  snapshot filtering. Each side must have at least ten distinct prices; the
  selected book is not truncated to ten levels.
- Use ten costs `((11**(i/10) - 1) / 10) * 0.035`, for i=1,...,10.
  Each cost contributes `[minimum average sell price, -sell volume/1e6,
  maximum average buy price, buy volume/1e6]`. Selling consumes buy orders;
  buying consumes sell orders. Retain fractional trade volumes.
- Normalize each of the 40 columns using unperturbed snapshots from the preceding
  five trading days, with equal snapshot weights and ddof=0. Calculate statistics
  in float64 and save features in float32. Continue history across split boundaries.
- The cost convention and corrected selling/buying equations are project choices
  agreed with the user, rather than literal use of the paper's printed equations.

Full-year preparation and alignment checks passed for all 245 output days of
all five stocks. Each stock has 162 training days, 21 validation days, and
62 test days.

| Stock | Training windows | Validation windows | Test windows |
| --- | ---: | ---: | ---: |
| KGHM | 13,651,481 | 1,309,193 | 3,765,697 |
| PKNORLEN | 12,149,701 | 1,278,165 | 4,313,703 |
| PKOBP | 11,190,788 | 1,084,053 | 3,606,606 |
| PZU | 10,485,997 | 1,060,365 | 3,909,065 |
| PEKAO | 6,203,073 | 558,443 | 1,991,235 |

## Training and clean test

Use the unchanged DeepLOB network from experiment 3: 16 convolution channels,
three 32-channel Inception branches, LSTM hidden size 64, LeakyReLU slope 0.01,
time-preserving padding, BatchNorm after all 14 convolution activations, and
dropout 0.2 after Inception. Initialize convolution weights with He normal;
retain the existing LSTM and classifier initialization.

Use Adam with learning rate 0.0001, epsilon 1e-8, betas (0.9, 0.999), and weight
decay 0. Batch size is 64, num_workers=0, and drop_last=False. Model and shuffle
seeds are 42. Train for at most 50 epochs; select strictly higher September
validation accuracy and stop after 20 epochs without improvement. Load the best
checkpoint before evaluating the complete clean test set. Report loss, accuracy,
weighted F1, and per-class scores.

From the project root on Quartz, set the Python path for your environment and
replace `YOUR_ACCOUNT` with your Slurm account:

```bash
export TRAINING_PYTHON="/path/to/deep-learning/bin/python"
sbatch --account=YOUR_ACCOUNT --partition=h100-single --gres=gpu:h100:1 \
  --job-name=experiment_05_KGHM train.sbatch \
  --stock KGHM --features ofn --horizon 20 --label-alpha 0.0000263 \
  --window-length 50 --n-levels 10 \
  --batch-norm --dropout 0.2 --initialization he_normal \
  --learning-rate 0.0001 --eps 1e-8 --batch-size 64 --seed 42 \
  --max-epochs 50 --patience 20 --run-dir experiments/experiment_05/KGHM
```

Here `--n-levels 10` specifies ten four-column input groups, corresponding to
ten OFN costs. It does not truncate the reconstructed book. The run directory is
created by training; do not create it before submission.

For another stock, change `--stock`, `--job-name`, and `--run-dir` to that
stock, and use its alpha listed above. All other settings stay the same.

Save the existing basic outputs in `<STOCK>/`: `history.csv`, `best_model.pt`,
`test_results.txt`, `test_metrics.npz`, `training_curves.png`,
`confusion_matrix.png`, and `test_metrics.png`. Keep one Slurm console output.

## Perturbed test

After clean training and testing, keep the selected weights fixed. Use experiment
4's probabilities 1%, 5%, 10%, 25%, and 50%, tick size 0.01, and unit added volume.
Fill only vacant ticks between each side's original first and tenth prices.
Process asks before bids and vacant ticks in ascending order; start one random
generator per probability with seed 42 and advance it across the same test rows.

Retain deeper orders when computing OFN from the perturbed snapshot. Preserve
the reconstruction state, original labels, and clean normalization history.
Perturb each snapshot once before forming overlapping windows. Compare clean
scores with experiment 3 and perturbed scores with experiment 4. Test scores do
not select checkpoints or tune training parameters.

Run each stock independently in its notebook:

| Stock | Notebook |
| --- | --- |
| KGHM | [experiment05_KGHM.ipynb](experiment05_KGHM.ipynb) |
| PKNORLEN | [experiment05_PKNORLEN.ipynb](experiment05_PKNORLEN.ipynb) |
| PKOBP | [experiment05_PKOBP.ipynb](experiment05_PKOBP.ipynb) |
| PEKAO | [experiment05_PEKAO.ipynb](experiment05_PEKAO.ipynb) |
| PZU | [experiment05_PZU.ipynb](experiment05_PZU.ipynb) |

Use the `deep-learning` environment from the project root or `experiment_05/`.
Each notebook needs the stock's raw HDF, processed daily files, prepared OFN
features, labels, and collected model and clean-test metrics. Run one notebook
at a time on the laptop. Splitting by stock does not reduce the total inference
work. CUDA is used when available, otherwise MPS or CPU, as in experiment 4.

Reconstruct the five preceding trading days and all 62 test days once per run.
For each test snapshot, calculate clean OFN and all five perturbed versions.
Check reconstructed timestamps and mid-prices against processed data; require
clean normalization to reproduce every saved test-day OFN array exactly. Use
unperturbed raw OFN statistics for normalization, not processed price-level
statistics. Retain one day of feature arrays and five days of small statistics
in memory; do not save perturbed feature caches.

After a full run, save the same tables and NumPy result format as experiment 4
under `<STOCK>/perturbation/`: `metrics.csv`, `class_metrics.csv`, and
`test_metrics.npz`. The NumPy stock axis has length one; the probability axis
is `[0, 0.01, 0.05, 0.10, 0.25, 0.50]`. Keep clean training/test files separate.
Compare both absolute scores and changes from each model's own clean baseline.
P=0% reuses saved CUDA scores; all five perturbation runs used MPS. This
retains experiment 4's evaluation convention. Small numerical differences
between devices are part of the reported comparisons. All five full
perturbation runs are complete.

## Perturbation results

The user confirmed completion for all five stocks. Each run covers the same
62 test days and class supports as experiment 4. All 30 overall rows and 90
per-class rows agree with their saved confusion matrices and losses. Positive
probabilities cover 87,931,530 evaluated windows across the five stocks.
Tables below use percentages; changes are percentage points (pp).

### OFN test accuracy (%)

| Stock | P=0% | P=1% | P=5% | P=10% | P=25% | P=50% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KGHM | 67.5721 | 67.5729 | 67.5738 | 67.5691 | 67.5680 | 67.5662 |
| PKNORLEN | 66.0433 | 66.0423 | 66.0446 | 66.0418 | 66.0365 | 66.0349 |
| PKOBP | 66.1163 | 66.1154 | 66.1152 | 66.1144 | 66.1148 | 66.1149 |
| PEKAO | 64.9389 | 64.9384 | 64.9349 | 64.9331 | 64.9394 | 64.9407 |
| PZU | 65.9359 | 65.9353 | 65.9365 | 65.9369 | 65.9342 | 65.9357 |

### OFN test weighted F1 (%)

| Stock | P=0% | P=1% | P=5% | P=10% | P=25% | P=50% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KGHM | 67.8740 | 67.8747 | 67.8758 | 67.8713 | 67.8710 | 67.8698 |
| PKNORLEN | 66.3838 | 66.3832 | 66.3863 | 66.3847 | 66.3820 | 66.3827 |
| PKOBP | 66.5260 | 66.5251 | 66.5249 | 66.5241 | 66.5246 | 66.5247 |
| PEKAO | 65.2285 | 65.2281 | 65.2250 | 65.2236 | 65.2307 | 65.2330 |
| PZU | 66.3374 | 66.3368 | 66.3379 | 66.3385 | 66.3357 | 66.3372 |

### Change from each model's clean baseline at P=50%

Price-level changes come from experiment 4; OFN changes come from experiment 5.
Negative values indicate lower scores than the respective P=0% baseline.

| Stock | Price-level accuracy change (pp) | OFN accuracy change (pp) | Price-level weighted F1 change (pp) | OFN weighted F1 change (pp) |
| --- | ---: | ---: | ---: | ---: |
| KGHM | -25.4274 | -0.0059 | -31.2943 | -0.0042 |
| PKNORLEN | -15.2127 | -0.0084 | -19.2005 | -0.0010 |
| PKOBP | -15.3879 | -0.0014 | -20.2748 | -0.0013 |
| PEKAO | -24.8873 | +0.0017 | -31.4022 | +0.0045 |
| PZU | -15.8143 | -0.0003 | -18.5003 | -0.0002 |

Across all 25 perturbed OFN conditions, the maximum absolute accuracy change
is 0.00844 pp and the maximum absolute weighted F1 change is 0.00497 pp.
Thus, OFN-model scores remain nearly unchanged under the tested unit-volume
insertions between the original first and tenth levels. At P=50%, the
price-level models lose 15.21-25.43 pp of accuracy and 18.50-31.40 pp of
weighted F1.

This pattern is consistent with small-volume insertions changing VWAP-based
maximum trade volumes only slightly, while disrupting fixed price-level
positions. The experiments compare the two complete input schemes: OFN uses
deeper orders as well as a different feature mapping. Each stock uses one
trained model and one perturbation seed. The result concerns this insertion
rule; it does not establish robustness to large-volume additions, deletions,
or changes in best quotes.

Keep the original executed notebooks and per-stock `perturbation/` outputs.
No model training or evaluation was rerun to produce this summary.
