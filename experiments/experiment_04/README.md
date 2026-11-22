# Experiment 4: Price-level inputs under test-time perturbation

Purpose: measure how small orders at previously empty prices affect the five
DeepLOB models from [experiment 3](../experiment_03/README.md). Keep the selected
weights fixed and evaluate all 62 October–December 2017 test days for each stock.
All five stocks and all five perturbation probabilities have been evaluated.

## Method

- Use the original labels, k=20, 50-snapshot windows, stride 1, and batch size 64.
  Data preparation and thresholds are described in the
  [project README](../../README.md#data-and-labels).
- For each snapshot and side, find vacant ticks strictly between the original
  first and tenth prices. Independently add volume 1 at each vacant tick with
  probability P = 1%, 5%, 10%, 25%, or 50%. Tick size is 0.01.
- Re-sort prices and keep ten levels per side. The best quotes and original
  mid-price do not change. Perturb each snapshot once before forming windows.
- Use `OrderBook.add_perturbation` in [wse_lob.py](../../wse_lob.py). Process asks
  before bids and vacant ticks in ascending order. Start a separate NumPy
  generator with seed 42 for each stock and P, and advance it across test days.
  The same draws are used across probabilities.
- Normalize with the clean preceding five trading days. Check that clean
  normalization reproduces every saved test-day feature array exactly.
- Disable gradients and dropout. Aggregate confusion counts and sample-weighted
  loss across days; check that class supports match the clean baseline.

P=0% reuses experiment 3's saved CUDA test scores. The perturbed evaluations
used MPS. Thus, changes from P=0% include possible numerical differences between
devices. This experiment uses one trained model and one perturbation seed per
stock, without retraining or selecting weights using perturbed scores.

## Results

Scores below are percentages. Weighted F1 weights each class by its true support.
The values come from [metrics.csv](metrics.csv).

### Test accuracy (%)

| Stock | P=0% | P=1% | P=5% | P=10% | P=25% | P=50% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KGHM | 66.8938 | 50.9545 | 41.1994 | 39.5586 | 38.9316 | 41.4664 |
| PEKAO | 64.7312 | 48.7033 | 40.8524 | 39.6620 | 38.9420 | 39.8439 |
| PKNORLEN | 66.0485 | 57.6086 | 51.3202 | 50.1929 | 50.2014 | 50.8358 |
| PKOBP | 66.1876 | 64.7881 | 59.7424 | 55.9550 | 51.6300 | 50.7997 |
| PZU | 64.1748 | 62.6776 | 57.7966 | 54.0758 | 49.4259 | 48.3605 |

### Test weighted F1 (%)

| Stock | P=0% | P=1% | P=5% | P=10% | P=25% | P=50% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KGHM | 67.2066 | 48.8717 | 36.0890 | 33.7423 | 31.5336 | 35.9123 |
| PEKAO | 65.0079 | 43.2924 | 33.6951 | 32.5941 | 32.0495 | 33.6057 |
| PKNORLEN | 66.3843 | 57.2496 | 50.7141 | 49.6875 | 49.0269 | 47.1838 |
| PKOBP | 66.6272 | 65.1317 | 59.0182 | 53.6375 | 47.4651 | 46.3524 |
| PZU | 64.5666 | 62.9710 | 57.5985 | 53.2266 | 47.4018 | 46.0663 |

At P=50%, accuracy falls by 15.21–25.43 percentage points and weighted F1 by
18.50–31.40 points. The decline is not monotonic for every stock. Even though
the inserted orders have volume 1, they change which prices occupy the ten
input levels. These results measure sensitivity to this particular insertion
rule. [Experiment 5](../experiment_05/README.md#perturbation-results) compares
the same perturbations using OFN inputs.

## Files and use

| File | Contents |
| --- | --- |
| [experiment_04.ipynb](experiment_04.ipynb) | Executed preparation and evaluation workflow for all five stocks |
| [metrics.csv](metrics.csv) | 30 rows: five stocks × six probabilities, including clean scores and changes |
| [class_metrics.csv](class_metrics.csv) | 90 rows of class support, precision, recall, and F1 |
| [test_metrics.npz](test_metrics.npz) | Confusion matrices, losses, dates, selected epochs, and evaluation settings |

The NPZ stock order is KGHM, PEKAO, PKNORLEN, PKOBP, PZU. Probabilities are
`[0, 0.01, 0.05, 0.10, 0.25, 0.50]`; confusion rows are true classes and columns
are predictions, ordered down, stationary, up.

To inspect results, open the tables or saved notebook outputs. To rerun, use the
installed project environment and open the notebook from the project root or
this directory. It needs experiment 3's checkpoints and clean metrics, processed
daily snapshots including the five preceding history days, prepared price-level
features, and labels. CUDA is selected when available, otherwise MPS or CPU.
Running the evaluation cells processes the full test period; the final save
cell replaces this experiment's three result files.

