# Deep Learning on WSELOB-2017

This is an project on
predicting short-term price movements from limit order books. It uses 2017 data
for five Warsaw Stock Exchange stocks: KGHM, PEKAO, PKNORLEN, PKOBP, and PZU.

This project reconstructs limit order books from raw messages and implements DeepLOB in PyTorch. It compares the model’s robustness using two input representations—prices and volumes at the first ten levels, and ordered fuzzy numbers (OFN)—under different levels of input perturbation.

[Installation and use](#installation-and-use) ·
[Project structure](#project-structure) ·
[Data and labels](#data-and-labels) ·
[Models and inputs](#models-and-inputs) ·
[Experiments and results](#experiments-and-results) ·
[Next steps](#next-steps) · [References](#references)

## Installation and use

Use Python 3.11. From the project root:

```bash
conda create -n deep-learning python=3.11
conda activate deep-learning
python -m pip install -r requirements.txt
```

If the environment already exists, activate it and install the requirements.
A Python virtual environment also works. Package versions are listed in
[requirements.txt](requirements.txt). Run the commands below from the project root.

### 1. Download and prepare data

Download the order-message HDF5 files from
[WSELOB-2017, version 1](https://data.mendeley.com/datasets/3g4mhdp899/1)
and place them in `data/wselob_2017_source/orders/`.
KGHM uses `KGHM_lob_2017_zlib.h5`. Trade files are not needed.

For a fresh copy with no processed outputs, prepare KGHM's full year with:

```bash
python prepare_data.py all \
  --stock KGHM \
  --source-file data/wselob_2017_source/orders/KGHM_lob_2017_zlib.h5 \
  --processed-dir data/wselob_2017_processed/KGHM \
  --output-dir data/wselob_2017_prepared/KGHM \
  --history-days 5 --k 20 --alpha 0.0000263
```

For another stock, change the stock name in every path and use its threshold
from the [label table](#labels). 

The five preparation stages are:

| Stage | Input → output |
| --- | --- |
| `extract` | Raw messages → daily snapshots |
| `prepare` | Daily snapshots → normalized features and labels |
| `labels` | Existing snapshots → labels for another horizon |
| `all` | Run extraction and preparation together |
| `ofn` | Raw messages → normalized OFN features, aligned with existing snapshots |

Existing output files cause an error. When extending a small run, select only
new output dates and use the appropriate separate stage. Do not repeat
`all` over existing files. See `python prepare_data.py --help` for examples.

After price-level preparation, generate OFN inputs with:

```bash
python prepare_data.py ofn \
  --stock KGHM \
  --source-file data/wselob_2017_source/orders/KGHM_lob_2017_zlib.h5 \
  --processed-dir data/wselob_2017_processed/KGHM \
  --output-dir data/wselob_2017_prepared/KGHM \
  --history-days 5
```

This reads raw messages again because OFN needs deeper orders. It checks
snapshot times and mid-prices against the processed files, writes
`features_ofn/`, and reuses the existing labels.

### 2. Train and evaluate

This command uses experiment 3's settings for KGHM. Replace `NEW_RUN` with
a new directory name; training creates it and refuses to overwrite a run.

```bash
python train_deeplob.py \
  --stock KGHM --features nlevels --horizon 20 --label-alpha 0.0000263 \
  --window-length 50 --n-levels 10 \
  --batch-norm --dropout 0.2 --initialization he_normal \
  --learning-rate 0.0001 --eps 1e-8 --batch-size 64 --seed 42 \
  --max-epochs 50 --patience 20 --device cuda \
  --run-dir experiments/NEW_RUN/KGHM
```

Use `--features ofn` and a different run directory for OFN. Here
`--n-levels 10` means ten groups of four features: price levels or OFN cost
limits. `--label-alpha` describes labels already on disk; it does not generate
them. Use the matching stock threshold.

Training selects the best validation checkpoint, evaluates it on the test set,
and saves metrics and figures. The reported runs used CUDA GPUs.
`--device cpu` is supported but much slower. For Slurm, adapt the account,
partition, and Python path in [train.sbatch](train.sbatch).

For interactive use, run `python -m jupyterlab` and open
[deeplob.ipynb](deeplob.ipynb). Select the installed environment and review the
configuration. Choose a new run directory before enabling `RUN_TRAINING` or
`RUN_TESTING`; both are disabled by default. Perturbation notebooks are linked
from the experiment READMEs. Their execution cells run full test evaluations;
open saved tables directly if you only want to inspect results.

## Project structure

| File | Role |
| --- | --- |
| [wse_lob.py](wse_lob.py) | Reconstruct orders, sample snapshots, compute both representations, normalize, label, and add perturbations |
| [prepare_data.py](prepare_data.py) | Run preparation and save daily NumPy arrays |
| [deeplob.py](deeplob.py) | Define DeepLOB, BatchNorm, dropout, and initialization |
| [train_deeplob.py](train_deeplob.py) | Load daily windows, train, select a checkpoint, evaluate, and plot results |
| [deeplob.ipynb](deeplob.ipynb) | Configure and run the shared training functions interactively |
| [train.sbatch](train.sbatch) | Submit one CUDA training run to Slurm |
| `experiments/experiment_01/` through `experiment_05/` | Experiment READMEs, results, and evaluation notebooks |
| `licenses/` | Third-party license notices |

Data is stored separately:

```text
data/
  wselob_2017_source/orders/<STOCK>_lob_2017_zlib.h5
  wselob_2017_processed/<STOCK>/YYYYMMDD.npz
  wselob_2017_prepared/<STOCK>/
    features/YYYYMMDD.npz
    features_ofn/YYYYMMDD.npz
    labels/k20/YYYYMMDD.npz
```

Processed files contain raw float64 `features` of shape `(N, 40)`,
`mid_price`, and UTC `time_ns`. They also contain `skipped_time_ns`,
`skipped_message_position`, and `skipped_reason` for rejected states
(1: crossed book; 2: insufficient depth). Prepared feature files contain only
float32 `features`; label files contain only int64 `labels`.
Dates refer to the Warsaw trading day. Raw data is preserved. Overlapping
windows are sliced when requested, rather than stored as separate copies.

## Data and labels

### Reconstructing and sampling the book

Raw records are order messages, not snapshots. Timestamps are converted to
`Europe/Warsaw`, and raw prices are divided by `10 ** price_level`.
Each day starts with an empty book. Messages are replayed in file order:
`F` clears the book, `Y` retransmits an order, and `A/M/D` add, modify,
or delete an order. A modification for a missing order is recorded and added;
a deletion for a missing order is recorded and skipped.

After applying all messages with the same timestamp, the code:

1. Considers snapshots from 09:00 inclusive to 16:50 exclusive. Messages outside
   these hours still update the book.
2. Finds the best positive bid and ask and their midpoint. Zero-price orders
   stay in the full book but do not define these quotes. Missing positive
   quotes raise an error. Crossed orders (bid price is larger than ask price) are skipped; equal bid and ask are allowed.
3. Selects orders within 10% of the midpoint and requires at least ten distinct
   prices on each side.
4. Saves the first eligible snapshot, then saves another only when the selected
   orders change. This compares order identities and all stored fields,
   not just the first ten prices.

Price-level features sum displayed volume at each price across order types.
A row contains `[ask price, ask volume, bid price, bid volume]` for levels
1–10, with asks ascending and bids descending. A horizon counts saved
snapshots, so it is neither a fixed number of seconds nor a raw-message count.

### Normalization and time splits

Each column is standardized using all snapshots from the preceding five trading
days, equally weighted, with population variance (`ddof=0`). Statistics use
float64 and saved features use float32. Each representation has its own
statistics. The current day and future days never enter those statistics.

Each stock has 250 extracted days and 245 prepared days after the initial five
history days. Training uses January–August (162 prepared days), validation uses
September (21), and testing uses October–December (62). Normalization history
continues across splits. Input windows contain 50 snapshots with stride 1;
windows and label intervals stay within a single day and split. Each stock
is trained separately.

### Labels

Let `p[t]` be the original mid-price at the window's final snapshot:

```text
past_mean   = mean(p[t-k], ..., p[t])       # k+1 prices
future_mean = mean(p[t+1], ..., p[t+k])     # k prices
change      = (future_mean - past_mean) / past_mean
```

Class 0 is **down** when `change < -alpha`; class 2 is **up** when
`change > alpha`; class 1 is **stationary** otherwise, including both thresholds.
Incomplete positions are stored as `-1` and excluded. For a day of `N` rows,
valid window endpoints are `max(49, k)` through `N-k-1` (zero-based).
Labels always use original prices.

Thresholds come from Table 2 of the [OFN paper](https://doi.org/10.1016/j.asoc.2024.111555):

| Stock | Alpha for k=20 |
| --- | ---: |
| KGHM | 0.0000263 |
| PEKAO | 0.00005 |
| PKNORLEN | 0.000022 |
| PKOBP | 0.0000207 |
| PZU | 0.000029 |

The past mean uses `k+1` in the denominator, correcting the printed DeepLOB
equation's count. The future mean excludes the current price, which also
differs from the OFN paper's printed label interval.

## Models and inputs

### DeepLOB

Both representations produce a `50 × 40` input. DeepLOB uses three convolution
blocks with 16 channels, three Inception branches with 32 channels each, an LSTM
with 64 hidden units, and three output logits. Padding preserves all 50 time
positions. Training uses cross-entropy.

Experiments 3 and 5 share these settings:

| Setting | Value |
| --- | --- |
| Activation and BatchNorm | LeakyReLU, slope 0.01; BatchNorm after all 14 convolution activations |
| Dropout | 0.2 after Inception; a channel mask shared across time within each sample |
| Initialization | He normal for convolutions; Xavier uniform for LSTM input weights and classifier; orthogonal stacked recurrent weights; total forget-gate bias 1 |
| Optimizer | Adam; learning rate 0.0001; epsilon 1e-8; betas (0.9, 0.999); no weight decay |
| Batches and seeds | Batch size 64; model and shuffle seed 42; shuffle training only; keep partial batches |
| Selection | At most 50 epochs; select higher validation accuracy; stop after 20 epochs without improvement |
| Evaluation | Best checkpoint; accuracy, F1 weighted by class support, per-class scores, and a training-majority baseline |

This follows the DeepLOB architecture with the stated project settings.
Published scores are reference values rather than an exact reproduction:
the OFN paper uses a different validation split, Keras/TensorFlow, and five
training runs per configuration. This project uses one run per stock.

### OFN implementation

OFN represents the maximum buy and sell volumes for a given cost limit,
measured relative to the mid-price. [orderbook_to_ofn](wse_lob.py) uses all
price levels within 10% of the mid-price and sums order volumes at each price.

At time $t$, $p_a^i(t)$ and $v_a^i(t)$ are the ask price and volume at level
$i$; $p_b^i(t)$ and $v_b^i(t)$ are the corresponding bid price and volume. Assume there are $L_a$ ask levels and $L_b$ bid levels. The
mid-price is

$$
p_r(t)=\frac{p_a^1(t)+p_b^1(t)}{2}.
$$

The volume-weighted average prices over the first $i$ levels are

$$
\mu_a^i(t)=\frac{\sum_{j=1}^{i}p_a^j(t)v_a^j(t)}{\sum_{j=1}^{i}v_a^j(t)},
\qquad
\mu_b^i(t)=\frac{\sum_{j=1}^{i}p_b^j(t)v_b^j(t)}{\sum_{j=1}^{i}v_b^j(t)}.
$$

For cost limit $x$ (percentage deviation of the actural execution price from the mid-price), the average sell price must be at least $(1-x)p_r(t)$,
and the average buy price must be at most $(1+x)p_r(t)$. For example,
$x=0.01$ allows a 1% difference from the mid-price. Sell volume is calculated
from the bid side, and buy volume from the ask side.

Let $l_b$ and $l_a$ be the largest bid and ask level indices satisfying

$$
\mu_b^{l_b}(t)\geq(1-x)p_r(t),
\qquad
\mu_a^{l_a}(t)\leq(1+x)p_r(t).
$$

Set the corresponding index to zero if no level satisfies the condition.
The maximum sell volume is recorded with a negative sign:

$$
f_t(x)=-\begin{cases}
0, & l_b=0, \\
\displaystyle\sum_{i=1}^{L_b}v_b^i(t), & l_b=L_b, \\
\displaystyle\left(\sum_{i=1}^{l_b}v_b^i(t)\right)
\left(1+\frac{\mu_b^{l_b}(t)-(1-x)p_r(t)}
{(1-x)p_r(t)-p_b^{l_b+1}(t)}\right), & 0<l_b<L_b.
\end{cases}
$$

The maximum buy volume is

$$
g_t(x)=\begin{cases}
0, & l_a=0, \\
\displaystyle\sum_{i=1}^{L_a}v_a^i(t), & l_a=L_a, \\
\displaystyle\left(\sum_{i=1}^{l_a}v_a^i(t)\right)
\left(1+\frac{\mu_a^{l_a}(t)-(1+x)p_r(t)}
{(1+x)p_r(t)-p_a^{l_a+1}(t)}\right), & 0<l_a<L_a.
\end{cases}
$$

The three cases correspond to zero volume, all available volume, and a
partial trade at the next level. In the third case, the total includes all
volume through level $l_b$ or $l_a$, plus enough of the next level to make the
average trade price equal to the price limit. Fractional volumes are retained.

The code uses ten cost limits $r_i r_{\max}$, where

$$
r_i=\frac{11^{i/10}-1}{10},\qquad i=1,\ldots,10,
\qquad r_{\max}=0.035.
$$

For each cost limit, the four features are the minimum average sell price,
negative sell volume, maximum average buy price, and buy volume. Volumes are
expressed in millions of shares. The ten groups form a 40-dimensional vector:

$$
x_t^{(2)}=\left[
(1-r_i r_{\max})p_r(t),\;
\frac{f_t(r_i r_{\max})}{10^6},\;
(1+r_i r_{\max})p_r(t),\;
\frac{g_t(r_i r_{\max})}{10^6}
\right]_{i=1}^{10}\in\mathbb{R}^{40}.
$$

Each feature is standardized using the preceding five trading days. OFN uses
the same snapshots and labels as the price-level input, including snapshots
with repeated OFN values.

### Test-time perturbations

Experiments 4 and 5 keep weights and labels fixed. For each snapshot and side,
they find vacant ticks strictly between the original first and tenth levels.
Each vacant tick independently receives volume 1 with probability **1%, 5%,
10%, 25%, or 50%**. Tick size is 0.01. Best quotes and mid-price remain unchanged.

`OrderBook.add_perturbation` inserts orders into price-level arrays and keeps
the new top ten. `perturb_orderbook` inserts them into a copied book for OFN,
retaining deeper orders. Both process asks before bids and vacant ticks in
ascending order. A seed-42 generator for each stock and probability advances
across test days, giving matching draws across the two experiments. Each
snapshot is perturbed once before windows are formed. Normalization uses the
clean preceding five days.

## Experiments and results

Experiments 3–5 contain the main comparisons:

| Experiment | Description and result index |
| --- | --- |
| [3: Price-level DeepLOB](experiments/experiment_03/README.md) | Five-stock training, clean test scores, selected epochs, and comparison with the OFN paper |
| [4: Price-level perturbations](experiments/experiment_04/README.md) | Fixed experiment 3 models evaluated at five insertion probabilities |
| [5: OFN DeepLOB](experiments/experiment_05/README.md) | OFN training, clean scores, perturbation notebooks, and comparison with experiment 4 |

[Experiment 1](experiments/experiment_01/README.md) contains early training
diagnostics. [Experiment 2](experiments/experiment_02/README.md) retains a
stopped baseline run. They provide background; experiment 2 uses macro-F1,
which should not be compared directly with the weighted F1 below.

### Clean test results, k=20

All scores are percentages on October–December 2017, from the saved test
metrics in experiments 3 and 5.

| Stock | Price-level accuracy | OFN accuracy | Price-level weighted F1 | OFN weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| KGHM | 66.8938 | 67.5721 | 67.2066 | 67.8740 |
| PEKAO | 64.7312 | 64.9389 | 65.0079 | 65.2285 |
| PKNORLEN | 66.0485 | 66.0433 | 66.3843 | 66.3838 |
| PKOBP | 66.1876 | 66.1163 | 66.6272 | 66.5260 |
| PZU | 64.1748 | 65.9359 | 64.5666 | 66.3374 |

Clean scores are close for several stocks; OFN does not improve every stock.
The largest clean accuracy gain is for PZU, about 1.76 percentage points.

### Perturbation results

At a 50% insertion probability, accuracy changes from each model's own clean
baseline are:

| Stock | Price-level change (pp) | OFN change (pp) |
| --- | ---: | ---: |
| KGHM | -25.4274 | -0.0059 |
| PEKAO | -24.8873 | +0.0017 |
| PKNORLEN | -15.2127 | -0.0084 |
| PKOBP | -15.3879 | -0.0014 |
| PZU | -15.8143 | -0.0003 |

Across all five probabilities and stocks, OFN's absolute accuracy change stays
below 0.0085 percentage points. Price-level inputs are much more affected by
this insertion rule. Full probability and per-class tables are in the
experiment directories.

These results cover unit-volume insertions inside the original first ten levels,
with one training seed and one perturbation seed. OFN also uses deeper orders,
so both the representation and available depth differ. Clean evaluation used
CUDA and perturbed evaluation used MPS; small device differences are included.
These results do not establish robustness to other order changes or trading
profitability.

For experiments 3 and 5, each stock directory contains:

- `history.csv` and `best_model.pt` with the selected model's configuration.
- `test_results.txt` and `test_metrics.npz` for the test scores.
- `training_curves.png`, `confusion_matrix.png`, and `test_metrics.png`.

Perturbation outputs are `metrics.csv`, `class_metrics.csv`, and `test_metrics.npz`,
at the experiment 4 root and in each experiment 5 stock's `perturbation/`
directory. Local training runs also retain one `console.log`; these logs are
excluded by `.gitignore`.

## Next steps

- **Longer horizons:** compare `k=50` and `k=100` with the current `k=20`
  results. Reuse features and create separate `labels/k50/` and `labels/k100/`
  arrays using the `labels` stage. Thresholds remain to be chosen for each
  new horizon; the current thresholds apply to k=20.
- **TLOB:** implement it and compare with DeepLOB on the same data, labels, and
  time splits. TLOB is not implemented here yet. Its paper and official code
  are linked below.

## References

- Zhang, Z., Zohren, S., and Roberts, S. (2019).
  [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://doi.org/10.1109/TSP.2019.2907260).
  *IEEE Transactions on Signal Processing*, 67(11).
  [Preprint](https://arxiv.org/abs/1808.03668) ·
  [Authors' repository](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books).
  Architecture and implementation reference; reviewed at commit
  `ff14d7c2fd38bdfc143389786993d0f0236d4eb8`.
- Marszałek, A., and Burczyński, T. (2024).
  [Modeling of limit order book data with ordered fuzzy numbers](https://doi.org/10.1016/j.asoc.2024.111555).
  *Applied Soft Computing*, 158, 111555. Source for the OFN representation,
  thresholds, and perturbation comparison.
- Marszałek, A. (2023).
  [WSELOB-2017: The year-long database of limit order books for the five biggest companies listed on the Warsaw Stock Exchange](https://doi.org/10.17632/3g4mhdp899.1).
  Mendeley Data, version 1, CC BY 4.0. The dataset's `orderbook2.py` example
  informed the vacant-tick insertion procedure.
- Marszałek, A. [pyofn](https://github.com/amarszalek/pyofn).
  `OrderBook` was adapted from `MinOrderBook` in
  [utils.py at cacb97b](https://github.com/amarszalek/pyofn/blob/cacb97bc163ccca2a76659f9b28728beb54262a5/pyofn/utils.py).
  The source notice and changes are recorded in `wse_lob.py`; the Apache-2.0
  license is retained in [licenses/pyofn_LICENSE](licenses/pyofn_LICENSE).
- Berti, L., and Kasneci, G. (2025).
  [TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data](https://arxiv.org/abs/2502.15757).
  [Official repository](https://github.com/LeonardoBerti00/TLOB).
  Reference for the planned comparison; no TLOB results are reported here.
