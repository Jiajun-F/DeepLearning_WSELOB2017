# Experiment 3: DeepLOB with BatchNorm and dropout

Purpose: evaluate the same DeepLOB configuration on five stocks after the
deterioration in experiment 2. Train one model from fresh weights for each stock
and select its checkpoint using September validation accuracy.

## Shared configuration

- Data: WSELOB-2017; January-August training, September validation, and
  October-December testing. Keep each stock's data separate.
- Inputs: 50 snapshots, ten levels per side, stride 1, and k=20. Use the
  prepared data and preprocessing rules in the [project README](../../README.md#data-and-labels).
  Each stock's label threshold is listed below.
- Model: 16 convolution channels, 32 channels per Inception branch, LSTM hidden
  size 64, LeakyReLU slope 0.01, and padding that preserves all 50 time positions.
- BatchNorm: after each of the 14 convolution activations.
- Dropout: 0.2 after Inception and before the LSTM, sharing the channel mask
  across time within each sample; disabled during validation and testing.
- Initialization: He normal for convolutions; Xavier uniform for LSTM input
  weights and the classifier; orthogonal initialization applied once to the
  concatenated LSTM recurrent weights; total forget-gate bias 1 and other biases 0.
- Adam: learning rate 0.0001, epsilon 1e-8, betas (0.9, 0.999), weight decay 0.
- Training: batch size 64, model and shuffle seeds 42, num_workers=0,
  drop_last=False; shuffle training only.
- Limit: 50 epochs; save strictly higher validation accuracy; stop after 20
  consecutive epochs without improvement.
- Metrics: loss, accuracy, and F1 weighted by true class support, following OFN.
  Experiment 2 used macro-F1, so its F1 values use a different averaging rule.
  Evaluate October-December once using the best validation checkpoint.

All runs used Python 3.11.18 and PyTorch 2.11.0+cu126, with one GPU, one CPU core,
and 16 GB host memory. Cluster jobs used QOS `allocated`.

| Stock and results | Alpha | Platform | GPU | Job ID |
| --- | ---: | --- | --- | --- |
| [KGHM](KGHM/) | 0.0000263 | Big Red 200 | A100 40 GB | 8889315 |
| [PEKAO](PEKAO/) | 0.00005 | Quartz | H100 80 GB | 12754561 |
| [PKNORLEN](PKNORLEN/) | 0.000022 | Quartz | H100 80 GB | 12754562 |
| [PKOBP](PKOBP/) | 0.0000207 | Quartz | H100 80 GB | 12754563 |
| [PZU](PZU/) | 0.000029 | Quartz | H100 80 GB | 12754564 |

## Results

All five runs finished training and testing. Each stopped after 20 consecutive
epochs without a higher validation accuracy. The validation scores below refer
to the selected checkpoint; that checkpoint produced the October-December test
scores. Scores are percentages except for test loss.

| Stock | Epochs trained | Best epoch | Validation accuracy | Test accuracy | Test weighted F1 | Test loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KGHM | 48 | 28 | 67.9391 | 66.8938 | 67.2066 | 0.742697 |
| PEKAO | 27 | 7 | 64.5847 | 64.7312 | 65.0079 | 0.787261 |
| PKNORLEN | 43 | 23 | 69.9347 | 66.0485 | 66.3843 | 0.766637 |
| PKOBP | 44 | 24 | 67.1821 | 66.1876 | 66.6272 | 0.757033 |
| PZU | 26 | 6 | 65.9027 | 64.1748 | 64.5666 | 0.790356 |

## Reference: DeepLOB in the OFN paper

The reference is the traditional level-based input X^(1), k=20, with no test
perturbation (P=0%). Each stock has its own trained model. Values come from the
[OFN paper](https://doi.org/10.1016/j.asoc.2024.111555), Appendix A, Tables A.7-A.11 (pages 13-14).
Scores are percentages; F1 is weighted by class support. Differences are
experiment 3 minus the published values, in percentage points (pp).

| Stock | Paper accuracy | Experiment 3 accuracy | Difference (pp) | Paper F1 | Experiment 3 F1 | Difference (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KGHM | 65.97 | 66.89 | +0.92 | 66.30 | 67.21 | +0.91 |
| PEKAO | 64.00 | 64.73 | +0.73 | 64.33 | 65.01 | +0.68 |
| PKNORLEN | 64.09 | 66.05 | +1.96 | 64.60 | 66.38 | +1.78 |
| PKOBP | 64.75 | 66.19 | +1.44 | 64.96 | 66.63 | +1.67 |
| PZU | 63.71 | 64.17 | +0.46 | 64.05 | 64.57 | +0.52 |

The paper's Table 3 separately reports a five-stock summary of 64.53% accuracy
and 64.90% weighted F1. These are the published summary values, not an arithmetic
mean recomputed from the rounded Appendix A rows above.

Experiment 3 has higher reported accuracy and weighted F1 for all five stocks.
This is a reference comparison, not evidence of a model improvement under
identical conditions:

- Labels: our future mean excludes the current price; the paper's printed
  Eq. (12) includes it. We average the k+1 past/current observations with a
  denominator of k+1. The labeling conventions are therefore not identical.
- Splits: we train on January-August and validate on complete September days;
  the paper reserves the last 10% of January-September records for validation.
- Repeats: this experiment uses one run per stock with seed 42; Section 5.4 of
  the paper reports five independent training runs per configuration.
- Implementation: our PyTorch configuration is specified above; the paper uses
  Keras/TensorFlow. Exact agreement in preprocessing and training implementation
  has not been established.

A controlled comparison should hold data, labels, splits, and training settings
fixed, and use multiple random seeds before attributing differences to a model
or input representation.

## Files

Each stock has the same directory structure:

```text
experiment_03/
├── README.md
├── KGHM/
├── PEKAO/
├── PKNORLEN/
├── PKOBP/
└── PZU/
```

Every stock directory contains these eight files:

| File | Contents |
| --- | --- |
| `history.csv` | Training and validation metrics and elapsed time for each completed epoch |
| `best_model.pt` | Best validation checkpoint and its configuration |
| `test_results.txt` | Test scores, class scores, majority baseline, and confusion matrix |
| `test_metrics.npz` | Numeric test metrics and confusion counts |
| `training_curves.png` | Training and validation loss, accuracy, and weighted F1 |
| `confusion_matrix.png` | Test confusion counts and row percentages |
| `test_metrics.png` | Model, baseline, and per-class test comparisons |
| `console.log` | The run's single training console output |

Open `<STOCK>/test_results.txt` for a full test report. The notebook selects
`experiments/<EXPERIMENT>/<STOCK>/` through `EXPERIMENT` and `STOCK`.

The local results and completed outputs on Slate use this same layout. The
remote experiment directory is `/N/slate/$USER/Project/experiments/experiment_03/`.
