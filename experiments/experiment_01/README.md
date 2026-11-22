# Experiment 1: Diagnose DeepLOB training

**Purpose:** Investigate poor learning and nearly constant predictions, then select a configuration for a full-data comparison. This is a diagnostic study, not a final project result.

**Data:** KGHM, WSELOB-2017. Use January-August training windows and September validation windows. October-December test data was never used.

**Sequence:** First compare learning rates. Then compare BatchNorm, dropout, initialization, epsilon, and batch size using a larger training budget. The two stages answer the same training-stability question; their different budgets mean their scores should not be directly ranked against each other.

**Outcome:** Choose learning rate 0.001 and test BatchNorm with no dropout and PyTorch initialization on the full training split. These subset comparisons do not establish a universal optimum or guarantee stable full-data training.

## Step 1: Learning rate

### Protocol

- Use 32,768 windows sampled without replacement from the January-August training split and 8,192 windows from the September validation split. Reuse `DailyWindowDataset` from `deeplob.ipynb`; retain within-day window and label boundaries.
- Use subset seed 42, model seed 42, and shuffle seed 42. Reset model weights and the sampler for each trial. The two initial train losses and validation losses match exactly.
- Train for ten subset epochs (10,240 optimizer steps) per learning rate. Keep the existing architecture, batch size 32, Adam epsilon 1e-8, betas (0.9, 0.999), and weight decay 0. Use the Apple GPU through MPS.
- Evaluate the same fixed subsets before training and after each epoch. Inspect gradients on a fixed training batch without taking an optimizer step. Never load the test split.
- Training label counts: 11,115 / 10,556 / 11,097. Validation label counts: 2,925 / 2,457 / 2,810. The down-class validation baseline is 35.7056%.

### Results

Both trials start with training loss 1.099773 and validation loss 1.095969. Losses below are evaluated after epoch 10; best accuracy is selected across epochs 1-10.

| Learning rate | Final train loss | Final validation loss | Final validation accuracy | Final validation macro-F1 | Best validation accuracy | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 1.059999 | 1.063771 | 41.6626% | 41.6255% | 42.3462% | 8 |
| 0.0001 | 1.050298 | 1.056121 | 40.9668% | 37.9957% | 40.9668% | 10 |

At epoch 10, validation prediction counts (down / stationary / up) are 2,920 / 2,962 / 2,310 for 0.001 and 1,061 / 4,969 / 2,162 for 0.0001. Mean per-class probability standard deviation across validation inputs rises from 0.000001836 to 0.068033 and 0.095994, respectively. Feature-extractor gradient norms on the fixed training probe are 0.334082 and 1.420075; LSTM input RMS values are 6.728228 and 0.163449.

Both trials show learning and input-dependent predictions. Select 0.001 for the next full-year run using validation accuracy, the existing primary selection metric. The 0.0001 trial finishes with lower loss; this short comparison does not establish that 0.001 will remain better over a long run. These subset results are not full-validation or test results.


## Step 2: Broader factor comparisons

### Selected configuration

Selected by mean complete-September validation accuracy across seeds 42, 43, and 44: **batch_norm**.

- Initialization: pytorch; BatchNorm: True; dropout: 0.
- Adam learning rate: 0.001; epsilon: 1e-08; batch size: 32.
- The model retains the paper's channel counts, activations, temporal padding, and LSTM size.
- These results identify the best configuration within the tested search and training budget; they do not establish a universal optimum or a final test score.

### Confirmation results

| Configuration | Accuracy mean ± SD | Macro-F1 mean ± SD | Mean loss |
| --- | ---: | ---: | ---: |
| batch_norm | 64.91% ± 0.10% | 65.08% ± 0.11% | 0.78247 |
| batch_norm_dropout | 64.79% ± 0.14% | 64.94% ± 0.14% | 0.78290 |
| baseline | 63.94% ± 0.30% | 64.13% ± 0.32% | 0.79922 |

#### Interpretation

- Adding BatchNorm changes mean complete-September accuracy by +0.965 percentage points relative to the baseline. Adding dropout to BatchNorm changes it by -0.117 percentage points.
- In single-seed screening, BatchNorm and the custom initialization presets learned faster than the baseline. Xavier and Kaiming did not improve the best accuracy achieved within twenty epochs.
- Larger batches reduced local run time but did not improve screening accuracy under equal sample exposure. They perform fewer optimizer updates per epoch.
- Increasing Adam epsilon without BatchNorm produced nearly constant predictions within this budget. The paper-Adam setting with BatchNorm learned, showing that these factors interact.
- This staged search does not test every interaction. Initialization comparisons change convolution, LSTM, and output-layer initialization together; they do not isolate the LSTM initializer.
- Confirmation trains on 65,536 selected windows, not the complete training split. Its ranking needs confirmation in subsequent full-data training.

#### Individual confirmation runs

All scores below use the complete September validation set. Predicted counts are ordered down / stationary / up. Probability SD is the mean of the three per-class probability standard deviations across validation inputs.

| Configuration | Seed | Selected epoch | Accuracy | Macro-F1 | Loss | Probability SD | Predicted counts | Subset run minutes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| baseline | 42 | 17 | 63.63% | 63.79% | 0.80829 | 0.2753 | 357116 / 566163 / 385914 | 8.97 |
| baseline | 43 | 20 | 64.23% | 64.42% | 0.79182 | 0.2623 | 349904 / 581492 / 377797 | 9.13 |
| baseline | 44 | 16 | 63.96% | 64.17% | 0.79754 | 0.2585 | 363868 / 605063 / 340262 | 8.26 |
| batch_norm | 42 | 16 | 64.92% | 65.10% | 0.78001 | 0.2838 | 362474 / 574786 / 371933 | 10.48 |
| batch_norm | 43 | 17 | 64.80% | 64.96% | 0.78471 | 0.2886 | 384486 / 582082 / 342625 | 10.44 |
| batch_norm | 44 | 14 | 65.00% | 65.18% | 0.78268 | 0.2893 | 378063 / 567394 / 363736 | 10.50 |
| batch_norm_dropout | 42 | 20 | 64.92% | 65.04% | 0.78185 | 0.2813 | 395837 / 539569 / 373787 | 10.49 |
| batch_norm_dropout | 43 | 13 | 64.64% | 64.79% | 0.78724 | 0.2826 | 364482 / 554262 / 390449 | 10.02 |
| batch_norm_dropout | 44 | 18 | 64.81% | 65.00% | 0.77962 | 0.2826 | 366638 / 585702 / 356853 | 9.88 |

Subset run time includes twenty training epochs and their subset evaluations and probes. It excludes the subsequent complete-September evaluation.

### Screening results

Scores use each trial's selected checkpoint on the fixed validation subset. Every run uses seed 42 and twenty epochs.

| Configuration | Selected epoch | Accuracy | Macro-F1 | Loss | Subset run minutes |
| --- | ---: | ---: | ---: | ---: | ---: |
| batch_norm | 15 | 63.98% | 64.13% | 0.82612 | 4.89 |
| batch_norm_dropout | 16 | 63.79% | 63.98% | 0.81224 | 4.91 |
| baseline | 19 | 62.89% | 63.04% | 0.82497 | 4.05 |
| xavier | 15 | 62.60% | 62.76% | 0.83494 | 4.02 |
| batch_64 | 19 | 62.46% | 62.61% | 0.83566 | 2.21 |
| kaiming | 12 | 62.43% | 62.56% | 0.84251 | 4.00 |
| batch_128 | 19 | 62.38% | 62.46% | 0.84316 | 1.86 |
| dropout | 19 | 61.68% | 61.75% | 0.85874 | 3.86 |
| combined_paper | 20 | 60.56% | 60.66% | 0.86518 | 5.30 |
| lr_1e4 | 20 | 53.81% | 53.39% | 0.95801 | 4.01 |
| eps_1e4 | 3 | 35.71% | 17.54% | 1.09659 | 4.00 |
| eps_1e2 | 1 | 35.71% | 17.54% | 1.09780 | 3.98 |
| paper_adam | 1 | 35.71% | 17.54% | 1.09705 | 3.95 |

### Protocol

- Screening includes 13 distinct settings, starting from twelve factor comparisons. Each uses the same 32,768 January-August training windows and 8,192 September validation windows for twenty epochs, seed 42.
- Confirmation uses the two best nonbaseline settings and the baseline, each trained from scratch for twenty epochs on 65,536 windows with a 32,768-window validation subset, for each of three seeds.
- Each checkpoint is selected by strict improvement in subset validation accuracy, then evaluated on the complete September validation set. Macro-F1 breaks ties when ranking configurations. October-December test data is never loaded.
- Larger subsets extend the screening subsets without duplicates. Training examples per epoch are equal across batch sizes; optimizer-update counts differ and are recorded.
- BatchNorm follows convolution activations and uses training running statistics during evaluation. Dropout uses one channel mask shared across the input's time dimension and is disabled during evaluation.
- Xavier and Kaiming presets both use Xavier LSTM input weights, an orthogonal combined recurrent matrix, zero biases except total forget-gate bias one, and a Xavier output layer. Only their convolution weight rule differs. PyTorch initialization preserves all original defaults.
- Adam epsilon is distinct from BatchNorm's epsilon. The paper-Adam comparison changes both learning rate and epsilon; it does not isolate either one.
- Standard deviations describe variability across only three seeds, not a statistical confidence interval. Longer full-year training may change the ranking.


## Files

- `learning_rate_history.csv` and `learning_rate_comparison.png`: the first learning-rate check.
- `screen_summary.csv`, `confirm_summary.csv`, and `ranking.csv`: the later factor comparisons and confirmation.
- The remaining PNG files show screening, confirmation, and validation confusion counts.
- Earlier trial checkpoints, subset indices, and diagnostic source code remain in an external cleanup backup and are not included here.
- Experiment 2 is the ongoing full-data baseline; experiment 3 is the planned full-data BatchNorm run. Their status is maintained in the project plan.
