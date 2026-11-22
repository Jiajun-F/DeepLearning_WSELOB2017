# Experiment 2: Full-data DeepLOB baseline

Purpose: measure the original model on the full training split. The user stopped
this experiment during epoch 17 after 16 complete epochs. The best checkpoint
has been evaluated on October-December; this experiment has not been selected
as a final project result.

Use KGHM 2017, input length 50, ten levels per side, k=20, and alpha=0.0000263.
Train on January-August, validate on September, and test on October-December
only after training. The model has no BatchNorm or dropout and uses PyTorch
initialization. Adam uses learning rate 0.001, epsilon 1e-8, batch size 32,
and seed 42. Train at most 100 epochs; select validation accuracy with patience 20.

- `history.csv`: completed epoch metrics.
- `best_model.pt`: the best validation checkpoint and its configuration.
- `console.log`: training output up to the stop.
- `evaluate.py`: the October-December evaluation code; prints results without saving files.
- This README: the experiment description and test results.

Stopped on 2026-11-08 at the user's request. The best saved checkpoint is epoch 1:
validation accuracy 66.1436%, macro-F1 66.3426%, and loss 0.749159.
The epoch-17 partial training pass is not counted as a completed epoch.
The model, completed history, and console output were preserved without changing
their contents.

## Test evaluation

Evaluated on 2026-11-08 using the saved epoch-1 model, PyTorch 2.11.0, and the
Apple GPU through MPS. September validation data was not included in this pass.
The test period contains 62 trading days and 3,765,697 eligible windows.

Use the existing normalized features, labels, and within-day windows. Evaluate
in chronological order with batch size 32, num_workers=0, drop_last=False,
model evaluation mode, and gradients disabled. Overall loss is weighted by
sample count; overall classification scores are computed from the summed
confusion matrix, not by averaging the monthly scores. The pass took about
690 seconds. The checkpoint was unchanged, and no output files were written
during evaluation. The code and these results were added afterward at the
user's request.

| Period | Trading days | Samples | Loss | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| October | 22 | 1,498,952 | 0.779294 | 64.9895% | 65.1450% |
| November | 21 | 1,202,593 | 0.767741 | 64.9829% | 65.0692% |
| December | 19 | 1,064,152 | 0.743582 | 66.6299% | 66.8275% |
| **Overall** | **62** | **3,765,697** | **0.765513** | **65.4509%** | **65.6166%** |

Overall class metrics:

| Class | Precision | Recall | F1 | True samples | Predicted samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Down (0) | 77.7254% | 59.8517% | 67.6275% | 1,351,666 | 1,040,837 |
| Stationary (1) | 51.3009% | 76.9451% | 61.5591% | 1,086,291 | 1,629,305 |
| Up (2) | 74.8334% | 61.7471% | 67.6633% | 1,327,740 | 1,095,555 |

Confusion matrix: rows are true classes; columns are predicted classes.

| True / predicted | Down | Stationary | Up |
| --- | ---: | ---: | ---: |
| Down | 808,995 | 403,740 | 138,931 |
| Stationary | 113,660 | 835,848 | 136,783 |
| Up | 118,182 | 389,717 | 819,841 |

Training label counts are 4,630,331 down, 4,435,554 stationary, and 4,585,596 up.
The training-majority baseline therefore always predicts down. On the same
test set, it reaches accuracy 35.8942% and macro-F1 17.6089%.

The saved first-epoch model retains predictive ability despite the subsequent
training deterioration. Monthly scores are similar. A common error is predicting
stationary for directional moves: stationary recall is 76.95%, but its precision
is 51.30%. These results concern this checkpoint, stock, and test period.

To rerun the same evaluation from the project root:

```bash
python -B experiments/experiment_02/evaluate.py
```

The script uses the checkpoint's configuration and requires an available Apple
GPU. It reuses the existing Dataset and metric definitions in memory; it avoids
importing plotting code because the original evaluation was requested without
new files. It only prints results and does not train or save a model.

See the experiment index in the [project README](../../README.md#experiments-and-results).
