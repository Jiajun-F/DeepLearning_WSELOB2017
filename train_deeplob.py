"""Load daily windows, train DeepLOB, and save evaluation reports and figures.

The notebook imports these functions; the command line runs the same workflow.
Run from the project root. New runs do not overwrite existing directories.
See --help for commands and parameters.
"""

import argparse
import csv
import os
from pathlib import Path
import random
import sys
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

from deeplob import DeepLOB


# Daily arrays and input windows

class DailyWindowDataset(torch.utils.data.Dataset):
    """Slice daily feature arrays into input windows when samples are requested.

    daily_features and daily_labels contain matching arrays for each day.
    Each sample returns a float32 window and the integer label at its end.
    """

    def __init__(self, daily_features, daily_labels, window_length=50, horizon=20):
        if len(daily_features) != len(daily_labels):
            raise ValueError("Provide one label array for each daily feature array.")
        if window_length < 1 or horizon < 1:
            raise ValueError("window_length and horizon must be positive.")

        self.daily_features = daily_features
        self.daily_labels = daily_labels
        self.window_length = window_length
        # A prediction needs its input window and k past prices plus the current one.
        self.first_position = max(window_length - 1, horizon)
        sample_counts = []
        for features, labels in zip(daily_features, daily_labels):
            if features.ndim != 2 or labels.shape != (len(features),):
                raise ValueError("Daily features and labels must have matching rows.")
            if not np.issubdtype(labels.dtype, np.integer):
                raise ValueError("Labels must be integer class IDs.")
            # Leave k future rows for the label; all indices stay within this day.
            count = max(0, len(features) - horizon - self.first_position)
            sample_counts.append(count)
        self.day_offsets = np.cumsum([0] + sample_counts, dtype=np.int64)

    def __len__(self):
        return int(self.day_offsets[-1])

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)

        # Locate the day and the prediction position for this sample index.
        day = int(np.searchsorted(self.day_offsets, index, side="right") - 1)
        position = self.first_position + int(index - self.day_offsets[day])
        start = position - self.window_length + 1
        window = self.daily_features[day][start:position + 1]
        label = self.daily_labels[day][position]
        return (
            torch.as_tensor(window, dtype=torch.float32),
            torch.as_tensor(label, dtype=torch.long),
        )


def load_daily_dataset(feature_dir, label_dir, dates, *, window_length=50, horizon=20):
    """Read matching daily NPZ files and retain arrays, not expanded windows."""
    daily_features, daily_labels = [], []
    for date in dates:
        with np.load(Path(feature_dir) / f"{date}.npz", allow_pickle=False) as arrays:
            daily_features.append(arrays["features"])
        with np.load(Path(label_dir) / f"{date}.npz", allow_pickle=False) as arrays:
            daily_labels.append(arrays["labels"])
    return DailyWindowDataset(daily_features, daily_labels, window_length, horizon)


# Training and metrics

def classification_metrics(confusion):
    """Return accuracy and class scores; rows are true classes, columns predicted."""
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    correct = np.diag(confusion)
    # A class with no predictions or no true examples receives a zero score.
    precision = np.divide(
        correct, predicted, out=np.zeros(len(support)), where=predicted > 0,
    )
    recall = np.divide(
        correct, support, out=np.zeros(len(support)), where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(len(support)), where=precision + recall > 0,
    )
    return {
        "accuracy": float(correct.sum() / support.sum()),
        # OFN weights each class F1 by its true sample count.
        "weighted_f1": float(np.dot(f1, support) / support.sum()),
        # Retained for the saved experiment 2 evaluation code.
        "macro_f1": float(f1.mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "confusion": confusion,
    }


def run_epoch(model, loader, criterion, device, optimizer=None, *, log_every=1000):
    """Run one training or evaluation pass with a mean-reduced loss function."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    confusion = torch.zeros((3, 3), dtype=torch.int64)
    started = perf_counter()

    with torch.set_grad_enabled(training):
        for batch_index, (features, labels) in enumerate(loader, start=1):
            features = features.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            # Weight by samples so the final partial batch contributes correctly.
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            predictions = logits.detach().argmax(dim=1).cpu()
            pairs = labels.detach().cpu() * 3 + predictions
            confusion += torch.bincount(pairs, minlength=9).reshape(3, 3)

            if log_every and batch_index % log_every == 0:
                phase = "train" if training else "evaluation"
                # Near-zero variation flags predictions that ignore the current inputs.
                probabilities = logits.detach().softmax(dim=1)
                probability_std = probabilities.std(dim=0, unbiased=False).mean().item()
                print(
                    f"{phase}: batch {batch_index}/{len(loader)}; "
                    f"loss={total_loss / sample_count:.6f}; "
                    f"batch_probability_std={probability_std:.6f}; "
                    f"predicted_counts={confusion.sum(dim=0).tolist()}; "
                    f"elapsed={perf_counter() - started:.1f}s",
                    flush=True,
                )

    metrics = classification_metrics(confusion.numpy())
    metrics["loss"] = total_loss / sample_count
    return metrics


def save_checkpoint(checkpoint, path):
    """Replace a checkpoint only after its new file has been written completely."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_model(
    model, train_loader, validation_loader, optimizer, criterion, device,
    run_dir, *, epochs, patience, selection_metric, config, log_every=1000,
):
    """Train from the supplied weights, record epoch metrics, and save the best model."""
    if selection_metric not in ("accuracy", "weighted_f1"):
        raise ValueError("Select checkpoints by accuracy or weighted_f1.")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    history = []
    best_score = -float("inf")
    epochs_without_improvement = 0
    columns = [
        "epoch", "train_loss", "train_accuracy", "train_weighted_f1",
        "validation_loss", "validation_accuracy", "validation_weighted_f1",
        "best", "elapsed_seconds",
    ]

    with (run_dir / "history.csv").open("x", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=columns)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            started = perf_counter()
            print(f"epoch={epoch}/{epochs}", flush=True)
            train_metrics = run_epoch(
                model, train_loader, criterion, device, optimizer, log_every=log_every,
            )
            validation_metrics = run_epoch(
                model, validation_loader, criterion, device, log_every=log_every,
            )
            score = validation_metrics[selection_metric]
            improved = score > best_score
            if improved:
                best_score = score
                epochs_without_improvement = 0
                save_checkpoint({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch, "selection_metric": selection_metric,
                    "best_score": best_score, "config": config,
                }, run_dir / "best_model.pt")
            else:
                epochs_without_improvement += 1

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_weighted_f1": train_metrics["weighted_f1"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_weighted_f1": validation_metrics["weighted_f1"],
                "best": improved,
                "elapsed_seconds": perf_counter() - started,
            }
            history.append(row)
            writer.writerow(row)
            history_file.flush()
            print(
                f"train_loss={row['train_loss']:.6f}; "
                f"validation_loss={row['validation_loss']:.6f}; "
                f"validation_accuracy={row['validation_accuracy']:.6f}; "
                f"validation_weighted_f1={row['validation_weighted_f1']:.6f}; "
                f"best={improved}; elapsed={row['elapsed_seconds']:.1f}s", flush=True,
            )
            if epochs_without_improvement >= patience:
                print(f"early_stop: no validation improvement for {patience} epochs", flush=True)
                break
    print("status=finished", flush=True)
    return history


# Figures from saved metrics

def plot_training_history(history, run_dir):
    """Plot epoch losses and scores, marking the best validation checkpoint."""
    epochs = [row["epoch"] for row in history]
    best_epoch = [row["epoch"] for row in history if row["best"]][-1]
    # Existing experiment histories used macro-F1; do not relabel those scores.
    f1_metric = "weighted_f1" if "train_weighted_f1" in history[0] else "macro_f1"
    f1_title = "Weighted F1" if f1_metric == "weighted_f1" else "Macro-F1"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, metric, title in zip(
        axes, ("loss", "accuracy", f1_metric), ("Loss", "Accuracy", f1_title),
    ):
        for split, label, color in (
            ("train", "Training", "#2878B5"),
            ("validation", "Validation", "#E07A32"),
        ):
            values = [row[f"{split}_{metric}"] for row in history]
            ax.plot(epochs, values, label=label, color=color, marker=".")
        ax.axvline(best_epoch, color="0.4", linestyle="--", label="Best checkpoint")
        ax.set(title=title, xlabel="Epoch", ylabel=title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if metric != "loss":
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("DeepLOB training and validation")
    path = Path(run_dir) / "training_curves.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_confusion_matrix(metrics, class_names, run_dir):
    """Save test counts and percentages normalized within each true class."""
    confusion = metrics["confusion"]
    support = confusion.sum(axis=1, keepdims=True)
    fractions = np.divide(
        confusion, support, out=np.zeros_like(confusion, dtype=float), where=support > 0,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, values, title, normalized in (
        (axes[0], confusion, "Sample counts", False),
        (axes[1], fractions, "Percentage of each true class", True),
    ):
        maximum = 1 if normalized else max(1, int(confusion.max()))
        image = ax.imshow(values, cmap="Blues", vmin=0, vmax=maximum)
        for row in range(len(class_names)):
            for column in range(len(class_names)):
                value = values[row, column]
                label = f"{value:.1%}" if normalized else f"{value:,}"
                ax.text(
                    column, row, label, ha="center", va="center",
                    color="white" if value > maximum / 2 else "black",
                )
        ax.set(
            title=title, xlabel="Predicted class", ylabel="True class",
            xticks=range(len(class_names)), xticklabels=class_names,
            yticks=range(len(class_names)), yticklabels=class_names,
        )
        colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
        if normalized:
            colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    fig.suptitle("DeepLOB test confusion matrix")
    path = Path(run_dir) / "confusion_matrix.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_test_metrics(metrics, baseline, class_names, run_dir):
    """Compare test scores with the training-majority baseline and show class scores."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    positions = np.arange(2)
    width = 0.35
    for offset, scores, name, color in (
        (-width / 2, metrics, "DeepLOB", "#2878B5"),
        (width / 2, baseline, "Training-majority baseline", "#888888"),
    ):
        bars = axes[0].bar(
            positions + offset, [scores["accuracy"], scores["weighted_f1"]],
            width, label=name, color=color,
        )
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axes[0].set(title="Overall test scores", xticks=positions, xticklabels=["Accuracy", "Weighted F1"])

    positions = np.arange(len(class_names))
    width = 0.25
    for offset, key, label, color in (
        (-width, "precision", "Precision", "#2878B5"),
        (0, "recall", "Recall", "#E07A32"),
        (width, "f1", "F1", "#409779"),
    ):
        axes[1].bar(positions + offset, metrics[key], width, label=label, color=color)
    axes[1].set(title="DeepLOB scores by class", xticks=positions, xticklabels=class_names)
    for ax in axes:
        ax.set(ylabel="Score", ylim=(0, 1.08), yticks=np.arange(0, 1.01, 0.2))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    path = Path(run_dir) / "test_metrics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# Command-line workflow and final evaluation

def evaluate_test(model, train_dataset, test_dataset, config, run_dir, device):
    """Evaluate once, then save class scores, a majority baseline, and figures."""
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"])
    metrics = run_epoch(model, loader, torch.nn.CrossEntropyLoss(), device)
    with (run_dir / "test_metrics.npz").open("xb") as output:
        np.savez(output, **{name: value for name, value in metrics.items() if name != "macro_f1"})

    counts = np.zeros(3, dtype=np.int64)
    for labels in train_dataset.daily_labels:
        targets = labels[train_dataset.first_position:len(labels) - config["horizon"]]
        counts += np.bincount(targets, minlength=3)
    majority = int(counts.argmax())
    confusion = np.zeros((3, 3), dtype=np.int64)
    confusion[:, majority] = metrics["support"]
    baseline = classification_metrics(confusion)
    names = config["class_names"]
    summary = pd.DataFrame([
        {"Model": "DeepLOB", "Accuracy": float(metrics["accuracy"]),
         "Weighted F1": float(metrics["weighted_f1"])},
        {"Model": "Training-majority baseline", "Accuracy": baseline["accuracy"],
         "Weighted F1": baseline["weighted_f1"]},
    ]).set_index("Model")
    per_class = pd.DataFrame({
        "Precision": metrics["precision"], "Recall": metrics["recall"],
        "F1": metrics["f1"], "Support": metrics["support"],
    }, index=names)
    table = pd.DataFrame(metrics["confusion"], index=names, columns=names)
    table.index.name = "True class"
    table.columns.name = "Predicted class"
    report = (
        f"Checkpoint epoch: {checkpoint['epoch']}\n"
        f"Selection: validation accuracy = {checkpoint['best_score']:.6f}\n"
        f"Test loss: {metrics['loss']:.6f}\n"
        "F1 averaging: weighted by true class support\n"
        f"Training-majority class: {majority} ({names[majority]})\n\n"
        + summary.to_string() + "\n\n" + per_class.to_string() + "\n\n"
        + table.to_string() + "\n"
    )
    (run_dir / "test_results.txt").write_text(report)
    print(report, flush=True)
    plot_confusion_matrix(metrics, names, run_dir)
    plot_test_metrics(metrics, baseline, names, run_dir)


def main(args):
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use a GPU allocation.")
    if args.run_dir.exists():
        raise FileExistsError(f"Choose a new experiment directory: {args.run_dir}")
    print(f"Slurm job: {os.environ.get('SLURM_JOB_ID')}; device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}; CUDA runtime: {torch.version.cuda}", flush=True)

    # Select complete trading dates; load test arrays only after training has finished.
    root = args.prepared_root / args.stock
    feature_dir = root / ("features_ofn" if args.features == "ofn" else "features")
    label_dir = root / "labels" / f"k{args.horizon}"
    print(f"features={args.features}; feature_dir={feature_dir}", flush=True)
    dates = [path.stem for path in sorted(feature_dir.glob("*.npz"))]
    ranges = {"train": ("0101", "0831"), "validation": ("0901", "0930"),
              "test": ("1001", "1231")}
    data_dates = {}
    for split, (start, end) in ranges.items():
        data_dates[split] = [date for date in dates if f"{args.year}{start}" <= date <= f"{args.year}{end}"]
        if not data_dates[split]:
            raise ValueError(f"No prepared {split} dates found in {feature_dir} for {args.year}.")
    datasets = {}
    for split in ("train", "validation"):
        datasets[split] = load_daily_dataset(
            feature_dir, label_dir, data_dates[split],
            window_length=args.window_length, horizon=args.horizon,
        )
        print(f"{split}: {len(data_dates[split])} days; {len(datasets[split]):,} samples", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0,
        drop_last=False, generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = torch.utils.data.DataLoader(
        datasets["validation"], batch_size=args.batch_size, shuffle=False,
        num_workers=0, drop_last=False,
    )
    model = DeepLOB(n_levels=args.n_levels, batch_norm=args.batch_norm,
                    dropout=args.dropout, initialization=args.initialization).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.eps,
                                 betas=(0.9, 0.999), weight_decay=0)
    config = {
        "stock": args.stock, "window_length": args.window_length, "n_levels": args.n_levels,
        "feature_type": args.features,
        "horizon": args.horizon, "label_alpha": args.label_alpha,
        "label_formula": "past=sum(p[t-k:t+1])/(k+1); future=sum(p[t+1:t+k+1])/k; change=(future-past)/past",
        "normalization": "per column; preceding five trading days; equal snapshot weights; ddof=0; history continues across splits",
        "feature_dir": str(feature_dir), "label_dir": str(label_dir), "data_dates": data_dates,
        "class_names": ["down", "stationary", "up"],
        "sample_counts": {split: len(data) for split, data in datasets.items()},
        "training_seed": args.seed, "loader_seed": args.seed,
        "batch_norm": args.batch_norm, "dropout": args.dropout, "initialization": args.initialization,
        "batch_size": args.batch_size, "num_workers": 0, "drop_last": False,
        "optimizer": "Adam", "learning_rate": args.learning_rate, "eps": args.eps,
        "betas": [0.9, 0.999], "weight_decay": 0,
        "max_epochs": args.max_epochs, "patience": args.patience, "selection_metric": "accuracy",
        "f1_average": "weighted",
        "device": str(device), "python": sys.version.split()[0], "numpy": np.__version__,
        "pytorch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cpu_threads": torch.get_num_threads(),
    }
    history = train_model(
        model, train_loader, validation_loader, optimizer, torch.nn.CrossEntropyLoss(), device,
        args.run_dir, epochs=args.max_epochs, patience=args.patience,
        selection_metric="accuracy", config=config, log_every=1000,
    )
    plot_training_history(history, args.run_dir)
    test_dataset = load_daily_dataset(
        feature_dir, label_dir, data_dates["test"],
        window_length=args.window_length, horizon=args.horizon,
    )
    print(f"test: {len(data_dates['test'])} days; {len(test_dataset):,} samples", flush=True)
    evaluate_test(model, datasets["train"], test_dataset, config, args.run_dir, device)
    print("status=finished; test report and figures saved", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stock", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("data/wselob_2017_prepared"))
    parser.add_argument("--features", choices=("nlevels", "ofn"), default="nlevels",
                        help="Input representation: features/ or features_ofn/; labels are shared.")
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--label-alpha", type=float, default=0.0000263,
                        help="Describe the existing labels; this does not regenerate them.")
    parser.add_argument("--window-length", type=int, default=50)
    parser.add_argument("--n-levels", type=int, default=10,
                        help="Number of four-column input groups: price levels or OFN costs.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--initialization", choices=("pytorch", "xavier", "kaiming", "he_normal"), default="pytorch")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    sys.exit(main(parser.parse_args()))
