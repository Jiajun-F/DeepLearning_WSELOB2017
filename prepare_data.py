"""Prepare daily order book data for model training.

Read raw messages, reconstruct snapshots, normalize features, and create labels.
Each step can also run separately. Save daily NPZ files and print progress.
See the usage guide at the bottom of this file for commands and examples.
"""

import argparse
from datetime import time
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from wse_lob import (
    build_daily_orderbooks,
    compute_daily_statistics,
    compute_history_statistics,
    compute_labels,
    extract_daily_nlevels,
    extract_daily_ofn,
    normalize_features,
)


def save_npz(path, **arrays):
    """Save named arrays once; an existing path raises FileExistsError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        np.savez_compressed(output, **arrays)


def run_preparation(
    stage, stock, processed_dir, *,
    output_dir=None,
    source_file=None,
    feature_dir=None,
    start_date=None,
    end_date=None,
    history_days=5, k=None,
    alpha=None, daily_start=time(9),
    daily_end=time(16, 50),
    price_range=0.10,
    n_levels=10,
):
    """Select dates, check outputs, then run extraction and/or daily preparation."""
    # 1. Resolve paths and identify which stages this run needs.
    extract_raw = stage in ("extract", "all")
    prepare_arrays = stage != "extract"
    processed_dir = Path(processed_dir).resolve()
    label_dir = None
    if history_days < 1:
        raise ValueError("history_days must be positive.")
    if stage == "ofn":
        source_file = Path(source_file).resolve()
        feature_dir = Path(output_dir).resolve() / "features_ofn"
        dates, day_keys, selected, extracted = select_trading_days(
            stage, source_file, processed_dir, start_date, end_date, history_days,
        )
        check_output_paths(
            stage, dates, selected, extracted, processed_dir, feature_dir, None,
        )
        print(
            f"ofn: stock={stock}; output_days={len(selected)}; "
            f"history_days={history_days}; max_cost=0.035", flush=True,
        )
        prepare_ofn_daily_files(
            source_file, processed_dir, feature_dir, day_keys, dates, selected,
            history_days=history_days, daily_start=daily_start, daily_end=daily_end,
            price_range=price_range, min_levels=n_levels,
        )
        print("Finished.", flush=True)
        return
    if extract_raw:
        source_file = Path(source_file).resolve()
    if prepare_arrays:
        if k is None or alpha is None:
            raise ValueError("Provide both k and alpha for label preparation.")
        output_dir = Path(output_dir).resolve()
        label_dir = output_dir / "labels" / f"k{k}"
        if stage == "labels":
            feature_dir = Path(feature_dir).resolve()
        else:
            feature_dir = output_dir / "features"

    # 2. Select target dates while retaining earlier dates for normalization history.
    dates, day_keys, selected, extracted = select_trading_days(
        stage, source_file, processed_dir, start_date, end_date, history_days,
    )

    # 3. Reject every existing output before processing.
    check_output_paths(
        stage, dates, selected, extracted, processed_dir, feature_dir, label_dir,
    )

    # 4. Execute the requested stages. Progress is printed, not saved to log files.
    print(f"{stage}: stock={stock}; days={len(selected)}; k={k}; alpha={alpha}", flush=True)
    if extract_raw:
        extract_processed_days(
            source_file, processed_dir, day_keys, dates, extracted,
            daily_start=daily_start, daily_end=daily_end,
            price_range=price_range, n_levels=n_levels,
        )
    if prepare_arrays:
        prepare_daily_files(
            processed_dir, feature_dir, label_dir, dates, selected,
            history_days=history_days, k=k, alpha=alpha,
            labels_only=stage == "labels",
        )
    print("Finished.", flush=True)


def select_trading_days(stage, source_file, processed_dir, start_date, end_date, history_days):
    """Return all dates, raw HDF keys, and target/extraction indices into those lists."""
    day_keys = []
    # Select targets from the full trading-date list, before applying date bounds.
    # Earlier dates remain available for normalization across split boundaries.
    if stage in ("extract", "all", "ofn"):
        with pd.HDFStore(source_file, mode="r") as store:
            day_keys = sorted(store.keys())
        dates = [key.removeprefix("/d") for key in day_keys]
    else:
        dates = [path.stem for path in sorted(processed_dir.glob("*.npz"))]

    first_eligible = 0 if stage == "extract" else history_days
    selected = []
    for index in range(first_eligible, len(dates)):
        date = dates[index]
        if start_date is not None and date < start_date:
            continue
        if end_date is not None and date > end_date:
            continue
        selected.append(index)
    if not selected:
        raise ValueError("No eligible trading dates in the requested range.")

    extracted = []
    if stage == "extract":
        extracted = selected
    elif stage == "all":
        extracted = range(selected[0] - history_days, selected[-1] + 1)
    return dates, day_keys, selected, extracted


def check_output_paths(
    stage, dates, selected, extracted, processed_dir, feature_dir, label_dir,
):
    """Check all output collisions and the existing features needed for labels-only runs."""
    outputs = []
    for index in extracted:
        outputs.append(processed_dir / f"{dates[index]}.npz")
    if stage != "extract":
        for index in selected:
            date = dates[index]
            if stage != "ofn":
                outputs.append(label_dir / f"{date}.npz")
            feature_path = feature_dir / f"{date}.npz"
            if stage == "labels":
                if not feature_path.is_file():
                    raise FileNotFoundError(feature_path)
            else:
                outputs.append(feature_path)
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Output already exists; choose a new output path: {path}")


def extract_processed_days(
    source_file, processed_dir, day_keys, dates, extracted, *,
    daily_start, daily_end, price_range, n_levels,
):
    """Replay selected raw days and save daily snapshots and issue arrays."""
    with pd.HDFStore(source_file, mode="r") as store:
        for index in extracted:
            started = perf_counter()
            date = dates[index]
            messages = store[day_keys[index]]
            # Convert raw units before replaying every message of this day.
            for column in ("time", "priority_date", "order_date"):
                messages[column] = pd.to_datetime(
                    messages[column], unit="ns", utc=True
                ).dt.tz_convert("Europe/Warsaw")
            messages["price"] = messages["price"] / (10 ** messages["price_level"])
            # Reconstruct and filter snapshots, then aggregate price-level features.
            orderbooks, order_issues, snapshot_issues = build_daily_orderbooks(
                messages, daily_start, daily_end,
                price_range=price_range, min_levels=n_levels,
            )
            snapshot_info, features = extract_daily_nlevels(orderbooks, n=n_levels)
            # Snapshot arrays and issue arrays retain their separate row orders.
            save_npz(
                processed_dir / f"{date}.npz",
                features=features,
                mid_price=snapshot_info["mid_price"].to_numpy(dtype=np.float64),
                time_ns=snapshot_info["time"].array.as_unit("ns").asi8,
                skipped_time_ns=np.array(
                    [issue["time"].value for issue in snapshot_issues], dtype=np.int64
                ),
                skipped_message_position=np.array(
                    [issue["message_position"] for issue in snapshot_issues], dtype=np.int64
                ),
                skipped_reason=np.array(
                    [issue["reason"] for issue in snapshot_issues], dtype=np.uint8
                ),
            )
            crossed = sum(issue["reason"] == 1 for issue in snapshot_issues)
            insufficient = sum(issue["reason"] == 2 for issue in snapshot_issues)
            print(
                f"{date}: messages={len(messages)}; snapshots={len(features)}; "
                f"missing_orders={len(order_issues)}; crossed={crossed}; "
                f"insufficient_levels={insufficient}; elapsed={perf_counter() - started:.2f}s",
                flush=True,
            )
            del messages, orderbooks, snapshot_info, features, order_issues, snapshot_issues


def prepare_daily_files(
    processed_dir, feature_dir, label_dir, dates, selected, *,
    history_days, k, alpha, labels_only=False,
):
    """Save aligned labels and, unless labels_only, rolling-normalized daily features."""

    # 1. Initialize history from days before the first target.
    daily_statistics = []
    if not labels_only:
        for index in range(selected[0] - history_days, selected[0]):
            daily_statistics.append(
                compute_daily_statistics(processed_dir / f"{dates[index]}.npz")
            )

    # 2. Prepare one target day at a time, preserving every snapshot row.
    for index in selected:
        started = perf_counter()
        date = dates[index]
        source_path = processed_dir / f"{date}.npz"
        with np.load(source_path, allow_pickle=False) as arrays:
            mid_price = arrays["mid_price"]
            if not labels_only:
                features = arrays["features"]
        # Labels always use original mid-prices, independently of normalization.
        labels = compute_labels(mid_price, k, alpha)
        if labels_only:
            # Pair by date and retain the original snapshot row alignment.
            with np.load(feature_dir / f"{date}.npz", allow_pickle=False) as arrays:
                feature_count = len(arrays["features"])
            if feature_count != len(labels):
                raise ValueError(f"Feature and label row counts differ for {date}.")
        else:
            mean, std = compute_history_statistics(daily_statistics)
            normalized = normalize_features(features, mean, std).astype(np.float32)
            save_npz(feature_dir / f"{date}.npz", features=normalized)
            del features, normalized
        save_npz(label_dir / f"{date}.npz", labels=labels)
        counts = np.bincount(labels[labels >= 0], minlength=3)
        print(
            f"{date}: labels={len(labels)}; valid={counts.sum()}; "
            f"down={counts[0]}; stationary={counts[1]}; up={counts[2]}; "
            f"elapsed={perf_counter() - started:.2f}s", flush=True,
        )
        # 3. Drop the oldest day and add the current raw day for the next target.
        # Cache small daily statistics, not full historical feature arrays.
        if not labels_only and index != selected[-1]:
            daily_statistics.pop(0)
            daily_statistics.append(compute_daily_statistics(source_path))
        del mid_price, labels


def prepare_ofn_daily_files(
    source_file, processed_dir, feature_dir, day_keys, dates, selected, *,
    history_days, daily_start, daily_end, price_range, min_levels,
):
    """Rebuild daily books and save normalized OFN features aligned with processed data."""
    daily_statistics = []
    # Rebuild the preceding days too, even when the requested output starts later.
    first_day = selected[0] - history_days
    with pd.HDFStore(source_file, mode="r") as store:
        for index in range(first_day, selected[-1] + 1):
            started = perf_counter()
            date = dates[index]
            messages = store[day_keys[index]]
            for column in ("time", "priority_date", "order_date"):
                messages[column] = pd.to_datetime(
                    messages[column], unit="ns", utc=True
                ).dt.tz_convert("Europe/Warsaw")
            messages["price"] = messages["price"] / (10 ** messages["price_level"])
            # Match the existing minimum-depth filter for label alignment.
            # OFN still uses all orders in the selected price band.
            orderbooks, order_issues, snapshot_issues = build_daily_orderbooks(
                messages, daily_start, daily_end,
                price_range=price_range, min_levels=min_levels,
            )
            snapshot_info, features = extract_daily_ofn(orderbooks, max_cost=0.035)

            # The existing labels refer to these exact rows. Matching counts alone
            # would not detect a skipped or shifted snapshot.
            with np.load(processed_dir / f"{date}.npz", allow_pickle=False) as arrays:
                if not np.array_equal(
                    snapshot_info["time"].array.as_unit("ns").asi8, arrays["time_ns"]
                ):
                    raise ValueError(f"OFN snapshot times differ from processed data for {date}.")
                if not np.array_equal(snapshot_info["mid_price"].to_numpy(), arrays["mid_price"]):
                    raise ValueError(f"OFN mid-prices differ from processed data for {date}.")

            count = len(features)
            current_statistics = (
                count,
                features.mean(axis=0, dtype=np.float64),
                features.std(axis=0, ddof=0, dtype=np.float64),
            )
            if index >= selected[0]:
                # History still contains only preceding days. Do not include
                # today's values in the mean or standard deviation used for today.
                mean, std = compute_history_statistics(daily_statistics)
                normalized = normalize_features(features, mean, std).astype(np.float32)
                save_npz(feature_dir / f"{date}.npz", features=normalized)
                del normalized
                status = "saved"
            else:
                status = "history only"

            daily_statistics.append(current_statistics)
            if len(daily_statistics) > history_days:
                daily_statistics.pop(0)
            print(
                f"{date}: snapshots={count}; alignment=matched; {status}; "
                f"elapsed={perf_counter() - started:.2f}s", flush=True,
            )
            del messages, orderbooks, snapshot_info, features, order_issues, snapshot_issues


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage", choices=["extract", "prepare", "labels", "all", "ofn"])
    parser.add_argument("--stock", required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--feature-dir", type=Path, help="Existing features, required for labels stage.")
    parser.add_argument("--start-date", help="Inclusive YYYYMMDD target date.")
    parser.add_argument("--end-date", help="Inclusive YYYYMMDD target date.")
    parser.add_argument("--history-days", type=int, default=5)
    parser.add_argument("--k", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--daily-start", type=time.fromisoformat, default=time(9))
    parser.add_argument("--daily-end", type=time.fromisoformat, default=time(16, 50))
    parser.add_argument("--price-range", type=float, default=0.10)
    parser.add_argument("--n-levels", type=int, default=10)
    args = parser.parse_args()
    if args.stage in ("extract", "all", "ofn") and args.source_file is None:
        parser.error("--source-file is required for extract, all, and ofn.")
    if args.stage in ("prepare", "labels", "all") and (args.output_dir is None or args.k is None or args.alpha is None):
        parser.error("--output-dir, --k, and --alpha are required for preparation.")
    if args.stage == "ofn" and args.output_dir is None:
        parser.error("--output-dir is required for ofn.")
    if args.stage == "labels" and args.feature_dir is None:
        parser.error("--feature-dir is required for labels.")
    run_preparation(**vars(args))


# Usage is kept here and also shown by --help.
USAGE = r"""
Run from the project root with your Python environment activated.
Install dependencies with: python -m pip install -r requirements.txt
All example paths are relative to the project root.

Choose a stage:
  extract  Raw HDF messages -> processed daily snapshots.
  prepare  Processed snapshots -> normalized features and labels.
  labels   Create labels using existing processed data and normalized features.
  all      Run extract and prepare together for new output files.
  ofn      Raw HDF messages -> normalized OFN features; reuse existing labels.

Every stage needs --stock and --processed-dir.
extract/all/ofn need --source-file; prepare/labels/all need --output-dir, --k,
and --alpha. The labels stage also needs --feature-dir.
ofn needs --output-dir and existing processed days for row alignment checks.
It writes only features_ofn/, without creating labels or processed OFN files.
--n-levels sets the number of levels per side for n-levels features and the
minimum required per side when selecting snapshots. For ofn, it only sets
this minimum; OFN uses all orders within the selected price band.

Example: prepare KGHM from raw messages (only when these outputs do not exist):

  python prepare_data.py all \
    --stock KGHM \
    --source-file data/wselob_2017_source/orders/KGHM_lob_2017_zlib.h5 \
    --processed-dir data/wselob_2017_processed/KGHM \
    --output-dir data/wselob_2017_prepared/KGHM \
    --history-days 5 --k 20 --alpha 0.0000263

For another stock, change the stock name and matching paths, and choose its
threshold. Changing --stock alone does not change the input or output paths.
The threshold above is currently agreed only for KGHM with k=20.

Example: add labels for a new horizon while reusing KGHM features.
Replace K_VALUE and ALPHA_VALUE with chosen numbers:

  python prepare_data.py labels \
    --stock KGHM \
    --processed-dir data/wselob_2017_processed/KGHM \
    --feature-dir data/wselob_2017_prepared/KGHM/features \
    --output-dir data/wselob_2017_prepared/KGHM \
    --history-days 5 --k K_VALUE --alpha ALPHA_VALUE

--k counts prediction length; --alpha is the classification threshold.

Example: generate OFN features for KGHM's sixth trading day, using the first
five trading days for normalization history:

  python prepare_data.py ofn \
    --stock KGHM \
    --source-file data/wselob_2017_source/orders/KGHM_lob_2017_zlib.h5 \
    --processed-dir data/wselob_2017_processed/KGHM \
    --output-dir data/wselob_2017_prepared/KGHM \
    --history-days 5 --start-date 20170110 --end-date 20170110

OFN uses ten costs: ((11**(i/10) - 1) / 10) * 0.035, for i=1,...,10.
Each group contains [minimum average sell price, -sell volume/1e6,
maximum average buy price, buy volume/1e6] before column-wise normalization.
Date bounds select output days. Earlier raw days are also read for history.

Existing NPZ files are never overwritten or silently skipped. Use
prepare if processed files already exist, or labels if features also exist.
Choose a separate output directory when changing settings for existing outputs.
Feature files contain only features; labels are saved in labels/k<k>/.
See README.md for normalization, labels, snapshot rules, and NPZ formats.
"""


if __name__ == "__main__":
    main()
