"""Order book reconstruction and daily feature preparation for WSELOB-2017."""

# Adapted from MinOrderBook in pyofn by Adam Marszalek.
# Source: https://github.com/amarszalek/pyofn/blob/
# cacb97bc163ccca2a76659f9b28728beb54262a5/pyofn/utils.py
# License: Apache-2.0; see licenses/pyofn_LICENSE.
# Changes: standalone message dispatch, structured missing-order diagnostics,
# explicit retransmission handling, copies of incoming order records,
# daily sampling with a mid-price band and positive-price quotes,
# crossed/depth filtering with separate snapshot diagnostics,
# and price-level features aggregated across order types.

import numpy as np
import pandas as pd


class OrderBook:
    """Maintain individual orders in buy_map and sell_map."""

    def __init__(self):
        self.buy_map = {}
        self.sell_map = {}
        self.current_time = None
        self.nmsg = 0
        self.issues = []

    def process_message(self, message):
        """Apply one message containing time and the 12 order fields."""
        self.current_time = message["time"]
        self.nmsg += 1

        order = message.copy()
        del order["time"]
        action = order["action_type"]

        if action == "F":
            self.clear_orderbook()
        elif action == "Y":
            self.retransmit_order(order)
        elif action == "A":
            self.add_order(order)
        elif action == "M":
            self.mod_order(order)
        elif action == "D":
            self.del_order(order)
        else:
            raise ValueError(f"Unsupported action_type: {action}")

    def clear_orderbook(self):
        """Clear both sides for an F message; retain the message counter."""
        self.buy_map.clear()
        self.sell_map.clear()

    def _get_key(self, order):
        return order["order_date"], order["order_id"]

    def _get_order_map(self, side):
        if side == 1:
            return self.buy_map
        if side in (2, 5):
            return self.sell_map
        raise KeyError(f"Unsupported side: {side}")

    def _record_missing_order(self, order, handling):
        issue = order.copy()
        issue["time"] = self.current_time
        issue["message_number"] = self.nmsg
        issue["issue"] = "missing_order"
        issue["handling"] = handling
        self.issues.append(issue)

    def add_order(self, order):
        """Insert an A message; an existing key raises KeyError."""
        order_map = self._get_order_map(order["side"])
        key = self._get_key(order)
        if key in order_map:
            raise KeyError(f"Order already exists: {key}")
        order_map[key] = order.copy()

    def mod_order(self, order):
        """Apply supplied fields; record and add an absent order."""
        order_map = self._get_order_map(order["side"])
        key = self._get_key(order)
        if key not in order_map:
            self._record_missing_order(order, "add")
            self.add_order(order)
            return

        existing = order_map[key]
        # Match pyofn: -1 marks a field that this message does not update.
        for field, value in order.items():
            if value != -1:
                existing[field] = value

    def del_order(self, order):
        """Remove a D message's order; record and skip an absent order."""
        order_map = self._get_order_map(order["side"])
        key = self._get_key(order)
        if key not in order_map:
            self._record_missing_order(order, "skip")
            return
        del order_map[key]

    def retransmit_order(self, order):
        """Insert a Y message, or update the order if its key exists."""
        order_map = self._get_order_map(order["side"])
        key = self._get_key(order)
        if key in order_map:
            self.mod_order(order)
        else:
            self.add_order(order)

    @staticmethod
    def add_perturbation(features, probability, rng, tick_size=0.01):
        """Return perturbed raw (N, 4*n_levels) features using the supplied NumPy RNG.

        Keep the original row order and depth; leave the input array unchanged.
        """
        if not 0 <= probability <= 1:
            raise ValueError("probability must be between 0 and 1.")
        if tick_size <= 0:
            raise ValueError("tick_size must be positive.")

        # Probability-based filling follows the WSELOB-2017 dataset example:
        # data/wselob_2017_source/orderbook2.py, OrderBook.add_perturbation.
        # Each level is [ask_price, ask_volume, bid_price, bid_volume].
        n_levels = features.shape[1] // 4
        perturbed = features.copy()
        scaled_prices = features[:, 0::2] / tick_size
        price_ticks = np.rint(scaled_prices).astype(np.int64)
        # Rounding removes floating-point error, not genuine off-grid prices.
        if not np.allclose(scaled_prices, price_ticks, rtol=0, atol=1e-6):
            raise ValueError("Raw prices must lie on the specified tick grid.")

        for row in range(len(features)):
            # Process asks first, then bids, always using the original snapshot.
            for side in (0, 1):
                price_column = 2 * side
                prices = features[row, price_column::4]
                volumes = features[row, price_column + 1::4]
                ticks = price_ticks[row, side::2]

                # The original first and last levels bound the candidate ticks.
                lower_tick = min(ticks[0], ticks[-1])
                upper_tick = max(ticks[0], ticks[-1])
                interior_ticks = np.arange(lower_tick + 1, upper_tick)
                vacant_ticks = np.setdiff1d(interior_ticks, ticks, assume_unique=True)
                # Draw once per vacant tick, including ticks later outside the top ten.
                selected_ticks = vacant_ticks[rng.random(len(vacant_ticks)) < probability]
                if len(selected_ticks) == 0:
                    continue

                # Keep original prices exactly; only new orders use the tick conversion.
                combined_prices = np.concatenate((prices, selected_ticks * tick_size))
                combined_volumes = np.concatenate((volumes, np.ones(len(selected_ticks))))
                order = np.argsort(combined_prices)
                if side == 1:
                    order = order[::-1]
                order = order[:n_levels]
                perturbed[row, price_column::4] = combined_prices[order]
                perturbed[row, price_column + 1::4] = combined_volumes[order]

        return perturbed


def perturb_orderbook(orderbook, probability, rng, tick_size=0.01, n_levels=10):
    """Copy a snapshot and add unit-volume orders between its original L1 and Ln.

    Keep deeper orders and the original snapshot unchanged. New orders contain
    only price and volume because this copy is used only for feature extraction.
    """
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1.")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive.")

    perturbed = orderbook.copy()
    # Match experiment 4: asks before bids, vacant ticks in ascending order.
    for map_name, descending in (("sell_map", False), ("buy_map", True)):
        orders = orderbook[map_name]
        prices = sorted({order["price"] for order in orders.values()}, reverse=descending)
        prices = np.asarray(prices[:n_levels], dtype=np.float64)
        scaled_prices = prices / tick_size
        ticks = np.rint(scaled_prices).astype(np.int64)
        if not np.allclose(scaled_prices, ticks, rtol=0, atol=1e-6):
            raise ValueError("Raw prices must lie on the specified tick grid.")

        lower_tick = min(ticks[0], ticks[-1])
        upper_tick = max(ticks[0], ticks[-1])
        interior_ticks = np.arange(lower_tick + 1, upper_tick)
        vacant_ticks = np.setdiff1d(interior_ticks, ticks, assume_unique=True)
        selected_ticks = vacant_ticks[rng.random(len(vacant_ticks)) < probability]

        # Copy the map; do not insert orders into the reconstruction state.
        perturbed[map_name] = orders.copy()
        for tick in selected_ticks:
            # Real keys are (order_date, order_id), so these keys cannot replace them.
            perturbed[map_name][("perturbation", int(tick))] = {
                "price": tick * tick_size,
                "volume": 1.0,
            }
    return perturbed


def build_daily_orderbooks(messages, daily_start, daily_end, price_range=0.10, min_levels=10):
    """Return an order-book iterator, order issues, and snapshot issues.

    messages contains one stock-day in file order, with scaled prices and
    Europe/Warsaw timestamps. daily_start/daily_end select the output time
    interval: daily_start <= timestamp < daily_end, in Europe/Warsaw.
    price_range=0.10 selects prices within +/-10% of mid-price. Skip crossed
    books and books with fewer than min_levels selected prices on either side.
    Return all orders in the price band; min_levels does not limit book depth.

    The iterator supplies time, mid_price, buy_map, and sell_map one at a time.
    Both issue lists fill as the iterator processes messages.
    """
    if daily_start >= daily_end:
        raise ValueError("daily_start must be earlier than daily_end.")
    if not 0 < price_range < 1:
        raise ValueError("price_range must be between 0 and 1.")
    if not isinstance(min_levels, int) or min_levels < 1:
        raise ValueError("min_levels must be a positive integer.")

    timestamps = messages["time"]
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Message times must be present and in file-time order.")
    if timestamps.dt.normalize().nunique() != 1:
        raise ValueError("Provide messages from one trading day.")
    if messages["symbol_idx"].nunique() != 1:
        raise ValueError("Provide messages from one stock.")
    book = OrderBook()
    snapshot_issues = []

    def iter_orderbooks():
        columns = [
            "time", "priority_date", "order_date", "symbol_idx", "price",
            "agg_volume", "volume", "order_id", "num_orders", "side",
            "order_type", "action_type", "price_level",
        ]
        group_ends = timestamps.ne(timestamps.shift(-1)).to_numpy()
        previous_buy = None
        previous_sell = None

        for position, values in enumerate(messages[columns].itertuples(index=False, name=None)):
            # Pre-open messages also contribute to the current full book.
            book.process_message(dict(zip(columns, values)))
            if not group_ends[position]:
                continue

            timestamp = book.current_time
            # skip if the timestamp doesn't lie in the selected time interval
            if not daily_start <= timestamp.time() < daily_end:
                continue

            # zero-price order (typically market order) cannot be used to compute mid-price
            best_bid = max(
                (order["price"] for order in book.buy_map.values() if order["price"] > 0),
                default=None,
            )
            best_ask = min(
                (order["price"] for order in book.sell_map.values() if order["price"] > 0),
                default=None,
            )
            if best_bid is None or best_ask is None:
                raise ValueError(
                    f"Missing positive quote at {timestamp}; "
                    f"raw message position {position} (zero-based); "
                    f"best_bid={best_bid}, best_ask={best_ask}."
                )

            # skip if best_bid > best_ask and record the timestamp and position
            if best_bid > best_ask:
                snapshot_issues.append({
                    "time": timestamp,
                    "message_position": position,
                    "reason": 1,  # Crossed
                })
                continue

            mid_price = (best_bid + best_ask) / 2
            lower_price = (1 - price_range) * mid_price
            upper_price = (1 + price_range) * mid_price
            selected_buy = {}
            selected_sell = {}
            for key, order in book.buy_map.items():
                if lower_price <= order["price"] <= upper_price:
                    selected_buy[key] = order
            for key, order in book.sell_map.items():
                if lower_price <= order["price"] <= upper_price:
                    selected_sell[key] = order

            buy_prices = {order["price"] for order in selected_buy.values()}
            sell_prices = {order["price"] for order in selected_sell.values()}
            # Skip and record snapshots with too few distinct prices on either side.
            if len(buy_prices) < min_levels or len(sell_prices) < min_levels:
                snapshot_issues.append({
                    "time": timestamp,
                    "message_position": position,
                    "reason": 2,  # Insufficient distinct levels in the price band.
                })
                continue

            # skip if the current timestamp has totally the same market state as the previous saved timestamp
            if selected_buy == previous_buy and selected_sell == previous_sell:
                continue


            previous_buy = {key: order.copy() for key, order in selected_buy.items()}
            previous_sell = {key: order.copy() for key, order in selected_sell.items()}
            yield {
                "time": timestamp,
                "mid_price": mid_price,
                "buy_map": previous_buy,
                "sell_map": previous_sell,
                "message_position": position,  
            }

    return iter_orderbooks(), book.issues, snapshot_issues


def extract_daily_nlevels(orderbooks, n=10):
    """Read an order-book iterator and return snapshot_info and features.

    snapshot_info contains time and mid_price, which is used to calculate labels.
    Each feature row contains [ask_price, ask_volume, bid_price, bid_volume]
    for levels 1 through n and matches one snapshot_info row. The feature
    array has shape (number of snapshots, 4 * n).
    Use the same n as build_daily_orderbooks' min_levels.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")

    snapshot_info_rows = []
    feature_rows = []
    for orderbook in orderbooks:
        # Sum displayed volumes at each price across order types.
        buy_volumes = {}
        sell_volumes = {}
        for order in orderbook["buy_map"].values():
            price = order["price"]
            buy_volumes[price] = buy_volumes.get(price, 0) + order["volume"]
        for order in orderbook["sell_map"].values():
            price = order["price"]
            sell_volumes[price] = sell_volumes.get(price, 0) + order["volume"]

        buy_prices = sorted(buy_volumes, reverse=True)
        sell_prices = sorted(sell_volumes)
        feature_row = []
        for level in range(n):
            ask_price = sell_prices[level]
            bid_price = buy_prices[level]
            feature_row.extend([
                ask_price, sell_volumes[ask_price],
                bid_price, buy_volumes[bid_price],
            ])

        snapshot_info_rows.append((orderbook["time"], orderbook["mid_price"]))
        feature_rows.append(feature_row)

    snapshot_info = pd.DataFrame(snapshot_info_rows, columns=["time", "mid_price"])
    snapshot_info = snapshot_info.astype({
        "time": "datetime64[ns, Europe/Warsaw]", "mid_price": "float64"
    })
    features = np.asarray(feature_rows, dtype=np.float64)

    return snapshot_info, features


def orderbook_to_ofn(orderbook, costs):
    """Return a flat float64 feature array for one book and its cost limits.

    costs are allowed average price deviations from mid-price: 0.01 means 1%.
    Each cost gives [minimum average sell price, -sell volume / 1e6,
    maximum average buy price, buy volume / 1e6].
    """
    costs = np.asarray(costs, dtype=np.float64)
    mid_price = orderbook["mid_price"]
    features = np.empty((len(costs), 4), dtype=np.float64)
    features[:, 0] = (1 - costs) * mid_price
    features[:, 2] = (1 + costs) * mid_price

    # OFN reference: https://doi.org/10.1016/j.asoc.2024.111555 (Sections 4.2 and 5.2).
    # selling uses buy orders (f is negative); buying uses sell orders (g is positive).
    for side in ("buy", "sell"):
        # Add the volumes of orders at the same price, including different order types.
        volumes_by_price = {}
        for order in orderbook[f"{side}_map"].values():
            price = order["price"]
            volumes_by_price[price] = volumes_by_price.get(price, 0) + order["volume"]

        # Buy orders: highest price first. Sell orders: lowest price first.
        # Use all price levels supplied by build_daily_orderbooks.
        prices = np.asarray(sorted(volumes_by_price, reverse=side == "buy"), dtype=np.float64)
        volumes = np.asarray([volumes_by_price[price] for price in prices], dtype=np.float64)
        cumulative_volume = np.cumsum(volumes)
        # Entry i is the average price after trading all shares in levels 0 through i.
        cumulative_average_price = np.cumsum(prices * volumes) / cumulative_volume

        # Count how many levels can be completely traded for each cost limit.
        # side="right" includes a level whose cumulative average equals the limit.
        if side == "buy":
            price_limits = features[:, 0]
            # Count average prices >= the minimum allowed selling price.
            # Negate both arrays because searchsorted requires increasing order.
            num_full_levels = np.searchsorted(-cumulative_average_price, -price_limits, side="right")
        else:
            price_limits = features[:, 2]
            # Count average prices <= the maximum allowed buying price.
            num_full_levels = np.searchsorted(cumulative_average_price, price_limits, side="right")

        # Zero levels: nothing can be traded. All levels: use the total volume.
        max_trade_volume = np.zeros(len(costs), dtype=np.float64)
        max_trade_volume[num_full_levels == len(prices)] = cumulative_volume[-1]

        # For the other costs, trade all shares through the last full level, then
        # part of the next level so the overall average price equals the limit.
        # Array indices start at zero: count - 1 is the last full level's index.
        needs_partial_calculation = (num_full_levels > 0) & (num_full_levels < len(prices))
        last_full_level_index = num_full_levels[needs_partial_calculation] - 1
        next_level_index = num_full_levels[needs_partial_calculation]
        max_trade_volume[needs_partial_calculation] = cumulative_volume[last_full_level_index] * (
            1 + (cumulative_average_price[last_full_level_index] - price_limits[needs_partial_calculation])
            / (price_limits[needs_partial_calculation] - prices[next_level_index])
        )

        if side == "buy":
            features[:, 1] = -max_trade_volume / 1e6
        else:
            features[:, 3] = max_trade_volume / 1e6

    return features.ravel()


def extract_daily_ofn(orderbooks, max_cost=0.035):
    """Return snapshot_info and raw float64 (N, 40) OFN features from an iterator."""
    # costs are r_i * r_max in the paper, we reproduce the paper's r_i by the following
    # formula. Note that we valuate f and g at r_i * r_max, instead of r_i as in the paper, for consistency.
    cost_fractions = (11.0 ** (np.arange(1, 11) / 10) - 1) / 10
    costs = cost_fractions * max_cost

    snapshot_info_rows = []
    feature_rows = []
    for orderbook in orderbooks:
        feature_rows.append(orderbook_to_ofn(orderbook, costs))
        snapshot_info_rows.append((orderbook["time"], orderbook["mid_price"]))

    # Keep every supplied snapshot, including repeated OFN rows, so existing
    # times, mid-prices and labels remain aligned with the same sampling rules.
    snapshot_info = pd.DataFrame(snapshot_info_rows, columns=["time", "mid_price"])
    snapshot_info = snapshot_info.astype({
        "time": "datetime64[ns, Europe/Warsaw]", "mid_price": "float64"
    })
    features = np.asarray(feature_rows, dtype=np.float64)
    return snapshot_info, features


def compute_daily_statistics(npz_path):
    """Read one processed day and return its snapshot count, column means and stds."""
    with np.load(npz_path, allow_pickle=False) as arrays:
        features = arrays["features"]

    # Weight saved snapshots equally and compute statistics separately for each column.
    count = len(features)
    mean = features.mean(axis=0, dtype=np.float64)
    std = features.std(axis=0, ddof=0, dtype=np.float64)
    return count, mean, std


def compute_history_statistics(daily_statistics):
    """Combine preceding (five) days' (count, mean, std) tuples into column means and stds."""
    daily_counts = []
    daily_means = []
    daily_stds = []
    for count, mean, std in daily_statistics:
        daily_counts.append(count)
        daily_means.append(mean)
        daily_stds.append(std)

    counts = np.asarray(daily_counts, dtype=np.int64)
    means = np.asarray(daily_means, dtype=np.float64)
    stds = np.asarray(daily_stds, dtype=np.float64)
    snapshot_count = int(counts.sum())

    mean = (counts[:, None] * means).sum(axis=0) / snapshot_count
    variance = (
        counts[:, None] * (stds ** 2 + (means - mean) ** 2)
    ).sum(axis=0) / snapshot_count
    std = np.sqrt(variance)
    return mean, std


def normalize_features(features, mean, std):
    # Broadcast the historical column statistics across all rows of this day.
    return (features - mean) / std


def compute_labels(mid_price, k, alpha):
    """Compare past/current and future mean prices; return aligned int64 labels."""
    if k < 1:
        raise ValueError("k must be positive.")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative.")
    # Incomplete intervals have label -1 and cannot be used as prediction targets.
    labels = np.full(len(mid_price), -1, dtype=np.int64)
    positions = np.arange(k, len(mid_price) - k)
    cumulative_price = np.concatenate(([0.0], np.cumsum(mid_price, dtype=np.float64)))

    # The past/current interval has k+1 prices; the future interval has k prices.
    past_mean = (
        cumulative_price[positions + 1] - cumulative_price[positions - k]
    ) / (k + 1)
    future_mean = (
        cumulative_price[positions + k + 1] - cumulative_price[positions + 1]
    ) / k
    change = (future_mean - past_mean) / past_mean

    labels[positions] = 1
    labels[positions[change < -alpha]] = 0
    labels[positions[change > alpha]] = 2
    return labels
