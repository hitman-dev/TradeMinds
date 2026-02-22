"""
Professional Trade Simulation Engine

Key improvements over basic backtester:
  - Tight stop: uses breakout candle low/high instead of zone boundary
  - Partial exit at 1R: books 50% at 1R, moves stop to breakeven for rest
    → dramatically improves win rate (trade is only a loss if hit before 1R)
  - Trailing stop: optional, runs after partial exit
  - Correct P&L with partial_lots tracking
"""

import pandas as pd
import numpy as np


def _to_date(ts):
    return pd.Timestamp(ts).date()


def run_backtest(
    df: pd.DataFrame,
    rr_ratio: float = 2.0,
    intraday_mode: bool = True,
    trailing_stop: bool = False,
    lot_size: int = 1,
    use_tight_stop: bool = True,
    partial_exit_r: float = 1.0,
    partial_exit_pct: float = 0.5,
) -> pd.DataFrame:
    """
    Simulate trades on signals from strategy.generate_signals().

    Args:
        df:               DataFrame with signal, entry_zone_high/low, entry_sl columns
        rr_ratio:         Full target = entry ± rr_ratio × stop_distance
        intraday_mode:    Force-close open trades at day end
        trailing_stop:    Trail SL at 1 stop_distance below highest high (long) after 1R
        lot_size:         Shares per trade
        use_tight_stop:   Use entry_sl (breakout candle low/high) instead of zone boundary
        partial_exit_r:   Take partial exit at this R-multiple (default 1.0 = 1R)
        partial_exit_pct: Fraction of position to exit at partial target (default 0.5 = 50%)

    Win/Loss classification (with partial exit enabled):
        WIN      → trade reached partial_exit_r OR full target was hit
        LOSS     → stop hit BEFORE reaching partial_exit_r
        BREAKEVEN→ exit at zero P&L (e.g., stop moved to breakeven after partial)

    Returns:
        pd.DataFrame with columns:
            trade_id, direction, entry_time, exit_time,
            entry_price, exit_price, stop_loss, target,
            pnl, pnl_pct, result, exit_reason, partial_exit
    """
    trades      = []
    trade_id    = 0
    open_trade  = None

    for i in range(len(df)):
        row          = df.iloc[i]
        current_time = row["timestamp"]
        current_date = _to_date(current_time)

        # ── Manage open trade ────────────────────────────────────────────────
        if open_trade is not None:
            direction    = open_trade["direction"]
            hi           = float(row["high"])
            lo           = float(row["low"])
            sl           = open_trade["stop_loss"]
            target       = open_trade["target"]
            entry        = open_trade["entry_price"]
            stop_dist    = open_trade["stop_distance"]
            partial_done = open_trade["partial_done"]

            # ── Partial exit check ───────────────────────────────────────────
            if not partial_done and partial_exit_pct > 0 and partial_exit_r > 0:
                partial_trigger = (
                    entry + partial_exit_r * stop_dist if direction == 1
                    else entry - partial_exit_r * stop_dist
                )
                hit_partial = (hi >= partial_trigger) if direction == 1 else (lo <= partial_trigger)

                if hit_partial:
                    partial_lots    = open_trade["remaining_lots"] * partial_exit_pct
                    remaining_lots  = open_trade["remaining_lots"] * (1 - partial_exit_pct)
                    partial_pnl     = (partial_trigger - entry) * direction * partial_lots

                    open_trade["partial_done"]    = True
                    open_trade["remaining_lots"]  = remaining_lots
                    open_trade["partial_pnl"]     = partial_pnl
                    open_trade["partial_price"]   = partial_trigger
                    # Move stop to breakeven after partial exit
                    open_trade["stop_loss"] = entry
                    sl = entry

            # ── Trailing stop ────────────────────────────────────────────────
            if trailing_stop:
                if direction == 1:
                    new_sl = max(sl, hi - stop_dist)
                    if new_sl > sl:
                        open_trade["stop_loss"] = new_sl
                        sl = new_sl
                else:
                    new_sl = min(sl, lo + stop_dist)
                    if new_sl < sl:
                        open_trade["stop_loss"] = new_sl
                        sl = new_sl

            # ── Full exit check ──────────────────────────────────────────────
            exit_price  = None
            exit_reason = None

            if direction == 1:
                if lo <= sl:
                    exit_price  = sl
                    exit_reason = "stop_loss"
                elif hi >= target:
                    exit_price  = target
                    exit_reason = "target"
            else:
                if hi >= sl:
                    exit_price  = sl
                    exit_reason = "stop_loss"
                elif lo <= target:
                    exit_price  = target
                    exit_reason = "target"

            # Intraday: force-exit at last candle of the day
            if intraday_mode and exit_price is None:
                is_last = (i == len(df) - 1) or (
                    _to_date(df.iloc[i + 1]["timestamp"]) != current_date
                )
                if is_last:
                    exit_price  = float(row["close"])
                    exit_reason = "eod_exit"

            if exit_price is not None:
                ep  = open_trade["entry_price"]
                rem = open_trade["remaining_lots"]

                remaining_pnl      = (exit_price - ep) * direction * rem
                partial_pnl_booked = open_trade.get("partial_pnl", 0.0)
                total_pnl          = remaining_pnl + partial_pnl_booked
                pnl_pct            = total_pnl / (ep * lot_size) * 100 if ep > 0 else 0.0

                # Win classification: only a LOSS if stop hit before any partial exit
                if exit_reason == "stop_loss" and not open_trade["partial_done"]:
                    result = "loss"
                elif total_pnl > 0.01:
                    result = "win"
                elif total_pnl < -0.01:
                    result = "loss"
                else:
                    result = "breakeven"

                trades.append({
                    "trade_id":    open_trade["trade_id"],
                    "direction":   "long" if direction == 1 else "short",
                    "entry_time":  open_trade["entry_time"],
                    "exit_time":   current_time,
                    "entry_price": round(ep, 2),
                    "exit_price":  round(exit_price, 2),
                    "stop_loss":   round(open_trade["initial_sl"], 2),
                    "target":      round(target, 2),
                    "pnl":         round(total_pnl, 2),
                    "pnl_pct":     round(pnl_pct, 3),
                    "result":      result,
                    "exit_reason": exit_reason,
                    "partial_exit": open_trade["partial_done"],
                })
                open_trade = None

        # ── New signal ────────────────────────────────────────────────────────
        if open_trade is None and row["signal"] != 0:
            direction   = int(row["signal"])
            entry_price = float(row["close"])
            zh          = row["entry_zone_high"]
            zl          = row["entry_zone_low"]

            if pd.isna(zh) or pd.isna(zl):
                continue

            zh = float(zh)
            zl = float(zl)

            # Determine stop loss: tight (breakout candle) or zone boundary
            if use_tight_stop and "entry_sl" in df.columns and not pd.isna(row["entry_sl"]):
                stop_loss = float(row["entry_sl"])
            else:
                stop_loss = zl if direction == 1 else zh

            if direction == 1:
                stop_distance = entry_price - stop_loss
            else:
                stop_distance = stop_loss - entry_price

            if stop_distance <= 0:
                continue   # entry inside or below zone; skip

            target = (
                entry_price + rr_ratio * stop_distance
                if direction == 1
                else entry_price - rr_ratio * stop_distance
            )

            trade_id += 1
            open_trade = {
                "trade_id":      trade_id,
                "direction":     direction,
                "entry_time":    current_time,
                "entry_price":   entry_price,
                "stop_loss":     stop_loss,
                "initial_sl":    stop_loss,
                "target":        target,
                "stop_distance": stop_distance,
                "partial_done":  False,
                "remaining_lots": float(lot_size),
                "partial_pnl":   0.0,
            }

    # ── Force-close any trade open at end of data ─────────────────────────────
    if open_trade is not None:
        last      = df.iloc[-1]
        ep        = open_trade["entry_price"]
        exit_price= float(last["close"])
        direction = open_trade["direction"]
        rem       = open_trade["remaining_lots"]

        remaining_pnl      = (exit_price - ep) * direction * rem
        partial_pnl_booked = open_trade.get("partial_pnl", 0.0)
        total_pnl          = remaining_pnl + partial_pnl_booked
        pnl_pct            = total_pnl / (ep * lot_size) * 100 if ep > 0 else 0.0

        result = "win" if total_pnl > 0.01 else ("loss" if total_pnl < -0.01 else "breakeven")

        trades.append({
            "trade_id":    open_trade["trade_id"],
            "direction":   "long" if direction == 1 else "short",
            "entry_time":  open_trade["entry_time"],
            "exit_time":   last["timestamp"],
            "entry_price": round(ep, 2),
            "exit_price":  round(exit_price, 2),
            "stop_loss":   round(open_trade["initial_sl"], 2),
            "target":      round(open_trade["target"], 2),
            "pnl":         round(total_pnl, 2),
            "pnl_pct":     round(pnl_pct, 3),
            "result":      result,
            "exit_reason": "end_of_data",
            "partial_exit": open_trade["partial_done"],
        })

    if not trades:
        return pd.DataFrame(columns=[
            "trade_id", "direction", "entry_time", "exit_time",
            "entry_price", "exit_price", "stop_loss", "target",
            "pnl", "pnl_pct", "result", "exit_reason", "partial_exit",
        ])

    return pd.DataFrame(trades)


def calculate_metrics(trades_df: pd.DataFrame) -> dict:
    """Compute performance metrics from a trades DataFrame."""
    empty = {
        "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
        "net_pnl": 0.0, "avg_pnl": 0.0, "best_trade": 0.0,
        "worst_trade": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0,
        "wins": 0, "losses": 0, "gross_profit": 0.0, "gross_loss": 0.0,
        "partial_exits": 0,
    }
    if trades_df.empty:
        return empty

    total   = len(trades_df)
    wins    = int((trades_df["result"] == "win").sum())
    losses  = int((trades_df["result"] == "loss").sum())
    partials= int(trades_df.get("partial_exit", pd.Series([False] * total)).sum())

    gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
    gross_loss   = abs(float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()))
    profit_factor= (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    cumulative  = trades_df["pnl"].cumsum()
    max_dd      = float((cumulative - cumulative.cummax()).min())

    pnl_series  = trades_df["pnl_pct"]
    std         = float(pnl_series.std())
    sharpe      = float(pnl_series.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    return {
        "total_trades":  total,
        "win_rate":      round(wins / total * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "net_pnl":       round(float(trades_df["pnl"].sum()), 2),
        "avg_pnl":       round(float(trades_df["pnl"].mean()), 2),
        "best_trade":    round(float(trades_df["pnl"].max()), 2),
        "worst_trade":   round(float(trades_df["pnl"].min()), 2),
        "max_drawdown":  round(max_dd, 2),
        "sharpe_ratio":  round(sharpe, 2),
        "wins":          wins,
        "losses":        losses,
        "gross_profit":  round(gross_profit, 2),
        "gross_loss":    round(gross_loss, 2),
        "partial_exits": partials,
    }
