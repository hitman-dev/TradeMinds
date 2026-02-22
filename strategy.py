"""
Professional Intraday Strategy — Multi-Indicator Confluence

Indicators
----------
  VWAP + ±1σ / ±2σ bands     — intraday price anchor (daily reset)
  Volume Profile (daily)      — POC, VAH, VAL from previous session
  EMA 9 / 21 / 50             — trend stack
  Supertrend (10, 2.5)        — trend direction & trailing stop
  RSI (14)                    — momentum
  ATR (14)                    — volatility / stop sizing
  Volume MA (20)              — volume context

Signals
-------
  EMA 9/21 crossover is the PRIMARY TRIGGER.
  Each additional confirmation adds +1 to a confluence score.
  Trade fires when score >= min_score (default 1 → many trades).

  Extra signal: VWAP-band mean reversion
    Long  when price touches VWAP -2σ band (oversold)
    Short when price touches VWAP +2σ band (overbought)
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Low-level indicator helpers
# ──────────────────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _supertrend(df: pd.DataFrame, period: int = 10, mult: float = 2.5):
    """Returns (line, direction)  direction: 1=bull, -1=bear"""
    atr     = _atr(df, period)
    hl2     = (df["high"] + df["low"]) / 2
    up_raw  = (hl2 + mult * atr).values
    lo_raw  = (hl2 - mult * atr).values
    cl      = df["close"].values
    n       = len(df)
    fu = up_raw.copy(); fl = lo_raw.copy()
    st = np.full(n, np.nan); di = np.zeros(n, dtype=int)

    for i in range(1, n):
        if np.isnan(atr.iloc[i]):
            continue
        fu[i] = up_raw[i] if up_raw[i] < fu[i-1] or cl[i-1] > fu[i-1] else fu[i-1]
        fl[i] = lo_raw[i] if lo_raw[i] > fl[i-1] or cl[i-1] < fl[i-1] else fl[i-1]
        prev = st[i-1]
        if np.isnan(prev):
            di[i], st[i] = 1, fl[i]
        elif abs(prev - fu[i-1]) < 1e-9:       # was bearish
            if cl[i] > fu[i]:   di[i], st[i] = 1, fl[i]
            else:               di[i], st[i] = -1, fu[i]
        else:                                   # was bullish
            if cl[i] < fl[i]:   di[i], st[i] = -1, fu[i]
            else:               di[i], st[i] = 1, fl[i]
    return pd.Series(st, index=df.index), pd.Series(di, index=df.index)


def _vwap_bands(df: pd.DataFrame):
    """
    Session VWAP with ±1σ and ±2σ bands (daily reset).
    Returns vwap, upper1, lower1, upper2, lower2 as pd.Series.
    """
    df    = df.copy()
    dates = pd.to_datetime(df["timestamp"]).dt.date
    tp    = (df["high"] + df["low"] + df["close"]) / 3
    vol   = df["volume"]

    vwap_arr  = np.full(len(df), np.nan)
    upper1    = np.full(len(df), np.nan)
    lower1    = np.full(len(df), np.nan)
    upper2    = np.full(len(df), np.nan)
    lower2    = np.full(len(df), np.nan)

    for d in sorted(dates.unique()):
        mask = (dates == d).values
        idx  = np.where(mask)[0]
        tp_d = tp.values[mask]
        v_d  = vol.values[mask].astype(float)

        cum_tv  = np.cumsum(tp_d * v_d)
        cum_v   = np.cumsum(v_d)
        cum_v   = np.where(cum_v == 0, np.nan, cum_v)
        vw      = cum_tv / cum_v

        # Rolling variance of (tp - vwap)² weighted by volume
        var_arr = np.full(len(idx), np.nan)
        for j in range(1, len(idx)):
            weights = v_d[:j+1]
            wsum    = weights.sum()
            if wsum == 0:
                continue
            dev2   = (tp_d[:j+1] - vw[j]) ** 2
            var_arr[j] = (weights * dev2).sum() / wsum

        sd = np.sqrt(np.where(np.isnan(var_arr), 0, var_arr))

        for j, gi in enumerate(idx):
            vwap_arr[gi] = vw[j]
            upper1[gi]   = vw[j] + 1 * sd[j]
            lower1[gi]   = vw[j] - 1 * sd[j]
            upper2[gi]   = vw[j] + 2 * sd[j]
            lower2[gi]   = vw[j] - 2 * sd[j]

    idx_s = df.index
    return (
        pd.Series(vwap_arr, index=idx_s),
        pd.Series(upper1,   index=idx_s),
        pd.Series(lower1,   index=idx_s),
        pd.Series(upper2,   index=idx_s),
        pd.Series(lower2,   index=idx_s),
    )


def _daily_volume_profile(df: pd.DataFrame, bins: int = 80):
    """
    Compute daily Volume Profile → POC, VAH, VAL.
    Each row gets the *previous* day's levels (standard institutional approach).

    Returns poc, vah, val as pd.Series (NaN for first day's candles).
    """
    df    = df.copy()
    dates = pd.to_datetime(df["timestamp"]).dt.date
    day_levels: dict = {}

    for d in sorted(dates.unique()):
        day_df = df[dates == d]
        lo_d   = float(day_df["low"].min())
        hi_d   = float(day_df["high"].max())
        if hi_d - lo_d < 1e-6 or len(day_df) < 3:
            continue

        bin_sz = (hi_d - lo_d) / bins
        vol_bin: dict = {}

        for _, row in day_df.iterrows():
            c_lo = float(row["low"])
            c_hi = float(row["high"])
            vol  = float(row["volume"])
            b_lo = int((c_lo - lo_d) / bin_sz)
            b_hi = int((c_hi - lo_d) / bin_sz) + 1
            n_b  = max(b_hi - b_lo, 1)
            vb   = vol / n_b
            for b in range(b_lo, min(b_hi, bins)):
                vol_bin[b] = vol_bin.get(b, 0.0) + vb

        if not vol_bin:
            continue

        # POC
        poc_b     = max(vol_bin, key=vol_bin.get)
        poc_price = lo_d + (poc_b + 0.5) * bin_sz

        # Value Area (70%)
        total_vol = sum(vol_bin.values())
        target    = total_vol * 0.70
        bins_list = sorted(vol_bin.keys())
        if poc_b not in bins_list:
            continue
        poc_idx   = bins_list.index(poc_b)

        va_set  = {poc_b}
        va_vol  = vol_bin[poc_b]
        lo_ptr  = poc_idx - 1
        hi_ptr  = poc_idx + 1

        while va_vol < target:
            lo_v = vol_bin.get(bins_list[lo_ptr], 0) if lo_ptr >= 0 else 0
            hi_v = vol_bin.get(bins_list[hi_ptr], 0) if hi_ptr < len(bins_list) else 0
            if lo_v == 0 and hi_v == 0:
                break
            if hi_v >= lo_v:
                va_set.add(bins_list[hi_ptr]); va_vol += hi_v; hi_ptr += 1
            else:
                va_set.add(bins_list[lo_ptr]); va_vol += lo_v; lo_ptr -= 1

        vah_price = lo_d + (max(va_set) + 1) * bin_sz
        val_price = lo_d + min(va_set) * bin_sz

        day_levels[d] = {"poc": poc_price, "vah": vah_price, "val": val_price}

    # Assign previous day's levels to each candle row
    poc_s = pd.Series(np.nan, index=df.index)
    vah_s = pd.Series(np.nan, index=df.index)
    val_s = pd.Series(np.nan, index=df.index)
    prev  = None
    for d in sorted(dates.unique()):
        mask = (dates == d).values
        if prev:
            poc_s.values[mask] = prev["poc"]
            vah_s.values[mask] = prev["vah"]
            val_s.values[mask] = prev["val"]
        if d in day_levels:
            prev = day_levels[d]

    return poc_s, vah_s, val_s, day_levels


# ──────────────────────────────────────────────────────────────────────────────
#  Main signal generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_signals(
    df: pd.DataFrame,
    direction: str        = "both",

    # ── Confluece scoring ────────────────────────────────────────────────────
    min_score: int        = 1,    # 0 = every crossover, 3 = pro-grade quality

    # ── Extra signal types ───────────────────────────────────────────────────
    vwap_reversion: bool  = True,  # enter at ±2σ VWAP band extremes

    # ── Optional hard gates (block trade even if score passes) ───────────────
    require_supertrend: bool = False,   # ST must agree
    require_vwap_side:  bool = False,   # price above/below VWAP
    require_ema_stack:  bool = False,   # EMA 9 > 21 > 50
    use_time_filter:    bool = False,
    time_start:         str  = "09:45",
    time_end:           str  = "14:30",

    # ── Indicator params ─────────────────────────────────────────────────────
    ema_fast:   int   = 9,
    ema_slow:   int   = 21,
    ema_trend:  int   = 50,
    st_period:  int   = 10,
    st_mult:    float = 2.5,
    rsi_period: int   = 14,
    atr_period: int   = 14,
) -> pd.DataFrame:
    """
    Compute all indicators and generate signals.

    Columns added to returned DataFrame
    ------------------------------------
    ema_fast, ema_slow, ema_trend  — EMA lines
    supertrend, st_direction       — Supertrend line + direction
    rsi, atr                       — RSI, ATR
    vwap, vwap_u1, vwap_l1,
    vwap_u2, vwap_l2               — VWAP + bands
    poc, vah, val                  — Previous-day volume profile levels
    vol_ma                         — 20-bar volume average
    signal_score                   — Confluence score at entry
    signal                         — 1 (long) / -1 (short) / 0
    entry_sl                       — ATR-based stop loss
    entry_zone_high/low            — required by backtester
    """
    df    = df.copy()
    close = df["close"]

    # ── Compute indicators ───────────────────────────────────────────────────
    ef           = _ema(close, ema_fast)
    es           = _ema(close, ema_slow)
    et           = _ema(close, ema_trend)
    rsi_s        = _rsi(close, rsi_period)
    atr_s        = _atr(df, atr_period)
    st_line, st_di = _supertrend(df, st_period, st_mult)
    vol_ma       = df["volume"].rolling(20).mean()

    vwap, vu1, vl1, vu2, vl2 = _vwap_bands(df)
    poc, vah, val, _          = _daily_volume_profile(df)

    # ── Attach columns for charting ──────────────────────────────────────────
    df["ema_fast"]     = ef
    df["ema_slow"]     = es
    df["ema_trend"]    = et
    df["rsi"]          = rsi_s
    df["atr"]          = atr_s
    df["supertrend"]   = st_line
    df["st_direction"] = st_di
    df["vwap"]         = vwap
    df["vwap_u1"]      = vu1
    df["vwap_l1"]      = vl1
    df["vwap_u2"]      = vu2
    df["vwap_l2"]      = vl2
    df["poc"]          = poc
    df["vah"]          = vah
    df["val"]          = val
    df["vol_ma"]       = vol_ma

    # ── EMA crossover detection ──────────────────────────────────────────────
    cross_up = (ef > es) & (ef.shift(1) <= es.shift(1))
    cross_dn = (ef < es) & (ef.shift(1) >= es.shift(1))

    # ── Initialise output columns ────────────────────────────────────────────
    df["signal"]          = 0
    df["signal_score"]    = 0
    df["entry_sl"]        = np.nan
    df["entry_zone_high"] = np.nan
    df["entry_zone_low"]  = np.nan

    warmup = max(ema_trend, 50) + 2

    for i in range(warmup, len(df)):
        row     = df.iloc[i]
        ts      = row["timestamp"]
        cur_cl  = float(row["close"])
        cur_atr = float(atr_s.iloc[i])
        cur_rsi = float(rsi_s.iloc[i])
        cur_st  = int(st_di.iloc[i])
        cur_vwap= float(vwap.iloc[i])
        cur_vu2 = float(vu2.iloc[i])
        cur_vl2 = float(vl2.iloc[i])
        cur_poc = float(poc.iloc[i]) if not np.isnan(poc.iloc[i]) else np.nan
        cur_vah = float(vah.iloc[i]) if not np.isnan(vah.iloc[i]) else np.nan
        cur_val = float(val.iloc[i]) if not np.isnan(val.iloc[i]) else np.nan
        cur_vol = float(row["volume"])
        cur_vm  = float(vol_ma.iloc[i]) if not np.isnan(vol_ma.iloc[i]) else 1.0

        if pd.isna(cur_atr) or cur_atr <= 0:
            continue

        # ── Time filter ──────────────────────────────────────────────────────
        if use_time_filter:
            hm = pd.Timestamp(ts).strftime("%H:%M")
            if not (time_start <= hm <= time_end):
                continue

        idx = df.index[i]

        # ════════════════════════════════════════════════════════════════════
        # SIGNAL A — EMA crossover with confluence score
        # ════════════════════════════════════════════════════════════════════
        for sig_dir, is_cross, st_need, rsi_lo, rsi_hi, vwap_ok in [
            ( 1, cross_up.iloc[i],  1, 40, 72, cur_cl > cur_vwap),
            (-1, cross_dn.iloc[i], -1, 28, 60, cur_cl < cur_vwap),
        ]:
            if direction == "long"  and sig_dir == -1: continue
            if direction == "short" and sig_dir ==  1: continue
            if not is_cross: continue

            # ── Hard gates ───────────────────────────────────────────────────
            if require_supertrend and cur_st != st_need:       continue
            if require_vwap_side  and not vwap_ok:             continue
            if require_ema_stack:
                ef_v, es_v, et_v = float(ef.iloc[i]), float(es.iloc[i]), float(et.iloc[i])
                if sig_dir == 1  and not (ef_v > es_v > et_v): continue
                if sig_dir == -1 and not (ef_v < es_v < et_v): continue

            # ── Confluence score ─────────────────────────────────────────────
            score = 0
            if cur_st == st_need:                         score += 1  # Supertrend aligned
            if vwap_ok:                                   score += 1  # VWAP side
            if rsi_lo <= cur_rsi <= rsi_hi:               score += 1  # RSI range
            if cur_vol > 1.3 * cur_vm:                    score += 1  # Volume surge
            # EMA full stack
            ef_v, es_v, et_v = float(ef.iloc[i]), float(es.iloc[i]), float(et.iloc[i])
            if sig_dir == 1  and ef_v > es_v > et_v:     score += 1
            if sig_dir == -1 and ef_v < es_v < et_v:     score += 1
            # VP level alignment
            if not np.isnan(cur_poc):
                if sig_dir == 1  and cur_cl > cur_poc:    score += 1
                if sig_dir == -1 and cur_cl < cur_poc:    score += 1
            if not np.isnan(cur_vah) and not np.isnan(cur_val):
                # Breakout above VAH or below VAL = extra conviction
                if sig_dir == 1  and cur_cl > cur_vah:   score += 1
                if sig_dir == -1 and cur_cl < cur_val:    score += 1

            if score < min_score:
                continue

            # ── Entry ────────────────────────────────────────────────────────
            sl = cur_cl - 1.5 * cur_atr * sig_dir   # below for long, above for short
            df.at[idx, "signal"]          = sig_dir
            df.at[idx, "signal_score"]    = score
            df.at[idx, "entry_sl"]        = sl
            df.at[idx, "entry_zone_high"] = cur_cl if sig_dir == 1 else sl
            df.at[idx, "entry_zone_low"]  = sl     if sig_dir == 1 else cur_cl
            break   # one signal per candle

        if df.at[idx, "signal"] != 0:
            continue   # already have a signal this candle

        # ════════════════════════════════════════════════════════════════════
        # SIGNAL B — VWAP band mean reversion
        #   Long  : price touches -2σ band, RSI < 40 (oversold extreme)
        #   Short : price touches +2σ band, RSI > 60 (overbought extreme)
        # ════════════════════════════════════════════════════════════════════
        if vwap_reversion and not pd.isna(cur_vl2) and not pd.isna(cur_vu2):
            rev_long  = (direction in ("long",  "both") and float(row["low"])  <= cur_vl2 and cur_rsi < 40)
            rev_short = (direction in ("short", "both") and float(row["high"]) >= cur_vu2 and cur_rsi > 60)

            if rev_long:
                sl = cur_cl - 1.5 * cur_atr
                df.at[idx, "signal"]          = 1
                df.at[idx, "signal_score"]    = -1   # mark as reversion trade
                df.at[idx, "entry_sl"]        = sl
                df.at[idx, "entry_zone_high"] = cur_cl
                df.at[idx, "entry_zone_low"]  = sl
            elif rev_short:
                sl = cur_cl + 1.5 * cur_atr
                df.at[idx, "signal"]          = -1
                df.at[idx, "signal_score"]    = -1
                df.at[idx, "entry_sl"]        = sl
                df.at[idx, "entry_zone_high"] = sl
                df.at[idx, "entry_zone_low"]  = cur_cl

    return df
