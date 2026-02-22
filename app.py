"""
Professional Intraday Backtester & Live Monitor — Streamlit App

Strategy: EMA 9/21 crossover + VWAP + Volume Profile + Supertrend + RSI
          Confluence scoring — minimum N confirmations required per trade

Tabs
----
  📡 Live      — auto-refreshing real-time chart & signal dashboard
  📊 Backtest  — historical simulation with full trade log
  📈 Metrics   — performance KPIs, drawdown, distribution charts
  🔍 Scanner   — scan Nifty 50 for current active setups

Run: pip install -r requirements.txt
     streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

from data_fetcher import GrowwDataFetcher
from strategy    import generate_signals
from backtester  import run_backtest, calculate_metrics

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pro Intraday Algo",
    page_icon="📡",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Pro Intraday Algo")
    st.caption("VWAP · Volume Profile · EMA · Supertrend · RSI")
    st.divider()

    # ── Data ─────────────────────────────────────────────────────────────────
    st.subheader("📅 Data")
    symbol = st.text_input("NSE Symbol", value="RELIANCE").upper().strip()

    INTERVALS = ["1min", "5min", "10min", "1hour", "4hours", "1day"]
    interval  = st.selectbox("Interval", INTERVALS, index=INTERVALS.index("10min"))
    intraday  = interval in ["1min", "5min", "10min"]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=date.today() - timedelta(days=60))
    with col2:
        end_date   = st.date_input("End",   value=date.today() - timedelta(days=1))

    st.divider()

    # ── Signal quality ────────────────────────────────────────────────────────
    st.subheader("🎯 Signal Quality")
    direction = st.radio("Direction", ["both", "long", "short"], horizontal=True)

    min_score = st.slider(
        "Min Confluence Score", 0, 5, 1,
        help=(
            "How many of 8 confirmations must align:\n"
            "  0–1 = max trades (all crossovers)\n"
            "  2–3 = balanced quality\n"
            "  4–5 = only top-tier setups\n\n"
            "Confirmations: Supertrend · VWAP side · RSI range · "
            "Volume surge · EMA stack · Price vs POC · Break VAH/VAL"
        ),
    )

    vwap_reversion = st.checkbox(
        "VWAP band reversion signals", value=True,
        help="Extra trades when price hits ±2σ VWAP band. "
             "Long at -2σ (oversold), Short at +2σ (overbought).",
    )

    st.markdown("**Hard gates** (block trade regardless of score)")
    req_st    = st.checkbox("Require Supertrend aligned", value=False)
    req_vwap  = st.checkbox("Require VWAP side correct",  value=False)
    req_stack = st.checkbox("Require full EMA stack",      value=False)
    time_gate = st.checkbox("Time filter (09:45–14:30)",   value=intraday)
    if time_gate:
        tc1, tc2 = st.columns(2)
        with tc1:  t_start = st.text_input("From", "09:45")
        with tc2:  t_end   = st.text_input("To",   "14:30")
    else:
        t_start, t_end = "09:15", "15:30"

    st.divider()

    # ── Trade management ─────────────────────────────────────────────────────
    st.subheader("📈 Trade Management")
    rr_ratio     = st.slider("Risk : Reward", 1.0, 5.0, 2.0, 0.5)
    partial      = st.checkbox("Partial exit at 1R", value=True,
                               help="Exit 50% at 1R → move stop to breakeven")
    p_pct        = st.slider("Partial %", 0.25, 0.75, 0.50, 0.05) if partial else 0.0
    p_r          = st.slider("Partial at R", 0.5, 2.0, 1.0, 0.25) if partial else 1.0
    trail        = st.checkbox("Trailing stop", value=False)
    lot_size     = st.number_input("Lot size", min_value=1, value=1, step=1)

    st.divider()

    # ── Live mode ─────────────────────────────────────────────────────────────
    st.subheader("📡 Live Mode")
    live_mode    = st.toggle("Enable auto-refresh", value=False)
    refresh_secs = st.slider("Refresh every (s)", 30, 300, 60, 10) if live_mode else 60
    if live_mode:
        st.success(f"Auto-refreshing every {refresh_secs}s")

    st.divider()
    run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_bt, tab_met, tab_scan = st.tabs(
    ["📡 Live", "📊 Backtest", "📈 Metrics", "🔍 Scanner"]
)


# ══════════════════════════════════════════════════════════════════════════════
#  Shared: fetch + compute (used by both Live and Backtest tabs)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_and_compute(sym, s_date, e_date, iv, msg=""):
    """Returns (df_with_signals, error_str | None)"""
    try:
        fetcher = GrowwDataFetcher()
        df = fetcher.fetch_candles(
            symbol=sym,
            start_date=str(s_date),
            end_date=str(e_date),
            interval=iv,
            exchange="NSE",
        )
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"API error: {e}"

    if df.empty:
        return None, "No candle data returned. Check symbol / date range / API token."

    df = generate_signals(
        df,
        direction=direction,
        min_score=min_score,
        vwap_reversion=vwap_reversion,
        require_supertrend=req_st,
        require_vwap_side=req_vwap,
        require_ema_stack=req_stack,
        use_time_filter=time_gate,
        time_start=t_start,
        time_end=t_end,
    )
    return df, None


def _build_chart(df, trades_df=None, title=""):
    """Build the full multi-pane Plotly chart."""
    # 3 rows: price (0.65), volume (0.15), RSI (0.20)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.13, 0.22],
        vertical_spacing=0.02,
    )

    # ── Candlestick ────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name=title or "Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # ── VWAP + bands ──────────────────────────────────────────────────────
    if "vwap" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["vwap"],
            name="VWAP", line=dict(color="#2196F3", width=1.8),
        ), row=1, col=1)
    if "vwap_u1" in df.columns and "vwap_l1" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["timestamp"], df["timestamp"].iloc[::-1]]),
            y=pd.concat([df["vwap_u1"], df["vwap_l1"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(33,150,243,0.06)",
            line=dict(color="rgba(33,150,243,0.25)", width=0.5),
            name="VWAP ±1σ", showlegend=True,
        ), row=1, col=1)
    if "vwap_u2" in df.columns and "vwap_l2" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["timestamp"], df["timestamp"].iloc[::-1]]),
            y=pd.concat([df["vwap_u2"], df["vwap_l2"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(33,150,243,0.03)",
            line=dict(color="rgba(33,150,243,0.15)", width=0.5),
            name="VWAP ±2σ", showlegend=True,
        ), row=1, col=1)

    # ── EMA lines ────────────────────────────────────────────────────────
    for col_n, lbl, clr, wid in [
        ("ema_fast",  "EMA 9",  "#64b5f6", 1.0),
        ("ema_slow",  "EMA 21", "#ffd54f", 1.2),
        ("ema_trend", "EMA 50", "#ff8a65", 1.8),
    ]:
        if col_n in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df[col_n],
                name=lbl, line=dict(color=clr, width=wid), opacity=0.8,
            ), row=1, col=1)

    # ── Supertrend ────────────────────────────────────────────────────────
    if "supertrend" in df.columns and "st_direction" in df.columns:
        for dv, nm, cl in [(1, "ST Bull", "#26a69a"), (-1, "ST Bear", "#ef5350")]:
            seg = df[df["st_direction"] == dv]
            if not seg.empty:
                fig.add_trace(go.Scatter(
                    x=seg["timestamp"], y=seg["supertrend"],
                    name=nm, mode="lines",
                    line=dict(color=cl, width=2, dash="dot"),
                    opacity=0.7,
                ), row=1, col=1)

    # ── Volume Profile levels (horizontal lines) ──────────────────────────
    last_row = df.dropna(subset=["poc"]).iloc[-1] if df["poc"].notna().any() else None
    if last_row is not None:
        poc_v = float(last_row["poc"])
        vah_v = float(last_row["vah"])
        val_v = float(last_row["val"])
        for yv, clr, lbl in [
            (poc_v, "rgba(255,235,59,0.9)",  f"POC {poc_v:.1f}"),
            (vah_v, "rgba(38,166,154,0.7)",  f"VAH {vah_v:.1f}"),
            (val_v, "rgba(239,83,80,0.7)",   f"VAL {val_v:.1f}"),
        ]:
            fig.add_hline(y=yv, line_dash="dash", line_color=clr,
                          annotation_text=lbl, annotation_position="right",
                          row=1, col=1)

    # ── Entry signal markers ──────────────────────────────────────────────
    longs  = df[df["signal"] == 1]
    shorts = df[df["signal"] == -1]
    if not longs.empty:
        fig.add_trace(go.Scatter(
            x=longs["timestamp"], y=longs["low"] * 0.9985,
            mode="markers", name="Long ▲",
            marker=dict(symbol="triangle-up", size=14,
                        color="#26a69a", line=dict(width=1, color="white")),
        ), row=1, col=1)
    if not shorts.empty:
        fig.add_trace(go.Scatter(
            x=shorts["timestamp"], y=shorts["high"] * 1.0015,
            mode="markers", name="Short ▼",
            marker=dict(symbol="triangle-down", size=14,
                        color="#ef5350", line=dict(width=1, color="white")),
        ), row=1, col=1)

    # ── Trade result lines ────────────────────────────────────────────────
    if trades_df is not None and not trades_df.empty:
        for _, t in trades_df.iterrows():
            clr = "#26a69a" if t["result"] == "win" else (
                  "#9e9e9e" if t["result"] == "breakeven" else "#ef5350")
            fig.add_trace(go.Scatter(
                x=[t["entry_time"], t["exit_time"]],
                y=[t["entry_price"], t["exit_price"]],
                mode="lines+markers",
                line=dict(color=clr, width=1.5, dash="dot"),
                marker=dict(size=5, color=clr),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)

    # ── Volume bars ───────────────────────────────────────────────────────
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(go.Bar(
        x=df["timestamp"], y=df["volume"],
        name="Volume", marker_color=vol_colors, opacity=0.6,
        showlegend=False,
    ), row=2, col=1)
    if "vol_ma" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["vol_ma"],
            name="Vol MA(20)", line=dict(color="#ffa726", width=1.2),
        ), row=2, col=1)

    # ── RSI ───────────────────────────────────────────────────────────────
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["rsi"],
            name="RSI(14)", line=dict(color="#b39ddb", width=1.3),
        ), row=3, col=1)
        for lvl, clr, lbl in [
            (70, "rgba(239,83,80,0.6)",  "70"),
            (50, "rgba(150,150,150,0.4)","50"),
            (30, "rgba(38,166,154,0.6)", "30"),
        ]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=clr,
                          annotation_text=lbl, row=3, col=1)

    fig.update_layout(
        height=720,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=11)),
    )
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Vol",        row=2, col=1)
    fig.update_yaxes(title_text="RSI",        row=3, col=1, range=[0, 100])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    live_end   = date.today()
    live_start = live_end - timedelta(days=5 if intraday else 30)

    hdr_col, status_col = st.columns([3, 1])
    with hdr_col:
        st.subheader(f"📡 Live — {symbol} ({interval})")
    with status_col:
        if live_mode:
            st.success("🟢 LIVE")
        else:
            if st.button("🔄 Refresh once"):
                st.rerun()

    with st.spinner("Loading latest data…"):
        df_live, err = _fetch_and_compute(symbol, live_start, live_end, interval)

    if err:
        st.error(err)
    elif df_live is not None:
        st.plotly_chart(_build_chart(df_live, title=symbol), width="stretch")

        # ── Current market status ──────────────────────────────────────────
        last = df_live.iloc[-1]
        st.divider()
        st.markdown("### 📊 Current Market Status")
        m1, m2, m3, m4, m5, m6 = st.columns(6)

        cur_close = float(last["close"])
        cur_vwap  = float(last.get("vwap", np.nan))
        cur_rsi   = float(last.get("rsi",  np.nan))
        cur_atr   = float(last.get("atr",  np.nan))
        cur_st    = int(last.get("st_direction", 0))
        cur_poc   = float(last["poc"]) if not np.isnan(last.get("poc", np.nan)) else None

        vwap_side = "Above VWAP 🟢" if cur_close > cur_vwap else "Below VWAP 🔴"
        st_status = "Bullish 🟢"    if cur_st ==  1 else ("Bearish 🔴" if cur_st == -1 else "—")

        m1.metric("Price",     f"₹{cur_close:,.2f}")
        m2.metric("VWAP",      f"₹{cur_vwap:,.2f}" if not np.isnan(cur_vwap) else "—", vwap_side)
        m3.metric("RSI",       f"{cur_rsi:.1f}"     if not np.isnan(cur_rsi)  else "—")
        m4.metric("ATR",       f"₹{cur_atr:.2f}"   if not np.isnan(cur_atr)  else "—")
        m5.metric("Supertrend", st_status)
        m6.metric("Prev POC",  f"₹{cur_poc:.2f}" if cur_poc else "—")

        # ── Recent signals ─────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🔔 Recent Signals")
        sigs = df_live[df_live["signal"] != 0].tail(10).copy()
        if sigs.empty:
            st.info(
                "No signals in the last 5 days. Try:\n"
                "- Lowering **Min Confluence Score** to 0\n"
                "- Disabling hard gates\n"
                "- Using a wider date range in Backtest"
            )
        else:
            sigs["Type"] = sigs["signal"].map({1: "🟢 LONG", -1: "🔴 SHORT"})
            sigs["Score"] = sigs["signal_score"].apply(
                lambda x: f"{x}/8" if x >= 0 else "Reversion"
            )
            sigs["entry_sl_disp"] = sigs["entry_sl"].apply(
                lambda x: f"₹{x:.2f}" if not pd.isna(x) else "—"
            )
            tbl = sigs[["timestamp", "Type", "close", "entry_sl_disp", "Score", "rsi", "vwap"]].rename(columns={
                "timestamp": "Time", "close": "Entry ₹",
                "entry_sl_disp": "Stop ₹", "rsi": "RSI", "vwap": "VWAP",
            })
            tbl["Time"] = tbl["Time"].astype(str)
            tbl["RSI"]  = tbl["RSI"].round(1)
            tbl["VWAP"] = tbl["VWAP"].round(2)

            def _sig_style(row):
                bg = "rgba(38,166,154,0.2)" if "LONG" in str(row.get("Type", "")) else "rgba(239,83,80,0.15)"
                return [f"background-color: {bg}"] * len(row)

            st.dataframe(tbl.style.apply(_sig_style, axis=1), width="stretch", hide_index=True)

        if live_mode:
            st.caption(f"Auto-refreshing every {refresh_secs}s · Last update: {pd.Timestamp.now().strftime('%H:%M:%S')}")
            st.markdown(
                f'<meta http-equiv="refresh" content="{refresh_secs}">',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_bt:
    if run_btn:
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            st.stop()

        with st.spinner(f"Fetching {symbol} {interval} ({start_date} → {end_date})…"):
            df_bt, err = _fetch_and_compute(symbol, start_date, end_date, interval)

        if err:
            st.error(err)
        elif df_bt is not None:
            st.success(f"Fetched **{len(df_bt):,}** candles")

            with st.spinner("Simulating trades…"):
                trades_df = run_backtest(
                    df_bt,
                    rr_ratio=rr_ratio,
                    intraday_mode=intraday,
                    trailing_stop=trail,
                    lot_size=lot_size,
                    use_tight_stop=True,
                    partial_exit_r=p_r,
                    partial_exit_pct=p_pct,
                )
                metrics = calculate_metrics(trades_df)

            # ── Stats bar ─────────────────────────────────────────────────
            n_long  = int((df_bt["signal"] == 1).sum())
            n_short = int((df_bt["signal"] == -1).sum())
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Signals",   n_long + n_short)
            c2.metric("Long",      n_long)
            c3.metric("Short",     n_short)
            c4.metric("Trades",    metrics["total_trades"])
            c5.metric("Win Rate",  f"{metrics['win_rate']}%")

            if n_long + n_short == 0:
                st.error(
                    "0 signals. Try:\n"
                    "- **Min Confluence Score = 0**\n"
                    "- Disable all Hard Gates\n"
                    "- 60+ days of data"
                )

            # ── Chart ─────────────────────────────────────────────────────
            st.plotly_chart(
                _build_chart(df_bt, trades_df if not trades_df.empty else None, symbol),
                width="stretch",
            )

            # ── Trade log ─────────────────────────────────────────────────
            if not trades_df.empty:
                cum_pnl    = trades_df["pnl"].cumsum()
                profitable = cum_pnl.iloc[-1] >= 0
                fig_cum = go.Figure(go.Scatter(
                    x=trades_df["exit_time"], y=cum_pnl,
                    fill="tozeroy",
                    fillcolor="rgba(38,166,154,0.15)" if profitable else "rgba(239,83,80,0.15)",
                    line=dict(color="#26a69a" if profitable else "#ef5350", width=2),
                ))
                fig_cum.update_layout(
                    height=200, template="plotly_dark",
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis_title="Cumulative P&L (₹)",
                )
                st.plotly_chart(fig_cum, width="stretch")

                d = trades_df.copy()
                d["entry_time"] = d["entry_time"].astype(str)
                d["exit_time"]  = d["exit_time"].astype(str)
                d["P&L"]  = d["pnl"].apply(lambda x: f"₹{x:+,.2f}")
                d["P&L%"] = d["pnl_pct"].apply(lambda x: f"{x:+.2f}%")
                show = d[[
                    "trade_id","direction","entry_time","exit_time",
                    "entry_price","exit_price","stop_loss","target",
                    "P&L","P&L%","result","exit_reason",
                ]].rename(columns={
                    "trade_id":"#","entry_time":"Entry","exit_time":"Exit",
                    "entry_price":"Entry ₹","exit_price":"Exit ₹",
                    "stop_loss":"SL","target":"Target","exit_reason":"Reason",
                })

                def _rbg(row):
                    c = ("rgba(38,166,154,0.18)" if row["result"] == "win"
                         else "rgba(239,83,80,0.15)" if row["result"] == "loss" else "")
                    return [f"background-color: {c}"] * len(row)

                st.dataframe(show.style.apply(_rbg, axis=1), width="stretch", height=380)

                # Store for Metrics tab
                st.session_state["trades_df"] = trades_df
                st.session_state["metrics"]   = metrics
    else:
        st.info(
            "**Quick start for maximum trades:**\n"
            "1. Symbol: any Nifty 50 (RELIANCE, INFY, TCS…)\n"
            "2. 60 days · 10min interval\n"
            "3. Min Confluence Score = **0** or **1**\n"
            "4. All Hard Gates = **OFF**\n"
            "5. Click ▶ Run Backtest\n\n"
            "Then raise the score to 2–3 to filter for quality."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_met:
    if "metrics" not in st.session_state:
        st.info("Run a Backtest first to see performance metrics.")
    else:
        metrics   = st.session_state["metrics"]
        trades_df = st.session_state["trades_df"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades",  metrics["total_trades"])
        c2.metric("Win Rate",      f"{metrics['win_rate']}%",
                  delta=f"{metrics['wins']}W / {metrics['losses']}L")
        c3.metric("Profit Factor", f"{metrics['profit_factor']}x")
        c4.metric("Net P&L",       f"₹{metrics['net_pnl']:,.2f}",
                  delta=f"Avg ₹{metrics['avg_pnl']:,.2f}/trade")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Max Drawdown", f"₹{metrics['max_drawdown']:,.2f}")
        c6.metric("Sharpe Ratio",  metrics["sharpe_ratio"])
        c7.metric("Best Trade",   f"₹{metrics['best_trade']:,.2f}")
        c8.metric("Worst Trade",  f"₹{metrics['worst_trade']:,.2f}")

        st.divider()
        ca, cb = st.columns(2)

        with ca:
            st.subheader("P&L per Trade")
            fig_b = go.Figure(go.Bar(
                x=list(range(1, len(trades_df) + 1)),
                y=trades_df["pnl"],
                marker_color=["#26a69a" if p >= 0 else "#ef5350" for p in trades_df["pnl"]],
            ))
            fig_b.update_layout(template="plotly_dark", height=260,
                                margin=dict(l=10,r=10,t=10,b=10),
                                xaxis_title="Trade #", yaxis_title="₹")
            st.plotly_chart(fig_b, width="stretch")

        with cb:
            st.subheader("Drawdown Curve")
            cum = trades_df["pnl"].cumsum()
            dd  = cum - cum.cummax()
            fig_d = go.Figure(go.Scatter(
                x=trades_df["exit_time"], y=dd,
                fill="tozeroy", fillcolor="rgba(239,83,80,0.2)",
                line=dict(color="#ef5350", width=2),
            ))
            fig_d.update_layout(template="plotly_dark", height=260,
                                margin=dict(l=10,r=10,t=10,b=10),
                                yaxis_title="Drawdown ₹")
            st.plotly_chart(fig_d, width="stretch")

        st.divider()
        cc, cd = st.columns(2)

        with cc:
            st.subheader("Win / Loss / Breakeven")
            be = metrics["total_trades"] - metrics["wins"] - metrics["losses"]
            fig_p = go.Figure(go.Pie(
                labels=["Wins","Losses","Breakeven"],
                values=[metrics["wins"], metrics["losses"], be],
                hole=0.45,
                marker=dict(colors=["#26a69a","#ef5350","#9e9e9e"]),
            ))
            fig_p.update_layout(template="plotly_dark", height=260,
                                margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_p, width="stretch")

        with cd:
            st.subheader("Exit Reasons")
            ec = trades_df["exit_reason"].value_counts()
            fig_e = go.Figure(go.Bar(
                x=ec.index.tolist(), y=ec.values.tolist(),
                marker_color=["#26a69a","#9e9e9e","#ef5350","#ffa726"][:len(ec)],
            ))
            fig_e.update_layout(template="plotly_dark", height=260,
                                margin=dict(l=10,r=10,t=10,b=10),
                                yaxis_title="# Trades")
            st.plotly_chart(fig_e, width="stretch")

        # ── Confluence score distribution ──────────────────────────────────
        if "signal_score" in trades_df.columns:
            st.divider()
            st.subheader("📊 Confluence Score vs Win Rate")
            score_stats = (
                trades_df.groupby("signal_score")
                .agg(
                    trades=("pnl", "count"),
                    wins=("result", lambda x: (x == "win").sum()),
                    net_pnl=("pnl", "sum"),
                )
                .reset_index()
            )
            score_stats["win_rate"] = (score_stats["wins"] / score_stats["trades"] * 100).round(1)
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Bar(
                x=score_stats["signal_score"],
                y=score_stats["trades"],
                name="# Trades",
                marker_color="#42a5f5",
                yaxis="y",
            ))
            fig_sc.add_trace(go.Scatter(
                x=score_stats["signal_score"],
                y=score_stats["win_rate"],
                name="Win Rate %",
                line=dict(color="#ffd54f", width=2),
                yaxis="y2",
                mode="lines+markers",
            ))
            fig_sc.update_layout(
                template="plotly_dark", height=280,
                margin=dict(l=10,r=60,t=10,b=10),
                xaxis_title="Confluence Score",
                yaxis=dict(title="# Trades"),
                yaxis2=dict(title="Win Rate %", overlaying="y", side="right", range=[0,100]),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_sc, width="stretch")
            st.caption("Higher confluence score = fewer trades but potentially higher win rate.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — SCANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab_scan:
    st.subheader("🔍 Multi-Stock Signal Scanner")
    st.caption(
        "Scans each stock with the same strategy settings. "
        "Shows current Supertrend, VWAP alignment, RSI, and whether a signal fired recently."
    )

    with st.expander("📋 Nifty 50 symbols", expanded=False):
        st.code(
            "RELIANCE,TCS,HDFCBANK,ICICIBANK,INFY,HINDUNILVR,ITC,BAJFINANCE,"
            "BHARTIARTL,KOTAKBANK,LT,SBIN,AXISBANK,ASIANPAINT,MARUTI,TITAN,"
            "NESTLEIND,ULTRACEMCO,BAJAJFINSV,WIPRO,HCLTECH,TECHM,SUNPHARMA,"
            "POWERGRID,NTPC,ONGC,TATAMOTORS,TATASTEEL,JSWSTEEL,HINDALCO,"
            "COALINDIA,ADANIPORTS,DRREDDY,DIVISLAB,CIPLA,GRASIM,BRITANNIA,"
            "BPCL,IOC,HEROMOTOCO,EICHERMOT,BAJAJ-AUTO,M&M,APOLLOHOSP,INDUSINDBK,"
            "ADANIENT,TATACONSUM,LTIM,HDFCLIFE,SBILIFE"
        )

    sc1, sc2, sc3 = st.columns([3, 1, 1])
    with sc1:
        scan_raw = st.text_input(
            "Symbols",
            value="RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,SBIN,WIPRO,MARUTI,ITC,BAJFINANCE",
        )
    with sc2:
        scan_days = st.number_input("Last N days", min_value=5, max_value=30, value=10)
    with sc3:
        SIVAL = ["1min", "5min", "10min", "1hour"]
        scan_iv = st.selectbox("Interval", SIVAL, index=SIVAL.index("10min"))

    if st.button("🔍 Scan Now", type="primary"):
        syms     = [s.strip().upper() for s in scan_raw.split(",") if s.strip()]
        s_end    = date.today() - timedelta(days=1)
        s_start  = s_end - timedelta(days=scan_days)
        results, errors = [], []
        prog   = st.progress(0)
        status = st.empty()

        try:
            sfetch = GrowwDataFetcher()
        except ValueError as e:
            st.error(str(e))
            st.stop()

        for idx, sym in enumerate(syms):
            status.text(f"Scanning {sym} ({idx+1}/{len(syms)})…")
            try:
                sdf = sfetch.fetch_candles(
                    symbol=sym, start_date=str(s_start),
                    end_date=str(s_end), interval=scan_iv, exchange="NSE",
                )
                if sdf.empty or len(sdf) < 60:
                    errors.append(f"{sym}: insufficient data")
                    continue

                sdf = generate_signals(
                    sdf, direction="both", min_score=0,
                    vwap_reversion=True, use_time_filter=False,
                )

                last    = sdf.iloc[-1]
                price   = float(last["close"])
                st_dir  = int(last.get("st_direction", 0))
                st_val  = float(last.get("supertrend", np.nan))
                rsi_v   = float(last.get("rsi", np.nan))
                vwap_v  = float(last.get("vwap", np.nan))
                poc_v   = float(last["poc"]) if not np.isnan(last.get("poc", np.nan)) else None

                trend = ("🟢 Bullish" if st_dir == 1 else "🔴 Bearish" if st_dir == -1 else "⚪")
                vwap_side = "Above 🟢" if price > vwap_v else "Below 🔴"

                recent_sig = "—"
                for _, sr in sdf.tail(5).iloc[::-1].iterrows():
                    if sr["signal"] == 1:
                        sc_v = int(sr["signal_score"])
                        recent_sig = f"🟢 Long (score {sc_v}/8)" if sc_v >= 0 else "🟢 Long (rev)"
                        break
                    elif sr["signal"] == -1:
                        sc_v = int(sr["signal_score"])
                        recent_sig = f"🔴 Short (score {sc_v}/8)" if sc_v >= 0 else "🔴 Short (rev)"
                        break

                pct_vwap = round((price - vwap_v) / price * 100, 2) if not np.isnan(vwap_v) else 999

                results.append({
                    "Symbol":   sym,
                    "ST Trend": trend,
                    "VWAP":     vwap_side,
                    "RSI":      round(rsi_v, 1) if not np.isnan(rsi_v) else "—",
                    "Price ₹":  round(price, 2),
                    "VWAP ₹":   round(vwap_v, 2) if not np.isnan(vwap_v) else "—",
                    "% vs VWAP":pct_vwap,
                    "Prev POC": round(poc_v, 2) if poc_v else "—",
                    "Signal":   recent_sig,
                })
            except Exception as e:
                errors.append(f"{sym}: {e}")
            prog.progress((idx+1) / len(syms))

        status.empty()
        prog.empty()

        if errors:
            with st.expander(f"⚠️ {len(errors)} errors"):
                for e in errors: st.text(e)

        if results:
            rdf = pd.DataFrame(results).sort_values("% vs VWAP")

            def _ss(row):
                has_sig = row.get("Signal", "—") != "—"
                if has_sig and "Long" in str(row.get("Signal", "")):
                    return ["background-color: rgba(38,166,154,0.3)"] * len(row)
                if has_sig and "Short" in str(row.get("Signal", "")):
                    return ["background-color: rgba(239,83,80,0.25)"] * len(row)
                if "Bullish" in str(row.get("ST Trend", "")):
                    return ["background-color: rgba(38,166,154,0.12)"] * len(row)
                if "Bearish" in str(row.get("ST Trend", "")):
                    return ["background-color: rgba(239,83,80,0.08)"] * len(row)
                return [""] * len(row)

            st.dataframe(rdf.style.apply(_ss, axis=1), width="stretch", hide_index=True)

            # Summary chart
            bull_n = rdf["ST Trend"].str.contains("Bullish").sum()
            bear_n = rdf["ST Trend"].str.contains("Bearish").sum()
            fig_sc = go.Figure(go.Bar(
                x=rdf["Symbol"],
                y=rdf["% vs VWAP"].apply(lambda x: x if x < 999 else 0),
                marker_color=[
                    "#26a69a" if "Bullish" in t else "#ef5350" if "Bearish" in t else "#9e9e9e"
                    for t in rdf["ST Trend"]
                ],
                text=rdf["Signal"],
                textposition="outside",
            ))
            fig_sc.update_layout(
                template="plotly_dark", height=300,
                title=f"🟢 {bull_n} Bullish · 🔴 {bear_n} Bearish — sorted by VWAP distance",
                margin=dict(l=10,r=10,t=50,b=10),
                yaxis_title="% distance from VWAP",
            )
            st.plotly_chart(fig_sc, width="stretch")
        else:
            st.warning("No results. Check API token or symbols.")
    else:
        st.info("Enter symbols and click **🔍 Scan Now**.")
