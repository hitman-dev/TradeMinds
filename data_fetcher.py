"""
Groww API wrapper for fetching historical candle data.
Uses get_historical_candle_data() with interval_in_minutes as integer.

Available intervals (from Groww docs):
  1min → 1     | max 7 days/call   | last 3 months history
  5min → 5     | max 15 days/call  | last 3 months history
  10min → 10   | max 30 days/call  | last 3 months history
  1hour → 60   | max 150 days/call | last 3 months history
  4hours → 240 | max 365 days/call | last 3 months history
  1day → 1440  | max 1080 days/call| full history
  1week → 10080| no limit          | full history
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Interval label → interval_in_minutes value for the API
INTERVAL_MINUTES = {
    "1min":   1,
    "5min":   5,
    "10min":  10,
    "1hour":  60,
    "4hours": 240,
    "1day":   1440,
    "1week":  10080,
}

# Max days per single API call (from Groww docs screenshot)
MAX_DAYS_PER_CALL = {
    "1min":   7,
    "5min":   15,
    "10min":  30,
    "1hour":  150,
    "4hours": 365,
    "1day":   1080,
    "1week":  9999,
}


class GrowwDataFetcher:
    def __init__(self, token: str = None):
        self.token = token or os.getenv("GROWW_API_TOKEN")
        if not self.token or self.token == "your_token_here":
            raise ValueError(
                "GROWW_API_TOKEN not set. Please add it to your .env file."
            )
        from growwapi import GrowwAPI
        self.client = GrowwAPI(self.token)

    def _parse_candles(self, raw) -> pd.DataFrame:
        """
        Parse candle data from API response.
        Handles multiple possible response formats from Groww SDK.
        """
        candles = None

        if isinstance(raw, dict):
            # Backtesting API format: {"candles": [...], ...}
            candles = raw.get("candles") or raw.get("data") or raw.get("ohlc")
        elif isinstance(raw, list):
            candles = raw

        if not candles:
            return pd.DataFrame()

        rows = []
        for c in candles:
            if isinstance(c, (list, tuple)):
                # [timestamp, open, high, low, close, volume, oi?]
                ts = c[0]
                o, h, l, cl = float(c[1]), float(c[2]), float(c[3]), float(c[4])
                vol = int(c[5]) if len(c) > 5 and c[5] is not None else 0
            elif isinstance(c, dict):
                ts = c.get("timestamp") or c.get("time") or c.get("t")
                o = float(c.get("open", 0))
                h = float(c.get("high", 0))
                l = float(c.get("low", 0))
                cl = float(c.get("close", 0))
                vol = int(c.get("volume", 0) or 0)
            else:
                continue
            rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": cl, "volume": vol})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce").fillna(
            pd.to_datetime(df["timestamp"], errors="coerce")
        )
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def fetch_candles(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "10min",
        exchange: str = "NSE",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for a symbol over a date range.

        Automatically chunks requests if the date range exceeds the API's
        per-call limit for the given interval.

        Args:
            symbol:     Stock symbol e.g. "RELIANCE"
            start_date: "YYYY-MM-DD"
            end_date:   "YYYY-MM-DD"
            interval:   One of: 1min, 5min, 10min, 1hour, 4hours, 1day, 1week
            exchange:   "NSE" or "BSE"

        Returns:
            pd.DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if interval not in INTERVAL_MINUTES:
            raise ValueError(
                f"Unsupported interval '{interval}'. "
                f"Choose from: {list(INTERVAL_MINUTES.keys())}"
            )

        interval_mins = INTERVAL_MINUTES[interval]
        max_days = MAX_DAYS_PER_CALL[interval]
        symbol = symbol.upper().strip()

        exchange_const = (
            self.client.EXCHANGE_NSE if exchange.upper() == "NSE"
            else self.client.EXCHANGE_BSE
        )

        # Parse dates
        if isinstance(start_date, str) and len(start_date) == 10:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_dt = pd.to_datetime(start_date).to_pydatetime()

        if isinstance(end_date, str) and len(end_date) == 10:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59
            )
        else:
            end_dt = pd.to_datetime(end_date).to_pydatetime()

        all_frames = []
        chunk_start = start_dt

        while chunk_start < end_dt:
            chunk_end = min(chunk_start + timedelta(days=max_days), end_dt)
            start_str = chunk_start.strftime("%Y-%m-%d %H:%M:%S")
            end_str = chunk_end.strftime("%Y-%m-%d %H:%M:%S")

            response = self.client.get_historical_candle_data(
                trading_symbol=symbol,
                exchange=exchange_const,
                segment=self.client.SEGMENT_CASH,
                start_time=start_str,
                end_time=end_str,
                interval_in_minutes=interval_mins,
            )

            chunk_df = self._parse_candles(response)
            if not chunk_df.empty:
                all_frames.append(chunk_df)

            chunk_start = chunk_end + timedelta(seconds=1)

        if not all_frames:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        result = pd.concat(all_frames, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        result = result.reset_index(drop=True)
        return result
