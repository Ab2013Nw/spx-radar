"""
leadlag_test.py
----------------
اختبار مباشر لفرضية: "هل حركة الشركات (اتساع السوق) تسبق حركة SPX
زمنياً، أو تتزامن معه، أو تتأخر عنه؟"

بدون أي قواعد تداول أو عتبات — بس ارتباط إحصائي صريح (Cross-Correlation)
بين قراءة الاتساع الآن، وحركة SPX بعد 0/15/30/45/60 دقيقة.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr

WEIGHTS_FILE = "weights.json"
INTERVAL = "15m"
PERIOD = "60d"
LAGS = [0, 1, 2, 3, 4]


def load_weights():
    with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["weights"]


def fetch_all_data(tickers):
    print(f"جلب بيانات {len(tickers)} شركة لفترة {PERIOD} ...")
    return yf.download(
        tickers=tickers, period=PERIOD, interval=INTERVAL,
        group_by="ticker", threads=True, progress=False, auto_adjust=False,
    )


def get_close_series(symbol):
    data = yf.download(symbol, period=PERIOD, interval=INTERVAL, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"].iloc[:, 0]
    return data["Close"]


def compute_signal_series(df):
    prev_avg = (df["Open"] + df["High"] + df["Low"] + df["Close"]).shift(1) / 4
    signal = pd.Series(0, index=df.index)
    signal[df["Close"] > prev_avg] = 1
    signal[df["Close"] < prev_avg] = -1
    signal[prev_avg.isna()] = np.nan
    return signal


def build_signals_matrix(candles, tickers):
    signals = {}
    for symbol in tickers:
        try:
            df = candles[symbol]
            if df["Close"].dropna().empty:
                continue
            signals[symbol] = compute_signal_series(df)
        except Exception:
            continue
    return pd.DataFrame(signals)


def aggregate_row(row, weights):
    valid = row.dropna()
    if valid.empty:
        return None, None
    w = np.array([weights.get(sym, 0) for sym in valid.index])
    total_w = w.sum()
    if total_w == 0:
        return None, None
    bullish_w = w[valid.values == 1].sum()
    weighted_pct = bullish_w / total_w * 100
    breadth_pct = (valid.values == 1).sum() / len(valid) * 100
    return weighted_pct, breadth_pct


def main():
    weights = load_weights()
    tickers = list(weights.keys())

    candles = fetch_all_data(tickers)
    signals_df = build_signals_matrix(candles, tickers)
    print(f"عدد الأوقات (الشموع): {len(signals_df)}")

    spx_close = get_close_series("^GSPC")

    series = []
    for ts, row in signals_df.iterrows():
        wp, bp = aggregate_row(row, weights)
        series.append({"time": ts, "weighted": wp, "breadth": bp})
    series_df = pd.DataFrame(series).set_index("time").dropna()

    series_df["weighted_delta"] = series_df["weighted"].diff(1)
    series_df["breadth_delta"] = series_df["breadth"].diff(1)
    series_df = series_df.dropna()

    print("=" * 65)
    print("اختبار الارتباط المتقدم/المتأخر: الاتساع مقابل حركة SPX")
    print("=" * 65)

    for lag in LAGS:
        future_ret = pd.Series(index=series_df.index, dtype=float)
        for ts in series_df.index:
            try:
                entry = spx_close.loc[spx_close.index >= ts].iloc[0]
                future_idx = spx_close.index[spx_close.index >= ts][lag]
                exitp = spx_close.loc[future_idx]
                future_ret.loc[ts] = (exitp - entry) / entry * 100
            except (IndexError, KeyError):
                future_ret.loc[ts] = np.nan

        merged = series_df.join(future_ret.rename("future_spx_move")).dropna()
        if len(merged) < 30:
            continue

        print(f"\n[الأفق: {lag*15} دقيقة قدام]  (عينة: {len(merged)})")
        for col in ["weighted", "breadth", "weighted_delta", "breadth_delta"]:
            corr, p = pearsonr(merged[col], merged["future_spx_move"])
            flag = "  <-- دلالة إحصائية (p<0.05)" if p < 0.05 else ""
            print(f"  {col:18s}: correlation = {corr:+.4f}   p-value = {p:.4f}{flag}")


if __name__ == "__main__":
    main()
