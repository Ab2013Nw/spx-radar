"""
backtest_v3.py
--------------
نفس منطق backtest_v2.py (تسارع الاتساع + تأكيد اتجاه SPX)، بإضافة
فلتر جودة جديد: نتجاهل القرار لو مؤشر VIX كان بحالة تطرف تاريخي
(متقلب جداً مقارنة بالفترة الأخيرة) وقت صدور الإشارة.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import percentileofscore

WEIGHTS_FILE = "weights.json"
INTERVAL = "15m"
BACKTEST_PERIOD = "60d"
SMA_WINDOW = 3
MOMENTUM_LOOKBACK = 4
HISTORY_WINDOW = 30
PERCENTILE_HIGH = 85
PERCENTILE_LOW = 15
VIX_EXTREME_PERCENTILE = 85


def load_weights():
    with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["weights"]


def fetch_all_data(tickers):
    print(f"جلب بيانات {len(tickers)} شركة لفترة {BACKTEST_PERIOD} ...")
    return yf.download(
        tickers=tickers, period=BACKTEST_PERIOD, interval=INTERVAL,
        group_by="ticker", threads=True, progress=False, auto_adjust=False,
    )


def get_close_series(symbol):
    data = yf.download(symbol, period=BACKTEST_PERIOD, interval=INTERVAL, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"].iloc[:, 0]
    return data["Close"]


def compute_signal_series(df):
    typical = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    sma = typical.rolling(SMA_WINDOW).mean().shift(1)
    signal = pd.Series(0, index=df.index)
    signal[df["Close"] > sma] = 1
    signal[df["Close"] < sma] = -1
    signal[sma.isna()] = np.nan
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


def run_backtest():
    weights = load_weights()
    tickers = list(weights.keys())

    candles = fetch_all_data(tickers)
    signals_df = build_signals_matrix(candles, tickers)
    print(f"عدد الأوقات (الشموع): {len(signals_df)}")

    spx_close = get_close_series("^GSPC")
    vix_close = get_close_series("^VIX")
    vix_aligned = vix_close.reindex(signals_df.index, method="ffill")

    series = []
    for ts, row in signals_df.iterrows():
        wp, bp = aggregate_row(row, weights)
        series.append({"time": ts, "weighted": wp, "breadth": bp})
    series_df = pd.DataFrame(series).set_index("time").dropna()

    series_df["weighted_accel"] = series_df["weighted"].diff(MOMENTUM_LOOKBACK)
    series_df["breadth_accel"] = series_df["breadth"].diff(MOMENTUM_LOOKBACK)
    series_df = series_df.dropna()

    history = []
    vix_history = []
    last_decision = None
    trades = []
    skipped_vix = 0

    for ts, row in series_df.iterrows():
        w_accel, b_accel = row["weighted_accel"], row["breadth_accel"]

        w_pctl = percentileofscore(
            [h["w"] for h in history] + [w_accel], w_accel
        ) if history else 50.0
        b_pctl = percentileofscore(
            [h["b"] for h in history] + [b_accel], b_accel
        ) if history else 50.0

        history.append({"w": w_accel, "b": b_accel})
        history[:] = history[-HISTORY_WINDOW:]

        vix_val = vix_aligned.loc[ts] if ts in vix_aligned.index else np.nan
        vix_pctl = percentileofscore(vix_history + [vix_val], vix_val) if vix_history else 50.0
        vix_history.append(vix_val)
        vix_history[:] = vix_history[-HISTORY_WINDOW:]

        try:
            spx_recent = spx_close.loc[spx_close.index <= ts].iloc[-3:]
            spx_trend_up = spx_recent.iloc[-1] > spx_recent.iloc[0]
        except Exception:
            continue

        if w_pctl >= PERCENTILE_HIGH and b_pctl >= PERCENTILE_HIGH and spx_trend_up:
            decision = "CALL"
        elif w_pctl <= PERCENTILE_LOW and b_pctl <= PERCENTILE_LOW and not spx_trend_up:
            decision = "PUT"
        else:
            decision = "NEUTRAL"

        if decision in ("CALL", "PUT") and vix_pctl >= VIX_EXTREME_PERCENTILE:
            skipped_vix += 1
            decision = "NEUTRAL"

        if decision in ("CALL", "PUT") and decision != last_decision:
            trades.append({"time": ts, "decision": decision})
            last_decision = decision

    print(f"عدد القرارات (تغيّرات): {len(trades)}")
    print(f"عدد القرارات اللي استبعدت بسبب فلتر VIX: {skipped_vix}")

    if not trades:
        print("ما فيه قرارات كافية للتقييم.")
        return

    trades_sorted = sorted(trades, key=lambda t: t["time"])
    mid = len(trades_sorted) // 2
    first_half = trades_sorted[:mid]
    second_half = trades_sorted[mid:]

    print("\n" + "#" * 60)
    print(f"النصف الأول (الأقدم زمنياً) — {len(first_half)} قرار")
    print("#" * 60)
    for steps, label in ((1, "15 دقيقة"), (2, "30 دقيقة")):
        evaluate_horizon(first_half, spx_close, steps, label)

    print("\n" + "#" * 60)
    print(f"النصف الثاني (الأحدث زمنياً) — {len(second_half)} قرار")
    print("#" * 60)
    for steps, label in ((1, "15 دقيقة"), (2, "30 دقيقة")):
        evaluate_horizon(second_half, spx_close, steps, label)

    print("\n" + "#" * 60)
    print(f"الإجمالي الكامل — {len(trades_sorted)} قرار")
    print("#" * 60)
    for steps, label in ((1, "15 دقيقة"), (2, "30 دقيقة")):
        evaluate_horizon(trades_sorted, spx_close, steps, label)


def evaluate_horizon(trades, spx_close, steps, label):
    results = []
    for trade in trades:
        ts = trade["time"]
        try:
            entry_price = spx_close.loc[spx_close.index >= ts].iloc[0]
            future_idx = spx_close.index[spx_close.index >= ts][steps]
            exit_price = spx_close.loc[future_idx]
        except (IndexError, KeyError):
            continue

        move_pct = (exit_price - entry_price) / entry_price * 100
        won = (trade["decision"] == "CALL" and move_pct > 0) or \
              (trade["decision"] == "PUT" and move_pct < 0)
        results.append({"decision": trade["decision"], "won": won})

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print(f"[أفق {label}] ما فيه قرارات كافية للتقييم.")
        return

    total = len(results_df)
    wins = results_df["won"].sum()
    win_rate = wins / total * 100
    call_df = results_df[results_df["decision"] == "CALL"]
    put_df = results_df[results_df["decision"] == "PUT"]

    print("-" * 50)
    print(f"[أفق {label}] إجمالي: {total} | نجاح إجمالي: {win_rate:.1f}%")
    if len(call_df):
        print(f"  CALL: {len(call_df)} قرار، نجاح {call_df['won'].mean()*100:.1f}%")
    if len(put_df):
        print(f"  PUT: {len(put_df)} قرار، نجاح {put_df['won'].mean()*100:.1f}%")


if __name__ == "__main__":
    run_backtest()
