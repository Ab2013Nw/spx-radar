"""
backtest.py
-----------
يشغّل نفس منطق spx_radar.py بالضبط، بس على بيانات تاريخية (آخر ~6 أشهر)،
عشان نشوف: كل مرة صار فيها قرار Call/Put، هل السوق فعلاً تحرك بنفس
الاتجاه بعدها، ولا لأ؟

النتيجة: نسبة نجاح حقيقية مبنية على بيانات، مو تخمين.

⚠️ هذا اختبار تاريخي (Backtest) بس — الأداء بالماضي ما يضمن نفس الأداء
بالمستقبل. الهدف إنه يعطيك فكرة أولية عن قوة الفكرة، مو ضمان ربح.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import percentileofscore

WEIGHTS_FILE = "weights.json"
INTERVAL = "1h"
BACKTEST_PERIOD = "180d"
HISTORY_WINDOW = 30
PERCENTILE_HIGH = 85
PERCENTILE_LOW = 15
FORWARD_HOURS = 4
OUTPUT_FILE = "backtest_results.csv"


def load_weights():
    with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["weights"]


def fetch_all_data(tickers):
    print(f"جلب بيانات {len(tickers)} شركة لفترة {BACKTEST_PERIOD} ...")
    data = yf.download(
        tickers=tickers,
        period=BACKTEST_PERIOD,
        interval=INTERVAL,
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )
    return data


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


def run_backtest():
    weights = load_weights()
    tickers = list(weights.keys())

    candles = fetch_all_data(tickers)
    signals_df = build_signals_matrix(candles, tickers)
    print(f"عدد الأوقات (الشموع) بالفترة التاريخية: {len(signals_df)}")

    spx = yf.download("^GSPC", period=BACKTEST_PERIOD, interval=INTERVAL, progress=False)
    if isinstance(spx.columns, pd.MultiIndex):
        spx_close = spx["Close"].iloc[:, 0]
    else:
        spx_close = spx["Close"]

    history = []
    last_decision = None
    trades = []

    for ts, row in signals_df.iterrows():
        weighted_pct, breadth_pct = aggregate_row(row, weights)
        if weighted_pct is None:
            continue

        weighted_pctl = percentileofscore(
            [h["weighted"] for h in history] + [weighted_pct], weighted_pct
        ) if history else 50.0
        breadth_pctl = percentileofscore(
            [h["breadth"] for h in history] + [breadth_pct], breadth_pct
        ) if history else 50.0

        history.append({"weighted": weighted_pct, "breadth": breadth_pct})
        history[:] = history[-HISTORY_WINDOW:]

        if weighted_pctl >= PERCENTILE_HIGH and breadth_pctl >= PERCENTILE_HIGH:
            decision = "CALL"
        elif weighted_pctl <= PERCENTILE_LOW and breadth_pctl <= PERCENTILE_LOW:
            decision = "PUT"
        else:
            decision = "NEUTRAL"

        if decision in ("CALL", "PUT") and decision != last_decision:
            trades.append({"time": ts, "decision": decision})
            last_decision = decision

    print(f"عدد القرارات (تغيّرات) اللي صارت بالفترة التاريخية: {len(trades)}")

    for forward_hours in (1, 2, 4):
        evaluate_horizon(trades, spx_close, forward_hours)


def evaluate_horizon(trades, spx_close, forward_hours):
    results = []
    for trade in trades:
        ts = trade["time"]
        try:
            entry_price = spx_close.loc[spx_close.index >= ts].iloc[0]
            future_idx = spx_close.index[spx_close.index >= ts][forward_hours]
            exit_price = spx_close.loc[future_idx]
        except (IndexError, KeyError):
            continue

        move_pct = (exit_price - entry_price) / entry_price * 100
        won = (trade["decision"] == "CALL" and move_pct > 0) or \
              (trade["decision"] == "PUT" and move_pct < 0)

        results.append({
            "time": ts,
            "decision": trade["decision"],
            "move_pct": round(float(move_pct), 3),
            "won": won,
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print(f"[أفق {forward_hours} ساعة] ما فيه قرارات كافية للتقييم.")
        return

    if forward_hours == 4:
        results_df.to_csv(OUTPUT_FILE, index=False)

    total = len(results_df)
    wins = results_df["won"].sum()
    win_rate = wins / total * 100
    call_df = results_df[results_df["decision"] == "CALL"]
    put_df = results_df[results_df["decision"] == "PUT"]

    print("-" * 50)
    print(f"[أفق {forward_hours} ساعة] إجمالي: {total} | نجاح إجمالي: {win_rate:.1f}%")
    if len(call_df):
        print(f"  CALL: {len(call_df)} قرار، نجاح {call_df['won'].mean()*100:.1f}%")
    if len(put_df):
        print(f"  PUT: {len(put_df)} قرار، نجاح {put_df['won'].mean()*100:.1f}%")


if __name__ == "__main__":
    run_backtest()
