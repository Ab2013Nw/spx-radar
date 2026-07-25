"""
spx_radar.py
------------
رادار لشركات S&P 500:
1) يفحص كل شركة: هل إغلاق الشمعة الحالية فوق أو تحت متوسط OHLC للشمعة السابقة؟
2) يجمع النتائج بطريقتين: موزونة بالقيمة السوقية (Weighted%) وغير موزونة (Breadth%)
3) يقارن القراءة الحالية بتاريخها القريب عبر Percentile Rank (عتبة ديناميكية، مو رقم ثابت)
4) لو صار "اتفاق" واضح بين الطريقتين على اتجاه متطرف تاريخياً، يقترح قرار Call/Put على SPX
5) يقترح 3 مستويات Strike (محافظ/متوازن/جريء) بناءً على Delta محسوب من Black-Scholes
6) يرسل تنبيه تيليجرام فقط لو القرار تغيّر عن آخر مرة

⚠️ هذا الكود أداة آلية مبنية على قاعدة فنية بسيطة، مو توصية استثمارية.
القرار النهائي بالتداول يرجع لك دائماً.
"""

import os
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from scipy.stats import norm, percentileofscore

# ---------------------------------------------------------------------------
# الإعدادات
# ---------------------------------------------------------------------------
WEIGHTS_FILE = "weights.json"
STATE_FILE = "state.json"
INTERVAL = "1h"          # فريم الشموع المستخدم للإشارة
LOOKBACK_PERIOD = "5d"   # مدة سحب البيانات (تكفي لحساب الشمعة السابقة)
HISTORY_WINDOW = 30      # عدد القراءات المحفوظة لحساب الـ Percentile
PERCENTILE_HIGH = 85     # عتبة الاعتراف بحركة "قوية صعودية"
PERCENTILE_LOW = 15      # عتبة الاعتراف بحركة "قوية هبوطية"
RISK_FREE_RATE = 0.05

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ---------------------------------------------------------------------------
# 1) تحميل الأوزان (القيمة السوقية) المحفوظة مسبقاً
# ---------------------------------------------------------------------------
def load_weights():
    with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["weights"]


# ---------------------------------------------------------------------------
# 2) جلب بيانات الشموع لكل الشركات دفعة وحدة (طلب واحد بدل 500 طلب)
# ---------------------------------------------------------------------------
def fetch_candles(tickers):
    data = yf.download(
        tickers=tickers,
        period=LOOKBACK_PERIOD,
        interval=INTERVAL,
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )
    return data


# ---------------------------------------------------------------------------
# 3) حساب إشارة الاختراق لكل شركة: إغلاق الشمعة الحالية مقابل متوسط
#    OHLC للشمعة السابقة
# ---------------------------------------------------------------------------
def compute_signals(candles, tickers):
    signals = {}
    for symbol in tickers:
        try:
            df = candles[symbol].dropna()
            if len(df) < 2:
                continue
            prev = df.iloc[-2]
            curr = df.iloc[-1]
            prev_avg = (prev["Open"] + prev["High"] + prev["Low"] + prev["Close"]) / 4
            if curr["Close"] > prev_avg:
                signals[symbol] = 1
            elif curr["Close"] < prev_avg:
                signals[symbol] = -1
            else:
                signals[symbol] = 0
        except Exception:
            continue
    return signals


# ---------------------------------------------------------------------------
# 4) تجميع الإشارات: نسبة موزونة بالقيمة السوقية + نسبة غير موزونة (Breadth)
# ---------------------------------------------------------------------------
def aggregate(signals, weights):
    total_weight = 0.0
    bullish_weight = 0.0
    bullish_count = 0
    total_count = 0

    for symbol, sig in signals.items():
        w = weights.get(symbol, 0)
        total_weight += w
        total_count += 1
        if sig == 1:
            bullish_weight += w
            bullish_count += 1

    weighted_pct = (bullish_weight / total_weight * 100) if total_weight else 0
    breadth_pct = (bullish_count / total_count * 100) if total_count else 0
    return weighted_pct, breadth_pct


# ---------------------------------------------------------------------------
# 5) تحميل/تحديث السجل التاريخي وحساب الـ Percentile Rank (العتبة الديناميكية)
# ---------------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"history": [], "last_decision": None}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def update_history_and_get_percentiles(state, weighted_pct, breadth_pct):
    history = state["history"]

    weighted_percentile = percentileofscore(
        [h["weighted"] for h in history] + [weighted_pct], weighted_pct
    ) if history else 50.0

    breadth_percentile = percentileofscore(
        [h["breadth"] for h in history] + [breadth_pct], breadth_pct
    ) if history else 50.0

    history.append({
        "weighted": weighted_pct,
        "breadth": breadth_pct,
        "time": datetime.now(timezone.utc).isoformat(),
    })
    state["history"] = history[-HISTORY_WINDOW:]

    return weighted_percentile, breadth_percentile


# ---------------------------------------------------------------------------
# 6) تحديد القرار النهائي (شرط مزدوج: الوزن + الاتساع لازم يتفقوا)
# ---------------------------------------------------------------------------
def decide(weighted_percentile, breadth_percentile):
    if weighted_percentile >= PERCENTILE_HIGH and breadth_percentile >= PERCENTILE_HIGH:
        return "CALL"
    if weighted_percentile <= PERCENTILE_LOW and breadth_percentile <= PERCENTILE_LOW:
        return "PUT"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# 7) جلب سعر SPX الحالي
# ---------------------------------------------------------------------------
def get_spx_price():
    t = yf.Ticker("^GSPC")
    return float(t.fast_info["last_price"])


# ---------------------------------------------------------------------------
# 8) جلب سلسلة الخيارات. مؤشر SPX نفسه (^GSPC) ما عنده سلسلة خيارات في
#    yfinance، فنستخدم خيارات SPY (وهو صندوق يتتبع نفس المؤشر تقريباً
#    بمقياس 1/10) كبديل عملي، ثم نحوّل الأسعار لمقياس SPX.
# ---------------------------------------------------------------------------
def get_option_chain(option_type):
    spy = yf.Ticker("SPY")
    expirations = spy.options
    if not expirations:
        return None, None, None

    nearest_expiry = expirations[0]
    chain = spy.option_chain(nearest_expiry)
    df = chain.calls if option_type == "CALL" else chain.puts

    spx_price = get_spx_price()
    spy_price = float(yf.Ticker("SPY").fast_info["last_price"])
    scale = spx_price / spy_price

    return df, nearest_expiry, scale


# ---------------------------------------------------------------------------
# 9) حساب Delta عبر معادلة Black-Scholes
# ---------------------------------------------------------------------------
def bs_delta(S, K, T, r, sigma, option_type):
    if sigma <= 0 or T <= 0:
        return 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if option_type == "CALL":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1  # للـ Put


def suggest_strikes(option_type):
    df, expiry, scale = get_option_chain(option_type)
    if df is None or df.empty:
        return None

    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    T = max((expiry_date - datetime.now(timezone.utc)).days, 1) / 365.0

    spx_price = get_spx_price()
    spy_price = spx_price / scale

    targets = {"محافظ": 0.65, "متوازن": 0.45, "جريء": 0.25}
    results = {}

    for label, target_delta in targets.items():
        best_row, best_diff = None, 999
        for _, row in df.iterrows():
            iv = row.get("impliedVolatility", 0)
            strike = row.get("strike", 0)
            if iv <= 0 or strike <= 0:
                continue
            delta = bs_delta(spy_price, strike, T, RISK_FREE_RATE, iv,
                              option_type if option_type == "CALL" else "PUT")
            diff = abs(abs(delta) - target_delta)
            if diff < best_diff:
                best_diff = diff
                best_row = row

        if best_row is not None:
            spx_equiv_strike = round(best_row["strike"] * scale)
            results[label] = {
                "strike_spx_approx": spx_equiv_strike,
                "strike_spy_actual": best_row["strike"],
                "expiry": expiry,
            }

    return results


# ---------------------------------------------------------------------------
# 10) إرسال رسالة تيليجرام
# ---------------------------------------------------------------------------
def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("تحذير: توكن تيليجرام أو Chat ID غير موجودين بمتغيرات البيئة.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})


def build_message(decision, weighted_pct, breadth_pct, weighted_pctl, breadth_pctl,
                   bullish, bearish, strikes):
    spx_price = get_spx_price()
    lines = [
        f"🎯 رادار S&P 500 — قرار جديد: {decision}",
        f"سعر SPX الحالي: {spx_price:.2f}",
        "",
        f"النسبة الموزونة (Weighted%): {weighted_pct:.1f}%  (رتبة تاريخية: {weighted_pctl:.0f}%)",
        f"نسبة الاتساع (Breadth%): {breadth_pct:.1f}%  (رتبة تاريخية: {breadth_pctl:.0f}%)",
        f"عدد الشركات الصعودية: {bullish} | الهبوطية: {bearish}",
        "",
    ]

    if strikes:
        lines.append(f"اقتراح Strike (عقد {list(strikes.values())[0]['expiry']}):")
        for label, info in strikes.items():
            lines.append(f"  • {label}: ≈ {info['strike_spx_approx']} (SPX)")

    lines.append("")
    lines.append("⚠️ أداة آلية مساعدة، مو توصية استثمارية.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# التشغيل الرئيسي
# ---------------------------------------------------------------------------
def main():
    weights = load_weights()
    tickers = list(weights.keys())

    print(f"جلب بيانات {len(tickers)} شركة ...")
    candles = fetch_candles(tickers)

    signals = compute_signals(candles, tickers)
    print(f"عدد الشركات اللي طلعت لها إشارة: {len(signals)}")

    weighted_pct, breadth_pct = aggregate(signals, weights)
    bullish = sum(1 for s in signals.values() if s == 1)
    bearish = sum(1 for s in signals.values() if s == -1)

    state = load_state()
    weighted_pctl, breadth_pctl = update_history_and_get_percentiles(
        state, weighted_pct, breadth_pct
    )

    decision = decide(weighted_pctl, breadth_pctl)
    print(f"Weighted%={weighted_pct:.1f} Breadth%={breadth_pct:.1f} "
          f"WeightedPctl={weighted_pctl:.0f} BreadthPctl={breadth_pctl:.0f} "
          f"Decision={decision}")

    last_decision = state.get("last_decision")

    if decision in ("CALL", "PUT") and decision != last_decision:
        strikes = suggest_strikes(decision)
        message = build_message(
            decision, weighted_pct, breadth_pct, weighted_pctl, breadth_pctl,
            bullish, bearish, strikes
        )
        send_telegram(message)
        state["last_decision"] = decision
        print("تم إرسال تنبيه تيليجرام (تغيّر القرار).")
    else:
        print("لا تغيير بالقرار — ما راح يرسل تنبيه.")

    save_state(state)


if __name__ == "__main__":
    main()
