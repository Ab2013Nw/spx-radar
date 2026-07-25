"""
update_weights.py
------------------
يجيب قائمة شركات S&P 500 الحالية من ويكيبيديا، ثم يجيب القيمة السوقية
(Market Cap) لكل شركة عبر yfinance، ويحفظها بملف weights.json.

يشتغل هذا الملف أسبوعياً بس (مو كل 5 دقايق) لأن تركيبة S&P 500 والأوزان
السوقية ما تتغير بسرعة، وهذا يوفر وقت التشغيل ويقلل الطلبات.
"""

import io
import json
import time
import requests
import pandas as pd
import yfinance as yf

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUTPUT_FILE = "weights.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
}


def get_sp500_tickers():
    """يجيب رموز شركات S&P 500 من ويكيبيديا ويصلحها لصيغة Yahoo Finance."""
    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    tickers = df["Symbol"].tolist()
    # بعض الرموز فيها نقطة (BRK.B) ولازم تصير شرطة (BRK-B) عشان yfinance
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


def fetch_market_caps(tickers):
    """يجيب القيمة السوقية لكل شركة. يستخدم fast_info وهو خفيف وسريع."""
    caps = {}
    for i, symbol in enumerate(tickers):
        try:
            t = yf.Ticker(symbol)
            cap = t.fast_info.get("market_cap")
            if cap:
                caps[symbol] = float(cap)
        except Exception as e:
            print(f"تعذر جلب القيمة السوقية لـ {symbol}: {e}")
        # وقفة بسيطة كل شوي عشان ما نضغط على الخادم
        if i % 50 == 0 and i > 0:
            time.sleep(2)
    return caps


def main():
    print("جلب قائمة شركات S&P 500 ...")
    tickers = get_sp500_tickers()
    print(f"عدد الشركات: {len(tickers)}")

    print("جلب القيم السوقية (هذي الخطوة تاخذ عدة دقايق) ...")
    caps = fetch_market_caps(tickers)

    total = sum(caps.values())
    weights = {sym: cap / total for sym, cap in caps.items()}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"updated_tickers": list(weights.keys()), "weights": weights},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"تم الحفظ في {OUTPUT_FILE} — عدد الشركات المحفوظة: {len(weights)}")


if __name__ == "__main__":
    main()
