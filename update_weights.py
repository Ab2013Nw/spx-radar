"""
update_weights.py
------------------
يجيب أوزان مؤشر S&P 500 (نسبة كل شركة من المؤشر) من موقع SlickCharts
مباشرة بطلب واحد بس، ويحفظها بملف weights.json.

هذي الطريقة أخف وأوثق من إنه نسأل Yahoo Finance عن كل شركة لحالها
(503 طلب منفصل)، واللي تبين إنه يتعرض للحجب من سيرفرات GitHub.

يشتغل هذا الملف أسبوعياً بس، لأن أوزان المؤشر ما تتغير بسرعة.
"""

import io
import json
import requests
import pandas as pd

URL = "https://www.slickcharts.com/sp500"
OUTPUT_FILE = "weights.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
}


def get_sp500_weights():
    """يجيب جدول أوزان S&P 500 من SlickCharts ويرجعه كـ dict {رمز: وزن}."""
    resp = requests.get(URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]

    # بعض الرموز فيها نقطة (BRK.B) ولازم تصير شرطة (BRK-B) عشان yfinance
    df["Symbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)

    # عمود الوزن يجي كنص فيه علامة % مثل "7.13%" -> نحوله لرقم عشري 0.0713
    df["Weight"] = (
        df["Weight"].astype(str).str.replace("%", "", regex=False).astype(float) / 100
    )

    weights = dict(zip(df["Symbol"], df["Weight"]))
    return weights


def main():
    print("جلب أوزان S&P 500 من SlickCharts ...")
    weights = get_sp500_weights()
    print(f"عدد الشركات: {len(weights)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"updated_tickers": list(weights.keys()), "weights": weights},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"تم الحفظ في {OUTPUT_FILE} بنجاح — {len(weights)} شركة")


if __name__ == "__main__":
    main()
