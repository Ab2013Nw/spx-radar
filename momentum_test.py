"""
momentum_test.py
-----------------
فكرة مختلفة كليًا عن فكرة "إجماع 500 شركة": نختبر هل مؤشر SPX نفسه
عنده نمط زخم أو ارتداد قصير المدى (15-60 دقيقة)، بدل الاعتماد على
إشارة مجمّعة من شركات كثيرة.

السؤال: لو SPX تحرك X% بالفترة الماضية، هل يميل يكمل بنفس الاتجاه
(Momentum) أو يرتد بالعكس (Mean Reversion)؟ نختبر الاثنين بدون
افتراض مسبق، ونشوف وش تقوله البيانات فعليًا.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr

INTERVAL = "15m"
PERIOD = "60d"

LOOKBACK_OPTIONS = [1, 2, 4]
FORWARD_OPTIONS = [1, 2]


def get_spx_close():
    spx = yf.download("^GSPC", period=PERIOD, interval=INTERVAL, progress=False)
    if isinstance(spx.columns, pd.MultiIndex):
        return spx["Close"].iloc[:, 0]
    return spx["Close"]


def analyze(close):
    returns = close.pct_change() * 100

    print("=" * 60)
    print("اختبار الزخم/الارتداد القصير المدى لمؤشر SPX")
    print("=" * 60)

    for lookback in LOOKBACK_OPTIONS:
        past_move = close.pct_change(lookback) * 100
        for forward in FORWARD_OPTIONS:
            future_move = (close.shift(-forward) - close) / close * 100

            df = pd.DataFrame({"past": past_move, "future": future_move}).dropna()
            if len(df) < 30:
                continue

            corr, p_value = pearsonr(df["past"], df["future"])

            same_direction = np.sign(df["past"]) == np.sign(df["future"])
            same_direction = same_direction[df["past"] != 0]
            momentum_win_rate = same_direction.mean() * 100 if len(same_direction) else 0

            print(f"[نظرة {lookback*15} دقيقة للخلف -> توقع {forward*15} دقيقة للأمام]")
            print(f"  الارتباط (Correlation): {corr:+.4f}  (p-value: {p_value:.4f})")
            print(f"  عينة: {len(df)} قراءة")
            print(f"  نسبة نجاح 'كمّل بنفس الاتجاه' (Momentum): {momentum_win_rate:.1f}%")
            print(f"  نسبة نجاح 'ارتد بالعكس' (Mean Reversion): {100 - momentum_win_rate:.1f}%")
            print("-" * 60)


def main():
    print("جلب بيانات SPX (15 دقيقة، آخر 60 يوم) ...")
    close = get_spx_close()
    print(f"عدد الشموع: {len(close)}")
    analyze(close)


if __name__ == "__main__":
    main()
