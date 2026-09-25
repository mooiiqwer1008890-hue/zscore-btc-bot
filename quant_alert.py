"""
quant_alert.py
===============
Python replacement for the C/Termux BTC analysis bot.

What changed vs. the original C version, and why:

1. SECRETS: Bot token / chat ID are read from environment variables,
   never hardcoded. Set them in Railway's "Variables" tab, not in code.
   NEVER commit a token to git, even in a private repo.

2. NO SHELL INJECTION: The original C code built a curl command by
   string-concatenating the message text, so a literal "$" in the
   message (e.g. "$67234.50") could trigger shell variable expansion
   and silently corrupt or truncate the message. Here, `requests`
   sends the message as a proper HTTP POST body — no shell involved,
   so there is nothing to inject into.

3. NO system()/curl/jq: `requests` talks to Binance and Telegram
   directly over HTTPS. Fewer moving parts, proper error handling,
   no dependency on external CLI tools being installed correctly.

4. RUNS FOREVER, SAFELY: A loop with a sleep interval, wrapped so a
   single failed request (network blip) doesn't crash the whole
   process — it logs the error and tries again next cycle. This is
   the actual requirement for "works all the time", and it has
   nothing to do with C vs Python — it's about where it's hosted
   (Railway, always-on) vs where it isn't (a phone Android kills).

Indicators (same formulas as your C version):
- Z-Score of the last close vs. a rolling mean/std (mean-reversion signal)
- Garman-Klass volatility (uses OHLC, more efficient than close-only vol)

IMPORTANT CAVEAT (carried over from our earlier discussion):
The signal thresholds below (|z| > 2) are the same ones from your C
code — they are a REASONABLE STARTING POINT, not a validated
strategy. Before trusting this for real capital, backtest these
thresholds properly (walk-forward, out-of-sample) using the
feature-engineering pipeline we already built. This script is the
DATA/ALERT layer, not a substitute for that validation step.
"""

import os
import time
import logging
import requests
import numpy as np

# ---------------------------------------------------------------------
# CONFIG — all secrets come from environment variables, set these in
# Railway's dashboard under Variables, never in this file.
# ---------------------------------------------------------------------
BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")
CHAT_ID = os.environ.get("TG_CHAT_ID")
SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
INTERVAL = os.environ.get("INTERVAL", "15m")
LOOKBACK = int(os.environ.get("LOOKBACK", "20"))
CANDLE_LIMIT = int(os.environ.get("CANDLE_LIMIT", "100"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "900"))  # 15 min default

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("quant_alert")


# ---------------------------------------------------------------------
# 1. DATA FETCHING
# ---------------------------------------------------------------------
def fetch_candles(symbol: str, interval: str, limit: int) -> np.ndarray:
    """
    Returns an (N, 4) array of [open, high, low, close] as floats,
    oldest first. Raises on network/API error — caller decides how
    to handle it (we catch it in the main loop so one bad request
    doesn't kill the process).
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    # Binance kline fields: [open_time, open, high, low, close, volume, ...]
    candles = np.array(
        [[float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in raw]
    )
    return candles  # columns: open, high, low, close


# ---------------------------------------------------------------------
# 2. INDICATORS (same math as the C version)
# ---------------------------------------------------------------------
def z_score(closes: np.ndarray, period: int) -> float:
    window = closes[-period:]
    mean = window.mean()
    std = window.std()
    if std == 0:
        return 0.0
    return (closes[-1] - mean) / std


def garman_klass_volatility(candles: np.ndarray, period: int) -> float:
    """candles columns: open, high, low, close"""
    window = candles[-period:]
    o, h, l, c = window[:, 0], window[:, 1], window[:, 2], window[:, 3]
    log_hl = np.log(h / l)
    log_co = np.log(c / o)
    term1 = 0.5 * log_hl**2
    term2 = (2 * np.log(2) - 1) * log_co**2
    variance = (term1 - term2).mean()
    return np.sqrt(max(variance, 0)) * 100.0  # guard against tiny negative from float noise


def classify_signal(z: float) -> str:
    if z < -2.0:
        return "🟢 شراء إحصائي (انحراف سلبي قوي عن المتوسط)"
    elif z > 2.0:
        return "🔴 بيع إحصائي (انحراف إيجابي قوي عن المتوسط)"
    else:
        return "⚪ محايد - لا يوجد شذوذ إحصائي واضح"


# ---------------------------------------------------------------------
# 3. TELEGRAM DELIVERY — safe, no shell involved
# ---------------------------------------------------------------------
def send_telegram_message(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        log.error("TG_BOT_TOKEN / TG_CHAT_ID not set — cannot send message.")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=payload, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error(f"Telegram send failed: {e}")
        return False


def build_report(symbol: str, interval: str, price: float, z: float, vol: float) -> str:
    signal = classify_signal(z)
    return (
        f"🧠 *تحليل إحصائي - {symbol} ({interval})*\n"
        f"-------------------------------------\n"
        f"💵 *السعر:* ${price:,.2f}\n\n"
        f"📐 *المؤشرات:*\n"
        f"• Z-Score: {z:.2f}\n"
        f"• تقلب Garman-Klass: {vol:.3f}%\n\n"
        f"-------------------------------------\n"
        f"🎯 *الإشارة:* {signal}\n"
        f"-------------------------------------\n"
        f"⚠️ _تذكير: هذه إشارة إحصائية أولية غير مُختبرة تاريخياً (Backtested).  "
        f"لا تُستخدم كقرار تنفيذ مباشر._"
    )


# ---------------------------------------------------------------------
# 4. MAIN LOOP — runs forever, survives individual failures
# ---------------------------------------------------------------------
def run_cycle():
    candles = fetch_candles(SYMBOL, INTERVAL, CANDLE_LIMIT)
    closes = candles[:, 3]
    price = closes[-1]
    z = z_score(closes, LOOKBACK)
    vol = garman_klass_volatility(candles, LOOKBACK)
    report = build_report(SYMBOL, INTERVAL, price, z, vol)
    sent = send_telegram_message(report)
    log.info(
        f"{SYMBOL} price={price:.2f} z={z:.2f} gk_vol={vol:.3f}% "
        f"telegram_sent={sent}"
    )


def main():
    log.info(f"Starting quant_alert for {SYMBOL} ({INTERVAL}), polling every {POLL_SECONDS}s")
    while True:
        try:
            run_cycle()
        except requests.RequestException as e:
            log.error(f"Network error this cycle, will retry next cycle: {e}")
        except Exception as e:
            # Catch-all so a transient bug never kills the whole process —
            # but we still log it loudly so you notice and fix it.
            log.exception(f"Unexpected error this cycle: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
