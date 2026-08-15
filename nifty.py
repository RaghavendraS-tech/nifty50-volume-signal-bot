#!/usr/bin/env python3
import os
import sys
import time
import csv
import json
import logging
import requests
import numpy as np
import pandas as pd
import pytz

from datetime import datetime, timedelta
from collections import deque

# === CONFIG ===
BASE_URL           = "https://www.nseindia.com"
LIVE_PAGE_URL      = BASE_URL + "/market-data/live-equity-market?symbol=NIFTY%2050"
API_URL            = BASE_URL + "/api/equity-stockIndices?index=NIFTY%2050"

POLL_INTERVAL      = 5                              # seconds
CALIBRATE_INTERVAL = timedelta(minutes=1)
SMOOTH_ALPHA       = 0.6
MAX_PCT_CHANGE     = 0.05

MIN_MOM_WINDOW     = 1
MAX_MOM_WINDOW     = 3
WHIPSAW_THRESHOLD  = 3
MIN_PRICE_DELTAPCT = 0.0005

TRADE_LOG          = "trade_log.csv"
STATE_FILE         = "state.json"
LOG_FILE           = "bot.log"
IST                = pytz.timezone("Asia/Kolkata")

HEADERS = {
    "User-Agent":       "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept":           "application/json, text/plain, */*",
    "Referer":          LIVE_PAGE_URL,
    "X-Requested-With": "XMLHttpRequest",
    "Cache-Control":    "no-cache",
    "Pragma":           "no-cache",
}

class C:
    BUY   = "\033[92m"
    SELL  = "\033[91m"
    HOLD  = "\033[93m"
    RESET = "\033[0m"

# === LOGGING SETUP ===
logger = logging.getLogger("nifty_bot")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
logger.addHandler(ch)

# === SIMPLE FILE LOCK FOR CSV APPEND ===
from threading import Lock
file_lock = Lock()
def locked_csv_append(path, row, header):
    exists = os.path.isfile(path)
    with file_lock:
        with open(path, "a+", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(header)
            w.writerow(row)

# === GLOBAL STATE ===
last_signal     = None
entry_price     = None
last_trade_ts   = None
ratio_threshold = 1.0
MOMENTUM_WINDOW = MIN_MOM_WINDOW
volume_hist     = deque(maxlen=500)
raw_hist        = deque(maxlen=500)
consec_fails    = 0

# === STATE I/O ===
def load_state():
    global last_signal, entry_price, last_trade_ts, ratio_threshold, MOMENTUM_WINDOW
    if not os.path.isfile(STATE_FILE):
        return
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        last_signal     = data.get("last_signal")
        entry_price     = data.get("entry_price")
        ratio_threshold = data.get("ratio_threshold", ratio_threshold)
        MOMENTUM_WINDOW = data.get("momentum_window", MOMENTUM_WINDOW)
        ts = data.get("last_trade_ts")
        if ts:
            last_trade_ts = datetime.fromisoformat(ts)
        logger.info("State loaded")
    except Exception:
        logger.exception("Failed to load state")

def save_state():
    s = {
        "last_signal":     last_signal,
        "entry_price":     entry_price,
        "last_trade_ts":   last_trade_ts.isoformat() if last_trade_ts else None,
        "ratio_threshold": ratio_threshold,
        "momentum_window": MOMENTUM_WINDOW
    }
    try:
        with open(STATE_FILE, "w+") as f:
            json.dump(s, f)
        os.chmod(STATE_FILE, 0o666)
    except Exception:
        logger.exception("Failed to save state")

# === HTTP SESSION SETUP ===
def setup_session(retries=3, backoff=1):
    for attempt in range(retries):
        try:
            sess = requests.Session()
            sess.headers.update(HEADERS)
            sess.get(BASE_URL, timeout=3)
            sess.get(LIVE_PAGE_URL, timeout=3)
            return sess
        except Exception:
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError("Unable to establish session after retries")

# === DATA EXTRACTION ===
def fetch_data(sess):
    global consec_fails
    try:
        resp = sess.get(API_URL, params={"_": int(time.time() * 1e3)}, timeout=5)
        resp.raise_for_status()
        consec_fails = 0
        return resp.json().get("data", [])
    except Exception:
        consec_fails += 1
        logger.warning(f"Fetch failure #{consec_fails}")
        return []

def extract_volume(data):
    df = pd.json_normalize(data)
    idx = df[df["symbol"] == "NIFTY 50"]
    if idx.empty:
        raise ValueError("NIFTY 50 not found in data")
    price = float(idx["lastPrice"].iloc[0])
    cons  = df[df["symbol"] != "NIFTY 50"].copy()
    cons["totalTradedVolume"] = pd.to_numeric(cons["totalTradedVolume"], errors="coerce").fillna(0)
    cons["pChange"]           = pd.to_numeric(cons["pChange"], errors="coerce").fillna(0)
    adv   = int(cons[cons["pChange"] > 0]["totalTradedVolume"].sum())
    bear  = int(cons[cons["pChange"] < 0]["totalTradedVolume"].sum()) or 1
    return adv, bear, round(price, 2)

# === THRESHOLD & MOMENTUM ADJUSTMENTS ===
def recompute_threshold(now):
    global ratio_threshold
    cutoff = now - CALIBRATE_INTERVAL
    while volume_hist and volume_hist[0][0] < cutoff:
        volume_hist.popleft()
    ratios = [adv/bear for _, adv, bear in volume_hist]
    if ratios:
        raw75 = float(np.percentile(ratios, 75))
        target = ratio_threshold + SMOOTH_ALPHA * (raw75 - ratio_threshold)
        maxd   = ratio_threshold * MAX_PCT_CHANGE
        delta  = max(-maxd, min(target - ratio_threshold, maxd))
        ratio_threshold = round(ratio_threshold + delta, 4)
    return ratio_threshold

def adjust_momentum(now):
    global MOMENTUM_WINDOW
    cutoff = now - CALIBRATE_INTERVAL
    while raw_hist and raw_hist[0][0] < cutoff:
        raw_hist.popleft()
    sigs  = [s for _, s in raw_hist]
    flips = sum(1 for i in range(1, len(sigs)) if sigs[i] != sigs[i-1])
    if flips >= WHIPSAW_THRESHOLD and MOMENTUM_WINDOW < MAX_MOM_WINDOW:
        MOMENTUM_WINDOW += 1
    elif flips == 0 and MOMENTUM_WINDOW > MIN_MOM_WINDOW:
        MOMENTUM_WINDOW -= 1

# === TRADE LOGGING ===
def log_trade(ts, side, ent, ext):
    pnl = round((ext - ent) if side == "BUY" else (ent - ext), 2)
    locked_csv_append(
        TRADE_LOG,
        [ts, side, ent, ext, f"{pnl:+.2f}"],
        ["Timestamp", "Signal", "Entry", "Exit", "PnL"]
    )

# === MAIN LOOP ===
def run_live():
    global last_signal, entry_price, last_trade_ts, consec_fails

    sess = setup_session()
    load_state()

    next_cal = datetime.now(IST) + CALIBRATE_INTERVAL

    try:
        while True:
            if consec_fails >= 3:
                sess = setup_session()
                consec_fails = 0

            data = fetch_data(sess)
            if not data:
                time.sleep(POLL_INTERVAL)
                continue

            now = datetime.now(IST)
            adv, bear, price = extract_volume(data)
            volume_hist.append((now, adv, bear))

            # Recalibrate periodically
            if now >= next_cal:
                recompute_threshold(now)
                adjust_momentum(now)
                save_state()
                next_cal = now + CALIBRATE_INTERVAL
                logger.info(f"Recalibrated → Thr={ratio_threshold:.2f}, MomWin={MOMENTUM_WINDOW}")

            # Raw signal
            raw = (
                "BUY"  if adv > bear * ratio_threshold else
                "SELL" if bear > adv * ratio_threshold else
                "HOLD"
            )
            raw_hist.append((now, raw))

            # Momentum-confirmed signal
            window = [s for _, s in list(raw_hist)[-MOMENTUM_WINDOW:]]
            if len(window) == MOMENTUM_WINDOW and len(set(window)) == 1:
                signal = window[0]
            else:
                signal = last_signal or "HOLD"

            # Suppress tiny oscillations
            if last_signal and entry_price:
                delta_pct = abs(price - entry_price) / entry_price
                if delta_pct < MIN_PRICE_DELTAPCT:
                    signal = last_signal

            ts = now.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"[{ts}] AdvVol={adv:,}  BearVol={bear:,}  Thr={ratio_threshold:.2f}  Price={price}")
            color = C.BUY if signal=="BUY" else C.SELL if signal=="SELL" else C.HOLD
            logger.info(f" SIGNAL → {color}{signal}{C.RESET}")

            # Execute entry/exit on signal changes
            if signal in ("BUY","SELL") and signal != last_signal:
                # exit previous
                if last_signal and entry_price:
                    log_trade(ts, last_signal, entry_price, price)
                    logger.info(f" EXIT {last_signal} @ {price}")
                # enter new
                last_signal, entry_price, last_trade_ts = signal, price, now
                save_state()
                logger.info(f" ENTER {signal} @ {price}")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Interrupted by user; saving state and exiting.")
        save_state()
        sys.exit(0)
    except Exception:
        logger.exception("Unhandled error in run_live; restarting loop.")
        time.sleep(POLL_INTERVAL)
        run_live()  # restart on any unexpected crash

if __name__ == "__main__":
    run_live()
