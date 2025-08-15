
import streamlit as st
import requests, json, re, time
import pandas as pd
import numpy as np
from datetime import datetime, timezone

st.set_page_config(page_title="Forex Scanner — Twelve Data (M5/M15/H1/H4)", layout="wide")

# ------------- Indicators & helpers -------------
def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    d = series.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    macd_line = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]).abs(),
                    (df["high"] - prev_close).abs(),
                    (df["low"]  - prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr(df, period=14): return true_range(df).rolling(period).mean()

def compute_indicators(df):
    df = df.copy()
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"]  = rsi(df["close"], 14)
    ml, sl, hist = macd(df["close"])
    df["macd"] = ml; df["macd_signal"] = sl; df["macd_hist"] = hist
    df["atr14"] = atr(df, 14)
    return df

def slope(series, window=20):
    if len(series) < window+1: return 0.0
    y = np.array(series[-window:]); x = np.arange(len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)

def m15_signal_row(df_m15, df_h1, idx):
    if idx <= 1: return None
    row = df_m15.iloc[idx]; h1 = df_h1.iloc[-2]
    entry = float(row["close"])
    long_ok  = (entry > row["ema200"]) and (float(h1["close"]) > float(h1["ema200"])) and (row["macd"] > row["macd_signal"]) and (50 <= row["rsi14"] <= 70)
    short_ok = (entry < row["ema200"]) and (float(h1["close"]) < float(h1["ema200"])) and (row["macd"] < row["macd_signal"]) and (30 <= row["rsi14"] <= 50)
    if not (long_ok or short_ok): return None
    return {"time": row["datetime"], "entry": float(entry), "atr": float(row["atr14"]), "direction": "LONG" if long_ok else "SHORT"}

def forward_outcome(df_m15, sig, atr_mult, rr, max_hold_bars):
    idxs = df_m15.index[df_m15["datetime"] == sig["time"]]
    if len(idxs)==0: return None
    i = int(idxs[0]); entry = sig["entry"]; sl_dist = sig["atr"]*atr_mult
    if sig["direction"]=="LONG":
        sl = entry - sl_dist; tp = entry + sl_dist*rr
    else:
        sl = entry + sl_dist; tp = entry - sl_dist*rr
    fwd = df_m15.iloc[i+1:i+1+max_hold_bars]
    hit_tp=False; hit_sl=False
    for _, r in fwd.iterrows():
        if sig["direction"]=="LONG":
            if r["low"] <= sl: hit_sl=True
            if r["high"]>= tp: hit_tp=True
        else:
            if r["high"]>= sl: hit_sl=True
            if r["low"] <= tp: hit_tp=True
        if hit_tp or hit_sl: break
    if hit_tp and not hit_sl: return True
    if hit_sl and not hit_tp: return False
    return False

def estimate_probabilities(df_m15, df_h1, atr_mult, rr_list, max_hold_bars, window_signals):
    signals = []
    for i in range(50, len(df_m15)-1):
        s = m15_signal_row(df_m15, df_h1, i)
        if s: signals.append(s)
    signals = signals[-window_signals:]
    probs = {}
    for rr in rr_list:
        wins=0; total=0
        for sig in signals:
            res = forward_outcome(df_m15, sig, atr_mult, rr, max_hold_bars)
            if res is None: continue
            wins += 1 if res else 0; total += 1
        probs[str(rr)] = (wins/total) if total>0 else None
    return probs

# ------------- Sidebar config -------------
st.sidebar.header("🔧 Scanner Settings")
td_key_default = st.secrets.get("TWELVEDATA_API_KEY", "")
tg_token_default = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
tg_chat_default = st.secrets.get("TELEGRAM_CHAT_ID", "")
openai_key_default = st.secrets.get("OPENAI_API_KEY", "")

td_key = st.sidebar.text_input("Twelve Data API Key", value=td_key_default, type="password")
symbols_default = "EUR/USD, GBP/USD, USD/JPY, AUD/USD, NZD/USD, USD/CAD, USD/CHF, EUR/JPY, GBP/JPY, XAU/USD"
symbols_text = st.sidebar.text_area("Assets (10 comma-separated)", value=symbols_default, height=80)

interval_minutes = st.sidebar.number_input("Scan every N minutes", min_value=5, max_value=60, value=15, step=5)
window_signals = st.sidebar.slider("Past signals window (for probabilities)", 40, 200, 80, step=10)
threshold_rr1 = st.sidebar.slider("Threshold P(win) for 1:1", 0.50, 0.95, 0.80, step=0.01)
threshold_rr2 = st.sidebar.slider("Threshold P(win) for 1:2", 0.50, 0.95, 0.65, step=0.01)
atr_multiple_sl = st.sidebar.slider("ATR multiple for SL", 0.5, 2.0, 1.0, step=0.1)
max_hold_bars_m15 = st.sidebar.slider("Max M15 bars to hold", 8, 48, 24, step=2)

st.sidebar.markdown("---")
use_m5_confirmation = st.sidebar.checkbox("Require M5 micro-structure confirmation", value=True,
    help="For LONG: last closed M5 above EMA200 and MACD>signal (opp for SHORT).")
use_h4_regime = st.sidebar.checkbox("Require H4 regime filter", value=True,
    help="For LONG: H4 close above EMA200 and EMA200 slope ≥ 0 (opp for SHORT).")

st.sidebar.markdown("---")
use_telegram = st.sidebar.checkbox("Send Telegram alerts", value=bool(tg_token_default and tg_chat_default))
tg_token = st.sidebar.text_input("Telegram Bot Token", value=tg_token_default, type="password", disabled=not use_telegram)
tg_chat = st.sidebar.text_input("Telegram Chat ID", value=tg_chat_default, disabled=not use_telegram)

st.sidebar.markdown("---")
use_openai = st.sidebar.checkbox("Use OpenAI sanity check (optional)", value=bool(openai_key_default))
openai_key = st.sidebar.text_input("OpenAI API Key", value=openai_key_default, type="password", disabled=not use_openai)

# Auto-refresh to rescan
st.markdown(f"<meta http-equiv='refresh' content='{int(interval_minutes)*60}'>", unsafe_allow_html=True)

# ------------- State -------------
if "alerts" not in st.session_state: st.session_state["alerts"] = []
if "last_run" not in st.session_state: st.session_state["last_run"] = None

# ------------- Data fetch -------------
def fetch_td(symbol, interval, outputsize, api_key):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": api_key}
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    data = r.json()
    if "values" not in data: return None
    df = pd.DataFrame(data["values"])
    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def telegram_send(token, chat_id, text):
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=20)
    except Exception as e:
        st.toast(f"Telegram error: {e}")

def openai_sanity(payload, api_key):
    try:
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        system = ("ROLE: Risk checker for intraday FX signals. Approve only if P1>=0.8 or P2>=0.65 "
                  "and no red flags (ATR spike, RSI extreme, regime mismatch). Return JSON: {\"approved\":bool,\"reason\":str}")
        user = "PAYLOAD:\n" + json.dumps(payload, separators=(',',':'))
        r = requests.post("https://api.openai.com/v1/chat/completions",
                          headers=headers,
                          json={"model":"gpt-4o-mini","temperature":0.0,
                                "messages":[{"role":"system","content":system},
                                            {"role":"user","content":user}],
                                "response_format":{"type":"json_object"}},
                          timeout=30)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        return {"approved": True, "reason": f"sanity_error:{e}"}

# ------------- Core scan -------------
def run_scan():
    if not td_key:
        st.error("Enter your Twelve Data API key in the sidebar."); return []

    symbols = [s.strip() for s in symbols_text.split(",") if s.strip()]
    if len(symbols) > 10:
        st.warning("Only the first 10 symbols will be scanned.")
        symbols = symbols[:10]

    intervals = {"m5":"5min","m15":"15min","h1":"1h","h4":"4h"}
    lookback  = {"m5":300, "m15":300, "h1":500, "h4":500}

    hits = []
    for sym in symbols:
        try:
            df_m5  = fetch_td(sym, intervals["m5"],  lookback["m5"],  td_key)
            df_m15 = fetch_td(sym, intervals["m15"], lookback["m15"], td_key)
            df_h1  = fetch_td(sym, intervals["h1"],  lookback["h1"],  td_key)
            df_h4  = fetch_td(sym, intervals["h4"],  lookback["h4"],  td_key)
            if any(x is None for x in [df_m5, df_m15, df_h1, df_h4]):
                st.warning(f"{sym}: missing data from Twelve Data"); continue

            df_m5  = compute_indicators(df_m5)
            df_m15 = compute_indicators(df_m15)
            df_h1  = compute_indicators(df_h1)
            df_h4  = compute_indicators(df_h4)

            if len(df_m15) < 210 or len(df_h1) < 210:
                st.info(f"{sym}: not enough bars yet"); continue

            # --- Base signal on M15 + H1 context ---
            idx = len(df_m15)-2  # last completed M15 bar
            row = df_m15.iloc[idx]; h1 = df_h1.iloc[-2]; h4 = df_h4.iloc[-2]
            entry = float(row["close"])
            long_ok  = (entry > row["ema200"]) and (float(h1["close"]) > float(h1["ema200"])) and (row["macd"] > row["macd_signal"]) and (50 <= row["rsi14"] <= 70)
            short_ok = (entry < row["ema200"]) and (float(h1["close"]) < float(h1["ema200"])) and (row["macd"] < row["macd_signal"]) and (30 <= row["rsi14"] <= 50)
            if not (long_ok or short_ok):
                continue
            direction = "LONG" if long_ok else "SHORT"

            # --- NEW: M5 micro-structure confirmation (toggle) ---
            m5_last = df_m5.iloc[-2]
            if use_m5_confirmation:
                if direction=="LONG":
                    if not (m5_last["close"] > m5_last["ema200"] and m5_last["macd"] > m5_last["macd_signal"]):
                        continue
                else:
                    if not (m5_last["close"] < m5_last["ema200"] and m5_last["macd"] < m5_last["macd_signal"]):
                        continue

            # --- NEW: H4 regime filter (toggle) ---
            if use_h4_regime:
                h4_slope = slope(df_h4["ema200"].fillna(method="ffill").fillna(0).tolist(), window=20)
                if direction=="LONG":
                    if not (float(h4["close"]) > float(h4["ema200"]) and h4_slope >= 0):
                        continue
                else:
                    if not (float(h4["close"]) < float(h4["ema200"]) and h4_slope <= 0):
                        continue

            atr14 = float(row["atr14"]); sl_dist = atr14 * atr_multiple_sl
            sl = entry - sl_dist if direction=="LONG" else entry + sl_dist
            tp_rr1 = entry + sl_dist if direction=="LONG" else entry - sl_dist

            # Probabilities from history of M15 signals (validated with H1) — unchanged
            probs = estimate_probabilities(df_m15, df_h1, atr_multiple_sl, [1.0, 2.0], max_hold_bars_m15, window_signals)
            p1, p2 = probs.get("1.0"), probs.get("2.0")
            passes = ((p1 is not None and p1 >= threshold_rr1) or (p2 is not None and p2 >= threshold_rr2))
            if not passes:
                continue

            # Optional sanity check
            approved = True; reason = "ok"
            if use_openai and openai_key:
                payload = {"symbol": sym, "direction": direction, "entry": entry, "sl": sl, "tp1": tp_rr1,
                           "p_rr1": p1, "p_rr2": p2, "rsi14": float(row["rsi14"]), "atr14": float(atr14),
                           "h1_vs_ema": float(h1["close"] - h1["ema200"]), "h4_vs_ema": float(h4["close"] - h4["ema200"])}
                verdict = openai_sanity(payload, openai_key)
                approved = bool(verdict.get("approved", True)); reason = verdict.get("reason", "ok")
            if not approved:
                st.info(f"{sym}: sanity rejected ({reason})")
                continue

            when = pd.Timestamp(row["datetime"]).tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")
            msg = (
                f"✅ {sym} {direction} | {when}\n"
                f"Entry {entry:.5f} | SL {sl:.5f} | TP(1:1) {tp_rr1:.5f}\n"
                f"P1:1={p1:.2% if p1 is not None else 'n/a'} | P1:2={p2:.2% if p2 is not None else 'n/a'}\n"
                f"Filters: M5={'ON' if use_m5_confirmation else 'OFF'}, H4={'ON' if use_h4_regime else 'OFF'} | Sanity: {reason}"
            )
            st.session_state["alerts"].append(msg)
            if use_telegram and tg_token and tg_chat:
                telegram_send(tg_token, tg_chat, msg)

        except Exception as e:
            st.warning(f"{sym}: error {e}")

# ------------- UI -------------
st.title("📈 Forex Scanner — Twelve Data (M5 + M15 + H1 + H4)")
st.caption("Scans up to 10 assets. Base signal on M15 with H1 context, plus optional M5 confirmation and H4 regime filter. Telegram alerts when thresholds are met.")

colA, colB = st.columns([1,1])
with colA:
    run_now = st.button("▶️ Run Scan Now")
with colB:
    if st.session_state["last_run"]:
        st.write(f"Last run: {st.session_state['last_run']}")

# Trigger
should_run = run_now or (st.session_state["last_run"] is None)
if not should_run and st.session_state["last_run"]:
    should_run = True  # each reload triggers a run

if should_run:
    with st.spinner("Scanning…"):
        run_scan()
    st.session_state["last_run"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# Alerts panel
st.subheader("🔔 Alerts")
if st.session_state["alerts"]:
    for a in reversed(st.session_state["alerts"][-50:]):
        st.code(a)
else:
    st.info("No alerts yet. They will appear here when thresholds are met.")

# Secrets hint
st.markdown(
    """
**Secrets example** (recommended):  
```
TWELVEDATA_API_KEY = "your_td_key"
TELEGRAM_BOT_TOKEN = "123456:ABC..."
TELEGRAM_CHAT_ID   = "123456789"
OPENAI_API_KEY     = "sk-..."  # optional
```
"""
)
