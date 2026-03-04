# app.py
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# -------------------- Page Config --------------------
st.set_page_config(page_title="MEXC Small-Cap Scanner", layout="wide")

# -------------------- Session State Init --------------------
if 'all_pairs' not in st.session_state:
    st.session_state.all_pairs = []
if 'scanned_results' not in st.session_state:
    st.session_state.scanned_results = []
if 'batch_index' not in st.session_state:
    st.session_state.batch_index = 0
if 'batch_size' not in st.session_state:
    st.session_state.batch_size = 50
if 'scan_complete' not in st.session_state:
    st.session_state.scan_complete = False
if 'filtered_watchlist' not in st.session_state:
    st.session_state.filtered_watchlist = pd.DataFrame()

# -------------------- Caching Helpers --------------------
@st.cache_data(ttl=3600)  # 1 hour
def get_small_cap_futures_pairs():
    """Fetch all USDT perpetuals from MEXC and filter by market cap (if possible)."""
    try:
        exchange = ccxt.mexc({'enableRateLimit': True})
        markets = exchange.load_markets()
        futures_pairs = [
            symbol for symbol in markets
            if symbol.endswith('/USDT')
            and markets[symbol].get('future', False)
            and markets[symbol]['active']
        ]
    except Exception as e:
        st.error(f"Failed to load markets from MEXC: {e}")
        return []

    # Try CoinGecko for market cap filter
    try:
        coingecko_url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_asc&per_page=250&page=1&sparkline=false'
        resp = requests.get(coingecko_url, timeout=10)
        if resp.status_code == 200:
            coins = resp.json()
            small_caps = {
                coin['symbol'].upper() + '/USDT': coin['market_cap']
                for coin in coins
                if coin['market_cap'] and coin['market_cap'] < 50_000_000
            }
            # Intersection with MEXC pairs
            pairs = [p for p in futures_pairs if p in small_caps]
            if pairs:
                return pairs[:500]
    except Exception as e:
        st.warning(f"CoinGecko API failed: {e}. Falling back to all MEXC futures (may include large caps).")

    # Fallback: return all futures (user can filter later)
    return futures_pairs[:500]

@st.cache_data(ttl=1800)  # 30 minutes
def fetch_ohlcv(exchange_id, symbol, timeframe='1d', limit=500):
    """Fetch OHLCV data for a symbol with error handling."""
    try:
        exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.warning(f"Failed to fetch {symbol} {timeframe}: {e}")
        return None

# -------------------- Analysis Function (Rule-Based) --------------------
def analyze_pair(pair):
    """Run technical analysis on a pair and return a signal dict or None."""
    # Fetch daily and hourly data
    df_daily = fetch_ohlcv('mexc', pair, '1d', 300)  # 300 days enough for most indicators
    df_hourly = fetch_ohlcv('mexc', pair, '1h', 168)  # 7 days

    if df_daily is None or df_hourly is None or len(df_daily) < 50:
        return None

    try:
        # --- Daily indicators ---
        close = df_daily['close']
        low = df_daily['low']
        volume = df_daily['volume']

        # Moving averages
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal

        # Bollinger Bands (20,2)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / bb_mid

        # Volume surge
        vol_ma20 = volume.rolling(20).mean()
        vol_surge_pct = ((volume / vol_ma20) - 1) * 100

        # Proximity to 52-week low
        year_low = low.rolling(365).min()
        proximity_to_low = ((close - year_low) / year_low) * 100

        # Trend strength: 50-day slope (linear regression)
        x = np.arange(50)
        y = close[-50:].values
        if len(y) == 50 and not np.any(np.isnan(y)):
            slope = np.polyfit(x, y, 1)[0] / close.iloc[-1]  # normalized slope
        else:
            slope = 0

        # Volatility (ATR)
        tr = pd.concat([high - low for high, low in zip(df_daily['high'], df_daily['low'])], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1] / close.iloc[-1] * 100

        # --- Hourly early pump detection ---
        hourly_close = df_hourly['close']
        hourly_vol = df_hourly['volume']
        vol_ma4 = hourly_vol.rolling(4).mean()
        hourly_vol_surge = hourly_vol.iloc[-1] / vol_ma4.iloc[-1] if vol_ma4.iloc[-1] > 0 else 1
        price_change_24h = (hourly_close.iloc[-1] - hourly_close.iloc[-24]) / hourly_close.iloc[-24] if len(hourly_close) >= 24 else 0
        early_pump = (hourly_vol_surge > 2) and (abs(price_change_24h) < 0.1)

        # --- Composite score (0-100) ---
        # Factors: RSI (oversold good), volume surge, distance from low, trend slope, volatility (low good)
        score = 0
        # RSI: below 30 is excellent, below 40 good
        rsi_val = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        if rsi_val < 30:
            score += 30
        elif rsi_val < 40:
            score += 20
        elif rsi_val < 50:
            score += 10

        # Proximity to low: closer to low is better (inverse)
        prox = proximity_to_low.iloc[-1] if not pd.isna(proximity_to_low.iloc[-1]) else 100
        if prox < 5:
            score += 30
        elif prox < 10:
            score += 20
        elif prox < 20:
            score += 10

        # Volume surge: >100% gives points
        vs = vol_surge_pct.iloc[-1] if not pd.isna(vol_surge_pct.iloc[-1]) else 0
        if vs > 200:
            score += 20
        elif vs > 100:
            score += 15
        elif vs > 50:
            score += 5

        # Trend slope: positive slope adds points
        if slope > 0.01:
            score += 20
        elif slope > 0:
            score += 10

        # Volatility: lower is better for safety
        if atr < 5:
            score += 10
        elif atr < 10:
            score += 5

        # Bonus for early pump
        if early_pump:
            score += 15

        # Grade based on score
        if score >= 80:
            grade = 'A+'
        elif score >= 70:
            grade = 'A'
        elif score >= 60:
            grade = 'B+'
        elif score >= 50:
            grade = 'B'
        elif score >= 40:
            grade = 'C+'
        else:
            grade = 'C'

        # Entry/SL/TP
        entry = close.iloc[-1]
        sl = entry * 0.92  # 8% stop
        tp = entry * 2.0   # target 100% gain (adjustable later)

        return {
            'Pair': pair,
            'Grade': grade,
            'Score': score,
            'Current Price': entry,
            'Entry': entry,
            'Stop Loss': sl,
            'Take Profit': tp,
            'Exit Condition': 'Sell on 100% gain or RSI > 70',
            'RSI': round(rsi_val, 2),
            'ATL Proximity %': round(prox, 2),
            'Volume Surge %': round(vs, 2),
            'Early Pump': early_pump,
            'Trend Slope %': round(slope * 100, 2),
            'ATR %': round(atr, 2)
        }

    except Exception as e:
        st.warning(f"Error analyzing {pair}: {e}")
        traceback.print_exc()
        return None

# -------------------- Batch Processing with Threads --------------------
def scan_batch(pairs, max_workers=5):
    """Analyze a list of pairs concurrently with a thread pool."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {executor.submit(analyze_pair, pair): pair for pair in pairs}
        for future in as_completed(future_to_pair):
            pair = future_to_pair[future]
            try:
                res = future.result(timeout=30)
                if res:
                    results.append(res)
            except Exception as e:
                st.warning(f"Exception for {pair}: {e}")
    return results

# -------------------- UI --------------------
st.title("🚀 MEXC Small‑Cap Scanner (Improved)")
st.markdown("Scan for potential swing trade setups using rule‑based scoring (no ML).")

scan_mode = st.radio("Scan Mode", ["Bottoms (Near Lows)", "Early Pumps"], horizontal=True)

col1, col2, col3 = st.columns(3)
with col1:
    batch_size = st.slider("Batch size (pairs per batch)", 20, 200, st.session_state.batch_size, 10)
    st.session_state.batch_size = batch_size
with col2:
    concurrency = st.slider("Concurrent threads", 1, 10, 5)
with col3:
    if st.button("🔄 Reset & New Scan"):
        for key in list(st.session_state.keys()):
            if key not in ['batch_size']:
                del st.session_state[key]
        st.rerun()

# Load pairs if not already
if not st.session_state.all_pairs:
    with st.spinner("Loading small‑cap pairs..."):
        st.session_state.all_pairs = get_small_cap_futures_pairs()
        st.session_state.batch_index = 0
        st.session_state.scanned_results = []
        st.session_state.scan_complete = False
    st.success(f"Found {len(st.session_state.all_pairs)} pairs to scan")

total_pairs = len(st.session_state.all_pairs)
scanned_so_far = len(st.session_state.scanned_results)
progress = scanned_so_far / total_pairs if total_pairs > 0 else 0

st.progress(progress)
st.caption(f"Scanned: {scanned_so_far} / {total_pairs} pairs")

# Scan button
if st.button("▶️ Scan Next Batch", disabled=(scanned_so_far >= total_pairs)):
    start_idx = scanned_so_far
    end_idx = min(start_idx + st.session_state.batch_size, total_pairs)
    batch_pairs = st.session_state.all_pairs[start_idx:end_idx]

    with st.status(f"Scanning batch {st.session_state.batch_index+1}...", expanded=True) as status:
        st.write(f"Analyzing {len(batch_pairs)} pairs...")
        new_results = scan_batch(batch_pairs, max_workers=concurrency)
        st.session_state.scanned_results.extend(new_results)
        st.session_state.batch_index += 1
        status.update(label=f"Batch complete! {len(new_results)} signals found.", state="complete")

    st.rerun()

# Once scanning is done (or anytime), apply filters
if st.session_state.scanned_results:
    df_all = pd.DataFrame(st.session_state.scanned_results)

    # Filter based on mode
    if scan_mode == "Bottoms (Near Lows)":
        filtered = df_all[
            (df_all['ATL Proximity %'] < 20) &
            (df_all['RSI'] < 45) &
            (df_all['Volume Surge %'] > 30) &
            (df_all['Grade'].isin(['A+', 'A', 'B+', 'B']))
        ].sort_values('Score', ascending=False)
    else:  # Early Pumps
        filtered = df_all[
            (df_all['Early Pump'] == True) &
            (df_all['Volume Surge %'] > 80) &
            (df_all['Grade'].isin(['A+', 'A', 'B+']))
        ].sort_values('Score', ascending=False)

    st.session_state.filtered_watchlist = filtered

    # Display watchlist
    st.subheader("📊 Your Watchlist")
    if not filtered.empty:
        # Select columns to show
        cols = ['Pair', 'Grade', 'Score', 'Current Price', 'Entry', 'Stop Loss', 'Take Profit',
                'Exit Condition', 'RSI', 'ATL Proximity %', 'Volume Surge %', 'ATR %']
        st.dataframe(filtered[cols].style.format({
            'Current Price': '{:.8f}',
            'Entry': '{:.8f}',
            'Stop Loss': '{:.8f}',
            'Take Profit': '{:.8f}',
            'RSI': '{:.1f}',
            'ATL Proximity %': '{:.1f}',
            'Volume Surge %': '{:.1f}',
            'ATR %': '{:.2f}'
        }), use_container_width=True)
    else:
        st.info("No signals match your current filters. Try adjusting the scan mode or batch more pairs.")

    # Option to show all raw data
    with st.expander("Show all scanned results (raw)"):
        st.dataframe(df_all)

# Sidebar info
st.sidebar.markdown("""
**How it works**
- Fetches USDT perpetuals from MEXC.
- Filters by market cap via CoinGecko (fallback to all).
- Computes RSI, volume surge, distance from 52‑week low, trend, volatility.
- Assigns a composite score (0‑100) and grade.
- Two scan modes: bottoms (near low) or early pumps (volume spike without price move).
- Batch scanning with concurrency for speed.
""")
