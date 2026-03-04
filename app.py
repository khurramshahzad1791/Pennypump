# app.py
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# -------------------- Page Config --------------------
st.set_page_config(page_title="MEXC Penny Stock Breakout Scanner", layout="wide")

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

# -------------------- Helper Functions --------------------
@st.cache_data(ttl=3600)
def get_mexc_futures_pairs():
    """Get all USDT perpetual futures from MEXC."""
    try:
        exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 30000})
        markets = exchange.load_markets()
        pairs = [
            symbol for symbol in markets
            if symbol.endswith('/USDT')
            and markets[symbol].get('future', False)
            and markets[symbol]['active']
        ]
        return pairs
    except Exception as e:
        st.error(f"❌ MEXC API error: {e}")
        return []

def get_small_cap_futures_pairs():
    """
    Attempt to load small‑cap pairs via CoinGecko.
    If that fails, return all MEXC futures.
    """
    mexc_pairs = get_mexc_futures_pairs()
    if not mexc_pairs:
        st.error("No pairs retrieved from MEXC. Please use manual input.")
        return []

    st.info(f"✅ Retrieved {len(mexc_pairs)} USDT perpetuals from MEXC")

    # Try CoinGecko for market cap filter
    try:
        coingecko_url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_asc&per_page=250&page=1&sparkline=false'
        resp = requests.get(coingecko_url, timeout=10)
        if resp.status_code == 200:
            coins = resp.json()
            small_caps = {}
            for coin in coins:
                if coin.get('market_cap') and coin['market_cap'] < 50_000_000:
                    symbol = coin['symbol'].upper() + '/USDT'
                    small_caps[symbol] = coin['market_cap']
            st.info(f"✅ CoinGecko: {len(small_caps)} small‑cap coins found")
            if small_caps:
                filtered = [p for p in mexc_pairs if p in small_caps]
                if filtered:
                    st.success(f"✅ {len(filtered)} small‑cap futures on MEXC")
                    return filtered[:500]
                else:
                    st.warning("No overlap between MEXC and CoinGecko small caps.")
            else:
                st.warning("CoinGecko returned zero small‑cap coins.")
        else:
            st.warning(f"CoinGecko HTTP {resp.status_code}")
    except Exception as e:
        st.warning(f"CoinGecko API failed: {e}")

    # Fallback: return all MEXC futures
    st.warning("Falling back to ALL MEXC futures (may include large caps).")
    return mexc_pairs[:500]

@st.cache_data(ttl=1800)
def fetch_ohlcv(symbol, timeframe='1d', limit=500):
    """Fetch OHLCV data for a symbol from MEXC."""
    try:
        exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 30000})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.warning(f"Failed to fetch {symbol} {timeframe}: {e}")
        return None

def detect_breakout(df_daily, df_hourly):
    """
    Enhanced breakout detection:
    - Price above 20-day high (breakout)
    - Volume surge
    - RSI > 50 (momentum)
    - Recent consolidation (tight range)
    Returns a score and breakout flag.
    """
    close = df_daily['close']
    high = df_daily['high']
    low = df_daily['low']
    volume = df_daily['volume']

    # 20-day high
    recent_high = high.rolling(20).max()
    above_high = close.iloc[-1] > recent_high.iloc[-2]  # close above previous 20-day high

    # Volume surge (10-day avg)
    vol_ma10 = volume.rolling(10).mean()
    vol_surge = volume.iloc[-1] / vol_ma10.iloc[-1] if vol_ma10.iloc[-1] > 0 else 1

    # RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    # Consolidation: recent range narrow compared to longer range
    range_10 = (high.rolling(10).max() - low.rolling(10).min()) / close.rolling(10).mean()
    range_50 = (high.rolling(50).max() - low.rolling(50).min()) / close.rolling(50).mean()
    consolidation = (range_10.iloc[-1] / range_50.iloc[-1]) < 0.5 if not pd.isna(range_50.iloc[-1]) else False

    # Breakout score
    score = 0
    if above_high:
        score += 30
    if vol_surge > 2:
        score += 30
    elif vol_surge > 1.5:
        score += 15
    if rsi_val > 60:
        score += 20
    elif rsi_val > 50:
        score += 10
    if consolidation:
        score += 20

    # Early pump detection (hourly)
    hourly_close = df_hourly['close']
    hourly_vol = df_hourly['volume']
    vol_ma4 = hourly_vol.rolling(4).mean()
    hourly_vol_surge = hourly_vol.iloc[-1] / vol_ma4.iloc[-1] if vol_ma4.iloc[-1] > 0 else 1
    price_change_24h = (hourly_close.iloc[-1] - hourly_close.iloc[-24]) / hourly_close.iloc[-24] if len(hourly_close) >= 24 else 0
    early_pump = (hourly_vol_surge > 2) and (abs(price_change_24h) < 0.05)

    if early_pump:
        score += 25

    # Grade
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

    entry = close.iloc[-1]
    sl = entry * 0.92
    tp = entry * 2.0

    return {
        'Breakout Score': score,
        'Grade': grade,
        'Above 20d High': above_high,
        'Volume Surge': round(vol_surge, 2),
        'RSI': round(rsi_val, 2),
        'Consolidation': consolidation,
        'Early Pump': early_pump,
        'Entry': entry,
        'Stop Loss': sl,
        'Take Profit': tp,
        'Exit Condition': 'Sell on 100% gain or RSI > 75'
    }

def analyze_pair(pair):
    """Analyze a pair for breakout signals."""
    df_daily = fetch_ohlcv(pair, '1d', 200)
    df_hourly = fetch_ohlcv(pair, '1h', 168)

    if df_daily is None or df_hourly is None or len(df_daily) < 50:
        return None

    try:
        breakout = detect_breakout(df_daily, df_hourly)
        if breakout['Breakout Score'] < 30:  # ignore very weak signals
            return None

        result = {
            'Pair': pair,
            'Current Price': df_daily['close'].iloc[-1],
            **breakout
        }
        return result
    except Exception as e:
        st.warning(f"Error analyzing {pair}: {e}")
        return None

def scan_batch(pairs, max_workers=5):
    """Analyze a list of pairs concurrently."""
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
st.title("🚀 MEXC Penny Stock Breakout Scanner")
st.markdown("Detect early breakouts in small‑cap futures using volume, price action, and momentum.")

# Sidebar: manual override and default list
with st.sidebar:
    st.header("Pair Selection")
    option = st.radio("Source", ["Auto (MEXC + CoinGecko)", "Manual Input", "Use Default List"])

    if option == "Manual Input":
        manual_pairs = st.text_area("Enter pairs (one per line, e.g. BTC/USDT)")
        if st.button("Load Manual Pairs"):
            if manual_pairs.strip():
                pairs = [p.strip() for p in manual_pairs.split('\n') if p.strip()]
                st.session_state.all_pairs = pairs
                st.session_state.batch_index = 0
                st.session_state.scanned_results = []
                st.rerun()
    elif option == "Use Default List":
        default_pairs = [
            "ALICE/USDT", "BAKE/USDT", "CELR/USDT", "CHZ/USDT", "DENT/USDT",
            "DOGE/USDT", "ELF/USDT", "ENJ/USDT", "FET/USDT", "GRT/USDT",
            "HOT/USDT", "IOST/USDT", "IOTA/USDT", "KAVA/USDT", "KSM/USDT",
            "LINA/USDT", "LIT/USDT", "MANA/USDT", "MATIC/USDT", "NEAR/USDT",
            "OCEAN/USDT", "OMG/USDT", "ONE/USDT", "ONT/USDT", "QTUM/USDT",
            "REN/USDT", "RVN/USDT", "SAND/USDT", "SKL/USDT", "SNX/USDT",
            "STMX/USDT", "STORJ/USDT", "SUSHI/USDT", "TOMO/USDT", "TRB/USDT",
            "UNI/USDT", "VET/USDT", "WAVES/USDT", "XEM/USDT", "ZEC/USDT",
            "ZIL/USDT", "ZRX/USDT"
        ]
        if st.button("Load Default Pairs"):
            st.session_state.all_pairs = default_pairs
            st.session_state.batch_index = 0
            st.session_state.scanned_results = []
            st.rerun()

    st.markdown("---")
    st.header("Scan Settings")
    scan_mode = st.radio("Focus", ["Breakouts", "Early Pumps"], horizontal=True)

col1, col2, col3 = st.columns(3)
with col1:
    batch_size = st.slider("Batch size", 20, 200, st.session_state.batch_size, 10)
    st.session_state.batch_size = batch_size
with col2:
    concurrency = st.slider("Concurrent threads", 1, 10, 5)
with col3:
    if st.button("🔄 Reset All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Load pairs if not already
if not st.session_state.all_pairs:
    if option == "Auto (MEXC + CoinGecko)":
        with st.spinner("Loading pairs from MEXC and CoinGecko..."):
            st.session_state.all_pairs = get_small_cap_futures_pairs()
    elif option == "Manual Input":
        st.info("Please enter pairs manually in the sidebar.")
        st.stop()
    else:  # Default list will be loaded by button, so here we just wait
        st.info("Select a source from the sidebar and click 'Load'.")
        st.stop()

if not st.session_state.all_pairs:
    st.error("No pairs loaded. Please use manual input or default list.")
    st.stop()

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

# Apply filters and display watchlist
if st.session_state.scanned_results:
    df_all = pd.DataFrame(st.session_state.scanned_results)

    if scan_mode == "Breakouts":
        filtered = df_all[
            (df_all['Above 20d High'] == True) &
            (df_all['Volume Surge'] > 1.5) &
            (df_all['RSI'] > 50) &
            (df_all['Grade'].isin(['A+', 'A', 'B+']))
        ].sort_values('Breakout Score', ascending=False)
    else:  # Early Pumps
        filtered = df_all[
            (df_all['Early Pump'] == True) &
            (df_all['Volume Surge'] > 2) &
            (df_all['Grade'].isin(['A+', 'A']))
        ].sort_values('Breakout Score', ascending=False)

    st.session_state.filtered_watchlist = filtered

    st.subheader("📊 Breakout Signals")
    if not filtered.empty:
        cols = ['Pair', 'Grade', 'Breakout Score', 'Current Price', 'Entry', 'Stop Loss', 'Take Profit',
                'Above 20d High', 'Volume Surge', 'RSI', 'Consolidation', 'Early Pump']
        st.dataframe(filtered[cols].style.format({
            'Current Price': '{:.8f}',
            'Entry': '{:.8f}',
            'Stop Loss': '{:.8f}',
            'Take Profit': '{:.8f}',
            'Breakout Score': '{:.0f}',
            'Volume Surge': '{:.2f}',
            'RSI': '{:.1f}'
        }), use_container_width=True)
    else:
        st.info("No breakout signals match your criteria. Try scanning more pairs or adjusting filters.")

    with st.expander("Show all scanned results"):
        st.dataframe(df_all)
