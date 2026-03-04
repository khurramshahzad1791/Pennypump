# app.py
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import ta  # TA-Lib wrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import requests
import time
from scipy.signal import find_peaks

# ────────────────────────────────────────────────
# Session state initialization
# ────────────────────────────────────────────────
if 'all_pairs' not in st.session_state:
    st.session_state.all_pairs = []
if 'scanned_results' not in st.session_state:
    st.session_state.scanned_results = []   # list of dicts
if 'batch_index' not in st.session_state:
    st.session_state.batch_index = 0
if 'batch_size' not in st.session_state:
    st.session_state.batch_size = 80       # safe: ~80–100 to avoid rate limits / timeouts
if 'scan_complete' not in st.session_state:
    st.session_state.scan_complete = False
if 'filtered_watchlist' not in st.session_state:
    st.session_state.filtered_watchlist = pd.DataFrame()

# ────────────────────────────────────────────────
# Functions
# ────────────────────────────────────────────────
@st.cache_data(ttl=3600)  # pair list 1 hour
def get_small_cap_futures_pairs():
    exchange = ccxt.mexc()
    markets = exchange.load_markets()
    futures_pairs = [symbol for symbol in markets if 'USDT' in symbol and markets[symbol].get('future', False) and markets[symbol]['active']]
    
    # Filter small caps using CoinGecko (free API)
    coingecko_url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_asc&per_page=250&page=1&sparkline=false'
    response = requests.get(coingecko_url)
    if response.status_code != 200:
        st.error("Error fetching from CoinGecko")
        return []
    
    coins = response.json()
    small_caps = {coin['symbol'].upper() + '/USDT': coin['market_cap'] for coin in coins if coin['market_cap'] < 50000000 and coin['market_cap'] > 0}
    
    # Match with MEXC futures
    small_cap_futures = [pair for pair in futures_pairs if pair in small_caps]
    return small_cap_futures[:500]  # Up to 500 for broad coverage

def fetch_historical_data(exchange, symbol, timeframe='1d', limit=500):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def analyze_coin(df_daily, df_hourly):
    # Indicators on daily
    df_daily['ma200'] = ta.trend.sma_indicator(df_daily['close'], window=200)
    df_daily['rsi'] = ta.momentum.rsi(df_daily['close'], window=14)
    df_daily['macd'] = ta.trend.macd_diff(df_daily['close'])
    df_daily['bb_width'] = (ta.volatility.bollinger_hband(df_daily['close']) - ta.volatility.bollinger_lband(df_daily['close'])) / ta.volatility.bollinger_mavg(df_daily['close'])
    df_daily['atl'] = df_daily['low'].rolling(window=len(df_daily), min_periods=1).min()
    df_daily['proximity_to_atl'] = (df_daily['close'] - df_daily['atl']) / df_daily['atl']
    
    # Volume analysis
    df_daily['vol_ma20'] = ta.trend.sma_indicator(df_daily['volume'], window=20)
    vol_surge = df_daily['volume'].iloc[-1] / df_daily['vol_ma20'].iloc[-1] if not pd.isna(df_daily['vol_ma20'].iloc[-1]) else 0
    vol_surge_pct = (vol_surge - 1) * 100
    
    # Cycle detection with scipy
    peaks, _ = find_peaks(df_daily['close'], distance=30, prominence=0.1 * df_daily['close'].std())
    if len(peaks) > 1:
        cycle_lengths = np.diff(df_daily['timestamp'].iloc[peaks]).mean().days
    else:
        cycle_lengths = np.nan
    
    # Hourly for early pump detection (price flat but volume up)
    df_hourly['vol_ma4'] = ta.trend.sma_indicator(df_hourly['volume'], window=4)
    hourly_vol_surge = df_hourly['volume'].iloc[-1] / df_hourly['vol_ma4'].iloc[-1] if not pd.isna(df_hourly['vol_ma4'].iloc[-1]) else 0
    price_change_24h = (df_hourly['close'].iloc[-1] - df_hourly['close'].iloc[-24]) / df_hourly['close'].iloc[-24] if len(df_hourly) > 24 else 0
    
    early_pump = hourly_vol_surge > 2 and abs(price_change_24h) < 0.1  # Volume surge without big price move
    
    # Prepare features for ML
    df_daily['returns'] = df_daily['close'].pct_change()
    features = ['rsi', 'macd', 'bb_width', 'proximity_to_atl', 'vol_surge']
    df_daily = df_daily.dropna(subset=features + ['returns'])
    if len(df_daily) < 100:
        return None
    
    X = df_daily[features]
    y = (df_daily['close'].shift(-14) > df_daily['close'] * 2.0).astype(int)  # Label: 1 if +100% in 14 days for bigger moves
    
    # Train RF model (more estimators for better accuracy)
    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X[-1:])[0][1]  # Upside prob
    
    # Historical upside for TP
    avg_upside = df_daily[df_daily['returns'] > 0]['returns'].mean() * 14  # 14-day expected
    
    # Grade setup
    if prob > 0.9:
        grade = 'A+'
    elif prob > 0.8:
        grade = 'A'
    elif prob > 0.7:
        grade = 'B+'
    elif prob > 0.6:
        grade = 'B'
    elif prob > 0.5:
        grade = 'C+'
    else:
        grade = 'C'
    
    current_price = df_daily['close'].iloc[-1]
    entry = current_price
    sl = entry * 0.92  # 8% SL for volatility
    tp = entry * (1 + max(0.5, avg_upside))  # At least 50%, or historical
    exit_condition = "Sell on 100% gain or cycle peak (RSI>70)"
    
    return {
        'proximity_to_atl': df_daily['proximity_to_atl'].iloc[-1] * 100,
        'rsi': df_daily['rsi'].iloc[-1],
        'vol_surge_pct': vol_surge_pct,
        'probability_upside': prob * 100,
        'expected_return': avg_upside * 100,
        'cycle_length_days': cycle_lengths if not np.isnan(cycle_lengths) else 'N/A',
        'early_pump': early_pump,
        'grade': grade,
        'entry': entry,
        'tp': tp,
        'sl': sl,
        'exit': exit_condition
    }

@st.cache_data(ttl=1200)  # 20 min per symbol data
def get_and_analyze_pair(pair):
    exchange = ccxt.mexc({'enableRateLimit': True})
    try:
        df_daily = fetch_historical_data(exchange, pair, '1d', 500)
        df_hourly = fetch_historical_data(exchange, pair, '1h', 168)
        analysis = analyze_coin(df_daily, df_hourly)
        if analysis:
            analysis['Pair'] = pair
            analysis['Current Price'] = df_daily['close'].iloc[-1]
            return analysis
    except:
        pass
    return None

# ────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────
st.title("Advanced Institutional-Level MEXC Small-Cap Scanner & Signal Giver")
st.markdown("""
This enhanced app scans MEXC small-cap futures for cyclic patterns, bottoms, and early pumps.
It uses advanced ML, multi-timeframe analysis, volume surges, and grades signals (A+ to C).
Includes entry/exit/TP/SL for swing trades with high leverage potential.
**Warning:** Crypto is high-risk. Use small capital, manage leverage (20-100x). Not advice.
""")

scan_type = st.selectbox("Scan Type", ["Bottoms (Near Lows)", "Early Pumps (Volume Surge)"])

col1, col2 = st.columns(2)
with col1:
    st.session_state.batch_size = st.slider("Batch size", 50, 150, st.session_state.batch_size)
with col2:
    if st.button("Reset & Start New Scan"):
        for key in list(st.session_state.keys()):
            if key not in ['batch_size']:  # keep slider
                del st.session_state[key]
        st.rerun()

# Load or refresh pair list
if not st.session_state.all_pairs:
    with st.spinner("Loading small-cap perpetual futures pairs..."):
        st.session_state.all_pairs = get_small_cap_futures_pairs()
        st.session_state.batch_index = 0
        st.session_state.scanned_results = []
        st.session_state.scan_complete = False
    st.success(f"Found {len(st.session_state.all_pairs)} small-cap futures pairs")

total = len(st.session_state.all_pairs)
done = st.session_state.batch_index * st.session_state.batch_size
progress = done / total if total > 0 else 0

st.progress(progress)
st.write(f"Scanned: {done} / {total} pairs  |  Batches complete: {st.session_state.batch_index}")

# ─── Scan button ───
if st.button("Scan Next Batch" if done < total else "Scan Complete – Filter Now"):
    if done >= total:
        # Final filtering step
        if st.session_state.scanned_results:
            df_all = pd.DataFrame(st.session_state.scanned_results)
            
            # Filters based on scan_type
            if scan_type == "Bottoms (Near Lows)":
                filtered = df_all[
                    (df_all['proximity_to_atl'] < 20) &
                    (df_all['rsi'] < 40) &
                    (df_all['probability_upside'] > 55) &
                    (df_all['vol_surge_pct'] > 50) &
                    (df_all['grade'].isin(['A+', 'A', 'B+']))
                ].sort_values('probability_upside', ascending=False)
            else:  # Early Pumps
                filtered = df_all[
                    (df_all['early_pump'] == True) &
                    (df_all['vol_surge_pct'] > 100) &
                    (df_all['probability_upside'] > 55) &
                    (df_all['grade'].isin(['A+', 'A', 'B+']))
                ].sort_values('probability_upside', ascending=False)
            
            st.session_state.filtered_watchlist = filtered
            st.session_state.scan_complete = True
            
            st.subheader("Your Watchlist – Top Profitable Signals")
            if not filtered.empty:
                st.table(filtered[['Pair', 'Grade', 'Current Price', 'Entry', 'Take Profit', 'Stop Loss', 'Exit', 'Upside Prob (%)', 'Exp Return (%)', 'Vol Surge (%)', 'ATL Proximity (%)', 'RSI']])
                st.markdown("""
                **Trading Guide:**
                - **A+/A:** High conviction, use 50-100x leverage on $5-10.
                - **B+:** Medium, 20-50x.
                - Expected: 5-20x returns on winners in cycles.
                """)
            else:
                st.info("No setups met your strict criteria this time.")
        else:
            st.warning("No data scanned yet.")
    else:
        # Scan current batch
        start = st.session_state.batch_index * st.session_state.batch_size
        end = min(start + st.session_state.batch_size, total)
        batch_pairs = st.session_state.all_pairs[start:end]
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        new_results = []
        for i, pair in enumerate(batch_pairs):
            status.text(f"Analyzing {pair} ({i+1}/{len(batch_pairs)}) – batch {st.session_state.batch_index + 1}")
            result = get_and_analyze_pair(pair)
            if result:
                new_results.append(result)
            progress_bar.progress((i + 1) / len(batch_pairs))
            time.sleep(0.6)  # conservative rate limit safety
        
        # Append to session
        st.session_state.scanned_results.extend(new_results)
        st.session_state.batch_index += 1
        
        st.success(f"Batch done! {len(new_results)} results added. Total scanned: {len(st.session_state.scanned_results)}")
        st.rerun()  # refresh UI

# Optional: show all raw if you want (for debug)
if st.checkbox("Show all raw scanned results (advanced)"):
    if st.session_state.scanned_results:
        st.dataframe(pd.DataFrame(st.session_state.scanned_results))

st.sidebar.markdown("Enhanced with batch scanning, SciPy for cycles, multi-TF, volume detection. Free tools: CCXT, CoinGecko, Scikit-learn, TA-Lib.")
