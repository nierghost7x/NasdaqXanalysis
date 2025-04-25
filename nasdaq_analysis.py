import streamlit as st
import requests
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Helper Functions ---
def get_nasdaq_100_tickers():
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}
    response = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=headers)
    return response.json()['data']['data']['rows']

def get_analyst_ratings(ticker):
    stock = yf.Ticker(ticker)
    recommendations = stock.recommendations
    if recommendations is None or recommendations.empty:
        return None
    latest = recommendations.iloc[0]
    total = latest[['strongBuy', 'buy', 'hold', 'sell', 'strongSell']].sum()
    percentages = {
        'Buy': ((latest['strongBuy'] + latest['buy']) / total) * 100,
        'Hold': (latest['hold'] / total) * 100,
        'Sell': ((latest['sell'] + latest['strongSell']) / total) * 100
    }
    score = (latest['strongBuy'] * 1 + latest['buy'] * 2 + latest['hold'] * 3 + latest['sell'] * 4 + latest['strongSell'] * 5) / total
    return {'Buy': round(percentages['Buy'], 1), 'Hold': round(percentages['Hold'], 1), 'Sell': round(percentages['Sell'], 1), 'Score': round(score, 2)}

def get_momentum_score(symbol):
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
        df.dropna(inplace=True)
        if len(df) < 200:
            return 0.0
        df_close = df['Close'].squeeze()
        p_1y = df_close.iloc[0]
        p_6m = df_close.iloc[int(len(df) * 0.5)]
        p_1m = df_close.iloc[int(len(df) * 11 / 12)]
        p_now = df_close.iloc[-1]
        change_1y = ((p_now - p_1y) / p_1y) * 100
        change_6m = ((p_now - p_6m) / p_6m) * 100
        change_1m = ((p_now - p_1m) / p_1m) * 100
        score = 1.5 * (change_1m - change_6m) + (change_1m - change_1y)
        return round(score, 2)
    except:
        return 0.0

def short_term_regression_momentum(symbol, window=20):
    try:
        df = yf.download(symbol, period="2mo", interval="1d", progress=False)
        df.dropna(inplace=True)
        if len(df) < window:
            return 0.0
        y = df['Close'].squeeze().values[-window:]
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        return round(slope / y[0] * 100, 3)  # normalized %
    except:
        return 0.0

def price_acceleration(symbol, window=10):
    try:
        df = yf.download(symbol, period="1mo", interval="1d", progress=False)
        df.dropna(inplace=True)
        close = df['Close'].squeeze().values[-window:]
        returns = np.diff(close)
        acc = np.mean(np.diff(returns))  # second derivative
        return round(acc, 4)
    except:
        return 0.0

def volume_weighted_momentum(symbol):
    try:
        df = yf.download(symbol, period="1mo", interval="1d", progress=False)
        df.dropna(inplace=True)
        df['return'] = df['Close'].pct_change()
        df['vw_return'] = df['return'] * df['Volume']
        return round(df['vw_return'].sum(), 4)
    except:
        return 0.0

def short_term_composite_momentum(symbol):
    reg = short_term_regression_momentum(symbol)
    accel = price_acceleration(symbol)
    vwm = volume_weighted_momentum(symbol)
    return round((reg * 0.5) + (accel * 100 * 0.3) + (vwm * 0.2), 2)

# --- Streamlit UI ---
st.title("NASDAQ-100 Stock Momentum Dashboard")

# Choose Sorting Option
sort_by = st.selectbox("Sort By", ["Drop Percentage", "Analyst Rating", "Momentum Score", "Composite Momentum"])

# Display data
tickers = get_nasdaq_100_tickers()
for ticker in tickers:
    ticker['symbol'] = ticker['symbol'].replace('.', '-')
    summary = get_analyst_ratings(ticker['symbol'])
    if summary:
        ticker['Buy'] = summary['Buy']
    else:
        ticker['Buy'] = -1  # Use -1 so that tickers with no data go last

# Sorting Configuration
if sort_by == "Drop Percentage":
    tickers_sorted = sorted(tickers, key=lambda x: parse_percent_change(x.get('percentageChange', '0%')))
elif sort_by == "Analyst Rating":
    tickers_sorted = sorted(tickers, key=lambda x: x['Buy'], reverse=True)
elif sort_by == "Momentum Score":
    for ticker in tickers:
        ticker['momentum_score'] = get_momentum_score(ticker['symbol'])
    tickers_sorted = sorted(tickers, key=lambda x: x['momentum_score'], reverse=True)
elif sort_by == "Composite Momentum":
    for ticker in tickers:
        ticker['momentum_score'] = short_term_composite_momentum(ticker['symbol'])
    tickers_sorted = sorted(tickers, key=lambda x: x['momentum_score'], reverse=True)

# Display Top 25 Stocks
for ticker in tickers_sorted[:25]:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3))
    score = ticker.get('momentum_score', '')
    change_str = ticker.get('percentageChange', 'N/A')
    fig.suptitle(f"{ticker['companyName']} ({change_str})  | Score: {score}", fontsize=16)
    symbol = ticker['symbol']

    local_timezone = pytz.timezone('Europe/Berlin')
    now = datetime.now(local_timezone)
    timeframes = {
        "8 Years": (now - timedelta(days=365*8)).strftime("%Y-%m-%d"),
        "1 Year": (now - timedelta(days=365)).strftime("%Y-%m-%d"),
        "6 Months": (now - timedelta(days=30*6)).strftime("%Y-%m-%d")
    }

    colors = {
        "8 Years": "green",
        "1 Year": "orange",
        "6 Months": "purple"
    }

    for ax, (label, start_date) in zip(axes[:3], timeframes.items()):
        df = yf.download(symbol, start=start_date, end=datetime.now().strftime("%Y-%m-%d"), progress=False)
        df.dropna(inplace=True)
        ax.plot(df['Close'].squeeze(), color=colors[label])
        ax.grid(axis="y")
        ax.set_xlabel(label)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=45)

    summary = get_analyst_ratings(symbol)
    axes[3].axis("off")
    if summary:
        axes[3].text(0.05, 0.9, "Analyst Ratings", fontsize=11, fontweight='bold')
        axes[3].text(0.05, 0.7, f"Buy: {summary['Buy']}%", color="green", fontweight='bold')
        axes[3].text(0.05, 0.6, f"Hold: {summary['Hold']}%", color="orange", fontweight='bold')
        axes[3].text(0.05, 0.5, f"Sell: {summary['Sell']}%", color="red", fontweight='bold')
        axes[3].text(0.05, 0.3, f"Score: {summary['Score']}", fontsize=10)
    else:
        axes[3].text(0.2, 0.5, "No Analyst Data", fontsize=11, color="gray")

    st.pyplot(fig)
    plt.close(fig)