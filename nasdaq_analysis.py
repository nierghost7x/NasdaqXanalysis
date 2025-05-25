import streamlit as st
import requests
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import pandas as pd

# --- Helper Functions ---
def get_nasdaq_100_tickers():
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}
    response = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=headers)
    return response.json()['data']['data']['rows']

def parse_percent_change(p):
    try:
        return float(p.replace('%', '').replace('+', '').strip())
    except:
        return 0.0

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

def fetch_stock_data(symbol, period_years=8):
    """Fetch stock data once for the longest period needed"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * period_years)
        df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), 
                        end=end_date.strftime("%Y-%m-%d"), progress=False)
        df.dropna(inplace=True)
        return df
    except:
        return None

def filter_data_by_period(df, months_back):
    """Filter dataframe to get data from specific months back"""
    if df is None or df.empty:
        return None
    
    end_date = df.index[-1]
    start_date = end_date - timedelta(days=30 * months_back)
    
    # Find the closest date in the dataframe
    filtered_df = df[df.index >= start_date]
    return filtered_df if not filtered_df.empty else df

def get_momentum_score_from_data(df):
    """Calculate momentum score using pre-fetched data"""
    try:
        if df is None or len(df) < 200:
            return 0.0
        
        df_close = df['Close'].squeeze()
        
        # Get data points for different periods
        p_1y = df_close.iloc[0] if len(df_close) > 252 else df_close.iloc[0]
        p_6m = df_close.iloc[max(0, len(df) - 126)] if len(df_close) > 126 else df_close.iloc[0]
        p_1m = df_close.iloc[max(0, len(df) - 21)] if len(df_close) > 21 else df_close.iloc[0]
        p_now = df_close.iloc[-1]
        
        change_1y = ((p_now - p_1y) / p_1y) * 100
        change_6m = ((p_now - p_6m) / p_6m) * 100
        change_1m = ((p_now - p_1m) / p_1m) * 100
        
        score = 1.5 * (change_1m - change_6m) + (change_1m - change_1y)
        return round(score, 2)
    except:
        return 0.0

def short_term_regression_momentum_from_data(df, window=20):
    """Calculate regression momentum using pre-fetched data"""
    try:
        if df is None or len(df) < window:
            return 0.0
        
        y = df['Close'].squeeze().values[-window:]
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        return round(slope / y[0] * 100, 3)  # normalized %
    except:
        return 0.0

def price_acceleration_from_data(df, window=10):
    """Calculate price acceleration using pre-fetched data"""
    try:
        if df is None or len(df) < window:
            return 0.0
        
        close = df['Close'].squeeze().values[-window:]
        returns = np.diff(close)
        acc = np.mean(np.diff(returns))  # second derivative
        return round(acc, 4)
    except:
        return 0.0

def volume_weighted_momentum_from_data(df):
    """Calculate volume weighted momentum using pre-fetched data"""
    try:
        if df is None or df.empty:
            return 0.0
        
        # Use last month of data
        df_last_month = df.tail(21)  # Approximately 1 month
        df_calc = df_last_month.copy()
        df_calc['return'] = df_calc['Close'].pct_change()
        df_calc['vw_return'] = df_calc['return'] * df_calc['Volume']
        return round(df_calc['vw_return'].sum(), 4)
    except:
        return 0.0

def short_term_composite_momentum_from_data(df):
    """Calculate composite momentum using pre-fetched data"""
    reg = short_term_regression_momentum_from_data(df)
    accel = price_acceleration_from_data(df)
    vwm = volume_weighted_momentum_from_data(df)
    return round((reg * 0.5) + (accel * 100 * 0.3) + (vwm * 0.2), 2)

# --- Streamlit UI ---
st.title("NASDAQ-100 Stock Momentum Dashboard")

# Choose Sorting Option
sort_by = st.selectbox("Sort By", ["Drop Percentage", "Analyst Rating", "Momentum Score", "Composite Momentum"])

# Display data
tickers = get_nasdaq_100_tickers()

# Pre-process tickers and add momentum scores if needed
with st.spinner("Loading stock data and calculating metrics..."):
    for ticker in tickers:
        ticker['symbol'] = ticker['symbol'].replace('.', '-')
        
        # Get analyst ratings
        summary = get_analyst_ratings(ticker['symbol'])
        if summary:
            ticker['Buy'] = summary['Buy']
        else:
            ticker['Buy'] = -1  # Use -1 so that tickers with no data go last
        
        # Fetch stock data once if momentum calculation is needed
        if sort_by in ["Momentum Score", "Composite Momentum"]:
            stock_data = fetch_stock_data(ticker['symbol'])
            
            if sort_by == "Momentum Score":
                ticker['momentum_score'] = get_momentum_score_from_data(stock_data)
            elif sort_by == "Composite Momentum":
                ticker['momentum_score'] = short_term_composite_momentum_from_data(stock_data)
            
            # Store the data for later use in charts
            ticker['stock_data'] = stock_data

# Sorting Configuration
if sort_by == "Drop Percentage":
    tickers_sorted = sorted(tickers, key=lambda x: parse_percent_change(x.get('percentageChange', '0%')))
elif sort_by == "Analyst Rating":
    tickers_sorted = sorted(tickers, key=lambda x: x['Buy'], reverse=True)
elif sort_by in ["Momentum Score", "Composite Momentum"]:
    tickers_sorted = sorted(tickers, key=lambda x: x['momentum_score'], reverse=True)

# Display Top 25 Stocks
for ticker in tickers_sorted[:25]:
    # Set up the figure with dark style and better spacing
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.patch.set_facecolor('#0E1117')  # Streamlit dark background
    
    score = ticker.get('momentum_score', '')
    change_str = ticker.get('percentageChange', 'N/A')
    
    # Enhanced title with better formatting
    title_color = '#32CD32' if parse_percent_change(change_str) >= 0 else '#FF6B6B'
    fig.suptitle(f"{ticker['companyName']} ({ticker['symbol']}) | {change_str} | Score: {score}", 
                fontsize=14, fontweight='bold', color='white', y=0.95)
    
    symbol = ticker['symbol']

    # Fetch data once for this ticker if not already fetched
    if 'stock_data' not in ticker:
        stock_data = fetch_stock_data(symbol)
    else:
        stock_data = ticker['stock_data']

    local_timezone = pytz.timezone('Europe/Berlin')
    now = datetime.now(local_timezone)
    
    # Define timeframes in months
    timeframes = {
        "5 Years": 5*12,  # 5 * 12 months
        "1 Year": 12,   # 12 months
        "3 Months": 3   # 6 months
    }

    # Enhanced color scheme with gradients
    colors = {
        "5 Years": "#00FF88",    # Bright green
        "1 Year": "#FF9500",     # Bright orange  
        "3 Months": "#9D4EDD"    # Purple
    }

    # Plot charts using filtered data from the same dataset
    for i, (ax, (label, months_back)) in enumerate(zip(axes[:3], timeframes.items())):
        ax.set_facecolor('#1E1E1E')  # Dark chart background
        
        if stock_data is not None and not stock_data.empty:
            filtered_df = filter_data_by_period(stock_data, months_back)
            if filtered_df is not None and not filtered_df.empty:
                close_prices = filtered_df['Close'].squeeze()
                dates = filtered_df.index
                
                # Plot with proper date x-axis
                ax.plot(dates, close_prices, color=colors[label], linewidth=2.5, alpha=0.9)
                
                # Add fill under the curve for better visual appeal
                ax.fill_between(dates, close_prices, alpha=0.2, color=colors[label])
                
                # Enhanced grid
                ax.grid(axis="y", alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
                ax.grid(axis="x", alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
                
                # Format x-axis dates properly
                if len(dates) > 0:
                    # Limit number of ticks to avoid overcrowding
                    max_ticks = 6
                    tick_indices = np.linspace(0, len(dates)-1, min(max_ticks, len(dates)), dtype=int)
                    tick_dates = [dates[i] for i in tick_indices]
                    tick_labels = [date.strftime('%b %Y') if months_back >= 12 else date.strftime('%b %d') for date in tick_dates]
                    
                    ax.set_xticks(tick_dates)
                    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
                
                # Style the axes
                ax.set_xlabel(label, fontsize=11, fontweight='bold', color='white')
                ax.tick_params(axis='x', colors='lightgray', labelsize=8)
                ax.tick_params(axis='y', rotation=0, colors='lightgray', labelsize=9)
                
                # Add price range info
                price_change = ((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]) * 100
                
                # Color code the price change
                change_color = '#32CD32' if price_change >= 0 else '#FF6B6B'
                ax.text(0.02, 0.95, f'{price_change:+.1f}%', transform=ax.transAxes, 
                       fontsize=10, fontweight='bold', color=change_color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                
                # Format y-axis to show currency and set proper limits
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
                
                # Set y-axis limits to actual price range with small padding
                price_min, price_max = close_prices.min(), close_prices.max()
                padding = (price_max - price_min) * 0.05  # 5% padding
                ax.set_ylim(price_min - padding, price_max + padding)
                
                # Spine styling
                for spine in ax.spines.values():
                    spine.set_color('gray')
                    spine.set_alpha(0.5)
                    
            else:
                ax.text(0.5, 0.5, 'No Data\nAvailable', ha='center', va='center', 
                       transform=ax.transAxes, color='gray', fontsize=12, fontweight='bold')
                ax.set_facecolor('#2E2E2E')
        else:
            ax.text(0.5, 0.5, 'No Data\nAvailable', ha='center', va='center', 
                   transform=ax.transAxes, color='gray', fontsize=12, fontweight='bold')
            ax.set_facecolor('#2E2E2E')

    # Enhanced Analyst ratings panel
    summary = get_analyst_ratings(symbol)
    axes[3].axis("off")
    axes[3].set_facecolor('#1E1E1E')
    
    if summary:
        # Create a more visually appealing ratings display
        axes[3].text(0.1, 0.85, "📊 Analyst Ratings", fontsize=12, fontweight='bold', color='white')
        
        # Rating bars visualization
        ratings = [('Buy', summary['Buy'], '#32CD32'), 
                  ('Hold', summary['Hold'], '#FFD700'), 
                  ('Sell', summary['Sell'], '#FF6B6B')]
        
        for i, (rating, pct, color) in enumerate(ratings):
            y_pos = 0.65 - i * 0.15
            # Rating label
            axes[3].text(0.1, y_pos, f"{rating}:", fontsize=10, color='white', fontweight='bold')
            # Percentage
            axes[3].text(0.3, y_pos, f"{pct}%", fontsize=10, color=color, fontweight='bold')
            # Visual bar
            bar_width = pct / 100 * 0.4  # Scale to fit
            axes[3].add_patch(plt.Rectangle((0.45, y_pos-0.02), bar_width, 0.04, 
                                          facecolor=color, alpha=0.7, edgecolor=color))
        
        # Overall score with styling
        score_color = '#32CD32' if summary['Score'] <= 2.5 else '#FFD700' if summary['Score'] <= 3.5 else '#FF6B6B'
        axes[3].text(0.1, 0.15, f"Overall Score: {summary['Score']}/5", 
                    fontsize=11, fontweight='bold', color=score_color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    else:
        axes[3].text(0.5, 0.5, "📊\nNo Analyst\nData Available", ha='center', va='center', 
                    fontsize=11, color='gray', fontweight='bold')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15, left=0.05, right=0.98, wspace=0.25)
    
    st.pyplot(fig)
    plt.close(fig)
    plt.style.use('default')  # Reset style for next iteration