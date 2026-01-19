"""
Fetch 10 years of historical silver prices and save to CSV
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Calculate date range (last 10 years)
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10)

# Silver is traded as SI=F (Silver Futures) on Yahoo Finance
# SLV is the iShares Silver Trust ETF which tracks silver prices closely
print("Fetching silver price data from Yahoo Finance...")
print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Fetch Silver Futures data
silver = yf.download("SI=F", start=start_date, end=end_date, progress=False)

if silver.empty:
    print("Silver Futures data not available, trying SLV ETF...")
    silver = yf.download("SLV", start=start_date, end=end_date, progress=False)

# Reset index to make Date a column
silver = silver.reset_index()

# Flatten multi-level columns if present
if isinstance(silver.columns, pd.MultiIndex):
    silver.columns = [col[0] if col[1] == '' else col[0] for col in silver.columns]

# Select and rename columns for ML training
silver = silver[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
silver.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# Format date column
silver['Date'] = pd.to_datetime(silver['Date']).dt.strftime('%Y-%m-%d')

# Save to CSV
output_file = "silver_prices_10years.csv"
silver.to_csv(output_file, index=False)

print(f"\n✓ Data saved to: {output_file}")
print(f"✓ Total records: {len(silver)}")
print(f"\nFirst 5 rows:")
print(silver.head())
print(f"\nLast 5 rows:")
print(silver.tail())
print(f"\nDataset Info:")
print(f"  - Columns: {list(silver.columns)}")
print(f"  - Date Range: {silver['Date'].iloc[0]} to {silver['Date'].iloc[-1]}")
