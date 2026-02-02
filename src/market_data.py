import requests
import pandas as pd
import datetime

class MarketData:
    def __init__(self, start_date: str, end_date: str, api_key: str = "AizU0ZNVK6PkG9YGaK43AJkjEQV1hbi2"):
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        
    def fetch_data(self, tickers: list) -> pd.DataFrame:
        """
        Fetches historical adjusted close prices using the new FMP 'stable' endpoint.
        """
        print(f"Fetching data for {tickers} from {self.start_date} to {self.end_date} using FMP (Stable API)...")
        
        all_data = {}
        
        for ticker in tickers:
            try:
                # New stable endpoint found in PDF documentation
                url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={ticker}&from={self.start_date}&to={self.end_date}&apikey={self.api_key}"
                
                response = requests.get(url)
                
                if response.status_code != 200:
                    print(f"Error {response.status_code} for {ticker}: {response.text[:100]}")
                    continue

                data = response.json()
                
                # Check for list format (common in new FMP endpoints) or dict with 'historical'
                records = []
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict) and "historical" in data:
                    records = data["historical"]
                elif isinstance(data, dict) and "symbol" in data:
                     # sometimes returns just the object if one day? uncommon for 'full'
                     pass
                
                if not records:
                     print(f"No records found for {ticker}")
                     continue

                df = pd.DataFrame(records)
                if not df.empty:
                    # Ensure 'date' column exists (sometimes it's 'date' or 'symbol' needs parsing? usually 'date')
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        df.sort_index(inplace=True)
                        
                        # Use adjClose if available
                        col = 'adjClose' if 'adjClose' in df.columns else 'close'
                        if col in df.columns:
                             all_data[ticker] = df[col]
                        else:
                             print(f"Missing price column for {ticker}. Columns: {df.columns}")
                    else:
                        print(f"Missing 'date' column for {ticker}. Columns: {df.columns}")
                        
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                
        # Combine into a single DataFrame
        if all_data:
            prices = pd.DataFrame(all_data)
            prices = prices.ffill().bfill()
            # Filter strict range
            prices = prices.loc[self.start_date:self.end_date]
            return prices
        else:
            return pd.DataFrame()

if __name__ == "__main__":
    # Test
    md = MarketData("2023-01-01", "2023-01-10")
    df = md.fetch_data(["SPY", "AGG"])
    print(df.head())
