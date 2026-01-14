import bql
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DualClassSpreadDashboard:
    def __init__(self):
        """Initialize dual-class spread monitoring dashboard"""
        try:
            self.bq = bql.Service()
            self.has_bql = True
            self.connection_status = "✅ Connected to Bloomberg BQL"
        except Exception as e:
            self.has_bql = False
            self.connection_status = f"❌ Bloomberg connection failed: {str(e)}"
        """Raising Error"""
        # Store bulk analysis results and company names cache
        self.bulk_results = {}
        self.company_names_cache = {}
        self.setup_dashboard()
    
    def get_company_names(self, tickers):
        """Get company names for a list of tickers using BQL"""
        if not self.has_bql:
            # Return placeholder names for demo mode
            return {ticker: ticker.replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '') 
                    for ticker in tickers}
        
        try:
            # Filter out already cached names
            uncached_tickers = [t for t in tickers if t not in self.company_names_cache]
            
            if uncached_tickers:
                data_item = {
                    'name': self.bq.data.SHORT_COMPANY_NAME()
                }
                request = bql.Request(uncached_tickers, data_item)
                response = self.bq.execute(request)
                
                if response and len(response) > 0:
                    df = response[0].df()
                    
                    if not df.empty:
                        # Look for the name column
                        name_col = None
                        for col in df.columns:
                            if 'name' in str(col).lower():
                                name_col = col
                                break
                        
                        if name_col is not None:
                            # Process each row - the DataFrame index should correspond to tickers
                            for i, (idx, row) in enumerate(df.iterrows()):
                                if i < len(uncached_tickers):
                                    ticker = uncached_tickers[i]
                                    company_name = row[name_col] if pd.notna(row[name_col]) else ticker.replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '')
                                    self.company_names_cache[ticker] = str(company_name).strip()
                        else:
                            # Fallback if no name column is found
                            for ticker in uncached_tickers:
                                self.company_names_cache[ticker] = ticker.replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '')
                    else:
                        # Fallback for empty DataFrame
                        for ticker in uncached_tickers:
                            self.company_names_cache[ticker] = ticker.replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '')
                else:
                    # Fallback for failed requests
                    for ticker in uncached_tickers:
                        self.company_names_cache[ticker] = ticker.replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '')
            
            # Return names for all requested tickers
            result = {}
            for ticker in tickers:
                result[ticker] = self.company_names_cache.get(ticker, ticker.replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', ''))
            
            return result
            
        except Exception as e:
            # Return fallback names
            return {ticker: ticker.replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '') 
                    for ticker in tickers}
    
    def setup_dashboard(self):
        """Create the spread analysis dashboard"""
        
        # Header
        header = widgets.HTML(
            value="""
            <div style='background: linear-gradient(90deg, #2E86AB, #A23B72); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='color: white; margin: 0; text-align: center; font-family: Arial, sans-serif;'>
                    📊 DUAL-CLASS SHARE SPREAD DASHBOARD
                </h2>
                <p style='color: white; margin: 5px 0 0 0; text-align: center; font-size: 14px;'>
                    Monitor price disparities between ordinary and preferred share classes
                </p>
            </div>
            """
        )
        
        # Connection status
        status = widgets.HTML(
            value=f"<div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px; color: #333;'><strong>{self.connection_status}</strong></div>"
        )
        
        # ============ SINGLE PAIR ANALYSIS SECTION ============
        self.single_ticker1 = widgets.Text(
            value='',
            placeholder='Ordinary shares (e.g., GOOG US EQUITY)',
            description='Ordinary:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.single_ticker2 = widgets.Text(
            value='',
            placeholder='Preferred shares (e.g., GOOGL US EQUITY)', 
            description='Preferred:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.single_share_ratio = widgets.FloatText(
            value=1.0,
            placeholder='1.0',
            description='Share Ratio:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        # Single pair time period
        self.single_period_dropdown = widgets.Dropdown(
            options=[
                ('30 Days', 30),
                ('3 Months', 90), 
                ('6 Months', 180),
                ('1 Year', 365),
                ('Custom', 0)
            ],
            value=90,
            description='Time Period:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        # Single pair custom dates
        default_end = datetime.now().date()
        default_start = default_end - timedelta(days=90)
        
        self.single_start_date = widgets.DatePicker(
            description='Start Date:',
            value=default_start,
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px', display='none')
        )
        
        self.single_end_date = widgets.DatePicker(
            description='End Date:',
            value=default_end,
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px', display='none')
        )
        
        # Single pair spread method
        self.single_spread_method = widgets.Dropdown(
            options=[
                ('Percentage Spread: 1 - Pref/Ord', 'percentage'),
                ('Price Difference: Ord - Pref', 'absolute'),
                ('Price Ratio: Ord/Pref', 'ratio')
            ],
            value='percentage',
            description='Method:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.single_analyze_btn = widgets.Button(
            description='🔍 Analyze Single Pair',
            button_style='success',
            layout=widgets.Layout(width='200px', height='35px')
        )
        
        # ============ BULK ANALYSIS SECTION ============
        self.bulk_t1_input = widgets.Textarea(
            value='',
            placeholder='Enter Ordinary shares list (CSV):\nGOOG US EQUITY, BRK/B US EQUITY, RACE US EQUITY',
            description='Ordinary List:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px', height='80px')
        )
        
        self.bulk_t2_input = widgets.Textarea(
            value='',
            placeholder='Enter Preferred shares list (CSV):\nGOOGL US EQUITY, BRK/A US EQUITY, RACE IM EQUITY',
            description='Preferred List:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px', height='80px')
        )
        
        self.bulk_ratios_input = widgets.Text(
            value='1,1,1',
            placeholder='Share ratios (CSV): 1,1,1',
            description='Share Ratios:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        # Bulk analysis time period
        self.bulk_period_dropdown = widgets.Dropdown(
            options=[
                ('30 Days', 30),
                ('3 Months', 90), 
                ('6 Months', 180),
                ('1 Year', 365),
                ('Custom', 0)
            ],
            value=90,
            description='Time Period:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        # Bulk custom dates
        self.bulk_start_date = widgets.DatePicker(
            description='Start Date:',
            value=default_start,
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px', display='none')
        )
        
        self.bulk_end_date = widgets.DatePicker(
            description='End Date:',
            value=default_end,
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px', display='none')
        )
        
        # Bulk spread method
        self.bulk_spread_method = widgets.Dropdown(
            options=[
                ('Percentage Spread: 1 - Pref/Ord', 'percentage'),
                ('Price Difference: Ord - Pref', 'absolute'),
                ('Price Ratio: Ord/Pref', 'ratio')
            ],
            value='percentage',
            description='Method:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.bulk_analyze_btn = widgets.Button(
            description='📊 Bulk Analysis',
            button_style='info',
            layout=widgets.Layout(width='200px', height='35px')
        )
        
        # Results area
        self.bulk_results_area = widgets.Output()
        self.output_area = widgets.Output()
        
        # Wire up events
        self.single_analyze_btn.on_click(self._analyze_single_pair)
        self.bulk_analyze_btn.on_click(self._analyze_bulk_pairs)
        
        # Wire up period change observers
        self.single_period_dropdown.observe(self._on_single_period_change, names='value')
        self.bulk_period_dropdown.observe(self._on_bulk_period_change, names='value')
        
        # Create sections
        single_section = widgets.VBox([
            widgets.HTML("<h3 style='color: #2E86AB; font-family: Arial, sans-serif; margin: 15px 0 10px 0;'>📈 Single Pair Analysis</h3>"),
            self.single_ticker1,
            self.single_ticker2,
            self.single_share_ratio,
            widgets.HTML("<p style='color: #666; font-size: 12px; margin: 5px 0;'>Share Ratio: Adjustment factor for preferred shares (if needed).</p>"),
            self.single_period_dropdown,
            self.single_start_date,
            self.single_end_date,
            self.single_spread_method,
            self.single_analyze_btn
        ], layout=widgets.Layout(padding='15px', border='1px solid #ddd', border_radius='8px', margin='10px 0'))
        
        bulk_section = widgets.VBox([
            widgets.HTML("<h3 style='color: #A23B72; font-family: Arial, sans-serif; margin: 15px 0 10px 0;'>📊 Bulk Analysis</h3>"),
            widgets.HTML("<p style='color: #666; font-size: 12px; margin: 5px 0;'>Enter comma-separated lists. Same number of entries required for each field.</p>"),
            self.bulk_t1_input,
            self.bulk_t2_input,
            self.bulk_ratios_input,
            widgets.HTML("<p style='color: #666; font-size: 12px; margin: 5px 0;'>Share Ratios: CSV format (e.g., 1,1,1). Adjustment factor for preferred shares if needed.</p>"),
            self.bulk_period_dropdown,
            self.bulk_start_date,
            self.bulk_end_date,
            self.bulk_spread_method,
            self.bulk_analyze_btn
        ], layout=widgets.Layout(padding='15px', border='1px solid #ddd', border_radius='8px', margin='10px 0'))
        
        # Main dashboard layout
        dashboard = widgets.VBox([
            header,
            status,
            single_section,
            bulk_section,
            widgets.HTML("<hr style='margin: 20px 0;'>"),
            widgets.HTML("<h3 style='color: #333; font-family: Arial, sans-serif;'>📊 Results</h3>"),
            self.bulk_results_area,
            self.output_area
        ])
        
        display(dashboard)
    
    def _on_single_period_change(self, change):
        """Show/hide custom date inputs for single analysis"""
        if change['new'] == 0:  # Custom selected
            self.single_start_date.layout.display = 'block'
            self.single_end_date.layout.display = 'block'
        else:
            self.single_start_date.layout.display = 'none'
            self.single_end_date.layout.display = 'none'
    
    def _on_bulk_period_change(self, change):
        """Show/hide custom date inputs for bulk analysis"""
        if change['new'] == 0:  # Custom selected
            self.bulk_start_date.layout.display = 'block'
            self.bulk_end_date.layout.display = 'block'
        else:
            self.bulk_start_date.layout.display = 'none'
            self.bulk_end_date.layout.display = 'none'
    
    def get_single_analysis_period(self):
        """Get start and end dates for single analysis"""
        if self.single_period_dropdown.value == 0:  # Custom
            if self.single_start_date.value and self.single_end_date.value:
                start_dt = datetime.combine(self.single_start_date.value, datetime.min.time())
                end_dt = datetime.combine(self.single_end_date.value, datetime.min.time())
                
                if start_dt >= end_dt or (end_dt - start_dt).days < 10:
                    return None, None
                    
                return start_dt, end_dt
            else:
                return None, None
        else:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=self.single_period_dropdown.value)
            return start_dt, end_dt
    
    def get_bulk_analysis_period(self):
        """Get start and end dates for bulk analysis"""
        if self.bulk_period_dropdown.value == 0:  # Custom
            if self.bulk_start_date.value and self.bulk_end_date.value:
                start_dt = datetime.combine(self.bulk_start_date.value, datetime.min.time())
                end_dt = datetime.combine(self.bulk_end_date.value, datetime.min.time())
                
                if start_dt >= end_dt or (end_dt - start_dt).days < 10:
                    return None, None
                    
                return start_dt, end_dt
            else:
                return None, None
        else:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=self.bulk_period_dropdown.value)
            return start_dt, end_dt
    
    def normalize_ticker(self, ticker):
        """Ensure ticker is in Bloomberg format with validation"""
        if not ticker or not isinstance(ticker, str):
            return None
            
        ticker = ticker.strip().upper()
        
        if not ticker:
            return None
            
        if 'EQUITY' not in ticker and 'INDEX' not in ticker:
            return f"{ticker} US EQUITY"
        return ticker
    
    def validate_share_ratio(self, ratio):
        """Validate share ratio input"""
        if not isinstance(ratio, (int, float)):
            return False
        if ratio <= 0 or ratio > 10000:
            return False
        return True
    
    def get_price_history(self, ticker, start_date, end_date):
        """Get historical price data using BQL with enhanced currency handling"""
        if not self.has_bql:
            # Mock data for demo
            days = (end_date - start_date).days
            dates = pd.date_range(start=start_date, end=end_date, freq='D')[:days]
            base_price = 100 + hash(ticker) % 100  # Different base price per ticker
            returns = np.random.randn(len(dates)) * 0.015 + 0.0002
            prices = [base_price]
            for r in returns[1:]:
                prices.append(prices[-1] * (1 + r))
            
            return pd.DataFrame({
                'date': dates[:len(prices)], 
                'price': prices
            })
        
        try:
            # First, try with currency conversion
            query_with_currency = f"""
            for('{ticker}') 
            get(px_last(dates=range('{start_date.strftime("%Y-%m-%d")}', '{end_date.strftime("%Y-%m-%d")}'), currency='USD'))
            """
            
            response = self.bq.execute(query_with_currency)
            
            if response and len(response) > 0:
                df = response[0].df()
                
                if not df.empty:
                    price_cols = [col for col in df.columns if 'PX_LAST' in str(col).upper()]
                    if price_cols:
                        price_col = price_cols[0]
                        df = df.rename(columns={price_col: 'price'})
                        
                        date_cols = [col for col in df.columns if 'DATE' in str(col).upper()]
                        if date_cols:
                            df['date'] = pd.to_datetime(df[date_cols[0]])
                            df = df[['date', 'price']].dropna()
                            
                            if len(df) >= 10 and df['price'].min() > 0:
                                return df.sort_values('date').reset_index(drop=True)
            
            # If currency conversion failed, try without currency conversion
            query_local = f"""
            for('{ticker}') 
            get(px_last(dates=range('{start_date.strftime("%Y-%m-%d")}', '{end_date.strftime("%Y-%m-%d")}')))
            """
            
            response = self.bq.execute(query_local)
            
            if not response or len(response) == 0:
                return None
                
            df = response[0].df()
            
            if df.empty:
                return None
                
            price_cols = [col for col in df.columns if 'PX_LAST' in str(col).upper()]
            if not price_cols:
                return None
                
            price_col = price_cols[0]
            df = df.rename(columns={price_col: 'price'})
            
            date_cols = [col for col in df.columns if 'DATE' in str(col).upper()]
            if date_cols:
                df['date'] = pd.to_datetime(df[date_cols[0]])
            else:
                return None
            
            # Get currency from metadata
            currency_cols = [col for col in df.columns if 'CURRENCY' in str(col).upper()]
            original_currency = df[currency_cols[0]].iloc[0] if currency_cols else None
            
            df = df[['date', 'price']].dropna()
            
            if len(df) < 10 or df['price'].min() <= 0:
                return None
            
            # Manual currency conversion if needed
            if original_currency and original_currency != 'USD':
                df = self.convert_currency_manually(df, original_currency, start_date, end_date)
                if df is None:
                    return None
            
            return df.sort_values('date').reset_index(drop=True)
            
        except Exception as e:
            return None
    
    def convert_currency_manually(self, df, from_currency, start_date, end_date):
        """Manual currency conversion using BQL FX rates when automatic conversion fails"""
        try:
            if from_currency == 'USD':
                return df  # Already in USD
            
            # Get FX rates for the period
            fx_ticker = f"{from_currency}USD Curncy"  # e.g., HKDUSD Curncy
            
            fx_query = f"""
            for('{fx_ticker}') 
            get(px_last(dates=range('{start_date.strftime("%Y-%m-%d")}', '{end_date.strftime("%Y-%m-%d")}')))
            """
            
            response = self.bq.execute(fx_query)
            
            if not response or len(response) == 0:
                return None
                
            fx_df = response[0].df()
            
            if fx_df.empty:
                return None
            
            # Process FX data
            price_cols = [col for col in fx_df.columns if 'PX_LAST' in str(col).upper()]
            if not price_cols:
                return None
                
            fx_df = fx_df.rename(columns={price_cols[0]: 'fx_rate'})
            
            date_cols = [col for col in fx_df.columns if 'DATE' in str(col).upper()]
            if date_cols:
                fx_df['date'] = pd.to_datetime(fx_df[date_cols[0]])
            else:
                return None
            
            fx_df = fx_df[['date', 'fx_rate']].dropna()
            
            if fx_df.empty:
                return None
            
            # Merge with price data and convert
            merged = pd.merge(df, fx_df, on='date', how='left')
            
            # Forward fill missing FX rates
            merged['fx_rate'] = merged['fx_rate'].fillna(method='ffill')
            
            # Convert prices: local_currency * fx_rate = USD
            merged['price'] = merged['price'] * merged['fx_rate']
            
            # Return only date and converted price
            result = merged[['date', 'price']].dropna()
            
            if result.empty:
                return None
            
            return result
            
        except Exception as e:
            return None
    
    def calculate_spread_metrics(self, ord_data, pref_data, method='percentage', share_ratio=1.0):
        """Calculate spread metrics using correct formula: 1 - pref_price/ord_price"""
        
        if ord_data is None or pref_data is None:
            raise ValueError("Missing price data for one or both tickers")
        
        if ord_data.empty or pref_data.empty:
            raise ValueError("Empty price data for one or both tickers")
        
        if not self.validate_share_ratio(share_ratio):
            raise ValueError(f"Invalid share ratio: {share_ratio}")
        
        ord_data_adj = ord_data.copy()
        pref_data_adj = pref_data.copy()
        
        # Apply share ratio adjustment to preferred shares if needed
        if share_ratio != 1.0:
            pref_data_adj['price'] = pref_data_adj['price'] * share_ratio
        
        # Align dates
        merged = pd.merge(ord_data_adj, pref_data_adj, on='date', suffixes=('_ord', '_pref'), how='inner')
        
        if merged.empty or len(merged) < 10:
            raise ValueError(f"Insufficient overlapping data points: {len(merged)}")
        
        if (merged['price_ord'] <= 0).any() or (merged['price_pref'] <= 0).any():
            raise ValueError("Found zero or negative prices in data")
        
        # Calculate spread using CORRECT FORMULA
        if method == 'percentage':
            # FIXED: Formula is now 1 - pref_price/ord_price (as percentage)
            merged['spread'] = (1 - (merged['price_pref'] / merged['price_ord'])) * 100
            spread_label = 'Spread (% Discount of Pref vs Ord)'
        elif method == 'absolute':
            merged['spread'] = merged['price_ord'] - merged['price_pref'] 
            spread_label = 'Price Difference ($)'
        else:  # ratio
            merged['spread'] = merged['price_ord'] / merged['price_pref']
            spread_label = 'Price Ratio (Ord/Pref)'
        
        # Validate calculations
        if merged['spread'].isna().any() or np.isinf(merged['spread']).any():
            raise ValueError("Spread calculation produced invalid values")
        
        # Statistics
        current_spread = merged['spread'].iloc[-1]
        avg_spread = merged['spread'].mean()
        std_spread = merged['spread'].std()
        max_spread = merged['spread'].max()
        min_spread = merged['spread'].min()
        
        current_zscore = (current_spread - avg_spread) / std_spread if std_spread > 0 else 0
        
        # Rolling stats for visualization
        try:
            window = min(30, len(merged))
            merged['spread_mean_30d'] = merged['spread'].rolling(window=window).mean()
            merged['spread_std_30d'] = merged['spread'].rolling(window=window).std()
            
            rolling_std = merged['spread_std_30d'].fillna(0)
            rolling_mean = merged['spread_mean_30d'].fillna(merged['spread'])
            
            merged['z_score_rolling'] = np.where(
                rolling_std > 0,
                (merged['spread'] - rolling_mean) / rolling_std,
                0
            )
        except Exception:
            merged['z_score_rolling'] = 0
        
        stats = {
            'current_spread': float(current_spread),
            'average_spread': float(avg_spread),
            'current_zscore': float(current_zscore),
            'max_spread': float(max_spread),
            'min_spread': float(min_spread),
            'std_spread': float(std_spread),
            'spread_label': spread_label,
            'share_ratio': float(share_ratio),
            'spread_range': f"{min_spread:.3f} to {max_spread:.3f}",
            'data_points': len(merged)
        }
        
        return merged, stats
    
    def create_bulk_summary_table(self, bulk_results):
        """Create interactive summary table with company names and sorting"""
        
        if not bulk_results:
            return widgets.HTML("<p>No bulk analysis results to display.</p>")
        
        valid_results = {k: v for k, v in bulk_results.items() if 'error' not in v}
        
        if not valid_results:
            error_count = len([v for v in bulk_results.values() if 'error' in v])
            return widgets.HTML(f"<p style='color: red;'>All {error_count} analyses failed. Check ticker symbols and data availability.</p>")
        
        # Get all unique tickers for company name lookup
        all_tickers = set()
        for pair_key in valid_results.keys():
            ord_ticker, pref_ticker = pair_key.split(' vs ')
            all_tickers.add(ord_ticker)
            all_tickers.add(pref_ticker)
        
        # Fetch company names
        company_names = self.get_company_names(list(all_tickers))
        
        table_html = """
        <style>
        .results-table {
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
            font-size: 14px;
            margin: 10px 0;
        }
        .results-table th {
            background-color: #2E86AB;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
            cursor: pointer;
            user-select: none;
            position: relative;
        }
        .results-table th:hover {
            background-color: #1d5f85;
        }
        .results-table th::after {
            content: ' ↕️';
            font-size: 12px;
            opacity: 0.7;
        }
        .results-table th.sort-asc::after {
            content: ' ↑';
            color: #FFD700;
        }
        .results-table th.sort-desc::after {
            content: ' ↓';
            color: #FFD700;
        }
        .results-table td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        .results-table tr:hover {
            background-color: #f5f5f5;
        }
        .share-ratio-highlight {
            background-color: #fff3cd !important;
            font-weight: bold;
        }
        .company-name {
            font-weight: bold;
            color: #2E86AB;
        }
        </style>
        <table class="results-table" id="resultsTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">Ordinary</th>
                    <th onclick="sortTable(1)">Preferred</th>
                    <th onclick="sortTable(2)">Company</th>
                    <th onclick="sortTable(3)">Ratio</th>
                    <th onclick="sortTable(4)" class="sort-desc">Current Spread</th>
                    <th onclick="sortTable(5)">Average Spread</th>
                    <th onclick="sortTable(6)">Spread Range</th>
                    <th onclick="sortTable(7)">Z-Score</th>
                    <th onclick="sortTable(8)">Std Dev</th>
                    <th onclick="sortTable(9)">Data Points</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Convert to list for sorting
        results_list = []
        for pair_key, result in valid_results.items():
            stats = result['stats']
            ord_ticker, pref_ticker = pair_key.split(' vs ')
            
            # Use ordinary ticker company name for consistency
            company_name = company_names.get(ord_ticker, ord_ticker.replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', ''))
            
            results_list.append({
                'pair_key': pair_key,
                'ord_ticker': ord_ticker,
                'pref_ticker': pref_ticker,
                'company_name': company_name,
                'stats': stats
            })
        
        # Sort by current spread (highest to lowest by default)
        results_list.sort(key=lambda x: x['stats']['current_spread'], reverse=True)
        
        for item in results_list:
            # Highlight rows with non-1.0 ratios
            ratio_class = "share-ratio-highlight" if item['stats']['share_ratio'] != 1.0 else ""
            
            table_html += f"""
            <tr class="{ratio_class}">
                <td>{item['ord_ticker'].replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '')}</td>
                <td>{item['pref_ticker'].replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '')}</td>
                <td class="company-name">{item['company_name']}</td>
                <td>{item['stats']['share_ratio']:.1f}</td>
                <td>{item['stats']['current_spread']:.3f}</td>
                <td>{item['stats']['average_spread']:.3f}</td>
                <td>{item['stats']['spread_range']}</td>
                <td>{item['stats']['current_zscore']:.2f}</td>
                <td>{item['stats']['std_spread']:.3f}</td>
                <td>{item['stats']['data_points']}</td>
            </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        
        <script>
        function sortTable(columnIndex) {
            var table = document.getElementById("resultsTable");
            var tbody = table.getElementsByTagName("tbody")[0];
            var rows = Array.from(tbody.getElementsByTagName("tr"));
            var headers = table.getElementsByTagName("th");
            
            // Clear all existing sort classes
            for (var i = 0; i < headers.length; i++) {
                headers[i].classList.remove("sort-asc", "sort-desc");
            }
            
            // Determine sort direction
            var currentHeader = headers[columnIndex];
            var isAscending = !currentHeader.hasAttribute("data-sort-desc");
            
            // Sort rows
            rows.sort(function(a, b) {
                var aText = a.getElementsByTagName("td")[columnIndex].textContent.trim();
                var bText = b.getElementsByTagName("td")[columnIndex].textContent.trim();
                
                // Handle numeric columns (columns 3-9)
                if (columnIndex >= 3) {
                    // Extract first number for range columns
                    var aVal = parseFloat(aText.split(' ')[0]) || 0;
                    var bVal = parseFloat(bText.split(' ')[0]) || 0;
                    
                    if (isAscending) {
                        return aVal - bVal;
                    } else {
                        return bVal - aVal;
                    }
                } else {
                    // String comparison for text columns
                    if (isAscending) {
                        return aText.localeCompare(bText);
                    } else {
                        return bText.localeCompare(aText);
                    }
                }
            });
            
            // Update header styling
            if (isAscending) {
                currentHeader.classList.add("sort-asc");
                currentHeader.removeAttribute("data-sort-desc");
            } else {
                currentHeader.classList.add("sort-desc");
                currentHeader.setAttribute("data-sort-desc", "true");
            }
            
            // Rebuild table body
            rows.forEach(function(row) {
                tbody.appendChild(row);
            });
        }
        </script>
        """
        
        # Add error summary
        error_results = {k: v for k, v in bulk_results.items() if 'error' in v}
        if error_results:
            table_html += f"<p style='color: orange; font-size: 12px;'>⚠️ {len(error_results)} pairs failed analysis: "
            failed_pairs = [k.replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '') for k in error_results.keys()]
            table_html += ", ".join(failed_pairs[:3])
            if len(failed_pairs) > 3:
                table_html += f" and {len(failed_pairs)-3} more"
            table_html += "</p>"
        
        table_html += "<p style='color: #666; font-size: 12px; margin-top: 10px;'>💡 Highlighted rows have share ratios ≠ 1.0. Click buttons below for detailed analysis.</p>"
        
        table_widget = widgets.HTML(value=table_html)
        
        # Create buttons for each valid pair (using sorted order)
        pair_buttons = []
        for item in results_list:
            btn = widgets.Button(
                description=f"📊 {item['company_name'][:15]}{'...' if len(item['company_name']) > 15 else ''}",
                button_style='',
                layout=widgets.Layout(width='300px', margin='2px'),
                tooltip=f"{item['ord_ticker'].replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '')} vs {item['pref_ticker'].replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '')} - {item['company_name']}"
            )
            btn.pair_key = item['pair_key']
            btn.on_click(self._show_detailed_analysis)
            pair_buttons.append(btn)
        
        return widgets.VBox([
            widgets.HTML(f"<h4 style='color: #2E86AB;'>📈 Bulk Analysis Summary ({len(valid_results)} successful)</h4>"),
            table_widget,
            widgets.HTML("<h5 style='color: #666;'>Click buttons below for detailed analysis:</h5>"),
            widgets.VBox(pair_buttons)
        ])
    
    def _show_detailed_analysis(self, btn):
        """Show detailed analysis for a specific pair"""
        pair_key = btn.pair_key
        result = self.bulk_results[pair_key]
        
        with self.output_area:
            clear_output()
            
            ord_ticker, pref_ticker = pair_key.split(' vs ')
            period_days = (result['end_date'] - result['start_date']).days
            summary_html = self.generate_summary_html(result['stats'], period_days, ord_ticker, pref_ticker)
            display(widgets.HTML(summary_html))
            
            self.create_spread_plots(result['data'], ord_ticker, pref_ticker, result['stats'], period_days)
    
    def create_spread_plots(self, data, ord_ticker, pref_ticker, stats, period_days):
        """Create comprehensive spread analysis plots"""
        
        try:
            plt.style.use('default')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.patch.set_facecolor('white')
            fig.suptitle(f'Spread Analysis: {ord_ticker.replace(" US EQUITY", "").replace(" KS EQUITY", "").replace(" IM EQUITY", "")} vs {pref_ticker.replace(" US EQUITY", "").replace(" KS EQUITY", "").replace(" IM EQUITY", "")}', 
                         fontsize=16, fontweight='bold', color='#2E86AB')
            
            # Plot 1: Normalized price comparison
            norm_ord = data['price_ord'] / data['price_ord'].iloc[0] * 100
            norm_pref = data['price_pref'] / data['price_pref'].iloc[0] * 100
            
            ax1.plot(data['date'], norm_ord, 
                    label=f"Ordinary ({ord_ticker.replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '')})", 
                    linewidth=2.5, color='#2E86AB')
            ax1.plot(data['date'], norm_pref, 
                    label=f"Preferred ({pref_ticker.replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '')})", 
                    linewidth=2.5, color='#A23B72')
            ax1.set_title('Normalized Price Performance (Base = 100)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Normalized Price', fontsize=11)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#f8f9fa')
            
            # Plot 2: Spread over time
            ax2.plot(data['date'], data['spread'], color='#F18F01', linewidth=2.5, label='Spread')
            ax2.axhline(y=stats['average_spread'], color='red', linestyle='--', alpha=0.8, linewidth=2, label='Average')
            
            if stats['std_spread'] > 0 and not np.isinf(stats['std_spread']):
                ax2.fill_between(data['date'], 
                                stats['average_spread'] - stats['std_spread'],
                                stats['average_spread'] + stats['std_spread'],
                                alpha=0.2, color='red', label='±1 Std Dev')
            
            ax2.set_title(f'{stats["spread_label"]} Over Time', fontsize=12, fontweight='bold')
            ax2.set_ylabel(stats['spread_label'], fontsize=11)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#f8f9fa')
            
            # Plot 3: Z-score
            if 'z_score_rolling' in data.columns and not data['z_score_rolling'].isna().all():
                ax3.plot(data['date'], data['z_score_rolling'], color='#C73E1D', linewidth=2.5, label='Rolling Z-Score')
            
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            ax3.axhline(y=2, color='red', linestyle='--', alpha=0.7, linewidth=2, label='±2 Threshold')
            ax3.axhline(y=-2, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax3.fill_between(data['date'], -2, 2, alpha=0.1, color='green', label='Normal Range')
            ax3.set_title('Rolling Z-Score (30-day)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Z-Score', fontsize=11)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.set_facecolor('#f8f9fa')
            
            # Plot 4: Spread distribution
            ax4.hist(data['spread'], bins=min(30, len(data)//2), alpha=0.7, color='#59A5D8', edgecolor='black', linewidth=1)
            ax4.axvline(x=stats['current_spread'], color='red', linestyle='--', linewidth=3, label='Current')
            ax4.axvline(x=stats['average_spread'], color='green', linestyle='--', linewidth=3, label='Average')
            ax4.set_title('Spread Distribution', fontsize=12, fontweight='bold')
            ax4.set_xlabel(stats['spread_label'], fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)
            ax4.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            plt.show()
            
            return fig
            
        except Exception as e:
            return None
    
    def generate_summary_html(self, stats, period_days, ord_ticker, pref_ticker):
        """Generate HTML summary with updated ratio explanation"""
        
        ratio_note = ""
        if stats['share_ratio'] != 1.0:
            ratio_note = f"<p style='margin: 5px 0; color: #555;'><strong>Share Ratio Applied:</strong> {stats['share_ratio']:.2f} (Preferred prices adjusted)</p>"
        
        html = f"""
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 20px 0;'>
            <h3 style='color: #2E86AB; margin-top: 0; font-family: Arial, sans-serif;'>
                📈 Analysis Summary ({period_days} days, {stats['data_points']} data points)
            </h3>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;'>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB;'>
                    <h4 style='margin: 0 0 10px 0; color: #333;'>📊 Spread Metrics</h4>
                    <p style='margin: 5px 0; color: #555;'><strong>Current {stats['spread_label']}:</strong> {stats['current_spread']:.3f}</p>
                    <p style='margin: 5px 0; color: #555;'><strong>Average {stats['spread_label']}:</strong> {stats['average_spread']:.3f}</p>
                    <p style='margin: 5px 0; color: #555;'><strong>Max {stats['spread_label']}:</strong> {stats['max_spread']:.3f}</p>
                    <p style='margin: 5px 0; color: #555;'><strong>Min {stats['spread_label']}:</strong> {stats['min_spread']:.3f}</p>
                    <p style='margin: 5px 0; color: #555;'><strong>Volatility (Std Dev):</strong> {stats['std_spread']:.3f}</p>
                </div>
                
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #A23B72;'>
                    <h4 style='margin: 0 0 10px 0; color: #333;'>📊 Statistical Measures</h4>
                    <p style='margin: 5px 0; color: #555;'><strong>Current Z-Score:</strong> {stats['current_zscore']:.2f}</p>
                    <p style='margin: 5px 0; color: #555;'><strong>Spread Range:</strong> {stats['spread_range']}</p>
                    <p style='margin: 5px 0; color: #555;'><strong>Mean Deviation:</strong> {abs(stats['current_spread'] - stats['average_spread']):.3f}</p>
                    {ratio_note}
                </div>
            </div>
            
            <div style='background: linear-gradient(90deg, #e3f2fd, #fce4ec); padding: 15px; border-radius: 8px; margin-top: 20px;'>
                <h4 style='margin: 0 0 10px 0; color: #333;'>ℹ️ Analysis Details</h4>
                <p style='margin: 5px 0; color: #555; font-size: 14px;'>
                    <strong>Ordinary Shares:</strong> {ord_ticker.replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '')}<br>
                    <strong>Preferred Shares:</strong> {pref_ticker.replace(' US EQUITY', '').replace(' KS EQUITY', '').replace(' IM EQUITY', '')}<br>
                    <strong>Period:</strong> {period_days} days<br>
                    <strong>Currency:</strong> USD (all prices converted)<br>
                    <strong>Spread Formula:</strong> 1 - preferred_price/ordinary_price (positive = preferred discount)<br>
                    <strong>Z-Score:</strong> (Current - Overall Mean) / Overall Std Dev
                </p>
            </div>
        </div>
        """
        
        return html
    
    def _analyze_single_pair(self, btn):
        """Analyze single ticker pair"""
        ord_ticker = self.normalize_ticker(self.single_ticker1.value)
        pref_ticker = self.normalize_ticker(self.single_ticker2.value)
        share_ratio = self.single_share_ratio.value
        
        if not ord_ticker or not pref_ticker:
            with self.output_area:
                clear_output()
                display(widgets.HTML(
                    "<div style='background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;'>"
                    "❌ <strong>Error:</strong> Please enter valid tickers for both fields</div>"
                ))
            return
        
        if not self.validate_share_ratio(share_ratio):
            with self.output_area:
                clear_output()
                display(widgets.HTML(
                    "<div style='background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;'>"
                    f"❌ <strong>Error:</strong> Invalid share ratio: {share_ratio}</div>"
                ))
            return
            
        start_date, end_date = self.get_single_analysis_period()
        if not start_date or not end_date:
            with self.output_area:
                clear_output()
                display(widgets.HTML(
                    "<div style='background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;'>"
                    "❌ <strong>Error:</strong> Please select a valid date range</div>"
                ))
            return
        
        method = self.single_spread_method.value
        
        with self.output_area:
            clear_output()
            print(f"🔍 Analyzing: {ord_ticker} vs {pref_ticker} (ratio: {share_ratio})...")
            
        self._run_single_analysis(ord_ticker, pref_ticker, start_date, end_date, method, share_ratio)
    
    def _analyze_bulk_pairs(self, btn):
        """Analyze multiple ticker pairs"""
        
        try:
            ord_raw = self.bulk_t1_input.value.strip()
            pref_raw = self.bulk_t2_input.value.strip()
            ratios_raw = self.bulk_ratios_input.value.strip()
            
            if not ord_raw or not pref_raw:
                raise ValueError("Ticker lists cannot be empty")
            
            ord_list = [self.normalize_ticker(t.strip()) for t in ord_raw.split(',')]
            pref_list = [self.normalize_ticker(t.strip()) for t in pref_raw.split(',')]
            
            ord_list = [t for t in ord_list if t is not None]
            pref_list = [t for t in pref_list if t is not None]
            
            if len(ord_list) == 0 or len(pref_list) == 0:
                raise ValueError("No valid tickers found")
            
            if ratios_raw:
                ratio_strings = [r.strip() for r in ratios_raw.split(',') if r.strip()]
                ratio_list = []
                for r_str in ratio_strings:
                    try:
                        ratio = float(r_str)
                        if not self.validate_share_ratio(ratio):
                            ratio = 1.0
                        ratio_list.append(ratio)
                    except ValueError:
                        ratio_list.append(1.0)
            else:
                ratio_list = []
            
        except Exception as e:
            with self.bulk_results_area:
                clear_output()
                display(widgets.HTML(
                    f"<div style='background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;'>"
                    f"❌ <strong>Parsing Error:</strong> {str(e)}</div>"
                ))
            return
            
        if len(ord_list) != len(pref_list):
            with self.bulk_results_area:
                clear_output()
                display(widgets.HTML(
                    "<div style='background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;'>"
                    f"❌ <strong>Error:</strong> Ordinary list has {len(ord_list)} tickers but Preferred list has {len(pref_list)}. Lists must have same length.</div>"
                ))
            return
        
        while len(ratio_list) < len(ord_list):
            ratio_list.append(1.0)
        
        start_date, end_date = self.get_bulk_analysis_period()
        if not start_date or not end_date:
            with self.bulk_results_area:
                clear_output()
                display(widgets.HTML(
                    "<div style='background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;'>"
                    "❌ <strong>Error:</strong> Please select a valid date range</div>"
                ))
            return
        
        method = self.bulk_spread_method.value
        
        with self.bulk_results_area:
            clear_output()
            print(f"🔍 Analyzing {len(ord_list)} pairs...")
        
        self.bulk_results = {}
        successful = 0
        
        for i, (ord, pref, ratio) in enumerate(zip(ord_list, pref_list, ratio_list)):
            pair_key = f"{ord} vs {pref}"
            print(f"Processing pair {i+1}/{len(ord_list)}: {pair_key} (ratio: {ratio})")
            
            try:
                result = self._run_bulk_pair_analysis(ord, pref, start_date, end_date, method, ratio)
                self.bulk_results[pair_key] = result
                successful += 1
                    
            except Exception as e:
                self.bulk_results[pair_key] = {'error': str(e)}
        
        print(f"✅ Completed: {successful}/{len(ord_list)} successful")
        
        with self.bulk_results_area:
            clear_output()
            summary_table = self.create_bulk_summary_table(self.bulk_results)
            display(summary_table)
    
    def _run_bulk_pair_analysis(self, ord_ticker, pref_ticker, start_date, end_date, method, share_ratio):
        """Run analysis for a single pair"""
        
        ord_data = self.get_price_history(ord_ticker, start_date, end_date)
        pref_data = self.get_price_history(pref_ticker, start_date, end_date)
        
        if ord_data is None:
            raise Exception(f"Failed to get price data for {ord_ticker}")
        if pref_data is None:
            raise Exception(f"Failed to get price data for {pref_ticker}")
        
        spread_data, stats = self.calculate_spread_metrics(ord_data, pref_data, method, share_ratio)
        
        return {
            'data': spread_data,
            'stats': stats,
            'start_date': start_date,
            'end_date': end_date
        }
    
    def _run_single_analysis(self, ord_ticker, pref_ticker, start_date, end_date, method, share_ratio):
        """Execute single pair analysis"""
        
        try:
            result = self._run_bulk_pair_analysis(ord_ticker, pref_ticker, start_date, end_date, method, share_ratio)
            
            with self.output_area:
                clear_output()
                
                period_days = (end_date - start_date).days
                summary_html = self.generate_summary_html(result['stats'], period_days, ord_ticker, pref_ticker)
                display(widgets.HTML(summary_html))
                
                self.create_spread_plots(result['data'], ord_ticker, pref_ticker, result['stats'], period_days)
                
        except Exception as e:
            with self.output_area:
                clear_output()
                display(widgets.HTML(
                    f"<div style='background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;'>"
                    f"❌ <strong>Analysis Failed:</strong> {str(e)}</div>"
                ))

# Launch the dashboard
def launch_spread_dashboard():
    """Launch the dual-class spread dashboard"""
    return DualClassSpreadDashboard()

# Initialize
dashboard = launch_spread_dashboard()