import bql
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProductionBQLDashboard:
    def __init__(self):
        """Initialize production BQL dashboard"""
        try:
            self.bq = bql.Service()
            print("Connected to Bloomberg BQL service")
            self.has_bql = True
        except Exception as e:
            print(f"Bloomberg connection failed: {e}")
            self.has_bql = False
            
        # Default years
        self.current_year = datetime.now().year
        self.base_year = 2025
        self.target_year = 2027
        self.cagr_years = 2
        
        self.setup_dashboard()
    
    def extract_metric_value(self, bql_response_df):
        """Extract metric value, return None if fails"""
        if bql_response_df is None or bql_response_df.empty:
            return None
        
        # Skip metadata columns
        metadata_cols = ['REVISION_DATE', 'AS_OF_DATE', 'PERIOD_END_DATE', 'CURRENCY']
        metric_cols = [col for col in bql_response_df.columns if col not in metadata_cols]
        
        if metric_cols:
            try:
                value = bql_response_df[metric_cols[-1]].iloc[0]
                # Convert to float, return None if fails
                if pd.isna(value):
                    return None
                return float(value)
            except:
                return None
        return None
    
    def normalize_ticker(self, ticker):
        """Convert ticker to Bloomberg format"""
        ticker = ticker.strip().upper()
        if 'EQUITY' in ticker:
            return ticker
        return f"{ticker} US EQUITY"
    
    def setup_dashboard(self):
        """Create dashboard with custom metric support"""
        
        # Individual ticker management
        self.ticker_input = widgets.Text(
            value='',
            placeholder='Single ticker (e.g., AAPL, MSFT)'
        )
        
        self.add_ticker_btn = widgets.Button(
            description='Add Single',
            button_style='success'
        )
        
        # Bulk ticker management
        self.bulk_ticker_input = widgets.Textarea(
            value='',
            placeholder='Paste comma-separated tickers from Bloomberg:\nAAPL US EQUITY, MSFT US EQUITY, GOOGL US EQUITY, NVDA US EQUITY',
            description='Bulk Add:',
            rows=3
        )
        
        self.bulk_add_btn = widgets.Button(
            description='Bulk Add Tickers',
            button_style='info',
            icon='list'
        )
        
        self.ticker_list = widgets.SelectMultiple(
            options=['AAPL US EQUITY', 'MSFT US EQUITY', 'GOOGL US EQUITY', 'NVDA US EQUITY', 'CRM US EQUITY'],
            value=['AAPL US EQUITY', 'MSFT US EQUITY', 'GOOGL US EQUITY', 'NVDA US EQUITY'],
            rows=6
        )
        
        self.remove_ticker_btn = widgets.Button(
            description='Remove Selected',
            button_style='warning'
        )
        
        # Y-axis (vertical) - Current Valuation Multiples
        y_axis_options = [
            ('EV / Sales', 'HEADLINE_EV_TO_SALES'),
            ('EV / EBITDA', 'EV_TO_EBITDA'), 
            ('EV / EBITA', 'EV_TO_EBITA'),
            ('EV / EBIT', 'EV_TO_EBIT'),
            ('P / E', 'HEADLINE_PE_RATIO'),
            ('FCF yield', 'FCF_EV_YIELD'),
            ('LFCF yield', 'FCF_YIELD_WITH_CUR_MKT_CAP')
        ]
        
        self.y_metric = widgets.Dropdown(
            options=y_axis_options,
            value='HEADLINE_EV_TO_SALES',
            description='Y-axis:'
        )
        
        # Custom Y input
        self.custom_y_input = widgets.Text(
            value='',
            placeholder='Enter BQL Mnemonic (e.g., PE_RATIO, TOT_RETURN_1YR)',
            description='Custom Y:'
        )
        
        # X-axis (horizontal) - Growth Metrics 
        x_axis_options = [
            ('Rev', 'IS_COMP_SALES'),
            ('EBITDA', 'IS_COMPARABLE_EBITDA'),
            ('EBITA', 'EBITDA'),
            ('EBIT', 'IS_COMPARABLE_EBIT'),
            ('Adj EPS', 'IS_DIL_EPS_CONT_OPS'),
            ('LFCF per share', 'FCF_PER_DIL_SHR')
        ]
        
        self.x_metric = widgets.Dropdown(
            options=x_axis_options,
            value='IS_COMP_SALES',
            description='X-axis:'
        )
        
        # Custom X input
        self.custom_x_input = widgets.Text(
            value='',
            placeholder='Enter BQL Mnemonic (e.g., IS_COMP_NET_INCOME)',
            description='Custom X:'
        )
        
        # Year configuration
        self.base_year_input = widgets.IntText(
            value=2025,
            description='Base Year:'
        )
        
        self.cagr_years_input = widgets.IntText(
            value=self.cagr_years,
            description='CAGR Years:'
        )
        
        # Analysis button
        self.analyze_btn = widgets.Button(
            description='Run Analysis',
            button_style='primary'
        )
        
        # Output area
        self.output = widgets.Output()
        
        # Event handlers
        self.add_ticker_btn.on_click(self._add_single_ticker)
        self.bulk_add_btn.on_click(self._bulk_add_tickers)
        self.remove_ticker_btn.on_click(self._remove_ticker)
        self.analyze_btn.on_click(self._run_analysis)
        
        self._display_dashboard()
    
    def _add_single_ticker(self, btn):
        """Add single ticker"""
        raw_ticker = self.ticker_input.value.strip()
        if not raw_ticker:
            return
            
        normalized_ticker = self.normalize_ticker(raw_ticker)
        
        if normalized_ticker not in self.ticker_list.options:
            current_options = list(self.ticker_list.options)
            current_values = list(self.ticker_list.value)
            
            current_options.append(normalized_ticker)
            current_values.append(normalized_ticker)
            
            self.ticker_list.options = current_options
            self.ticker_list.value = current_values
            self.ticker_input.value = ''
            
            with self.output:
                print(f"Added: {normalized_ticker}")
    
    def _bulk_add_tickers(self, btn):
        """Bulk add comma-separated tickers"""
        bulk_input = self.bulk_ticker_input.value.strip()
        if not bulk_input:
            with self.output:
                clear_output(wait=True)
                print("Please enter comma-separated tickers")
            return
        
        # Split by comma and clean up
        raw_tickers = [t.strip() for t in bulk_input.split(',') if t.strip()]
        normalized_tickers = [self.normalize_ticker(t) for t in raw_tickers]
        
        with self.output:
            clear_output(wait=True)
            print(f"Processing {len(raw_tickers)} tickers...")
        
        # Process each ticker
        current_options = list(self.ticker_list.options)
        current_values = list(self.ticker_list.value)
        added_count = 0
        skipped_count = 0
        
        for ticker in normalized_tickers:
            if ticker and ticker not in current_options:
                current_options.append(ticker)
                current_values.append(ticker)
                added_count += 1
                print(f"  Added: {ticker}")
            else:
                skipped_count += 1
                print(f"  Skipped: {ticker} (already exists or invalid)")
        
        # Update the ticker list
        self.ticker_list.options = current_options
        self.ticker_list.value = current_values
        self.bulk_ticker_input.value = ''
        
        print(f"\nBulk add complete: {added_count} added, {skipped_count} skipped")
    
    def _remove_ticker(self, btn):
        """Remove selected tickers"""
        selected = list(self.ticker_list.value)
        remaining = [opt for opt in self.ticker_list.options if opt not in selected]
        
        self.ticker_list.options = remaining
        self.ticker_list.value = remaining[:min(4, len(remaining))]
    
    def _get_ticker_data(self, ticker, x_field_code, y_field_code, is_custom_y=False, is_custom_x=False):
        """Get data for one ticker - return NaN if fails, no filler data"""
        if not self.has_bql:
            return {
                'ticker': ticker,
                'company_name': ticker.replace(' US EQUITY', ''),
                'x_value': np.nan,
                'y_value': np.nan
            }
        
        try:
            # Y-axis: Current multiple
            if is_custom_y:
                # Use exact mnemonic as typed - NO PARAMETERS
                y_query = f"for('{ticker}') get({y_field_code.lower()})"
            else:
                # Use standard format with parameters for predefined fields
                y_query = f"for('{ticker}') get({y_field_code.lower()}(ae=E))"
                
            y_response = self.bq.execute(y_query)
            y_value = self.extract_metric_value(y_response[0].df())
            
            # X-axis: CAGR calculation
            if is_custom_x:
                # Use exact mnemonic as typed - NO ae=E, only fpr for years
                base_query = f"for('{ticker}') get({x_field_code.lower()}(fpr='{self.base_year}'))"
                target_query = f"for('{ticker}') get({x_field_code.lower()}(fpr='{self.target_year}'))"
            else:
                # Use standard format for predefined fields
                base_query = f"for('{ticker}') get({x_field_code.lower()}(fpr='{self.base_year}', ae=E))"
                target_query = f"for('{ticker}') get({x_field_code.lower()}(fpr='{self.target_year}', ae=E))"
            
            base_response = self.bq.execute(base_query)
            target_response = self.bq.execute(target_query)
            
            base_value = self.extract_metric_value(base_response[0].df())
            target_value = self.extract_metric_value(target_response[0].df())
            
            # Calculate CAGR or return NaN
            x_value = np.nan
            if base_value is not None and target_value is not None and base_value > 0:
                x_value = ((target_value / base_value) ** (1/self.cagr_years) - 1) * 100
            
            return {
                'ticker': ticker,
                'company_name': ticker.replace(' US EQUITY', '').replace(' EQUITY', ''),
                'x_value': x_value,
                'y_value': y_value
            }
            
        except Exception as e:
            # Return NaN values when error occurs
            return {
                'ticker': ticker,
                'company_name': ticker.replace(' US EQUITY', '').replace(' EQUITY', ''),
                'x_value': np.nan,
                'y_value': np.nan
            }
    
    def _run_analysis(self, btn):
        """Execute analysis"""
        # Clear any existing plots
        plt.close('all')
        
        # Update configuration
        self.base_year = self.base_year_input.value
        self.cagr_years = self.cagr_years_input.value
        self.target_year = self.base_year + self.cagr_years
        
        with self.output:
            clear_output(wait=True)
            
            tickers = list(self.ticker_list.value)
            
            # Check if custom inputs are being used
            is_custom_y = bool(self.custom_y_input.value.strip())
            is_custom_x = bool(self.custom_x_input.value.strip())
            
            # Custom inputs automatically override dropdown selections
            if is_custom_y:
                y_field_code = self.custom_y_input.value.strip().upper()
                y_label = f"Custom ({y_field_code})"
            else:
                y_field_code = self.y_metric.value
                y_label = [opt[0] for opt in self.y_metric.options if opt[1] == y_field_code][0]
                
            if is_custom_x:
                x_field_code = self.custom_x_input.value.strip().upper()
                x_label = f"Custom ({x_field_code})"
            else:
                x_field_code = self.x_metric.value
                x_label = [opt[0] for opt in self.x_metric.options if opt[1] == x_field_code][0]
            
            if not tickers:
                print("Please select at least one ticker")
                return
                
            print(f"Running analysis for {len(tickers)} securities...")
            print(f"Y-axis: {y_label} (current)")
            print(f"X-axis: {x_label} ({self.cagr_years}Y CAGR, {self.base_year}-{self.target_year})")
            print(f"Using BQL fields: {y_field_code} vs {x_field_code}")
            
            # Fetch data for all tickers
            data_points = []
            for ticker in tickers:
                print(f"  Fetching {ticker}...")
                point = self._get_ticker_data(ticker, x_field_code, y_field_code, is_custom_y, is_custom_x)
                data_points.append(point)
            
            # Create DataFrame with all results (including NaN)
            df = pd.DataFrame(data_points)
            
            print(f"Retrieved data for {len(df)} securities")
            
            # Display data table showing NaN for failed tickers
            print("\nData Summary:")
            display_df = df[['company_name', 'x_value', 'y_value']].copy()
            display_df.columns = ['Company', f'{x_label} (%)', f'{y_label}']
            # Don't round NaN values
            display_df[f'{x_label} (%)'] = display_df[f'{x_label} (%)'].apply(lambda x: round(x, 1) if pd.notna(x) else np.nan)
            display_df[f'{y_label}'] = display_df[f'{y_label}'].apply(lambda x: round(x, 2) if pd.notna(x) else np.nan)
            display(display_df)
            
            # Create plot with only valid data points
            valid_df = df.dropna(subset=['x_value', 'y_value'])
            
            if len(valid_df) > 0:
                self._create_plot(valid_df, x_label, y_label)
            else:
                print("No valid data points to plot")
    
    def _create_plot(self, df, x_label, y_label):
        """Create single plot with valid data only"""
        
        plt.figure(figsize=(12, 8))
        
        # Plot points
        colors = plt.cm.Set1(np.linspace(0, 1, len(df)))
        
        for i, (_, row) in enumerate(df.iterrows()):
            plt.scatter(row['x_value'], row['y_value'], 
                       c=[colors[i]], s=120, alpha=0.8,
                       edgecolors='black', linewidth=2)
            
            # Add labels
            plt.annotate(row['company_name'], 
                        (row['x_value'], row['y_value']),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        # Add trend line if enough points
        if len(df) > 1:
            z = np.polyfit(df['x_value'], df['y_value'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(df['x_value'].min(), df['x_value'].max(), 100)
            plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
        
        # Formatting
        plt.xlabel(f'{x_label} ({self.cagr_years}Y CAGR %)', fontsize=12, fontweight='bold')
        plt.ylabel(f'{y_label} (Multiple)', fontsize=12, fontweight='bold')
        plt.title(f'{y_label} vs {x_label} Growth Analysis', fontsize=14, fontweight='bold')
        
        # Set axis limits to start from 0
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()
        
        # Print analysis
        if len(df) > 1:
            corr = df['x_value'].corr(df['y_value'])
            print(f"\nAnalysis Summary:")
            print(f"  Correlation: {corr:.3f}")
            print(f"  Valid data points: {len(df)}")
    
    def _display_dashboard(self):
        """Display dashboard"""
        
        title = widgets.HTML("<h2>Bloomberg BQL Dashboard - Production</h2>")
        
        # Year configuration
        year_section = widgets.VBox([
            widgets.HTML("<h3>Configuration</h3>"),
            self.base_year_input,
            self.cagr_years_input
        ])
        
        # Ticker section
        ticker_section = widgets.VBox([
            widgets.HTML("<h3>Ticker Management</h3>"),
            widgets.HTML("<p><b>Single Add:</b></p>"),
            widgets.HBox([self.ticker_input, self.add_ticker_btn]),
            widgets.HTML("<p><b>Bulk Add:</b> Copy from Bloomberg Terminal</p>"),
            self.bulk_ticker_input,
            self.bulk_add_btn,
            widgets.HTML("<p><b>Selected Tickers:</b></p>"),
            self.ticker_list,
            self.remove_ticker_btn
        ])
        
        # Metrics section with clear Y/X labeling
        metrics_section = widgets.VBox([
            widgets.HTML("<h3>Metrics</h3>"),
            widgets.HTML("<p><b>Y-AXIS (Vertical) - Current Valuation:</b></p>"),
            self.y_metric,
            widgets.HTML("<p><b>Custom Y (auto-overrides dropdown):</b></p>"),
            self.custom_y_input,
            widgets.HTML("<p><b>X-AXIS (Horizontal) - Growth CAGR:</b></p>"),
            self.x_metric,
            widgets.HTML("<p><b>Custom X (auto-overrides dropdown):</b></p>"),
            self.custom_x_input,
            self.analyze_btn
        ])
        
        # Layout
        main_layout = widgets.VBox([
            title,
            widgets.HBox([year_section, ticker_section, metrics_section]),
            widgets.HTML("<hr>"),
            self.output
        ])
        
        display(main_layout)

def launch_production_dashboard():
    """Launch the production dashboard"""
    return ProductionBQLDashboard()

def bulk_analysis_from_paste(ticker_string, x_field='IS_COMP_SALES', y_field='HEADLINE_EV_TO_SALES', base_year=2025, cagr_years=2):
    """Analyze tickers directly from pasted Bloomberg list"""
    dashboard = ProductionBQLDashboard()
    
    # Configure years
    dashboard.base_year = base_year
    dashboard.cagr_years = cagr_years
    dashboard.target_year = base_year + cagr_years
    
    # Parse comma-separated ticker string
    raw_tickers = [t.strip() for t in ticker_string.split(',') if t.strip()]
    normalized_tickers = [dashboard.normalize_ticker(t) for t in raw_tickers]
    
    print(f"Bulk Analysis from paste: {len(raw_tickers)} tickers")
    print(f"BQL fields: {y_field} vs {x_field}")
    print(f"Tickers: {', '.join([t.replace(' US EQUITY', '') for t in normalized_tickers])}")
    
    # Get data for all tickers - determine if custom fields
    is_custom_y = y_field not in ['HEADLINE_EV_TO_SALES', 'EV_TO_EBITDA', 'EV_TO_EBITA', 'EV_TO_EBIT', 'HEADLINE_PE_RATIO', 'FCF_EV_YIELD', 'FCF_YIELD_WITH_CUR_MKT_CAP']
    is_custom_x = x_field not in ['IS_COMP_SALES', 'IS_COMPARABLE_EBITDA', 'EBITDA', 'IS_COMPARABLE_EBIT', 'IS_DIL_EPS_CONT_OPS', 'FCF_PER_DIL_SHR']
    
    data_points = []
    for ticker in normalized_tickers:
        print(f"  Processing {ticker}...")
        point = dashboard._get_ticker_data(ticker, x_field, y_field, is_custom_y, is_custom_x)
        data_points.append(point)
    
    df = pd.DataFrame(data_points)
    
    # Show results table with NaN
    print(f"\nResults:")
    display_df = df[['company_name', 'x_value', 'y_value']].copy()
    display_df.columns = ['Company', 'Growth CAGR (%)', 'Multiple']
    display_df['Growth CAGR (%)'] = display_df['Growth CAGR (%)'].apply(lambda x: round(x, 1) if pd.notna(x) else np.nan)
    display_df['Multiple'] = display_df['Multiple'].apply(lambda x: round(x, 2) if pd.notna(x) else np.nan)
    display(display_df)
    
    # Create plot with valid points only - NO DUPLICATE
    valid_df = df.dropna(subset=['x_value', 'y_value'])
    if len(valid_df) > 0:
        # Clear existing plots first
        plt.close('all')
        dashboard._create_plot(valid_df, 'Growth', 'Multiple')
        print(f"\nPlotted {len(valid_df)} valid points out of {len(df)} total")
    else:
        print("No valid data points to plot - all returned NaN")
    
    return df

# Launch dashboard
dashboard = launch_production_dashboard()