"""
Data loading and preprocessing module for renewable energy forecasting
"""
import pandas as pd
import numpy as np
from pathlib import Path


class RenewableDataLoader:
    """Load and preprocess renewable energy generation data"""
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
    
    def load_yearly_data(self, year):
        """Load data for a specific year"""
        filename = f'energy-charts_Public_net_electricity_generation_in_Germany_in_{year}.csv'
        filepath = self.data_dir / filename
        
        # Skip row 1 (units row), keep row 0 (technology names)
        df = pd.read_csv(filepath, skiprows=[1])
        return df
    
    def load_all_years(self, years=[2022, 2023, 2024]):
        """Load and combine multiple years"""
        dfs = []
        for year in years:
            df = self.load_yearly_data(year)
            dfs.append(df)
        
        # Combine
        combined = pd.concat(dfs, ignore_index=True)
        
        # Parse dates
        date_col = combined.columns[0]
        combined[date_col] = pd.to_datetime(combined[date_col], utc=True)
        combined = combined.set_index(date_col)
        combined.index = combined.index.tz_localize(None)
        combined.index.name = 'DateTime'
        
        return combined
    
    def aggregate_by_technology(self, df):
        """Aggregate generation by technology type across all TSOs"""
        df_agg = pd.DataFrame(index=df.index)
        
        # Sum across all TSOs for each technology
        df_agg['Solar'] = df[[col for col in df.columns if 'Solar' in col]].sum(axis=1)
        df_agg['Wind_Onshore'] = df[[col for col in df.columns if 'Wind onshore' in col]].sum(axis=1)
        df_agg['Wind_Offshore'] = df[[col for col in df.columns if 'Wind offshore' in col]].sum(axis=1)
        
        return df_agg
    
    def resample_to_hourly(self, df):
        """Resample 15-minute data to hourly averages"""
        return df.resample('1h').mean()
    
    def load_and_preprocess(self, years=[2022, 2023, 2024], save_path=None):
        """
        Complete pipeline: Load, aggregate, resample
        
        Parameters:
        -----------
        years : list
            Years to load
        save_path : str, optional
            Path to save processed data
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed hourly renewable generation data
        """
        # Load
        df = self.load_all_years(years)
        
        # Aggregate
        df_agg = self.aggregate_by_technology(df)
        
        # Resample
        df_hourly = self.resample_to_hourly(df_agg)
        
        # Add total
        df_hourly['Total'] = df_hourly.sum(axis=1)
        
        # Save if requested
        if save_path:
            df_hourly.to_csv(save_path)
        
        return df_hourly


if __name__ == '__main__':
    # Example usage
    loader = RenewableDataLoader()
    renewable = loader.load_and_preprocess(save_path='data/processed/renewable_hourly.csv')
    print(f"Loaded and processed {len(renewable)} hours of data")
    print(f"Columns: {renewable.columns.tolist()}")
    print(renewable.describe())

