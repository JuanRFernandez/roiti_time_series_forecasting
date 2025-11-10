"""
Feature engineering for time series forecasting
"""
import pandas as pd
import numpy as np


class TimeSeriesFeatureEngineer:
    """Create features for time series forecasting"""
    
    def __init__(self, target_col='Total'):
        self.target_col = target_col
    
    def add_lag_features(self, df, lags=[24, 48, 168]):
        """
        Add lagged features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with datetime index
        lags : list
            List of lag periods (in hours)
        """
        df_feat = df.copy()
        
        for lag in lags:
            df_feat[f'lag_{lag}h'] = df_feat[self.target_col].shift(lag)
        
        return df_feat
    
    def add_rolling_features(self, df, windows=[24, 48]):
        """
        Add rolling window statistics
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        windows : list
            Window sizes (in hours)
        """
        df_feat = df.copy()
        
        for window in windows:
            # Shift by 1 to avoid leakage (don't include current hour)
            df_feat[f'rolling_mean_{window}h'] = (
                df_feat[self.target_col].shift(1).rolling(window=window).mean()
            )
            df_feat[f'rolling_std_{window}h'] = (
                df_feat[self.target_col].shift(1).rolling(window=window).std()
            )
        
        return df_feat
    
    def add_time_features(self, df):
        """Add time-based features"""
        df_feat = df.copy()
        
        # Basic time features
        df_feat['hour'] = df_feat.index.hour
        df_feat['day_of_week'] = df_feat.index.dayofweek
        df_feat['month'] = df_feat.index.month
        df_feat['day_of_year'] = df_feat.index.dayofyear
        
        # Cyclical encoding for periodic features
        df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
        df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
        df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
        df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
        
        return df_feat
    
    def create_all_features(self, df, lags=[24, 48, 168], windows=[24, 48]):
        """
        Create complete feature set
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with target column
        lags : list
            Lag periods to create
        windows : list
            Rolling window sizes
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with all features
        """
        df_feat = df.copy()
        
        # Add all feature types
        df_feat = self.add_lag_features(df_feat, lags)
        df_feat = self.add_rolling_features(df_feat, windows)
        df_feat = self.add_time_features(df_feat)
        
        # Drop rows with NaN (from rolling windows)
        df_feat = df_feat.dropna()
        
        return df_feat
    
    def get_feature_columns(self, df):
        """Get list of feature columns (excluding target and original data)"""
        exclude = [self.target_col, 'Solar', 'Wind_Onshore', 'Wind_Offshore']
        return [col for col in df.columns if col not in exclude]


if __name__ == '__main__':
    # Example usage
    from data_loader import RenewableDataLoader
    
    loader = RenewableDataLoader()
    df = loader.load_and_preprocess()
    
    engineer = TimeSeriesFeatureEngineer(target_col='Total')
    df_features = engineer.create_all_features(df)
    
    feature_cols = engineer.get_feature_columns(df_features)
    print(f"Created {len(feature_cols)} features:")
    for col in feature_cols:
        print(f"  - {col}")

