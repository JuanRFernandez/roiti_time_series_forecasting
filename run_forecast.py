"""
Main script to run renewable energy forecasting pipeline

Usage:
    python run_forecast.py
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from data_loader import RenewableDataLoader
from feature_engineer import TimeSeriesFeatureEngineer
from models import BaselineModel, SARIMAXModel, SVRModel, ModelEvaluator
import config


def main():
    print("="*70)
    print("DAY-AHEAD RENEWABLE ENERGY FORECASTING")
    print("="*70)
    
    # 1. Load data
    print("\n1. Loading data...")
    loader = RenewableDataLoader(data_dir=config.DATA_RAW_DIR)
    renewable = loader.load_and_preprocess(
        years=config.DATA_YEARS,
        save_path=f'{config.DATA_PROCESSED_DIR}/renewable_hourly.csv'
    )
    print(f"   OK Loaded {len(renewable):,} hours ({len(renewable)/24:.0f} days)")
    print(f"   Period: {renewable.index.min()} to {renewable.index.max()}")
    
    # 2. Feature engineering
    print("\n2. Creating features...")
    engineer = TimeSeriesFeatureEngineer(target_col='Total')
    df_features = engineer.create_all_features(renewable)
    feature_cols = engineer.get_feature_columns(df_features)
    print(f"   OK Created {len(feature_cols)} features")
    
    # 3. Train-test split
    print("\n3. Splitting data...")
    test_days = config.TEST_DAYS
    test_hours = test_days * config.FORECAST_HORIZON
    
    train_df = df_features[:-test_hours]
    test_df = df_features[-test_hours:]
    
    print(f"   Training: {len(train_df):,} hours")
    print(f"   Testing: {len(test_df):,} hours ({test_days} days)")
    print(f"   Test period: {test_df.index[0].date()} to {test_df.index[-1].date()}")
    
    # 4. Train and evaluate models
    print("\n4. Training models...")
    
    # Baseline
    print("   - Baseline (Yesterday=Tomorrow)... ", end="", flush=True)
    baseline = BaselineModel()
    baseline_pred = baseline.predict(test_df)
    print("OK")
    
    # SARIMAX
    print("   - SARIMAX (Day-Ahead Walk-Forward)... ", flush=True)
    sarimax = SARIMAXModel(order=config.SARIMAX_ORDER, seasonal_order=config.SARIMAX_SEASONAL_ORDER)
    sarimax_pred = sarimax.walk_forward_predict(
        train_series=renewable['Total'][:-test_hours],
        test_df=test_df[['Total']], 
        steps_per_iteration=24
    )
    print("   OK")
    
    # SVR
    print("   - SVR (with Features)... ", end="", flush=True)
    svr = SVRModel(kernel=config.SVR_KERNEL, C=config.SVR_C, epsilon=config.SVR_EPSILON)
    svr.fit(train_df[feature_cols], train_df['Total'])
    svr_pred = svr.predict(test_df[feature_cols])
    print("OK")
    
    # 5. Evaluate
    print("\n5. Evaluating performance...")
    evaluator = ModelEvaluator(penalty_per_mw=config.PENALTY_PER_MWH)
    
    y_true = test_df['Total'].values
    predictions = {
        'Baseline': baseline_pred,
        'SARIMAX': sarimax_pred,
        'SVR': svr_pred
    }
    
    results = evaluator.compare_models(y_true, predictions)
    
    # 6. Print results
    print("\n" + "="*70)
    print("RESULTS - 7 DAY BACKTEST")
    print("="*70)
    print(results[['MAE', 'RMSE', 'Daily_Cost', 'Savings_vs_Baseline', 'Improvement_Pct']].round(2))
    print("="*70)
    
    # Best model
    best_model = results['MAE'].idxmin()
    best_savings = results.loc[best_model, 'Savings_vs_Baseline']
    best_improvement = results.loc[best_model, 'Improvement_Pct']
    
    print(f"\nBest Model: {best_model}")
    print(f"  Daily Savings: EUR {best_savings:,.0f}")
    print(f"  Improvement: {best_improvement:.1f}%")
    print(f"  Yearly Projection: EUR {best_savings * 365:,.0f}")
    
    # Save results
    output_path = f'{config.RESULTS_DIR}/model_comparison.csv'
    results.to_csv(output_path)
    print(f"\nOK Results saved to: {output_path}")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    
    return results


if __name__ == '__main__':
    results = main()

