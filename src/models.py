"""
Forecasting models for renewable energy generation
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BaselineModel:
    """Naive baseline: Tomorrow = Yesterday (24-hour lag)"""
    
    def __init__(self):
        self.name = "Baseline"
    
    def predict(self, df, horizon=24):
        """
        Predict using 24-hour lag
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'lag_24h' column
        horizon : int
            Forecast horizon (not used for baseline)
            
        Returns:
        --------
        np.array
            Predictions
        """
        return df['lag_24h'].values
    
    def fit(self, X, y):
        """Baseline doesn't need training"""
        pass


class SARIMAXModel:
    """SARIMAX model with walk-forward validation"""
    
    def __init__(self, order=(2, 1, 2), seasonal_order=(1, 0, 1, 24)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.name = "SARIMAX"
    
    def walk_forward_predict(self, train_series, test_df, steps_per_iteration=24):
        """
        Walk-forward prediction: Predict 24h, update, repeat
        
        Parameters:
        -----------
        train_series : pd.Series
            Training data (target variable only)
        test_df : pd.DataFrame
            Test dataframe
        steps_per_iteration : int
            Hours to predict at once (typically 24 for day-ahead)
            
        Returns:
        --------
        np.array
            All predictions concatenated
        """
        predictions = []
        history = train_series.copy()
        
        n_iterations = len(test_df) // steps_per_iteration
        
        for i in range(n_iterations):
            start_idx = i * steps_per_iteration
            end_idx = start_idx + steps_per_iteration
            
            # Train model on history
            model = SARIMAX(history, order=self.order, seasonal_order=self.seasonal_order)
            fit = model.fit(disp=False)
            
            # Predict next period
            forecast = fit.forecast(steps=steps_per_iteration)
            predictions.extend(forecast.values)
            
            # Update history with actual values
            actual_values = test_df.iloc[start_idx:end_idx][test_df.columns[0]]
            history = pd.concat([history, actual_values])
        
        return np.array(predictions)


class SVRModel:
    """SVR model with feature engineering"""
    
    def __init__(self, kernel='rbf', C=100, epsilon=0.1):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.name = "SVR"
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def fit(self, X, y):
        """
        Train SVR model
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature matrix
        y : pd.Series or np.array
            Target variable
        """
        # Scale features
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
        
        # Train
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, gamma='scale')
        self.model.fit(X_scaled, y_scaled)
    
    def predict(self, X):
        """
        Generate predictions
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature matrix
            
        Returns:
        --------
        np.array
            Predictions in original scale
        """
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return y_pred


class ModelEvaluator:
    """Evaluate and compare models"""
    
    def __init__(self, penalty_per_mw=50):
        self.penalty_per_mw = penalty_per_mw
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate performance metrics
        
        Returns:
        --------
        dict
            Dictionary with MAE, RMSE, R2, and cost
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Cost calculation
        total_cost = mae * self.penalty_per_mw * len(y_true)
        daily_cost = total_cost / (len(y_true) / 24)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Total_Cost': total_cost,
            'Daily_Cost': daily_cost
        }
    
    def compare_models(self, y_true, predictions_dict):
        """
        Compare multiple models
        
        Parameters:
        -----------
        y_true : np.array
            Actual values
        predictions_dict : dict
            Dictionary of {model_name: predictions}
            
        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        results = {}
        
        for model_name, y_pred in predictions_dict.items():
            metrics = self.calculate_metrics(y_true, y_pred)
            results[model_name] = metrics
        
        df_results = pd.DataFrame(results).T
        
        # Calculate savings vs baseline
        baseline_cost = df_results.loc['Baseline', 'Daily_Cost']
        df_results['Savings_vs_Baseline'] = baseline_cost - df_results['Daily_Cost']
        df_results['Improvement_Pct'] = (df_results['Savings_vs_Baseline'] / baseline_cost * 100)
        
        return df_results


if __name__ == '__main__':
    print("Model classes defined")
    print("  - BaselineModel: Naive forecast (yesterday = tomorrow)")
    print("  - SARIMAXModel: Statistical seasonal model")
    print("  - SVRModel: Machine learning with features")
    print("  - ModelEvaluator: Performance comparison")

