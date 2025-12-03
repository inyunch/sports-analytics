"""
Win rate prediction models using Pythagorean expectation and WAR features.
Implements multiple machine learning algorithms for comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, roc_auc_score, brier_score_loss,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class WinRatePredictor:
    """
    Predicts team win rates using Pythagorean expectation and WAR-based features.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize predictor.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}

    def prepare_data(self,
                    team_features: pd.DataFrame,
                    target_col: str = 'Actual_Win_Rate',
                    test_seasons: Optional[List[int]] = None,
                    feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for modeling with train/test split by season.

        Args:
            team_features: DataFrame with features and target
            target_col: Name of target column (actual win rate)
            test_seasons: List of seasons to use for testing (if None, uses latest season)
            feature_cols: List of feature column names (if None, uses default set)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        df = team_features.copy()

        # Define default features
        if feature_cols is None:
            feature_cols = [
                'Offensive_WAR',
                'Pitching_WAR',
                'Total_WAR',
                'RS',
                'RA',
                'Run_Differential',
                'Pythagorean_Expectation',
                'WAR_Balance',
                'Offensive_Strength',
                'Pitching_Strength'
            ]

        # Filter to available features
        available_features = [col for col in feature_cols if col in df.columns]

        # Check if target exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data. "
                           "Need actual win-loss records to train supervised models.")

        # Remove rows with missing target
        df = df.dropna(subset=[target_col] + available_features)

        # Determine test seasons
        if test_seasons is None:
            # Use latest season for testing
            test_seasons = [df['Season'].max()]

        # Split by season
        train_mask = ~df['Season'].isin(test_seasons)
        test_mask = df['Season'].isin(test_seasons)

        X_train = df.loc[train_mask, available_features]
        X_test = df.loc[test_mask, available_features]
        y_train = df.loc[train_mask, target_col]
        y_test = df.loc[test_mask, target_col]

        print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
        print(f"Training seasons: {sorted(df.loc[train_mask, 'Season'].unique())}")
        print(f"Test seasons: {sorted(df.loc[test_mask, 'Season'].unique())}")
        print(f"Features: {available_features}")

        return X_train, X_test, y_train, y_test

    def train_linear_models(self,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_test: pd.DataFrame,
                          y_test: pd.Series) -> Dict[str, Any]:
        """
        Train and evaluate linear regression models.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            Dictionary of trained models and metrics
        """
        results = {}

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['linear'] = scaler

        # 1. Linear Regression
        print("\n[1/4] Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        y_pred = lr.predict(X_test_scaled)

        results['LinearRegression'] = {
            'model': lr,
            'predictions': y_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'coefficients': dict(zip(X_train.columns, lr.coef_))
        }

        # 2. Ridge Regression
        print("[2/4] Training Ridge Regression...")
        ridge = Ridge(random_state=self.random_state)
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        ridge_cv = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
        ridge_cv.fit(X_train_scaled, y_train)

        y_pred = ridge_cv.predict(X_test_scaled)

        results['Ridge'] = {
            'model': ridge_cv.best_estimator_,
            'predictions': y_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'best_params': ridge_cv.best_params_,
            'coefficients': dict(zip(X_train.columns, ridge_cv.best_estimator_.coef_))
        }

        # 3. Lasso Regression
        print("[3/4] Training Lasso Regression...")
        lasso = Lasso(random_state=self.random_state, max_iter=10000)
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
        lasso_cv = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
        lasso_cv.fit(X_train_scaled, y_train)

        y_pred = lasso_cv.predict(X_test_scaled)

        results['Lasso'] = {
            'model': lasso_cv.best_estimator_,
            'predictions': y_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'best_params': lasso_cv.best_params_,
            'coefficients': dict(zip(X_train.columns, lasso_cv.best_estimator_.coef_))
        }

        # 4. ElasticNet
        print("[4/4] Training ElasticNet...")
        elastic = ElasticNet(random_state=self.random_state, max_iter=10000)
        param_grid = {
            'alpha': [0.01, 0.1, 1.0],
            'l1_ratio': [0.2, 0.5, 0.8]
        }
        elastic_cv = GridSearchCV(elastic, param_grid, cv=5, scoring='neg_mean_squared_error')
        elastic_cv.fit(X_train_scaled, y_train)

        y_pred = elastic_cv.predict(X_test_scaled)

        results['ElasticNet'] = {
            'model': elastic_cv.best_estimator_,
            'predictions': y_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'best_params': elastic_cv.best_params_,
            'coefficients': dict(zip(X_train.columns, elastic_cv.best_estimator_.coef_))
        }

        return results

    def train_tree_models(self,
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_test: pd.DataFrame,
                         y_test: pd.Series) -> Dict[str, Any]:
        """
        Train and evaluate tree-based models.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            Dictionary of trained models and metrics
        """
        results = {}

        # 1. Random Forest
        print("\n[1/2] Training Random Forest...")
        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
        rf_cv = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        rf_cv.fit(X_train, y_train)

        y_pred = rf_cv.predict(X_test)

        results['RandomForest'] = {
            'model': rf_cv.best_estimator_,
            'predictions': y_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'best_params': rf_cv.best_params_,
            'feature_importance': dict(zip(X_train.columns, rf_cv.best_estimator_.feature_importances_))
        }

        # 2. Gradient Boosting
        print("[2/2] Training Gradient Boosting...")
        gb = GradientBoostingRegressor(random_state=self.random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
        gb_cv = GridSearchCV(gb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gb_cv.fit(X_train, y_train)

        y_pred = gb_cv.predict(X_test)

        results['GradientBoosting'] = {
            'model': gb_cv.best_estimator_,
            'predictions': y_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'best_params': gb_cv.best_params_,
            'feature_importance': dict(zip(X_train.columns, gb_cv.best_estimator_.feature_importances_))
        }

        return results

    def compare_models(self,
                      team_features: pd.DataFrame,
                      target_col: str = 'Actual_Win_Rate',
                      test_seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Train and compare all models.

        Args:
            team_features: DataFrame with features and target
            target_col: Target column name
            test_seasons: Seasons to use for testing

        Returns:
            DataFrame with model comparison results
        """
        print("=" * 60)
        print("Model Training and Comparison")
        print("=" * 60)

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            team_features, target_col, test_seasons
        )

        # Train linear models
        print("\n--- Training Linear Models ---")
        linear_results = self.train_linear_models(X_train, y_train, X_test, y_test)
        self.models.update(linear_results)

        # Train tree models
        print("\n--- Training Tree-Based Models ---")
        tree_results = self.train_tree_models(X_train, y_train, X_test, y_test)
        self.models.update(tree_results)

        # Compile comparison results
        comparison = []
        for model_name, results in self.models.items():
            comparison.append({
                'Model': model_name,
                'RMSE': results['rmse'],
                'MAE': results['mae'],
                'R²': results['r2']
            })

        comparison_df = pd.DataFrame(comparison).sort_values('RMSE')

        # Store test data for visualization
        self.X_test = X_test
        self.y_test = y_test

        print("\n" + "=" * 60)
        print("Model Comparison Results:")
        print("=" * 60)
        print(comparison_df.to_string(index=False))

        return comparison_df

    def plot_model_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize model predictions vs actual values.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        n_models = len(self.models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, results) in enumerate(self.models.items()):
            ax = axes[idx]

            y_pred = results['predictions']

            # Scatter plot
            ax.scatter(self.y_test, y_pred, alpha=0.6, s=100)

            # Perfect prediction line
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

            # Metrics
            rmse = results['rmse']
            r2 = results['r2']

            ax.text(0.05, 0.95, f"RMSE: {rmse:.4f}\nR²: {r2:.4f}",
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel('Actual Win Rate', fontsize=11)
            ax.set_ylabel('Predicted Win Rate', fontsize=11)
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Model Predictions vs Actual Win Rate', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_feature_importance(self, model_name: str = 'GradientBoosting',
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance for tree-based models.

        Args:
            model_name: Name of model to visualize
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        results = self.models[model_name]

        if 'feature_importance' in results:
            importance = results['feature_importance']
        elif 'coefficients' in results:
            importance = {k: abs(v) for k, v in results['coefficients'].items()}
        else:
            raise ValueError(f"No importance/coefficients found for {model_name}")

        # Sort by importance
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        fig, ax = plt.subplots(figsize=(10, 6))

        features = list(sorted_importance.keys())
        importances = list(sorted_importance.values())

        ax.barh(features, importances, color='steelblue')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def save_models(self, output_dir: str = "models") -> None:
        """
        Save trained models to disk.

        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for model_name, results in self.models.items():
            model_file = output_path / f"{model_name.lower()}_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(results['model'], f)
            print(f"Saved {model_name} to {model_file}")

        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_file = output_path / f"{scaler_name}_scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)

        print(f"\nAll models saved to {output_path}")


if __name__ == "__main__":
    print("Win Rate Prediction Models - Ready for use")
