import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from utils import (DataProcessor, get_real_predictions, preprocess_test_data,
                   preprocess_train_data)

warnings.filterwarnings("ignore")

class SimpleEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Simple ensemble regressor with 3 tuned models: RandomForest, XGBoost, and Ridge
    """

    def __init__(self,
                 rf_weight=0.4,
                 xgb_weight=0.4,
                 ridge_weight=0.2,
                 tune_hyperparams=True,
                 random_state=42):
        """
        Initialize the simple ensemble regressor.

        Parameters:
        -----------
        rf_weight : float
            Weight for RandomForest predictions
        xgb_weight : float
            Weight for XGBoost predictions
        ridge_weight : float
            Weight for Ridge predictions
        tune_hyperparams : bool
            Whether to perform hyperparameter tuning
        random_state : int
            Random state for reproducibility
        """
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.ridge_weight = ridge_weight
        self.tune_hyperparams = tune_hyperparams
        self.random_state = random_state

        total_weight = rf_weight + xgb_weight + ridge_weight
        self.rf_weight = rf_weight / total_weight
        self.xgb_weight = xgb_weight / total_weight
        self.ridge_weight = ridge_weight / total_weight

        self.models = {}
        self.best_params = {}
        self.trained = False

    def _get_default_params(self):
        """Get default parameters for each model"""
        return {
            'rf': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state,
                'n_jobs': -1
            },
            'xgb': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1
            },
            'ridge': {
                'alpha': 1.0,
                'random_state': self.random_state
            }
        }

    def _get_param_grids(self):
        """Get parameter grids for hyperparameter tuning"""
        return {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 10, 12],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgb': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }

    def _tune_model(self, model, param_grid, X, y, model_name):
        """Tune hyperparameters for a single model"""
        print(f"  Tuning {model_name}...")

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X, y)
        self.best_params[model_name] = grid_search.best_params_

        print(f"    Best {model_name} params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def _create_models(self):
        """Create model instances with default or tuned parameters"""
        default_params = self._get_default_params()

        models = {
            'rf': RandomForestRegressor(**default_params['rf']),
            'xgb': xgb.XGBRegressor(**default_params['xgb']),
            'ridge': Ridge(**default_params['ridge'])
        }

        return models

    def fit(self, X, y):
        """Fit the ensemble model"""
        print("Training Simple Ensemble (RF + XGB + Ridge)...")

        if self.tune_hyperparams:
            print("Performing hyperparameter tuning...")
            param_grids = self._get_param_grids()
            base_models = self._create_models()

            for name, model in base_models.items():
                tuned_model = self._tune_model(
                    model, param_grids[name], X, y, name
                )
                self.models[name] = tuned_model
        else:
            print("Using default parameters...")
            self.models = self._create_models()

            for name, model in self.models.items():
                print(f"  Training {name}...")
                model.fit(X, y)

        self.trained = True
        return self

    def predict(self, X):
        """Make predictions using the weighted ensemble"""
        if not self.trained:
            raise ValueError("Model must be fitted before prediction")

        rf_pred = self.models['rf'].predict(X)
        xgb_pred = self.models['xgb'].predict(X)
        ridge_pred = self.models['ridge'].predict(X)

        ensemble_pred = (
            self.rf_weight * rf_pred +
            self.xgb_weight * xgb_pred +
            self.ridge_weight * ridge_pred
        )

        return ensemble_pred

    def get_model_predictions(self, X):
        """Get individual model predictions for analysis"""
        if not self.trained:
            raise ValueError("Model must be fitted before prediction")

        return {
            'rf': self.models['rf'].predict(X),
            'xgb': self.models['xgb'].predict(X),
            'ridge': self.models['ridge'].predict(X)
        }


class SimpleSalesPredictionPipeline:
    """
    Simplified sales prediction pipeline with 3-model ensemble
    """

    def __init__(self,
                 rf_weight=0.4,
                 xgb_weight=0.4,
                 ridge_weight=0.2,
                 tune_hyperparams=True):
        """
        Initialize the simple prediction pipeline.

        Parameters:
        -----------
        rf_weight : float
            Weight for RandomForest in ensemble
        xgb_weight : float
            Weight for XGBoost in ensemble
        ridge_weight : float
            Weight for Ridge in ensemble
        tune_hyperparams : bool
            Whether to perform hyperparameter tuning
        """
        self.processor = None
        self.model = None
        self.evaluation_metrics = {}
        self.trained = False
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.ridge_weight = ridge_weight
        self.tune_hyperparams = tune_hyperparams

    def fit(self, df):
        """Train the simple ensemble pipeline"""
        print("Starting Simple Ensemble Training...")

        X_train = df.drop(columns=['Item_Outlet_Sales'])
        y_train = df['Item_Outlet_Sales']

        self.processor = DataProcessor(
            target_col="Item_Outlet_Sales",
            outlier_method='iqr',
            outlier_threshold=1.5,
            scale_features=True,
            scale_target=True
        )

        X_train_processed, y_train_processed, fitted_processor = preprocess_train_data(
            X_train, y_train, self.processor
        )
        self.processor = fitted_processor

        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train_processed, y_train_processed, test_size=0.2, random_state=42
        )

        self.model = SimpleEnsembleRegressor(
            rf_weight=self.rf_weight,
            xgb_weight=self.xgb_weight,
            ridge_weight=self.ridge_weight,
            tune_hyperparams=self.tune_hyperparams,
            random_state=42
        )
        self.model.fit(X_train_split, y_train_split)

        self.trained = True

        self.evaluate_models(X_train_split, X_test_split, y_train_split, y_test_split)

        return self

    def evaluate_models(self, X_train, X_test, y_train, y_test):
        """Evaluate the ensemble model and individual models"""
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        test_individual = self.model.get_model_predictions(X_test)

        train_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred)
        }

        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred)
        }

        individual_metrics = {}
        for model_name, test_pred in test_individual.items():
            individual_metrics[model_name] = {
                'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'mae': mean_absolute_error(y_test, test_pred),
                'r2': r2_score(y_test, test_pred)
            }

        self.evaluation_metrics = {
            'ensemble': {'train': train_metrics, 'test': test_metrics},
            'individual': individual_metrics
        }

    def predict(self, df):
        """Make predictions"""
        if not self.trained:
            raise ValueError("Model must be trained first")

        X_test_processed = preprocess_test_data(df, self.processor)

        y_pred = self.model.predict(X_test_processed)

        return y_pred

    def print_evaluation_metrics(self):
        """Print evaluation results"""
        if not self.evaluation_metrics:
            return

        print("\n" + "="*80)
        print("SIMPLE ENSEMBLE MODEL EVALUATION RESULTS")
        print("="*80)

        ensemble_metrics = self.evaluation_metrics['ensemble']
        print("\nENSEMBLE RESULTS:")
        print(f"  Training   - RMSE: {ensemble_metrics['train']['rmse']:.2f}, "
              f"MAE: {ensemble_metrics['train']['mae']:.2f}, "
              f"R²: {ensemble_metrics['train']['r2']:.4f}")
        print(f"  Test       - RMSE: {ensemble_metrics['test']['rmse']:.2f}, "
              f"MAE: {ensemble_metrics['test']['mae']:.2f}, "
              f"R²: {ensemble_metrics['test']['r2']:.4f}")

        print("\nINDIVIDUAL MODEL RESULTS (Test Set):")
        individual_metrics = self.evaluation_metrics['individual']
        for model_name, metrics in individual_metrics.items():
            print(f"  {model_name.upper():8} - RMSE: {metrics['rmse']:.2f}, "
                  f"MAE: {metrics['mae']:.2f}, "
                  f"R²: {metrics['r2']:.4f}")

        print("\nENSEMBLE WEIGHTS:")
        print(f"  RandomForest: {self.model.rf_weight:.3f}")
        print(f"  XGBoost:      {self.model.xgb_weight:.3f}")
        print(f"  Ridge:        {self.model.ridge_weight:.3f}")

        test_r2 = ensemble_metrics['test']['r2']
        if test_r2 > 0.8:
            rating = "Excellent"
        elif test_r2 > 0.7:
            rating = "Good"
        elif test_r2 > 0.6:
            rating = "Fair"
        else:
            rating = "Needs Improvement"

        print(f"\nPerformance Rating: {rating}")

        train_r2 = ensemble_metrics['train']['r2']
        gap = abs(train_r2 - test_r2)
        if gap < 0.05:
            overfitting_status = "No overfitting"
        elif gap < 0.1:
            overfitting_status = "Slight overfitting"
        else:
            overfitting_status = "Significant overfitting"

        print(f"Overfitting Status: {overfitting_status}")

        best_model = max(individual_metrics.keys(),
                        key=lambda x: individual_metrics[x]['r2'])
        print(f"Best Individual Model: {best_model.upper()}")

        print("="*80)

        try:
            with open("scores.md", "a") as f:
                f.write(f"\n## Simple Ensemble Model Results ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
                f.write("### Ensemble Results\n")
                f.write(f"- **Train RMSE:** {ensemble_metrics['train']['rmse']:.2f}\n")
                f.write(f"- **Train MAE:** {ensemble_metrics['train']['mae']:.2f}\n")
                f.write(f"- **Train R²:** {ensemble_metrics['train']['r2']:.4f}\n")
                f.write(f"- **Test RMSE:** {ensemble_metrics['test']['rmse']:.2f}\n")
                f.write(f"- **Test MAE:** {ensemble_metrics['test']['mae']:.2f}\n")
                f.write(f"- **Test R²:** {ensemble_metrics['test']['r2']:.4f}\n")
                f.write("### Individual Models\n")
                for model_name, metrics in individual_metrics.items():
                    f.write(f"- **{model_name.upper()}:** RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}\n")
                f.write("### Configuration\n")
                f.write(f"- **Ensemble Weights:** RF: {self.model.rf_weight:.3f}, XGB: {self.model.xgb_weight:.3f}, Ridge: {self.model.ridge_weight:.3f}\n")
                f.write(f"- **Hyperparameter Tuning:** {'Yes' if self.tune_hyperparams else 'No'}\n")
                f.write(f"- **Performance Rating:** {rating}\n")
                f.write(f"- **Overfitting Status:** {overfitting_status}\n")
                f.write(f"- **Best Individual Model:** {best_model.upper()}\n")
        except Exception as e:
            print(f"[scores.md] Write error: {e}")


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    ddir = Path(__file__).parent
    df = pd.read_csv(ddir / "train_data.csv")

    pipeline = SimpleSalesPredictionPipeline(
        rf_weight=0.5,
        xgb_weight=0.3,
        ridge_weight=0.2,
        tune_hyperparams=True
    )
    pipeline.fit(df)

    pipeline.print_evaluation_metrics()

    test_data = pd.read_csv(ddir / "test_data.csv")
    # test_data = pd.read_csv(ddir / "train_data.csv")
    # y_real = test_data["Item_Outlet_Sales"]
    # test_data = test_data.drop(columns=["Item_Outlet_Sales"], errors="ignore")

    predictions = (
        get_real_predictions(pipeline.predict(test_data), pipeline.processor)
        .clip(0, df["Item_Outlet_Sales"].max())
        .reshape(-1, 1)
        .ravel()
    )

    # get the real predictions
    y_real_pred = get_real_predictions(predictions, pipeline.processor)

    # get the metrics
    # metrics = evaluate_model_performance(y_real, y_real_pred, pipeline.processor)
    # print(f"RMSE: {np.sqrt(mean_squared_error(y_real, y_real_pred)):.2f}")
    # print(f"MAE: {mean_absolute_error(y_real, y_real_pred):.2f}")
    # print(f"R²: {r2_score(y_real, y_real_pred):.4f}")
    # # print(f"MAPE: {mean_absolute_percentage_error(y_real, y_real_pred):.4f}")
    # df =pd.DataFrame({"Original": y_real, "Predicted": y_real_pred})
    # df['ratio'] = df['Original'] / df['Predicted']
    # print(df['ratio'].describe().to_string())
    # df.plot.scatter(x="Original", y="Predicted")
    # # df.plot.hist(x="ratio")
    # plt.show()

    submission = pd.DataFrame({
        'Item_Identifier': test_data['Item_Identifier'],
        'Outlet_Identifier': test_data['Outlet_Identifier'],
        'Item_Outlet_Sales': predictions
    })

    # timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    submission.to_csv(f"{ddir}/outputs/submission.csv", index=False)

    print(f"\nSubmission saved to: {ddir}/outputs/submission.csv")
    print("Prediction statistics:")
    print(f"  Mean: {predictions.mean():.2f}")
    print(f"  Median: {np.median(predictions):.2f}")
    print(f"  Std: {predictions.std():.2f}")
    print(f"  Min: {predictions.min():.2f}")
    print(f"  Max: {predictions.max():.2f}")

    if pipeline.model.best_params:
        print("\nBest Hyperparameters Found:")
        for model_name, params in pipeline.model.best_params.items():
            print(f"  {model_name.upper()}: {params}")

    print("\n" + "="*80)
    print("SIMPLE ENSEMBLE MODEL SUMMARY")
    print("="*80)
    print("Models Used:")
    print("  ✓ RandomForest Regressor")
    print("  ✓ XGBoost Regressor")
    print("  ✓ Ridge Regressor")
    print(f"Hyperparameter Tuning: {'Enabled' if pipeline.tune_hyperparams else 'Disabled'}")
    print(f"Ensemble Weights: RF={pipeline.model.rf_weight:.3f}, XGB={pipeline.model.xgb_weight:.3f}, Ridge={pipeline.model.ridge_weight:.3f}")
    print("="*80)
    print("="*80)
