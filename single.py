import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


class DataProcessor(BaseEstimator, TransformerMixin):
    """
    Enhanced data preprocessing pipeline with improved structure,
    StandardScaler integration, and outlier handling.
    """

    def __init__(
        self,
        target_col="Item_Outlet_Sales",
        outlier_method="iqr",
        outlier_threshold=1.5,
        scale_features=True,
        scale_target=True,
        log_transform_target=False,
    ):
        """
        Initialize the data processor.

        Parameters:
        -----------
        target_col : str
            Name of the target column
        outlier_method : str, {'iqr', 'zscore', 'isolation'}
            Method for outlier detection
        outlier_threshold : float
            Threshold for outlier detection (1.5 for IQR, 3 for Z-score)
        scale_features : bool
            Whether to apply StandardScaler to numerical features
        scale_target : bool
            Whether to apply MinMaxScaler(0,1) to target variable
        log_transform_target : bool
            Whether to apply log transformation to target variable before scaling
        """
        self.target_col = target_col
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.scale_features = scale_features
        self.scale_target = scale_target
        self.log_transform_target = log_transform_target

        # Initialize state variables
        self._fitted = False
        self.encoders = {}
        self.feature_scaler = None
        self.target_scaler = None
        self.outlier_bounds = {}
        self.outlet_target_encoding = {}
        self.global_sales_mean = None
        self.feature_columns = []
        self.mrp_quantiles = None
        self.item_weight_stats = {}
        self.item_mrp_stats = {}
        self.target_log_offset = 0  # For log transformation

        # Configuration dictionaries
        self._setup_configs()

    def _setup_configs(self):
        """Setup configuration dictionaries for data cleaning and feature types."""
        self.ordinal_cols = [
            "Item_Fat_Content",
            "Outlet_Location_Type",
            "Outlet_Size",
            # "Item_Category",
        ]
        self.nominal_cols = [
            "Item_Type",
            "Outlet_Type",
            "MRP_Category",
            "Item_Category",
        ]
        self.id_cols = ["Item_Identifier", "Outlet_Identifier"]
        self.numerical_cols = [
            "Item_Weight",
            "Item_Visibility",
            "Item_MRP",
            "Outlet_Establishment_Year",
        ]
        self.all_cols = (
            self.ordinal_cols + self.nominal_cols + self.id_cols + self.numerical_cols
        )

        self.DATA_CLEANING_DICT = {
            "Item_Fat_Content": {
                "standardization_mapping": {
                    "LF": "Low Fat",
                    "low fat": "Low Fat",
                    "reg": "Regular",
                },
                "feature_type": "ordinal",
                "ordinal_mapping": {"Low Fat": 0, "Regular": 1},
            },
            "Item_Type": {
                "standardization_mapping": {
                    "Fruits & Vegetables": "Fruits and Vegetables",
                    "Health & Hygiene": "Health and Hygiene",
                },
                "feature_type": "nominal",
            },
            "Outlet_Location_Type": {
                "standardization_mapping": {
                    "Tier1": "Tier 1",
                    "Tier2": "Tier 2",
                    "Tier3": "Tier 3",
                    "tier 1": "Tier 1",
                    "tier 2": "Tier 2",
                    "tier 3": "Tier 3",
                },
                "feature_type": "ordinal",
                "ordinal_mapping": {"Tier 3": 0, "Tier 2": 1, "Tier 1": 2},
            },
            "Outlet_Size": {
                "standardization_mapping": {
                    "high": "High",
                    "medium": "Medium",
                    "small": "Small",
                    "SMALL": "Small",
                    "MEDIUM": "Medium",
                    "HIGH": "High",
                },
                "feature_type": "ordinal",
                "ordinal_mapping": {"Small": 0, "Medium": 1, "High": 2},
                "missing_value_handling": "impute",
            },
            "Outlet_Type": {
                "standardization_mapping": {
                    "Grocery": "Grocery Store",
                    "Supermarket Type 1": "Supermarket Type1",
                    "Supermarket Type 2": "Supermarket Type2",
                    "Supermarket Type 3": "Supermarket Type3",
                },
                "feature_type": "nominal",
            },
        }

    def fit(self, X, y=None):
        """
        Fit the preprocessing pipeline.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series, optional
            Target variable

        Returns:
        --------
        self : object
            Returns the instance itself
        """
        # Validate input
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        df = X.copy()

        # Step 1: Data cleaning and standardization
        df = self._standardize_values(df)

        # Step 2: Missing value imputation (learn imputation statistics)
        df = self._learn_missing_value_handling(df)

        # Step 3: Outlier detection (learn outlier bounds)
        if y is not None and self.target_col in df.columns:
            df = self._learn_outlier_bounds(df)

        # Step 4: Feature engineering
        df = self._feature_engineering(df, fit=True)

        # Step 5: Encoding (learn encodings) - this changes the feature structure
        df = self._learn_encodings(df)
        # Apply encodings to get the final feature structure
        df = self._apply_encodings(df)

        # Step 6: Get numerical features for scaling (after all transformations)
        numerical_features = self._get_numerical_features(df)

        # Step 7: Learn scaling parameters on final feature set
        if self.scale_features and numerical_features:
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(df[numerical_features])

        if self.scale_target and y is not None:
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))

            # Prepare target for scaling
            y_for_scaling = y.copy()

            # Apply log transformation if requested
            if self.log_transform_target:
                # Handle zero/negative values by adding small offset if needed
                min_val = y_for_scaling.min()
                if min_val <= 0:
                    self.target_log_offset = abs(min_val) + 1
                    y_for_scaling = y_for_scaling + self.target_log_offset

                # Apply log transformation (using log1p for numerical stability)
                y_for_scaling = np.log1p(y_for_scaling)

            # Fit the scaler on (potentially log-transformed) target
            self.target_scaler.fit(y_for_scaling.values.reshape(-1, 1))

        # Store final feature columns
        self.feature_columns = self._get_final_feature_columns(df)
        self._fitted = True

        return self

    def transform(self, X):
        """
        Transform the input data using fitted preprocessing pipeline.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix to transform

        Returns:
        --------
        X_transformed : pd.DataFrame
            Transformed feature matrix
        """
        if not self._fitted:
            raise ValueError("This instance is not fitted yet. Call 'fit' first.")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        df = X.copy()

        # Apply the same transformation steps as in fit
        df = self._standardize_values(df)
        df = self._apply_missing_value_handling(df)
        df = self._remove_outliers(df)
        df = self._feature_engineering(df, fit=False)
        df = self._apply_encodings(df)

        # Get numerical features and apply scaling (after all transformations)
        numerical_features = self._get_numerical_features(df)

        if (
            self.scale_features
            and self.feature_scaler is not None
            and numerical_features
        ):
            # Only scale features that exist in both the scaler and current dataframe
            scaler_features = (
                self.feature_scaler.feature_names_in_
                if hasattr(self.feature_scaler, "feature_names_in_")
                else numerical_features
            )
            available_features = [col for col in scaler_features if col in df.columns]

            if available_features:
                df[available_features] = self.feature_scaler.transform(
                    df[available_features]
                )

        # Return only the final feature columns that were learned during fit
        final_features = [col for col in self.feature_columns if col in df.columns]
        return df[final_features]

    def fit_transform(self, X, y=None):
        """Fit the preprocessing pipeline and transform the data."""
        return self.fit(X, y).transform(X)

    def transform_target(self, y):
        """
        Transform the target variable using fitted scaler.

        Parameters:
        -----------
        y : pd.Series or np.array
            Target variable to transform

        Returns:
        --------
        y_transformed : np.array
            Transformed target variable
        """
        if not self._fitted:
            raise ValueError("This instance is not fitted yet. Call 'fit' first.")

        if self.scale_target and self.target_scaler is not None:
            y_transformed = y.copy()

            # Apply log transformation if it was used during fit
            if self.log_transform_target:
                # Apply the same offset if it was used during fit
                if self.target_log_offset > 0:
                    y_transformed = y_transformed + self.target_log_offset

                # Apply log transformation
                y_transformed = np.log1p(y_transformed)

            # Apply scaling
            y_values = (
                y_transformed.values
                if hasattr(y_transformed, "values")
                else y_transformed
            )
            return self.target_scaler.transform(y_values.reshape(-1, 1)).ravel()

        return y.values if hasattr(y, "values") else y

    def inverse_transform_target(self, y):
        """
        Inverse transform the target variable using fitted scaler.

        Parameters:
        -----------
        y : np.array
            Transformed target variable

        Returns:
        --------
        y_original : np.array
            Original scale target variable
        """
        if not self._fitted:
            raise ValueError("This instance is not fitted yet. Call 'fit' first.")

        if self.scale_target and self.target_scaler is not None:
            # First, inverse the scaling
            y_unscaled = self.target_scaler.inverse_transform(y.reshape(-1, 1)).ravel()

            # Then, inverse the log transformation if it was applied
            if self.log_transform_target:
                # Inverse log transformation (expm1 is inverse of log1p)
                y_unscaled = np.expm1(y_unscaled)

                # Remove the offset if it was applied
                if self.target_log_offset > 0:
                    y_unscaled = y_unscaled - self.target_log_offset

            return y_unscaled

        return y

    def _standardize_values(self, df):
        """Standardize categorical values according to configuration."""
        df = df.copy()
        for col, config in self.DATA_CLEANING_DICT.items():
            if col in df.columns and "standardization_mapping" in config:
                df[col] = df[col].replace(config["standardization_mapping"])
        return df

    def _learn_missing_value_handling(self, df):
        """Learn statistics for missing value imputation."""
        df = df.copy()

        # Learn Item_Weight imputation statistics
        if "Item_Weight" in df.columns:
            self.item_weight_stats["by_identifier"] = (
                df.groupby("Item_Identifier")["Item_Weight"].mean().to_dict()
            )
            self.item_weight_stats["by_type"] = (
                df.groupby("Item_Type")["Item_Weight"].mean().to_dict()
            )
            self.item_weight_stats["global_mean"] = df["Item_Weight"].mean()

            # Apply imputation during fit
            df = self._apply_item_weight_imputation(df)

        # Learn Item_MRP imputation statistics
        if "Item_MRP" in df.columns:
            self.item_mrp_stats["by_identifier"] = (
                df.groupby("Item_Identifier")["Item_MRP"].mean().to_dict()
            )
            self.item_mrp_stats["global_mean"] = df["Item_MRP"].mean()

            # Apply imputation during fit
            df = self._apply_item_mrp_imputation(df)

        # Learn Outlet_Size imputation
        if "Outlet_Size" in df.columns:
            df = self._apply_outlet_size_imputation(df)

        # Handle Item_Visibility = 0
        if "Item_Visibility" in df.columns:
            df = self._handle_item_visibility(df)

        return df

    def _apply_missing_value_handling(self, df):
        """Apply learned missing value imputation."""
        df = df.copy()

        if "Item_Weight" in df.columns:
            df = self._apply_item_weight_imputation(df)

        if "Item_MRP" in df.columns:
            df = self._apply_item_mrp_imputation(df)

        if "Outlet_Size" in df.columns:
            df = self._apply_outlet_size_imputation(df)

        if "Item_Visibility" in df.columns:
            df = self._handle_item_visibility(df)

        return df

    def _apply_item_weight_imputation(self, df):
        """Apply Item_Weight imputation using learned statistics."""
        if "Item_Weight" not in df.columns:
            return df

        df = df.copy()
        # Fill by Item_Identifier
        df["Item_Weight"] = df.apply(
            lambda row: self.item_weight_stats["by_identifier"].get(
                row["Item_Identifier"], row["Item_Weight"]
            )
            if pd.isna(row["Item_Weight"])
            else row["Item_Weight"],
            axis=1,
        )

        # Fill by Item_Type
        df["Item_Weight"] = df.apply(
            lambda row: self.item_weight_stats["by_type"].get(
                row["Item_Type"], row["Item_Weight"]
            )
            if pd.isna(row["Item_Weight"])
            else row["Item_Weight"],
            axis=1,
        )

        # Fill remaining with global mean
        df["Item_Weight"] = df["Item_Weight"].fillna(
            self.item_weight_stats["global_mean"]
        )

        return df

    def _apply_item_mrp_imputation(self, df):
        """Apply Item_MRP imputation using learned statistics."""
        if "Item_MRP" not in df.columns:
            return df

        df = df.copy()
        # Fill by Item_Identifier
        df["Item_MRP"] = df.apply(
            lambda row: self.item_mrp_stats["by_identifier"].get(
                row["Item_Identifier"], row["Item_MRP"]
            )
            if pd.isna(row["Item_MRP"])
            else row["Item_MRP"],
            axis=1,
        )

        # Fill remaining with global mean
        df["Item_MRP"] = df["Item_MRP"].fillna(self.item_mrp_stats["global_mean"])

        return df

    def _apply_outlet_size_imputation(self, df):
        """Apply Outlet_Size imputation."""
        if "Outlet_Size" not in df.columns:
            return df

        df = df.copy()
        # Fill by Outlet_Identifier
        df["Outlet_Size"] = df.groupby("Outlet_Identifier")["Outlet_Size"].transform(
            lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
        )

        # Fill by Outlet_Type + Location_Type
        if "Outlet_Type" in df.columns and "Outlet_Location_Type" in df.columns:
            df["Outlet_Size"] = df.groupby(["Outlet_Type", "Outlet_Location_Type"])[
                "Outlet_Size"
            ].transform(
                lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
            )

        # Fill remaining with global mode
        mode_val = df["Outlet_Size"].mode()
        if not mode_val.empty:
            df["Outlet_Size"] = df["Outlet_Size"].fillna(mode_val.iloc[0])

        return df

    def _handle_item_visibility(self, df):
        """Handle Item_Visibility = 0 by replacing with median by Item_Type."""
        if "Item_Visibility" not in df.columns:
            return df

        df = df.copy()
        zero_mask = df["Item_Visibility"] == 0

        # Calculate medians for non-zero visibility by Item_Type
        medians = (
            df[df["Item_Visibility"] > 0]
            .groupby("Item_Type")["Item_Visibility"]
            .median()
        )

        # Replace zeros with medians
        for item_type in medians.index:
            mask = (df["Item_Type"] == item_type) & zero_mask
            df.loc[mask, "Item_Visibility"] = medians[item_type]

        return df

    def _learn_outlier_bounds(self, df):
        """Learn outlier bounds for numerical features."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if col == self.target_col:
                continue

            if self.outlier_method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
                self.outlier_bounds[col] = (lower_bound, upper_bound)

            elif self.outlier_method == "zscore":
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - self.outlier_threshold * std
                upper_bound = mean + self.outlier_threshold * std
                self.outlier_bounds[col] = (lower_bound, upper_bound)

        return df

    def _remove_outliers(self, df):
        """Remove or cap outliers based on learned bounds."""
        df = df.copy()

        for col, (lower_bound, upper_bound) in self.outlier_bounds.items():
            if col in df.columns:
                # Cap outliers instead of removing them to preserve data
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return df

    def _feature_engineering(self, df, fit=False):
        """Create engineered features."""
        df = df.copy()

        # Basic engineered features
        df["Outlet_Age"] = 2024 - df["Outlet_Establishment_Year"]
        # df["Item_Category"] = df["Item_Identifier"].str[:2]

        # MRP categorization
        if fit and "Item_MRP" in df.columns:
            self.mrp_quantiles = df["Item_MRP"].quantile([0.2, 0.4, 0.6, 0.8]).values

        if "Item_MRP" in df.columns:
            df["MRP_Category"] = pd.cut(
                df["Item_MRP"],
                bins=[-np.inf] + list(self.mrp_quantiles) + [np.inf],
                labels=["Low", "Low-Med", "Medium", "Med-High", "High"],
            )
            df["MRP_Category"] = df["MRP_Category"].astype("category")

        # Ratio features
        if "Item_MRP" in df.columns and "Item_Weight" in df.columns:
            df["MRP_per_Weight"] = df["Item_MRP"] / df["Item_Weight"].replace(0, np.nan)

        if "Item_Visibility" in df.columns and "Item_MRP" in df.columns:
            df["Visibility_MRP_Ratio"] = df["Item_Visibility"] / df["Item_MRP"].replace(
                0, np.nan
            )

        if "Item_MRP" in df.columns and "Outlet_Age" in df.columns:
            df["MRP_x_Outlet_Age"] = df["Item_MRP"] * df["Outlet_Age"]

        # Target encoding for Outlet_Identifier (only during fit)
        if fit and self.target_col in df.columns:
            global_mean = df[self.target_col].mean()
            stats = df.groupby("Outlet_Identifier")[self.target_col].agg(
                ["mean", "count"]
            )
            alpha = 10
            smoothed = (stats["mean"] * stats["count"] + global_mean * alpha) / (
                stats["count"] + alpha
            )
            self.global_sales_mean = global_mean
            self.outlet_target_encoding = smoothed.to_dict()

        # Apply target encoding
        if "Outlet_Identifier" in df.columns:
            df["Outlet_Target_Encoded"] = (
                df["Outlet_Identifier"]
                .map(self.outlet_target_encoding)
                .fillna(self.global_sales_mean)
            )

        return df

    def _learn_encodings(self, df):
        """Learn encodings for categorical variables."""
        # Ordinal encoding
        # for col in self.ordinal_cols:
        #     if col in df.columns:
        #         config = self.DATA_CLEANING_DICT.get(col)
        #         if config and "ordinal_mapping" in config:
        #             df[col + "_encoded"] = df[col].map(config["ordinal_mapping"])
        # replace the values with the encoded values
        # df[col] = df[col].map(config["ordinal_mapping"])

        # One-hot encoding
        for col in self.nominal_cols:
            if col in df.columns:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                col_data = df[col].astype(str).values.reshape(-1, 1)
                encoder.fit(col_data)
                self.encoders[col] = encoder

        return df

    def _apply_encodings(self, df):
        """Apply learned encodings to categorical variables."""
        # Ordinal encoding
        for col in self.ordinal_cols:
            if col in df.columns:
                config = self.DATA_CLEANING_DICT.get(col)
                if config and "ordinal_mapping" in config:
                    df[col] = df[col].map(config["ordinal_mapping"])

        # One-hot encoding
        for col in self.nominal_cols:
            if col in df.columns and col in self.encoders:
                encoder = self.encoders[col]
                transformed = encoder.transform(
                    df[col].astype(str).values.reshape(-1, 1)
                )
                col_names = encoder.get_feature_names_out([col])
                ohe_df = pd.DataFrame(transformed, columns=col_names, index=df.index)
                df = pd.concat([df.drop(columns=[col]), ohe_df], axis=1)
                df.drop(columns=[col], inplace=True, errors="ignore")

        return df

    def _get_numerical_features(self, df):
        """Get list of numerical feature columns (after all transformations)."""
        # Exclude ID columns and target column
        exclude_cols = self.id_cols + [self.target_col]
        # Also exclude original categorical columns (before encoding)
        exclude_cols.extend(self.ordinal_cols + self.nominal_cols)
        exclude_cols = [col for col in exclude_cols if col in df.columns]

        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features = [
            col for col in numerical_features if col not in exclude_cols
        ]

        return numerical_features

    def _get_final_feature_columns(self, df):
        """Get the final list of feature columns after all transformations."""
        # The df passed here should already have all transformations applied
        # Exclude ID columns and target column
        exclude_cols = self.id_cols + [self.target_col]
        exclude_cols = [col for col in exclude_cols if col in df.columns]

        feature_columns = [col for col in df.columns if col not in exclude_cols]
        return feature_columns

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if not self._fitted:
            raise ValueError("This instance is not fitted yet. Call 'fit' first.")
        return self.feature_columns

    def get_outlier_summary(self):
        """Get summary of outlier bounds learned during fit."""
        if not self._fitted:
            raise ValueError("This instance is not fitted yet. Call 'fit' first.")
        return self.outlier_bounds.copy()


# Usage example and helper functions
def create_preprocessing_pipeline(
    target_col="Item_Outlet_Sales",
    outlier_method="iqr",
    scale_features=True,
    scale_target=False,
    log_transform_target=False,
):
    """
    Create a complete preprocessing pipeline.

    Parameters:
    -----------
    target_col : str
        Name of the target column
    outlier_method : str
        Method for outlier detection
    scale_features : bool
        Whether to scale numerical features
    scale_target : bool
        Whether to scale target variable
    log_transform_target : bool
        Whether to apply log transformation to target variable

    Returns:
    --------
    processor : DataProcessor
        Configured preprocessing pipeline
    """
    return DataProcessor(
        target_col=target_col,
        outlier_method=outlier_method,
        scale_features=scale_features,
        scale_target=scale_target,
        log_transform_target=log_transform_target,
    )


def preprocess_train_data(X_train, y_train, processor=None):
    """
    Preprocess training data.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    processor : DataProcessor, optional
        Preprocessing pipeline (if None, creates a new one)

    Returns:
    --------
    X_train_processed : pd.DataFrame
        Processed training features
    y_train_processed : np.array
        Processed training target
    processor : DataProcessor
        Fitted preprocessing pipeline
    """
    if processor is None:
        processor = create_preprocessing_pipeline()

    # Combine X and y for fitting (needed for target encoding)
    train_data = X_train.copy()
    train_data[processor.target_col] = y_train

    # Fit and transform features
    processor.fit(train_data, y_train)
    X_train_processed = processor.transform(train_data)

    # Transform target
    y_train_processed = processor.transform_target(y_train)

    return X_train_processed, y_train_processed, processor


def preprocess_test_data(X_test, processor, y_test=None):
    """
    Preprocess test data using fitted processor.

    Parameters:
    -----------
    X_test : pd.DataFrame
        Test features
    processor : DataProcessor
        Fitted preprocessing pipeline
    y_test : pd.Series, optional
        Test target (for evaluation)

    Returns:
    --------
    X_test_processed : pd.DataFrame
        Processed test features
    y_test_processed : np.array, optional
        Processed test target (if y_test provided)
    """
    X_test_processed = processor.transform(X_test)

    if y_test is not None:
        y_test_processed = processor.transform_target(y_test)
        return X_test_processed, y_test_processed

    return X_test_processed


def get_real_predictions(predictions, processor):
    """
    Convert scaled/transformed predictions back to real target values.

    Parameters:
    -----------
    predictions : np.array or pd.Series
        Model predictions (potentially scaled)
    processor : DataProcessor
        Fitted preprocessing pipeline

    Returns:
    --------
    real_predictions : np.array
        Predictions in original target scale
    """
    if not processor._fitted:
        raise ValueError(
            "Processor must be fitted before inverse transforming predictions."
        )

    return processor.inverse_transform_target(predictions)


def evaluate_model_performance(y_true, y_pred, processor):
    """
    Evaluate model performance with real-scale metrics.

    Parameters:
    -----------
    y_true : np.array or pd.Series
        True target values (can be scaled or unscaled)
    y_pred : np.array
        Model predictions (potentially scaled)
    processor : DataProcessor
        Fitted preprocessing pipeline

    Returns:
    --------
    metrics : dict
        Dictionary containing various performance metrics in real scale
    """
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)

    # Convert to real scale if needed
    if processor.scale_target:
        y_true_real = processor.inverse_transform_target(y_true)
        y_pred_real = processor.inverse_transform_target(y_pred)
    else:
        y_true_real = y_true.values if hasattr(y_true, "values") else y_true
        y_pred_real = y_pred

    # Calculate metrics
    mse = mean_squared_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_real, y_pred_real)
    r2 = r2_score(y_true_real, y_pred_real)

    # Additional regression metrics
    mape = np.mean(np.abs((y_true_real - y_pred_real) / y_true_real)) * 100

    metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

    return metrics, y_true_real, y_pred_real


def complete_ml_workflow_example():
    """
    Complete example workflow showing how to use the enhanced processor
    with real predictions.
    """

    # This is a demonstration - replace with your actual data loading
    print("=== Complete ML Workflow Example ===")
    print()

    # Step 1: Load and split your data
    print("Step 1: Load and prepare data")
    print("# X_train, X_test, y_train, y_test = load_and_split_your_data()")
    print()

    # Step 2: Create and configure processor
    print("Step 2: Create preprocessing pipeline")
    processor_code = """
    processor = DataProcessor(
        target_col="Item_Outlet_Sales",
        outlier_method='iqr',
        outlier_threshold=1.5,
        scale_features=True,
        scale_target=True,           # Scale target to 0-1 range
        log_transform_target=False   # Optional: log transform before scaling
    )
    """
    print(processor_code)

    # Step 3: Preprocess training data
    print("Step 3: Preprocess training data")
    preprocess_code = """
    # Fit the processor and transform training data
    X_train_processed, y_train_processed, fitted_processor = preprocess_train_data(
        X_train, y_train, processor
    )

    print(f"Original target range: {y_train.min():.2f} to {y_train.max():.2f}")
    if fitted_processor.scale_target:
        print(f"Scaled target range: {y_train_processed.min():.2f} to {y_train_processed.max():.2f}")
    """
    print(preprocess_code)

    # Step 4: Train model
    print("Step 4: Train your model")
    model_code = """
    from sklearn.ensemble import RandomForestRegressor
    # or any other regression model

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_processed, y_train_processed)
    """
    print(model_code)

    # Step 5: Make predictions
    print("Step 5: Make predictions and convert to real values")
    prediction_code = """
    # Preprocess test data
    X_test_processed = preprocess_test_data(X_test, fitted_processor)

    # Make predictions (these will be scaled if target was scaled)
    scaled_predictions = model.predict(X_test_processed)

    # Convert predictions back to real target values
    real_predictions = get_real_predictions(scaled_predictions, fitted_processor)

    print(f"Scaled predictions range: {scaled_predictions.min():.2f} to {scaled_predictions.max():.2f}")
    print(f"Real predictions range: {real_predictions.min():.2f} to {real_predictions.max():.2f}")
    """
    print(prediction_code)

    # Step 6: Evaluate performance
    print("Step 6: Evaluate model performance")
    evaluation_code = """
    # Get real-scale evaluation (if you have test target values)
    if y_test is not None:
        metrics, y_true_real, y_pred_real = evaluate_model_performance(
            y_test, scaled_predictions, fitted_processor
        )

        print("Model Performance (Real Scale):")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    """
    print(evaluation_code)

    return """
# Summary of key functions for getting real predictions:

1. get_real_predictions(predictions, processor)
   - Converts scaled predictions back to original scale

2. evaluate_model_performance(y_true, y_pred, processor)
   - Evaluates model with real-scale metrics
   - Returns metrics dict and real-scale values

3. processor.inverse_transform_target(predictions)
   - Direct method to inverse transform target values

4. processor.transform_target(y)
   - Transform target to scaled values (for training)
"""


# Utility function for batch prediction with real values
def predict_and_get_real_values(model, X_new, processor):
    """
    Make predictions on new data and return real-scale values.

    Parameters:
    -----------
    model : sklearn estimator
        Trained regression model
    X_new : pd.DataFrame
        New data to make predictions on
    processor : DataProcessor
        Fitted preprocessing pipeline

    Returns:
    --------
    real_predictions : np.array
        Predictions in original target scale
    processed_features : pd.DataFrame
        Processed features (for debugging)
    """
    # Preprocess features
    X_processed = processor.transform(X_new)

    # Make predictions
    scaled_predictions = model.predict(X_processed)

    # Convert to real scale
    real_predictions = get_real_predictions(scaled_predictions, processor)

    return real_predictions, X_processed


def create_prediction_dataframe(X_original, real_predictions, id_column=None):
    """
    Create a clean DataFrame with original identifiers and real predictions.

    Parameters:
    -----------
    X_original : pd.DataFrame
        Original input data (before preprocessing)
    real_predictions : np.array
        Predictions in real scale
    id_column : str, optional
        Column name to use as identifier

    Returns:
    --------
    prediction_df : pd.DataFrame
        DataFrame with identifiers and predictions
    """
    if id_column and id_column in X_original.columns:
        prediction_df = pd.DataFrame(
            {
                id_column: X_original[id_column].values,
                "Predicted_Sales": real_predictions,
            }
        )
    else:
        prediction_df = pd.DataFrame(
            {"Index": X_original.index, "Predicted_Sales": real_predictions}
        )

    return prediction_df


class SimpleEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Simple ensemble regressor with 3 tuned models: RandomForest, XGBoost, and Ridge
    """

    def __init__(
        self,
        rf_weight=0.4,
        xgb_weight=0.4,
        ridge_weight=0.2,
        tune_hyperparams=True,
        random_state=42,
    ):
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
            "rf": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": self.random_state,
                "n_jobs": -1,
            },
            "xgb": {
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "n_jobs": -1,
            },
            "ridge": {"alpha": 1.0, "random_state": self.random_state},
        }

    def _get_param_grids(self):
        """Get parameter grids for hyperparameter tuning"""
        return {
            "rf": {
                "n_estimators": [100, 200, 300],
                "max_depth": [8, 10, 12],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "xgb": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.15],
                "max_depth": [4, 6, 8],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            },
            "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
        }

    def _tune_model(self, model, param_grid, X, y, model_name):
        """Tune hyperparameters for a single model"""
        print(f"  Tuning {model_name}...")

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X, y)
        self.best_params[model_name] = grid_search.best_params_

        print(f"    Best {model_name} params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def _create_models(self):
        """Create model instances with default or tuned parameters"""
        default_params = self._get_default_params()

        models = {
            "rf": RandomForestRegressor(**default_params["rf"]),
            "xgb": xgb.XGBRegressor(**default_params["xgb"]),
            "ridge": Ridge(**default_params["ridge"]),
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
                tuned_model = self._tune_model(model, param_grids[name], X, y, name)
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

        rf_pred = self.models["rf"].predict(X)
        xgb_pred = self.models["xgb"].predict(X)
        ridge_pred = self.models["ridge"].predict(X)

        ensemble_pred = (
            self.rf_weight * rf_pred
            + self.xgb_weight * xgb_pred
            + self.ridge_weight * ridge_pred
        )

        return ensemble_pred

    def get_model_predictions(self, X):
        """Get individual model predictions for analysis"""
        if not self.trained:
            raise ValueError("Model must be fitted before prediction")

        return {
            "rf": self.models["rf"].predict(X),
            "xgb": self.models["xgb"].predict(X),
            "ridge": self.models["ridge"].predict(X),
        }


class SimpleSalesPredictionPipeline:
    """
    Simplified sales prediction pipeline with 3-model ensemble
    """

    def __init__(
        self, rf_weight=0.4, xgb_weight=0.4, ridge_weight=0.2, tune_hyperparams=True
    ):
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

        X_train = df.drop(columns=["Item_Outlet_Sales"])
        y_train = df["Item_Outlet_Sales"]

        self.processor = DataProcessor(
            target_col="Item_Outlet_Sales",
            outlier_method="iqr",
            outlier_threshold=1.5,
            scale_features=True,
            scale_target=True,
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
            random_state=42,
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
            "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "mae": mean_absolute_error(y_train, y_train_pred),
            "r2": r2_score(y_train, y_train_pred),
        }

        test_metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "mae": mean_absolute_error(y_test, y_test_pred),
            "r2": r2_score(y_test, y_test_pred),
        }

        individual_metrics = {}
        for model_name, test_pred in test_individual.items():
            individual_metrics[model_name] = {
                "rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
                "mae": mean_absolute_error(y_test, test_pred),
                "r2": r2_score(y_test, test_pred),
            }

        self.evaluation_metrics = {
            "ensemble": {"train": train_metrics, "test": test_metrics},
            "individual": individual_metrics,
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

        print("\n" + "=" * 80)
        print("SIMPLE ENSEMBLE MODEL EVALUATION RESULTS")
        print("=" * 80)

        ensemble_metrics = self.evaluation_metrics["ensemble"]
        print("\nENSEMBLE RESULTS:")
        print(
            f"  Training   - RMSE: {ensemble_metrics['train']['rmse']:.2f}, "
            f"MAE: {ensemble_metrics['train']['mae']:.2f}, "
            f"R²: {ensemble_metrics['train']['r2']:.4f}"
        )
        print(
            f"  Test       - RMSE: {ensemble_metrics['test']['rmse']:.2f}, "
            f"MAE: {ensemble_metrics['test']['mae']:.2f}, "
            f"R²: {ensemble_metrics['test']['r2']:.4f}"
        )

        print("\nINDIVIDUAL MODEL RESULTS (Test Set):")
        individual_metrics = self.evaluation_metrics["individual"]
        for model_name, metrics in individual_metrics.items():
            print(
                f"  {model_name.upper():8} - RMSE: {metrics['rmse']:.2f}, "
                f"MAE: {metrics['mae']:.2f}, "
                f"R²: {metrics['r2']:.4f}"
            )

        print("\nENSEMBLE WEIGHTS:")
        print(f"  RandomForest: {self.model.rf_weight:.3f}")
        print(f"  XGBoost:      {self.model.xgb_weight:.3f}")
        print(f"  Ridge:        {self.model.ridge_weight:.3f}")

        test_r2 = ensemble_metrics["test"]["r2"]
        if test_r2 > 0.8:
            rating = "Excellent"
        elif test_r2 > 0.7:
            rating = "Good"
        elif test_r2 > 0.6:
            rating = "Fair"
        else:
            rating = "Needs Improvement"

        print(f"\nPerformance Rating: {rating}")

        train_r2 = ensemble_metrics["train"]["r2"]
        gap = abs(train_r2 - test_r2)
        if gap < 0.05:
            overfitting_status = "No overfitting"
        elif gap < 0.1:
            overfitting_status = "Slight overfitting"
        else:
            overfitting_status = "Significant overfitting"

        print(f"Overfitting Status: {overfitting_status}")

        best_model = max(
            individual_metrics.keys(), key=lambda x: individual_metrics[x]["r2"]
        )
        print(f"Best Individual Model: {best_model.upper()}")

        print("=" * 80)

        try:
            with open("scores.md", "a") as f:
                f.write(
                    f"\n## Simple Ensemble Model Results ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
                )
                f.write("### Ensemble Results\n")
                f.write(f"- **Train RMSE:** {ensemble_metrics['train']['rmse']:.2f}\n")
                f.write(f"- **Train MAE:** {ensemble_metrics['train']['mae']:.2f}\n")
                f.write(f"- **Train R²:** {ensemble_metrics['train']['r2']:.4f}\n")
                f.write(f"- **Test RMSE:** {ensemble_metrics['test']['rmse']:.2f}\n")
                f.write(f"- **Test MAE:** {ensemble_metrics['test']['mae']:.2f}\n")
                f.write(f"- **Test R²:** {ensemble_metrics['test']['r2']:.4f}\n")
                f.write("### Individual Models\n")
                for model_name, metrics in individual_metrics.items():
                    f.write(
                        f"- **{model_name.upper()}:** RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}\n"
                    )
                f.write("### Configuration\n")
                f.write(
                    f"- **Ensemble Weights:** RF: {self.model.rf_weight:.3f}, XGB: {self.model.xgb_weight:.3f}, Ridge: {self.model.ridge_weight:.3f}\n"
                )
                f.write(
                    f"- **Hyperparameter Tuning:** {'Yes' if self.tune_hyperparams else 'No'}\n"
                )
                f.write(f"- **Performance Rating:** {rating}\n")
                f.write(f"- **Overfitting Status:** {overfitting_status}\n")
                f.write(f"- **Best Individual Model:** {best_model.upper()}\n")
        except Exception as e:
            print(f"[scores.md] Write error: {e}")


if __name__ == "__main__":
    from pathlib import Path

    # import matplotlib.pyplot as plt

    ddir = Path(__file__).parent
    df = pd.read_csv(ddir / "train_data.csv")

    pipeline = SimpleSalesPredictionPipeline(
        rf_weight=0.5, xgb_weight=0.3, ridge_weight=0.2, tune_hyperparams=True
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

    submission = pd.DataFrame(
        {
            "Item_Identifier": test_data["Item_Identifier"],
            "Outlet_Identifier": test_data["Outlet_Identifier"],
            "Item_Outlet_Sales": predictions,
        }
    )

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

    print("\n" + "=" * 80)
    print("SIMPLE ENSEMBLE MODEL SUMMARY")
    print("=" * 80)
    print("Models Used:")
    print("  ✓ RandomForest Regressor")
    print("  ✓ XGBoost Regressor")
    print("  ✓ Ridge Regressor")
    print(
        f"Hyperparameter Tuning: {'Enabled' if pipeline.tune_hyperparams else 'Disabled'}"
    )
    print(
        f"Ensemble Weights: RF={pipeline.model.rf_weight:.3f}, XGB={pipeline.model.xgb_weight:.3f}, Ridge={pipeline.model.ridge_weight:.3f}"
    )
    print("=" * 80)
    print("=" * 80)
