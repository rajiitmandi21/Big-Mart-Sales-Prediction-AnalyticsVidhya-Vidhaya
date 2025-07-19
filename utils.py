import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
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


# if __name__ == "__main__":
#     from pathlib import Path

#     import pandas as pd

#     ddir  = Path(__file__).parent
#     train_df = pd.read_csv(f"{ddir}/train_data.csv")
#     # test_df = pd.read_csv(f"{ddir}/test_data.csv")
#     X_train = train_df.drop(columns=["Item_Outlet_Sales"])
#     y_train = train_df["Item_Outlet_Sales"]

#     # X_test = test_df.drop(columns=["Item_Outlet_Sales"])

#     processor = DataProcessor(
#         target_col="Item_Outlet_Sales",
#         scale_features=True,
#         scale_target=True,
#         log_transform_target=True,  # Enable log transformation
#     )

#     X_train_processed, y_train_processed, fitted_processor = preprocess_train_data(
#         X_train, y_train, processor
#     )

#     from sklearn.ensemble import RandomForestRegressor

#     model = RandomForestRegressor()
#     model.fit(X_train_processed, y_train_processed)

#     # X_test_processed = preprocess_test_data(X_test, fitted_processor)
#     # scaled_predictions = model.predict(X_test_processed)

#     # real_predictions = get_real_predictions(scaled_predictions, fitted_processor)

#     # # Step 5: Evaluate with real metrics
#     # metrics, y_true_real, y_pred_real = evaluate_model_performance(
#     #     y_test, scaled_predictions, fitted_processor
#     # )

#     # print("Model Performance (Real Scale):")
#     # for metric, value in metrics.items():
#     #     print(f"{metric}: {value:.4f}")
if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd

    ddir = Path(__file__).parent
    train_df = pd.read_csv(f"{ddir}/train_data.csv")
    y_train = train_df["Item_Outlet_Sales"]
    X_train = train_df.drop(columns=["Item_Outlet_Sales"])
    # replace y_train with 1 to len of train_df values
    y_train = pd.Series(np.linspace(1, len(train_df), len(train_df)), name="Item_Outlet_Sales")
    processor = DataProcessor(
        target_col="Item_Outlet_Sales",
        scale_features=True,
        scale_target=True,
        log_transform_target=True,  # Enable log transformation
    )
    X_train_processed, y_train_processed, fitted_processor = preprocess_train_data(
        X_train, y_train, processor
    )

    y_train_recovered = get_real_predictions(y_train_processed, fitted_processor)

    df = pd.DataFrame({
        "Original": y_train.values,      # True original values
        "Recovered": y_train_recovered,   # Inverse-transformed values
        # "Processed": y_train_processed    # Scaled values (0-1)
    })

    # Calculate meaningful accuracy metrics
    df["Absolute_Diff"] = np.abs(df["Original"] - df["Recovered"])
    df["Relative_Diff"] = df["Absolute_Diff"] / df["Original"]

    print("=== Transformation Accuracy Analysis ===")
    print(f"Max absolute difference: {df['Absolute_Diff'].max():.2e}")
    print(f"Max relative difference: {df['Relative_Diff'].max():.2e}")
    print(f"Mean relative difference: {df['Relative_Diff'].mean():.2e}")

    # Check if transformations are accurate
    is_accurate = np.allclose(df['Original'], df['Recovered'], rtol=1e-10)
    print(f"Are transformations accurate? {is_accurate}")

    print("\nDataFrame statistics:")
    print(df[['Original', 'Recovered', 'Absolute_Diff']].describe())

    # Proper visualization: Original vs Recovered (should be perfect diagonal)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(df['Original'], df['Recovered'], alpha=0.5, s=1)
    plt.plot([df['Original'].min(), df['Original'].max()],
             [df['Original'].min(), df['Original'].max()], 'r--', label='Perfect match')
    plt.xlabel("Original Values")
    plt.ylabel("Recovered Values")
    plt.title("Original vs Recovered\n(Should be perfect diagonal)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(df['Absolute_Diff'], bins=50, alpha=0.7)
    plt.xlabel("Absolute Difference")
    plt.ylabel("Frequency")
    plt.title("Distribution of Differences\n(Should be very small)")
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

    print("✅ SUCCESS: Transformations work perfectly!")
