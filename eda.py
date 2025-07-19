
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Set style for better visualizations
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class SalesEDA:
    def __init__(
        self,
        df,
        output_dir="outputs",
        output_file="eda_report.md",
        image_dir="images",
        write_mode="w",
    ):
        """
        Initialize EDA class with the dataframe and output settings
        """
        self.df = df.copy()
        self.target = "Item_Outlet_Sales"
        self.numerical_cols = [
            "Item_Weight",
            "Item_Visibility",
            "Item_MRP",
            "Outlet_Establishment_Year",
            "Item_Outlet_Sales",
        ]
        self.categorical_cols = [
            "Item_Identifier",
            "Item_Fat_Content",
            "Item_Type",
            "Outlet_Identifier",
            "Outlet_Size",
            "Outlet_Location_Type",
            "Outlet_Type",
        ]
        self.output_dir = output_dir
        self.image_dir = os.path.join(self.output_dir, image_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        self.file_path = os.path.join(self.output_dir, output_file)
        open(self.file_path, write_mode).close()  # Clean file at start
        self._image_count = 0
        # replace duplicate keys with unique keys in "Item_Fat_Content"
        # LF: Low Fat
        # low fat: Low Fat
        # reg: Regular
        self.df["Item_Fat_Content"] = self.df["Item_Fat_Content"].replace(
            {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
        )

    def _df_to_markdown_table(self, df):
        """
        Convert a DataFrame to a Markdown-formatted table.
        Args:
            df (pd.DataFrame): DataFrame to convert.
        Returns:
            str: Markdown table as a string.
        """
        try:
            import tabulate

            return tabulate.tabulate(df, headers="keys", tablefmt="pipe")
        except ImportError:
            return df.to_string()

    def _format_for_markdown(self, obj):
        """
        Format DataFrame, Series, or list for Markdown output.
        Args:
            obj: DataFrame, Series, list, or other object.
        Returns:
            str: Markdown-formatted string.
        """
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            return self._df_to_markdown_table(obj)
        elif isinstance(obj, pd.Series):
            return self._df_to_markdown_table(obj.to_frame())
        elif isinstance(obj, list):
            if len(obj) == 0:
                return "(empty list)"
            # If list of lists or tuples, try to make a table
            if all(isinstance(x, (list, tuple)) for x in obj):
                import tabulate

                try:
                    return tabulate.tabulate(obj, tablefmt="pipe")
                except Exception:
                    return str(obj)
            # Otherwise, Markdown bullet list
            return "\n".join([f"- {x}" for x in obj])
        else:
            return str(obj)

    def _log_and_write(self, text):
        # If not a string, format for markdown
        if not isinstance(text, str):
            text = self._format_for_markdown(text)
        with open(self.file_path, "a") as f:
            f.write(str(text) + "\n\n")

    def _save_plot_and_log(self, fig, plot_name):
        image_path = os.path.join(self.image_dir, f"{plot_name}.png")
        fig.savefig(image_path, bbox_inches="tight")
        plt.close(fig)
        rel_path = os.path.relpath(image_path, os.path.dirname(self.file_path))
        self._log_and_write(f"![{plot_name}]({rel_path})")

    def basic_info(self):
        self._log_and_write("## DATASET OVERVIEW")
        self._log_and_write("-" * 4)
        self._log_and_write(f"Dataset Shape: {self.df.shape}")
        self._log_and_write(f"Target Variable: {self.target}")
        self._log_and_write("\nData Types:")
        self._log_and_write(self.df.dtypes)
        self._log_and_write("\nBasic Statistics:")
        self._log_and_write(self.df.describe())

    def missing_values_analysis(self):
        self._log_and_write("\n" + "-" * 4)
        self._log_and_write("## MISSING VALUES ANALYSIS")
        # self._log_and_write("-" * 4)
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame(
            {"Missing_Count": missing_data, "Missing_Percentage": missing_percent}
        )
        missing_df = missing_df[missing_df["Missing_Count"] > 0].sort_values(
            "Missing_Count", ascending=False
        )
        if len(missing_df) > 0:
            self._log_and_write("Missing Values Summary:")
            self._log_and_write(missing_df)
            plt.figure(figsize=(5, 3))
            missing_df["Missing_Percentage"].plot(kind="bar")
            plt.title("Missing Values Percentage by Column")
            plt.ylabel("Percentage (%)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            fig = plt.gcf()
            self._save_plot_and_log(fig, f"missing_values_{self._image_count}")
            self._image_count += 1
        else:
            self._log_and_write("No missing values found in the dataset!")

    def target_variable_analysis(self):
        self._log_and_write("\n" + "-" * 4)
        self._log_and_write("## TARGET VARIABLE ANALYSIS")
        # self._log_and_write("-" * 4)
        target_stats = self.df[self.target].describe()
        self._log_and_write("Target Variable Statistics:")
        self._log_and_write(target_stats)
        Q1 = self.df[self.target].quantile(0.25)
        Q3 = self.df[self.target].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df[
            (self.df[self.target] < lower_bound) | (self.df[self.target] > upper_bound)
        ]
        self._log_and_write("\nOutliers Analysis:")
        self._log_and_write(
            f"Number of outliers: {len(outliers)} ({len(outliers) / len(self.df) * 100:.2f}%)"
        )
        self._log_and_write(f"Lower bound: {lower_bound:.2f}")
        self._log_and_write(f"Upper bound: {upper_bound:.2f}")
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].hist(self.df[self.target], bins=50, alpha=0.7, edgecolor="black")
        axes[0, 0].set_title("Distribution of Item_Outlet_Sales")
        axes[0, 0].set_xlabel("Sales")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 1].boxplot(self.df[self.target])
        axes[0, 1].set_title("Box Plot of Item_Outlet_Sales")
        axes[0, 1].set_ylabel("Sales")
        stats.probplot(self.df[self.target], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot (Normal Distribution)")
        log_sales = np.log1p(self.df[self.target])
        axes[1, 1].hist(log_sales, bins=50, alpha=0.7, edgecolor="black")
        axes[1, 1].set_title("Log-transformed Sales Distribution")
        axes[1, 1].set_xlabel("Log(Sales + 1)")
        axes[1, 1].set_ylabel("Frequency")
        plt.tight_layout()
        fig = plt.gcf()
        self._save_plot_and_log(fig, f"target_variable_{self._image_count}")
        self._image_count += 1
        skewness = stats.skew(self.df[self.target])
        kurtosis = stats.kurtosis(self.df[self.target])
        self._log_and_write("\nDistribution Properties:")
        self._log_and_write(f"Skewness: {skewness:.3f}")
        self._log_and_write(f"Kurtosis: {kurtosis:.3f}")

    def numerical_features_analysis(self):
        self._log_and_write("\n" + "-" * 4)
        self._log_and_write("## NUMERICAL FEATURES ANALYSIS")
        # self._log_and_write("-" * 4)
        numerical_features = [col for col in self.numerical_cols if col != self.target]
        corr_matrix = self.df[self.numerical_cols].corr()
        self._log_and_write("Correlation Matrix:")
        self._log_and_write(corr_matrix)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
        )
        plt.title("Correlation Heatmap - Numerical Features")
        plt.tight_layout()
        fig = plt.gcf()
        self._save_plot_and_log(fig, f"num_corr_heatmap_{self._image_count}")
        self._image_count += 1
        n_features = len(numerical_features)
        fig, axes = plt.subplots(2, n_features, figsize=(5 * n_features, 8))
        for i, feature in enumerate(numerical_features):
            axes[0, i].hist(
                self.df[feature].dropna(), bins=30, alpha=0.7, edgecolor="black"
            )
            axes[0, i].set_title(f"Distribution of {feature}")
            axes[0, i].set_xlabel(feature)
            axes[0, i].set_ylabel("Frequency")
            axes[1, i].scatter(self.df[feature], self.df[self.target], alpha=0.6)
            axes[1, i].set_title(f"{feature} vs {self.target}")
            axes[1, i].set_xlabel(feature)
            axes[1, i].set_ylabel(self.target)
            corr_coef = self.df[feature].corr(self.df[self.target])
            axes[1, i].text(
                0.05,
                0.95,
                f"Correlation: {corr_coef:.3f}",
                transform=axes[1, i].transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        plt.tight_layout()
        fig = plt.gcf()
        self._save_plot_and_log(fig, f"num_features_{self._image_count}")
        self._image_count += 1

    def categorical_features_analysis(self):
        self._log_and_write("\n" + "-" * 4)
        self._log_and_write("## CATEGORICAL FEATURES ANALYSIS")
        # self._log_and_write("-" * 4)
        categorical_features = [
            col for col in self.categorical_cols if col != "Item_Identifier"
        ]
        for feature in categorical_features:
            self._log_and_write(f"\n{feature} Analysis:")
            value_counts = self.df[feature].value_counts()
            self._log_and_write(f"Unique values: {self.df[feature].nunique()}")
            self._log_and_write("Value counts:")
            self._log_and_write(value_counts)
            sales_by_category = self.df.groupby(feature)[self.target].agg(
                ["mean", "median", "std", "count"]
            )
            self._log_and_write(f"\nSales statistics by {feature}:")
            self._log_and_write(sales_by_category)
        n_features = len(categorical_features)
        fig, axes = plt.subplots(2, n_features, figsize=(3 * n_features, 10))
        for i, feature in enumerate(categorical_features):
            value_counts = self.df[feature].value_counts()
            axes[0, i].bar(range(len(value_counts)), value_counts.values)
            axes[0, i].set_title(f"Distribution of {feature}")
            axes[0, i].set_xlabel(feature)
            axes[0, i].set_ylabel("Count")
            axes[0, i].set_xticks(range(len(value_counts)))
            axes[0, i].set_xticklabels(value_counts.index, rotation=45)
            self.df.boxplot(column=self.target, by=feature, ax=axes[1, i])
            axes[1, i].set_title(f"{self.target} by {feature}")
            axes[1, i].set_xlabel(feature)
            axes[1, i].set_ylabel(self.target)
        plt.tight_layout()
        fig = plt.gcf()
        self._save_plot_and_log(fig, f"cat_features_{self._image_count}")
        self._image_count += 1

    def outlet_analysis(self):
        self._log_and_write("\n" + "-" * 4)
        self._log_and_write("## OUTLET ANALYSIS")
        # self._log_and_write("-" * 4)
        outlet_performance = (
            self.df.groupby("Outlet_Identifier")[self.target]
            .agg(["mean", "median", "sum", "count", "std"])
            .round(2)
        )
        outlet_performance["cv"] = (
            outlet_performance["std"] / outlet_performance["mean"]
        ).round(3)
        outlet_performance = outlet_performance.sort_values("mean", ascending=False)
        self._log_and_write("Top 10 Outlets by Average Sales:")
        self._log_and_write(outlet_performance.head(10))
        outlet_year_analysis = (
            self.df.groupby("Outlet_Establishment_Year")[self.target]
            .agg(["mean", "count"])
            .round(2)
        )
        self._log_and_write("\nSales by Outlet Establishment Year:")
        self._log_and_write(outlet_year_analysis)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        top_outlets = outlet_performance.head(10)
        axes[0, 0].bar(range(len(top_outlets)), top_outlets["mean"])
        axes[0, 0].set_title("Top 10 Outlets by Average Sales")
        axes[0, 0].set_xlabel("Outlet")
        axes[0, 0].set_ylabel("Average Sales")
        axes[0, 0].set_xticks(range(len(top_outlets)))
        axes[0, 0].set_xticklabels(top_outlets.index, rotation=45)
        axes[0, 1].plot(
            outlet_year_analysis.index, outlet_year_analysis["mean"], marker="o"
        )
        axes[0, 1].set_title("Average Sales by Outlet Establishment Year")
        axes[0, 1].set_xlabel("Establishment Year")
        axes[0, 1].set_ylabel("Average Sales")
        pivot_table = self.df.pivot_table(
            values=self.target,
            index="Outlet_Type",
            columns="Outlet_Size",
            aggfunc="mean",
        )
        sns.heatmap(pivot_table, annot=True, cmap="viridis", ax=axes[1, 0])
        axes[1, 0].set_title("Average Sales: Outlet Type vs Size")
        location_analysis = self.df.groupby("Outlet_Location_Type")[self.target].mean()
        axes[1, 1].bar(location_analysis.index, location_analysis.values)
        axes[1, 1].set_title("Average Sales by Location Type")
        axes[1, 1].set_xlabel("Location Type")
        axes[1, 1].set_ylabel("Average Sales")
        plt.tight_layout()
        fig = plt.gcf()
        self._save_plot_and_log(fig, f"outlet_analysis_{self._image_count}")
        self._image_count += 1

    def item_analysis(self):
        self._log_and_write("\n" + "-" * 4)
        self._log_and_write("## ITEM ANALYSIS")
        # self._log_and_write("-" * 4)
        item_type_stats = (
            self.df.groupby("Item_Type")[self.target]
            .agg(["mean", "median", "count", "std"])
            .round(2)
        )
        item_type_stats = item_type_stats.sort_values("mean", ascending=False)
        self._log_and_write("Sales by Item Type:")
        self._log_and_write(item_type_stats)
        fat_content_stats = (
            self.df.groupby("Item_Fat_Content")[self.target]
            .agg(["mean", "median", "count"])
            .round(2)
        )
        self._log_and_write("\nSales by Fat Content:")
        self._log_and_write(fat_content_stats)
        self.df["MRP_Category"] = pd.cut(
            self.df["Item_MRP"],
            bins=5,
            labels=["Low", "Low-Med", "Medium", "Med-High", "High"],
        )
        mrp_analysis = (
            self.df.groupby("MRP_Category")[self.target].agg(["mean", "count"]).round(2)
        )
        self._log_and_write("\nSales by MRP Category:")
        self._log_and_write(mrp_analysis)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        top_items = item_type_stats.head(10)
        axes[0, 0].barh(range(len(top_items)), top_items["mean"])
        axes[0, 0].set_title("Top 10 Item Types by Average Sales")
        axes[0, 0].set_xlabel("Average Sales")
        axes[0, 0].set_yticks(range(len(top_items)))
        axes[0, 0].set_yticklabels(top_items.index)
        fat_content_stats["mean"].plot(kind="bar", ax=axes[0, 1])
        axes[0, 1].set_title("Average Sales by Fat Content")
        axes[0, 1].set_ylabel("Average Sales")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[1, 0].scatter(self.df["Item_MRP"], self.df[self.target], alpha=0.6)
        axes[1, 0].set_title("Item MRP vs Sales")
        axes[1, 0].set_xlabel("Item MRP")
        axes[1, 0].set_ylabel("Sales")
        mrp_analysis["mean"].plot(kind="bar", ax=axes[1, 1])
        axes[1, 1].set_title("Average Sales by MRP Category")
        axes[1, 1].set_ylabel("Average Sales")
        axes[1, 1].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig = plt.gcf()
        self._save_plot_and_log(fig, f"item_analysis_{self._image_count}")
        self._image_count += 1

    def feature_interactions(self):
        self._log_and_write("\n" + "-" * 4)
        self._log_and_write("## FEATURE INTERACTIONS ANALYSIS")
        # self._log_and_write("-" * 4)
        interactions = [
            ("Item_Type", "Outlet_Type"),
            ("Item_Fat_Content", "Outlet_Size"),
            ("Outlet_Location_Type", "Outlet_Type"),
            ("Item_Type", "Outlet_Location_Type"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        for i, (feature1, feature2) in enumerate(interactions):
            pivot_table = self.df.pivot_table(
                values=self.target, index=feature1, columns=feature2, aggfunc="mean"
            )
            sns.heatmap(pivot_table, annot=True, cmap="viridis", ax=axes[i])
            axes[i].set_title(f"Average Sales: {feature1} vs {feature2}")
        plt.tight_layout()
        fig = plt.gcf()
        self._save_plot_and_log(fig, f"feature_interactions_{self._image_count}")
        self._image_count += 1

    def statistical_tests(self):
        self._log_and_write("\n" + "-" * 4)
        self._log_and_write("## STATISTICAL TESTS")
        # self._log_and_write("-" * 4)
        for feature in self.categorical_cols:
            if self.df[feature].nunique() > 1:
                groups = [
                    group[self.target].values
                    for name, group in self.df.groupby(feature)
                ]
                groups = [group for group in groups if len(group) > 0]
                if len(groups) > 1:
                    f_stat, p_value = stats.f_oneway(*groups)
                    self._log_and_write(f"\nANOVA Test for {feature}:")
                    self._log_and_write(f"F-statistic: {f_stat:.4f}")
                    self._log_and_write(f"P-value: {p_value:.4f}")
                    if p_value < 0.05:
                        self._log_and_write("Result: Significant difference between groups")
                    else:
                        self._log_and_write("Result: No significant difference between groups")

    def data_quality_checks(self):
        self._log_and_write("\n" + "-" * 4)
        self._log_and_write("## DATA QUALITY CHECKS")
        # self._log_and_write("-" * 4)
        duplicates = self.df.duplicated().sum()
        self._log_and_write(f"Number of duplicate rows: {duplicates}")
        negative_sales = (self.df[self.target] < 0).sum()
        self._log_and_write(f"Number of negative sales: {negative_sales}")
        zero_sales = (self.df[self.target] == 0).sum()
        self._log_and_write(f"Number of zero sales: {zero_sales}")
        fat_content_variations = self.df["Item_Fat_Content"].unique()
        self._log_and_write(f"Fat content variations: {fat_content_variations}")
        outlet_patterns = (
            self.df["Outlet_Identifier"].apply(lambda x: x[:3]).value_counts()
        )
        self._log_and_write(f"Outlet identifier patterns: {outlet_patterns}")

    def generate_insights(self):
        self._log_and_write("\n" + "-" * 4)
        self._log_and_write("## KEY INSIGHTS FOR MODELING")
        # self._log_and_write("-" * 4)
        insights = []
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            insights.append(
                "• Missing values detected in Item_Weight and Outlet_Size - consider imputation strategies"
            )
        skewness = stats.skew(self.df[self.target])
        if abs(skewness) > 1:
            insights.append(
                "• Target variable is highly skewed - consider log transformation"
            )
        corr_with_target = (
            self.df[self.numerical_cols]
            .corr()[self.target]
            .abs()
            .sort_values(ascending=False)
        )
        top_corr_feature = corr_with_target.index[1]  # Exclude target itself
        insights.append(
            f"• {top_corr_feature} shows highest correlation with target ({corr_with_target.iloc[1]:.3f})"
        )
        high_cardinality_features = []
        for col in self.categorical_cols:
            if self.df[col].nunique() > 20:
                high_cardinality_features.append(col)
        if high_cardinality_features:
            insights.append(
                f"• High cardinality features: {high_cardinality_features} - consider target encoding"
            )
        outlet_performance = self.df.groupby("Outlet_Identifier")[self.target].mean()
        cv_outlets = outlet_performance.std() / outlet_performance.mean()
        if cv_outlets > 0.3:
            insights.append(
                "• High variation in outlet performance - outlet features are important"
            )
        for insight in insights:
            self._log_and_write(insight)
        return insights

    def run_complete_eda(self):
        self._log_and_write("Starting Comprehensive EDA for Item_Outlet_Sales Estimation...")
        self._log_and_write("-" * 4)
        self.basic_info()
        # self._missing_values_analysis()
        self.missing_values_analysis()
        self.target_variable_analysis()
        self.numerical_features_analysis()
        self.categorical_features_analysis()
        self.outlet_analysis()
        self.item_analysis()
        self.feature_interactions()
        self.statistical_tests()
        self.data_quality_checks()
        insights = self.generate_insights()
        self._log_and_write("\n" + "-" * 4)
        self._log_and_write("EDA COMPLETE!")
        self._log_and_write("-" * 4)
        return insights


if __name__ == "__main__":
    df = pd.read_csv("train_data.csv")
    eda = SalesEDA(df)
    eda.run_complete_eda()
    # Usage Example:
    # Assuming your dataframe is loaded as 'df'
    eda = SalesEDA(df)
    insights = eda.run_complete_eda()
    # Or run individual analyses:
    # eda.basic_info()
    # eda.target_variable_analysis()
    # eda.numerical_features_analysis()

