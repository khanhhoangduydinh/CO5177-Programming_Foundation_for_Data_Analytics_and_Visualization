# %% [markdown]
# # Kepler Objects of Interest — Tabular EDA
#
# **Course:** CO5177 — Programming Foundation for Data Analytics and Visualization
#
# **Student:** Dinh Hoang Duy Khanh — 2670306
# **Group:** Trailblazer

# %% [markdown]
# ## 0. Import libraries

# %%
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

# %%
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# %%
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180})
colors = ["#d35b12", "#efae00", "#173f4c"]

# %%
project_folder = Path.cwd()
if project_folder.name == "notebooks":
    project_folder = project_folder.parent

figure_folder = project_folder / "reports" / "figures"
figure_folder.mkdir(parents=True, exist_ok=True)

# %%
def save_plot(name):
    plt.tight_layout()
    plt.savefig(figure_folder / name, bbox_inches="tight", facecolor="white")
    plt.show()


# %% [markdown]
# ## 1. Load the dataset

# %%
data_url = "https://raw.githubusercontent.com/khanhhoangduydinh/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/main/data/koi_cumulative.csv"
df = pd.read_csv(data_url)
df.head()

# %%
print("Shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Missing cells:", df.isnull().sum().sum())

# %%
df.info()

# %%
df.describe().round(2)

# %% [markdown]
# The dataset has 9,564 rows and 30 columns. It satisfies the assignment
# requirement of more than 2,000 rows and more than 10 columns.

# %% [markdown]
# ## 2. Dataset overview

# %%
overview = pd.DataFrame({
    "Rows": [df.shape[0]],
    "Columns": [df.shape[1]],
    "Numerical": [df.select_dtypes(include="number").shape[1]],
    "Categorical": [df.select_dtypes(exclude="number").shape[1]],
})
overview

# %%
df.head(6)

# %% [markdown]
# ## 3. Target distribution

# %%
target_counts = df["koi_disposition"].value_counts()
target_percent = df["koi_disposition"].value_counts(normalize=True) * 100
pd.DataFrame({"Count": target_counts, "Percent": target_percent.round(2)})

# %%
plt.figure(figsize=(8.4, 5.2))
ax = sns.barplot(x=target_counts.index, y=target_counts.values, hue=target_counts.index, palette=colors, legend=False)
ax.set(title="KOI disposition is moderately imbalanced", xlabel="Disposition", ylabel="Objects")
for bar in ax.containers:
    ax.bar_label(bar, fmt="{:,.0f}", padding=4)
save_plot("01_target_distribution.png")

# %% [markdown]
# False positives are the largest class. Therefore, accuracy is not enough;
# macro F1 and balanced accuracy are also used.

# %% [markdown]
# ## 4. Missing values

# %%
missing = pd.DataFrame({"Count": df.isnull().sum()})
missing["Percent"] = missing["Count"] / len(df) * 100
missing = missing[missing["Count"] > 0]
missing = missing.sort_values("Percent", ascending=False)
missing.round(2)

# %%
missing_plot = missing.head(15).sort_values("Percent")
plt.figure(figsize=(9.5, 6.8))
sns.barplot(x=missing_plot["Percent"], y=missing_plot.index, color="#d35b12")
plt.title("Top columns by missing-value share")
plt.xlabel("Missing values (%)")
plt.ylabel("")
save_plot("02_missing_values.png")

# %% [markdown]
# Missing numerical values are filled with the median. Missing categorical
# values are filled with the most common value.

# %% [markdown]
# ## 5. Numerical distributions

# %%
plt.figure(figsize=(9, 5.6))
sns.histplot(df["koi_period"].dropna(), bins=60, color="#173f4c")
plt.xscale("log")
plt.title("Orbital period spans several orders of magnitude")
plt.xlabel("Orbital period (days, log scale)")
plt.ylabel("Objects")
save_plot("03_orbital_period_distribution.png")

# %%
radius = df.loc[df["koi_prad"] > 0, "koi_prad"].dropna()
plt.figure(figsize=(9, 5.6))
sns.histplot(radius, bins=60, color="#d35b12")
plt.xscale("log")
plt.title("Estimated planetary radius contains extreme values")
plt.xlabel("Planet radius (Earth radii, log scale)")
plt.ylabel("Objects")
save_plot("04_planet_radius_distribution.png")

# %%
plt.figure(figsize=(9, 5.6))
sns.histplot(df["koi_steff"].dropna(), bins=45, color="#efae00")
plt.axvline(df["koi_steff"].median(), color="#173f4c", linestyle="--", label="Median")
plt.title("Host-star effective temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("Objects")
plt.legend()
save_plot("05_stellar_temperature_distribution.png")

# %% [markdown]
# ## 6. Categorical distribution

# %%
delivery = df["koi_tce_delivname"].fillna("Missing").value_counts()
plt.figure(figsize=(9, 5.6))
ax = sns.barplot(x=delivery.values, y=delivery.index, hue=delivery.index, palette=["#173f4c", "#d35b12", "#a9cfc1", "#efae00"], legend=False)
for bar in ax.containers:
    ax.bar_label(bar, fmt="{:,.0f}", padding=5)
ax.set(title="KOIs by TCE delivery catalogue", xlabel="Objects", ylabel="")
save_plot("15_tce_delivery_distribution.png")

# %% [markdown]
# ## 7. Outliers

# %%
numeric_cols = [
    "koi_period", "koi_impact", "koi_duration", "koi_depth",
    "koi_prad", "koi_teq", "koi_insol", "koi_model_snr",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
]

# %%
outlier_list = []
for col in numeric_cols:
    values = df[col].dropna()
    q1, q3 = values.quantile([0.25, 0.75])
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    count = ((values < lower) | (values > upper)).sum()
    outlier_list.append([col, count, count / len(values) * 100])

# %%
outliers = pd.DataFrame(outlier_list, columns=["Feature", "Count", "Percent"])
outliers = outliers.sort_values("Percent", ascending=False)
outliers.round(2)

# %%
outlier_plot = outliers.sort_values("Percent")
plt.figure(figsize=(9.5, 6.2))
sns.barplot(data=outlier_plot, x="Percent", y="Feature", color="#efae00")
plt.title("IQR outlier share by model feature")
plt.xlabel("Outliers (%)")
plt.ylabel("")
save_plot("06_outlier_share.png")

# %% [markdown]
# Outliers are kept because rare astronomical objects may still be valid.

# %% [markdown]
# ## 8. Target and numerical features

# %%
plot_df = df[df["koi_prad"].between(0, df["koi_prad"].quantile(0.99))]
plt.figure(figsize=(9, 5.8))
sns.boxplot(data=plot_df, x="koi_disposition", y="koi_prad", hue="koi_disposition", palette=colors, legend=False, showfliers=False)
plt.title("Planet radius differs across dispositions")
plt.xlabel("Disposition")
plt.ylabel("Planet radius (Earth radii; ≤99th percentile)")
save_plot("07_target_vs_planet_radius.png")

# %%
plot_df = df[df["koi_model_snr"].between(0, df["koi_model_snr"].quantile(0.99))]
plt.figure(figsize=(9, 5.8))
sns.boxplot(data=plot_df, x="koi_disposition", y="koi_model_snr", hue="koi_disposition", palette=colors, legend=False, showfliers=False)
plt.title("Transit signal-to-noise by disposition")
plt.xlabel("Disposition")
plt.ylabel("Model SNR (≤99th percentile)")
save_plot("08_target_vs_snr.png")

# %% [markdown]
# The classes overlap. Therefore, one feature alone cannot solve the problem.

# %% [markdown]
# ## 9. Correlation

# %%
corr_cols = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_steff", "koi_srad",
]
corr = df[corr_cols].corr(method="spearman")

# %%
plt.figure(figsize=(10, 7.8))
sns.heatmap(corr, cmap=sns.diverging_palette(220, 25, as_cmap=True), center=0,
            vmin=-1, vmax=1, annot=True, fmt=".2f", square=True,
            cbar_kws={"label": "Spearman correlation"})
plt.title("Rank correlations among selected measurements", pad=16)
save_plot("09_correlation_heatmap.png")

# %% [markdown]
# ## 10. Select features and split data
#
# Columns that directly reveal the result are not used. Examples are
# `koi_score`, `koi_pdisposition`, `kepler_name`, and false-positive flags.

# %%
category_cols = ["koi_tce_delivname"]
features = numeric_cols + category_cols
target = "koi_disposition"

X = df[features]
y = df[target]
groups = df["kepid"]

# %%
split1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_id, temp_id = next(split1.split(X, y, groups))

X_train, y_train = X.iloc[train_id], y.iloc[train_id]
temp = df.iloc[temp_id]

# %%
split2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_id, test_id = next(split2.split(temp[features], temp[target], temp["kepid"]))

X_val, y_val = temp.iloc[val_id][features], temp.iloc[val_id][target]
X_test, y_test = temp.iloc[test_id][features], temp.iloc[test_id][target]

# %%
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# %% [markdown]
# The split uses `kepid`, so objects from the same star cannot appear in two
# different sets.

# %% [markdown]
# ## 11. Before and after scaling

# %%
scale_cols = ["koi_period", "koi_depth", "koi_steff"]
before = X_train[scale_cols].melt(var_name="Feature", value_name="Value").dropna()
plt.figure(figsize=(9, 5.6))
sns.boxplot(data=before, x="Feature", y="Value", color="#f1c9b2", showfliers=False)
plt.title("Before scaling: feature magnitudes are incompatible")
plt.xlabel("")
plt.ylabel("Raw value")
save_plot("10_before_scaling.png")

# %%
scale_data = X_train[scale_cols].copy()
scale_data = scale_data.fillna(scale_data.median())
scale_data = pd.DataFrame(StandardScaler().fit_transform(scale_data), columns=scale_cols)
after = scale_data.melt(var_name="Feature", value_name="Value")

# %%
plt.figure(figsize=(9, 5.6))
sns.boxplot(data=after, x="Feature", y="Value", color="#a9cfc1", showfliers=False)
plt.title("After standardization: features share a comparable scale")
plt.xlabel("")
plt.ylabel("Standard deviations")
save_plot("11_after_scaling.png")

# %% [markdown]
# ## 12. Logistic Regression

# %%
log_prep = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
    (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), category_cols),
)

# %%
log_model = make_pipeline(log_prep, LogisticRegression(max_iter=2500, random_state=42))
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_val)

# %%
print("Logistic accuracy:", round(accuracy_score(y_val, log_pred), 4))
print("Logistic balanced accuracy:", round(balanced_accuracy_score(y_val, log_pred), 4))
print("Logistic macro F1:", round(f1_score(y_val, log_pred, average="macro"), 4))

# %% [markdown]
# ## 13. Random Forest

# %%
forest_prep = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), RobustScaler()), numeric_cols),
    (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), category_cols),
)

# %%
forest = RandomForestClassifier(n_estimators=350, min_samples_leaf=2,
                                class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)
forest_model = make_pipeline(forest_prep, forest)
forest_model.fit(X_train, y_train)
forest_pred = forest_model.predict(X_val)

# %%
print("Forest accuracy:", round(accuracy_score(y_val, forest_pred), 4))
print("Forest balanced accuracy:", round(balanced_accuracy_score(y_val, forest_pred), 4))
print("Forest macro F1:", round(f1_score(y_val, forest_pred, average="macro"), 4))

# %% [markdown]
# ## 14. Compare models

# %%
validation_results = pd.DataFrame({
    "accuracy": [accuracy_score(y_val, forest_pred), accuracy_score(y_val, log_pred)],
    "balanced_accuracy": [balanced_accuracy_score(y_val, forest_pred), balanced_accuracy_score(y_val, log_pred)],
    "macro_f1": [f1_score(y_val, forest_pred, average="macro"), f1_score(y_val, log_pred, average="macro")],
}, index=["Balanced Random Forest", "Logistic baseline"])
validation_results.index.name = "model"
validation_results.round(4)

# %%
plot_scores = validation_results.reset_index()
plot_scores = plot_scores.melt(id_vars="model", var_name="metric", value_name="score")

# %%
plt.figure(figsize=(10, 5.8))
ax = sns.barplot(data=plot_scores, x="metric", y="score", hue="model", palette=["#a9cfc1", "#d35b12"])
for bar in ax.containers:
    ax.bar_label(bar, fmt="%.3f", padding=3, fontsize=8)
ax.set(title="Validation comparison: baseline vs class-aware nonlinear model",
       xlabel="", ylabel="Score", ylim=(0, 1))
ax.legend(title="")
save_plot("12_model_comparison.png")

# %% [markdown]
# Random Forest has the higher macro F1, so it is selected as the final model.

# %% [markdown]
# ## 15. Test results

# %%
test_pred = forest_model.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, test_pred), 4))
print("Test balanced accuracy:", round(balanced_accuracy_score(y_test, test_pred), 4))
print("Test macro F1:", round(f1_score(y_test, test_pred, average="macro"), 4))

# %%
print(classification_report(y_test, test_pred))

# %%
plt.figure(figsize=(7.6, 6.4))
cm = ConfusionMatrixDisplay.from_predictions(
    y_test, test_pred, normalize="true", values_format=".2f",
    cmap=sns.light_palette("#173f4c", as_cmap=True), colorbar=False, ax=plt.gca(),
)
cm.ax_.set(title="Normalized test confusion matrix — Balanced Random Forest",
           xlabel="Predicted disposition", ylabel="True disposition")
save_plot("13_confusion_matrix.png")

# %% [markdown]
# Candidate objects are the hardest class to predict.

# %% [markdown]
# ## 16. Feature importance

# %%
result = permutation_importance(
    forest_model, X_test, y_test, n_repeats=8,
    random_state=42, scoring="f1_macro", n_jobs=-1,
)

# %%
importance = pd.DataFrame({
    "feature": features,
    "importance": result.importances_mean,
    "std": result.importances_std,
})
importance = importance.sort_values("importance", ascending=False)
importance.round(4)

# %%
top_features = importance.head(10).sort_values("importance")
plt.figure(figsize=(9.2, 6.2))
ax = sns.barplot(data=top_features, x="importance", y="feature", color="#173f4c")
ax.errorbar(top_features["importance"], np.arange(len(top_features)),
            xerr=top_features["std"], fmt="none", ecolor="#d35b12", capsize=3)
ax.set(title="Permutation importance on unseen host stars",
       xlabel="Decrease in macro F1 after shuffling", ylabel="")
save_plot("14_feature_importance.png")

# %% [markdown]
# ## 17. Conclusion
#
# - The dataset has missing values, imbalance, skewness, and outliers.
# - Random Forest performs better than Logistic Regression.
# - The final test macro F1 is about 0.727.
# - `koi_model_snr`, `koi_prad`, and `koi_impact` are important features.
# - The model supports analysis but cannot confirm a real planet.
