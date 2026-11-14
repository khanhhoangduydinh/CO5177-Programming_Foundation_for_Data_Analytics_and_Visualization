# %% [markdown]
# # Kepler Objects of Interest — Tabular EDA
#
# **Course:** CO5177 — Programming Foundation for Data Analytics and Visualization
#
# **Student:** Dinh Hoang Duy Khanh — 2670306
# **Group:** Trailblazer
#
# **Cách chạy:** Mở notebook trong Google Colab, chọn Runtime → Run all và chờ từng cell hoàn tất. Lần đầu cần Internet để tải dữ liệu. Đọc hướng dẫn tiếng Việt ngay trước mỗi cell và kiểm tra kết quả sau khi chạy.

# %% [markdown]
# ## 0. Import libraries

# %% [markdown]
# Nạp Path để quản lý đường dẫn, warnings để điều chỉnh cảnh báo và các thư viện vẽ, mảng, bảng dữ liệu. Sau cell này, ta dùng được pandas, NumPy, Matplotlib và Seaborn.

# %%
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% [markdown]
# Nạp các thành phần tiền xử lý và hai mô hình phân loại. ColumnTransformer áp dụng cách xử lý riêng cho từng nhóm cột; Pipeline nối các bước thành một quy trình.

# %%
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

# %% [markdown]
# Nạp thước đo đánh giá, cách chia theo nhóm sao và các bộ mã hóa/chuẩn hóa. Macro F1 xem các lớp có trọng số ngang nhau, hữu ích khi dữ liệu lệch lớp.

# %%
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# %% [markdown]
# Ẩn riêng cảnh báo FutureWarning, chọn kiểu lưới và màu cho biểu đồ. Các thiết lập này chỉ thay đổi cách trình bày, không thay đổi dữ liệu hay kết quả huấn luyện.

# %%
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180})
colors = ["#d35b12", "#efae00", "#173f4c"]

# %% [markdown]
# Lấy thư mục làm việc hiện tại; nếu đang ở notebooks thì đi lên thư mục gốc. Tạo reports/figures để những hình tiếp theo có nơi lưu.

# %%
project_folder = Path.cwd()
if project_folder.name == "notebooks":
    project_folder = project_folder.parent

figure_folder = project_folder / "reports" / "figures"
figure_folder.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# Định nghĩa save_plot: căn biểu đồ, lưu ảnh vào thư mục figures rồi hiển thị. Tên tệp được truyền vào khi gọi hàm ở từng cell vẽ hình.

# %%
def save_plot(name):
    plt.tight_layout()
    plt.savefig(figure_folder / name, bbox_inches="tight", facecolor="white")
    plt.show()


# %% [markdown]
# ## 1. Load the dataset

# %% [markdown]
# Tải bảng KOI từ tệp CSV của repository bằng pandas; df là toàn bộ bảng. head() hiển thị vài dòng đầu để kiểm tra tên cột và dữ liệu.

# %%
data_url = "https://raw.githubusercontent.com/khanhhoangduydinh/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/main/data/koi_cumulative.csv"
df = pd.read_csv(data_url)
df.head()

# %% [markdown]
# In số dòng và số cột qua shape, số dòng trùng hoàn toàn qua duplicated, và tổng số ô thiếu qua isnull. Các số này giúp ước lượng chất lượng dữ liệu ban đầu.

# %%
print("Shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Missing cells:", df.isnull().sum().sum())

# %% [markdown]
# Dùng info() để xem kiểu dữ liệu, số giá trị không thiếu và mức sử dụng bộ nhớ của từng cột. Đây là bước kiểm tra trước khi chọn cách xử lý.

# %%
df.info()

# %% [markdown]
# describe() tóm tắt các cột số bằng số lượng, trung bình, độ lệch chuẩn và các phân vị; round(2) làm tròn để dễ đọc.

# %%
df.describe().round(2)

# %% [markdown]
# The dataset has 9,564 rows and 30 columns. It satisfies the assignment
# requirement of more than 2,000 rows and more than 10 columns.

# %% [markdown]
# ## 2. Dataset overview

# %% [markdown]
# Tạo bảng overview gồm số dòng, số cột, số cột số và số cột không phải số. select_dtypes phân biệt kiểu lưu trữ, không tự quyết định ý nghĩa của biến.

# %%
overview = pd.DataFrame({
    "Rows": [df.shape[0]],
    "Columns": [df.shape[1]],
    "Numerical": [df.select_dtypes(include="number").shape[1]],
    "Categorical": [df.select_dtypes(exclude="number").shape[1]],
})
overview

# %% [markdown]
# Hiển thị sáu dòng mẫu để quan sát từng trường. Đây là dữ liệu gốc trước khi điền giá trị thiếu hoặc chia tập.

# %%
df.head(6)

# %% [markdown]
# ## 3. Target distribution

# %% [markdown]
# Đếm từng nhãn koi_disposition và tính tỷ lệ phần trăm. normalize=True chia số lượng mỗi lớp cho tổng số dòng.

# %%
target_counts = df["koi_disposition"].value_counts()
target_percent = df["koi_disposition"].value_counts(normalize=True) * 100
pd.DataFrame({"Count": target_counts, "Percent": target_percent.round(2)})

# %% [markdown]
# Vẽ biểu đồ cột cho số mẫu của ba lớp và ghi số lượng trên cột. So sánh chiều cao các cột để nhận biết mức mất cân bằng lớp.

# %%
display_labels = {
    "CONFIRMED": "Confirmed planet",
    "FALSE POSITIVE": "Non-planet signal",
    "CANDIDATE": "Planet candidate",
}
plot_counts = target_counts.rename(index=display_labels)
plt.figure(figsize=(8.4, 5.2))
ax = sns.barplot(x=plot_counts.index, y=plot_counts.values,
                 hue=plot_counts.index, palette=colors, legend=False)
ax.set(title="Catalogue category distribution", xlabel="Catalogue category", ylabel="KOI records")
for bar in ax.containers:
    ax.bar_label(bar, fmt="{:,.0f}", padding=4)
save_plot("01_target_distribution.png")

# %% [markdown]
# False positives are the largest class. Therefore, accuracy is not enough;
# macro F1 and balanced accuracy are also used.

# %% [markdown]
# ## 4. Missing values

# %% [markdown]
# Đếm số giá trị thiếu theo từng cột, đổi sang phần trăm, chỉ giữ cột có thiếu và sắp xếp giảm dần. Bảng này giúp ưu tiên kiểm tra các cột thiếu nhiều.

# %%
missing = pd.DataFrame({"Count": df.isnull().sum()})
missing["Percent"] = missing["Count"] / len(df) * 100
missing = missing[missing["Count"] > 0]
missing = missing.sort_values("Percent", ascending=False)
missing.round(2)

# %% [markdown]
# Vẽ tối đa 15 cột thiếu nhiều nhất. Chiều dài thanh là tỷ lệ thiếu, không phải tỷ lệ sai; hình được lưu bằng save_plot.

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

# %% [markdown]
# Chia các khoảng trên thang log; chỉ vẽ chu kỳ dương, không thiếu.

# %%
period = df.loc[df["koi_period"] > 0, "koi_period"].dropna()
plt.figure(figsize=(9, 5.6))
sns.histplot(period, bins=60, log_scale=True, color="#173f4c")
plt.title("Orbital period distribution")
plt.xlabel("Orbital period (days, log scale)")
plt.ylabel("Objects")
save_plot("03_orbital_period_distribution.png")

# %% [markdown]
# Chia các khoảng trên thang log; chỉ vẽ bán kính dương, không thiếu.

# %%
radius = df.loc[df["koi_prad"] > 0, "koi_prad"].dropna()
plt.figure(figsize=(9, 5.6))
sns.histplot(radius, bins=60, log_scale=True, color="#d35b12")
plt.title("Estimated planetary radius distribution")
plt.xlabel("Planet radius (Earth radii, log scale)")
plt.ylabel("Objects")
save_plot("04_planet_radius_distribution.png")

# %% [markdown]
# Vẽ phân phối nhiệt độ sao chủ và đánh dấu trung vị bằng đường nét đứt. Trung vị là giá trị ở giữa, ít bị cực trị chi phối hơn trung bình.

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
# Phân phối 9 biến số còn lại: chỉ hiển thị phân vị 1%–99%; dữ liệu gốc được giữ nguyên.

# %%
distribution_columns = [
    ("koi_impact", "Impact parameter"),
    ("koi_duration", "Transit duration (hours)"),
    ("koi_depth", "Transit depth (ppm)"),
    ("koi_teq", "Equilibrium temperature (K)"),
    ("koi_insol", "Insolation (Earth flux)"),
    ("koi_model_snr", "Transit model SNR"),
    ("koi_slogg", "Stellar surface gravity (log g)"),
    ("koi_srad", "Stellar radius (solar radii)"),
    ("koi_kepmag", "Kepler magnitude"),
]
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
for ax, (column, label) in zip(axes.flat, distribution_columns):
    values = df[column].dropna()
    lower, upper = values.quantile([0.01, 0.99])
    shown = values[values.between(lower, upper)]
    if column in {"koi_depth", "koi_insol", "koi_model_snr"}:
        shown = shown[shown > 0]
        ax.hist(shown, bins=np.geomspace(shown.min(), shown.max(), 36),
                color="#173f4c", edgecolor="white")
        ax.set_xscale("log")
        label += " (log scale)"
    else:
        ax.hist(shown, bins=35, color="#173f4c", edgecolor="white")
    ax.set(title=column, xlabel=label, ylabel="Objects")
    ax.text(0.98, 0.95, f"Shown: {len(shown):,} / {len(values):,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
fig.suptitle("Additional model features — 1st to 99th percentile shown", y=1.01)
save_plot("16_additional_numeric_distributions.png")

# %% [markdown]
# ## 6. Categorical distribution

# %% [markdown]
# Điền nhãn Missing tạm thời cho cột phân loại, đếm từng nhóm và vẽ biểu đồ thanh. Mục đích là thấy các nhóm danh mục và số giá trị thiếu.

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

# %% [markdown]
# Liệt kê các biến số được chọn để khảo sát và đưa vào mô hình. Danh sách rõ ràng giúp kiểm soát đặc trưng thay vì sử dụng mọi cột trong bảng.

# %%
numeric_cols = [
    "koi_period", "koi_impact", "koi_duration", "koi_depth",
    "koi_prad", "koi_teq", "koi_insol", "koi_model_snr",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
]

# %% [markdown]
# Với mỗi biến số, tính Q1, Q3 và khoảng IQR = Q3 − Q1; đếm giá trị ngoài [Q1 − 1,5 IQR; Q3 + 1,5 IQR]. Giá trị ngoại lệ thống kê không nhất thiết là dữ liệu sai.

# %%
outlier_list = []
for col in numeric_cols:
    values = df[col].dropna()
    q1, q3 = values.quantile([0.25, 0.75])
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    count = ((values < lower) | (values > upper)).sum()
    outlier_list.append([col, count, count / len(values) * 100])

# %% [markdown]
# Chuyển danh sách số ngoại lệ thành DataFrame, sắp theo tỷ lệ và làm tròn. Có thể so sánh mức độ cực trị giữa các biến trong bảng.

# %%
outliers = pd.DataFrame(outlier_list, columns=["Feature", "Count", "Percent"])
outliers = outliers.sort_values("Percent", ascending=False)
outliers.round(2)

# %% [markdown]
# Vẽ tỷ lệ ngoại lệ IQR của từng biến. Biểu đồ hỗ trợ lựa chọn cách xử lý, nhưng cell này không tự động loại bỏ mẫu.

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

# %% [markdown]
# Chỉ để vẽ, giới hạn bán kính đến phân vị 99% và tạo boxplot theo lớp mục tiêu. So sánh trung vị và độ phân tán giữa các lớp, không suy ra quan hệ nhân quả.

# %%
plot_df = df[df["koi_prad"].between(0, df["koi_prad"].quantile(0.99))]
plt.figure(figsize=(9, 5.8))
sns.boxplot(data=plot_df, x="koi_disposition", y="koi_prad", hue="koi_disposition", palette=colors, legend=False, showfliers=False)
plt.title("Planet radius differs across dispositions")
plt.xlabel("Disposition")
plt.ylabel("Planet radius (Earth radii; ≤99th percentile)")
save_plot("07_target_vs_planet_radius.png")

# %% [markdown]
# Làm tương tự với tỷ số tín hiệu trên nhiễu: giới hạn phần đuôi cho biểu đồ và so sánh các lớp bằng boxplot. Dữ liệu huấn luyện không bị cắt bởi cell vẽ này.

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

# %% [markdown]
# Chọn các biến số quan trọng và tính tương quan hạng Spearman giữa từng cặp. Giá trị gần +1 hoặc −1 cho thấy quan hệ đơn điệu mạnh hơn.

# %%
corr_cols = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_steff", "koi_srad",
]
corr = df[corr_cols].corr(method="spearman")

# %% [markdown]
# Hiển thị ma trận tương quan bằng heatmap, kèm giá trị làm tròn hai chữ số. Màu thể hiện chiều và độ mạnh tương quan, không chứng minh nhân quả.

# %%
plt.figure(figsize=(10, 7.8))
sns.heatmap(corr, cmap=sns.diverging_palette(220, 25, as_cmap=True), center=0,
            vmin=-1, vmax=1, annot=True, fmt=".2f", square=True,
            cbar_kws={"label": "Spearman correlation"})
plt.title("Rank correlations among selected measurements", pad=16)
save_plot("09_correlation_heatmap.png")

# %% [markdown]
# ## 10. Binary classification: select labels and split data
# EDA retains three labels. Exclude CANDIDATE before the binary group split.
#
# Columns that directly reveal the result are not used. Examples are
# `koi_score`, `koi_pdisposition`, `kepler_name`, and false-positive flags.

# %% [markdown]
# Tạo X từ biến số và biến phân loại, y là nhãn cần dự đoán, groups là mã sao kepid. Mã nhóm dùng để tránh cùng một sao xuất hiện ở nhiều tập.

# %%
category_cols = ["koi_tce_delivname"]
features = numeric_cols + category_cols
target = "koi_disposition"

model_df = df[df[target].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
X = model_df[features]
y = model_df[target]
groups = model_df["kepid"]
print("Binary modeling rows:", len(model_df))
print("Excluded CANDIDATE rows:", int((df[target] == "CANDIDATE").sum()))
print(y.value_counts())

# %% [markdown]
# Chia theo nhóm sao bằng GroupShuffleSplit: khoảng 70% vào train và phần còn lại vào tập tạm. random_state=42 giúp phép chia có thể lặp lại.

# %%
split1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_id, temp_id = next(split1.split(X, y, groups))

X_train, y_train = X.iloc[train_id], y.iloc[train_id]
temp = model_df.iloc[temp_id]

# %% [markdown]
# Chia tiếp tập tạm thành validation và test theo nhóm sao. Validation dùng so sánh mô hình; test giữ lại cho đánh giá sau cùng.

# %%
split2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_id, test_id = next(split2.split(temp[features], temp[target], temp["kepid"]))

X_val, y_val = temp.iloc[val_id][features], temp.iloc[val_id][target]
X_test, y_test = temp.iloc[test_id][features], temp.iloc[test_id][target]

# Check binary labels and ensure host-star groups never cross splits.
assert set(y.unique()) == {"CONFIRMED", "FALSE POSITIVE"}
assert set(model_df.iloc[train_id]["kepid"]).isdisjoint(temp["kepid"])
assert set(temp.iloc[val_id]["kepid"]).isdisjoint(temp.iloc[test_id]["kepid"])

# %% [markdown]
# In kích thước ba tập để kiểm tra số dòng và số đặc trưng. Do chia theo nhóm, tỷ lệ số dòng thực tế có thể không đúng chính xác 70/15/15.

# %%
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# %% [markdown]
# The split uses `kepid`, so objects from the same star cannot appear in two
# different sets.

# %% [markdown]
# ## 11. Before and after scaling

# %% [markdown]
# Chuyển ba biến số train sang dạng dài bằng melt rồi vẽ boxplot trước chuẩn hóa. Các thang đo khác nhau có thể khiến một biến lấn át biểu đồ.

# %%
scale_cols = ["koi_period", "koi_depth", "koi_steff"]
before = X_train[scale_cols].melt(var_name="Feature", value_name="Value").dropna()
plt.figure(figsize=(9, 5.6))
sns.boxplot(data=before, x="Feature", y="Value", color="#f1c9b2", showfliers=False)
plt.title("Before scaling: feature magnitudes are incompatible")
plt.xlabel("")
plt.ylabel("Raw value")
save_plot("10_before_scaling.png")

# %% [markdown]
# Trên bản sao train, điền thiếu bằng trung vị và fit StandardScaler để chuẩn hóa; sau đó chuyển về dạng dài. Chỉ dùng train để học trung vị và thang đo, tránh rò rỉ.

# %%
scale_data = X_train[scale_cols].copy()
scale_data = scale_data.fillna(scale_data.median())
scale_data = pd.DataFrame(StandardScaler().fit_transform(scale_data), columns=scale_cols)
after = scale_data.melt(var_name="Feature", value_name="Value")

# %% [markdown]
# Vẽ lại ba biến sau chuẩn hóa để so sánh với trước đó. Giá trị chuẩn hóa biểu thị khoảng cách theo độ lệch chuẩn, không còn đơn vị gốc.

# %%
plt.figure(figsize=(9, 5.6))
sns.boxplot(data=after, x="Feature", y="Value", color="#a9cfc1", showfliers=False)
plt.title("After standardization: features share a comparable scale")
plt.xlabel("")
plt.ylabel("Standard deviations")
save_plot("11_after_scaling.png")

# %% [markdown]
# ## 12. Logistic Regression

# %% [markdown]
# Tạo bộ tiền xử lý cho Logistic Regression: cột số được điền trung vị và chuẩn hóa; cột danh mục được điền giá trị thường gặp rồi one-hot. handle_unknown cho phép gặp danh mục mới khi dự đoán.

# %%
log_prep = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
    (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), category_cols),
)

# %% [markdown]
# Nối tiền xử lý với Logistic Regression, fit toàn bộ pipeline trên train rồi dự đoán validation. Chỉ bước fit được học tham số từ train.

# %%
log_model = make_pipeline(log_prep, LogisticRegression(max_iter=2500, random_state=42))
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_val)

# %% [markdown]
# Tính accuracy, balanced accuracy và macro F1 trên validation. Balanced accuracy và macro F1 giúp quan sát tốt hơn khi các lớp có kích thước khác nhau.

# %%
print("Logistic accuracy:", round(accuracy_score(y_val, log_pred), 4))
print("Logistic balanced accuracy:", round(balanced_accuracy_score(y_val, log_pred), 4))
print("Logistic macro F1:", round(f1_score(y_val, log_pred, average="macro"), 4))

# %% [markdown]
# ## 13. Random Forest

# %% [markdown]
# Tạo pipeline cho Random Forest: điền thiếu, dùng RobustScaler cho biến số và one-hot cho biến phân loại. RobustScaler dựa trên trung vị và IQR, ít nhạy với cực trị.

# %%
forest_prep = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), RobustScaler()), numeric_cols),
    (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), category_cols),
)

# %% [markdown]
# Tạo rừng gồm 350 cây, giới hạn số mẫu tối thiểu ở lá và dùng trọng số lớp cân bằng; fit trên train rồi dự đoán validation. random_state cố định phép lấy mẫu ngẫu nhiên.

# %%
forest = RandomForestClassifier(n_estimators=350, min_samples_leaf=2,
                                class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)
forest_model = make_pipeline(forest_prep, forest)
forest_model.fit(X_train, y_train)
forest_pred = forest_model.predict(X_val)

# %% [markdown]
# Tính cùng ba thước đo trên validation cho Random Forest. Dùng cùng tập và cùng thước đo để so sánh với Logistic Regression công bằng hơn.

# %%
print("Forest accuracy:", round(accuracy_score(y_val, forest_pred), 4))
print("Forest balanced accuracy:", round(balanced_accuracy_score(y_val, forest_pred), 4))
print("Forest macro F1:", round(f1_score(y_val, forest_pred, average="macro"), 4))

# %% [markdown]
# ## 14. Compare models

# %% [markdown]
# Ghép kết quả validation của hai mô hình thành bảng: mỗi hàng là mô hình, mỗi cột là một thước đo. So sánh các giá trị trước khi đọc kết quả test.

# %%
validation_results = pd.DataFrame({
    "accuracy": [accuracy_score(y_val, forest_pred), accuracy_score(y_val, log_pred)],
    "balanced_accuracy": [balanced_accuracy_score(y_val, forest_pred), balanced_accuracy_score(y_val, log_pred)],
    "macro_f1": [f1_score(y_val, forest_pred, average="macro"), f1_score(y_val, log_pred, average="macro")],
}, index=["Balanced Random Forest", "Logistic baseline"])
validation_results.index.name = "model"
validation_results.round(4)

# %% [markdown]
# Đổi bảng so sánh từ dạng rộng sang dạng dài bằng reset_index và melt để Seaborn vẽ nhóm cột theo từng thước đo.

# %%
plot_scores = validation_results.reset_index()
plot_scores = plot_scores.melt(id_vars="model", var_name="metric", value_name="score")

# %% [markdown]
# Vẽ và ghi điểm validation của hai mô hình theo từng metric. Chiều cao cột là điểm trong khoảng 0–1; hình lưu để đưa vào báo cáo.

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

# %% [markdown]
# Dùng Random Forest đã chọn để dự đoán X_test và in ba thước đo cuối cùng. Kết quả này mô tả hiệu quả trên tập chưa dùng để fit mô hình.

# %%
test_pred = forest_model.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, test_pred), 4))
print("Test balanced accuracy:", round(balanced_accuracy_score(y_test, test_pred), 4))
print("Test macro F1:", round(f1_score(y_test, test_pred, average="macro"), 4))

# %% [markdown]
# In classification_report gồm precision, recall, F1 và support của từng lớp. Support là số mẫu thật của lớp trong tập test.

# %%
print(classification_report(y_test, test_pred))

# %% [markdown]
# Vẽ confusion matrix chuẩn hóa theo hàng: hàng là nhãn thật, cột là nhãn dự đoán. Đường chéo thể hiện tỷ lệ dự đoán đúng cho từng lớp.

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
# The confusion matrix covers the two known classes; CANDIDATE objects are excluded.

# %% [markdown]
# ## 16. Feature importance

# %% [markdown]
# Tính permutation importance trên test: lần lượt xáo trộn một đặc trưng và đo mức giảm macro F1, lặp 8 lần. Đây là phân tích sau đánh giá, không dùng để điều chỉnh mô hình trên test.

# %%
result = permutation_importance(
    forest_model, X_test, y_test, n_repeats=8,
    random_state=42, scoring="f1_macro", n_jobs=-1,
)

# %% [markdown]
# Tạo bảng tên đặc trưng, độ quan trọng trung bình và độ lệch giữa các lần xáo trộn; sắp xếp từ ảnh hưởng lớn đến nhỏ.

# %%
importance = pd.DataFrame({
    "feature": features,
    "importance": result.importances_mean,
    "std": result.importances_std,
})
importance = importance.sort_values("importance", ascending=False)
importance.round(4)

# %% [markdown]
# Vẽ 10 đặc trưng có permutation importance cao nhất, kèm thanh sai lệch. Mức giảm F1 lớn cho biết mô hình phụ thuộc nhiều hơn vào biến đó trong phép đo này.

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
# - The final test macro F1 is reported above for the two known classes.
# - Permutation importance ranks features for the fitted binary model.
# - The model supports analysis but cannot confirm a real planet.
