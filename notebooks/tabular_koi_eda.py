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
# ### Feature Descriptions & Domain Grouping
#
# Below is the feature catalog grouping all 30 Kepler KOI attributes into 6 domain categories with their data types, missing rates, modeling roles, and detailed Vietnamese descriptions:
# 1. **Identification & Disposition** (`kepid`, `kepoi_name`, `kepler_name`, `koi_disposition`, `koi_pdisposition`, `koi_score`)
# 2. **Vetting Flags** (`koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec`)
# 3. **Transit & Orbital Properties** (`koi_period`, `koi_period_err1`, `koi_period_err2`, `koi_impact`, `koi_duration`, `koi_depth`, `koi_model_snr`)
# 4. **Planetary Properties** (`koi_prad`, `koi_prad_err1`, `koi_prad_err2`, `koi_teq`, `koi_insol`)
# 5. **Stellar Properties** (`koi_steff`, `koi_slogg`, `koi_srad`, `koi_kepmag`)
# 6. **Coordinates & Pipeline Metadata** (`ra`, `dec`, `koi_tce_plnt_num`, `koi_tce_delivname`)

# %%
feature_catalog = [
    # 1. Identification & Disposition
    {"Group": "1. Identification & Disposition", "Column": "kepid", "Role": "Group ID", "Description": "Mã định danh duy nhất của sao chủ trong Danh mục Kepler Input Catalog (KIC). Dùng để chia nhóm sao (group split)."},
    {"Group": "1. Identification & Disposition", "Column": "kepoi_name", "Role": "Identifier", "Description": "Tên định danh theo dõi của ứng viên Kepler Object of Interest (KOI) (ví dụ: K00752.01)."},
    {"Group": "1. Identification & Disposition", "Column": "kepler_name", "Role": "Target Leakage (Drop)", "Description": "Tên định danh chính thức của hành tinh Kepler đã xác nhận (chỉ gán cho CONFIRMED, rò rỉ nhãn 100%)."},
    {"Group": "1. Identification & Disposition", "Column": "koi_disposition", "Role": "Target", "Description": "Nhãn thẩm định cuối cùng của NASA Exoplanet Archive (CONFIRMED, CANDIDATE, FALSE POSITIVE)."},
    {"Group": "1. Identification & Disposition", "Column": "koi_pdisposition", "Role": "Target Leakage (Drop)", "Description": "Nhãn phân loại sơ bộ của pipeline tự động Kepler chỉ dựa trên dữ liệu trắc quang Kepler."},
    {"Group": "1. Identification & Disposition", "Column": "koi_score", "Role": "Target Leakage (Drop)", "Description": "Điểm xác suất tin cậy của thuật toán tự động Robovetter trong khoảng từ 0.0 đến 1.0."},

    # 2. Vetting Flags
    {"Group": "2. Vetting Flags", "Column": "koi_fpflag_nt", "Role": "Target Leakage (Drop)", "Description": "Cờ Not-Transit-Like (0 hoặc 1): Đường cong ánh sáng không mang hình dạng hành tinh quá cảnh (nhiễu thiết bị hoặc biến quang sao)."},
    {"Group": "2. Vetting Flags", "Column": "koi_fpflag_ss", "Role": "Target Leakage (Drop)", "Description": "Cờ Stellar-Eclipse (0 hoặc 1): Tín hiệu giảm sáng thực chất là do hệ sao đôi che khuất lẫn nhau (Eclipsing Binary)."},
    {"Group": "2. Vetting Flags", "Column": "koi_fpflag_co", "Role": "Target Leakage (Drop)", "Description": "Cờ Centroid-Offset (0 hoặc 1): Trọng tâm nguồn sáng bị lệch, nguồn gốc giảm sáng từ sao nền lân cận."},
    {"Group": "2. Vetting Flags", "Column": "koi_fpflag_ec", "Role": "Target Leakage (Drop)", "Description": "Cờ Ephemeris-Match Contamination (0 hoặc 1): Trùng khớp chu kỳ và thời điểm quá cảnh với tạp âm hoặc hệ sao đôi xung quanh."},

    # 3. Transit & Orbital Properties
    {"Group": "3. Transit & Orbital Properties", "Column": "koi_period", "Role": "Numerical Feature", "Description": "Chu kỳ quỹ đạo: Khoảng thời gian giữa hai lần quá cảnh liên tiếp (đơn vị: Ngày)."},
    {"Group": "3. Transit & Orbital Properties", "Column": "koi_period_err1", "Role": "Numerical Feature", "Description": "Độ không đảm bảo đo trên (+) của chu kỳ quỹ đạo (đơn vị: Ngày)."},
    {"Group": "3. Transit & Orbital Properties", "Column": "koi_period_err2", "Role": "Numerical Feature", "Description": "Độ không đảm bảo đo dưới (-) của chu kỳ quỹ đạo (đơn vị: Ngày)."},
    {"Group": "3. Transit & Orbital Properties", "Column": "koi_impact", "Role": "Numerical Feature", "Description": "Tham số va chạm: Khoảng cách hình chiếu giữa tâm sao và tâm quỹ đạo hành tinh lúc cực đại, chuẩn hóa theo bán kính sao."},
    {"Group": "3. Transit & Orbital Properties", "Column": "koi_duration", "Role": "Numerical Feature", "Description": "Thời gian quá cảnh: Khoảng thời gian từ lần tiếp xúc đầu tiên đến lần tiếp xúc cuối cùng (đơn vị: Giờ)."},
    {"Group": "3. Transit & Orbital Properties", "Column": "koi_depth", "Role": "Numerical Feature", "Description": "Độ sâu quá cảnh: Tỷ lệ quang thông của sao bị sụt giảm tại điểm cực tiểu quá cảnh (đơn vị: ppm - phần triệu)."},
    {"Group": "3. Transit & Orbital Properties", "Column": "koi_model_snr", "Role": "Numerical Feature", "Description": "Tỷ số tín hiệu trên nhiễu (SNR) của mô hình: Tỷ số giữa độ sâu quá cảnh và độ bất định trắc quang trung bình."},

    # 4. Planetary Properties
    {"Group": "4. Planetary Properties", "Column": "koi_prad", "Role": "Numerical Feature", "Description": "Bán kính hành tinh: Bán kính tính toán của ứng viên hành tinh chuẩn hóa theo bán kính Trái Đất (đơn vị: R_Earth)."},
    {"Group": "4. Planetary Properties", "Column": "koi_prad_err1", "Role": "Numerical Feature", "Description": "Độ không đảm bảo đo trên (+) của bán kính hành tinh (đơn vị: R_Earth)."},
    {"Group": "4. Planetary Properties", "Column": "koi_prad_err2", "Role": "Numerical Feature", "Description": "Độ không đảm bảo đo dưới (-) của bán kính hành tinh (đơn vị: R_Earth)."},
    {"Group": "4. Planetary Properties", "Column": "koi_teq", "Role": "Numerical Feature", "Description": "Nhiệt độ cân bằng: Nhiệt độ cân bằng bức xạ lý thuyết của hành tinh theo mô hình vật đen (đơn vị: Kelvin)."},
    {"Group": "4. Planetary Properties", "Column": "koi_insol", "Role": "Numerical Feature", "Description": "Mật độ thông lượng bức xạ: Lượng bức xạ nhận từ sao chủ chuẩn hóa theo mức Trái Đất nhận từ Mặt Trời (đơn vị: F_Earth)."},

    # 5. Stellar Properties
    {"Group": "5. Stellar Properties", "Column": "koi_steff", "Role": "Numerical Feature", "Description": "Nhiệt độ hiệu dụng của sao: Nhiệt độ quang cầu của ngôi sao chủ (đơn vị: Kelvin)."},
    {"Group": "5. Stellar Properties", "Column": "koi_slogg", "Role": "Numerical Feature", "Description": "Trọng trường bề mặt sao: Logarit cơ số 10 của gia tốc trọng trường tại bề mặt sao (đơn vị: log10(cm/s²))."},
    {"Group": "5. Stellar Properties", "Column": "koi_srad", "Role": "Numerical Feature", "Description": "Bán kính sao chủ: Bán kính quang cầu của sao chủ chuẩn hóa theo bán kính Mặt Trời (đơn vị: R_Sun)."},
    {"Group": "5. Stellar Properties", "Column": "koi_kepmag", "Role": "Numerical Feature", "Description": "Cấp sao Kepler: Độ sáng biểu kiến của ngôi sao đo qua dải thông quang học của kính viễn vọng Kepler (Magnitude)."},

    # 6. Coordinates & Pipeline Metadata
    {"Group": "6. Coordinates & Pipeline Metadata", "Column": "ra", "Role": "Numerical Feature", "Description": "Xích kinh: Tọa độ thiên văn ngang của sao trên thiên cầu theo hệ J2000 (đơn vị: độ)."},
    {"Group": "6. Coordinates & Pipeline Metadata", "Column": "dec", "Role": "Numerical Feature", "Description": "Xích vĩ: Tọa độ thiên văn dọc của sao trên thiên cầu theo hệ J2000 (đơn vị: độ)."},
    {"Group": "6. Coordinates & Pipeline Metadata", "Column": "koi_tce_plnt_num", "Role": "Numerical Feature", "Description": "Số thứ tự ứng viên hành tinh trong hệ sự kiện vượt ngưỡng TCE (Threshold Crossing Event)."},
    {"Group": "6. Coordinates & Pipeline Metadata", "Column": "koi_tce_delivname", "Role": "Categorical Feature", "Description": "Tên phiên bản bàn giao dữ liệu của pipeline sự kiện vượt ngưỡng TCE (ví dụ: q1_q17_dr25_tce)."},
]

feature_df = pd.DataFrame(feature_catalog)
feature_df["Data Type"] = feature_df["Column"].apply(lambda c: str(df[c].dtype))
feature_df["Missing Count"] = feature_df["Column"].apply(lambda c: int(df[c].isnull().sum()))
feature_df["Missing Rate (%)"] = feature_df["Column"].apply(lambda c: round(df[c].isnull().mean() * 100, 2))

# Display detailed feature catalog
pd.set_option("display.max_rows", 35)
pd.set_option("display.max_colwidth", None)
feature_df[["Group", "Column", "Data Type", "Missing Count", "Missing Rate (%)", "Role", "Description"]]

# %% [markdown]
# Visualize feature distribution across domain groups and modeling roles:

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Subplot 1: Feature count by domain group
group_counts = feature_df["Group"].value_counts().sort_values(ascending=True)
group_colors = ["#173f4c", "#1f5f73", "#2a809b", "#efae00", "#d35b12", "#8b3807"]
bars1 = ax1.barh(group_counts.index, group_counts.values, color=group_colors[:len(group_counts)], height=0.6)
ax1.set_title("Feature Count by Astronomical Domain Group", fontsize=12, fontweight="bold", pad=12)
ax1.set_xlabel("Number of Features", fontsize=10)
ax1.bar_label(bars1, padding=5, fontsize=10, fontweight="bold")
ax1.grid(axis="x", linestyle="--", alpha=0.5)

# Subplot 2: Feature distribution by modeling role
role_counts = feature_df["Role"].value_counts().sort_values(ascending=True)
role_colors = sns.color_palette("Set2", len(role_counts))
bars2 = ax2.barh(role_counts.index, role_counts.values, color=role_colors, height=0.6)
ax2.set_title("Feature Distribution by Modeling Role", fontsize=12, fontweight="bold", pad=12)
ax2.set_xlabel("Number of Features", fontsize=10)
ax2.bar_label(bars2, padding=5, fontsize=10, fontweight="bold")
ax2.grid(axis="x", linestyle="--", alpha=0.5)

plt.suptitle("KEPLER KOI FEATURE CATALOG & DOMAIN GROUPING", fontsize=14, fontweight="bold", y=1.02)
save_plot("00_feature_groups_overview.png")

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
# ## 3. Missing values

# %% [markdown]
# Đếm số giá trị thiếu theo từng cột, đổi sang phần trăm, chỉ giữ cột có thiếu và sắp xếp giảm dần. Bảng này giúp ưu tiên kiểm tra các cột thiếu nhiều.

# %%
missing = pd.DataFrame({"Count": df.isnull().sum()})
missing["Percent"] = missing["Count"] / len(df) * 100
missing = missing[missing["Count"] > 0]
missing = missing.sort_values("Percent", ascending=False)
missing.round(2)

# %% [markdown]
# Vẽ tối đa 15 cột thiếu nhiều nhất. Chiều dài thanh là tỷ lệ thiếu; hình được lưu bằng save_plot.

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
# ## 4. Numerical features distribution

# %% [markdown]
# Khảo sát phân phối của chu kỳ quỹ đạo (`koi_period`) trên thang log (chỉ lấy giá trị dương, không thiếu):

# %%
period = df.loc[df["koi_period"] > 0, "koi_period"].dropna()
plt.figure(figsize=(9, 5.6))
sns.histplot(period, bins=60, log_scale=True, color="#173f4c")
plt.title("Orbital period distribution")
plt.xlabel("Orbital period (days, log scale)")
plt.ylabel("Sample")
save_plot("03_orbital_period_distribution.png")

# %% [markdown]
# Khảo sát phân phối của bán kính hành tinh ước tính (`koi_prad`) trên thang log:

# %%
radius = df.loc[df["koi_prad"] > 0, "koi_prad"].dropna()
plt.figure(figsize=(9, 5.6))
sns.histplot(radius, bins=60, log_scale=True, color="#d35b12")
plt.title("Estimated planetary radius distribution")
plt.xlabel("Planet radius (Earth radii, log scale)")
plt.ylabel("Sample")
save_plot("04_planet_radius_distribution.png")

# %% [markdown]
# Khảo sát phân phối nhiệt độ hiệu dụng của sao chủ (`koi_steff`) và đánh dấu trung vị bằng đường nét đứt:

# %%
plt.figure(figsize=(9, 5.6))
sns.histplot(df["koi_steff"].dropna(), bins=45, color="#efae00")
plt.axvline(df["koi_steff"].median(), color="#173f4c", linestyle="--", label="Median")
plt.title("Host-star effective temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("Sample")
plt.legend()
save_plot("05_stellar_temperature_distribution.png")

# %% [markdown]
# ### Phân phối toàn bộ 20 đặc trưng số liên tục (All 20 Continuous Numerical Features)
#
# Dưới đây là lưới đồ thị phân phối (Histograms) của toàn bộ 20 đặc trưng số liên tục trong tập dữ liệu (bao gồm thông số quỹ đạo, bán kính, nhiệt độ, quang thông, độ sâu quá cảnh, tọa độ và sai số đo đạc). Mỗi đồ thị thể hiện dải phân vị 1%–99% cùng đường trung vị (màu cam nét đứt):

# %%
all_numeric_cols = [
    ("koi_period", "Orbital period (days, log)"),
    ("koi_period_err1", "Period err1 (days)"),
    ("koi_period_err2", "Period err2 (days)"),
    ("koi_impact", "Impact parameter"),
    ("koi_duration", "Transit duration (hours)"),
    ("koi_depth", "Transit depth (ppm, log)"),
    ("koi_prad", "Planet radius (R_Earth, log)"),
    ("koi_prad_err1", "Planet radius err1 (R_Earth)"),
    ("koi_prad_err2", "Planet radius err2 (R_Earth)"),
    ("koi_teq", "Equilibrium temp (K)"),
    ("koi_insol", "Insolation flux (log)"),
    ("koi_model_snr", "Transit model SNR (log)"),
    ("koi_tce_plnt_num", "TCE planet number"),
    ("koi_steff", "Stellar eff temp (K)"),
    ("koi_slogg", "Stellar log(g)"),
    ("koi_srad", "Stellar radius (R_Sun, log)"),
    ("ra", "Right Ascension (deg)"),
    ("dec", "Declination (deg)"),
    ("koi_kepmag", "Kepler magnitude"),
    ("koi_score", "Robovetter score [0-1]"),
]

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
for ax, (column, label) in zip(axes.flat, all_numeric_cols):
    values = df[column].dropna()
    lower, upper = values.quantile([0.01, 0.99])
    shown = values[values.between(lower, upper)]
    if column in {"koi_period", "koi_depth", "koi_prad", "koi_insol", "koi_model_snr", "koi_srad"}:
        shown = shown[shown > 0]
        ax.hist(shown, bins=np.geomspace(shown.min(), shown.max(), 35),
                color="#173f4c", edgecolor="white")
        ax.set_xscale("log")
    else:
        ax.hist(shown, bins=35, color="#173f4c", edgecolor="white")
    median_val = values.median()
    ax.axvline(median_val, color="#d35b12", linestyle="--", linewidth=1.2)
    ax.set_title(column, fontsize=11, fontweight="bold")
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("Sample", fontsize=9)
    ax.text(0.97, 0.93, f"Median: {median_val:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#d35b12", fontweight="bold")

plt.suptitle("DISTRIBUTIONS OF ALL 20 CONTINUOUS NUMERICAL FEATURES", fontsize=15, fontweight="bold", y=1.01)
save_plot("16_additional_numeric_distributions.png")

# %% [markdown]
# ### Phân phối các đặc trưng số quan trọng trong học máy (Key Machine Learning Feature Distributions)
#
# Biểu đồ cột phân phối tần suất (binned histograms & density) của 6 đặc trưng số quan trọng nhất đối với mô hình học máy phân loại ngoại hành tinh (được kiểm chứng qua Permutation Importance ở phần sau). Các đặc trưng này phản ánh các đặc tính vật lý cốt lõi phân biệt giữa hành tinh thực sự (`CONFIRMED`) và các dạng dương tính giả (`FALSE POSITIVE` do sao đôi che khuất hoặc nhiễu thiết bị):

# %%
ml_key_features = [
    ("koi_prad", "Planetary Radius (Earth radii, log scale)", True, 0.1, 100),
    ("koi_period", "Orbital Period (days, log scale)", True, 0.5, 500),
    ("koi_model_snr", "Transit Model SNR (log scale)", True, 5, 2000),
    ("koi_depth", "Transit Depth (ppm, log scale)", True, 10, 100000),
    ("koi_duration", "Transit Duration (hours)", False, 0.5, 20),
    ("koi_impact", "Impact Parameter", False, 0.0, 1.5),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
palette = {"CONFIRMED": "#173f4c", "FALSE POSITIVE": "#d35b12", "CANDIDATE": "#efae00"}

for ax, (col, label, use_log, vmin, vmax) in zip(axes.flat, ml_key_features):
    sub_df = df[df[col].notnull()].copy()
    if use_log:
        sub_df = sub_df[(sub_df[col] >= vmin) & (sub_df[col] <= vmax)]
        bins = np.geomspace(sub_df[col].min(), sub_df[col].max(), 35)
        sns.histplot(data=sub_df, x=col, hue="koi_disposition", bins=bins,
                     element="bars", common_norm=False, stat="density",
                     palette=palette, alpha=0.55, edgecolor="white", ax=ax)
        ax.set_xscale("log")
    else:
        q_low, q_high = sub_df[col].quantile([0.005, 0.995])
        sub_df = sub_df[sub_df[col].between(q_low, q_high)]
        sns.histplot(data=sub_df, x=col, hue="koi_disposition", bins=35,
                     element="bars", common_norm=False, stat="density",
                     palette=palette, alpha=0.55, edgecolor="white", ax=ax)
        
    median_val = df[col].median()
    ax.axvline(median_val, color="#333333", linestyle="--", linewidth=1.5)
    ax.set_title(f"Distribution of {col}\n({label})", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)

plt.suptitle("KEY MACHINE LEARNING FEATURE DISTRIBUTIONS ACROSS DISPOSITIONS", fontsize=14, fontweight="bold", y=1.01)
save_plot("17_key_ml_features_distribution.png")

# %% [markdown]
# **Ý nghĩa vật lý và vai trò trong học máy của các đặc trưng quan trọng:**
#
# 1. **Bán kính hành tinh (`koi_prad` — Importance ~ 0.11, Quan trọng nhất):**
#    - *Phân phối:* Phân bố lệch phải cực mạnh. Đa phần các ngoại hành tinh đã xác nhận (`CONFIRMED`) tập trung dày đặc ở kích thước nhỏ ($1 - 4 R_{Earth}$, gồm các hành tinh đất đá siêu Trái Đất và tiểu Hải Vương Tinh).
#    - *Ý nghĩa ML:* Đây là ranh giới vật lý sắc bén nhất. Các ứng viên có bán kính ước tính vượt quá $20 - 30 R_{Earth}$ hầu như chắc chắn là sao thứ cấp trong hệ sao đôi che khuất (`FALSE POSITIVE`), vì ngay cả hành tinh khí khổng lồ lớn nhất (như Sao Mộc) cũng chỉ đạt cỡ $11 - 14 R_{Earth}$.
#
# 2. **Chu kỳ quỹ đạo (`koi_period` — Importance ~ 0.10):**
#    - *Phân phối:* Trải rộng trên thang log từ dưới 1 ngày đến hàng trăm ngày, đạt đỉnh mật độ trong khoảng 3 đến 20 ngày.
#    - *Ý nghĩa ML:* Các hệ sao đôi che khuất giả mạo thường bị khóa thủy triều ở chu kỳ cực ngắn ($< 2 - 3$ ngày), trong khi hành tinh thực sự phân bố dàn trải hơn và tuân theo định luật cơ học Kepler III.
#
# 3. **Tỷ số tín hiệu trên nhiễu (`koi_model_snr` — Importance ~ 0.06):**
#    - *Phân phối:* Đa số tín hiệu ứng viên tập trung ở vùng SNR từ 10 đến 50, với phần đuôi kéo dài tới hàng nghìn.
#    - *Ý nghĩa ML:* Đo lường độ tin cậy thống kê của tín hiệu quá cảnh so với nhiễu nền trắc quang. Tín hiệu quá yếu ($SNR < 10$) thường là nhiễu thiết bị quang học; tín hiệu quá mạnh bất thường ($SNR > 500$) thường phản ánh độ sụt sáng sâu của sao đôi.
#
# 4. **Độ sâu quá cảnh (`koi_depth` — Importance ~ 0.03):**
#    - *Phân phối:* Tỷ lệ quang thông sao bị che khuất ($\\Delta F / F \\approx (R_p / R_*)^2$). Hành tinh thực sự làm mờ sao mẹ rất ít (chỉ từ vài chục đến vài nghìn ppm, tương đương $0.005\\% - 0.3\\%$).
#    - *Ý nghĩa ML:* Các sự kiện sụt giảm độ sáng lớn ($> 10,000 - 50,000$ ppm, tức giảm $1\\% - 5\\%$ ánh sáng sao) vượt xa kích thước tối đa của một hành tinh, cung cấp cho mô hình bằng chứng phân loại trực tiếp để đánh dấu `FALSE POSITIVE`.
#
# 5. **Thời gian quá cảnh (`koi_duration` — Importance ~ 0.06):**
#    - *Phân phối:* Phân bố dạng chuông lệch phải, tập trung chủ yếu trong khoảng 2 đến 6 giờ.
#    - *Ý nghĩa ML:* Bị ràng buộc chặt chẽ bởi chu kỳ quỹ đạo và kích thước sao chủ ($T_{dur} \\propto P^{1/3}$). Thời lượng quá cảnh quá ngắn hoặc quá dài so với chu kỳ quỹ đạo dự kiến là đặc trưng chỉ điểm của tín hiệu nhiễu hoặc sai lệch tâm quang học.
#
# 6. **Tham số va chạm (`koi_impact` — Importance ~ 0.06):**
#    - *Phân phối:* Khoảng cách hình chiếu từ tâm sao lúc quá cảnh ($b \\in [0, 1]$ là cắt qua đĩa sao, $b > 1$ là trượt mép / grazing).
#    - *Ý nghĩa ML:* Các sự kiện trượt mép ($b \\approx 1$ hoặc $b > 1$) tạo ra đường cong ánh sáng chữ V thoai thoải, là trường hợp gây nhầm lẫn nhiều nhất giữa hành tinh trượt mép và hệ sao đôi che khuất một phần.

# %% [markdown]
# ## 5. Categorical features distribution

# %% [markdown]
# Khảo sát phân phối của toàn bộ các đặc trưng phân loại và chẩn đoán trong tập dữ liệu (phiên bản TCE delivery, nhãn pipeline sơ bộ `koi_pdisposition`, tình trạng đặt tên hành tinh `kepler_name` và 4 cờ chẩn đoán vetting false positive flags):

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Subplot 1: TCE Delivery Catalogue
deliv = df["koi_tce_delivname"].fillna("Missing").value_counts()
ax1 = axes[0, 0]
sns.barplot(x=deliv.values, y=deliv.index, hue=deliv.index, palette=["#173f4c", "#d35b12", "#2a809b", "#efae00"], legend=False, ax=ax1)
for bar in ax1.containers:
    ax1.bar_label(bar, fmt="{:,.0f}", padding=5, fontsize=9)
ax1.set_title("TCE Delivery Catalogue Version", fontsize=12, fontweight="bold")
ax1.set_xlabel("Sample count", fontsize=10)

# Subplot 2: Kepler Pipeline Disposition (koi_pdisposition)
pdisp = df["koi_pdisposition"].value_counts()
ax2 = axes[0, 1]
sns.barplot(x=pdisp.index, y=pdisp.values, hue=pdisp.index, palette=["#d35b12", "#efae00"], legend=False, ax=ax2)
for bar in ax2.containers:
    ax2.bar_label(bar, fmt="{:,.0f}", padding=5, fontsize=9)
ax2.set_title("Kepler Pipeline Disposition (koi_pdisposition)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Sample count", fontsize=10)

# Subplot 3: Official Kepler Planet Name Status
kname_status = pd.Series({
    "Named (Confirmed Planet)": df["kepler_name"].notnull().sum(),
    "Unnamed (No Kepler Name)": df["kepler_name"].isnull().sum(),
})
ax3 = axes[1, 0]
sns.barplot(x=kname_status.index, y=kname_status.values, hue=kname_status.index, palette=["#173f4c", "#999999"], legend=False, ax=ax3)
for bar in ax3.containers:
    ax3.bar_label(bar, fmt="{:,.0f}", padding=5, fontsize=9)
ax3.set_title("Official Kepler Planet Name Status", fontsize=12, fontweight="bold")
ax3.set_ylabel("Sample count", fontsize=10)

# Subplot 4: False Positive Vetting Flags Frequency
flag_counts = pd.Series({
    "Not-Transit-Like (nt)": (df["koi_fpflag_nt"] == 1).sum(),
    "Stellar-Eclipse (ss)": (df["koi_fpflag_ss"] == 1).sum(),
    "Centroid-Offset (co)": (df["koi_fpflag_co"] == 1).sum(),
    "Ephemeris-Match (ec)": (df["koi_fpflag_ec"] == 1).sum(),
})
ax4 = axes[1, 1]
sns.barplot(x=flag_counts.values, y=flag_counts.index, hue=flag_counts.index, palette=["#d35b12", "#e67e22", "#f39c12", "#d98880"], legend=False, ax=ax4)
for bar in ax4.containers:
    ax4.bar_label(bar, fmt="{:,.0f}", padding=5, fontsize=9)
ax4.set_title("False Positive Vetting Flags (Flag = 1 count)", fontsize=12, fontweight="bold")
ax4.set_xlabel("Positive flag count", fontsize=10)

plt.suptitle("CATEGORICAL & DIAGNOSTIC FEATURES DISTRIBUTION", fontsize=14, fontweight="bold", y=1.01)
save_plot("15_tce_delivery_distribution.png")

# %% [markdown]
# **Nhận xét về đặc trưng phân loại:**
# - `koi_tce_delivname`: Đợt bàn giao `q1_q17_dr25_tce` chiếm đại đa số (hơn 84% mẫu). Có 346 mẫu khuyết thiếu được mã hóa bằng "Missing" trong quá trình chuẩn bị dữ liệu.
# - `koi_pdisposition`, `kepler_name`, và 4 cờ `koi_fpflag_*`: Phản ánh kết quả chẩn đoán tự động của NASA Kepler. Do tiềm ẩn nguy cơ gây rò rỉ nhãn nghiêm trọng (Target Leakage), các trường này bắt buộc bị loại bỏ trước khi huấn luyện mô hình học máy.

# %% [markdown]
# ## 6. Target variable distribution

# %% [markdown]
# Phân tích biến mục tiêu `koi_disposition` (Nhãn thẩm định cuối cùng của NASA Exoplanet Archive). Đếm số lượng và tính tỷ lệ phần trăm phân bố trên toàn bộ 9.564 mẫu:

# %%
target_counts = df["koi_disposition"].value_counts()
target_percent = df["koi_disposition"].value_counts(normalize=True) * 100
pd.DataFrame({"Count": target_counts, "Percent": target_percent.round(2)})

# %% [markdown]
# Vẽ biểu đồ cột phân phối biến mục tiêu với số lượng mẫu hiển thị trên từng cột:

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
ax.set(title="Target Variable Distribution (koi_disposition)", xlabel="Target Disposition", ylabel="Sample Count")
for bar in ax.containers:
    ax.bar_label(bar, fmt="{:,.0f}", padding=4)
save_plot("01_target_distribution.png")

# %% [markdown]
# **Phân tích phân phối biến mục tiêu:**
# - **Mất cân bằng lớp (Class Imbalance):** `FALSE POSITIVE` là lớp chiếm đa số với 5.023 mẫu (52,52%), trong khi `CONFIRMED` chiếm 2.293 mẫu (23,97%) và `CANDIDATE` chiếm 2.248 mẫu (23,51%). Do dữ liệu mất cân bằng, độ chính xác thuần túy (Accuracy) không phản ánh đúng năng lực phân loại; các thước đo cân bằng như **Macro F1** và **Balanced Accuracy** là bắt buộc.
# - **Lý do loại trừ `CANDIDATE` trong bài toán học máy:** Nhãn `CANDIDATE` là các tín hiệu ứng viên chưa có kết luận thẩm định chắc chắn từ các quan sát kiểm chứng độc lập. Do đó, bài toán học máy nhị phân sẽ giữ lại hai lớp có nhãn rõ ràng: `CONFIRMED` (1 - ngoại hành tinh thực sự) và `FALSE POSITIVE` (0 - tín hiệu thiên thực sao đôi hoặc nhiễu), giúp mô hình học được ranh giới phân lớp vật lý tường minh.

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
