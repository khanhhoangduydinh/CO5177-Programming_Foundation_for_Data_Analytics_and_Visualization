# %% [markdown]
# # ViCTSD — Vietnamese Text Classification
#
# **Course:** CO5177 — Programming Foundation for Data Analytics and Visualization
#
# **Student:** Dinh Hoang Duy Khanh — 2670306
#
# **Group:** Trailblazer
#
# **Cách chạy:** Mở notebook trong Google Colab, chọn Runtime → Run all và chờ từng cell hoàn tất. Lần đầu cần Internet để tải dữ liệu. Đọc hướng dẫn tiếng Việt ngay trước mỗi cell và kiểm tra kết quả sau khi chạy.

# %% [markdown]
# This notebook studies Vietnamese social-media comments from ViCTSD. The main
# task is toxicity classification. Constructiveness is reported as a second task.

# %% [markdown]
# ## 0. Import libraries

# %% [markdown]
# Nạp thư viện đường dẫn, JSON, biểu thức chính quy, vẽ biểu đồ và xử lý bảng. Chúng hỗ trợ đọc dữ liệu, làm sạch văn bản và lưu kết quả.

# %%
from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% [markdown]
# Nạp TF-IDF, Logistic Regression, Naive Bayes và các thước đo phân loại. Các mô hình nhận đặc trưng số được chuyển từ văn bản.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB

# %% [markdown]
# Đặt giao diện và màu chung cho các biểu đồ. Thiết lập này không ảnh hưởng đến nhãn hoặc kết quả mô hình.

# %%
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180})
colors = ["#d35b12", "#efae00", "#173f4c"]

# %% [markdown]
# Xác định thư mục dự án và tạo reports/figures để lưu hình. Khi chạy trong Colab, đường dẫn dựa trên thư mục làm việc hiện tại.

# %%
project_folder = Path.cwd()
if project_folder.name == "notebooks":
    project_folder = project_folder.parent

figure_folder = project_folder / "reports" / "figures"
figure_folder.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# Định nghĩa save_plot để căn hình, lưu tệp và hiển thị. Việc đặt chung tên hàm giúp các cell biểu đồ có cùng cách lưu.

# %%
def save_plot(name):
    plt.tight_layout()
    plt.savefig(figure_folder / name, bbox_inches="tight", facecolor="white")
    plt.show()


# %% [markdown]
# ## 1. Load the official splits

# %% [markdown]
# Định nghĩa load_split: ưu tiên CSV trong data nếu có, nếu không thì đọc bản công khai từ repository ViCTSD. Tham số name là train, valid hoặc test.

# %%
def load_split(name):
    local_file = project_folder / "data" / f"victsd_{name}.csv"
    source_url = f"https://raw.githubusercontent.com/tarudesu/ViCTSD/main/ViCTSD_{name}.csv"
    return pd.read_csv(local_file if local_file.exists() else source_url)


# %% [markdown]
# Đọc ba phần chia chính thức thành train, valid và test. Việc giữ riêng ba tập hỗ trợ huấn luyện, lựa chọn và đánh giá theo từng bước.

# %%
train = load_split("train")
valid = load_split("valid")
test = load_split("test")

# %% [markdown]
# Bỏ cột chỉ số Unnamed: 0 trong từng bảng rồi xem vài dòng train. Cột này là chỉ số lưu kèm CSV, không phải nội dung bình luận.

# %%
for frame in [train, valid, test]:
    frame.drop(columns="Unnamed: 0", inplace=True)

train.head()

# %% [markdown]
# In kích thước từng tập và tổng số bình luận. shape gồm số dòng và số cột; tổng cho biết quy mô dữ liệu được sử dụng.

# %%
print("Train:", train.shape)
print("Validation:", valid.shape)
print("Test:", test.shape)
print("All comments:", len(train) + len(valid) + len(test))

# %% [markdown]
# Đếm giá trị thiếu theo cột và số dòng trùng hoàn toàn trong train. Kiểm tra này giúp nhận biết vấn đề trước khi làm sạch.

# %%
print("Missing values:")
print(train.isnull().sum())
print("Duplicate rows in train:", train.duplicated().sum())

# %% [markdown]
# ViCTSD has 10,000 human-annotated Vietnamese comments. The official split is
# 7,000 train, 2,000 validation, and 1,000 test comments.

# %% [markdown]
# ## 2. Dataset overview

# %% [markdown]
# Lập bảng số bình luận của train, validation và test. Bảng này mô tả cách chia có sẵn của nguồn dữ liệu.

# %%
split_sizes = pd.DataFrame({
    "Split": ["Train", "Validation", "Test"],
    "Comments": [len(train), len(valid), len(test)],
})
split_sizes

# %% [markdown]
# Vẽ số bình luận trong từng tập và ghi số trên mỗi cột. Biểu đồ thể hiện quy mô, không đánh giá chất lượng nhãn.

# %%
plt.figure(figsize=(7.8, 4.8))
ax = sns.barplot(data=split_sizes, x="Split", y="Comments", hue="Split", palette=colors, legend=False)
for bar in ax.containers:
    ax.bar_label(bar, fmt="{:,.0f}", padding=4)
ax.set(title="Official ViCTSD split", xlabel="", ylabel="Comments")
save_plot("text_01_split_sizes.png")

# %% [markdown]
# Gắn tên tập vào từng bảng rồi nối lại để mô tả toàn bộ dữ liệu. all_data phục vụ EDA; mô hình vẫn được fit trên train riêng.

# %%
train["Split"] = "Train"
valid["Split"] = "Validation"
test["Split"] = "Test"
all_data = pd.concat([train, valid, test], ignore_index=True)

# %% [markdown]
# Đếm số chủ đề và số bình luận trùng chính xác, rồi lập bảng chéo giữa hai nhãn. Bảng chéo cho biết số lượng từng cặp nhãn.

# %%
print("Topics:", all_data["Topic"].nunique())
print("Exact duplicate comments:", all_data["Comment"].duplicated().sum())
pd.crosstab(all_data["Constructiveness"], all_data["Toxicity"])

# %% [markdown]
# ## 3. Label and topic distributions

# %% [markdown]
# Dùng melt để đưa hai cột nhãn về dạng dài, sau đó ánh xạ 0 và 1 thành tên lớp chung cho biểu đồ.

# %%
label_data = all_data.melt(
    value_vars=["Constructiveness", "Toxicity"],
    var_name="Task", value_name="Label",
)
label_data["Class"] = label_data["Label"].map({0: "Class 0", 1: "Class 1"})

# %% [markdown]
# Vẽ số lượng lớp 0 và lớp 1 cho từng nhiệm vụ, kèm số trên cột. Từ đó xem nhãn nào mất cân bằng và chọn metric phù hợp.

# %%
plt.figure(figsize=(8.5, 5.2))
ax = sns.countplot(data=label_data, x="Task", hue="Class", palette=["#a9cfc1", "#d35b12"])
for bar in ax.containers:
    ax.bar_label(bar, fmt="{:,.0f}", padding=3)
ax.set(title="Label distribution for both tasks", xlabel="", ylabel="Comments")
save_plot("text_02_label_distribution.png")

# %% [markdown]
# Đếm số bình luận theo chủ đề và vẽ biểu đồ thanh. So sánh độ dài các thanh để kiểm tra mức phân bố chủ đề.

# %%
topic_counts = all_data["Topic"].value_counts().sort_values()
plt.figure(figsize=(9, 5.8))
sns.barplot(x=topic_counts.values, y=topic_counts.index, color="#173f4c")
plt.title("Comments in each news topic")
plt.xlabel("Comments")
plt.ylabel("")
save_plot("text_03_topic_distribution.png")

# %% [markdown]
# The dataset is balanced by topic (1,000 comments each) but toxicity is rare.
# This makes accuracy alone a misleading metric for the toxicity task.

# %% [markdown]
# ## 4. Clean text and audit duplicates

# %% [markdown]
# Định nghĩa clean_text: đổi chữ thường, bỏ URL và dấu không thuộc ký tự/tách từ, rồi thu gọn khoảng trắng. Đây là làm sạch đơn giản, không phải tách từ tiếng Việt chuyên sâu.

# %%
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# %% [markdown]
# Áp dụng clean_text cho ba tập; đồng thời đếm ký tự và số từ tách theo khoảng trắng. Sau đó nối bảng để trực quan thống kê.

# %%
for frame in [train, valid, test]:
    frame["CleanComment"] = frame["Comment"].map(clean_text)
    frame["Characters"] = frame["Comment"].str.len()
    frame["Words"] = frame["CleanComment"].str.split().str.len()

all_data = pd.concat([train, valid, test], ignore_index=True)

# %% [markdown]
# Tạo tập bình luận train đã chuẩn hóa và loại khỏi validation/test những nội dung trùng train. Phép lọc này không loại trùng giữa validation và test; trong bản dữ liệu đã kiểm tra còn một dòng test trùng validation.

# %%
known_comments = set(train["CleanComment"])
valid_eval = valid[~valid["CleanComment"].isin(known_comments)].reset_index(drop=True)
test_eval = test[~test["CleanComment"].isin(known_comments)].reset_index(drop=True)

# %% [markdown]
# In số dòng bị loại vì trùng train và kích thước validation/test còn lại. Điểm số về sau áp dụng cho phần được giữ lại, không phải toàn bộ tập gốc.

# %%
print("Validation comments removed for overlap:", len(valid) - len(valid_eval))
print("Test comments removed for overlap:", len(test) - len(test_eval))
print("Validation used for evaluation:", len(valid_eval))
print("Test used for evaluation:", len(test_eval))

# %% [markdown]
# Vẽ histogram độ dài bình luận theo nhãn độc hại; số từ dựa trên phép tách khoảng trắng đơn giản. Quan sát phân phối, không kết luận độ dài gây độc hại.

# %%
plt.figure(figsize=(9, 5.6))
sns.histplot(data=all_data, x="Words", hue="Toxicity", bins=45,
             palette=["#a9cfc1", "#d35b12"], element="step")
plt.title("Comment length by toxicity label")
plt.xlabel("Words after simple cleaning")
save_plot("text_04_comment_length.png")

# %% [markdown]
# Only simple cleaning is used: lowercase, remove URLs and punctuation, then
# collapse spaces. Titles and topics are not used as model inputs.

# %% [markdown]
# ## 5. TF-IDF features

# %% [markdown]
# Khai báo bộ TF-IDF theo từ, dùng unigram và bigram, giữ từ xuất hiện ít nhất hai lần và giới hạn 8.000 đặc trưng. sublinear_tf làm giảm ảnh hưởng của tần suất lớn.

# %%
word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), min_df=2, max_features=8000, sublinear_tf=True,
)

# %% [markdown]
# Fit bộ từ vựng và trọng số IDF chỉ trên train, sau đó transform validation và test. In số đặc trưng để kiểm tra kích thước biểu diễn.

# %%
X_train_word = word_vectorizer.fit_transform(train["CleanComment"])
X_valid_word = word_vectorizer.transform(valid_eval["CleanComment"])
X_test_word = word_vectorizer.transform(test_eval["CleanComment"])
print("Word TF-IDF features:", X_train_word.shape[1])

# %% [markdown]
# Khai báo TF-IDF theo chuỗi ký tự dài 3–5 trong ranh giới từ, tối đa 12.000 đặc trưng. Cách này mô tả mẫu ký tự thay vì chỉ nguyên từ.

# %%
char_vectorizer = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=12000, sublinear_tf=True,
)

# %% [markdown]
# Fit biểu diễn ký tự trên train và transform validation; in số đặc trưng. Không học từ vựng từ validation để tránh rò rỉ.

# %%
X_train_char = char_vectorizer.fit_transform(train["CleanComment"])
X_valid_char = char_vectorizer.transform(valid_eval["CleanComment"])
print("Character TF-IDF features:", X_train_char.shape[1])

# %% [markdown]
# ## 6. Words and phrases related to toxicity

# %% [markdown]
# Huấn luyện Logistic Regression cân bằng trọng số lớp trên TF-IDF theo từ rồi dự đoán validation. max_iter giới hạn số vòng tối ưu.

# %%
toxic_word_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
toxic_word_model.fit(X_train_word, train["Toxicity"])
toxic_valid_pred = toxic_word_model.predict(X_valid_word)

# %% [markdown]
# Lấy tên đặc trưng và hệ số của Logistic Regression; chọn các trọng số dương và âm lớn nhất. Hệ số mô tả liên hệ với nhãn trong mô hình này, không chứng minh nguyên nhân.

# %%
terms = word_vectorizer.get_feature_names_out()
weights = pd.DataFrame({"Term": terms, "Weight": toxic_word_model.coef_[0]})
toxic_terms = weights.nlargest(15, "Weight").sort_values("Weight")
safe_terms = weights.nsmallest(15, "Weight").sort_values("Weight", ascending=False)

# %% [markdown]
# Vẽ các từ/cụm từ có hệ số dương lớn nhất liên quan đến dự đoán độc hại. Đọc biểu đồ như tín hiệu mô hình học được, không phải đánh giá mọi ngữ cảnh.

# %%
plt.figure(figsize=(9, 6))
sns.barplot(data=toxic_terms, x="Weight", y="Term", color="#d35b12")
plt.title("TF-IDF terms most associated with toxic comments")
plt.xlabel("Logistic Regression weight")
plt.ylabel("")
save_plot("text_05_top_toxic_ngrams.png")

# %% [markdown]
# Vẽ các từ/cụm từ có hệ số âm nổi bật liên quan đến dự đoán không độc hại. Nội dung cần được diễn giải trong giới hạn dữ liệu và mô hình.

# %%
plt.figure(figsize=(9, 6))
sns.barplot(data=safe_terms, x="Weight", y="Term", color="#a9cfc1")
plt.title("TF-IDF terms most associated with non-toxic comments")
plt.xlabel("Logistic Regression weight")
plt.ylabel("")
save_plot("text_06_top_safe_ngrams.png")

# %% [markdown]
# These weights explain which n-grams push a comment toward toxic or non-toxic.
# They are model signals, not a list of words that always determine toxicity.

# %% [markdown]
# ## 7. Toxicity model comparison

# %% [markdown]
# Huấn luyện Multinomial Naive Bayes làm baseline trên TF-IDF từ và dự đoán validation. Đây là mô hình đơn giản để đối chiếu.

# %%
toxic_nb = MultinomialNB()
toxic_nb.fit(X_train_word, train["Toxicity"])
toxic_nb_pred = toxic_nb.predict(X_valid_word)

# %% [markdown]
# Huấn luyện Logistic Regression trên TF-IDF ký tự và dự đoán validation. So sánh với mô hình từ cho biết biểu diễn nào phù hợp hơn theo metric được chọn.

# %%
toxic_char_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
toxic_char_model.fit(X_train_char, train["Toxicity"])
toxic_char_pred = toxic_char_model.predict(X_valid_char)

# %% [markdown]
# Định nghĩa score_model trả về accuracy, balanced accuracy, F1 của lớp 1 và macro F1. Truyền nhãn thật và nhãn dự đoán để tính cùng một cách cho mọi mô hình.

# %%
def score_model(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        balanced_accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, pos_label=1),
        f1_score(y_true, y_pred, average="macro"),
    ]


# %% [markdown]
# Tạo bảng điểm validation của ba phương án theo bốn metric. Chú ý accuracy cao không bảo đảm F1 của lớp độc hại cao khi dữ liệu lệch lớp.

# %%
toxic_scores = pd.DataFrame(
    [score_model(valid_eval["Toxicity"], toxic_nb_pred),
     score_model(valid_eval["Toxicity"], toxic_valid_pred),
     score_model(valid_eval["Toxicity"], toxic_char_pred)],
    columns=["Accuracy", "Balanced accuracy", "F1 toxic", "Macro F1"],
    index=["Naive Bayes", "Word TF-IDF + Logistic", "Char TF-IDF + Logistic"],
)
toxic_scores.round(4)

# %% [markdown]
# Chuyển bảng điểm sang dạng dài và vẽ biểu đồ so sánh ba mô hình. Các cột chỉ dùng kết quả validation, không dùng test để chọn.

# %%
score_plot = toxic_scores.reset_index().melt(
    id_vars="index", var_name="Metric", value_name="Score",
)
plt.figure(figsize=(10.5, 5.8))
ax = sns.barplot(data=score_plot, x="Metric", y="Score", hue="index", palette=colors)
ax.set(title="Toxicity validation: model comparison", xlabel="", ylabel="Score", ylim=(0, 1))
ax.legend(title="")
save_plot("text_07_toxic_model_comparison.png")

# %% [markdown]
# Lấy riêng hai mô hình Logistic theo từ và theo ký tự, rồi chọn F1 toxic và macro F1 để phân tích chi tiết.

# %%
word_char = toxic_scores.loc[["Word TF-IDF + Logistic", "Char TF-IDF + Logistic"]]
word_char = word_char[["F1 toxic", "Macro F1"]].reset_index().melt(
    id_vars="index", var_name="Metric", value_name="Score",
)

# %% [markdown]
# Vẽ so sánh F1 toxic và macro F1 giữa hai cách biểu diễn. Đối chiếu nhãn và độ cao cột để xem phương án nào tốt hơn trên validation.

# %%
plt.figure(figsize=(8.8, 5.3))
ax = sns.barplot(data=word_char, x="Metric", y="Score", hue="index", palette=["#173f4c", "#efae00"])
ax.set(title="Word vs character TF-IDF", xlabel="", ylabel="Score", ylim=(0, 1))
ax.legend(title="")
save_plot("text_08_word_char_comparison.png")

# %% [markdown]
# Word TF-IDF plus Logistic Regression has the best validation macro F1. It is
# selected for the final toxicity test.

# %% [markdown]
# ## 8. Toxicity test results

# %% [markdown]
# Dùng mô hình Logistic theo từ đã huấn luyện để dự đoán tập test đã loại trùng train; in bốn thước đo. Các giá trị áp dụng cho 985 dòng test được giữ lại ở lần chạy lưu sẵn; chạy lại cần kiểm tra số dòng thực tế.

# %%
toxic_test_pred = toxic_word_model.predict(X_test_word)
print("Test accuracy:", round(accuracy_score(test_eval["Toxicity"], toxic_test_pred), 4))
print("Test balanced accuracy:", round(balanced_accuracy_score(test_eval["Toxicity"], toxic_test_pred), 4))
print("Test F1 toxic:", round(f1_score(test_eval["Toxicity"], toxic_test_pred), 4))
print("Test macro F1:", round(f1_score(test_eval["Toxicity"], toxic_test_pred, average="macro"), 4))

# %% [markdown]
# In precision, recall, F1 và số mẫu của hai lớp trên test. Đặc biệt xem recall và F1 của lớp độc hại thay vì chỉ đọc accuracy.

# %%
print(classification_report(test_eval["Toxicity"], toxic_test_pred, target_names=["Non-toxic", "Toxic"]))

# %% [markdown]
# Vẽ confusion matrix chuẩn hóa theo hàng, với nhãn thật ở hàng và nhãn dự đoán ở cột. Ô ngoài đường chéo cho biết tỷ lệ nhầm giữa hai lớp.

# %%
plt.figure(figsize=(7, 5.8))
cm = ConfusionMatrixDisplay.from_predictions(
    test_eval["Toxicity"], toxic_test_pred, normalize="true", values_format=".2f",
    display_labels=["Non-toxic", "Toxic"], cmap=sns.light_palette("#173f4c", as_cmap=True),
)
cm.ax_.set(title="Toxicity test confusion matrix", xlabel="Predicted", ylabel="True")
save_plot("text_09_toxic_confusion_matrix.png")

# %% [markdown]
# Tạo bảng các bình luận test dự đoán sai cùng chủ đề, nhãn thật và nhãn dự đoán; xem tám ví dụ đầu. Bảng giúp kiểm tra ngữ cảnh mà con số tổng hợp không cho thấy.

# %%
error_table = test_eval[["Comment", "Topic", "Toxicity"]].copy()
error_table["Prediction"] = toxic_test_pred
error_table = error_table[error_table["Toxicity"] != error_table["Prediction"]]
error_table.head(8)

# %% [markdown]
# The error table shows real false positives and false negatives. It is useful
# for checking where short or context-dependent comments confuse the model.

# %% [markdown]
# ## 9. Constructiveness classification

# %% [markdown]
# Huấn luyện Logistic Regression thứ hai cho nhãn tính xây dựng trên cùng TF-IDF từ, rồi dự đoán validation và test. Đây là nhiệm vụ riêng với toxicity.

# %%
constructive_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
constructive_model.fit(X_train_word, train["Constructiveness"])
constructive_valid_pred = constructive_model.predict(X_valid_word)
constructive_test_pred = constructive_model.predict(X_test_word)

# %% [markdown]
# Tạo bảng so sánh metric của nhiệm vụ tính xây dựng trên validation và test. Phân biệt hai tập khi diễn giải; test không dùng để chọn tham số.

# %%
constructive_scores = pd.DataFrame(
    [score_model(valid_eval["Constructiveness"], constructive_valid_pred),
     score_model(test_eval["Constructiveness"], constructive_test_pred)],
    columns=["Accuracy", "Balanced accuracy", "F1 constructive", "Macro F1"],
    index=["Validation", "Test"],
)
constructive_scores.round(4)

# %% [markdown]
# Vẽ confusion matrix chuẩn hóa cho nhãn tính xây dựng. Quan sát lỗi giữa hai nhóm non-constructive và constructive.

# %%
plt.figure(figsize=(7, 5.8))
cm = ConfusionMatrixDisplay.from_predictions(
    test_eval["Constructiveness"], constructive_test_pred, normalize="true", values_format=".2f",
    display_labels=["Non-constructive", "Constructive"], cmap=sns.light_palette("#d35b12", as_cmap=True),
)
cm.ax_.set(title="Constructiveness test confusion matrix", xlabel="Predicted", ylabel="True")
save_plot("text_10_constructive_confusion_matrix.png")

# %% [markdown]
# Constructiveness is easier than toxicity in this baseline because it has a
# larger positive class and clearer text patterns in the official split.

# %% [markdown]
# ## 10. Save results for the website

# %% [markdown]
# Tạo dictionary summary chứa quy mô, chủ đề, số trùng và số dòng loại do trùng train. Dictionary này chuẩn bị thông tin để lưu dưới dạng JSON.

# %%
summary = {
    "rows": len(all_data), "topics": all_data["Topic"].nunique(),
    "duplicate_comments": int(all_data["Comment"].duplicated().sum()),
    "validation_overlap_removed": len(valid) - len(valid_eval),
    "test_overlap_removed": len(test) - len(test_eval),
}

# %% [markdown]
# Bổ sung các điểm test và bảng validation vào summary. Các số được tính từ dự đoán hiện tại, không nhập tay vào tệp kết quả.

# %%
summary["toxicity_test"] = score_model(test_eval["Toxicity"], toxic_test_pred)
summary["constructiveness_test"] = score_model(test_eval["Constructiveness"], constructive_test_pred)
summary["toxicity_validation"] = toxic_scores.round(6).to_dict()

# %% [markdown]
# Ghi summary ra reports/text_victsd_summary.json bằng UTF-8 và in vị trí tệp. Website hoặc người đọc có thể sử dụng JSON này để kiểm tra kết quả.

# %%
summary_file = project_folder / "reports" / "text_victsd_summary.json"
summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:", summary_file)

# %% [markdown]
# ## 11. Conclusion
#
# - ViCTSD provides 10,000 Vietnamese comments from 10 topics.
# - The toxicity label is imbalanced, so macro F1 and balanced accuracy matter.
# - Word TF-IDF + Logistic Regression beats Naive Bayes and char TF-IDF here.
# - Removing train overlap reduces direct leakage; one validation/test overlap remains in the checked data.
# - The models help moderation analysis but cannot replace human judgement.
