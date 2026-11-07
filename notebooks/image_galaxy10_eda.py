# %% [markdown]
# # Galaxy10 DECaLS - Image Classification
#
# **Course:** CO5177 - Programming Foundation for Data Analytics and Visualization  
# **Student:** Dinh Hoang Duy Khanh - 2670306  
# **Group:** Traiblazer
#
# This notebook classifies galaxy shapes into 10 classes. It uses a simple
# pretrained MobileNetV3 feature extractor, then compares Logistic Regression
# with a small transfer-learning head.

# %% [markdown]
# ## 0. Install and import libraries
#
# Run this cell in Google Colab first. The download is large, so use a High RAM
# runtime if Colab asks for more memory.

# %%
# !pip -q install h5py scikit-learn torch torchvision

# %%
from pathlib import Path
import copy
import json
import urllib.request

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

# %%
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180})
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

project_folder = Path.cwd()
if project_folder.name == "notebooks":
    project_folder = project_folder.parent

data_folder = project_folder / "data"
figure_folder = project_folder / "reports" / "figures"
data_folder.mkdir(exist_ok=True)
figure_folder.mkdir(parents=True, exist_ok=True)

def save_plot(name):
    plt.tight_layout()
    plt.savefig(figure_folder / name, bbox_inches="tight", facecolor="white")
    plt.show()


def md5_matches(file_path, expected_hash):
    """Return True when a downloaded file has the expected MD5 hash."""
    import hashlib

    digest = hashlib.md5()
    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower() == expected_hash.lower()

# %% [markdown]
# ## 1. Download and load Galaxy10 DECaLS
#
# The official no-duplicate HDF5 file is used. It stores images, labels, sky
# coordinates, redshift, and pixel scale. Only image pixels are used as model
# inputs; the other fields are kept for dataset inspection.

# %%
DATA_URL = (
    "https://zenodo.org/records/10845026/files/"
    "Galaxy10_DECals_NoDuplicated.h5"
)
OFFICIAL_MD5 = "a920094d5e470e3705f4691c0d6dff54"
data_file = data_folder / "Galaxy10_DECals_NoDuplicated.h5"

if not data_file.exists():
    print("Downloading the official Galaxy10 DECaLS file...")
    urllib.request.urlretrieve(DATA_URL, data_file)

if not md5_matches(data_file, OFFICIAL_MD5):
    raise RuntimeError(
        "The downloaded HDF5 file does not match the official MD5 hash. "
        "Delete it and run this cell again."
    )

print("File:", data_file)
print("Size (GB):", round(data_file.stat().st_size / 1e9, 2))

# %%
with h5py.File(data_file, "r") as file:
    required_keys = {"images", "ans", "redshift", "pxscale"}
    missing_keys = required_keys.difference(file.keys())
    if missing_keys:
        raise KeyError(f"Missing HDF5 keys: {sorted(missing_keys)}")
    print("Keys:", list(file.keys()))
    images = file["images"][:]
    labels = file["ans"][:].astype(int)
    redshift = file["redshift"][:]
    pxscale = file["pxscale"][:]

print("Images:", images.shape, images.dtype)
print("Labels:", labels.shape, labels.dtype)
print("Pixel range:", images.min(), "to", images.max())

# %%
CLASS_NAMES = [
    "Disturbed",
    "Merging",
    "Round smooth",
    "In-between round",
    "Cigar shaped",
    "Barred spiral",
    "Unbarred tight spiral",
    "Unbarred loose spiral",
    "Edge-on without bulge",
    "Edge-on with bulge",
]

metadata = pd.DataFrame(
    {
        "Field": ["images", "ans", "redshift", "pxscale"],
        "Meaning": ["RGB galaxy image", "morphology label", "redshift", "arcsec/pixel"],
        "Missing values": [
            0,
            0,
            int(np.isnan(redshift).sum()),
            int(np.isnan(pxscale).sum()),
        ],
    }
)
metadata

# %% [markdown]
# ## 2. Image EDA
#
# One image from each class is shown first. The images have three channels made
# from the g, r, and z survey bands. A class label describes visual morphology,
# not whether an object is scientifically confirmed as one specific galaxy type.

# %%
rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(2, 5, figsize=(15, 7.4))

for class_id, ax in enumerate(axes.flat):
    image_id = rng.choice(np.where(labels == class_id)[0])
    ax.imshow(images[image_id])
    ax.set_title(f"{class_id}: {CLASS_NAMES[class_id]}", fontsize=9)
    ax.axis("off")

plt.suptitle("One Galaxy10 DECaLS example from each class", y=0.99)
save_plot("image_01_class_examples.png")

# %%
class_counts = pd.Series(labels).value_counts().sort_index()
class_table = pd.DataFrame(
    {
        "Class": CLASS_NAMES,
        "Images": class_counts.values,
        "Percent": (class_counts.values / len(labels) * 100).round(2),
    }
)
class_table

# %%
plt.figure(figsize=(10, 5.8))
ax = sns.barplot(data=class_table, x="Images", y="Class", color="#173f4c")
ax.bar_label(ax.containers[0], fmt="{:,.0f}", padding=3)
plt.title("Galaxy images per morphology class")
plt.xlabel("Number of images")
plt.ylabel("")
save_plot("image_02_class_distribution.png")

# %% [markdown]
# The classes are imbalanced. For example, Cigar shaped galaxies are much less
# common than Round smooth galaxies. Therefore, accuracy alone is not enough:
# this notebook also reports balanced accuracy and macro F1.

# %%
sample_ids = np.concatenate(
    [rng.choice(np.where(labels == class_id)[0], size=min(100, class_counts[class_id]), replace=False)
     for class_id in range(10)]
)
brightness = images[sample_ids].mean(axis=(1, 2, 3))
brightness_data = pd.DataFrame(
    {"Class": [CLASS_NAMES[labels[i]] for i in sample_ids], "Mean pixel value": brightness}
)

plt.figure(figsize=(11, 5.8))
sns.boxplot(data=brightness_data, x="Mean pixel value", y="Class", color="#a9cfc1")
plt.title("Brightness distribution from 100 images per class")
plt.xlabel("Mean RGB pixel value")
plt.ylabel("")
save_plot("image_03_brightness_by_class.png")

# %% [markdown]
# This plot uses a balanced sample of at most 100 images per class. It is an
# EDA check, not a feature used by the model. The full images are used later.

# %% [markdown]
# ## 3. Stratified train, validation, and test split
#
# The dataset has no official split. We split indices into 70% train, 15%
# validation, and 15% test. Stratification keeps all 10 classes in similar
# proportions in every split.

# %%
all_ids = np.arange(len(labels))
train_ids, holdout_ids = train_test_split(
    all_ids, test_size=0.30, stratify=labels, random_state=SEED
)
valid_ids, test_ids = train_test_split(
    holdout_ids, test_size=0.50, stratify=labels[holdout_ids], random_state=SEED
)

split_table = pd.DataFrame(
    {
        "Split": ["Train", "Validation", "Test"],
        "Images": [len(train_ids), len(valid_ids), len(test_ids)],
    }
)
split_table

# %%
split_labels = pd.DataFrame(
    {
        "Train": np.bincount(labels[train_ids], minlength=10),
        "Validation": np.bincount(labels[valid_ids], minlength=10),
        "Test": np.bincount(labels[test_ids], minlength=10),
    },
    index=CLASS_NAMES,
)
split_labels

# %% [markdown]
# ## 4. Preprocessing and augmentation
#
# MobileNetV3 was pretrained on ImageNet. Images are resized to 160x160 and
# normalized with ImageNet values. Horizontal flip is used only for training:
# a galaxy's left-right orientation does not change its morphology label.

# %%
IMAGE_SIZE = 160
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)
eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

# %%
example = Image.fromarray(images[train_ids[0]])
flipped = transforms.RandomHorizontalFlip(p=1)(example)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(example)
axes[0].set_title("Original")
axes[1].imshow(flipped)
axes[1].set_title("Horizontal flip")
for ax in axes:
    ax.axis("off")
save_plot("image_04_augmentation_example.png")

# %%
class GalaxyDataset(Dataset):
    def __init__(self, image_array, label_array, ids, transform):
        self.image_array = image_array
        self.label_array = label_array
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = Image.fromarray(self.image_array[image_id])
        return self.transform(image), int(self.label_array[image_id])

# %%
BATCH_SIZE = 32
train_set = GalaxyDataset(images, labels, train_ids, train_transform)
train_eval_set = GalaxyDataset(images, labels, train_ids, eval_transform)
valid_set = GalaxyDataset(images, labels, valid_ids, eval_transform)
test_set = GalaxyDataset(images, labels, test_ids, eval_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
train_eval_loader = DataLoader(train_eval_set, batch_size=BATCH_SIZE, num_workers=0)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, num_workers=0)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0)

# %% [markdown]
# ## 5. Baseline: pretrained MobileNetV3 features + Logistic Regression
#
# The pretrained backbone converts every image into a short numerical feature
# vector. Logistic Regression is then a simple Scikit-learn classifier on these
# vectors. This is the baseline for the transfer-learning model below.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = MobileNet_V3_Small_Weights.DEFAULT
feature_model = models.mobilenet_v3_small(weights=weights)
feature_model.classifier = nn.Identity()
feature_model = feature_model.to(device).eval()

for parameter in feature_model.parameters():
    parameter.requires_grad = False

print("Device:", device)

# %%
@torch.no_grad()
def extract_features(loader):
    vectors, targets = [], []
    for batch_images, batch_labels in loader:
        vectors.append(feature_model(batch_images.to(device)).cpu().numpy())
        targets.append(batch_labels.numpy())
    return np.vstack(vectors), np.concatenate(targets)

# %%
X_train, y_train = extract_features(train_eval_loader)
X_valid, y_valid = extract_features(valid_loader)
X_test, y_test = extract_features(test_loader)
print("Feature vector size:", X_train.shape)

# %%
def metrics(y_true, prediction):
    return {
        "accuracy": round(accuracy_score(y_true, prediction), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, prediction), 4),
        "macro_f1": round(f1_score(y_true, prediction, average="macro"), 4),
    }


def select_model(logistic_scores, transfer_scores):
    if logistic_scores["macro_f1"] >= transfer_scores["macro_f1"]:
        return "Pretrained features + Logistic"
    return "Frozen MobileNetV3 head"

# %%
logistic = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=SEED
)
logistic.fit(X_train, y_train)
logistic_valid_pred = logistic.predict(X_valid)
logistic_valid_scores = metrics(y_valid, logistic_valid_pred)
logistic_valid_scores

# %% [markdown]
# ## 6. Transfer learning: frozen MobileNetV3 + small classification head
#
# The pretrained image layers are frozen. Only the small new classifier head is
# trained. Class weights give minority classes a fairer training loss.

# %%
model = models.mobilenet_v3_small(weights=weights)
for parameter in model.features.parameters():
    parameter.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(model.classifier[0].in_features, 256),
    nn.Hardswish(),
    nn.Dropout(0.2),
    nn.Linear(256, 10),
)
model = model.to(device)

class_weight = len(y_train) / (10 * np.bincount(y_train, minlength=10))
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weight, dtype=torch.float32, device=device)
)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=3e-4)

print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# %%
def predict(model, loader):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            outputs = model(batch_images.to(device))
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            targets.extend(batch_labels.numpy())
    return np.array(targets), np.array(predictions)

# %%
def train_one_epoch(model, loader):
    model.train()
    correct, total = 0, 0
    for batch_images, batch_labels in loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        correct += (outputs.argmax(dim=1) == batch_labels).sum().item()
        total += len(batch_labels)
    return correct / total

# %%
history = []
best_state = None
best_macro_f1 = -1

for epoch in range(5):
    train_accuracy = train_one_epoch(model, train_loader)
    valid_true, valid_pred = predict(model, valid_loader)
    valid_scores = metrics(valid_true, valid_pred)
    history.append({"Epoch": epoch + 1, "Train accuracy": train_accuracy, **valid_scores})

    if valid_scores["macro_f1"] > best_macro_f1:
        best_macro_f1 = valid_scores["macro_f1"]
        best_state = copy.deepcopy(model.state_dict())

    print(f"Epoch {epoch + 1}: train={train_accuracy:.3f}, val macro F1={valid_scores['macro_f1']:.3f}")

model.load_state_dict(best_state)
history = pd.DataFrame(history)
history

# %%
plt.figure(figsize=(8.5, 5.2))
plt.plot(history["Epoch"], history["Train accuracy"], marker="o", label="Train accuracy")
plt.plot(history["Epoch"], history["accuracy"], marker="o", label="Validation accuracy")
plt.plot(history["Epoch"], history["macro_f1"], marker="o", label="Validation macro F1")
plt.title("Frozen MobileNetV3 training history")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
save_plot("image_05_training_history.png")

# %% [markdown]
# ## 7. Validation comparison and final test
#
# The model with the stronger validation macro F1 is selected. The test split
# is evaluated only after this decision.

# %%
valid_true, transfer_valid_pred = predict(model, valid_loader)
transfer_valid_scores = metrics(valid_true, transfer_valid_pred)

validation_results = pd.DataFrame(
    [logistic_valid_scores, transfer_valid_scores],
    index=["Pretrained features + Logistic", "Frozen MobileNetV3 head"],
)
validation_results

selected_model = select_model(logistic_valid_scores, transfer_valid_scores)
print("Selected model:", selected_model)

# %%
plot_results = validation_results.reset_index(names="Model").melt(
    id_vars="Model", var_name="Metric", value_name="Score"
)
plt.figure(figsize=(10, 5.5))
sns.barplot(data=plot_results, x="Metric", y="Score", hue="Model", palette=["#a9cfc1", "#173f4c"])
plt.ylim(0, 1)
plt.title("Validation comparison")
save_plot("image_06_model_comparison.png")

# %%
if selected_model == "Pretrained features + Logistic":
    test_true = y_test
    test_pred = logistic.predict(X_test)
else:
    test_true, test_pred = predict(model, test_loader)

test_scores = metrics(test_true, test_pred)
print("Test scores:", test_scores)

report = classification_report(test_true, test_pred, target_names=CLASS_NAMES, output_dict=True)
class_report = pd.DataFrame(report).T.round(3)
class_report

# %%
fig, ax = plt.subplots(figsize=(13, 11))
ConfusionMatrixDisplay.from_predictions(
    test_true,
    test_pred,
    display_labels=CLASS_NAMES,
    normalize="true",
    values_format=".2f",
    xticks_rotation=90,
    cmap=sns.light_palette("#173f4c", as_cmap=True),
    ax=ax,
)
ax.set_title(f"Normalized test confusion matrix: {selected_model}")
save_plot("image_07_confusion_matrix.png")

# %%
error_ids = test_ids[test_true != test_pred]
show_ids = error_ids[:12]
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for ax, image_id in zip(axes.flat, show_ids):
    position = np.where(test_ids == image_id)[0][0]
    true_name = CLASS_NAMES[test_true[position]]
    pred_name = CLASS_NAMES[test_pred[position]]
    ax.imshow(images[image_id])
    ax.set_title(f"True: {true_name}\nPred: {pred_name}", color="#d35b12", fontsize=8)
    ax.axis("off")

for ax in axes.flat[len(show_ids):]:
    ax.axis("off")

plt.suptitle(f"First 12 incorrect test predictions: {selected_model}", y=0.98)
save_plot("image_08_error_examples.png")

# %% [markdown]
# ## 8. Save a small result file for the report and website

# %%
summary = {
    "dataset": "Galaxy10 DECaLS NoDuplicated",
    "total_images": int(len(labels)),
    "image_shape": list(images.shape[1:]),
    "classes": CLASS_NAMES,
    "split_sizes": {row.Split: int(row.Images) for row in split_table.itertuples()},
    "logistic_validation": logistic_valid_scores,
    "transfer_validation": transfer_valid_scores,
    "selected_model": selected_model,
    "selected_test": test_scores,
}

summary_file = project_folder / "reports" / "image_galaxy10_summary.json"
summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
prediction_file = project_folder / "reports" / "image_galaxy10_test_predictions.csv"
pd.DataFrame(
    {
        "image_id": test_ids,
        "true_label": [CLASS_NAMES[label] for label in test_true],
        "predicted_label": [CLASS_NAMES[label] for label in test_pred],
    }
).to_csv(prediction_file, index=False)
print("Saved:", summary_file)
print("Saved:", prediction_file)

# %% [markdown]
# ## 9. Conclusion
#
# - Galaxy10 DECaLS has more than 5,000 images and 10 morphology classes.
# - The class distribution is imbalanced, so macro F1 and balanced accuracy are
#   reported with accuracy.
# - Horizontal flip is suitable because flipping does not change a galaxy's
#   morphology class.
# - Pretrained MobileNetV3 features provide a simple baseline. The frozen
#   transfer-learning head is compared on validation before one final test.
# - Error images and the confusion matrix show which morphology classes remain
#   visually difficult to separate.
