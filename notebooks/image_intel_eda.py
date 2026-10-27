# %% [markdown]
# # Intel Image Classification - Image EDA
#
# This notebook classifies six outdoor-scene labels: buildings, forest,
# glacier, mountain, sea, and street. It uses the public Intel Image
# Classification dataset and two small pretrained MobileNetV3 approaches.

# %%
# !pip -q install kagglehub scikit-learn torch torchvision matplotlib seaborn

# %%
from pathlib import Path
import json
import random

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

# %%
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

project_folder = Path.cwd().resolve()
if project_folder.name == "notebooks":
    project_folder = project_folder.parent

data_folder = project_folder / "data" / "intel_image"
figure_folder = project_folder / "reports" / "figures"
report_folder = project_folder / "reports"
figure_folder.mkdir(parents=True, exist_ok=True)
report_folder.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# %% [markdown]
# ## 1. Download and inspect the dataset
#
# KaggleHub downloads the public archive once. The archive contains a labelled
# `seg_train` folder and a separate labelled `seg_test` folder. The unlabelled
# `seg_pred` images are not used in this assignment.

# %%
local_copy = project_folder / "data" / "intel_image_data" / "data"
local_sample = local_copy / "seg_train" / "buildings" / "0.jpg"
if local_sample.exists() and local_sample.stat().st_size > 1000:
    dataset_path = local_copy
else:
    dataset_path = Path(
        kagglehub.dataset_download(
            "puneet6060/intel-image-classification",
            output_dir=str(data_folder),
        )
    )
train_folder = dataset_path / "seg_train"
test_folder = dataset_path / "seg_test"

if not train_folder.exists() or not test_folder.exists():
    raise FileNotFoundError("Expected seg_train and seg_test after the Intel dataset download.")

raw_train = datasets.ImageFolder(train_folder)
raw_test = datasets.ImageFolder(test_folder)
class_names = raw_train.classes

print("Classes:", class_names)
print("Source train images:", len(raw_train))
print("Source test images:", len(raw_test))
print("Image size:", Image.open(raw_train.samples[0][0]).size)

# %% [markdown]
# ## 2. Create train, validation, and test splits
#
# The source training folder is split into train and validation data with a
# fixed seed. The source test folder stays untouched until final evaluation.

# %%
train_ids = np.arange(len(raw_train))
train_labels = np.array(raw_train.targets)
fit_ids, validation_ids = train_test_split(
    train_ids,
    test_size=0.20,
    random_state=SEED,
    stratify=train_labels,
)

print("Train:", len(fit_ids))
print("Validation:", len(validation_ids))
print("Test:", len(raw_test))

# %% [markdown]
# ## 3. EDA: examples, class balance, and brightness

# %%
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for label, axis in enumerate(axes.flat):
    image_id = next(i for i, value in enumerate(raw_train.targets) if value == label)
    image, _ = raw_train[image_id]
    axis.imshow(image)
    axis.set_title(class_names[label])
    axis.axis("off")
fig.suptitle("One Intel scene from each class", fontsize=16)
fig.tight_layout()
fig.savefig(figure_folder / "intel_01_scene_examples.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
train_counts = pd.Series(raw_train.targets).value_counts().sort_index()
test_counts = pd.Series(raw_test.targets).value_counts().sort_index()
count_table = pd.DataFrame(
    {"source train": train_counts.values, "source test": test_counts.values},
    index=class_names,
)

ax = count_table.plot(kind="bar", figsize=(11, 5), color=["#174653", "#dc5b0b"])
ax.set_title("Label count in the source train and source test folders")
ax.set_xlabel("Scene class")
ax.set_ylabel("Images")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(figure_folder / "intel_02_class_distribution.png", dpi=180, bbox_inches="tight")
plt.show()
print(count_table)

# %%
brightness = []
for label, name in enumerate(class_names):
    image_ids = [i for i, value in enumerate(raw_train.targets) if value == label][:100]
    for image_id in image_ids:
        image, _ = raw_train[image_id]
        brightness.append({"class": name, "brightness": np.asarray(image).mean() / 255})

brightness_table = pd.DataFrame(brightness)
plt.figure(figsize=(11, 5))
sns.boxplot(data=brightness_table, x="class", y="brightness", palette="crest")
plt.title("Brightness from 100 training images per class")
plt.xlabel("Scene class")
plt.ylabel("Mean RGB value (0 to 1)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(figure_folder / "intel_03_brightness_by_class.png", dpi=180, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Preprocessing and augmentation
#
# Images are resized and normalized in the scale expected by ImageNet-pretrained
# MobileNetV3. A horizontal flip is used only in training because a mirrored
# scene still has the same class.

# %%
weights = MobileNet_V3_Small_Weights.DEFAULT
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
evaluation_transform = weights.transforms()
training_transform = transforms.Compose(
    [
        transforms.Resize(232),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)

example_path = raw_train.samples[fit_ids[0]][0]
example_image = Image.open(example_path).convert("RGB")
flipped_image = transforms.functional.hflip(example_image)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(example_image)
axes[0].set_title("Original")
axes[1].imshow(flipped_image)
axes[1].set_title("Horizontal flip")
for axis in axes:
    axis.axis("off")
fig.tight_layout()
fig.savefig(figure_folder / "intel_04_augmentation_example.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
train_data = Subset(datasets.ImageFolder(train_folder, transform=training_transform), fit_ids)
validation_data = Subset(datasets.ImageFolder(train_folder, transform=evaluation_transform), validation_ids)
test_data = datasets.ImageFolder(test_folder, transform=evaluation_transform)

loader_args = {"batch_size": 64, "num_workers": 0, "pin_memory": device == "cuda"}
train_loader = DataLoader(train_data, shuffle=True, **loader_args)
validation_loader = DataLoader(validation_data, shuffle=False, **loader_args)
test_loader = DataLoader(test_data, shuffle=False, **loader_args)

# %% [markdown]
# ## 5. Model A: pretrained features + Logistic Regression
#
# MobileNetV3 describes each image as a vector. Logistic Regression then chooses
# one of the six scene labels. This is the simplest pretrained baseline.

# %%
feature_model = models.mobilenet_v3_small(weights=weights).to(device).eval()
feature_model.classifier = nn.Identity()

def get_features(loader):
    vectors, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            vectors.append(feature_model(images.to(device)).cpu().numpy())
            labels.append(targets.numpy())
    return np.vstack(vectors), np.concatenate(labels)


feature_train = Subset(datasets.ImageFolder(train_folder, transform=evaluation_transform), fit_ids)
fit_x, fit_y = get_features(DataLoader(feature_train, shuffle=False, **loader_args))
validation_x, validation_y = get_features(validation_loader)
test_x, test_y = get_features(test_loader)

logistic = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
logistic.fit(fit_x, fit_y)
logistic_validation = logistic.predict(validation_x)
logistic_test = logistic.predict(test_x)

# %% [markdown]
# ## 6. Model B: frozen MobileNetV3 classifier head
#
# This model keeps the pretrained visual backbone fixed and trains only its
# classifier layers for five epochs.

# %%
head_model = models.mobilenet_v3_small(weights=weights).to(device)
for parameter in head_model.features.parameters():
    parameter.requires_grad = False
head_model.classifier[3] = nn.Linear(head_model.classifier[3].in_features, len(class_names))
head_model = head_model.to(device)

class_weight = np.bincount(fit_y, minlength=len(class_names))
class_weight = class_weight.sum() / (len(class_names) * class_weight)
loss_function = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32, device=device))
optimizer = torch.optim.Adam(head_model.classifier.parameters(), lr=0.0003)

history = {"train_loss": [], "validation_macro_f1": []}
for epoch in range(5):
    head_model.train()
    losses = []
    for images, targets in train_loader:
        optimizer.zero_grad()
        loss = loss_function(head_model(images.to(device)), targets.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    head_model.eval()
    predicted = []
    with torch.no_grad():
        for images, _ in validation_loader:
            predicted.extend(head_model(images.to(device)).argmax(1).cpu().numpy())

    history["train_loss"].append(float(np.mean(losses)))
    history["validation_macro_f1"].append(float(f1_score(validation_y, predicted, average="macro")))
    print(f"Epoch {epoch + 1}: loss={history['train_loss'][-1]:.3f}, macro F1={history['validation_macro_f1'][-1]:.3f}")

# %%
plt.figure(figsize=(8, 4))
plt.plot(range(1, 6), history["train_loss"], marker="o", label="train loss")
plt.plot(range(1, 6), history["validation_macro_f1"], marker="o", label="validation macro F1")
plt.title("Frozen MobileNetV3 classifier-head training")
plt.xlabel("Epoch")
plt.xticks(range(1, 6))
plt.legend()
plt.tight_layout()
plt.savefig(figure_folder / "intel_05_training_history.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
head_model.eval()
head_validation, head_test = [], []
with torch.no_grad():
    for images, _ in validation_loader:
        head_validation.extend(head_model(images.to(device)).argmax(1).cpu().numpy())
    for images, _ in test_loader:
        head_test.extend(head_model(images.to(device)).argmax(1).cpu().numpy())

def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


validation_scores = {
    "Pretrained features + Logistic Regression": metrics(validation_y, logistic_validation),
    "Frozen MobileNetV3 head": metrics(validation_y, head_validation),
}
selected_model = max(validation_scores, key=lambda name: validation_scores[name]["macro_f1"])
selected_test = logistic_test if selected_model.startswith("Pretrained") else np.array(head_test)
selected_probability = logistic.predict_proba(test_x).max(axis=1) if selected_model.startswith("Pretrained") else np.full(len(test_y), np.nan)
test_scores = metrics(test_y, selected_test)

# %%
model_table = pd.DataFrame(validation_scores).T
print(model_table.round(3))
model_table.plot(kind="bar", figsize=(10, 5), color=["#174653", "#dc5b0b", "#e9b949"])
plt.title("Validation comparison of two pretrained approaches")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(figure_folder / "intel_06_model_comparison.png", dpi=180, bbox_inches="tight")
plt.show()
print("Selected model:", selected_model)

# %% [markdown]
# ## 7. Final test evaluation and error review
#
# The test folder is evaluated once after choosing the model from validation
# macro F1. Macro F1 is reported because every scene class should matter.

# %%
matrix = confusion_matrix(test_y, selected_test, normalize="true")
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title(f"Test confusion matrix: {selected_model}")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig(figure_folder / "intel_07_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
wrong_ids = np.flatnonzero(test_y != selected_test)[:12]
fig, axes = plt.subplots(3, 4, figsize=(13, 9))
for axis, image_id in zip(axes.flat, wrong_ids):
    image_path, _ = raw_test.samples[image_id]
    axis.imshow(Image.open(image_path).convert("RGB"))
    axis.set_title(f"true: {class_names[test_y[image_id]]}\npred: {class_names[selected_test[image_id]]}", fontsize=9)
    axis.axis("off")
for axis in axes.flat[len(wrong_ids):]:
    axis.axis("off")
fig.suptitle("First incorrect predictions from the final test set", fontsize=15)
fig.tight_layout()
fig.savefig(figure_folder / "intel_08_error_examples.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
prediction_table = pd.DataFrame(
    {
        "image_path": [path for path, _ in raw_test.samples],
        "true_label": [class_names[label] for label in test_y],
        "predicted_label": [class_names[label] for label in selected_test],
        "confidence": selected_probability,
        "correct": test_y == selected_test,
    }
)
prediction_table.to_csv(report_folder / "image_intel_test_predictions.csv", index=False)

summary = {
    "dataset": "Intel Image Classification",
    "dataset_source": "https://www.kaggle.com/datasets/puneet6060/intel-image-classification",
    "class_names": class_names,
    "image_size": [150, 150],
    "source_rows": {"train": len(raw_train), "test": len(raw_test)},
    "split_rows": {"train": len(fit_ids), "validation": len(validation_ids), "test": len(raw_test)},
    "random_seed": SEED,
    "selection_metric": "validation_macro_f1",
    "validation_scores": validation_scores,
    "selected_model": selected_model,
    "test_scores": test_scores,
    "per_class_test": classification_report(test_y, selected_test, target_names=class_names, output_dict=True),
}
(report_folder / "image_intel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(pd.Series(test_scores).round(3))
print("Saved figures, summary, and test predictions to reports/.")

# %% [markdown]
# ## 8. Conclusion
#
# This dataset is not a toy dataset: it has 17,034 labelled 150x150 scene
# images and six classes. The notebook uses an explicit train/validation/test
# protocol, pretrained MobileNetV3, a meaningful augmentation, a model
# comparison, and test errors. The selected model is useful as a classroom
# baseline, but real scenes can still mix visual cues such as mountains,
# glaciers, buildings, and streets.
