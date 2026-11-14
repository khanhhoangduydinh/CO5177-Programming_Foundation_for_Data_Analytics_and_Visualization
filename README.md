![CO5177 — From data to insight: tabular, text and image](assets/readme-banner.svg)

<p align="center">
  <a href="https://khanhhoangduydinh.github.io/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/"><img alt="CO5177 website" src="https://img.shields.io/badge/Website-CO5177-173f4c?style=flat-square"></a>
  <a href="https://colab.research.google.com/github/khanhhoangduydinh/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/blob/main/notebooks/tabular_koi_eda.ipynb"><img alt="Open in Google Colab" src="https://img.shields.io/badge/Notebooks-Google%20Colab-d35b12?style=flat-square&amp;logo=googlecolab&amp;logoColor=white"></a>
  <a href="CO5177%20Report%20Latex/report.pdf"><img alt="Combined PDF report" src="https://img.shields.io/badge/Report-PDF-618e81?style=flat-square"></a>
  <img alt="Three data types" src="https://img.shields.io/badge/Data-Tabular%20%7C%20Text%20%7C%20Image-efae00?style=flat-square">
</p>

# CO5177 — Programming Foundation for Data Analytics and Visualization

This repository contains the course assignments for **CO5177**, Ho Chi Minh City University of Technology — Vietnam National University Ho Chi Minh City, semester HK261 (2026–2027). The project uses Python to explore, visualize and prepare three types of data, build classification models, and evaluate their results.

**Group:** Trailblazer · **Author:** Dinh Hoang Duy Khanh · **Lecturer:** Dr. Le Thanh Sach

## Quick links

| Task | Dataset and problem | EDA page | Google Colab |
| --- | --- | --- | --- |
| Tabular | NASA Kepler KOI: explore three labels; classify two known classes | [Tabular EDA](https://khanhhoangduydinh.github.io/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/tabular-eda.html) | [Open Tabular](https://colab.research.google.com/github/khanhhoangduydinh/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/blob/main/notebooks/tabular_koi_eda.ipynb) |
| Text | ViCTSD: classify Vietnamese comments using two labels | [Text EDA](https://khanhhoangduydinh.github.io/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/text-eda.html) | [Open Text](https://colab.research.google.com/github/khanhhoangduydinh/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/blob/main/notebooks/text_victsd_eda.ipynb) |
| Image | Intel Image Classification: classify six scene categories | [Image EDA](https://khanhhoangduydinh.github.io/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/image-eda.html) | [Open Image](https://colab.research.google.com/github/khanhhoangduydinh/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/blob/main/notebooks/image_intel_eda.ipynb) |

- [Project website](https://khanhhoangduydinh.github.io/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/) · [Combined PDF report](CO5177%20Report%20Latex/report.pdf) · [Assignment and grading rubric](assignment-vne-v4.pdf)
- Use the **EN / VI** switch on the website to change language; the selection is remembered across pages. Static figure labels remain in the original notebook language.
- Presentation videos are **not yet available**.

## Task details

### 1. Tabular — Kepler Objects of Interest

- **Data:** 9,564 rows and 30 columns in a snapshot of the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html). EDA retains three catalogue labels: `CONFIRMED`, `CANDIDATE`, and `FALSE POSITIVE`. Binary modeling excludes the 1,977 `CANDIDATE` rows and uses the other 7,587 rows.
- **Approach:** inspect missing values and outliers, visualize features, exclude information directly related to the target from model inputs, and compare classification models.
- **Saved result:** Balanced Random Forest achieved binary test accuracy **0.923** and macro F1 **0.919**. See [tabular_koi_summary.json](reports/tabular_koi_summary.json).
- **Files:** [notebook](notebooks/tabular_koi_eda.ipynb) · [Python source](notebooks/tabular_koi_eda.py) · [data snapshot](data/koi_cumulative.csv).

### 2. Text — ViCTSD

- **Data:** [ViCTSD](https://github.com/tarudesu/ViCTSD) contains 10,000 Vietnamese comments. The original split has 7,000 training, 2,000 validation, and 1,000 test comments. The tasks separately predict `Toxicity` and `Constructiveness`.
- **Approach:** summarize and visualize the text data, check for duplicate comments across splits, transform text with TF-IDF, and compare classifiers.
- **Saved results:** on the filtered test set, Toxicity achieved accuracy **0.842** and macro F1 **0.679**; Constructiveness achieved accuracy **0.807** and macro F1 **0.800**. See [text_victsd_summary.json](reports/text_victsd_summary.json).
- **Limitation:** after removing comments that also appear in training, one normalized comment still occurs in both validation and test. The saved results include that case; remove it and rerun the experiments before treating the two sets as fully independent.
- **Files:** [notebook](notebooks/text_victsd_eda.ipynb) · [Python source](notebooks/text_victsd_eda.py) · [data information](data/README.md).

### 3. Image — Intel Image Classification

- **Data:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) has 17,034 labeled 150 × 150 images in six classes: buildings, forest, glacier, mountain, sea, and street.
- **Approach:** inspect class distributions and samples, augment training images, extract features with pretrained MobileNetV3, and compare two classification approaches. Validation selects the model; a separate test set is used for final evaluation.
- **Saved result:** Logistic Regression on pretrained features achieved test accuracy **0.914** and macro F1 **0.916** on 3,000 images. See [image_intel_summary.json](reports/image_intel_summary.json).
- **Limitation:** when training the alternative classifier head, backbone parameters are frozen, but Batch Normalization running statistics may still change in training mode.
- **Files:** [notebook](notebooks/image_intel_eda.ipynb) · [Python source](notebooks/image_intel_eda.py) · [test predictions](reports/image_intel_test_predictions.csv).

These are results saved from previous runs, not a guarantee that every run will produce identical scores.

## Run a notebook in Google Colab

1. Open a notebook using the **Google Colab** links above.
2. Read the Vietnamese guidance before each code cell and select **Runtime → Run all**. The first run needs an internet connection to download required packages or data.
3. For the Image task, a GPU is recommended under **Runtime → Change runtime type**. The notebook downloads Intel images and pretrained weights if needed; runtime and storage requirements depend on the Colab environment.
4. To save changes, choose **File → Save a copy in Drive** or save a copy to GitHub. The links above open notebooks from the GitHub `main` branch; changes that have not been saved and pushed do not appear there automatically.

Each task has a notebook (`.ipynb`) and a corresponding percent-format Python file (`.py`). These are separate files: check and update both when needed, because they do not synchronize automatically.

## Run the website locally

Install Python 3, open a terminal in the repository folder, and run:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/` in a browser. The website uses static HTML, CSS, and JavaScript and does not require a build step. Use a local server instead of opening the file directly so that pages can load their JSON data correctly.

For local notebook dependencies, run `pip install -r requirements-eda.txt` in a virtual environment. If the default PyTorch/torchvision installation does not work, choose versions appropriate for your operating system and CPU/GPU.

## Repository structure

```text
├── index.html                         # Landing page
├── tabular-eda.html, text-eda.html, image-eda.html
├── assets/
│   ├── css/                           # Landing page and EDA styles
│   ├── js/                            # JavaScript files
│   └── ...                            # Logos, banner, and dataset images
├── notebooks/
│   ├── archive/                       # Older Intel notebook copy
│   └── ...                            # Current notebooks and Python sources
├── data/                              # KOI snapshot and ViCTSD splits
├── reports/                           # Figures and saved results used by the website
├── CO5177 Report Latex/               # LaTeX source and combined PDF report
├── tests/                             # Selected integrity checks
├── assignment-vne-v4.pdf              # Assignment
└── requirements-eda.txt               # Main Python dependencies
```

The website links to the three current notebooks in `notebooks/` and the single combined report at `CO5177 Report Latex/report.pdf`. The `reports/` folder contains figures and experiment results used by the website, not separate reports. Earlier Galaxy10 notebook files and the older Intel notebook in `notebooks/archive/` remain available.

## Data sources

- [NASA Exoplanet Archive — Kepler Objects of Interest](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html)
- [UIT-ViCTSD — Vietnamese Constructive and Toxic Speech Detection](https://github.com/tarudesu/ViCTSD)
- [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

## Author

[Dinh Hoang Duy Khanh](https://github.com/khanhhoangduydinh) · Trailblazer · HCMUT
