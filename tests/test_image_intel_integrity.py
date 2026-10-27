import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SUMMARY = PROJECT / "reports" / "image_intel_summary.json"
PREDICTIONS = PROJECT / "reports" / "image_intel_test_predictions.csv"
FIGURES = [
    PROJECT / "reports" / "figures" / f"intel_{number:02d}_{name}.png"
    for number, name in [
        (1, "scene_examples"),
        (2, "class_distribution"),
        (3, "brightness_by_class"),
        (4, "augmentation_example"),
        (5, "training_history"),
        (6, "model_comparison"),
        (7, "confusion_matrix"),
        (8, "error_examples"),
    ]
]


def test_intel_summary_keeps_source_test_for_final_evaluation():
    """Catch a split that mixes source-test images into model selection."""
    assert SUMMARY.exists(), "Run notebooks/image_intel_eda.py to create the summary."

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert summary["dataset"] == "Intel Image Classification"
    assert summary["image_size"] == [150, 150]
    assert summary["class_names"] == [
        "buildings",
        "forest",
        "glacier",
        "mountain",
        "sea",
        "street",
    ]
    assert summary["source_rows"] == {"train": 14034, "test": 3000}
    assert summary["split_rows"] == {"train": 11227, "validation": 2807, "test": 3000}
    assert summary["selection_metric"] == "validation_macro_f1"


def test_intel_run_writes_final_predictions_and_all_figures():
    """Catch incomplete notebook runs before the website or PDF uses their outputs."""
    assert PREDICTIONS.exists(), "Run notebooks/image_intel_eda.py to create test predictions."
    assert len(PREDICTIONS.read_text(encoding="utf-8").splitlines()) == 3001
    assert all(figure.exists() and figure.stat().st_size > 0 for figure in FIGURES)


def test_image_page_exposes_the_intel_notebook_and_generated_figures():
    """Catch an Image tab that points to the old task or an incomplete static page."""
    page = PROJECT / "image-eda.html"
    landing_script = PROJECT / "script.js"
    assert page.exists(), "Create image-eda.html before publishing the Image task."

    html = page.read_text(encoding="utf-8")
    script = landing_script.read_text(encoding="utf-8")
    assert "Intel Image Classification" in html
    assert "notebooks/image_intel_eda.ipynb" in html
    assert "puneet6060/intel-image-classification" in html
    assert all(figure.name in html for figure in FIGURES)
    assert 'title: "Intel Image Classification"' in script
    assert 'reportLink: "image-eda.html"' in script


def test_course_report_uses_the_intel_image_chapter():
    """Catch a PDF source that still presents the replaced Galaxy10 task."""
    report = PROJECT / "CO5177 Report Latex" / "report.tex"
    chapter = PROJECT / "CO5177 Report Latex" / "chapters" / "image_intel.tex"

    assert chapter.exists(), "Add the Intel Image Classification report chapter."
    assert r"\input{chapters/image_intel}" in report.read_text(encoding="utf-8")

    text = chapter.read_text(encoding="utf-8")
    assert "Image Data: Intel Image Classification" in text
    assert "17,034" in text
    assert "3,000" in text
    assert "MobileNetV3" in text
