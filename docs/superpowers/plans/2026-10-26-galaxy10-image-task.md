# Galaxy10 Image Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, beginner-readable Galaxy10 image task and add its verified results to the CO5177 report.

**Architecture:** One linear notebook owns data download, EDA, preprocessing, two model paths, visual results, and a small JSON summary. The existing LaTeX report consumes only generated figure files and verified summary values. A focused regression test protects the download checksum guard.

**Tech Stack:** Python 3.13, h5py, NumPy, pandas, Matplotlib, seaborn, scikit-learn, PyTorch/torchvision, Jupyter, LaTeX.

**Spec:** `docs/superpowers/specs/2026-10-26-galaxy10-image-design.md`

## Global Constraints

- Keep notebook cells short, linear, and readable for a new Python student.
- Use Galaxy10 DECaLS NoDuplicated HDF5 and MD5 `a920094d5e470e3705f4691c0d6dff54`.
- Use seed 42 and a stratified 70%/15%/15% split.
- Use only verified execution output in report prose and tables.
- Do not push or commit Git changes unless the user asks.

## Review Focus

- Interrupted or overlapping HDF5 downloads must fail checksum validation before data loading.
- A missing required `images` or `ans` HDF5 key must stop the notebook with a clear error.
- Every class must appear in each stratified split.
- Baseline and transfer-learning results must use identical test indices.
- Report figure paths and reported metrics must correspond to files from the latest notebook run.

---

### Task 1: Preserve data-file integrity

**Files:**
- Create: `tests/test_image_galaxy10_integrity.py`
- Modify: `notebooks/image_galaxy10_eda.py`
- Modify: `notebooks/image_galaxy10_eda.ipynb`

**Interfaces:**
- Produces: `md5_matches(file_path, expected_hash) -> bool` in the notebook source.

- [x] Write a failing AST-based test that requires `md5_matches` and checks a known correct and incorrect MD5 value.
- [x] Run `pytest tests/test_image_galaxy10_integrity.py -v`; confirm failure because `md5_matches` is missing.
- [x] Add the smallest streaming checksum function and a pre-load checksum error to the notebook source.
- [x] Regenerate the executable notebook from the source and rerun the test.

### Task 2: Execute the analysis and retain real artifacts

**Files:**
- Modify: `notebooks/image_galaxy10_eda.py`
- Modify: `notebooks/image_galaxy10_eda.ipynb`
- Create: `reports/image_galaxy10_summary.json`
- Create: `reports/figures/image_01_class_examples.png` through `image_08_incorrect_predictions.png`

**Interfaces:**
- Consumes: valid official HDF5 file from Task 1.
- Produces: real EDA plots, model figures, test predictions, and summary metrics.

- [x] Replace the known-corrupt download with a single clean download and verify its published MD5.
- [x] Run the notebook from the BTL root; confirm HDF5 keys, all ten stratified classes, figures, and JSON summary.
- [x] Inspect the summary, plot dimensions, and saved predictions; rerun source syntax validation.

### Task 3: Add the report chapter

**Files:**
- Modify: `CO5177 Report Latex/report.tex`
- Modify: `CO5177 Report Latex/report.pdf`

**Interfaces:**
- Consumes: generated images and values from `reports/image_galaxy10_summary.json`.
- Produces: a source-cited Galaxy10 Image Data chapter with EDA, preprocessing, comparison, error analysis, limitations, and conclusion.

- [x] Read the fresh JSON and retain only supported values.
- [x] Add a new chapter starting on a new page with appropriate figures and beginner-friendly explanations.
- [x] Compile LaTeX twice, render selected pages, and inspect the result for overflows, missing figures, and header/footer consistency.

### Task 4: Final verification

**Files:**
- Verify: all Task 1–3 files.

- [x] Run the focused regression test and Python syntax check.
- [x] Verify source notebook and `.ipynb` content agree on checksum behavior.
- [x] Verify each report figure exists and PDF output is newly compiled.
- [x] Re-read this plan and report any unmet item rather than claiming completion.
