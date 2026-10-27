# Intel Image Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a simple, reproducible Intel scene-classification Image assignment and include its verified outcomes in the DataFlow site and CO5177 report.

**Architecture:** `notebooks/image_intel_eda.py` is the source of truth and writes compact JSON, CSV, and PNG artifacts. The static Image EDA page and LaTeX chapter only render those generated artifacts and metrics, keeping results consistent across the notebook, web page, and PDF.

**Tech Stack:** Python, Hugging Face Datasets, PyTorch/Torchvision, scikit-learn, Matplotlib/Seaborn, static HTML/CSS/JS, LaTeX.

**Spec:** `docs/superpowers/specs/2026-10-26-intel-image-design.md`

## Global Constraints

- Use 17,034 labelled 150x150 RGB Intel scene images in six classes.
- Stratify the source training split into train/validation with seed 42 and reserve the source test split for final evaluation.
- Compare only pretrained-feature Logistic Regression and one frozen MobileNetV3-Small head.
- Generate independent, caption-ready figures and preserve all Galaxy10 artifacts.
- The website and report use exact metrics emitted by the notebook.

## Review Focus

- A missing or partially downloaded dataset must fail with a clear message instead of producing partial results.
- Test data must never enter feature fitting, model selection, or augmentation.
- The published page must not contain a Galaxy10 link or metric as the active Image task.
- The static report must link to an existing notebook, source dataset, and PDF.
- The report must compile without unresolved figure references or overfull layout warnings.

---

### Task 1: Dataset loader and reproducibility contract

**Files:**
- Create: `tests/test_image_intel_integrity.py`
- Create: `notebooks/image_intel_eda.py`
- Create: `notebooks/image_intel_eda.ipynb`

**Interfaces:**
- Produces: `load_splits()` returning labelled source train, validation, and final test datasets; `reports/image_intel_summary.json`; generated figures and prediction CSV.

- [x] **Step 1: Write failing artifact tests**

Test the expected six class names, 14,034 source train rows, 3,000 source test rows, and required generated artifact paths.

- [x] **Step 2: Run the test to verify red**

Run: `pytest tests/test_image_intel_integrity.py -v`
Expected: FAIL because the Intel notebook and artifacts do not exist.

- [x] **Step 3: Implement the compact notebook source and notebook**

Download the public Hugging Face data, create the fixed stratified split, generate EDA, train the two required pretrained approaches, select by validation macro F1, and write all outputs.

- [x] **Step 4: Run the full notebook source and focused tests**

Run: `python notebooks/image_intel_eda.py` then `pytest tests/test_image_intel_integrity.py -v`
Expected: generated artifacts exist and tests pass.

### Task 2: Static Image EDA page and landing page integration

**Files:**
- Create: `image-eda.html`
- Modify: `script.js`
- Modify: `README.md`
- Test: `tests/test_image_intel_integrity.py`

**Interfaces:**
- Consumes: metrics and figures emitted by Task 1.
- Produces: a public static page linked from the Image tab.

- [x] **Step 1: Extend the test with required content and link assertions**

Assert that the Image page, landing-page tab link, Colab link, and source link exist and name Intel scenes.

- [x] **Step 2: Run the expanded test to verify red**

Run: `pytest tests/test_image_intel_integrity.py -v`
Expected: FAIL because no Image EDA page or landing-page link exists.

- [x] **Step 3: Add the page and narrow landing-page changes**

Reuse the existing EDA layout and copy the generated metrics. Update only the Image tab’s title, description, preview labels, count, source name, and report path.

- [x] **Step 4: Run focused static checks**

Run: `pytest tests/test_image_intel_integrity.py -v`
Expected: PASS.

### Task 3: CO5177 report update and publication checks

**Files:**
- Modify: `CO5177 Report Latex/report.tex`
- Create: `CO5177 Report Latex/images/intel_*.png`
- Modify: `tests/test_image_intel_integrity.py`

**Interfaces:**
- Consumes: generated notebook figures/summary.
- Produces: compiled `CO5177 Report Latex/report.pdf` with the Intel Image chapter.

- [x] **Step 1: Extend the test with report requirements**

Assert that report source names Intel Image Classification, references 17,034 labelled images and six classes, and no longer names Galaxy10 in the active Image section.

- [x] **Step 2: Run the expanded test to verify red**

Run: `pytest tests/test_image_intel_integrity.py -v`
Expected: FAIL because the report contains the previous Image task.

- [x] **Step 3: Replace only the active Image chapter and summary/reproducibility references**

Use the actual generated metrics and figures, explain split, augmentation, fair model comparison, errors, limits, and the source dataset.

- [x] **Step 4: Compile and verify report outputs**

Run: `pdflatex -interaction=nonstopmode report.tex` twice in `CO5177 Report Latex`, then run the complete test suite.
Expected: no unresolved references, report test passes, and PDF contains the Intel chapter.

### Task 4: GitHub Pages publication

**Files:**
- Modify: only files produced in Tasks 1–3.

**Interfaces:**
- Consumes: verified static site, notebook, report, and generated artifacts.
- Produces: a commit pushed to `origin/main` for GitHub Pages publication.

- [x] **Step 1: Inspect the final diff and run full validation**

Run: `git status --short`, `pytest -v`, `python -m py_compile notebooks/image_intel_eda.py`, and static HTML/link checks.

- [x] **Step 2: Commit only the new Intel task files and deliberate report/site updates**

Use a conventional commit message naming the Intel Image task.

- [x] **Step 3: Push `main` and verify the remote commit**

Run: `git push origin main` and `git ls-remote origin refs/heads/main`.
Expected: the remote branch resolves to the committed SHA.
