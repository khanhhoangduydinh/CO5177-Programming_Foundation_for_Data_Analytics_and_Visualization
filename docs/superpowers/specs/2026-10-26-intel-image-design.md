# Intel Image Classification Design

## Intent

Replace the Galaxy10 image task as the public Image assignment with a simpler
but non-toy scene-classification project. The work must be readable and
reproducible for a first programming course while satisfying the CO5177 image
requirements.

## Chosen approach

Use the labelled Intel Image Classification distribution from Hugging Face.
It contains 14,034 source training images and 3,000 source test images, each
150 by 150 RGB pixels, across six scene classes: buildings, forest, glacier,
mountain, sea, and street. The source training split is stratified into 80%
training and 20% validation with seed 42; the source test split is retained
for one final evaluation.

The notebook uses only RGB pixels and supplied class labels. It compares two
pretrained MobileNetV3-Small approaches: fixed feature vectors with balanced
Logistic Regression, then a frozen backbone with a small classifier head.
Validation macro F1 selects the final approach; the original test split is not
used for selection.

## Scope

- Create a compact, Colab-runnable Python notebook and `.ipynb` counterpart.
- Produce independent EDA figures: class examples, train/test distribution,
  per-class brightness, augmentation, model comparison, confusion matrix, and
  incorrect test examples.
- Create `image-eda.html` using the established DataFlow EDA design.
- Point the Image tab on the landing page to the new report page.
- Replace the public Image chapter in `CO5177 Report Latex/report.tex` and
  compile its PDF from actual notebook outputs.
- Preserve existing Galaxy10 files and data; do not present them as the active
  Image task.

## Constraints

- At least 5,000 labelled images, at least 3 classes, and image side at least
  128 pixels: satisfied by 17,034 labelled 150x150 images and six classes.
- Do not use a toy dataset.
- Use a pretrained model, state the split, prevent leakage, and justify the
  horizontal-flip augmentation.
- Keep notebook cells short and beginner-readable. Do not add unrequested
  model families or hyperparameter searches.
- Preserve the existing public DataFlow visual language rather than creating a
  separate website style.

## Verification

1. Artifact tests verify the source split, generated summary, prediction rows,
   expected figures, page links, and compiled report text.
2. Run the complete notebook script locally on the labelled source data.
3. Compile the LaTeX report twice and inspect the PDF text/pages.
4. Review the static page locally and check all published paths before GitHub
   push.
