# Galaxy10 DECaLS Image Task Design

## Intent

Add the third CO5177 assignment task: a beginner-readable image analysis and
classification notebook for Galaxy10 DECaLS, followed by an evidence-based
section in the existing LaTeX report. The work must use results produced by
the notebook rather than invented values.

## Goals

- Use the official no-duplicate Galaxy10 DECaLS HDF5 dataset: 17,675 RGB
  256 by 256 images in ten galaxy-morphology classes.
- Meet the image-task requirements in `assignment-vne-v4.pdf`: at least 5,000
  images, at least three classes, a declared train/validation/test split,
  image EDA, preprocessing, pretrained deep learning, model comparison,
  confusion matrix, and error analysis.
- Keep the notebook linear and understandable to a new Python student:
  short cells, direct plots, common libraries, and no custom framework.
- Compare logistic regression on frozen MobileNetV3 embeddings against a
  small MobileNetV3 transfer-learning classifier on the exact same split.
- Add only verified counts, metrics, figures, and observations to the report.

## Non-goals

- Do not claim scientific discovery or use galaxy coordinates/redshift as
  model features.
- Do not tune many architectures, fabricate metrics, or add Git commits,
  pushes, or website changes in this task.
- Do not copy prose or figures from the reference reports.

## Chosen approach

The notebook downloads the official no-duplicate file and first checks its
published MD5 hash (`a920094d5e470e3705f4691c0d6dff54`). It creates a
stratified 70%/15%/15% train/validation/test split with seed 42, so every
class is represented and all reported comparisons use the same untouched test
set. Training images receive only horizontal flip augmentation; validation and
test images receive resize, tensor conversion, and ImageNet normalization.

MobileNetV3-Small uses ImageNet pretrained weights. The baseline fits logistic
regression to frozen image embeddings. The transfer-learning model freezes
MobileNet features and learns a small 256-unit classifier head for five
epochs, selecting the epoch with highest validation macro F1. This keeps the
learning objective transparent and realistic for a first course.

## Dataset integrity and transparency

The HDF5 key `images` supplies the pixels and `ans` supplies the morphology
labels. The notebook verifies the checksum before opening the file, checks the
required HDF5 keys and array shapes, and stops with an actionable error if the
download is incomplete. Labels are citizen-science morphology categories, not
physical confirmation of a galaxy property. The report will state this
limitation and will not overinterpret the classifier.

## Deliverables and validation

1. `notebooks/image_galaxy10_eda.py` and its executable `.ipynb` counterpart.
2. Image EDA plots, model figures, a JSON summary, and saved test predictions
   in `reports/`.
3. An Image Data chapter in `CO5177 Report Latex/report.tex` and a compiled,
   rendered `report.pdf`.
4. Validation evidence: checksum matches; HDF5 opens with expected keys;
   notebook run finishes; summary and all expected plots exist; source syntax
   check passes; LaTeX compiles twice; rendered report pages are visually
   inspected.

## Alternatives considered

- A CNN trained from scratch is shorter in concept but conflicts with the
  assignment's pretrained-model expectation and needs more epochs.
- Fine-tuning many backbone layers can improve accuracy but makes the learning
  process less suitable for a first-course, easy-to-explain notebook.
- A single deep model has less explanatory value than the selected baseline
  versus transfer-learning comparison.

## Execution note

The user asked to continue the already specified work, which authorizes native
execution without a separate design-review pause. Environment policy also
disallows delegating this task to subagents.
