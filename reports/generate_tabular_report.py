"""Generate the PDF report for the CO5177 Kepler KOI tabular assignment."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
OUTPUT = ROOT / "reports" / "tabular-koi-eda-report.pdf"
INK = colors.HexColor("#173f4c")
ORANGE = colors.HexColor("#d35b12")
MINT = colors.HexColor("#d8e9e2")
MUTED = colors.HexColor("#61777d")


def report_image(name: str, width: float = 172 * mm) -> Image:
    """Return a proportionally scaled report image."""
    image = Image(str(FIGURES / name))
    scale = width / image.imageWidth
    image.drawWidth = width
    image.drawHeight = image.imageHeight * scale
    return image


def footer(canvas, document) -> None:
    """Draw a running footer on every report page."""
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d7dedc"))
    canvas.line(20 * mm, 16 * mm, 190 * mm, 16 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 10 * mm, "CO5177 · TABULAR DATA · KEPLER KOI")
    canvas.drawRightString(190 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> None:
    """Build the complete report PDF from executed notebook artifacts."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=27, leading=30, textColor=INK, alignment=TA_CENTER, spaceAfter=8 * mm))
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=INK, spaceBefore=2 * mm, spaceAfter=5 * mm))
    styles.add(ParagraphStyle(name="SubTitle", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=ORANGE, spaceBefore=4 * mm, spaceAfter=2 * mm))
    styles.add(ParagraphStyle(name="BodyReport", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=14, textColor=colors.HexColor("#263f47"), spaceAfter=3 * mm))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=8, leading=11, textColor=MUTED, spaceAfter=4 * mm))

    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=19 * mm, leftMargin=19 * mm, topMargin=18 * mm, bottomMargin=22 * mm, title="Kepler Objects of Interest — Tabular EDA and Classification", author="Dinh Hoang Duy Khanh")
    story = [
        Spacer(1, 22 * mm),
        Paragraph("CO5177 · TABULAR DATA ASSIGNMENT", styles["SubTitle"]),
        Paragraph("Kepler Objects of Interest", styles["ReportTitle"]),
        Paragraph("Exploratory Data Analysis and Leakage-Safe Classification", ParagraphStyle(name="CoverSubtitle", parent=styles["BodyReport"], fontSize=14, leading=18, alignment=TA_CENTER, textColor=ORANGE, spaceAfter=14 * mm)),
    ]
    cover = Table(
        [
            ["Student", "Dinh Hoang Duy Khanh"],
            ["Student ID", "2670306"],
            ["Group", "Trailblazer"],
            ["Course", "Programming Foundation for Data Analytics and Visualization"],
            ["Semester", "HK261 · Academic year 2026–2027"],
            ["Dataset", "NASA Kepler Objects of Interest cumulative catalogue"],
        ],
        colWidths=[38 * mm, 112 * mm],
    )
    cover.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), MINT), ("BACKGROUND", (1, 0), (1, -1), colors.white), ("TEXTCOLOR", (0, 0), (-1, -1), INK), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTNAME", (1, 0), (1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 9), ("LEADING", (0, 0), (-1, -1), 13), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c6d3cf")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.extend([cover, Spacer(1, 13 * mm), Paragraph("This report studies 9,564 transit-like signals using reproducible data-quality checks, independent visualizations, grouped train/validation/test splits, and a quantitative comparison between Logistic Regression and a balanced Random Forest.", styles["BodyReport"]), PageBreak()])

    story.extend([Paragraph("1. Problem and dataset", styles["SectionTitle"]), Paragraph("The Kepler mission identifies periodic decreases in stellar brightness that may indicate a transiting planet. Each Kepler Object of Interest (KOI) is catalogued as CONFIRMED, CANDIDATE, or FALSE POSITIVE. The modelling task is to learn multivariate catalogue patterns while avoiding fields that directly encode the vetting outcome.", styles["BodyReport"]), Paragraph("Assignment compliance", styles["SubTitle"])])
    compliance = Table(
        [
            ["Requirement", "Observed"],
            [">= 2,000 samples", "9,564 objects"],
            [">= 10 columns", "30 selected columns"],
            ["Missing values", "13,878 missing cells"],
            ["Categorical encoding", "koi_tce_delivname · one-hot encoding"],
            ["Numerical scaling", "StandardScaler / RobustScaler"],
            ["Imbalance and outliers", "3 classes · IQR outliers retained and audited"],
        ],
        colWidths=[60 * mm, 105 * mm], repeatRows=1,
    )
    compliance.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("BACKGROUND", (0, 1), (-1, -1), colors.white), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d7dedc")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.extend([compliance, Spacer(1, 6 * mm), report_image("01_target_distribution.png", 150 * mm), Paragraph("Figure 1. The target is moderately imbalanced: 4,839 false positives, 2,748 confirmed planets, and 1,977 candidates.", styles["Caption"]), PageBreak()])

    story.extend([
        Paragraph("2. Data quality and exploratory findings", styles["SectionTitle"]),
        Paragraph("Missingness is not random in every field. For example, an official Kepler name is normally available only for confirmed objects. Such label-like columns are excluded rather than imputed into the model. Partially observed physical measurements are median-imputed inside the training pipeline.", styles["BodyReport"]),
        report_image("02_missing_values.png", 124 * mm),
        Paragraph("Figure 2. Missing-value share. kepler_name, koi_score, pipeline dispositions, diagnostic flags, and identifiers are excluded from predictors to prevent target leakage.", styles["Caption"]),
        report_image("06_outlier_share.png", 124 * mm),
        Paragraph("Figure 3. IQR outlier audit. Extreme catalogue values are retained because unusual astronomical objects may be valid; robust preprocessing limits their numerical influence.", styles["Caption"]),
        PageBreak(),
    ])

    story.extend([
        Paragraph("3. Relationships and preprocessing", styles["SectionTitle"]),
        Paragraph("Estimated planet radius and transit signal-to-noise differ across the three dispositions, but all class distributions overlap. The problem therefore requires multivariate evidence rather than a single physical cutoff.", styles["BodyReport"]),
        report_image("07_target_vs_planet_radius.png", 121 * mm),
        Paragraph("Figure 4. Planet radius by disposition after limiting only the visualization to the 99th percentile.", styles["Caption"]),
        report_image("08_target_vs_snr.png", 121 * mm),
        Paragraph("Figure 5. Model signal-to-noise by disposition. The overlap motivates a nonlinear multivariate model.", styles["Caption"]),
        Paragraph("Leakage-safe split and transformations", styles["SubTitle"]),
        Paragraph("Rows are split by kepid with GroupShuffleSplit, keeping every host star in exactly one of train, validation, or test. Numerical variables use median imputation and scaling; the categorical delivery field uses most-frequent imputation and one-hot encoding. All transformations are fitted inside a scikit-learn Pipeline using training data only.", styles["BodyReport"]),
        PageBreak(),
    ])

    story.extend([
        Paragraph("4. Quantitative model comparison", styles["SectionTitle"]),
        Paragraph("A multinomial Logistic Regression provides the linear baseline. The extended model is a 350-tree Random Forest with balanced subsample weights, robust scaling, and a minimum leaf size of two. Macro F1 is the primary selection score because it gives all dispositions equal weight.", styles["BodyReport"]),
        report_image("12_model_comparison.png", 158 * mm),
        Paragraph("Figure 6. On validation hosts, the balanced Random Forest increases macro F1 from 0.558 to 0.758 and balanced accuracy from 0.582 to 0.759.", styles["Caption"]),
        report_image("13_confusion_matrix.png", 118 * mm),
        Paragraph("Figure 7. Normalized test confusion matrix. The final test macro F1 is 0.727; candidates are the most difficult class, with approximately 54% recall.", styles["Caption"]),
        PageBreak(),
    ])

    story.extend([
        Paragraph("5. Interpretation, conclusions, and limitations", styles["SectionTitle"]),
        report_image("14_feature_importance.png", 155 * mm),
        Paragraph("Figure 8. Permutation importance on unseen host stars. Transit signal-to-noise is the strongest predictor, followed by estimated radius, impact parameter, and transit duration.", styles["Caption"]),
        Paragraph("Conclusions", styles["SubTitle"]),
        Paragraph("The dataset meets every tabular constraint in the assignment. Grouped splitting provides a more realistic evaluation than random row splitting. The nonlinear, class-aware extension produces a measurable improvement across all reported validation metrics. The model is most reliable for confirmed and false-positive objects, while candidates remain intrinsically ambiguous.", styles["BodyReport"]),
        Paragraph("Limitations", styles["SubTitle"]),
        Paragraph("Catalogue labels may change as new evidence is incorporated. Median imputation does not propagate measurement uncertainty. Permutation importance reflects predictive utility, not astronomical causality. Catalogue classification can prioritize review but cannot establish or reject planetary confirmation without physical validation and follow-up observations.", styles["BodyReport"]),
        Paragraph("Reproducibility", styles["SubTitle"]),
        Paragraph("The executed notebook, editable Python source, dataset snapshot, generated figures, machine-readable metrics, and this report are versioned together in the project repository. The GitHub-backed Colab link always opens the latest notebook from the main branch.", styles["BodyReport"]),
    ])

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build_report()
