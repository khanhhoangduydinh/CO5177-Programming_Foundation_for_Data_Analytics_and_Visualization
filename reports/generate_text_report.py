"""Create the PDF report for the ViCTSD text-classification assignment."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
OUTPUT = ROOT / "reports" / "text-victsd-eda-report.pdf"
INK = colors.HexColor("#173f4c")
ORANGE = colors.HexColor("#d35b12")
MINT = colors.HexColor("#d8e9e2")
MUTED = colors.HexColor("#61777d")
LINE = colors.HexColor("#c6d3cf")


def report_image(name: str, width: float = 170 * mm) -> Image:
    """Return one notebook figure with its original aspect ratio."""
    image = Image(str(FIGURES / name))
    scale = width / image.imageWidth
    image.drawWidth = width
    image.drawHeight = image.imageHeight * scale
    return image


def styled_table(rows: list[list[str]], widths: list[float]) -> Table:
    """Create a compact, readable table with a dark header."""
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def footer(canvas, document) -> None:
    """Draw the assignment footer on every report page."""
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d7dedc"))
    canvas.line(20 * mm, 16 * mm, 190 * mm, 16 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 10 * mm, "CO5177 · TEXT DATA · VICTSD")
    canvas.drawRightString(190 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> None:
    """Build a reproducible report from the executed simple notebook."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=27, leading=30, textColor=INK, alignment=TA_CENTER, spaceAfter=8 * mm))
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=INK, spaceBefore=2 * mm, spaceAfter=5 * mm))
    styles.add(ParagraphStyle(name="SubTitle", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=ORANGE, spaceBefore=4 * mm, spaceAfter=2 * mm))
    styles.add(ParagraphStyle(name="BodyReport", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=14, textColor=colors.HexColor("#263f47"), spaceAfter=3 * mm))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=8, leading=11, textColor=MUTED, spaceAfter=4 * mm))

    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=19 * mm, leftMargin=19 * mm, topMargin=18 * mm, bottomMargin=22 * mm, title="ViCTSD — Vietnamese Text EDA and Classification", author="Dinh Hoang Duy Khanh")
    story = [
        Spacer(1, 22 * mm),
        Paragraph("CO5177 · TEXT DATA ASSIGNMENT", styles["SubTitle"]),
        Paragraph("Vietnamese Constructiveness and Toxicity", styles["ReportTitle"]),
        Paragraph("Exploratory Data Analysis and Simple TF-IDF Classification", ParagraphStyle(name="CoverSubtitle", parent=styles["BodyReport"], fontSize=14, leading=18, alignment=TA_CENTER, textColor=ORANGE, spaceAfter=14 * mm)),
    ]
    cover = styled_table([
        ["Student", "Dinh Hoang Duy Khanh"],
        ["Student ID", "2670306"],
        ["Group", "APlus"],
        ["Course", "Programming Foundation for Data Analytics and Visualization"],
        ["Semester", "HK261 · Academic year 2026–2027"],
        ["Dataset", "UIT-ViCTSD: Vietnamese social-media comments"],
    ], [40 * mm, 110 * mm])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), MINT),
        ("BACKGROUND", (1, 0), (1, 0), colors.white),
        ("TEXTCOLOR", (1, 0), (1, 0), INK),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica"),
    ]))
    story.extend([cover, Spacer(1, 13 * mm), Paragraph("This report uses the official 10,000-comment ViCTSD split. It independently visualizes the data, audits duplicated text across splits, compares simple TF-IDF baselines, and reports held-out test metrics for toxicity and constructiveness.", styles["BodyReport"]), PageBreak()])

    story.extend([
        Paragraph("1. Dataset and assignment fit", styles["SectionTitle"]),
        Paragraph("ViCTSD contains Vietnamese online comments annotated for two binary tasks: Constructiveness and Toxicity. The official files contain 7,000 training rows, 2,000 validation rows, and 1,000 test rows across ten topics. The project retains these official splits and uses the comment column as the only model input.", styles["BodyReport"]),
        Paragraph("Assignment compliance", styles["SubTitle"]),
        styled_table([
            ["Requirement", "Observed in this project"],
            [">= 2,000 text samples", "10,000 Vietnamese comments"],
            ["Data split", "official train / validation / test = 7,000 / 2,000 / 1,000"],
            ["Exploratory analysis", "labels, topics, text length, n-grams, duplicates"],
            ["Text preprocessing", "lowercase, URL removal, punctuation removal, whitespace cleanup"],
            ["Model comparison", "Naive Bayes · word TF-IDF LR · char TF-IDF LR"],
            ["Evaluation", "accuracy, balanced accuracy, minority-class F1, macro F1, confusion matrices"],
        ], [59 * mm, 106 * mm]),
        Spacer(1, 5 * mm),
        report_image("text_01_split_sizes.png", 145 * mm),
        Paragraph("Figure 1. Official split sizes are preserved. Validation is used to compare simple models; the test split is opened only for the selected word-level Logistic Regression.", styles["Caption"]),
        PageBreak(),
    ])

    story.extend([
        Paragraph("2. Text exploration and data quality", styles["SectionTitle"]),
        Paragraph("Toxic comments are a minority: 1,101 of 10,000 rows. Accuracy alone can therefore look good while a classifier misses most toxic comments. The report consequently uses minority-class F1 and macro F1 beside accuracy.", styles["BodyReport"]),
        report_image("text_02_label_distribution.png", 135 * mm),
        Paragraph("Figure 2. Label balance for the two annotation tasks. Toxicity has a stronger imbalance than Constructiveness.", styles["Caption"]),
        report_image("text_03_topic_distribution.png", 135 * mm),
        Paragraph("Figure 3. The source provides an even 1,000 comments for each of the ten discussion topics.", styles["Caption"]),
        report_image("text_04_comment_length.png", 142 * mm),
        Paragraph("Figure 4. Comment length is right-skewed: mean 128.6 characters, median 87 characters, and a maximum of 1,494 characters.", styles["Caption"]),
        PageBreak(),
    ])

    story.extend([
        Paragraph("3. Leakage check and preprocessing", styles["SectionTitle"]),
        Paragraph("After simple normalization, 65 exact duplicate comments appear in the combined data. To avoid evaluating a test sentence that is already in training, the notebook removes validation or test rows whose normalized comment occurs in training. The final evaluation therefore uses 1,968 validation rows and 985 test rows. Training remains at 7,000 rows.", styles["BodyReport"]),
        Paragraph("Simple, reproducible representation", styles["SubTitle"]),
        styled_table([
            ["Step", "What the notebook does"],
            ["Cleaning", "lowercase; remove URLs and punctuation; normalize spaces"],
            ["Word features", "TF-IDF word 1–2 grams; at most 8,000 features"],
            ["Character features", "TF-IDF character 3–5 grams; at most 12,000 features"],
            ["Classifier", "Logistic Regression with balanced class weights"],
            ["Baseline", "Multinomial Naive Bayes on word TF-IDF"],
        ], [49 * mm, 116 * mm]),
        Spacer(1, 9 * mm),
        report_image("text_05_top_toxic_ngrams.png", 145 * mm),
        Paragraph("Figure 5. Highest positive word or phrase weights from the toxicity Logistic Regression. They are model signals, not a rule list for moderation.", styles["Caption"]),
        report_image("text_06_top_safe_ngrams.png", 145 * mm),
        Paragraph("Figure 6. Highest negative weights correspond to comments the model associates with the non-toxic class.", styles["Caption"]),
        PageBreak(),
    ])

    story.extend([
        Paragraph("4. Toxicity model comparison", styles["SectionTitle"]),
        Paragraph("The Naive Bayes baseline has the best validation accuracy (0.884) but almost no ability to recover toxic comments (toxic F1 = 0.026). Its predictions follow the majority non-toxic class. The word TF-IDF Logistic Regression is selected because its validation macro F1 is 0.677 and toxic F1 is 0.449, much more balanced results.", styles["BodyReport"]),
        report_image("text_07_toxic_model_comparison.png", 164 * mm),
        Paragraph("Figure 7. Validation comparison. Balanced accuracy, toxic F1, and macro F1 expose the failure of an accuracy-only baseline.", styles["Caption"]),
        report_image("text_08_word_char_comparison.png", 155 * mm),
        Paragraph("Figure 8. The word representation has the stronger macro F1 and toxic F1 here, so it is retained as the final simple model.", styles["Caption"]),
        Paragraph("Held-out toxicity result", styles["SubTitle"]),
        styled_table([
            ["Metric", "Test score"],
            ["Accuracy", "0.842"],
            ["Balanced accuracy", "0.732"],
            ["F1 toxic", "0.451"],
            ["Macro F1", "0.679"],
        ], [82 * mm, 83 * mm]),
        PageBreak(),
    ])

    story.extend([
        Paragraph("5. Test analysis and conclusion", styles["SectionTitle"]),
        report_image("text_09_toxic_confusion_matrix.png", 125 * mm),
        Paragraph("Figure 9. Test confusion matrix for the selected toxicity model. The model is useful as a first-pass signal, but minority-class mistakes remain and require human review.", styles["Caption"]),
        report_image("text_10_constructive_confusion_matrix.png", 125 * mm),
        Paragraph("Figure 10. Constructiveness Logistic Regression reaches 0.800 macro F1 on the filtered test split, compared with 0.679 macro F1 for toxicity.", styles["Caption"]),
        Paragraph("Conclusions and limitations", styles["SubTitle"]),
        Paragraph("This compact workflow meets the text-data assignment with independent plots, simple preprocessing, fair validation, multiple baselines, and error-oriented evaluation. It deliberately uses a student-friendly TF-IDF approach instead of a large pretrained model. Results should not be used for automatic moderation without policy review: labels reflect the dataset context, short or sarcastic Vietnamese comments are difficult, and excluding overlap rows reduces but cannot eliminate all real-world distribution differences.", styles["BodyReport"]),
        Paragraph("Reproducibility", styles["SubTitle"]),
        Paragraph("The repository includes the original CSV files, an editable short-cell Python notebook source, its executed notebook, figures, a JSON metric summary, this PDF, and a GitHub-backed Colab link. Re-running the notebook regenerates the charts and metrics.", styles["BodyReport"]),
    ])

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build_report()
