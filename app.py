import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
import shap

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, render_template, request, session

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

FEATURES = [
    "heavy_menstrual_bleeding",
    "menstrual_pain_dysmenorrhea",
    "pain_during_sex_dyspareunia",
    "general_pelvic_pain",
    "irregular_or_missed_periods",
    "abdominal_pain_or_pressure",
    "back_pain",
    "painful_bowel_movements",
    "nausea",
    "infertility",
    "chronic_diarrhea",
    "chronic_constipation",
    "chronic_vomiting",
    "chronic_fatigue",
    "painful_ovulation",
    "migraines",
    "fainting_syncope",
    "mood_swings",
    "depression",
    "ovarian_cysts",
    "painful_urination",
    "anaemia_iron_deficiency",
    "vaginal_pain_or_pressure",
    "anxiety",
    "fever",
    "hormonal_problems",
    "bloating",
    "insomnia_or_sleeplessness",
    "acne_or_pimples",
    "loss_of_appetite",
]

FEATURE_GROUPS = {
    "Menstrual symptoms": [
        "heavy_menstrual_bleeding",
        "menstrual_pain_dysmenorrhea",
        "irregular_or_missed_periods",
        "painful_ovulation",
    ],
    "Pelvic and abdominal pain": [
        "general_pelvic_pain",
        "pain_during_sex_dyspareunia",
        "abdominal_pain_or_pressure",
        "back_pain",
        "vaginal_pain_or_pressure",
        "painful_urination",
    ],
    "Gastrointestinal symptoms": [
        "painful_bowel_movements",
        "nausea",
        "chronic_diarrhea",
        "chronic_constipation",
        "chronic_vomiting",
        "bloating",
        "loss_of_appetite",
    ],
    "General and neurological symptoms": [
        "chronic_fatigue",
        "migraines",
        "fainting_syncope",
        "fever",
        "anaemia_iron_deficiency",
        "hormonal_problems",
    ],
    "Mental health and related conditions": [
        "mood_swings",
        "depression",
        "anxiety",
        "insomnia_or_sleeplessness",
        "acne_or_pimples",
        "ovarian_cysts",
        "infertility",
    ],
}


def to_label(feature_name: str) -> str:
    return feature_name.replace("_", " ").capitalize()


FEATURE_QUESTIONS = [
    ("heavy_menstrual_bleeding", "Do you experience heavy menstrual bleeding?"),
    ("menstrual_pain_dysmenorrhea", "Do you have menstrual pain or dysmenorrhea?"),
    ("pain_during_sex_dyspareunia", "Do you experience pain during sex (dyspareunia)?"),
    ("general_pelvic_pain", "Do you have general pelvic pain?"),
    ("irregular_or_missed_periods", "Do you have irregular or missed periods?"),
    ("abdominal_pain_or_pressure", "Do you feel abdominal pain or pressure?"),
    ("back_pain", "Do you experience back pain?"),
    ("painful_bowel_movements", "Do you have painful bowel movements?"),
    ("nausea", "Do you experience nausea?"),
    ("infertility", "Have you experienced infertility?"),
    ("chronic_diarrhea", "Do you have chronic diarrhea?"),
    ("chronic_constipation", "Do you have chronic constipation?"),
    ("chronic_vomiting", "Do you experience chronic vomiting?"),
    ("chronic_fatigue", "Do you have chronic fatigue?"),
    ("painful_ovulation", "Do you experience painful ovulation?"),
    ("migraines", "Do you get migraines?"),
    ("fainting_syncope", "Do you experience fainting or syncope?"),
    ("mood_swings", "Do you have mood swings?"),
    ("depression", "Do you experience depression?"),
    ("ovarian_cysts", "Have you had ovarian cysts?"),
    ("painful_urination", "Do you have painful urination?"),
    ("anaemia_iron_deficiency", "Do you have iron deficiency anemia?"),
    ("vaginal_pain_or_pressure", "Do you feel vaginal pain or pressure?"),
    ("anxiety", "Do you experience anxiety?"),
    ("fever", "Do you have fever?"),
    ("hormonal_problems", "Do you have hormonal problems?"),
    ("bloating", "Do you experience bloating?"),
    ("insomnia_or_sleeplessness", "Do you have insomnia or sleeplessness?"),
    ("acne_or_pimples", "Do you have acne or pimples?"),
    ("loss_of_appetite", "Have you experienced loss of appetite?"),
]
FEATURE_LABELS = {feature: label for feature, label in FEATURE_QUESTIONS}

GROUPED_QUESTIONS = [
    (group, [(feature, FEATURE_LABELS[feature]) for feature in feature_list])
    for group, feature_list in FEATURE_GROUPS.items()
]

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model_training"
DATA_FILE = PROJECT_ROOT / "data" / "cleaned_endometriosis_dataset.xlsx"
MODEL_CANDIDATES = [
    "endometriosis_svm_classifier.pkl",
    "endometriosis_logistic_classifier.pkl",
    "endometriosis_randomforest_classifier.pkl",
    "endometriosis_xgb_classifier.pkl",
    "endometriosis_mlp_classifier.pkl",
]


def model_display_name(model_file: str) -> str:
    name = model_file.replace("endometriosis_", "").replace("_classifier.pkl", "")
    display_map = {
        "svm": "SVM",
        "logistic": "Logistic Regression",
        "randomforest": "Random Forest",
        "xgb": "XGBoost",
        "mlp": "MLP",
    }
    return display_map.get(name, name.replace("_", " ").title())


def load_model(model_file: str):
    model_path = MODEL_DIR / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    return joblib.load(model_path)


def load_background_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE, engine="openpyxl")
    return df[FEATURES].head(150)


def compute_shap_explanation(model, input_df: pd.DataFrame):
    model_name = model.__class__.__name__
    supported = {"XGBClassifier", "RandomForestClassifier", "LogisticRegression"}

    if model_name not in supported:
        return None, (
            f"SHAP explanations are currently available only for Logistic Regression, Random Forest, and XGBoost models. "
            f"You selected {model_name}, which is not supported for SHAP output."
        )

    try:
        if model_name in {"XGBClassifier", "RandomForestClassifier"}:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(input_df)
            if isinstance(values, list) and len(values) == 2:
                row_values = np.asarray(values[1])
            else:
                row_values = np.asarray(values)
        else:
            explainer = shap.LinearExplainer(model, BACKGROUND_DATA)
            values = explainer.shap_values(input_df)
            row_values = np.asarray(values)

        row_values = row_values.reshape(-1)
        if row_values.size < len(FEATURES):
            return None, "SHAP explanation returned fewer values than expected."

        row_values = row_values[: len(FEATURES)]
        shap_pairs = list(zip(FEATURES, [float(value) for value in row_values]))
        shap_pairs.sort(key=lambda item: abs(item[1]), reverse=True)
        top_items = [{"feature": feature, "value": value} for feature, value in shap_pairs[:5]]
        return top_items, None
    except ModuleNotFoundError:
        return None, "SHAP package is not installed. Install shap to enable explanations."
    except Exception as exc:
        return None, f"SHAP explanation unavailable: {exc}"


def escape_pdf_text(text: str) -> str:
    # Escape the few special characters PDF text streams need.
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf_text(x: int, y: int, text: str, size: int = 10) -> str:
    # Build a single text drawing command for the PDF page.
    return f"BT /F1 {size} Tf {x} {y} Td ({escape_pdf_text(text)}) Tj ET"


def pdf_rect(x: int, y: int, width: int, height: int, color: str = "0.93 0.96 1") -> str:
    return f"q {color} rg {x} {y} {width} {height} re f Q"


def build_pdf_bytes(report: dict) -> bytes:
    summary_lines = [
        f"Generated: {report['timestamp']}",
        f"Model: {model_display_name(report['model'])}",
        f"Estimated probability: {report['probability_percent']}%",
    ]

    responses = [
        (FEATURE_LABELS[feature], "Yes" if report["responses"].get(feature) == 1 else "No")
        for feature in FEATURES
    ]

    shap_items = report.get("shap_top_features") or []
    max_shap = max((abs(item["value"]) for item in shap_items), default=1.0)
    shap_bars = []
    for item in shap_items:
        bar_width = int((abs(item["value"]) / max_shap) * 360)
        shap_bars.append(
            {
                "label": to_label(item["feature"]),
                "value": float(item["value"]),
                "width": bar_width,
                "direction": "positive" if item["value"] >= 0 else "negative",
            }
        )

    content_lines = []
    y = 760
    content_lines.append("BT /F1 18 Tf 50 760 Td (Endometriosis Survey Report) Tj ET")
    y -= 32
    for line in summary_lines:
        safe_text = escape_pdf_text(line)
        content_lines.append(f"BT /F1 12 Tf 50 {y} Td ({safe_text}) Tj ET")
        y -= 18

    y -= 14
    content_lines.append(pdf_text(50, y, "Responses", size=14))
    y -= 28

    # Build the responses table in two columns. This is just positioning math.
    responses_per_col = 15
    row_height = 24
    col1_x = 50
    col2_x = 330
    response_start_y = y

    for idx, (label, answer) in enumerate(responses):
        col = 0 if idx < responses_per_col else 1
        row = idx % responses_per_col
        row_y = response_start_y - row * row_height
        if row_y < 120:
            break

        col_x = col1_x if col == 0 else col2_x
        if row % 2 == 0:
            # Alternate row background for readability.
            content_lines.append(pdf_rect(col_x - 3, row_y - 6, 270, 20))

        content_lines.append(pdf_text(col_x, row_y, label, size=9))
        content_lines.append(pdf_text(col_x + 230, row_y, answer, size=9))

    y = response_start_y - responses_per_col * row_height - 32
    if y < 140:
        y = 140

    if shap_bars and y > 100:
        # SHAP block: header plus one row per top driver.
        content_lines.append(pdf_text(50, y, "Top SHAP Drivers", size=14))
        y -= 24

        for item in shap_bars:
            label_text = item["label"]
            if len(label_text) > 30:
                label_text = label_text[:28] + "..."
            row_label = escape_pdf_text(label_text)
            row_value = escape_pdf_text(f"{item['value']:+.5f}")
            content_lines.append(pdf_text(50, y, row_label, size=10))
            bar_y = y - 14
            content_lines.append(pdf_rect(50, bar_y, 400, 10, "0.90 0.93 0.97"))
            bar_width = min(item["width"], 400)
            bar_color = "0.07 0.65 0.65" if item["direction"] == "positive" else "0.80 0.10 0.12"
            content_lines.append(f"q {bar_color} rg 50 %d %d 10 re f Q" % (bar_y, bar_width))
            content_lines.append(pdf_text(460, bar_y + 2, row_value, size=9))
            y -= 34
            if y < 60:
                break

    content_stream = "\n".join(content_lines).encode("latin1")

    # Create the PDF page and attach the font/resource objects.
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    resources = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})})
    page[NameObject("/Resources")] = resources

    # Put all of the raw PDF drawing commands into the page content stream.
    contents = DecodedStreamObject()
    contents.set_data(content_stream)
    contents[NameObject("/Length")] = NumberObject(len(content_stream))
    page[NameObject("/Contents")] = writer._add_object(contents)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


AVAILABLE_MODELS = [c for c in MODEL_CANDIDATES if (MODEL_DIR / c).exists()]
if not AVAILABLE_MODELS:
    raise FileNotFoundError("No trained model found in model_training directory.")

BACKGROUND_DATA = load_background_data()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "endoaware-dev-secret")


@app.route("/", methods=["GET", "POST"])
def index():
    probability = None
    error = None
    shap_top_features = None
    shap_message = None
    shap_chart_data = None
    selected_model = AVAILABLE_MODELS[0]
    responses = {feature: None for feature in FEATURES}

    if request.method == "POST":
        selected_model = request.form.get("selected_model", AVAILABLE_MODELS[0])
        try:
            input_row = {feature: int(request.form.get(feature, "0")) for feature in FEATURES}
            responses = input_row
            input_df = pd.DataFrame([input_row], columns=FEATURES)
            model = load_model(selected_model)
            probability = float(model.predict_proba(input_df)[0][1]) * 100.0
            shap_top_features, shap_message = compute_shap_explanation(model, input_df)

            if shap_top_features:
                max_abs = max(abs(item["value"]) for item in shap_top_features) or 1.0
                shap_chart_data = [
                    {
                        "feature": item["feature"],
                        "value": float(item["value"]),
                        "width_pct": (abs(item["value"]) / max_abs) * 100.0,
                        "direction": "positive" if item["value"] >= 0 else "negative",
                    }
                    for item in shap_top_features
                ]

            session["last_result"] = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model": selected_model,
                "probability_percent": round(probability, 4),
                "responses": input_row,
                "shap_top_features": shap_top_features,
            }
        except Exception as exc:
            error = f"Unable to calculate probability: {exc}"

    return render_template(
        "index.html",
        grouped_questions=GROUPED_QUESTIONS,
        available_models=AVAILABLE_MODELS,
        selected_model=selected_model,
        probability=probability,
        shap_top_features=shap_top_features,
        shap_chart_data=shap_chart_data,
        shap_message=shap_message,
        responses=responses,
        error=error,
        model_display_name=model_display_name,
    )


@app.get("/download-report")
def download_report():
    report = session.get("last_result")
    if not report:
        return Response("No report available yet. Submit the survey first.", status=400)

    pdf_bytes = build_pdf_bytes(report)
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment; filename=endometriosis_report.pdf"},
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
