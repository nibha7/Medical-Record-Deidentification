
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from ensemble_engine import EnsembleEngine
from file_utils import read_pdf, write_pdf, read_excel, write_excel


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="MedDeID",
    page_icon="🔒",
    layout="wide"
)

st.title("Medical Record De-Identification")
st.caption(
    "CNN + Decision Tree + Random Forest + Regex Ensemble"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("results")


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_engine():
    return EnsembleEngine()


try:
    engine = load_engine()
except Exception as e:
    st.error("Could not load the trained models.")
    st.error("Check that the trained model files exist in the models folder.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# MODEL EVALUATION DASHBOARD
# --------------------------------------------------

st.header("📊 Model Evaluation Dashboard")

metrics_path = RESULTS_DIR / "metrics.csv"

if metrics_path.exists():
    try:
        metrics_df = pd.read_csv(metrics_path)

        if metrics_df.empty:
            st.info("The metrics file is empty. Run python evaluate.py.")
        else:
            st.subheader("Model Performance Comparison")

            # Show the complete evaluation table.
            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True
            )

            # Identify the model and metric columns.
            normalized = {
                str(col).strip().casefold().replace("_", " ").replace("-", " "): col
                for col in metrics_df.columns
            }

            model_col = next(
                (
                    normalized[name]
                    for name in ("model", "classifier", "algorithm")
                    if name in normalized
                ),
                None
            )

            metric_mapping = {
                "Accuracy": ("accuracy",),
                "Precision": ("precision",),
                "Recall": ("recall",),
                "F1-Score": ("f1 score", "f1")
            }

            display_columns = {}

            for display_name, possible_names in metric_mapping.items():
                for name in possible_names:
                    if name in normalized:
                        display_columns[display_name] = normalized[name]
                        break

            # Show metric cards when all four models are present.
            if model_col:
                for _, model_row in metrics_df.iterrows():
                    st.write(
                        f"**{model_row[model_col]}**"
                    )

                    metric_cols = st.columns(
                        max(1, len(display_columns))
                    )

                    for i, (metric_name, source_col) in enumerate(
                        display_columns.items()
                    ):
                        value = pd.to_numeric(
                            pd.Series([model_row[source_col]]),
                            errors="coerce"
                        ).iloc[0]

                        if pd.notna(value):
                            with metric_cols[i]:
                                st.metric(
                                    metric_name,
                                    f"{value:.2%}"
                                    if 0 <= value <= 1
                                    else f"{value:.2f}"
                                )

                # Comparison chart.
                if display_columns:
                    chart_df = metrics_df[
                        [model_col] + list(display_columns.values())
                    ].copy()

                    chart_df = chart_df.rename(
                        columns={
                            source: name
                            for name, source in display_columns.items()
                        }
                    )

                    for column in display_columns:
                        chart_df[column] = pd.to_numeric(
                            chart_df[column],
                            errors="coerce"
                        )

                    chart_df = chart_df.set_index(model_col)

                    st.subheader("Performance Metrics Chart")
                    st.bar_chart(chart_df)

                    # Caution if all scores are 100%.
                    values = chart_df.to_numpy().flatten()
                    values = values[pd.notna(values)]

                    if len(values) and all(
                        abs(float(value) - 1.0) < 1e-9
                        for value in values
                    ):
                        st.warning(
                            "All displayed scores are 100%. These results "
                            "should be treated as preliminary. Check for "
                            "overlap between training and test data and "
                            "evaluate on varied, unseen medical records."
                        )

            st.caption(
                "Metrics are loaded from results/metrics.csv, generated "
                "by evaluate.py."
            )

    except Exception as e:
        st.error("Could not load the evaluation metrics.")
        st.exception(e)
else:
    st.info(
        "Evaluation results are not available yet. Run this command "
        "in your project terminal: python evaluate.py"
    )


# Sample model predictions.
outputs_path = RESULTS_DIR / "model_outputs.csv"

if outputs_path.exists():
    with st.expander("View Sample Model Predictions"):
        try:
            outputs_df = pd.read_csv(outputs_path)

            st.dataframe(
                outputs_df.head(100),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Model Outputs CSV",
                data=outputs_path.read_bytes(),
                file_name="model_outputs.csv",
                mime="text/csv",
                key="download_model_outputs"
            )
        except Exception as e:
            st.warning(f"Could not display model predictions: {e}")


# Detailed classification reports.
report_files = [
    RESULTS_DIR / "cnn_report.csv",
    RESULTS_DIR / "decision_tree_report.csv",
    RESULTS_DIR / "random_forest_report.csv",
    RESULTS_DIR / "ensemble_report.csv"
]

existing_reports = [
    path for path in report_files if path.exists()
]

if existing_reports:
    with st.expander("Detailed Classification Reports"):
        for report_path in existing_reports:
            st.subheader(report_path.stem.replace("_", " ").title())

            try:
                report_df = pd.read_csv(report_path)
                st.dataframe(
                    report_df,
                    use_container_width=True,
                    hide_index=True
                )
            except Exception as e:
                st.warning(f"Could not read report: {e}")


st.divider()


# --------------------------------------------------
# USER OPTIONS
# --------------------------------------------------

st.header("🔒 De-identify Medical Records")

mode = st.sidebar.selectbox(
    "De-identification mode",
    ["mask", "pseudonymize"],
    help=(
        "Mask replaces detected information with labels such as [NAME]. "
        "Pseudonymize replaces it with consistent-looking identifiers."
    )
)

st.sidebar.info("Supported files: PDF, CSV, XLSX and XLS")


# --------------------------------------------------
# SENSITIVE COLUMN FALLBACK
# --------------------------------------------------

SENSITIVE_COLUMNS = {
    "name": "NAME",
    "patient name": "NAME",
    "patient_name": "NAME",
    "patientname": "NAME",
    "doctor": "DOCTOR",
    "doctor name": "DOCTOR",
    "doctor_name": "DOCTOR",
    "hospital": "HOSPITAL",
    "hospital name": "HOSPITAL",
    "hospital_name": "HOSPITAL",
    "insurance provider": "INSURANCE",
    "insurance_provider": "INSURANCE",
    "insurance": "INSURANCE",
    "room number": "ROOM_NUMBER",
    "room_number": "ROOM_NUMBER",
    "roomnumber": "ROOM_NUMBER",
    "age": "AGE",
    "date of admission": "DATE",
    "date_of_admission": "DATE",
    "admission date": "DATE",
    "admission_date": "DATE",
    "discharge date": "DATE",
    "discharge_date": "DATE",
    "phone": "PHONE",
    "phone number": "PHONE",
    "phone_number": "PHONE",
    "telephone": "PHONE",
    "email": "EMAIL",
    "email address": "EMAIL",
    "email_address": "EMAIL",
    "address": "ADDRESS",
    "patient address": "ADDRESS",
    "patient_address": "ADDRESS"
}


# --------------------------------------------------
# MASK ONE CELL
# --------------------------------------------------

def mask_value(value, column_name):
    if pd.isna(value):
        return ""

    original_value = str(value)

    if not original_value.strip():
        return original_value

    masked_text, detected_spans = engine.mask(
        original_value,
        mode
    )

    column = str(column_name).strip().casefold()
    label = SENSITIVE_COLUMNS.get(column)

    if not detected_spans and label:
        if mode == "mask":
            return f"[{label}]"

        digest = hashlib.sha256(
            f"{label}:{original_value.casefold()}".encode("utf-8")
        ).hexdigest()[:8]

        return f"{label}_{digest}"

    return masked_text


# --------------------------------------------------
# PROCESS DATAFRAME
# --------------------------------------------------

def process_dataframe(df):
    output_df = df.copy()

    for column in output_df.columns:
        output_df[column] = output_df[column].apply(
            lambda value: mask_value(value, column)
        )

    return output_df


# --------------------------------------------------
# UPLOAD MEDICAL RECORD
# --------------------------------------------------

uploaded = st.file_uploader(
    "Upload a medical record",
    type=["pdf", "xlsx", "xls", "csv"],
    key="medical_record_uploader"
)


# --------------------------------------------------
# PROCESS UPLOADED FILE
# --------------------------------------------------

if uploaded is not None:
    extension = Path(uploaded.name).suffix.lower()

    input_path = OUTPUT_DIR / Path(uploaded.name).name
    input_path.write_bytes(uploaded.getbuffer())

    st.success(f"Uploaded: {uploaded.name}")

    # PDF
    if extension == ".pdf":
        try:
            original_text = read_pdf(input_path)

            masked_text, spans = engine.mask(
                original_text,
                mode
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Medical Record")
                st.text_area(
                    "Original text",
                    original_text,
                    height=350,
                    key="original_pdf_text"
                )

            with col2:
                st.subheader("De-identified Medical Record")
                st.text_area(
                    "De-identified text",
                    masked_text,
                    height=350,
                    key="masked_pdf_text"
                )

            st.metric("Detected PHI Entities", len(spans))

            st.subheader("Detected Entities")

            if spans:
                entity_rows = []

                for start, end, label, confidence in spans:
                    entity_rows.append({
                        "Start": start,
                        "End": end,
                        "Label": label,
                        "Confidence": round(float(confidence), 3),
                        "Detected Text": original_text[start:end]
                    })

                st.dataframe(
                    pd.DataFrame(entity_rows),
                    use_container_width=True
                )
            else:
                st.info(
                    "No PHI entities were detected. Please review "
                    "the output manually."
                )

            output_path = OUTPUT_DIR / "deidentified_output.pdf"
            write_pdf(masked_text, output_path)

            st.download_button(
                "Download De-identified PDF",
                data=output_path.read_bytes(),
                file_name="deidentified_output.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error("Could not process the PDF file.")
            st.exception(e)

    # CSV
    elif extension == ".csv":
        try:
            original_df = pd.read_csv(
                input_path,
                dtype=str
            ).fillna("")

            st.subheader("Original Dataset")
            st.write(
                f"Rows: {len(original_df)} | "
                f"Columns: {len(original_df.columns)}"
            )

            st.dataframe(
                original_df.head(50),
                use_container_width=True
            )

            with st.spinner("De-identifying CSV data..."):
                deidentified_df = process_dataframe(original_df)

            st.success("CSV de-identification completed.")

            st.subheader("De-identified Dataset")

            preview_size = st.selectbox(
                "Number of rows to preview",
                ["50", "100", "500", "All"],
                index=0,
                key="csv_preview_size"
            )

            preview_df = (
                deidentified_df
                if preview_size == "All"
                else deidentified_df.head(int(preview_size))
            )

            st.dataframe(
                preview_df,
                use_container_width=True,
                height=450
            )

            output_path = OUTPUT_DIR / "deidentified_output.csv"
            deidentified_df.to_csv(output_path, index=False)

            st.download_button(
                "Download De-identified CSV",
                data=output_path.read_bytes(),
                file_name="deidentified_output.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Could not process the CSV file.")
            st.exception(e)

    # Excel
    elif extension in [".xlsx", ".xls"]:
        try:
            original_sheets = read_excel(input_path)
            deidentified_sheets = {}

            with st.spinner("De-identifying Excel data..."):
                for sheet_name, df in original_sheets.items():
                    deidentified_sheets[sheet_name] = process_dataframe(df)

            st.success("Excel de-identification completed.")

            for sheet_name, original_df in original_sheets.items():
                st.subheader(f"Sheet: {sheet_name}")

                deidentified_df = deidentified_sheets[sheet_name]

                st.write(
                    f"Rows: {len(original_df)} | "
                    f"Columns: {len(original_df.columns)}"
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original Data**")
                    st.dataframe(
                        original_df.head(50),
                        use_container_width=True,
                        height=350
                    )

                with col2:
                    st.markdown("**De-identified Data**")
                    st.dataframe(
                        deidentified_df.head(50),
                        use_container_width=True,
                        height=350
                    )

            output_path = OUTPUT_DIR / "deidentified_output.xlsx"
            write_excel(deidentified_sheets, output_path)

            st.download_button(
                "Download De-identified Excel",
                data=output_path.read_bytes(),
                file_name="deidentified_output.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
            )

        except Exception as e:
            st.error("Could not process the Excel file.")
            st.exception(e)

    else:
        st.error("Unsupported file format.")