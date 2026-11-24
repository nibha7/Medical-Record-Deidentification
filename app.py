
import streamlit as st
import pandas as pd
import re
import io
import json
import torch
from pathlib import Path
from datetime import datetime

from common import tokenize_with_offsets

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MedDeID | Healthcare Privacy",
    page_icon="🔒",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "bilstm_crf" / "model.pt"
VOCAB_PATH = BASE_DIR / "models" / "bilstm_crf" / "vocab.json"
LABELS_PATH = BASE_DIR / "models" / "bilstm_crf" / "labels.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================================
# PAGE STYLE
# =========================================================

st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: bold;
    color: #15365c;
}
.sub-title {
    color: #64748b;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# REGEX PHI DETECTION
# =========================================================

REGEX_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",

    "PHONE": r"(?<!\w)(?:\+91[\s-]?)?[6-9]\d{9}(?!\w)",

    "DATE": (
        r"\b(?:"
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"|"
        r"\d{4}-\d{2}-\d{2}"
        r")\b"
    ),

    "ID": (
        r"\b(?:MRN|Medical Record Number|Patient ID|"
        r"Patient Number|Health ID)\s*[:#-]?\s*"
        r"[A-Za-z0-9-]+\b"
    )
}


def regex_predictions(text):

    spans = []

    for label, pattern in REGEX_PATTERNS.items():

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            spans.append((
                match.start(),
                match.end(),
                label,
                1.0
            ))

    return spans


# =========================================================
# LOAD TRAINED BILSTM-CRF MODEL
# =========================================================

@st.cache_resource
def load_bilstm_model():

    from train_bilstm_crf import BiLSTMCRF

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "BiLSTM-CRF model checkpoint not found."
        )

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            "BiLSTM-CRF vocabulary file not found."
        )

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            "BiLSTM-CRF labels file not found."
        )

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        bio_labels = json.load(f)

    model = BiLSTMCRF(
        len(vocab),
        len(bio_labels)
    )

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model, vocab, bio_labels


# =========================================================
# MODEL PREDICTION
# =========================================================

def bilstm_predictions(text, model, vocab, bio_labels):

    if not isinstance(text, str) or not text.strip():
        return []

    tokens = tokenize_with_offsets(text)

    if not tokens:
        return []

    words = [
        token[0].lower()
        for token in tokens
    ]

    token_ids = [
        vocab.get(word, vocab.get("<UNK>", 1))
        for word in words
    ]

    input_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )

    mask = input_tensor != 0

    with torch.no_grad():
        predicted_ids = model.predict(
            input_tensor,
            mask
        )[0]

    predicted_tags = [
        bio_labels[index]
        for index in predicted_ids
    ]

    # Convert BIO token labels into character spans.
    spans = []

    active_label = None
    active_start = None
    active_end = None

    for index, tag in enumerate(predicted_tags):

        token, start, end = tokens[index]

        if tag == "O":

            if active_label is not None:
                spans.append((
                    active_start,
                    active_end,
                    active_label,
                    0.75
                ))

            active_label = None
            active_start = None
            active_end = None

            continue

        if "-" in tag:
            prefix, label = tag.split("-", 1)
        else:
            prefix, label = "B", tag

        if label == "O":
            continue

        if prefix == "B" or active_label != label:

            if active_label is not None:
                spans.append((
                    active_start,
                    active_end,
                    active_label,
                    0.75
                ))

            active_label = label
            active_start = start
            active_end = end

        else:
            # Continue the same entity, including whitespace.
            active_end = end

    if active_label is not None:

        spans.append((
            active_start,
            active_end,
            active_label,
            0.75
        ))

    return spans


# =========================================================
# MERGE MODEL AND REGEX PREDICTIONS
# =========================================================

def merge_predictions(text, model_spans, regex_spans):

    candidates = []

    for span in model_spans:
        candidates.append((
            span[0],
            span[1],
            span[2],
            span[3],
            False
        ))

    for span in regex_spans:
        candidates.append((
            span[0],
            span[1],
            span[2],
            span[3],
            True
        ))

    # Regex predictions have priority for overlapping spans.
    candidates.sort(
        key=lambda x: (
            x[4],
            x[3],
            x[1] - x[0]
        ),
        reverse=True
    )

    selected = []

    for start, end, label, confidence, is_regex in candidates:

        if start < 0 or end > len(text) or start >= end:
            continue

        overlaps = any(
            not (
                end <= old_start
                or start >= old_end
            )
            for old_start, old_end, _, _ in selected
        )

        if not overlaps:

            selected.append((
                start,
                end,
                label,
                confidence
            ))

    return sorted(
        selected,
        key=lambda x: x[0]
    )


# =========================================================
# REDACTION
# =========================================================

def redact_spans(text, spans):

    output = text

    for start, end, label, confidence in sorted(
        spans,
        key=lambda x: x[0],
        reverse=True
    ):

        output = (
            output[:start]
            + "[" + label + "]"
            + output[end:]
        )

    return output


def detect_and_redact(text, model_data):

    if not isinstance(text, str):
        return text, []

    model, vocab, bio_labels = model_data

    model_spans = bilstm_predictions(
        text,
        model,
        vocab,
        bio_labels
    )

    regex_spans = regex_predictions(text)

    combined = merge_predictions(
        text,
        model_spans,
        regex_spans
    )

    redacted = redact_spans(
        text,
        combined
    )

    return redacted, combined


# =========================================================
# STRUCTURED COLUMN DETECTION
# =========================================================

def normalize_column(column):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(column).lower()
    )


COLUMN_LABELS = {

    "name": "PATIENT",
    "patient": "PATIENT",
    "patientname": "PATIENT",
    "fullname": "PATIENT",

    "age": "AGE",
    "patientage": "AGE",

    "dateofadmission": "DATE",
    "admissiondate": "DATE",
    "dischargedate": "DATE",
    "dateofdischarge": "DATE",
    "date": "DATE",

    "doctor": "DOCTOR",
    "doctorname": "DOCTOR",
    "physician": "DOCTOR",

    "hospital": "HOSPITAL",
    "hospitalname": "HOSPITAL",

    "insurance": "INSURANCE",
    "insuranceprovider": "INSURANCE",

    "roomnumber": "ROOM_NUMBER",
    "roomno": "ROOM_NUMBER",

    "phone": "PHONE",
    "phonenumber": "PHONE",
    "mobile": "PHONE",
    "mobilenumber": "PHONE",

    "email": "EMAIL",
    "emailaddress": "EMAIL",

    "patientid": "ID",
    "medicalrecordnumber": "ID",
    "mrn": "ID",
    "uhid": "ID"
}


# =========================================================
# PROCESS A DATAFRAME
# =========================================================

def process_dataframe(df, model_data):

    df = df.copy()

    total_cells = 0
    total_detected = 0

    for column in df.columns:

        normalized = normalize_column(column)

        column_label = COLUMN_LABELS.get(
            normalized
        )

        for index in df.index:

            value = df.at[index, column]

            if pd.isna(value):
                continue

            value = str(value).strip()

            if not value:
                continue

            total_cells += 1

            # Structured column: replace the whole value.
            if column_label:

                df.at[index, column] = (
                    "[" + column_label + "]"
                )

                total_detected += 1

            else:

                # Run NER on longer free-text cells.
                # Regex still checks every nonempty cell.
                run_model = len(value) >= 40

                if run_model:
                    redacted, spans = detect_and_redact(
                        value,
                        model_data
                    )
                else:
                    spans = regex_predictions(value)
                    redacted = redact_spans(
                        value,
                        spans
                    )

                df.at[index, column] = redacted
                total_detected += len(spans)

    return df, total_cells, total_detected


# =========================================================
# SESSION STATE
# =========================================================

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []


def add_audit(
    filename,
    file_type,
    total_cells,
    detected,
    status
):

    st.session_state.audit_log.append({

        "Date and Time": datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        ),

        "File Name": filename,

        "File Type": file_type,

        "Values Processed": total_cells,

        "PHI Replaced/Detected": detected,

        "Status": status
    })


# =========================================================
# LOAD MODEL
# =========================================================

model_data = None
model_error = None

try:

    model_data = load_bilstm_model()

except Exception as error:

    model_error = str(error)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">MedDeID</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Medical Record De-identification using NLP and Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

if model_data is not None:

    st.success(
        "BiLSTM-CRF model loaded successfully."
    )

else:

    st.warning(
        "BiLSTM-CRF model could not be loaded: "
        + str(model_error)
        + ". Structured column replacement and regex detection "
        "will still work."
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("MedDeID")

page = st.sidebar.radio(
    "Navigation",
    [
        "Model Performance",
        "Healthcare Privacy"
    ]
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

if page == "Model Performance":

    st.title("Model Performance")

    performance_path = (
        BASE_DIR / "results" / "model_comparison.csv"
    )

    if performance_path.exists():

        results = pd.read_csv(performance_path)

        st.subheader("Model Comparison")

        st.dataframe(
            results,
            use_container_width=True
        )

        st.subheader("Performance Graph")

        numeric_columns = results.select_dtypes(
            include="number"
        ).columns.tolist()

        model_column = next(
            (
                column for column in results.columns
                if str(column).lower() in [
                    "model",
                    "model_name",
                    "algorithm"
                ]
            ),
            None
        )

        if model_column and numeric_columns:

            chart_data = results.set_index(
                model_column
            )

            st.bar_chart(
                chart_data[numeric_columns]
            )

        st.download_button(
            "Download Model Comparison",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "Model comparison file not found. "
            "Run evaluate_models.py first."
        )


# =========================================================
# HEALTHCARE PRIVACY
# =========================================================

elif page == "Healthcare Privacy":

    st.title("Healthcare Privacy")

    st.write(
        "Detect and replace personally identifiable "
        "information in medical records."
    )

    tab1, tab2, tab3 = st.tabs([
        "Single Record",
        "Bulk Documents and Excel",
        "Audit Report"
    ])


    # =====================================================
    # SINGLE RECORD
    # =====================================================

    with tab1:

        st.subheader("Single Medical Record")

        input_text = st.text_area(
            "Enter medical record",
            height=200,
            placeholder=(
                "Enter patient details or a medical report here."
            )
        )

        if st.button(
            "Detect and De-identify",
            key="single_button"
        ):

            if not input_text.strip():

                st.warning("Enter medical text first.")

            else:

                if model_data is not None:

                    redacted, spans = detect_and_redact(
                        input_text,
                        model_data
                    )

                else:

                    spans = regex_predictions(input_text)

                    redacted = redact_spans(
                        input_text,
                        spans
                    )

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("Original Record")

                    st.text_area(
                        "Original",
                        value=input_text,
                        height=200,
                        disabled=True,
                        key="single_original"
                    )

                with col2:

                    st.subheader("De-identified Record")

                    st.text_area(
                        "Redacted",
                        value=redacted,
                        height=200,
                        disabled=True,
                        key="single_redacted"
                    )

                if spans:

                    detection_df = pd.DataFrame([
                        {
                            "PHI Type": span[2],
                            "Detected Text": input_text[
                                span[0]:span[1]
                            ]
                        }
                        for span in spans
                    ])

                    st.subheader("Detected PHI")

                    st.dataframe(
                        detection_df,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No supported PHI patterns were detected."
                    )

                st.download_button(
                    "Download De-identified Record",
                    data=redacted,
                    file_name="deidentified_record.txt",
                    mime="text/plain"
                )

                add_audit(
                    "Single Record",
                    "TEXT",
                    1,
                    len(spans),
                    "Completed"
                )


    # =====================================================
    # BULK FILE PROCESSING
    # =====================================================

    with tab2:

        st.subheader("Upload Medical Documents")

        st.write(
            "Supported formats: CSV, XLSX, XLS and TXT."
        )

        uploaded_files = st.file_uploader(
            "Choose medical records",
            type=["csv", "xlsx", "xls", "txt"],
            accept_multiple_files=True
        )

        if uploaded_files:

            if st.button(
                "Process Uploaded Files",
                key="bulk_button"
            ):

                for uploaded_file in uploaded_files:

                    filename = uploaded_file.name

                    extension = filename.rsplit(
                        ".",
                        1
                    )[-1].lower()

                    st.divider()

                    st.write("Processing:", filename)

                    try:

                        # ---------------------------------
                        # CSV
                        # ---------------------------------

                        if extension == "csv":

                            uploaded_file.seek(0)

                            df = None

                            for encoding in [
                                "utf-8-sig",
                                "utf-8",
                                "cp1252",
                                "latin1"
                            ]:

                                try:

                                    uploaded_file.seek(0)

                                    df = pd.read_csv(
                                        uploaded_file,
                                        dtype=str,
                                        keep_default_na=False,
                                        encoding=encoding
                                    )

                                    break

                                except UnicodeDecodeError:
                                    continue

                            if df is None:
                                raise ValueError(
                                    "Could not read CSV file."
                                )

                            processed, cells, detected = (
                                process_dataframe(
                                    df,
                                    model_data
                                )
                            )

                            output = io.StringIO()

                            processed.to_csv(
                                output,
                                index=False
                            )

                            csv_bytes = output.getvalue().encode(
                                "utf-8-sig"
                            )

                            st.success(
                                "CSV de-identification completed."
                            )

                            st.write("Rows:", len(processed))
                            st.write("Values processed:", cells)
                            st.write("PHI replaced/detected:", detected)

                            st.download_button(
                                "Download De-identified CSV",
                                data=csv_bytes,
                                file_name=(
                                    "deidentified_"
                                    + filename.rsplit(".", 1)[0]
                                    + ".csv"
                                ),
                                mime="text/csv",
                                key="csv_" + filename
                            )

                            add_audit(
                                filename,
                                "CSV",
                                cells,
                                detected,
                                "Completed"
                            )


                        # ---------------------------------
                        # EXCEL
                        # ---------------------------------

                        elif extension in ["xlsx", "xls"]:

                            uploaded_file.seek(0)

                            sheets = pd.read_excel(
                                uploaded_file,
                                sheet_name=None,
                                dtype=str
                            )

                            output_sheets = {}

                            total_cells = 0
                            total_detected = 0

                            for sheet_name, df in sheets.items():

                                processed, cells, detected = (
                                    process_dataframe(
                                        df,
                                        model_data
                                    )
                                )

                                output_sheets[sheet_name] = processed

                                total_cells += cells
                                total_detected += detected

                            excel_output = io.BytesIO()

                            with pd.ExcelWriter(
                                excel_output,
                                engine="openpyxl"
                            ) as writer:

                                for sheet_name, df in output_sheets.items():

                                    df.to_excel(
                                        writer,
                                        sheet_name=sheet_name[:31],
                                        index=False
                                    )

                            st.success(
                                "Excel de-identification completed."
                            )

                            st.write(
                                "Sheets processed:",
                                len(output_sheets)
                            )

                            st.write(
                                "Values processed:",
                                total_cells
                            )

                            st.write(
                                "PHI replaced/detected:",
                                total_detected
                            )

                            st.download_button(
                                "Download De-identified Excel",
                                data=excel_output.getvalue(),
                                file_name=(
                                    "deidentified_"
                                    + filename.rsplit(".", 1)[0]
                                    + ".xlsx"
                                ),
                                mime=(
                                    "application/vnd.openxmlformats-"
                                    "officedocument.spreadsheetml.sheet"
                                ),
                                key="excel_" + filename
                            )

                            add_audit(
                                filename,
                                "EXCEL",
                                total_cells,
                                total_detected,
                                "Completed"
                            )


                        # ---------------------------------
                        # TXT
                        # ---------------------------------

                        elif extension == "txt":

                            uploaded_file.seek(0)

                            text = uploaded_file.read().decode(
                                "utf-8",
                                errors="replace"
                            )

                            if model_data is not None:

                                redacted, spans = detect_and_redact(
                                    text,
                                    model_data
                                )

                            else:

                                spans = regex_predictions(text)

                                redacted = redact_spans(
                                    text,
                                    spans
                                )

                            st.success(
                                "TXT de-identification completed."
                            )

                            st.write(
                                "PHI detected:",
                                len(spans)
                            )

                            st.download_button(
                                "Download De-identified TXT",
                                data=redacted.encode("utf-8"),
                                file_name=(
                                    "deidentified_"
                                    + filename.rsplit(".", 1)[0]
                                    + ".txt"
                                ),
                                mime="text/plain",
                                key="txt_" + filename
                            )

                            add_audit(
                                filename,
                                "TXT",
                                1,
                                len(spans),
                                "Completed"
                            )

                    except Exception as error:

                        st.error(
                            "Error processing "
                            + filename
                            + ": "
                            + str(error)
                        )

                        add_audit(
                            filename,
                            extension.upper(),
                            0,
                            0,
                            "Failed: " + str(error)
                        )


    # =====================================================
    # AUDIT REPORT
    # =====================================================

    with tab3:

        st.subheader("Processing Audit Report")

        if st.session_state.audit_log:

            audit_df = pd.DataFrame(
                st.session_state.audit_log
            )

            st.dataframe(
                audit_df,
                use_container_width=True
            )

            st.download_button(
                "Download Audit Report",
                data=audit_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="meddeid_audit_report.csv",
                mime="text/csv"
            )

        else:

            st.info(
                "No files have been processed in this session."
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "MedDeID | Medical Record De-identification"
)

st.caption(
    "This prototype combines structured column replacement, "
    "regex detection and BiLSTM-CRF predictions. "
    "Results require validation before real clinical use."
)
