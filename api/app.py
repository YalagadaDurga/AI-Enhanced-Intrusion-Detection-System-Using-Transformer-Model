import streamlit as st
import torch
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import sys
import os
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI-IDS",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------------------------
# HIGH CONTRAST DARK THEME
# -------------------------------------------------
st.markdown("""
<style>

/* GLOBAL TEXT FIX */
body {
    color: #FFFFFF;
}

/* File uploader label text */
label, .stFileUploader label {
    color: #FFFFFF !important;
    font-weight: 500;
}

/* Drag & drop text */
[data-testid="stFileUploader"] section {
    color: #FFFFFF !important;
}

/* Browse button text */
[data-testid="stFileUploader"] button {
    color: #FFFFFF !important;
    background-color: #1B2438 !important;
}

/* MAIN BACKGROUND */
.stApp {
    background-color: #0A0F1F;
    color: #FFFFFF;
    font-family: 'Segoe UI', sans-serif;
}

/* HEADINGS */
h1, h2, h3, h4 {
    color: #00F5D4 !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #111827;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* TOP HEADER */
.topbar {
    background: #111827;
    padding: 20px 10px 15px 10px;
    border-bottom: 1px solid #1f2937;
    text-align: center;
}
.topbar h1 {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    background: linear-gradient(90deg, #00BBF9, #3A86FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
    text-shadow: 0px 0px 10px rgba(76, 201, 240, 0.6);
}
            
.topbar p {
    font-size: 15px;
    color: #9CA3AF;
    margin: 0;
    letter-spacing: 0.5px;
}

/* CARDS */
.card {
    background: #1B2438;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #2a3655;
}

/* KPI NUMBERS */
.kpi-number {
    font-size: 28px;
    font-weight: bold;
    color: #FFFFFF !important;
}
            
/* REMOVE TOP WHITE SPACE */
.block-container {
    padding-top: 2rem !important;
}

header {
    visibility: hidden;
}

[data-testid="stToolbar"] {
    display: none;
}

/* BUTTON */
.stButton>button {
    background-color: #00BBF9;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}

/* FILE UPLOADER */
/* ===== FORCE DARK UPLOAD DROPZONE (LATEST STREAMLIT FIX) ===== */

/* Entire dropzone */
[data-testid="stFileUploaderDropzone"] {
    background-color: #1B2438 !important;
    border: 2px dashed #2a3655 !important;
    border-radius: 12px !important;
}

/* Main drag text */
[data-testid="stFileUploaderDropzone"] span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

/* Small helper text */
[data-testid="stFileUploaderDropzone"] small {
    color: #9CA3AF !important;
}

/* Remove light background wrapper */
[data-testid="stFileUploader"] {
    background: transparent !important;
}

/* DATAFRAME FIX */
[data-testid="stDataFrame"] {
    background-color: #1B2438 !important;
    color: white !important;
}

/* METRIC FIX */
[data-testid="stMetricValue"] {
    color: #00F5D4 !important;
    font-weight: bold;
}

/* ALERTS */
.alert-danger {
    background-color: #3A0000;
    border-left: 5px solid red;
    padding: 15px;
    border-radius: 8px;
}
.alert-safe {
    background-color: #003A1F;
    border-left: 5px solid lime;
    padding: 15px;
    border-radius: 8px;
}

footer {visibility:hidden;}
#MainMenu {visibility:hidden;}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
<div class="topbar">
    <h1>🛡️ AI Enhanced Intrusion Detection System</h1>
    <p>Transformer-Based Network Security Monitoring with Explainable AI (XAI)</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# MODEL IMPORT
# -------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.transformer_model import TransformerIDS

INPUT_DIM = 122
NUM_CLASSES = 5

label_map = {
    0: "Normal",
    1: "DoS",
    2: "Probe",
    3: "R2L",
    4: "U2R"
}

@st.cache_resource
def load_model():
    encoder = joblib.load("data/processed/encoder.pkl")

    model = TransformerIDS(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256
    )

    model.load_state_dict(torch.load("model/ids_transformer.pt", map_location="cpu"))
    model.eval()
    return encoder, model

encoder, model = load_model()

# -------------------------------------------------
# ATTACK ALERT POPUP
# -------------------------------------------------
@st.dialog("🚨 Security Alert")
def attack_alert():

    st.warning(
        """
        Intrusion activity has been detected in the uploaded network traffic.

        ⚠ Immediate investigation is recommended.

        For detailed analysis of the detected attack,
        please navigate to the **Threat Insights** page.
        """
    )

    if st.button("OK"):
        st.session_state.show_alert = False
        st.rerun()

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
st.sidebar.title("⚙️ Control Panel")
page = st.sidebar.radio(
    "Go To :",
    ["🔍 Detection", "📊 Metrics", "🧠 Threat Insights"]
)

with st.sidebar:
    st.markdown("### Controls")

    if st.button("🔄 Reset Detection", use_container_width=True):
        st.session_state.clear()
        st.rerun()


st.sidebar.markdown(
    """
    <div style="
        background-color:#1e293b;
        padding:15px;
        border-radius:10px;
        border:1px solid #334155;
        font-size:14px;
    ">
        <b>📌 Features:</b><br><br>
        • Upload RAW or PROCESSED dataset<br>
        • Automatic preprocessing<br>
        • Risk visualization<br>
        • Explainable results
    </div>
    """,
    unsafe_allow_html=True
)



# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "df_pred" not in st.session_state:
    st.session_state.df_pred = None
    st.session_state.y_pred = None
    st.session_state.y_true = None

if "show_alert" not in st.session_state:
    st.session_state.show_alert = False

# =================================================
# DETECTION
# =================================================
if page == "🔍 Detection":

    st.subheader("Upload Network Traffic Dataset")
    uploaded_file = st.file_uploader("📂 Upload CSV File", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Run Intrusion Detection", use_container_width=True):

            label_col = None
            for col in ["label", "attack", "attack_type"]:
                if col in df.columns:
                    label_col = col
                    break

            y_true = None
            if label_col:
                y_true = df[label_col].values
                df = df.drop(columns=[label_col])

            if df.shape[1] == INPUT_DIM:
                X = df.values
            else:
                X = encoder.transform(df)

            X = torch.tensor(X, dtype=torch.float32)

            with st.spinner("Analyzing network traffic..."):
                with torch.no_grad():
                    outputs = model(X)
                preds = torch.argmax(outputs, dim=1).numpy()

            df["Prediction"] = [label_map[p] for p in preds]

            st.session_state.df_pred = df
            st.session_state.y_pred = preds
            st.session_state.y_true = y_true

            st.success("Detection Completed Successfully")

            if (df["Prediction"] != "Normal").sum() > 0:
                st.session_state.show_alert = True

    if st.session_state.df_pred is not None:

        df = st.session_state.df_pred
        total = len(df)
        attacks = (df["Prediction"] != "Normal").sum()
        normal = total - attacks
        risk = int((attacks / total) * 100)
        if risk < 10:
            risk_color = "lime"
        elif risk < 40:
            risk_color = "orange"
        else:
            risk_color = "red"

        col1, col2, col3, col4 = st.columns(4)

        col1.markdown(f"<div class='card'><div>Total Traffic</div><div class='kpi-number'>{total}</div></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='card'><div>Attacks</div><div class='kpi-number' style='color:red'>{attacks}</div></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='card'><div>Normal</div><div class='kpi-number' style='color:lime'>{normal}</div></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='card'><div>Risk Level</div><div class='kpi-number' style='color:{risk_color}'>{risk}%</div></div>",unsafe_allow_html=True)
        

        st.write("")

        if attacks > 0:
            st.markdown("<div class='alert-danger'>🚨 ACTIVE THREAT DETECTED — Immediate Security Investigation Required</div>",unsafe_allow_html=True)

            if st.session_state.show_alert:
                attack_alert()
        else:
            st.markdown("<div class='alert-safe'>🟢 Network Secure</div>", unsafe_allow_html=True)

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### 📊 Attack Distribution")
            counts = df["Prediction"].value_counts()
            fig, ax = plt.subplots(facecolor="#0A0F1F")
            ax.bar(counts.index, counts.values)
            ax.set_facecolor("#0A0F1F")
            ax.tick_params(colors='white')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with colB:
            st.subheader("Live Detection Log")
            st.dataframe(df.tail(20), use_container_width=True)

# =================================================
# METRICS
# =================================================
elif page == "📊 Metrics":

    if st.session_state.df_pred is None:
        st.info("Run detection first.")
    else:
        y_pred = st.session_state.y_pred
        y_true = st.session_state.y_true

        if y_true is not None:

            # -----------------------------
            # METRIC CALCULATION (Macro only)
            # -----------------------------
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro')
            rec = recall_score(y_true, y_pred, average='macro')
            f1 = 2 * (prec * rec) / (prec + rec + 1e-8)

            st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)

            # -----------------------------
            # KPI CARDS
            # -----------------------------
            col1, col2, col3, col4 = st.columns(4)

            def metric_card(label, value):
                st.markdown(
                    f"""
                    <div style='background:#1B2438;padding:20px;border-radius:12px;
                                border:1px solid #2a3655;text-align:center'>
                        <div style='color:#9CA3AF;font-size:14px;margin-bottom:8px'>{label}</div>
                        <div style='font-size:34px;font-weight:bold;color:#00F5D4'>{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col1:
                metric_card("Accuracy", f"{acc*100:.2f}%")
            with col2:
                metric_card("Precision", f"{prec:.3f}")
            with col3:
                metric_card("Recall", f"{rec:.3f}")
            with col4:
                metric_card("F1 Score", f"{f1:.3f}")

            # -----------------------------
            # CONFUSION MATRIX (SMALL + CENTERED)
            # -----------------------------
            st.markdown("""
            <h3 style='color:#00BBF9; margin-top:60px; margin-bottom:25px;'>
            📊 Confusion Matrix
            </h3>
            """, unsafe_allow_html=True)

            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(4.5,3.8))
            im = ax.imshow(cm, cmap="Blues")

            ax.set_xticks(range(len(label_map)))
            ax.set_yticks(range(len(label_map)))
            ax.set_xticklabels(label_map.values(), fontsize=8)
            ax.set_yticklabels(label_map.values(), fontsize=8)

            for i in range(len(label_map)):
                for j in range(len(label_map)):
                    ax.text(j, i, cm[i, j],
                            ha="center", va="center",
                            fontsize=7, color="black")

            plt.tight_layout()

            col_left, col_mid, col_right = st.columns([1.2,2,1.2])
            with col_mid:
                st.pyplot(fig)

            # -----------------------------
            # CLASSIFICATION REPORT (Styled + Clean Accuracy)
            # -----------------------------
            st.markdown("""
            <h3 style='color:#3A86FF; margin-top:60px; margin-bottom:25px;'>
            📑 Classification Report
            </h3>
            """, unsafe_allow_html=True)

            report = classification_report(
                y_true,
                y_pred,
                target_names=label_map.values(),
                output_dict=True
            )

            report_df = pd.DataFrame(report).transpose().round(3)

            class_rows = list(label_map.values())
            class_df = report_df.loc[class_rows][
                ["precision", "recall", "f1-score", "support"]
            ]

            accuracy_value = float(report_df.loc["accuracy"][0])

            # Heat color function (clamped between 0–1)
            def heat(val):
                val = max(0, min(val, 1))
                return f"""
                    background-color: rgba(0,187,249,{val});
                    color:white;
                    padding:8px;
                    border:1px solid #2a3655;
                """

            html = """
            <table style='width:100%; border-collapse:collapse; text-align:center;
                    background:#111827; border-radius:8px; overflow:hidden;'>

            <tr style='background:#1B2438; color:white; font-weight:600;'>
                <th style='padding:10px;'>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>Support</th>
            </tr>
            """

            # Per-class rows
            for label in class_rows:
                row = class_df.loc[label]
                html += "<tr>"
                html += f"<td style='padding:8px; border:1px solid #2a3655;'>{label}</td>"
                html += f"<td style='{heat(row['precision'])}'>{row['precision']:.3f}</td>"
                html += f"<td style='{heat(row['recall'])}'>{row['recall']:.3f}</td>"
                html += f"<td style='{heat(row['f1-score'])}'>{row['f1-score']:.3f}</td>"
                html += f"<td style='padding:8px; border:1px solid #2a3655;'>{int(row['support'])}</td>"
                html += "</tr>"

            # Accuracy row (full width highlight)
            html += f"""
            <tr style='background:#0A0F1F; font-weight:bold; border-top:2px solid #00F5D4;'>
                <td colspan='5' style='padding:14px; color:#00F5D4; font-size:17px;'>
                    Overall Accuracy : {accuracy_value:.3f}
                </td>
            </tr>
            """

            html += "</table>"

            col1, col2, col3 = st.columns([1.2, 2, 1.2])
            with col2:
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.warning("No labels found in dataset.")
# =================================================
# Threat Insights
# =================================================
elif page == "🧠 Threat Insights":

    if st.session_state.df_pred is None:
        st.info("Run detection first.")
    else:

        df = st.session_state.df_pred
        attack_df = df[df["Prediction"] != "Normal"]

        if attack_df.empty:
            st.success("No attacks detected.")
        else:
            st.subheader("Select Attack Sample")

            idx = st.selectbox("Choose sample index", attack_df.index)

            sample = df.loc[idx]
            prediction = sample["Prediction"]

            st.markdown(f"## 🚨 Predicted Attack Type: **{prediction}**")

            # Remove prediction column
            feature_values = sample.drop("Prediction")

            

            # --------------------------------------------------
            # Dynamic Context Based on Attack Type
            # --------------------------------------------------

            attack_context = {
                "DoS": {
                    "risk": "🔴 High Risk",
                    "description": """
A Denial-of-Service (DoS) attack attempts to overwhelm network resources 
by flooding the system with excessive traffic.

This can cause service downtime, degraded performance, and resource exhaustion.
""",
                    "impact": """
• Service disruption  
• Server overload  
• Network bandwidth exhaustion  
• Legitimate users denied access  
""",
                    "mitigation": """
• Enable rate limiting  
• Deploy traffic filtering (WAF / Firewall rules)  
• Activate DDoS protection services  
• Monitor abnormal traffic spikes  
"""
                },

                "Probe": {
                    "risk": "🟠 Medium Risk",
                    "description": """
A Probe attack scans the network to gather information 
about open ports, vulnerabilities, and services.

It is typically reconnaissance before a larger attack.
""",
                    "impact": """
• Exposure of open ports  
• Vulnerability identification  
• Potential future exploitation  
""",
                    "mitigation": """
• Disable unused ports  
• Use intrusion prevention systems (IPS)  
• Enable network monitoring alerts  
• Patch exposed services  
"""
                },

                "R2L": {
                    "risk": "🔴 High Risk",
                    "description": """
Remote-to-Local (R2L) attacks attempt to gain local access 
from a remote machine without authorization.

These often involve password guessing or exploiting authentication flaws.
""",
                    "impact": """
• Unauthorized system access  
• Data theft  
• Credential compromise  
""",
                    "mitigation": """
• Enforce strong password policies  
• Enable multi-factor authentication  
• Monitor failed login attempts  
• Restrict remote access services  
"""
                },

                "U2R": {
                    "risk": "🔴 Critical Risk",
                    "description": """
User-to-Root (U2R) attacks attempt to escalate privileges 
from a normal user account to root/administrator level.

This is highly dangerous as it grants full system control.
""",
                    "impact": """
• Full system compromise  
• Privilege escalation  
• Malware installation  
• Sensitive data manipulation  
""",
                    "mitigation": """
• Apply strict access controls  
• Monitor privilege escalation logs  
• Keep OS and software updated  
• Implement endpoint protection  
"""
                }
            }

            if prediction in attack_context:
                context = attack_context[prediction]

                st.markdown(f"### ⚠ Risk Level: {context['risk']}")

                st.markdown("### 📖 Attack Description")
                st.markdown(context["description"])

                st.markdown("### 💥 Potential Impact")
                st.markdown(context["impact"])

                st.markdown("### 🛡 Recommended Mitigation")
                st.markdown(context["mitigation"])

            else:
                st.markdown("""
### ℹ Normal Traffic

This network flow is classified as normal behavior.

No immediate security action required.
""")