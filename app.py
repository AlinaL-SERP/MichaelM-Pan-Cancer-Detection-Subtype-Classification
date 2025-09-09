import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import io

m1_model = joblib.load("M1fin_model.pkl")
m1_scaler = joblib.load("M1fin_scaler.pkl")
m2_model = joblib.load("subtype_model.pkl")
m2_scaler = joblib.load("subtype_scaler.pkl")

def stacked_predict(sample, threshold=0.7):
    sample = np.array(sample)
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    y_pred, y_prob, y_binary = [], [], []
    for i in range(sample.shape[0]):
        s = sample[i].reshape(1, -1)
        s_scaled = m1_scaler.transform(s)
        cancer_prob = m1_model.predict_proba(s_scaled)[0, 1]
        if cancer_prob < threshold:
            y_pred.append("HC")
            y_binary.append("Healthy")
        else:
            s_scaled_m2 = m2_scaler.transform(s)
            subtype = m2_model.predict(s_scaled_m2)[0]
            y_pred.append(subtype)
            y_binary.append("Cancer")
        y_prob.append(cancer_prob)
    return y_pred, y_prob, y_binary


st.title("Pan-Cancer Detection & Subtype Classification")

st.write("""
Upload an Excel file (`.xlsx`) with your samples.
- Columns must match the 30 features used for training.
- Optionally include a 'label' column for evaluation.
""")

st.write("From Alina Liu & Michael Manganel, 2025 SERP")

uploaded_file = st.file_uploader("Choose Excel file", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    if 'label' in df.columns:
        y_true = df['label'].values
        X_input = df.drop(columns=['label'])
    else:
        y_true = None
        X_input = df

    features = ["ADAM10", "RAB7A", "PDCD6IP", "ZYX", "EHD3", "RP2", "NAPA", "GPX1", "EEF1G", "CDC42", "F11R", "KPNB1", "SDPR", "GNB2", "SERPINB1", "APOA4", "ACTR1A", "APMAP", "SLC4A1", "MDH1", "SLC44A1", "CCT4", "CD47", "RAP2B", "FLOT1", "A1BG", "ITGB2", "FERMT3", "EEF1D", "GNG5"]

    X_input = X_input[features].values
    y_pred, y_prob, y_binary = stacked_predict(X_input, threshold=0.62)

    df['Predicted_Label'] = y_pred
    df['Cancer_Probability'] = y_prob
    df['Healthy_or_Cancer'] = y_binary

    st.subheader("Predictions")
    st.dataframe(df)

    if y_true is not None:
        y_true = np.array([str(lbl).strip() for lbl in y_true])
        y_pred_clean = np.array([str(lbl).strip() for lbl in y_pred])

        # Binary ground truth (Healthy vs Cancer)
        y_true_binary = np.array(["Healthy" if lbl=="HC" else "Cancer" for lbl in y_true])

        # Metrics
        auroc = roc_auc_score([0 if lbl=="HC" else 1 for lbl in y_true], y_prob)
        acc_multi = accuracy_score(y_true, y_pred_clean)
        acc_binary = accuracy_score(y_true_binary, y_binary)

        labels_cm = ["HC","CRC","LC","GC","PC"]
        cm = confusion_matrix(y_true, y_pred_clean, labels=labels_cm)

        st.subheader("Evaluation Metrics")
        st.write(f"AUROC (Cancer vs Healthy): {auroc:.3f}")
        st.write(f"Accuracy (Binary Healthy vs Cancer): {acc_binary:.3f}")
        st.write(f"Accuracy (All Classes): {acc_multi:.3f}")
        st.write("Confusion Matrix:")
        st.write(pd.DataFrame(cm, index=labels_cm, columns=labels_cm))

    # --- Excel download using BytesIO ---
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)

    st.download_button(
        "Download Predictions",
        data=output,
        file_name="predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
