import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fetal Health Predictor", layout="centered")
st.caption("APP VERSION: Immortal")

@st.cache_resource
def load_bundle():
    return joblib.load("fetal_health_rf_bundle.pkl")

bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
top_features = bundle["top_features"]
all_features = bundle["all_features"]

label_map = {
    1: "Normal",
    2: "Suspect",
    3: "Pathological"
}

st.sidebar.header("About the Project")
st.sidebar.write(
    """
    This project uses a Random Forest machine learning model to classify fetal health 
    (Normal, Suspect, Pathological) using Cardiotocography (CTG) features.

    CTG devices monitor fetal heart rate, fetal movements, and uterine contractions.
    In low-resource settings, limited access to trained specialists can delay the 
    interpretation of CTG readings.

    This application provides probability-based predictions to support clinical 
    decision-making and early detection of fetal distress.
    """
)

st.title("Fetal Health Prediction (Random Forest)")
st.write(
    """
    Enter CTG feature values below. The app outputs **probabilities** for each class:

    - **1 = Normal**
    - **2 = Suspect**
    - **3 = Pathological**
    """
)

st.subheader("Input Features")

# Build inputs dynamically from the selected feature list
inputs = {}
for feat in top_features:
    inputs[feat] = st.number_input(
        label=feat,
        value=0.0,
        step=0.1,
        format="%.2f"
    )

# Predict button
if st.button("Predict"):
    # Build a full feature row because the scaler expects all original features
    full_row = {feat: 0.0 for feat in all_features}
    full_row.update(inputs)

    X_new_full = pd.DataFrame([full_row], columns=all_features)

    # Scale using the fitted scaler
    X_new_full_scaled = scaler.transform(X_new_full)

    # Convert back to DataFrame so we can select the same top features
    X_new_full_scaled_df = pd.DataFrame(X_new_full_scaled, columns=all_features)
    X_new_selected = X_new_full_scaled_df[top_features]

    # Predict probabilities
    probs = model.predict_proba(X_new_selected)[0]
    classes = model.classes_

    # Determine predicted class
    pred_index = int(np.argmax(probs))
    pred_class = int(classes[pred_index])

    # Show text probabilities
    st.subheader("Prediction Probabilities")
    for cls, p in zip(classes, probs):
        cls_int = int(cls)
        label = label_map.get(cls_int, str(cls_int))
        st.write(f"Class {cls_int} = {label}: {p*100:.2f}%")

    # Create probability dataframe
    prob_df = pd.DataFrame({
        "Class": [f"{int(cls)} - {label_map[int(cls)]}" for cls in classes],
        "Probability (%)": [p * 100 for p in probs]
    })

    # Plot colored probability bars with embedded percentages
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2E8B57", "#FFA500", "#DC143C"]  # green, orange, red

    bars = ax.bar(
        prob_df["Class"],
        prob_df["Probability (%)"],
        color=colors
    )

    ax.set_ylabel("Probability (%)")
    ax.set_title("Prediction Probabilities")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add labels inside bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,
            f"{height:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=11
        )

    st.pyplot(fig)

    # Final prediction
    st.success(
        f"Final Prediction = Class {pred_class} → {label_map.get(pred_class, pred_class)}"
    )