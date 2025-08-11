import streamlit as st
import pandas as pd
import requests
from PIL import Image
import io
import base64


# -----------------------
# API Base URL
# -----------------------
API_URL = "http://127.0.0.1:5000"

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Pneumonia Detection", "Clinician Feedback Stats"])

if page == "Pneumonia Detection":
    st.title("Pneumonia Detection with Grad-CAM Explainability and Clinician Feedback Integration")

    # -----------------------
    # Upload Image
    # -----------------------
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader('Upload a chest X-ray image',type=["jpg", "jpeg", "png"])

    # Variables to store for feedback later
    session_state = st.session_state
    if "image_id" not in session_state:
        session_state.image_id = None
    if "label" not in session_state:
        session_state.label = None
    if "cue" not in session_state:
        session_state.cue = None

    # -----------------------
    # Predict Section
    # -----------------------
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        if st.button("Predict Pneumonia Probability"):
            files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(f"{API_URL}/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Pneumonia Probability: {result['pneumonia_probability']:.4f}")
            else:
                st.error("Error calling /predict endpoint")


    # -----------------------
    # Explainability Section
    # -----------------------
    st.subheader("Explainability")
    st.markdown("""
    This model uses **Grad-CAM** to highlight the lung regions that most influenced its pneumonia prediction.
    We then map these heatmaps to **ecologically valid cues** — clinical terms that a radiologist or clinician can
    understand, such as *Lobar opacity*, *Pleural effusion*, or *Diffuse lung opacity*.  
    The goal is to bridge the gap between AI output and human decision-making.
    """)

    if uploaded_file and st.button("Explain Prediction"):
        files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{API_URL}/explain", files=files)

        if response.status_code == 200:
            result = response.json()

            # Show probability
            st.info(f"Pneumonia Probability: {result['pneumonia_probability']:.4f}")

            # Show cues and store for feedback
            explanations = result.get("explanations", [])
            if explanations:
                for exp in explanations:
                    st.write(f"**Label:** {exp['label']}")
                    st.write(f"**Probability:** {exp['probability']:.4f}")
                    st.write(f"**Cues:** {exp['label']}")
                    session_state.image_id = result.get("image_id")
                    session_state.label = exp['label']
                    session_state.cue = exp['cues']

            # Show original & heatmap side-by-side
            cols = st.columns(2)
            with cols[0]:
                st.image(uploaded_file, caption="Original X-ray", use_container_width=True)
            with cols[1]:
                heatmap_b64 = result.get("heatmap_base64")
                if heatmap_b64:
                    heatmap_img = Image.open(io.BytesIO(base64.b64decode(heatmap_b64)))
                    st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)

        else:
            st.error("Error calling /explain endpoint")

    else:
        st.info("Upload an image first to enable Explainability.")

    # -----------------------
    # Feedback Section
    # -----------------------
    st.subheader("Clinician Feedback")

    if session_state.image_id and session_state.label and session_state.cue:
        st.write(f"**Image ID:** {session_state.image_id}")
        st.write(f"**Label:** {session_state.label}")
        st.write(f"**Cue:** {session_state.cue}")

        comment = st.text_area("Your comment (why this cue is relevant/irrelevant):")

        if st.button("Submit Feedback"):
            feedback_payload = {
                "image_id": session_state.image_id,
                "label": session_state.label,
                "cue": session_state.cue,
                "comment": comment
            }
            resp = requests.post(f"{API_URL}/feedback", json=feedback_payload)
            if resp.status_code == 200:
                st.success("Feedback recorded successfully.")
            else:
                st.error("Error submitting feedback.")
    else:
        st.info("Run an explanation first to enable feedback form.")
    

elif page == "Clinician Feedback Stats":
    # -----------------------
    # Feedback Statistics
    # -----------------------
    st.subheader("Clinician Feedback Summary")
    st.write(
    "This section provides an overview of all feedback submitted by clinicians, "
    "including totals for correct and incorrect assessments, and a breakdown of "
    "comments linked to specific visual cues identified in the AI’s predictions."
)

    if st.button("Refresh Stats"):
        stats_resp = requests.get(f"{API_URL}/feedback_stats")
        if stats_resp.status_code == 200:
            stats = stats_resp.json()

            cols = st.columns(3)
            with cols[0]:
                st.metric("Total Feedback Entries", stats["total_feedback"])

            with cols[1]:
                st.metric("Total Positive Comments", stats["total_positive_comments"])

            with cols[2]:
                st.metric("Total Negative Comments", stats["total_negative_comments"])
            
            df_summary = pd.DataFrame(stats["summary"])
            st.dataframe(df_summary, use_container_width=True)
            
        else:
            st.error("Error fetching feedback statistics")





