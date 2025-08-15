# Clinician-in-the-Loop Explainable AI for Pneumonia Detection from Chest X-Rays

This project implements a complete workflow for detecting pneumonia from chest X-rays and providing **ecologically grounded explanations** that clinicians can query and validate.  
It integrates a backend AI model, a Grad-CAM explainability layer mapped to **clinically relevant cues**, and a **feedback loop** that stores clinician insights in a database for later analysis and refinement.

---

## 1) Problem Statement and How This Approach Addresses It

### The Challenge
While AI models for medical imaging have achieved impressive accuracy, they often act as **black boxes**, producing outputs such as “pneumonia: 0.72 probability” without providing reasoning in terms a clinician can understand.  
This creates three core issues:

1. **Loss of cognitive control** – Clinicians cannot interrogate or challenge the AI’s reasoning, reducing trust and adoption.
2. **Uninterpretable visual saliency** – Raw heatmaps may highlight areas but do not translate into clinical reasoning terms (e.g., lobar opacity, pleural effusion).
3. **Dataset vs. ecological mismatch** – Models are optimized for dataset labels, while clinicians reason in terms of nuanced findings and anatomical distributions.

### The Approach
This system addresses these issues through:

1. **Clinically bounded explanations**
   Explanations are mapped to a predefined set of cues for pneumonia:  
   - Lobar opacity (upper/lower, left/right, bilateral)  
   - Diffuse lung opacity  
   - Pleural effusion  
   This ensures outputs are interpretable in a real-world clinical context.

2. **Grad-CAM with cue mapping**  
   The model’s decision for “Pneumonia” is analyzed using Grad-CAM.  
   The activation map is then converted into one or more **explicit clinical cues** based on activation location and spread.

3. **Clinician-in-the-loop feedback**  
   Clinicians can review the explanation and submit feedback directly in the interface.  
   This feedback is stored in a database and can be analyzed to understand which cues are most often validated or disputed, supporting continuous improvement.

---

## 2) System Implementation Details


### Backend (Flask API)

#### Model
- Pretrained **DenseNet121** model from `torchxrayvision`
- Multi-label chest X-ray classification; only the **Pneumonia** class is used in this implementation.

#### Explainability
- **Grad-CAM** applied to the final convolutional layer.
- Heatmaps normalized and overlaid on the input X-ray.
- Activation patterns analyzed to assign **clinically bounded cues**.

#### API Endpoints
1. **`POST /predict`** – Returns pneumonia probability without explanation.  
2. **`POST /explain`** – Returns probability, mapped clinical cues, and a Grad-CAM overlay image.  
3. **`POST /feedback`** – Stores clinician feedback in the database.  
4. **`GET /feedback_stats`** – Returns aggregated cue/comment counts.  

---

### Database (SQLite)
- Schema:
```sql
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT,
    label TEXT,
    cue TEXT,
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

### Frontend (Streamlit)

#### Page 1: Pneumonia Detection & Explainability
- Upload image  
- **Predict** → Calls `/predict` and shows probability  
- **Explain** → Calls `/explain`, displays Grad-CAM overlay and cues  
- **Feedback Form** → Pre-filled with image ID, label, cue; clinician adds comment and submits to `/feedback`

#### Page 2: Feedback Statistics
- Calls `/feedback_stats`  
- Displays table: cue, comment, count

---

## 3) How to Use This Project

### Install Requirements
```bash
git clone https://github.com/Iam0-0ap/explainable-AI-with-clinician-feedback-radiology.git
cd <repository-folder>
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### Run Backend
```bash
python main.py
```
Backend runs at `http://127.0.0.1:5000`

---

### Run Frontend
```bash
streamlit run streamlit_app.py
```

---

### Example Usage Flow
1. Upload an X-ray image in the **Pneumonia Detection** page.  
2. Click **Predict** to see pneumonia probability.  
3. Click **Explain** to get Grad-CAM overlay + clinical cue mapping.  
4. Review the explanation and enter feedback.  
5. Go to the **Clinician Feedback Stats** page to see aggregated clinician input.

---
