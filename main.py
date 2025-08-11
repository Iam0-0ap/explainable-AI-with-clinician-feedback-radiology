import os
import base64
import sqlite3
import numpy as np
import torch
import tempfile

from flask import Flask, request, jsonify, send_file
from pneumonia_pipeline import (
    load_and_preprocess,
    load_model,
    predict_pneumonia_prob,
    run_gradcam,
    map_heatmap_to_cues,
    overlay_and_save,
    PROBABILITY_KEEP_THRESHOLD
)


app = Flask(__name__)

## DB setup

DB_PATH = "feedback.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT,
            label TEXT,
            cue TEXT,
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Load model once at startup
model = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files["image"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img_file.save(tmp.name)
        img_tensor, _ = load_and_preprocess(tmp.name, res=224)
        prob = predict_pneumonia_prob(model, img_tensor)

    os.remove(tmp.name)
    return jsonify({"pneumonia_probability": prob})



@app.route("/explain", methods=["POST"])
def explain():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files["image"]

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img_file.save(tmp.name)
        tmp_path = tmp.name

    # Preprocess
    img_tensor, img_numpy = load_and_preprocess(tmp_path, res=224)

    # Prediction
    prob = predict_pneumonia_prob(model, img_tensor)
    if prob < PROBABILITY_KEEP_THRESHOLD:
        os.remove(tmp_path)
        return jsonify({
            "pneumonia_probability": prob,
            "explanations": [
                {
                "label": "Not Pneumonia",
                "probability": prob,
                "cues": "Not Available"
            }],
            "image_id": os.path.basename(img_file.filename),
        })

    heatmap = run_gradcam(model, img_tensor, target_label="Pneumonia")
    cues = map_heatmap_to_cues(heatmap)

    # Save Grad-CAM overlay
    visual_img_rgb = np.stack([img_numpy[0], img_numpy[0], img_numpy[0]], axis=-1)
    visual_img_rgb = (visual_img_rgb - visual_img_rgb.min()) / (visual_img_rgb.max() - visual_img_rgb.min())
    heatmap_path = tmp_path.replace(".jpg", "_Pneumonia_cam.png")
    overlay_and_save(visual_img_rgb, heatmap, heatmap_path)

    # Encode heatmap as base64
    with open(heatmap_path, "rb") as f:
        heatmap_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Cleanup
    os.remove(tmp_path)
    os.remove(heatmap_path)

    return jsonify({
        "pneumonia_probability": prob,
        "explanations": [{
            "label": "Pneumonia",
            "probability": prob,
            "cues": cues
        }],
        "image_id": os.path.basename(img_file.filename),
        "heatmap_base64": heatmap_base64
    })

    # return jsonify({
    #     "pneumonia_probability": prob,
    #     "explanations": [{
    #         "label": "Pneumonia",
    #         "probability": prob,
    #         "cues": cues
    #     }],
    #     "heatmap_image_url": f"/heatmap/{os.path.basename(heatmap_path)}"
    # })


@app.route("/feedback", methods=["POST"])
def feedback():
    """Stores clinician feedback on cues (only comment is typed by clinician)."""
    data = request.get_json()

    # Required from frontend (auto-filled)
    image_id = data.get("image_id")
    label = data.get("label")
    cue = data.get("cue")

    # Only field the clinician types
    comment = data.get("comment", "")

    if not image_id or not label or not cue:
        return jsonify({"error": "Missing image_id, label, or cue from request"}), 400

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO feedback (image_id, label, cue, comment) VALUES (?, ?, ?, ?)",
        (image_id, label, cue, comment)
    )
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "message": "Feedback recorded"})


@app.route("/feedback_stats", methods=["GET"])
def feedback_stats():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Total feedback entries
    cur.execute("SELECT COUNT(*) FROM feedback")
    total_count = cur.fetchone()[0]

    # Total entries with positive comment
    cur.execute("""SELECT COUNT(*) FROM feedback WHERE comment="correct" """)
    positive_count = cur.fetchone()[0]

    # Total entries with negative comment
    cur.execute("""SELECT COUNT(*) FROM feedback WHERE comment="incorrect" """)
    negative_count = cur.fetchone()[0]

    cur.execute("""
        SELECT cue,
               comment,
               COUNT(*) AS comment_count, 
               label
        FROM feedback 
        GROUP BY cue, comment
        ORDER BY cue, comment
    """)
    rows = cur.fetchall()
    conn.close()

    summary = [
        {"cue": row[0], "comment": row[1], "comment_count": row[2], "label": row[3]}
        for row in rows
    ]

    return jsonify({
        "total_feedback": total_count,
        "total_positive_comments": positive_count, 
        "total_negative_comments": negative_count,
        "summary": summary
    })



if __name__ == "__main__":
    app.run(debug=False)
