import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image

# ===============================
# PATHS
# ===============================
BASE_DIR = r"D:\Infosys_springboard_project"
MODEL_PATH = os.path.join(BASE_DIR, "pcb_defect_cnn_model.h5")
OUTPUT_DIR = os.path.join(BASE_DIR, "streamlit_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH)

# IMPORTANT: Must match training classes EXACTLY
class_names = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="PCB Defect Detection", layout="wide")
st.title("🔍 PCB Defect Detection & Classification System")

st.markdown("""
This system detects **PCB defects** using:
- Image differencing
- ROI extraction
- CNN-based defect classification
""")

# ===============================
# FILE UPLOAD
# ===============================
template_file = st.file_uploader("📌 Upload TEMPLATE PCB Image", type=["jpg", "png"])
test_file = st.file_uploader("🧪 Upload TEST PCB Image", type=["jpg", "png"])

if template_file and test_file:
    template_img = Image.open(template_file).convert("RGB")
    test_img = Image.open(test_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(template_img, caption="Template PCB", use_column_width=True)
    with col2:
        st.image(test_img, caption="Test PCB", use_column_width=True)

    if st.button("🚀 Detect Defects"):
        # Convert PIL to OpenCV
        template = cv2.cvtColor(np.array(template_img), cv2.COLOR_RGB2BGR)
        test = cv2.cvtColor(np.array(test_img), cv2.COLOR_RGB2BGR)

        # ===============================
        # IMAGE PROCESSING
        # ===============================
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

        template_gray = cv2.resize(
            template_gray, (test_gray.shape[1], test_gray.shape[0])
        )

        diff = cv2.absdiff(template_gray, test_gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        results = []
        defect_count = 0

        # ===============================
        # DEFECT CLASSIFICATION
        # ===============================
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = test[y:y+h, x:x+w]

                # 🔴 CRITICAL FIX: 224x224 (MobileNetV2)
                roi_resized = cv2.resize(roi, (224, 224))
                roi_norm = roi_resized / 255.0
                roi_input = np.expand_dims(roi_norm, axis=0)

                pred = model.predict(roi_input, verbose=0)
                class_id = np.argmax(pred)
                defect_name = class_names[class_id]

                # Draw results
                cv2.rectangle(test, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(
                    test,
                    defect_name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                results.append({
                    "Defect_ID": defect_count,
                    "Defect_Type": defect_name
                })

                defect_count += 1

        # ===============================
        # SAVE OUTPUTS
        # ===============================
        output_image_path = os.path.join(OUTPUT_DIR, "annotated_output.jpg")
        cv2.imwrite(output_image_path, test)

        df = pd.DataFrame(results)
        csv_path = os.path.join(OUTPUT_DIR, "defect_results.csv")
        df.to_csv(csv_path, index=False)

        # ===============================
        # DISPLAY RESULTS
        # ===============================
        st.success(f"✅ Defects detected: {defect_count}")

        st.image(
            cv2.cvtColor(test, cv2.COLOR_BGR2RGB),
            caption="Detected & Classified Defects",
            use_column_width=True
        )

        st.download_button(
            "⬇️ Download Annotated Image",
            data=open(output_image_path, "rb"),
            file_name="pcb_defects.jpg"
        )

        st.download_button(
            "⬇️ Download CSV Report",
            data=df.to_csv(index=False),
            file_name="defect_report.csv"
        )
