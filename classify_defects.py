import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =============================
# PATHS
# =============================
BASE_DIR = r"D:\Infosys_springboard_project"
ROI_DIR = os.path.join(BASE_DIR, "detected_rois")
MODEL_PATH = os.path.join(BASE_DIR, "pcb_defect_cnn_model.h5")

# =============================
# CLASS LABELS (VERY IMPORTANT)
# =============================
CLASS_NAMES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

# =============================
# LOAD MODEL
# =============================
model = load_model(MODEL_PATH)
print("✅ CNN model loaded")

# =============================
# PROCESS ROIs
# =============================
for file in os.listdir(ROI_DIR):
    if not file.endswith(".jpg"):
        continue

    path = os.path.join(ROI_DIR, file)

    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_id = np.argmax(pred)
    defect_name = CLASS_NAMES[class_id]

    print(f"🧠 {file} → {defect_name}")
