import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# =====================
# PATHS
# =====================
BASE_DIR = r"D:\Infosys_springboard_project"
DATASET_DIR = os.path.join(BASE_DIR, "PCB_DATASET", "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "pcb_defect_cnn_model.h5")

IMG_SIZE = 128  # 🔥 MUST match training size

class_names = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

# =====================
# LOAD MODEL
# =====================
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# =====================
# LOAD TEST DATA
# =====================
X = []
y_true = []

for label, class_name in enumerate(class_names):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(class_dir):
        continue

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X.append(img)
        y_true.append(label)

X = np.array(X)
y_true = np.array(y_true)

print(f"📊 Test samples loaded: {len(X)}")

# =====================
# PREDICTION
# =====================
y_pred_probs = model.predict(X)
y_pred = np.argmax(y_pred_probs, axis=1)

# =====================
# RESULTS
# =====================
print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
