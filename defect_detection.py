import os
import cv2
import numpy as np

def load_image_safe(path):
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

BASE_DIR = r"D:\Infosys_springboard_project"
TEMPLATE_PATH = os.path.join(BASE_DIR, "template", "template_fixed.jpg")
TEST_PATH = os.path.join(BASE_DIR, "test", "test_fixed.jpg")
OUTPUT_DIR = os.path.join(BASE_DIR, "detected_rois")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📂 Base directory :", BASE_DIR)
print("🖼️ Template path :", TEMPLATE_PATH)
print("🧪 Test path     :", TEST_PATH)

template = load_image_safe(TEMPLATE_PATH)
test = load_image_safe(TEST_PATH)

if template is None:
    raise FileNotFoundError(f"❌ Cannot load template image: {TEMPLATE_PATH}")

if test is None:
    raise FileNotFoundError(f"❌ Cannot load test image: {TEST_PATH}")

print("✅ Images loaded successfully")

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

count = 0
for cnt in contours:
    if cv2.contourArea(cnt) > 300:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = test[y:y+h, x:x+w]
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"defect_{count}.jpg"),
            roi
        )
        cv2.rectangle(test, (x, y), (x+w, y+h), (0, 0, 255), 2)
        count += 1

print(f"✅ Defects detected: {count}")
print(f"📁 Saved in: {OUTPUT_DIR}")

cv2.imshow("Detected Defects", test)
cv2.waitKey(0)
cv2.destroyAllWindows()
