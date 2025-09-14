import cv2
import numpy as np
from ultralytics import YOLO

# Paths
model_path = "e:/Collage/MIA/Final Project/alphabet.pt"
output_file = "e:/Collage/MIA/Final Project/final_word.txt"

# Map class IDs → Letters A–Z
classes_map = {i: chr(65+i) for i in range(26)}

# Load YOLO model
model = YOLO(model_path)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("✅ Collecting 5 green letters to form a word...")

detected_letters = []

def is_green_background(frame, bbox, expand=10, green_thresh=0.12):
    """Check if the background near bbox is green (not red)."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1 - expand), max(0, y1 - expand)
    x2, y2 = min(w-1, x2 + expand), min(h-1, y2 + expand)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # green mask
    lower_green = np.array([35, 60, 60])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.count_nonzero(green_mask) / float(roi.shape[0] * roi.shape[1])

    return green_ratio > green_thresh


while len(detected_letters) < 5:
    ret, frame = cap.read()
    if not ret:
        break


    # ✅ Mirror frame (selfie view)
    frame = cv2.flip(frame, 1)


    results = model(frame, conf=0.25)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = classes_map.get(cls, str(cls))

            # check background
            if is_green_background(frame, (x1, y1, x2, y2)):
                if len(detected_letters) < 5:
                    detected_letters.append(label)
                    print(f"✅ Collected: {label} ({len(detected_letters)}/5)")

                color = (0, 255, 0)  # green box
            else:
                color = (0, 0, 255)  # red box

            # draw box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Letter Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Build word
final_word = "".join(detected_letters)
print(f"\n✨ Final Word: {final_word}")

# Save to file
with open(output_file, "w") as f:
    f.write(final_word)

print(f"📝 Saved to {output_file}")
