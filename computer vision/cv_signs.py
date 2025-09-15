import cv2
from ultralytics import YOLO

class Sign_Node:

    def __init__(self, model_path, label_map=None, conf_thresh=0.25, cap=None):
        self.model_path = model_path
        self.conf_thresh = conf_thresh


        # Use existing camera if passed
        self.cap = cap
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Camera did not open")

        self.model = YOLO(self.model_path)


        self.label_map = label_map if label_map else {1: "LEFT", 0: "RIGHT"}
        self.detections = []



    def detect_signs(self, frame):
        results = self.model(frame, conf=self.conf_thresh)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.label_map.get(cls_id, f"Class{cls_id}")
                detections.append({"label": label, "confidence": conf, "bbox": (x1, y1, x2, y2)})
        return detections



    def draw_detections(self, frame, detection):


        for det in detection:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame
