#!/usr/bin/env python3
import cv2
import rospy
from std_msgs.msg import String
from ultralytics import YOLO

class Sign_Node:
    def __init__(self, model_path, label_map=None, conf_thresh=0.25, cap=None):
        self.model_path = model_path
        self.conf_thresh = conf_thresh 

        self.cap = cap if cap is not None else cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError(" Camera did not open")

        self.model = YOLO(self.model_path)

        self.label_map = label_map if label_map else {
            1: "LEFT",
            0: "RIGHT"
        }

        self.pub = rospy.Publisher("/detected_sign", String, queue_size=10)
        rospy.init_node("sign_detector_node", anonymous=True)
        rospy.loginfo("✅ Sign detector node started, publishing to /detected_sign")

    def detect_signs(self, frame):
        results = self.model(frame, conf=self.conf_thresh)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.label_map.get(cls_id, f"Class{cls_id}")
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })
        return detections

    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
        return frame

    def run(self):
        rospy.loginfo(" Running sign detection... Press 'q' to quit")

        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("Frame not received, skipping...")
                continue

            # frame = cv2.flip(frame, 1)

            detections = self.detect_signs(frame)
            if detections:
                for det in detections:
                    rospy.loginfo(f"Detected: {det['label']} (conf={det['confidence']:.2f})")
                    self.pub.publish(det["label"]) 

            processed_frame = self.draw_detections(frame, detections)
            cv2.imshow("Sign Detector", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        sign_model = "/home/rawan/catkin_ws/src/Robot-Competition/models/SignModel.pt"
        detector = Sign_Node(sign_model)
        detector.run()
    except rospy.ROSInterruptException:
        pass
