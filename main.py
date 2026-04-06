from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("boxes.pt")
cap = cv2.VideoCapture("test.mp4")
pts_src = np.array([[0, 129], [1275, 303], [1274, 601], [3, 294]], dtype=np.float32)

width = int(np.linalg.norm(pts_src[0] - pts_src[1]))
height = int(np.linalg.norm(pts_src[0] - pts_src[3]))
pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
M_inv = cv2.getPerspectiveTransform(pts_dst, pts_src)

print(f"📍 Fixed tilted ROI defined ({width}x{height})")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    cv2.polylines(frame, [pts_src.astype(int)], True, (255, 0, 0), 2)
    roi_frame = cv2.warpPerspective(frame, M, (width, height))
    results = model(roi_frame, imgsz=640, conf=0.05, verbose=False)
    boxes = results[0].boxes
    class_names = model.names

    for box, cls_id in zip(boxes.xyxy, boxes.cls):
        class_name = class_names[int(cls_id)]
        if class_name == "box":
            x1, y1, x2, y2 = map(int, box[:4])
            pts_box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            pts_box_transformed = cv2.perspectiveTransform(np.array([pts_box]), M_inv)[0].astype(int)
            cv2.polylines(frame, [pts_box_transformed], True, (0, 255, 0), 1)
            label = "BOX"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
            x_text, y_text = pts_box_transformed[0]
            cv2.rectangle(frame, (x_text, y_text - text_h - 8), (x_text + text_w + 5, y_text), (0, 255, 0), -1)
            cv2.putText(frame, label, (x_text + 2, y_text - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 1, cv2.LINE_AA)

    frame_small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Logistics Counter (Fixed Tilted ROI)", frame_small)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
