import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Initialize model
model = YOLO("boxes.pt")
conf_threshold = 0.05

# ROI setup
pts_src = np.array([[0, 129], [1275, 303], [1274, 601], [3, 294]], dtype=np.float32)
width = int(np.linalg.norm(pts_src[0] - pts_src[1]))
height = int(np.linalg.norm(pts_src[0] - pts_src[3]))

M = cv2.getPerspectiveTransform(
    pts_src,
    np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
)

M_inv = cv2.getPerspectiveTransform(
    np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32),
    pts_src
)


def draw_detection(frame, box, cls_id, conf, class_name, use_roi=False):
    x1, y1, x2, y2 = map(int, box[:4])
    color = (0, 255, 0) if class_name == "box" else (255, 255, 0)
    label = f"{class_name.upper()}: {conf:.2f}"

    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    if use_roi:
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        pts_transformed = cv2.perspectiveTransform(np.array([pts]), M_inv)[0].astype(int)
        cv2.polylines(frame, [pts_transformed], True, color, 2)
        x_text, y_text = pts_transformed[0]
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        x_text, y_text = x1, y1

    cv2.rectangle(frame, (x_text, y_text - text_h - 10), (x_text + text_w + 5, y_text), color, -1)
    cv2.putText(frame, label, (x_text + 2, y_text - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def process_detections(frame, boxes, class_names, show_bg, show_box, use_roi=False):
    box_count = 0
    for idx, (box, cls_id) in enumerate(zip(boxes.xyxy, boxes.cls)):
        class_name = class_names[int(cls_id)]

        if class_name == "box":
            box_count += 1

        if (class_name == "bg" and not show_bg) or (class_name == "box" and not show_box):
            continue

        draw_detection(frame, box, cls_id, float(boxes.conf[idx]), class_name, use_roi)

    return box_count


def detect_image(image, show_bg, show_box):
    if image is None:
        return None, 0

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(frame, imgsz=640, conf=conf_threshold, verbose=False)

    box_count = process_detections(
        frame, results[0].boxes, model.names, show_bg, show_box, False
    )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), box_count


def detect_video(video_path, show_bg, use_roi, show_box, progress=gr.Progress()):
    if not video_path or not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if use_roi:
            cv2.polylines(frame, [pts_src.astype(int)], True, (255, 0, 0), 2)
            detection_frame = cv2.warpPerspective(frame, M, (width, height))
        else:
            detection_frame = frame

        results = model(detection_frame, imgsz=640, conf=conf_threshold, verbose=False)

        frame_box_count = process_detections(
            frame, results[0].boxes, model.names, show_bg, show_box, use_roi
        )

        out.write(frame)
        frame_count += 1

        progress(
            (frame_count / total_frames) if total_frames > 0 else 0,
            desc=f"Frame {frame_count}/{total_frames} | Boxes: {frame_box_count}"
        )

    cap.release()
    out.release()

    return output_path


# UI
default_video_path = "test.mp4" if os.path.exists("test.mp4") else None
default_image_path = "demo.jpg" if os.path.exists("demo.jpg") else None

with gr.Blocks() as app:
    gr.Markdown("# 📦 Logistics Box Detection System")

    with gr.Row():
        show_bg = gr.Checkbox(label="Show BG", value=True)
        use_roi = gr.Checkbox(label="Use ROI", value=True)
        show_box = gr.Checkbox(label="Show Box", value=True)

    with gr.Tabs():

        with gr.Tab("Image"):
            image_input = gr.Image(value=default_image_path)
            image_output = gr.Image()
            count_output = gr.Textbox(label="Box Count")

            def img_wrapper(img, bg, box):
                res, count = detect_image(img, bg, box)
                return res, str(count)

            gr.Button("Detect").click(
                img_wrapper,
                [image_input, show_bg, show_box],
                [image_output, count_output]
            )

        with gr.Tab("Video"):
            video_input = gr.Video(value=default_video_path)
            video_output = gr.Video()

            gr.Button("Process").click(
                detect_video,
                [video_input, show_bg, use_roi, show_box],
                video_output
            )


# ✅ CRITICAL FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.launch(server_name="0.0.0.0", server_port=port, share=False)
# import gradio as gr
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import tempfile
# import os

# # Initialize
# model = YOLO("boxes.pt")
# conf_threshold = 0.05

# # ROI setup
# pts_src = np.array([[0, 129], [1275, 303], [1274, 601], [3, 294]], dtype=np.float32)
# width = int(np.linalg.norm(pts_src[0] - pts_src[1]))
# height = int(np.linalg.norm(pts_src[0] - pts_src[3]))
# M = cv2.getPerspectiveTransform(pts_src, np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32))
# M_inv = cv2.getPerspectiveTransform(np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32), pts_src)

# def draw_detection(frame, box, cls_id, conf, class_name, use_roi=False):
#     """Draw single detection box and label"""
#     x1, y1, x2, y2 = map(int, box[:4])
#     color = (0, 255, 0) if class_name == "box" else (255, 255, 0)
#     label = f"{class_name.upper()}: {conf:.2f}"
#     (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
#     if use_roi:
#         pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
#         pts_transformed = cv2.perspectiveTransform(np.array([pts]), M_inv)[0].astype(int)
#         cv2.polylines(frame, [pts_transformed], True, color, 2)
#         x_text, y_text = pts_transformed[0]
#     else:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         x_text, y_text = x1, y1
    
#     cv2.rectangle(frame, (x_text, y_text - text_h - 10), (x_text + text_w + 5, y_text), color, -1)
#     cv2.putText(frame, label, (x_text + 2, y_text - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

# def process_detections(frame, boxes, class_names, show_bg, show_box, use_roi=False):
#     """Process all detections and draw them"""
#     box_count = 0
#     for idx, (box, cls_id) in enumerate(zip(boxes.xyxy, boxes.cls)):
#         class_name = class_names[int(cls_id)]
#         if class_name == "box" and use_roi:
#             box_count += 1
#         elif class_name == "box" and not use_roi:
#             box_count += 1
        
#         if (class_name == "bg" and not show_bg) or (class_name == "box" and not show_box):
#             continue
        
#         draw_detection(frame, box, cls_id, float(boxes.conf[idx]), class_name, use_roi)
#     return box_count

# def detect_image(image, show_bg, show_box):
#     """Detect boxes in image"""
#     if image is None:
#         return None, 0
    
#     frame = cv2.cvtColor(np.array(image) if not hasattr(image, 'shape') else image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image.copy()
#     results = model(frame, imgsz=640, conf=conf_threshold, verbose=False)
#     box_count = process_detections(frame, results[0].boxes, model.names, show_bg, show_box, False)
#     return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), box_count

# def detect_video(video_path, show_bg, use_roi, show_box, progress=gr.Progress()):
#     """Detect boxes in video"""
#     default_vid = "test.mp4" if os.path.exists("test.mp4") else None
#     video_path = video_path or default_vid
#     if not video_path or not os.path.exists(video_path):
#         return None
    
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return None
    
#     fps, w, h = int(cap.get(cv2.CAP_PROP_FPS)) or 30, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
#     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         if use_roi:
#             cv2.polylines(frame, [pts_src.astype(int)], True, (255, 0, 0), 2)
#             detection_frame = cv2.warpPerspective(frame, M, (width, height))
#         else:
#             detection_frame = frame
        
#         results = model(detection_frame, imgsz=640, conf=conf_threshold, verbose=False)
#         frame_box_count = process_detections(frame, results[0].boxes, model.names, show_bg, show_box, use_roi)
        
#         out.write(frame)
#         frame_count += 1
#         progress((frame_count / total_frames) if total_frames > 0 else 0, 
#                 desc=f"Processing frame {frame_count}/{total_frames} - Boxes: {frame_box_count}")
    
#     cap.release()
#     out.release()
#     return output_path

# # UI Setup
# default_video_path = "test.mp4" if os.path.exists("test.mp4") else None
# default_image_path = "demo.jpg" if os.path.exists("demo.jpg") else None

# css = """
# .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
# .header { text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
#           color: white; border-radius: 10px; margin-bottom: 20px; }
# .checkbox-group { background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; }
# .checkbox-group h3 { color: #000000 !important; }
# .checkbox-group label, .checkbox-group .block, .checkbox-group .wrap, .checkbox-group .info { color: #000000 !important; }
# .gr-checkbox label, .gr-checkbox .wrap, .gr-checkbox > label, .gr-checkbox label span,
# [data-testid*="checkbox"] label, .wrap label { color: #ffffff !important; }
# .gradio-app div:has(input[type="checkbox"]) label, div:has(input[type="checkbox"]) > label { color: #ffffff !important; }
# """

# with gr.Blocks(css=css, theme=gr.themes.Soft()) as app:
#     gr.Markdown("# 📦 Logistics Box Detection System\n### Warehouse YOLO-based Detection with Customizable Options", elem_classes=["header"])
    
#     with gr.Row():
#         with gr.Column(scale=1):
#             gr.Markdown("### ⚙️ Detection Settings", elem_classes=["checkbox-group"])
#             show_bg = gr.Checkbox(label="Show BG Class (Background)", value=True, info="Display background class detections")
#             use_roi = gr.Checkbox(label="Use ROI (Region of Interest)", value=True, info="Apply perspective transform for ROI-based detection (Video only)")
#             show_box = gr.Checkbox(label="Show Box Class", value=True, info="Display box class detections")
    
#     with gr.Tabs():
#         with gr.Tab("🖼️ Image Detection"):
#             with gr.Row():
#                 with gr.Column():
#                     image_input = gr.Image(label="Upload Image", height=400, value=default_image_path)
#                     image_button = gr.Button("Detect Boxes", variant="primary", size="lg")
#                 with gr.Column():
#                     image_output = gr.Image(label="Detection Result", type="numpy", height=400)
#                     image_box_count = gr.HTML(value="<div style='text-align: center; padding: 15px; background: #f0f0f0; border-radius: 8px; margin-top: 10px;'><h3 style='margin: 0; color: #333;'>📦 Total Boxes Detected: <span style='color: #28a745; font-size: 24px; font-weight: bold;'>0</span></h3></div>", label="")
            
#             def process_image_wrapper(img, bg, box):
#                 result, count = detect_image(img, bg, box)
#                 html = f"<div style='text-align: center; padding: 15px; background: #f0f0f0; border-radius: 8px; margin-top: 10px;'><h3 style='margin: 0; color: #333;'>📦 Total Boxes Detected: <span style='color: #28a745; font-size: 24px; font-weight: bold;'>{count}</span></h3></div>"
#                 return result, html
            
#             image_button.click(process_image_wrapper, [image_input, show_bg, show_box], [image_output, image_box_count])
        
#         with gr.Tab("🎥 Video Detection"):
#             with gr.Row():
#                 with gr.Column():
#                     video_input = gr.Video(label="Input Video", height=300, value=default_video_path)
#                     video_button = gr.Button("🎬 Process Video (Real-time)", variant="primary", size="lg")
#                 with gr.Column():
#                     video_output = gr.Video(label="Detection Result (Real-time)", height=300)
            
#             video_button.click(detect_video, [video_input, show_bg, use_roi, show_box], [video_output])

# if __name__ == "__main__":
#     app.launch(share=True, server_name="0.0.0.0", server_port=7860)

