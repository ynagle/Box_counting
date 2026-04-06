---
title: Warehousing Security Bag Count
emoji: 📦
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 📦 Warehousing Security Bag Count

A sophisticated real-time box/bag counting and detection system for warehouse security and logistics operations, powered by YOLOv8 and Gradio.

## 🌟 Features

- **Real-time Detection**: Detect and count boxes/bags in images and videos
- **ROI Support**: Region of Interest (ROI) detection with perspective transformation for accurate counting in specific zones
- **Customizable Display**: Toggle background and box class detections
- **Video Processing**: Process videos with real-time progress tracking
- **Modern UI**: Beautiful and intuitive Gradio interface

## 🚀 Quick Start

### Online Demo

Try the live demo on Hugging Face Spaces: [Your Space URL]

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/warehousing-security-bag-count.git
cd warehousing-security-bag-count
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The app will launch at `http://localhost:7860`

## 🎯 How to Use

### Image Detection
1. Navigate to the **🖼️ Image Detection** tab
2. Upload an image or use the provided demo image
3. Adjust detection settings:
   - **Show BG Class**: Display background detections
   - **Show Box Class**: Display box/bag detections
4. Click **Detect Boxes** to process
5. View results with box count

### Video Detection
1. Navigate to the **🎥 Video Detection** tab
2. Upload a video or use the provided demo video
3. Configure settings:
   - **Show BG Class**: Toggle background detection display
   - **Use ROI**: Enable region of interest detection with perspective transform
   - **Show Box Class**: Toggle box detection display
4. Click **🎬 Process Video** to start processing
5. View real-time progress and final annotated video

## 🛠️ Technology Stack

- **YOLOv8**: State-of-the-art object detection model (Ultralytics)
- **OpenCV**: Computer vision operations and video processing
- **Gradio**: Interactive web interface
- **NumPy**: Numerical computations
- **Python 3.11+**: Core programming language

## 📁 Project Structure

```
warehousing-security-bag-count/
├── app.py              # Main Gradio application
├── main.py             # Core detection logic (if separate)
├── boxes.pt            # YOLOv8 trained model weights
├── demo.jpg            # Demo image for testing
├── test.mp4            # Demo video for testing
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project configuration
└── README.md           # This file
```

## 🎨 Detection Classes

The model detects two primary classes:
- **box**: Boxes/bags to be counted
- **bg**: Background elements

## ⚙️ Configuration

Key parameters in `app.py`:
- `conf_threshold`: Detection confidence threshold (default: 0.05)
- `model`: YOLO model path (`boxes.pt`)
- `pts_src`: ROI coordinates for perspective transformation
- `server_port`: Gradio server port (default: 7860)

## 🔧 Model Information

The detection model (`boxes.pt`) is a custom-trained YOLOv8 model optimized for:
- Box/bag detection in warehouse environments
- Low confidence threshold for maximum recall
- Real-time inference performance

## 📊 Performance

- **Inference Speed**: Real-time processing on modern hardware
- **Model Size**: Optimized for deployment
- **Accuracy**: Custom-trained for warehouse scenarios

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- [Gradio](https://gradio.app/) for the web interface
- [OpenCV](https://opencv.org/) for computer vision capabilities

## 📞 Contact

For questions, issues, or collaborations, please open an issue on GitHub.

---

**Note**: This model is designed for warehouse security and logistics applications. Ensure proper lighting and camera positioning for optimal detection accuracy.
