# Face Detection with Deep Learning

A robust face detection system using OpenCV's deep learning-based face detectors. This project provides tools for detecting faces in both images and video streams with higher accuracy than traditional methods.

![Face Detection Example](https://github.com/user-attachments/assets/d54c7059-b05c-4cba-8342-e7ee0d83df7e)

## Features

- **Deep Learning Face Detection**: Uses pre-trained Caffe models for more accurate face detection compared to traditional Haar cascades
- **Confidence Thresholding**: Adjustable confidence threshold to filter weak detections
- **Support for Images and Video**: Process both still images and live video streams

## Requirements

- Python 3.6+
- OpenCV 4.x
- NumPy
- argparse
- imutils (for video processing)

```bash
pip install opencv-python numpy imutils
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/face-detection.git
   cd face-detection
   ```

2. Download the pre-trained Caffe model files:

   ```bash
   # You can download these from:
   # https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
   # Or use the included model files
   ```

3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv face_detection
   source face_detection/bin/activate  # On Windows: face_detection\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Image Detection

```bash
# Basic usage
python detect_faces.py --image path/to/your/image.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# With custom confidence threshold
python detect_faces.py --image path/to/your/image.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --confidence 0.7
```

### Video Detection

```bash
# Basic usage (uses webcam)
python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# With custom confidence threshold
python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --confidence 0.7
```

## Command Line Arguments

### detect_faces.py

| Argument             | Flag               | Description                                                  |
| -------------------- | ------------------ | ------------------------------------------------------------ |
| Image path           | `-i, --image`      | Path to input image                                          |
| Prototxt path        | `-p, --prototxt`   | Path to Caffe 'deploy' prototxt file                         |
| Model path           | `-m, --model`      | Path to Caffe pre-trained model                              |
| Confidence threshold | `-c, --confidence` | Minimum probability to filter weak detections (default: 0.5) |

### detect_faces_video.py

| Argument             | Flag               | Description                                                  |
| -------------------- | ------------------ | ------------------------------------------------------------ |
| Prototxt path        | `-p, --prototxt`   | Path to Caffe 'deploy' prototxt file                         |
| Model path           | `-m, --model`      | Path to Caffe pre-trained model                              |
| Confidence threshold | `-c, --confidence` | Minimum probability to filter weak detections (default: 0.5) |

## Model Information

The system uses OpenCV's DNN module with a pre-trained Caffe model for face detection:

- **Model Architecture**: Single Shot Multibox Detector (SSD) with ResNet-10 base network
- **Input Size**: 300x300 pixels
- **Pre-processing**: Mean subtraction (104.0, 177.0, 123.0) and scale factor of 1.0
- **Output**: Face detections with bounding boxes and confidence scores

## How It Works

1. The input image is loaded and preprocessed (resized to 300x300 pixels)
2. The image is passed through the deep neural network
3. Detections above the confidence threshold are collected
4. The final detections are drawn on the output image with confidence scores

## Example Outputs

Here are some example results from the face detection system:

### Image Detection Example

### Video Detection Example

## Directory Structure

```README.md
face-detection/
│
├── detect_faces.py         # Script for image-based face detection
├── detect_faces_video.py   # Script for video-based face detection
├── deploy.prototxt.txt     # Caffe model architecture
├── res10_300x300_ssd_iter_140000.caffemodel  # Pre-trained model weights
│
├── images/                 # Test images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── output/                 # Processed output images
│   └── ...
│
├── requirements.txt        # Project dependencies
├── .gitignore              # Git ignore file
└── README.md               # This documentation
```

## Troubleshooting

### Common Issues

1. **Image Not Found Error**:

   - Ensure the image path is correct
   - Check if the file exists and is readable

2. **Model Loading Failure**:

   - Verify the paths to model files
   - Make sure both the .prototxt and .caffemodel files are in the specified locations

3. **Low Detection Rate**:

   - Adjust the confidence threshold with `--confidence`
   - Ensure images have good lighting and resolution

4. **Performance Issues**:
   - Resize input images to smaller dimensions
   - Consider using GPU acceleration if available

### Tips for Better Results

- Optimal lighting conditions improve detection accuracy
- For group photos, ensure faces are reasonably sized in the frame

## Limitations

- May miss faces at extreme angles or with heavy occlusion
- Performance depends on image quality, lighting conditions, and face size
- The default model is designed primarily for frontal faces

## Acknowledgments

- This project uses OpenCV's DNN module and pre-trained models
- The deep learning face detector is based on the Single Shot Multibox Detector (SSD) framework with a ResNet-10 base network

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
