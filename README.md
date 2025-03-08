# Eye Detection in Face Images

This project detects and marks eyes in face images using facial landmark detection. It processes all images in an input directory, draws bounding boxes around detected eyes, and saves the results in an output directory.

## Requirements

- Python 3.6+
- OpenCV
- dlib
- numpy

## Installation

1. **Install dependencies**:
   ```bash
   pip install opencv-python dlib numpy
2. Download the facial landmark model:
    Download shape_predictor_68_face_landmarks.dat from dlib.net/files, extract it, and place it in your project directory.

Usage

Run the script with the following arguments:

    --input_dir: Directory containing input images.

    --output_dir: Directory to save processed images.

    --model: Path to the facial landmark model file (default: shape_predictor_68_face_landmarks.dat).

Example:
  ```bash 
  python eye_detection.py -i input_images -o output_images -m shape_predictor_68_face_landmarks.dat 

##How It Works

    Face Detection: Uses dlib's HOG-based face detector.

    Landmark Detection: Identifies 68 facial landmarks using the pre-trained model.

    Eye Localization: Extracts landmarks for the left and right eyes (points 36-47).

    Bounding Boxes: Draws green rectangles around detected eyes and saves the images.

Sample Output

Input Image → Output Image
Notes

    Ensure input images are in common formats (JPG, PNG).

    The script processes all faces detected in an image.

    Adjust the padding value in the code to modify the eye region margin.
