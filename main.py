import cv2
import dlib
import os
import argparse
import sys # Import the sys module

def detect_eyes(input_dir, output_dir, model_path):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)

                # Define eye regions (left and right)
                for i, (start, end) in enumerate([(36, 42), (42, 48)]):
                    eye_points = landmarks.parts()[start:end]
                    x_coords = [p.x for p in eye_points]
                    y_coords = [p.y for p in eye_points]

                    # Expand eye region by 5 pixels
                    padding = 5
                    x_min = max(0, min(x_coords) - padding)
                    x_max = min(img.shape[1], max(x_coords) + padding)
                    y_min = max(0, min(y_coords) - padding)
                    y_max = min(img.shape[0], max(y_coords) + padding)

                    # Draw rectangle around the eye
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Save the processed image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)
            print(f"Processed {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect eyes in face images.')
    parser.add_argument('-i', '--input_dir', required=True, help='Input directory containing face images')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory to save processed images')
    parser.add_argument('-m', '--model', default='shape_predictor_68_face_landmarks.dat', help='Path to facial landmark model file')
    
    # Simulate command-line arguments within Jupyter Notebook
    # Replace 'path/to/your/input/dir', 'path/to/your/output/dir' and 'path/to/your/model' with the actual paths
    sys.argv = ['detect_eyes.py',  # Replace with your script name
                '-i', 'path/to/your/input/dir', 
                '-o', 'path/to/your/output/dir',
                '-m', 'path/to/your/model']  # If you are using a custom model
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
    else:
        detect_eyes(args.input_dir, args.output_dir, args.model)
