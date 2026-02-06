import torch
from ultralytics import YOLO

def run_yolo_inference():
    # 1. Load the pretrained YOLOv8n model (weights will be downloaded automatically)
    # Using 'yolov8n.pt' which is the 'nano' version optimized for speed
    model = YOLO('yolov8n.pt')

    # 2. Define input source (can be a local image path, URL, or numpy array)
    # For this example, I'll use a placeholder URL from Ultralytics
    source = 'https://ultralytics.com/images/bus.jpg'

    # 3. Perform inference
    # imgsz=640: Resizes the input image to 640x640
    # batch=1: Since we are passing a single source string, batch size is 1 by default
    # device='cpu': You can change this to 'cuda' if you have a GPU
    results = model.predict(
        source=source, 
        imgsz=640, 
        batch=1,
        conf=0.25,      # Confidence threshold
        device='cpu'    # Force CPU for basic learning
    )

    # 4. Process the results
    # Since batch=1, we take the first item in the list
    result = results[0]

    # 5. Show and print info
    print(f"Detected {len(result.boxes)} objects in the image.")
    
    # Save or show the visual output (optional)
    result.show()  # Opens a window with detections
    result.save(filename='result.jpg') # Saves the image to disk

if __name__ == "__main__":
    run_yolo_inference()