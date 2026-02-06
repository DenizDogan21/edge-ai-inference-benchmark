import time
import psutil
import torch
import pandas as pd
from ultralytics import YOLO

def run_benchmark():
    # 1. Load the model
    model = YOLO('yolov8n.pt')
    
    # Configuration
    image_path = 'https://ultralytics.com/images/bus.jpg'
    num_iterations = 50 # How many times to run the test
    warmup_runs = 5     # Discard first few runs as they are usually slower
    
    latencies = []
    
    print(f"Starting benchmark with {num_iterations} iterations...")

    # 2. Warm-up (Initial runs to prepare the hardware/cache)
    for _ in range(warmup_runs):
        _ = model.predict(source=image_path, imgsz=640, verbose=False)

    # 3. Benchmark Loop
    for i in range(num_iterations):
        start_time = time.time()
        
        # Run inference
        results = model.predict(source=image_path, imgsz=640, verbose=False)
        
        end_time = time.time()
        
        # Calculate latency in milliseconds
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        if i % 10 == 0:
            print(f"Iteration {i}/{num_iterations}...")

    # 4. Measure Resource Usage
    # Getting RAM usage of the current process in MB
    process = psutil.Process()
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)

    # 5. Calculate Final Metrics
    avg_latency = sum(latencies) / len(latencies)
    fps = 1000 / avg_latency
    
    # 6. Create Results Table
    data = {
        "Model": ["YOLOv8n"],
        "Resolution": ["640x640"],
        "Batch Size": [1],
        "Avg Latency (ms)": [round(avg_latency, 2)],
        "FPS": [round(fps, 2)],
        "RAM Usage (MB)": [round(ram_usage_mb, 2)],
        "Device": ["CPU"] # Change to GPU if you use cuda
    }

    df = pd.DataFrame(data)
    
    # 7. Save to Excel and CSV
    df.to_excel("benchmark_results.xlsx", index=False)
    df.to_csv("benchmark_results.csv", index=False)
    
    print("\n--- Benchmark Results ---")
    print(df)
    print("\nResults saved to 'benchmark_results.xlsx'")

if __name__ == "__main__":
    run_benchmark()