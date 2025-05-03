import time
import torch
import requests
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
from optimum.onnxruntime import ORTModelForImageClassification, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

model_id = "google/vit-base-patch16-224-in21k"
url = "http://images.cocodataset.org/val2017/000000039769.jpg"

pytorch_save_dir = f"{model_id.split('/')[-1]}-pytorch"
onnx_save_dir = f"{model_id.split('/')[-1]}-onnx-optimized"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print(f"Using device: {device}")

    image = Image.open(requests.get(url, stream=True).raw)
    print("Sample image loaded.")

    print("\n---Baseline ---")
    pt_model = AutoModelForImageClassification.from_pretrained(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    pt_model.to(device)
    pt_model.eval()

    inputs = processor(images=image, return_tensors="pt").to(device)

    print("Warmup...")
    with torch.no_grad():
        for _ in range(10):
            _ = pt_model(**inputs)

    print("Измеряем время инференса...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            outputs = pt_model(**inputs)
    end_time = time.time()
    pytorch_time = end_time - start_time
    print(f"PyTorch Inference time (100 runs): {pytorch_time:.4f} seconds")

    predicted_class_idx = outputs.logits.argmax(-1).item()
    print("PyTorch Predicted class:", pt_model.config.id2label[predicted_class_idx])

    print("\n--- ONNX Export and Optimization ---")

    print("Exporting PyTorch model to ONNX...")
    ort_model_exporter = ORTModelForImageClassification.from_pretrained(
        model_id,
        export=True,
    )
    temp_onnx_dir = f"{model_id.split('/')[-1]}-onnx-temp"
    ort_model_exporter.save_pretrained(temp_onnx_dir)
    processor.save_pretrained(temp_onnx_dir)
    print(f"ONNX model exported  to {temp_onnx_dir}")

    optimization_config = OptimizationConfig(optimization_level=99)

    optimizer = ORTOptimizer.from_pretrained(temp_onnx_dir)

    print("Applying ONNX Runtime optimizations...")
    optimizer.optimize(
        save_dir=onnx_save_dir,
        optimization_config=optimization_config,
    )
    print(f"Optimized ONNX model saved to {onnx_save_dir}")
    processor.save_pretrained(onnx_save_dir)


    print("\n--- Optimized ONNX Inference ---")

    print("Loading optimized ONNX model for inference...")
    optimized_model = ORTModelForImageClassification.from_pretrained(
        onnx_save_dir,
        provider="CUDAExecutionProvider"
    )

    inputs_onnx = processor(images=image, return_tensors="pt").to(device)

    print("Warmup...")
    for _ in range(10):
        _ = optimized_model(**inputs_onnx)

    print("Timed Inference...")
    start_time = time.time()
    for _ in range(100):
        onnx_outputs = optimized_model(**inputs_onnx)
    end_time = time.time()
    onnx_time = end_time - start_time
    print(f"Optimized ONNX Inference time (100 runs): {onnx_time:.4f} seconds")

    onnx_predicted_class_idx = onnx_outputs.logits.argmax(-1).item()
    print("ONNX Predicted class:", optimized_model.config.id2label[onnx_predicted_class_idx]) # Use loaded config

    print("\n--- Comparison ---")
    print(f"PyTorch baseline time: {pytorch_time:.4f} seconds")
    print(f"Optimized ONNX time:   {onnx_time:.4f} seconds")
    speedup = pytorch_time / onnx_time if onnx_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")
