import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor


def get_memory_usage():
    """Get RAM usage in MB"""
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
    return ram_usage

def get_gpu_memory_usage():
    """Get VRAM usage in MB"""
    torch.cuda.synchronize()
    vram_usage = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    return vram_usage

def load_model_and_dataset(device):
    """Load ViT model and beans dataset"""
    print("Loading model and dataset...")
    model_name = "google/vit-base-patch16-224-in21k"
    model = ViTForImageClassification.from_pretrained(model_name).to(device)
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    # Load beans dataset
    dataset = load_dataset("beans")#, download_mode="force_redownload", verification_mode='no_checks')
    
    # Process dataset
    def transform(examples):
        return processor(examples["image"], return_tensors="pt").to(device)
    
    processed_dataset = dataset["test"].select(range(32))
    processed_inputs = transform(processed_dataset)
    
    # Get labels
    labels = processed_dataset["labels"]
    
    return model, processed_inputs, labels, processor

def measure_performance(model, inputs, labels, device, export_type="original"):
    """Measure model performance"""
    print(f"Measuring performance for {export_type} model on {device}...")
    
    # Move model and inputs to device
    model = model.to(device)
    model.eval()
    
    input_tensor = inputs["pixel_values"].to(device)
    
    # Measure RAM usage
    ram_usage = get_memory_usage()
    
    # Measure VRAM usage if using GPU
    if device == "cuda":
        vram_usage = get_gpu_memory_usage()
    else:
        vram_usage = 0
    
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            outputs = model(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / 100  # Average time in ms
    
    # Calculate accuracy
    predictions = outputs.logits.argmax(-1).cpu().numpy()
    accuracy = (predictions == labels).mean() * 100
    
    # Get model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    return {
        "Модель": "google/vit-base-patch16-224-in21k",
        "Метод": export_type,
        "Размер весов": f"{model_size_mb:.2f}Mb",
        "Время инференса (GPU, ms)": f"{avg_time_ms:.2f}ms",
        "Использование RAM (MB)": f"{ram_usage:.2f}Mb",
        "Использование VRAM (MB)": f"{vram_usage:.2f}Mb",
        "Качество (Accuracy)": f"{accuracy:.2f}%",
    }

def export_to_onnx(model, processor, output_path="hw5/model.onnx"):
    """Export model to ONNX format"""
    print("Exporting model to ONNX...")
    model.eval()
    
    # Create dummy input that matches the processor's expected input format
    dummy_image = torch.ones((1, 3, 224, 224), dtype=torch.float32).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_image,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    
    print(f"Model exported to {output_path}")
    return output_path

def load_onnx_model(onnx_path):
    """Load ONNX model using ONNX Runtime"""
    import onnxruntime as ort
    
    ort_session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
    
    # Create a wrapper class to match the PyTorch model interface
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, session):
            super().__init__()
            self.session = session
        
        def forward(self, x):
            # Run ONNX model
            outputs = self.session.run(
                None, 
                {"input": x.cpu().numpy()}
            )
            
            # Convert back to PyTorch and wrap in a structure matching the original model
            class OutputWrapper:
                pass
            
            result = OutputWrapper()
            result.logits = torch.tensor(outputs[0]).to(x.device)
            
            return result
    
    return ONNXWrapper(ort_session)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("No GPU available! This script requires GPU.")
        return
    
    os.makedirs("hw5", exist_ok=True)
    

    model, inputs, labels, processor = load_model_and_dataset(device)

    original_perf = measure_performance(model, inputs, labels, device, "Оригинал")
    
    onnx_path = export_to_onnx(model, processor, device=device)
    
    onnx_model = load_onnx_model(onnx_path)
    
    onnx_perf = measure_performance(onnx_model, inputs, labels, device, "ONNX")
    
    results = pd.DataFrame([original_perf, onnx_perf])
    print("\nРезультаты сравнения:")
    print(results.to_markdown(index=False))

if __name__ == "__main__":
    main() 