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

def export_to_torchscript(model, processor, device, output_path="hw5/model.pt"):
    """Export model to TorchScript format"""
    print("Exporting model to TorchScript...")
    model.eval()
    
    dummy_image = torch.ones((1, 3, 224, 224), dtype=torch.float32).to(device)
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            return self.model(x).logits
    
    wrapped_model = ModelWrapper(model)
    
    traced_model = torch.jit.trace(wrapped_model, dummy_image).to(device)
    
    optimized_model = torch.jit.optimize_for_inference(traced_model)
    
    optimized_model.save(output_path)
    
    print(f"Model exported to {output_path} (optimized for inference)")
    return optimized_model

def measure_performance(model, inputs, labels, device, export_type="original"):
    """Measure model performance"""
    print(f"Measuring performance for {export_type} model on {device}...")
    
    model = model.to(device)
    model.eval()
    
    input_tensor = inputs["pixel_values"].to(device)
    
    ram_usage = get_memory_usage()
    
    if device == "cuda":
        vram_usage = get_gpu_memory_usage()
    else:
        vram_usage = 0
    
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            outputs = model(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / 100
    
    if hasattr(outputs, "logits"):
        predictions = outputs.logits.argmax(-1).cpu().numpy()
    else:
        if isinstance(outputs, torch.Tensor):
            predictions = outputs.argmax(-1).cpu().numpy()
        else:
            try:
                predictions = outputs[0].argmax(-1).cpu().numpy()
            except:
                predictions = outputs.argmax(-1).cpu().numpy()
    
    accuracy = (predictions == labels).mean() * 100
    
    if export_type == "TorchScript":
        model_size_mb = os.path.getsize("hw5/model.pt") / (1024 * 1024)
    else:
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("No GPU available! This script requires GPU.")
        return
    
    os.makedirs("hw5", exist_ok=True)
    
    model, inputs, labels, processor = load_model_and_dataset(device)
    
    original_perf = measure_performance(model, inputs, labels, device, "Оригинал")
    
    torchscript_model = export_to_torchscript(model, processor)
    
    torchscript_perf = measure_performance(torchscript_model, inputs, labels, device, "TorchScript")
    
    results = pd.DataFrame([original_perf, torchscript_perf])
    print("\nРезультаты сравнения:")
    print(results.to_markdown(index=False))
    

if __name__ == "__main__":
    main() 