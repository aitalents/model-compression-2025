""" 
This is a module for benchmarking torch.quantization.quantize_dynamic. This methods 
"""


import torch
import timm
import time
import psutil
import os
import GPUtil
import numpy as np
from torch.amp import autocast
from torchvision import datasets, transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.metrics import precision_score, recall_score, f1_score


def print_memory_usage(label=""):
    """Выводит использование памяти в мегабайтах"""
    if label:
        print(f"\n--- Memory Usage ({label}) ---")
    else:
        print("\n--- Memory Usage ---")
        
    # CPU RAM в MB
    process = psutil.Process(os.getpid())
    ram_used = process.memory_info().rss / (1024 ** 2)
    print(f"CPU RAM used: {ram_used:.2f} MB")
    
    # GPU VRAM в MB
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        vram_used = gpu.memoryUsed
        vram_total = gpu.memoryTotal
        print(f"GPU {gpu.id} VRAM: {vram_used:.2f} MB / {vram_total:.2f} MB")

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2

def calculate_metrics(model, device, data_loader):
    model.eval()
    all_preds = []
    all_targets = []

    model.to(device)
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
    
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return precision, recall, f1

def load_imagenet_mini(dataset_path, model):
    # Создаем трансформы на основе модели
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    # Загружаем датасет
    dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, 'val'),
        transform=transform
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return data_loader

def benchmark_model(model, device, input_tensor, num_runs=10, warmup=3, use_amp=False):
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup
    print(f"\n🔥 Warming up ({warmup} runs) on {device}...")
    for _ in range(warmup):
        with torch.no_grad():
            if use_amp and device.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.float16):
                    _ = model(input_tensor)
            else:
                _ = model(input_tensor)
    
    # Benchmark
    print(f"🚀 Benchmarking ({num_runs} runs) on {device}...")
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            if use_amp and device.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.float16):
                    _ = model(input_tensor)
            else:
                _ = model(input_tensor)
    
    total_time = (time.time() - start_time) * 1000
    avg_time = total_time / num_runs
    print(f"✅ Average inference: {avg_time:.2f} ms")
    print(f"📊 Total time: {total_time:.2f} ms | FPS: {1000/(avg_time + 1e-9):.1f}")
    
    return avg_time

def main(model, device, dataset_path=None):
    device_cpu = torch.device('cpu')
    device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Бенчмарки
    input_tensor = torch.randn(1, 3, 224, 224)
    
    print("\n🧪 Benchmarking on CPU:")
    cpu_time = benchmark_model(model, device_cpu, input_tensor)
    print_memory_usage("After CPU benchmark")
    
    print(f"⏱️ CPU inference time: {cpu_time:.2f} ms")

    # Загрузка и расчет метрик качества
    if dataset_path:
        print("\n📊 Loading ImageNetMini dataset...")
        data_loader = load_imagenet_mini(dataset_path, model)
        
        print("\n🧮 Calculating metrics on CPU:")
        precision_cpu, recall_cpu, f1_cpu = calculate_metrics(model, device_cpu, data_loader)

        print("\n🎯 Quality Metrics Summary:")
        print("| Device | Precision | Recall  | F1-Score |")
        print("|--------|-----------|---------|----------|")
        print(f"| CPU    | {precision_cpu:.4f}  | {recall_cpu:.4f} | {f1_cpu:.4f}  |")


if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available else "cpu" 
    device = "cpu"

    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")
    path += "/imagenet-mini"

    print("\n🔍 Initial memory state:")
    print_memory_usage("Before loading model")

    model_fp32 = timm.create_model('efficientvit_b3.r256_in1k', pretrained=True)
    print(f"📏 Model model_fp32 size: {get_model_size(model_fp32):.2f} MB")

    print_memory_usage("Model loaded fp32")

    # model must be set to eval mode for static quantization logic to work
    model_fp32.eval()

    model_float16 = torch.quantization.quantize_dynamic(
        model_fp32.to('cpu'), {torch.nn.Linear}, dtype=torch.float16
    )
    torch.save(model_float16.state_dict(), "./model_float16.pth")
    print(f"📏 Model model_float16 size: {get_model_size(model_float16):.2f} MB")

    print_memory_usage("Model loaded fp16")

    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32.to('cpu'), {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(model_int8.state_dict(), "./model_int8.pth")
    print(f"📏 Model model_int8 size: {get_model_size(model_int8):.2f} MB")

    print_memory_usage("Model loaded int8")

    print('Benchmarking FLOAT16')
    main(model_float16, device='cpu', dataset_path=path)

    print('Benchmarking INT8')
    main(model_int8, device='cpu', dataset_path=path)


    