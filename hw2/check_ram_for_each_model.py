import torch
import timm
import psutil
import os
import GPUtil


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


if __name__ == "__main__":
    #print("\n🔍 Initial memory state:")
    #print_memory_usage("Before loading model")

    model = timm.create_model('efficientvit_b3.r256_in1k', pretrained=True)
    #print_memory_usage("Model loaded fp32")

    model = torch.quantization.quantize_dynamic(
        model.to('cpu'), {torch.nn.Linear}, dtype=torch.float16
    )
    print_memory_usage("Model loaded fp16")

    # model = torch.quantization.quantize_dynamic(
    #     model.to('cpu'), {torch.nn.Linear}, dtype=torch.qint8
    # )
    # print_memory_usage("Model loaded int8")

    