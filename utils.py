import torch

def yolo_metrics(metrics):
    speed = metrics.speed
    result_metrics = metrics.results_dict

    res = {
            "inference": speed["inference"],
            "precision": result_metrics["metrics/precision(B)"],
            "recall": result_metrics["metrics/recall(B)"]
        }
    return res

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2
