from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

import torch
from transformers import AutoTokenizer

try:
    import optimum.onnxruntime.utils as _ort_utils
    _ort_utils.validate_provider_availability = lambda provider: None
except ImportError:
    pass

from optimum.onnxruntime import ORTModelForSequenceClassification

@dataclass
class TextClassificationModelData:
    model_name: str
    label: str
    score: float

class BaseTextClassificationModel(ABC):
    """
    Abstract base class for text classification models.
    """
    def __init__(self, name: str, model_path: str, tokenizer_name: str):
        self.name = name
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        assert torch.cuda.is_available(), "CUDA is not available, GPU execution required"
        self.device = torch.device("cuda:0")
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def __call__(self, inputs: Dict[str, torch.Tensor]) -> List[TextClassificationModelData]:
        pass

class TransformerTextClassificationModel(BaseTextClassificationModel):

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            self.model_path,
            export=True,
            provider="CUDAExecutionProvider",
            use_io_binding=True
        )

    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        for key, tensor in inputs.items():
            inputs[key] = tensor.to(self.device)
        return inputs

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> List[TextClassificationModelData]:
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)
        id2label = self.model.config.id2label
        results: List[TextClassificationModelData] = []
        for idx, label_id in enumerate(pred_ids):
            lid = label_id.item()
            results.append(
                TextClassificationModelData(
                    model_name=self.name,
                    label=id2label[lid],
                    score=probs[idx, lid].item()
                )
            )
        return results
