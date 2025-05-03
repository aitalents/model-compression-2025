from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import onnxruntime as ort
import numpy as np
import json

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

@dataclass
class TextClassificationModelData:
    model_name: str
    label: str
    score: float


class BaseTextClassificationModel(ABC):

    def __init__(self, name: str, model_path: str, tokenizer: str):
        self.name = name
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self._load_model()

    @abstractmethod
    def _load_model(self):
        ...

    @abstractmethod
    def __call__(self, inputs) -> List[TextClassificationModelData]:
        ...


class TransformerTextClassificationModel(BaseTextClassificationModel):

    def _load_model(self):
        conf_path = f'{self.tokenizer}/model_config.json'
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        with open(conf_path, 'r') as f:
            self.id2label = json.load(f)
        self.id2label = {int(k): v for k, v in self.id2label.items()}
        # self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        # self.model = self.model.to(self.device)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = ort.InferenceSession(self.model_path, sess_options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        

    def tokenize_texts(self, texts: List[str]):
        # inputs = self.tokenizer.batch_encode_plus(
        #         texts,
        #         add_special_tokens=True,
        #         padding='longest',
        #         truncation=True,
        #         return_token_type_ids=True,
        #         return_tensors='pt'
        #         )
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to GPU
        inputs = self.tokenizer.batch_encode_plus(
                texts,
                add_special_tokens=True,
                padding='longest',
                truncation=True
                )
        inputs = {k: np.array(v, dtype=np.int64) for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}

        return inputs

    def _results_from_logits(self, logits: torch.Tensor):
        # id2label = self.model.config.id2label

        label_ids = logits.argmax(axis=1)
        scores = softmax(logits) # logits.softmax(dim=-1)
        results = [
                {
                    "label": self.id2label[label_id.item()],
                    "score": score[label_id.item()].item()
                }
                for label_id, score in zip(label_ids, scores)
            ]
        return results

    def __call__(self, inputs) -> List[TextClassificationModelData]:
        # logits = self.model(**inputs).logits
        # predictions = self._results_from_logits(logits)
        # predictions = [TextClassificationModelData(self.name, **prediction) for prediction in predictions]
        io_binding = self.model.io_binding()
        io_binding.bind_cpu_input('input_ids', inputs['input_ids'])
        io_binding.bind_cpu_input('input_mask', inputs['attention_mask'])
        io_binding.bind_output('output')
        self.model.run_with_iobinding(io_binding)
        logits = io_binding.copy_outputs_to_cpu()[0]

        predictions = self._results_from_logits(logits)
        predictions = [TextClassificationModelData(self.name, **prediction) for prediction in predictions]
        return predictions

