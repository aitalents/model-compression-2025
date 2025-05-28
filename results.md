# Final Evaluation Results:

- MODEL_TEACHER = "google-t5/t5-base"
- MODEL_STUDENT = "google-t5/t5-small"
- DATASET = "wmt16"
- DATASET_CONFIG = "ro-en"
- BATCH_SIZE = 4
- EPOCHS = 3
- LEARNING_RATE = 10e-5

| Model    | Size (params) | BLEU   | ROUGE  | Avg Latency (s) |
|----------|--------------|--------|--------|-----------------|
| Student  | 60M          | 1.24   | 0.0897 | 0.399           |
| Teacher  | 220M         | 0.98   | 0.0793 | 0.688           |
