# model-compression-2025

# Вывод

Model = openai/whisper-base
inference device = CPU

| optimization_level     | Average Inference Time (s) | WER (%) | RAM Usage (MB) | Model Size (MB) |
|------------------------|----------------------------|---------|----------------|-----------------|
| optimization_level=0   | 7.15                       | 19.23%  | 10482.39 MB    | 666.22 MB       |
| optimization_level=1   | 6.69                       | 19.23%  |  9564.21 MB    | 666.12 MB       |
| optimization_level=2   | 6.25                       | 19.23%  |  9311.27 MB    | 666.12 MB       |
| optimization_level=99  | 5.95                       | 19.23%  |  7598.21 MB    | 666.12 MB       |



Репозиторий курса по сжатию и ускорению моделей машинного обучения.
ИТМО 2025, направление магистратуры Искуственный интеллект.

https://docs.google.com/presentation/d/1SuKmMIwmccECF7T_TFMFR5BCiy4OzdpBr1uKcO6eGIY/edit#slide=id.g249d310d701_0_69

### Расписание

📅 Среда – лекция
📅 Пятница – практика

*План курса*

    Пара 6 - Автоматическая компрессия моделей, с применением Optimum, AWQ, AutoGPTQ:
        Что это за фреймворк?
                С какими видами моделей рабоатет?
        Практика - Рассмотрим примеры и способы компрессии моделей с применением фреймворка Optimum от Huggingface
        ДЗ: Применить к своим моделям и замерить производительность

### Презентации
[Занятие 6 Обзор Фреймворков](https://docs.google.com/presentation/d/1SuKmMIwmccECF7T_TFMFR5BCiy4OzdpBr1uKcO6eGIY/edit?usp=sharing)
