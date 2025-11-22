# Send Time Optimization

STO модель для выбора оптимального времени отправки офферов.

## Запуск

Обучение:
python train.py --gen

Инференс:
python predict.py data/raw/new_clients.parquet

## Структура

- src/ — весь код
- config/ — конфиги
- data/ — данные
- models/ — обученные модели