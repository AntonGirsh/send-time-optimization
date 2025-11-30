# Send-Time Optimization

Модель для выбора оптимального времени отправки офферов.

## Установка
pip install -r requirements.txt

## Генерация данных
python run.py data --n 5000

## Обучение
python run.py train data/synthetic.parquet --run-id 2025-11-25_v1

## Инференс
python run.py predict --model-run 2025-11-25_v1 data/raw/clients.parquet

## Визуализация
python run.py viz uplift-heatmap reports/last_predictions.parquet
