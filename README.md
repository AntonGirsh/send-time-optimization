# Send-Time Optimization. Модель для выбора оптимального времени отправки офферов
Задача: мы - некий подписочный сервис. После бесплатного месяца пользователи получают предложение продлить подписку на сервис в виде некоего оффера, который должен предварительно пройти банковские фильтры.
Гипотеза: время отправки влияет на вероятность обоих акцептов.
Решение: 
1. обучить бустинг на имеющихся данных
2. для новых кейсов совершать предсказание для каждого временного слота и выбрать то, что имеет максимальный скор (probe)

## Общая логика работы

для начала работы нужно данные о своём датасете отразить в config/base.yaml

Далее просто запускаем нужные куски пайплайна через командную строку

## Установка

pip install -r requirements.txt

## Запуск отдельных шагов пайплайна
шаблон + пример

### Генерация данных

вход: число строк

выход: синтетические данные в .parquet

python run.py data --n <число элементов> --output <адрес сохранения в .parquet>

python run.py data --n 50000 --output data/synthetic.parquet

### Обучение

вход: данные в .parquet

выход: обученная модель

python run.py train <датасет .parquet> --run-id <имя обученной модели>

python run.py train data/synthetic.parquet --run-id 2025-12-01_v1

### Инференс

вход: обученная модель, датасет в .parquet

выход: датафрейм с синтетическими данными

python run.py predict <имя обученной модели> <датасет .parquet> --output <датасет с результатом инференса в .parquet>

python run.py predict 2025-12-01_v1 data/synthetic.parquet --output reports/pred.parquet

### Визуализация

вход: датасет с результатом инференса в .parquet

выход: heatmap с временными слотами

python run.py viz uplift-heatmap reports/pred.parquet

python run.py viz uplift-heatmap <датасет с результатом инференса в .parquet>

### Интерпретация

вход: датасет с результатом инференса в .parquet

выход: перечисление наиболее заметных групп пользователей

python run.py interp profile-top-slots -p <датасет с результатом инференса в .parquet> --min-uplift <пороговый прирост> --min-clients-abs <минимальное число тразакций в группе>

python run.py interp profile-top-slots -p reports/pred.parquet --min-uplift 0.03 --min-clients-abs 100

### feature_importance

вход: обученная модель

выход: файлы с графиками feature_importance

python run.py fi show <имя обученной модели> --top <число признаков для анализа>

python run.py fi show 2025-12-01_v2 --top 15
