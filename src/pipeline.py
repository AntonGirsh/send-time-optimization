# src/pipeline.py
import joblib
from pathlib import Path
import shutil
from omegaconf import OmegaConf
import pandas as pd
import numpy as np

from src.features import create_time_features, build_time_grid
from src.models import train_bank_model, train_user_model, calibrate_model
from src.prediction import predict_best_time_for_dataset
from src.metrics import expected_conversion_score

def train_pipeline(data_path: Path, run_id: str):
    cfg = OmegaConf.load("config/base.yaml")

    # 1. Папка для эксперимента
    run_dir = Path("models") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Загрузка и препроцессинг
    df = pd.read_parquet(data_path)
    df = df.rename(columns={
        cfg.rename.send_ts: 'send_ts',
        cfg.rename.accept_bank: 'accept_bank',
        cfg.rename.accept_user: 'accept_user',
        cfg.rename.client_id: 'client_id',
    })
    df = create_time_features(df, 'send_ts')

    # 3. Определяем фичи
    cat_features = [c for c in cfg.features.categorical if c in df.columns]
    all_features = (
        cfg.features.categorical +
        cfg.features.numeric +
        cfg.features.time.all_generated
    )
    feature_cols = [f for f in all_features if f in df.columns]

    # защита от NaN в категориях
    for col in cat_features:
        df[col] = df[col].astype('string').fillna('missing')

    # 4. Разбиение
    df = df.sort_values('send_ts')
    train_df = df.iloc[:int(0.70 * len(df))]
    val_df   = df.iloc[int(0.70 * len(df)):int(0.85 * len(df))]
    test_df  = df.iloc[int(0.85 * len(df)):]

    # 5. Обучение
    bank_model = train_bank_model(
        train_df[feature_cols], train_df['accept_bank'],
        val_df[feature_cols],   val_df['accept_bank'],
        cat_features
    )

    train_user = train_df[train_df['accept_bank'] == 1]
    val_user   = val_df[val_df['accept_bank'] == 1]
    user_model = train_user_model(
        train_user[feature_cols], train_user['accept_user'],
        val_user[feature_cols],   val_user['accept_user'],
        cat_features
    )

    # 6. Калибровка
    bank_calibrator = calibrate_model(bank_model, val_df[feature_cols], val_df['accept_bank'])
    user_calibrator = calibrate_model(user_model, val_user[feature_cols], val_user['accept_user'])

    # 7. Сетка времени
    time_grid = build_time_grid()

    # 8. Валидационная метрика
    val_p_bank_raw = bank_model.predict_proba(val_df[feature_cols])[:, 1]
    val_p_bank = bank_calibrator.predict(val_p_bank_raw)

    # Предсказываем user только для тех, у кого bank == 1
    val_bank_positive_mask = val_df['accept_bank'] == 1
    if val_bank_positive_mask.any():
        val_p_user_raw = user_model.predict_proba(val_df.loc[val_bank_positive_mask, feature_cols])[:, 1]
        val_p_user_calibrated = user_calibrator.predict(val_p_user_raw)
    else:
        val_p_user_calibrated = np.array([])

    # Заполняем нулями там, где bank == 0
    val_p_user = np.zeros(len(val_df))
    val_p_user[val_bank_positive_mask] = val_p_user_calibrated

    val_metric = expected_conversion_score(val_p_bank, val_p_user)
    print(f"Validation Harmonic F1 (expected conversion): {val_metric:.5f}")

    # 9. Сохранение всего
    artifacts = {
        'bank_model': bank_model,
        'user_model': user_model,
        'bank_calibrator': bank_calibrator,
        'user_calibrator': user_calibrator,
        'time_grid': time_grid,
        'feature_cols': feature_cols,
        'cat_features': cat_features,
    }
    joblib.dump(artifacts, run_dir / "artifacts.joblib")

    # копия конфига + метрика
    shutil.copy("config/base.yaml", run_dir / "config_used.yaml")
    (run_dir / "metrics.json").write_text(f'{{"val_harmonic_f1": {val_metric:.5f}}}')

    print(f"Готово. Метрика на валидации: {val_metric:.5f}")


def predict_pipeline(model_run: str, data_path: Path, output_path: Path):
    run_dir = Path("models") / model_run
    if not run_dir.exists():
        raise FileNotFoundError(f"Модель {model_run} не найдена")

    cfg = OmegaConf.load("config/base.yaml")
    artifacts = joblib.load(run_dir / "artifacts.joblib")

    df = pd.read_parquet(data_path)
    result_df = predict_best_time_for_dataset(
        df=df.copy(),
        time_grid=artifacts['time_grid'],
        bank_model=artifacts['bank_model'],
        user_model=artifacts['user_model'],
        bank_calibrator=artifacts['bank_calibrator'],
        user_calibrator=artifacts['user_calibrator'],
        cfg=cfg,
        top_k=3
    )
    result_df.to_parquet(output_path, index=False)