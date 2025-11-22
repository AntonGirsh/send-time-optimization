from omegaconf import OmegaConf
from src.data import load_data, generate_synthetic_data
from src.features import create_time_features, build_time_grid
from src.models import train_bank_model, train_user_model, calibrate_model
from src.prediction import predict_best_time_for_dataset
import joblib

def full_pipeline(df_path: Optional[str] = None, use_generated: bool = False, output_dir: str = "models"):
    cfg = OmegaConf.load("config/base.yaml")
    
    if use_generated:
        df = generate_synthetic_data(5000)
    else:
        df = load_data(df_path)

    # Переименование
    df = df.rename(columns={
        cfg.rename.send_ts: 'send_ts',
        cfg.rename.accept_bank: 'accept_bank',
        cfg.rename.accept_user: 'accept_user',
        cfg.rename.client_id: 'client_id',
    })

    # Feature engineering
    df = create_time_features(df, 'send_ts')

    # Определяем признаки
    cat_features = [c for c in cfg.features.categorical if c in df.columns]

    # Защита от типов и NaN
    df = df.astype({col: 'string' for col in df.select_dtypes(include=['category', 'object']).columns})
    for col in cat_features:
        df[col] = df[col].fillna('missing').astype(str)

    # feature_cols из конфига
    all_features = cfg.features.categorical + cfg.features.numeric + cfg.features.time.all_generated
    feature_cols = [f for f in all_features if f in df.columns]

    # Разбиение
    df = df.sort_values('send_ts')
    train_df = df.iloc[:int(0.7*len(df))]
    val_df = df.iloc[int(0.7*len(df)):int(0.85*len(df))]
    test_df = df.iloc[int(0.85*len(df)):]

    # Обучение
    bank_model = train_bank_model(train_df[feature_cols], train_df['accept_bank'], val_df[feature_cols], val_df['accept_bank'], cat_features)
    train_user = train_df[train_df['accept_bank'] == 1]
    val_user = val_df[val_df['accept_bank'] == 1]
    user_model = train_user_model(train_user[feature_cols], train_user['accept_user'], val_user[feature_cols], val_user['accept_user'], cat_features)

    # Калибровка
    bank_calibrator = calibrate_model(bank_model, val_df[feature_cols], val_df['accept_bank'])
    user_calibrator = calibrate_model(user_model, val_user[feature_cols], val_user['accept_user'])

    # Time grid
    time_grid = build_time_grid()

    # Сохранение
    joblib.dump({'bank_model': bank_model, 'user_model': user_model, 'bank_calibrator': bank_calibrator, 'user_calibrator': user_calibrator, 'time_grid': time_grid, 'cfg': cfg}, f"{output_dir}/models.pkl")

    return {
        'bank_model': bank_model,
        'user_model': user_model,
        'bank_calibrator': bank_calibrator,
        'user_calibrator': user_calibrator,
        'time_grid': time_grid,
        'feature_cols': feature_cols,
        'cat_features': cat_features
    }