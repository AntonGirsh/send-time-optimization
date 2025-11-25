import pandas as pd
import numpy as np
from pathlib import Path

def generate_and_save(n_samples: int = 5000, path: Path = Path("data/synthetic.parquet")) -> None:
    """
    Генерирует синтетический датасет и сохраняет в parquet.
    """
    np.random.seed(42)
    df = pd.DataFrame({
        'client_id_gen': range(1, n_samples + 1),
        'send_ts_gen': pd.date_range('2025-01-01', periods=n_samples, freq='H'),
        'age_gen': np.random.randint(18, 70, n_samples),
        'app_intensity_gen': np.random.uniform(0, 10, n_samples),
        'country_gen': np.random.choice(['RU', 'KZ', 'BY', 'UA'], n_samples),
        'accept_bank_gen': np.random.binomial(1, 0.5, n_samples),
        'accept_user_gen': np.random.binomial(1, 0.3, n_samples),
    })
    
    # Фиктивные таргеты (в реальности — из логики)
    df['accept_user_gen'] = df['accept_user_gen'] * df['accept_bank_gen']
    
    path.parent.mkdir(exist_ok=True)
    df.to_parquet(path, index=False)