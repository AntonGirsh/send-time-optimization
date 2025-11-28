# src/prediction.py
import numpy as np
import pandas as pd

def predict_with_uplift(
    artifacts: dict,
    df: pd.DataFrame,
    n_random: int = 20,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Делает предикт + сравнение с random.
    Возвращает таблицу с колонкой uplift_pct.
    """
    np.random.seed(random_seed)
    
    bank_model = artifacts['bank_model']
    user_model = artifacts['user_model']
    bank_calibrator = artifacts['bank_calibrator']
    user_calibrator = artifacts['user_calibrator']
    time_grid = artifacts['time_grid']
    feature_cols = artifacts['feature_cols']

    results = []

    for _, row in df.iterrows():
        client_id = row['client_id']
        client_features = row[['country', 'age', 'app_intensity']].to_dict()

        # Создаём гриду 168 слотов для этого клиента
        grid = time_grid.copy()
        for k, v in client_features.items():
            grid[k] = v

        # P(bank)
        p_bank = bank_calibrator.predict(
            bank_model.predict_proba(grid[feature_cols])[:, 1]
        )

        # P(user | bank=1)
        positive_mask = p_bank > 0.0
        p_user_full = np.zeros(len(grid))
        if positive_mask.any():
            p_user_raw = user_model.predict_proba(grid.loc[positive_mask, feature_cols])[:, 1]
            p_user_full[positive_mask] = user_calibrator.predict(p_user_raw)

        # Итоговая конверсия
        score = p_bank * p_user_full

        # Лучшее время
        best_idx = score.argmax()
        best_score = score[best_idx]

        # Random времена
        random_indices = np.random.choice(len(grid), size=n_random, replace=True)
        random_score = score[random_indices].mean()

        # Uplift
        uplift_abs = best_score - random_score
        uplift_pct = uplift_abs / random_score if random_score > 0 else 0

        results.append({
            'client_id': int(client_id),
            'best_hour': int(time_grid.iloc[best_idx]['hour']),
            'best_dow': int(time_grid.iloc[best_idx]['dow']),
            'best_score': float(best_score),
            'random_score': float(random_score),
            'uplift_abs': float(uplift_abs),
            'uplift_pct': float(uplift_pct),
        })

    result_df = pd.DataFrame(results)
    
    print(f"Expected conversion (model):   {result_df['best_score'].mean():.4f}")
    print(f"Expected conversion (random):  {result_df['random_score'].mean():.4f}")
    print(f"Mean uplift:                  +{result_df['uplift_pct'].mean()*100:5.2f}%")

    return result_df