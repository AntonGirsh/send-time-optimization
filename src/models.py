from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

def train_bank_model(X_train, y_train, X_val, y_val, cat_features):
    """
    Модель P(accept_bank = 1 | x, t) — обучается на ВСЕХ данных
    """
    model = CatBoostClassifier(
        iterations=2000,
        depth=10,
        learning_rate=0.03,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=200,
        task_type="GPU" if __import__('torch').cuda.is_available() else "CPU",
        cat_features=cat_features
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    return model

def train_user_model(X_train, y_train, X_val, y_val, cat_features):
    """
    Модель P(accept_user = 1 | x, t, accept_bank=1) — обучается ТОЛЬКО на bank==1
    """
    model = CatBoostClassifier(
        iterations=1500,
        depth=8,
        learning_rate=0.05,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=200,
        task_type="GPU" if __import__('torch').cuda.is_available() else "CPU",
        cat_features=cat_features
    )
    
    model.fit(
        X_train_user, y_train_user,
        eval_set=(X_val_user, y_val_user),
        use_best_model=True
    )
    return model

def calibrate_model(model, X_calib, y_calib):
    """
    Isotonic Regression — монотонная калибровка, идеально для GBDT.
    Работает лучше Platt/temperature scaling на табличных данных.
    """
    raw_probs = model.predict_proba(X_calib)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_probs, y_calib)
    return calibrator