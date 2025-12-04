from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

# def train_bank_model(X_train, y_train, X_val, y_val, cat_features):
#     model = CatBoostClassifier(
#         iterations=800,
#         depth=6,
#         learning_rate=0.08,
#         l2_leaf_reg=20,
#         bagging_temperature=0.9,
#         random_strength=10,
#         random_seed=42,
#         eval_metric='AUC',
#         verbose=100,
#         early_stopping_rounds=80,
#         cat_features=cat_features,
#         border_count=128,
#         grow_policy='Lossguide'  # лучше для малых датасетов
#     )
#     model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
#     return model


def train_user_model(X_train, y_train, X_val, y_val, cat_features):
    model = CatBoostClassifier(
        iterations=600,
        depth=4,
        learning_rate=0.10,
        l2_leaf_reg=50,
        bagging_temperature=1.0,
        random_strength=10,
        random_seed=42,
        eval_metric='AUC',
        verbose=100,
        early_stopping_rounds=60,
        cat_features=cat_features,
        border_count=64,
        grow_policy='Lossguide'
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    return model



def train_bank_model(X_train, y_train, X_val, y_val, cat_features):
    model = CatBoostClassifier(
        iterations=600,
        depth=4,                    # было 10 → стало 4
        learning_rate=0.1,
        l2_leaf_reg=60,             # было 3 → стало 50!
        bagging_temperature=1.0,
        random_strength=10,
        subsample=0.8,
        border_count=32,            # меньше сплитов → меньше шанса на случайные
        early_stopping_rounds=50,
        eval_metric='AUC',
        verbose=100,
        cat_features=cat_features
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    return model


# def train_user_model(X_train, y_train, X_val, y_val, cat_features):
#     model = CatBoostClassifier(
#         iterations=500,
#         depth=4,
#         learning_rate=0.12,
#         l2_leaf_reg=80,             # критично!
#         bagging_temperature=1.0,
#         random_strength=15,
#         subsample=0.7,
#         border_count=32,
#         early_stopping_rounds=50,
#     )
#     model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
#     return model

def calibrate_model(model, X_calib, y_calib):
    raw_probs = model.predict_proba(X_calib)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_probs, y_calib)
    return calibrator