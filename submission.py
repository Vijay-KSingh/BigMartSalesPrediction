

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# LOAD DATA
# ==========================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_ids = test[['Item_Identifier', 'Outlet_Identifier']].copy()
train_len = len(train)

data = pd.concat([train, test], ignore_index=True)

# ==========================================================
# STABLE FEATURE ENGINEERING (BASELINE)
# ==========================================================

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'reg': 'Regular'
})

data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']

data['Item_Weight'] = data.groupby('Item_Identifier')['Item_Weight'] \
                           .transform(lambda x: x.fillna(x.mean()))

data['Outlet_Size'] = data.groupby('Outlet_Type')['Outlet_Size'] \
                           .transform(lambda x: x.fillna(x.mode()[0]))

data['Item_Visibility'] = data.groupby('Item_Identifier')['Item_Visibility'] \
                               .transform(lambda x: x.replace(0, x.mean()))

data['Item_Visibility_Ratio'] = data['Item_Visibility'] / \
    data.groupby('Item_Identifier')['Item_Visibility'].transform('mean')

# Encode categoricals
cat_cols = data.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

train = data.iloc[:train_len].copy()
test = data.iloc[train_len:].copy()

y = train['Item_Outlet_Sales']
y_log = np.log1p(y)

drop_cols = ['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']
X = train.drop(drop_cols, axis=1)
X_test = test.drop(drop_cols, axis=1)

# ==========================================================
# K-FOLD
# ==========================================================

kf = KFold(n_splits=5, shuffle=True, random_state=42)

n_models = 9
oof_preds = np.zeros((len(X), n_models))
test_preds = np.zeros((len(X_test), n_models))

model_idx = 0

# ==========================================================
# MODEL RUNNER
# ==========================================================

def run_model(model, name):
    global model_idx

    oof = np.zeros(len(X))
    test_fold = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr = y_log.iloc[tr_idx]

        model.fit(X_tr, y_tr)

        oof[val_idx] = model.predict(X_val)
        test_fold += model.predict(X_test) / 5

    oof_preds[:, model_idx] = oof
    test_preds[:, model_idx] = test_fold

    print(f"{name} done")
    model_idx += 1


# ==========================================================
# LIGHTGBM
# ==========================================================

run_model(LGBMRegressor(n_estimators=3500, learning_rate=0.007,
                        num_leaves=48, subsample=0.75,
                        colsample_bytree=0.75, random_state=42), "LGB1")

run_model(LGBMRegressor(n_estimators=4000, learning_rate=0.006,
                        num_leaves=64, subsample=0.8,
                        colsample_bytree=0.8, random_state=1), "LGB2")

run_model(LGBMRegressor(n_estimators=3200, learning_rate=0.01,
                        num_leaves=40, subsample=0.7,
                        colsample_bytree=0.7, random_state=7), "LGB3")

run_model(LGBMRegressor(n_estimators=3700, learning_rate=0.008,
                        num_leaves=56, subsample=0.8,
                        colsample_bytree=0.8, random_state=99), "LGB4")

# ==========================================================
# XGBOOST
# ==========================================================

run_model(XGBRegressor(n_estimators=3000, learning_rate=0.01,
                       max_depth=6, subsample=0.8,
                       colsample_bytree=0.8, tree_method='hist',
                       random_state=42), "XGB1")

run_model(XGBRegressor(n_estimators=3500, learning_rate=0.008,
                       max_depth=8, subsample=0.8,
                       colsample_bytree=0.8, tree_method='hist',
                       random_state=1), "XGB2")

run_model(XGBRegressor(n_estimators=2500, learning_rate=0.015,
                       max_depth=6, subsample=0.7,
                       colsample_bytree=0.7, tree_method='hist',
                       random_state=99), "XGB3")

# ==========================================================
# CATBOOST
# ==========================================================

run_model(CatBoostRegressor(iterations=3000, learning_rate=0.01,
                            depth=6, verbose=False,
                            random_seed=42), "CAT1")

run_model(CatBoostRegressor(iterations=3500, learning_rate=0.008,
                            depth=8, verbose=False,
                            random_seed=1), "CAT2")

# ==========================================================
# META LEARNER (RIDGE ONLY - BEST FOUND)
# ==========================================================

ridge = Ridge(alpha=1.0)
ridge.fit(oof_preds, y_log)

stack_oof = ridge.predict(oof_preds)
stack_test = ridge.predict(test_preds)

print("Stack OOF RMSE:",
      np.sqrt(mean_squared_error(y, np.expm1(stack_oof))))

# ==========================================================
# RESIDUAL MODELING
# ==========================================================

residuals = y_log - stack_oof

res_oof = np.zeros(len(X))
res_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    r_tr = residuals[tr_idx]

    res_model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=40,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=fold
    )

    res_model.fit(X_tr, r_tr)

    res_oof[val_idx] = res_model.predict(X_val)
    res_test += res_model.predict(X_test) / 5

# Add residual correction
final_oof = stack_oof + 0.5 * res_oof
final_test = stack_test + 0.5 * res_test

# ==========================================================
# FINAL EVALUATION
# ==========================================================

oof_original = np.expm1(final_oof)
rmse = np.sqrt(mean_squared_error(y, oof_original))
print("Final OOF RMSE after residual:", rmse)

# ==========================================================
# SUBMISSION
# ==========================================================

final_preds = np.expm1(final_test)

submission = pd.DataFrame({
    'Item_Identifier': test_ids['Item_Identifier'],
    'Outlet_Identifier': test_ids['Outlet_Identifier'],
    'Item_Outlet_Sales': final_preds
})

submission.to_csv("submission_full_stack_residual-3.csv", index=False)
print("Final submission created.")