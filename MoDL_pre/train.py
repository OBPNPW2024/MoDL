import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib



train_path = '../deform/function_pre/U87 cell.csv'

train_data = pd.read_csv(train_path)

X_train = train_data.iloc[:, :-5]
y_train_MMP = train_data.iloc[:, -5]
y_train_ATP = train_data.iloc[:, -4]
y_train_ROS = train_data.iloc[:, -3]
y_train_respiration = train_data.iloc[:, -2]
y_train_mitophagy = train_data.iloc[:, -1]


# MMP
xgb1_MMP = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    booster='gbtree',
    objective='reg:gamma',
    gamma=0.1,
    max_depth=5,
    reg_lambda=3,
    subsample=0.7,
    colsample_bytree=1,
    min_child_weight=5,
    eta=0.1,
    seed=5,
    nthread=4
)

lgb1_MMP = LGBMRegressor(
    boosting_type='gbdt',
    objective='regression',
    num_leaves=31,
    learning_rate=0.001,
    n_estimators=3000,
    max_depth=7,
    reg_lambda=0.01,
    subsample=1,
    colsample_bytree=0.7
)

xgb2_MMP = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    booster='gbtree',
    objective='reg:gamma',
    gamma=0.1,
    max_depth=5,
    reg_lambda=3,
    subsample=0.7,
    colsample_bytree=1,
    min_child_weight=5,
    eta=0.1,
    seed=5,
    nthread=4
)

xgb1_MMP.fit(X_train, y_train_MMP)
xgb1_preds_train_MMP = xgb1_MMP.predict(X_train)
X_train_2_MMP = pd.concat([X_train, pd.DataFrame(xgb1_preds_train_MMP, columns=['xgb1_preds_MMP'])], axis=1)
lgb1_MMP.fit(X_train_2_MMP, y_train_MMP)
lgb1_preds_train_MMP = lgb1_MMP.predict(X_train_2_MMP)
X_train_3_MMP = pd.concat([X_train_2_MMP, pd.DataFrame(lgb1_preds_train_MMP, columns=['lgb1_preds_MMP'])], axis=1)

xgb2_MMP.fit(X_train_3_MMP, y_train_MMP)

joblib.dump(xgb1_MMP, '../model/xgb1_MMP.pkl')
joblib.dump(lgb1_MMP, '../model/lgb1_MMP.pkl')
joblib.dump(xgb2_MMP, '../model/xgb2_MMP.pkl')
print("MMP model completed")

# ATP
xgb1_ATP = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    booster='gbtree',
    objective='reg:gamma',
    gamma=0.1,
    max_depth=5,
    reg_lambda=3,
    subsample=0.7,
    colsample_bytree=1,
    min_child_weight=5,
    eta=0.1,
    seed=5,
    nthread=4
)

lgb1_ATP = LGBMRegressor(
    boosting_type='gbdt',
    objective='regression',
    num_leaves=31,
    learning_rate=0.001,
    n_estimators=3000,
    max_depth=7,
    reg_lambda=0.01,
    subsample=1,
    colsample_bytree=0.7
)

xgb2_ATP = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    booster='gbtree',
    objective='reg:gamma',
    gamma=0.1,
    max_depth=5,
    reg_lambda=3,
    subsample=0.7,
    colsample_bytree=1,
    min_child_weight=5,
    eta=0.1,
    seed=5,
    nthread=4
)

xgb1_ATP.fit(X_train, y_train_ATP)
xgb1_preds_train_ATP = xgb1_ATP.predict(X_train)
X_train_2_ATP = pd.concat([X_train, pd.DataFrame(xgb1_preds_train_ATP, columns=['xgb1_preds_ATP'])], axis=1)
lgb1_ATP.fit(X_train_2_ATP, y_train_ATP)
lgb1_preds_train_ATP = lgb1_ATP.predict(X_train_2_ATP)
X_train_3_ATP = pd.concat([X_train_2_ATP, pd.DataFrame(lgb1_preds_train_ATP, columns=['lgb1_preds_ATP'])], axis=1)

xgb2_ATP.fit(X_train_3_ATP, y_train_ATP)

joblib.dump(xgb1_ATP, '../model/xgb1_ATP.pkl')
joblib.dump(lgb1_ATP, '../model/lgb1_ATP.pkl')
joblib.dump(xgb2_ATP, '../model/xgb2_ATP.pkl')
print("ATP model completed")

# ROS
xgb1_ROS = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    booster='gbtree',
    objective='reg:gamma',
    gamma=0.1,
    max_depth=5,
    reg_lambda=3,
    subsample=0.7,
    colsample_bytree=1,
    min_child_weight=5,
    eta=0.1,
    seed=5,
    nthread=4
)

lgb1_ROS = LGBMRegressor(
    boosting_type='gbdt',
    objective='regression',
    num_leaves=31,
    learning_rate=0.001,
    n_estimators=3000,
    max_depth=7,
    reg_lambda=0.01,
    subsample=1,
    colsample_bytree=0.7
)

xgb2_ROS = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    booster='gbtree',
    objective='reg:gamma',
    gamma=0.1,
    max_depth=5,
    reg_lambda=3,
    subsample=0.7,
    colsample_bytree=1,
    min_child_weight=5,
    eta=0.1,
    seed=5,
    nthread=4
)

xgb1_ROS.fit(X_train, y_train_ROS)
xgb1_preds_train_ROS = xgb1_ROS.predict(X_train)
X_train_2_ROS = pd.concat([X_train, pd.DataFrame(xgb1_preds_train_ROS, columns=['xgb1_preds_ROS'])], axis=1)
lgb1_ROS.fit(X_train_2_ROS, y_train_ROS)
lgb1_preds_train_ROS = lgb1_ROS.predict(X_train_2_ROS)
X_train_3_ROS = pd.concat([X_train_2_ROS, pd.DataFrame(lgb1_preds_train_ROS, columns=['lgb1_preds_ROS'])], axis=1)

xgb2_ROS.fit(X_train_3_ROS, y_train_ROS)

joblib.dump(xgb1_ROS, '../model/xgb1_ROS.pkl')
joblib.dump(lgb1_ROS, '../model/lgb1_ROS.pkl')
joblib.dump(xgb2_ROS, '../model/xgb2_ROS.pkl')
print("ROS model completed")

# respiration
xgb1_respiration = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    booster='gbtree',
    objective='reg:gamma',
    gamma=0.1,
    max_depth=5,
    reg_lambda=3,
    subsample=0.7,
    colsample_bytree=1,
    min_child_weight=5,
    eta=0.1,
    seed=5,
    nthread=4
)

lgb1_respiration = LGBMRegressor(
    boosting_type='gbdt',
    objective='regression',
    num_leaves=31,
    learning_rate=0.001,
    n_estimators=3000,
    max_depth=7,
    reg_lambda=0.01,
    subsample=1,
    colsample_bytree=0.7
)

xgb2_respiration = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    booster='gbtree',
    objective='reg:gamma',
    gamma=0.1,
    max_depth=5,
    reg_lambda=3,
    subsample=0.7,
    colsample_bytree=1,
    min_child_weight=5,
    eta=0.1,
    seed=5,
    nthread=4
)

xgb1_respiration.fit(X_train, y_train_respiration)
xgb1_preds_train_respiration = xgb1_respiration.predict(X_train)
X_train_2_respiration = pd.concat([X_train, pd.DataFrame(xgb1_preds_train_respiration, columns=['xgb1_preds_respiration'])], axis=1)
lgb1_respiration.fit(X_train_2_respiration, y_train_respiration)
lgb1_preds_train_respiration = lgb1_respiration.predict(X_train_2_respiration)
X_train_3_respiration = pd.concat([X_train_2_respiration, pd.DataFrame(lgb1_preds_train_respiration, columns=['lgb1_preds_respiration'])], axis=1)

xgb2_respiration.fit(X_train_3_respiration, y_train_respiration)

joblib.dump(xgb1_respiration, '../model/xgb1_respiration.pkl')
joblib.dump(lgb1_respiration, '../model/lgb1_respiration.pkl')
joblib.dump(xgb2_respiration, '../model/xgb2_respiration.pkl')
print("respiration model completed")

# mitophagy
xgb1_mitophagy = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    booster='gbtree',
    objective='reg:gamma',
    gamma=0.1,
    max_depth=5,
    reg_lambda=3,
    subsample=0.7,
    colsample_bytree=1,
    min_child_weight=5,
    eta=0.1,
    seed=5,
    nthread=4
)

lgb1_mitophagy = LGBMRegressor(
    boosting_type='gbdt',
    objective='regression',
    num_leaves=31,
    learning_rate=0.001,
    n_estimators=3000,
    max_depth=7,
    reg_lambda=0.01,
    subsample=1,
    colsample_bytree=0.7
)

xgb2_mitophagy = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    booster='gbtree',
    objective='reg:gamma',
    gamma=0.1,
    max_depth=5,
    reg_lambda=3,
    subsample=0.7,
    colsample_bytree=1,
    min_child_weight=5,
    eta=0.1,
    seed=5,
    nthread=4
)

xgb1_mitophagy.fit(X_train, y_train_mitophagy)
xgb1_preds_train_mitophagy = xgb1_mitophagy.predict(X_train)
X_train_2_mitophagy = pd.concat([X_train, pd.DataFrame(xgb1_preds_train_mitophagy, columns=['xgb1_preds_mitophagy'])], axis=1)
lgb1_mitophagy.fit(X_train_2_mitophagy, y_train_mitophagy)
lgb1_preds_train_mitophagy = lgb1_mitophagy.predict(X_train_2_mitophagy)
X_train_3_mitophagy = pd.concat([X_train_2_mitophagy, pd.DataFrame(lgb1_preds_train_mitophagy, columns=['lgb1_preds_mitophagy'])], axis=1)

xgb2_mitophagy.fit(X_train_3_mitophagy, y_train_mitophagy)

joblib.dump(xgb1_mitophagy, '../model/xgb1_mitophagy.pkl')
joblib.dump(lgb1_mitophagy, '../model/lgb1_mitophagy.pkl')
joblib.dump(xgb2_mitophagy, '../model/xgb2_mitophagy.pkl')
print("mitophagy model completed")
print("All done")
