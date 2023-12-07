# データをトレーニング、検証、テスト用に分割
train = df['2000-01-01':'2022-12-31']
val = df['2023-01-01':'2023-9-28']
test = df['2023-09-27':'2023-09-28']

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

feature_columns = [col for col in df.columns if col != 'Target']

# Instantiate a scaler for each feature
scalers = {}

# Create a copy of the dataframe for the scaled data
train_scaled = pd.DataFrame()
val_scaled = pd.DataFrame()
test_scaled = pd.DataFrame()

for feature in feature_columns:
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit on train data
    scaler.fit(train[feature].values.reshape(-1,1))

    # Transform on both train and val data
    train_scaled[feature] = scaler.transform(train[feature].values.reshape(-1,1)).reshape(-1)
    val_scaled[feature] = scaler.transform(val[feature].values.reshape(-1,1)).reshape(-1)
    test_scaled[feature] = scaler.transform(test[feature].values.reshape(-1,1)).reshape(-1)

    # Save the scaler for later use
    scalers[feature] = scaler

from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Define RMSE loss function
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

n_features = train_scaled.shape[1]
n_steps = 1  # number of time steps - it could be more depending on your data

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import plotly.graph_objects as go

# LGBMモデルのインスタンスを作成
model = LGBMRegressor(n_estimators=100, random_state=42)

# モデルをトレーニングデータにフィットさせる
model.fit(train_scaled[feature_columns], train['Target'])

# 予測を行う
y_train_pred = model.predict(train_scaled[feature_columns])
y_validation_pred = model.predict(val_scaled[feature_columns])

# 可視化のための日付インデックスを取得
train_dates = train.index
val_dates = val.index

# Train Dataの可視化
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_dates, y=train['Target'], mode='lines+markers', name='Actual'))
fig.add_trace(go.Scatter(x=train_dates, y=y_train_pred, mode='lines+markers', name='Predicted'))
fig.update_layout(title='Actual vs Predicted Train Data',
                 xaxis_title='Date',
                 yaxis_title='Value')
fig.show()

# Validation Dataの可視化
fig = go.Figure()
fig.add_trace(go.Scatter(x=val_dates, y=val['Target'], mode='lines+markers', name='Actual'))
fig.add_trace(go.Scatter(x=val_dates, y=y_validation_pred, mode='lines+markers', name='Predicted'))
fig.update_layout(title='Actual vs Predicted Validation Data',
                 xaxis_title='Date',
                 yaxis_title='Value')
fig.show()

# バリデーションデータに対する予測を行う
y_validation_pred = model.predict(val_scaled[feature_columns])

# 平均二乗誤差（MSE）を計算
mse = mean_squared_error(val['Target'], y_validation_pred)
print(f"Validation MSE: {mse}")

# ルート平均二乗誤差（RMSE）も計算
rmse = np.sqrt(mse)
print(f"Validation RMSE: {rmse}")

print(model.get_params())
