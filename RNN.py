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
    #scaler = RobustScaler()
    #scaler = StandardScaler()

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

# Define a function for windowing the data
def create_dataset(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Apply the windowing function
X_train, y_train = create_dataset(train_scaled[feature_columns], train['Target'], time_steps=n_steps)
X_validation, y_validation = create_dataset(val_scaled[feature_columns], val['Target'], time_steps=n_steps)
X_test, y_test = create_dataset(test_scaled[feature_columns], test['Target'], time_steps=n_steps)

from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.optimizers import Adam

# RNNモデルの定義
model = Sequential()
model.add(SimpleRNN(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(SimpleRNN(50, activation='relu'))
model.add(Dense(1))

# 学習率を指定したAdamオプティマイザ
optimizer = Adam(learning_rate=0.0001)

# モデルのコンパイル
model.compile(optimizer=optimizer, loss=rmse)  # ここではMSEを損失関数として使用

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], n_steps, n_features))
X_validation = X_validation.reshape((X_validation.shape[0], n_steps, n_features))
X_test = X_test.reshape((X_test.shape[0], n_steps, n_features))

model.summary()

# Train the model
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
history=model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=1000, batch_size=32, verbose=1, callbacks=[early_stop])

# 最良のval_lossの値を取得
best_val_loss = min(history.history['val_loss'])
print(f"The best val_loss is {best_val_loss}")

# Make predictions
y_train_pred = model.predict(X_train)
y_validation_pred = model.predict(X_validation)

# もともとのデータフレームから日付インデックスを取得
train_dates = train.index[n_steps:]  # train dataがn_stepsでwindowされるため、その分日付もずらす
val_dates = val.index[n_steps:]  # 同様にvalidation dataも

# Train Dataの可視化
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_dates, y=y_train, mode='lines+markers', name='Actual'))
fig.add_trace(go.Scatter(x=train_dates, y=y_train_pred.flatten(), mode='lines+markers', name='Predicted'))
fig.update_layout(title='Actual vs Predicted Train Data',
                 xaxis_title='Date',
                 yaxis_title='Value')
fig.show()

# Validation Dataの可視化
fig = go.Figure()
fig.add_trace(go.Scatter(x=val_dates, y=y_validation, mode='lines+markers', name='Actual'))
fig.add_trace(go.Scatter(x=val_dates, y=y_validation_pred.flatten(), mode='lines+markers', name='Predicted'))
fig.update_layout(title='Actual vs Predicted Validation Data',
                 xaxis_title='Date',
                 yaxis_title='Value')
fig.show()

from tensorflow.keras.utils import plot_model

from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.optimizers import Adam

# RNNモデルの定義
model = Sequential()
model.add(SimpleRNN(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(SimpleRNN(50, activation='relu'))
model.add(Dense(1))

# モデルのサマリー図をファイルに保存
plot_model(model, show_shapes=True, show_layer_names=True)
