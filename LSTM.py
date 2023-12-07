from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.layers import LeakyReLU
from keras.layers import ELU

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))

# 学習率を指定したAdamオプティマイザ
optimizer = Adam(learning_rate=0.0001)

# モデルのコンパイル
model.compile(optimizer=optimizer, loss=rmse)

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

test_pred = model.predict(X_test)
print(f"Predicted Closing Price for the Next Day: {test_pred[0][0]:.2f}") #9/29 31857
