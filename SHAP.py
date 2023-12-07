!pip install shap

import shap

# 特徴量の名前をリストとして保存
feature_names = list(df.columns)  # dfは元のDataFrameです

# SHAP値を計算するためのモデルの予測関数を定義（reshape後のデータを使用）
def f(X):
    return model.predict(X.reshape((X.shape[0], n_steps, n_features)))

# SHAP Kernel Explainerを使用してSHAP値を計算（reshape前のデータを使用）
explainer = shap.KernelExplainer(f, X_train[:100].reshape((100, -1)))

# X_validationの最初の10サンプルのSHAP値を計算（reshape前のデータ形式に戻してから渡す）
shap_values = explainer.shap_values(X_validation[:10].reshape((10, -1)))

# SHAP値の可視化（特徴量の名前を指定）
shap.summary_plot(shap_values, X_validation[:10].reshape((10, -1)), feature_names=feature_names, max_display=30)

top_shap_features = [
    'SMA_12_month', 'SMA_24_month', 'WMA_24_month', 'EMA_24_month',
    'EMA_12_month', 'WMA_12_month', 'SMA_50_week', 'BB_Upper_252_std3',
    'MACD_Signal_monthly', 'PSAR', 'BB_Upper_252_std2', 'SMA_6_month',
    'SMA_26_week', 'EMA_6_month', 'EMA_50_week', 'BB_Upper_252_std1', 
    'EMA_26_week', "SMA_75_day", 'BB_Lower_65_std3', 'BB_Lower_252_std1',
    'SMA_52_week', 'SMA_13_week', 'EMA_75_day', 'BB_Lower_252_std2',
    'WMA_50_week', 'WMA_6_month', 'WMA_26_week', 'SMA_50_day', 'WMA_75_day',
    'EMA_52_week', 'Target'
]

df = df[top_shap_features]
