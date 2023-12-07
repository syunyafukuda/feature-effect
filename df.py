# ライブラリの読み込み

from google.colab import drive
drive.mount('/content/drive')

%%shell
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
python -m pip install git+https://github.com/TA-Lib/ta-lib-python.git@TA_Lib-0.4.26

import talib as ta

# ライブラリのインポート
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.simplefilter('ignore')

!pip install japanize-matplotlib
import japanize_matplotlib
!pip install yfinance
import yfinance as yfin
yfin.pdr_override()
!pip install mplfinance
import mplfinance as mpf
!pip install requests
import requests
!pip install plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
!pip install fredapi pandas_datareader
from fredapi import Fred
from pandas_datareader.data import DataReader
!pip install pdfminer.six
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.converter import PDFPageAggregator
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.layout import LTTextBoxHorizontal
from pdfminer.pdfpage import PDFPage
from io import StringIO

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# データの取得と目的変数の作成

# 日経平均株価データの取得
start = '1990-01-01'
end = datetime.now().strftime('2023-09-30')
data_master = data.get_data_yahoo('^N225', start, end)

# 前処理
df = data_master.copy()
df.drop(columns=['Adj Close'], inplace=True)
df['Target'] = df['Close'].shift(-1)

# テクニカル指標系

# 実体部分の追加
df['Body'] = df['Open'] - df['Close']

# 日毎のRSI
df['RSI_9'] = ta.RSI(df['Close'], timeperiod=9) #9日
df['RSI_14'] = ta.RSI(df['Close'], timeperiod=14) #14日
df['RSI_22'] = ta.RSI(df['Close'], timeperiod=22) #22日
df['RSI_42'] = ta.RSI(df['Close'], timeperiod=42) #42日
df['RSI_52'] = ta.RSI(df['Close'], timeperiod=52) #52日

# 週毎のRSI（週5日のマーケットを仮定）
df['RSI_9_week'] = ta.RSI(df['Close'], timeperiod=9*5) #9週
df['RSI_13_week'] = ta.RSI(df['Close'], timeperiod=13*5) #13週
df['RSI_26_week'] = ta.RSI(df['Close'], timeperiod=26*5)  # 26週

# 月毎のRSI（週5日のマーケットを仮定）
df['RSI_3_month'] = ta.RSI(df['Close'], timeperiod=13*5) # 3ヶ月（約13週）
df['RSI_6_month'] = ta.RSI(df['Close'], timeperiod=26*5) # 6ヶ月（約26週）
df['RSI_12_month'] = ta.RSI(df['Close'], timeperiod=52*5) # 12ヶ月（約52週）

# 日単位の単純移動平均
df['SMA_5_day'] = ta.SMA(df['Close'].rolling(window=5).mean())
df['SMA_25_day'] = ta.SMA(df['Close'].rolling(window=25).mean())
df['SMA_50_day'] = ta.SMA(df['Close'].rolling(window=50).mean())
df['SMA_75_day'] = ta.SMA(df['Close'].rolling(window=75).mean())
df['SMA_200_day'] = ta.SMA(df['Close'].rolling(window=200).mean())

# 週単位の単純移動平均（週5日のマーケットを仮定）
df['SMA_9_week'] = ta.SMA(df['Close'].rolling(window=9*5).mean())
df['SMA_13_week'] = ta.SMA(df['Close'].rolling(window=13*5).mean())
df['SMA_26_week'] = ta.SMA(df['Close'].rolling(window=26*5).mean())
df['SMA_50_week'] = ta.SMA(df['Close'].rolling(window=50*5).mean())
df['SMA_52_week'] = ta.SMA(df['Close'].rolling(window=52*5).mean())

# 月単位の単純移動平均（月平均21取引日を仮定）
df['SMA_6_month'] = ta.SMA(df['Close'].rolling(window=6*21).mean())
df['SMA_12_month'] = ta.SMA(df['Close'].rolling(window=12*21).mean())
df['SMA_24_month'] = ta.SMA(df['Close'].rolling(window=24*21).mean())
df['SMA_60_month'] = ta.SMA(df['Close'].rolling(window=60*21).mean())

# 日単位のEMA (Exponential Moving Average)
df['EMA_5_day'] = ta.EMA(df['Close'], timeperiod=5)
df['EMA_25_day'] = ta.EMA(df['Close'], timeperiod=25)
df['EMA_50_day'] = ta.EMA(df['Close'], timeperiod=50)
df['EMA_75_day'] = ta.EMA(df['Close'], timeperiod=75)
df['EMA_200_day'] = ta.EMA(df['Close'], timeperiod=200)

# 週単位のEMA (Exponential Moving Average)(週に5取引日と仮定）
df['EMA_9_week'] = ta.EMA(df['Close'], timeperiod=9*5)
df['EMA_13_week'] = ta.EMA(df['Close'], timeperiod=13*5)
df['EMA_26_week'] = ta.EMA(df['Close'], timeperiod=26*5)
df['EMA_50_week'] = ta.EMA(df['Close'], timeperiod=50*5)
df['EMA_52_week'] = ta.EMA(df['Close'], timeperiod=52*5)

# 月単位のEMA (Exponential Moving Average)(月に21取引日と仮定）
df['EMA_6_month'] = ta.EMA(df['Close'], timeperiod=6*21)
df['EMA_12_month'] = ta.EMA(df['Close'], timeperiod=12*21)
df['EMA_24_month'] = ta.EMA(df['Close'], timeperiod=24*21)
df['EMA_60_month'] = ta.EMA(df['Close'], timeperiod=60*21)

# 日単位のWMA (Weighted Moving Average)
df['WMA_5_day'] = ta.WMA(df['Close'], timeperiod=5)
df['WMA_25_day'] = ta.WMA(df['Close'], timeperiod=25)
df['WMA_50_day'] = ta.WMA(df['Close'], timeperiod=50)
df['WMA_75_day'] = ta.WMA(df['Close'], timeperiod=75)
df['WMA_200_day'] = ta.WMA(df['Close'], timeperiod=200)

# 週単位のWMA (Exponential Moving Average)(週に5取引日と仮定）
df['WMA_9_week'] = ta.WMA(df['Close'], timeperiod=9*5)
df['WMA_13_week'] = ta.WMA(df['Close'], timeperiod=13*5)
df['WMA_26_week'] = ta.WMA(df['Close'], timeperiod=26*5)
df['WMA_50_week'] = ta.WMA(df['Close'], timeperiod=50*5)
df['WMA_52_week'] = ta.WMA(df['Close'], timeperiod=52*5)

# 月単位のWMA (Exponential Moving Average)(月に21取引日と仮定）
df['WMA_6_month'] = ta.WMA(df['Close'], timeperiod=6*21)
df['WMA_12_month'] = ta.WMA(df['Close'], timeperiod=12*21)
df['WMA_24_month'] = ta.WMA(df['Close'], timeperiod=24*21)
df['WMA_60_month'] = ta.WMA(df['Close'], timeperiod=60*21)

# 一目均衡表の追加
def ichimoku(df, high_col='High', low_col='Low', close_col='Close'):
    # 転換線 (Conversion Line)
    nine_period_high = df[high_col].rolling(window=9).max()
    nine_period_low = df[low_col].rolling(window=9).min()
    df['ichimoku_Conversion_Line'] = (nine_period_high + nine_period_low) / 2

    # 基準線 (Base Line)
    twenty_six_period_high = df[high_col].rolling(window=26).max()
    twenty_six_period_low = df[low_col].rolling(window=26).min()
    df['ichimoku_Base_Line'] = (twenty_six_period_high + twenty_six_period_low) / 2

    # 先行スパン A (Leading Span A)
    df['ichimoku_Leading_Span A'] = ((df['ichimoku_Conversion_Line'] + df['ichimoku_Base_Line']) / 2).shift(26)

    # 先行スパン B (Leading Span B)
    fifty_two_period_high = df[high_col].rolling(window=52).max()
    fifty_two_period_low = df[low_col].rolling(window=52).min()
    df['ichimoku_Leading_Span B'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)

    # 遅行スパン (Lagging Span)
    df['ichimoku_Lagging_Span'] = df[close_col].shift(-26)

ichimoku(df)

# ボリンジャーバンドの追加
periods = [5,
           20,
           25,
           65, #13週
           252 #12か月
           ]

num_stds = [1, 2, 3]

for period in periods:
    for std in num_stds:
        upper, middle, lower = ta.BBANDS(df['Close'].values, timeperiod=period, nbdevup=std, nbdevdn=std)
        df[f'BB_Upper_{period}_std{std}'] = upper
        df[f'BB_Lower_{period}_std{std}'] = lower

# 'SMA_'で始まる列名を自動的に探してリストに格納
sma_columns = df.filter(like='SMA_').columns.tolist()

# 移動平均乖離率の追加
for sma_column in sma_columns:
    df[f'{sma_column}_DevRate'] = (df['Close'] - df[sma_column]) / df[sma_column] * 100

# 日毎MACDの追加
def add_macd(df, close_col='Close', fastperiod=12, slowperiod=26, signalperiod=9):
    macd, macd_signal, macd_hist = ta.MACD(df[close_col], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist

add_macd(df)

# 週毎MACDの追加
def add_macd(df, close_col='Close', fastperiod=12, slowperiod=26, signalperiod=9):
    df_copy = df.copy()  # 元のDataFrameを変更しないためにコピーを作成
    macd, macd_signal, macd_hist = ta.MACD(df_copy[close_col], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    df_copy['MACD_weekly'] = macd
    df_copy['MACD_Signal_weekly'] = macd_signal
    df_copy['MACD_Hist_weekly'] = macd_hist
    return df_copy  # 変更を加えたDataFrameを返す

# 週毎にリサンプリングして終値の平均を取る
df_weekly = df.resample('W').agg({'Close':'last'})  # 'W'は週毎、'last'は週の最後のデータを取る
df_weekly.fillna(method='ffill', inplace=True)  # 前方補間

# 新しい周期でMACDを計算
df_weekly_with_macd = add_macd(df_weekly, close_col='Close')

# インデックスをDatetimeIndexに変換
df.index = pd.to_datetime(df.index)

# 週の最後の日付を取得
week_last_day = df.resample('W').last().index

# 新しいカラムに週の最後の日付をセット
df['Week_last_day'] = df.index.to_series().apply(lambda x: week_last_day[week_last_day >= x][0])

# 元の日毎のデータに週の最後の日付を追加
df_weekly_with_macd['Week_last_day'] = df_weekly_with_macd.index

# インデックスを保存
original_index = df.index

df = pd.merge(df, df_weekly_with_macd[['Week_last_day', 'MACD_weekly', 'MACD_Signal_weekly', 'MACD_Hist_weekly']], on='Week_last_day', how='left')

# インデックスを再設定
df.index = original_index

df.drop('Week_last_day', axis=1, inplace=True)

# 月毎MACDの追加
def add_monthly_macd(df, close_col='Close', fastperiod=12, slowperiod=26, signalperiod=9):
    df_copy = df.copy()  # 元のDataFrameを変更しないためにコピーを作成
    macd, macd_signal, macd_hist = ta.MACD(df_copy[close_col], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    df_copy['MACD_monthly'] = macd
    df_copy['MACD_Signal_monthly'] = macd_signal
    df_copy['MACD_Hist_monthly'] = macd_hist
    return df_copy  # 変更を加えたDataFrameを返す

# 月毎にリサンプリングして終値を取る
df_monthly = df.resample('M').agg({'Close':'last'})  # 'M'は月毎、'last'は月の最後のデータを取る
df_monthly.fillna(method='ffill', inplace=True)  # 前方補間

# 新しい周期でMACDを計算
df_monthly_with_macd = add_monthly_macd(df_monthly, close_col='Close')

# 月の最後の日付を取得
month_last_day = df.resample('M').last().index

# 新しいカラムに月の最後の日付をセット
df['Month_last_day'] = df.index.to_series().apply(lambda x: month_last_day[month_last_day >= x][0])

# 元の日毎のデータに月の最後の日付を追加
df_monthly_with_macd['Month_last_day'] = df_monthly_with_macd.index

# インデックスを保存
original_index = df.index

# マージ
df = pd.merge(df, df_monthly_with_macd[['Month_last_day', 'MACD_monthly', 'MACD_Signal_monthly', 'MACD_Hist_monthly']], on='Month_last_day', how='left')

# インデックスを再設定
df.index = original_index

# 不要なカラムを削除
df.drop('Month_last_day', axis=1, inplace=True)

# 日毎のストキャスティクスの追加
def add_stochastic(df, high_col='High', low_col='Low', close_col='Close', fastk_period=5, slowk_period=3, slowd_period=3):
    slowk, slowd = ta.STOCH(df[high_col], df[low_col], df[close_col], fastk_period=fastk_period, slowk_period=slowk_period, slowd_period=slowd_period)
    df['Slow_Stochastic_K'] = slowk
    df['Slow_Stochastic_D'] = slowd

    fastk, fastd = ta.STOCHF(df[high_col], df[low_col], df[close_col], fastk_period=fastk_period, fastd_period=slowd_period)
    df['Fast_Stochastic_K'] = fastk
    df['Fast_Stochastic_D'] = fastd

add_stochastic(df)

# 週毎ストキャスティクスの追加
def add_weekly_stochastics(df, high_col='High', low_col='Low', close_col='Close',
                           fastk_period=5, fastd_period=3, slowk_period=5, slowd_period=3):
    df_copy = df.copy()

    # Fast Stochasticを計算
    fastk, fastd = ta.STOCHF(df_copy[high_col], df_copy[low_col], df_copy[close_col],
                             fastk_period=fastk_period, fastd_period=fastd_period)
    df_copy['Fast_Stochastic_K_weekly'] = fastk
    df_copy['Fast_Stochastic_D_weekly'] = fastd

    # Slow Stochasticを計算
    slowk, slowd = ta.STOCH(df_copy[high_col], df_copy[low_col], df_copy[close_col],
                            fastk_period=slowk_period, slowd_period=slowd_period)
    df_copy['Slow_Stochastic_K_weekly'] = slowk
    df_copy['Slow_Stochastic_D_weekly'] = slowd

    return df_copy

# 週毎にリサンプリングして各種データを取る
df_weekly = df.resample('W').agg({'High':'max', 'Low':'min', 'Close':'last'})
df_weekly.fillna(method='ffill', inplace=True)

# 週毎のストキャスティクスを計算
df_weekly_with_stochastics = add_weekly_stochastics(df_weekly)  # 関数名を修正

# 新しいカラムに週の最後の日付をセット
df_weekly_with_stochastics['Week_last_day'] = df_weekly_with_stochastics.index

# インデックスをDatetimeIndexに変換
df.index = pd.to_datetime(df.index)

# 週の最後の日付を取得
week_last_day = df.resample('W').last().index

# 新しいカラムに週の最後の日付をセット
df['Week_last_day'] = df.index.to_series().apply(lambda x: week_last_day[week_last_day >= x][0])

# 元の日毎のデータに週毎のストキャスティクスをマージ
df = pd.merge(df, df_weekly_with_stochastics[['Week_last_day', 'Fast_Stochastic_K_weekly', 'Fast_Stochastic_D_weekly', 'Slow_Stochastic_K_weekly', 'Slow_Stochastic_D_weekly']], on='Week_last_day', how='left')  # カラムを追加

# インデックスを再設定
df.index = original_index

# Week_last_dayカラムを削除
df.drop('Week_last_day', axis=1, inplace=True)

# 月毎ストキャスティクスの追加
def add_monthly_stochastics(df, high_col='High', low_col='Low', close_col='Close',
                            fastk_period=5, fastd_period=3, slowk_period=5, slowd_period=3):
    df_copy = df.copy()

    # Fast Stochasticを計算
    fastk, fastd = ta.STOCHF(df_copy[high_col], df_copy[low_col], df_copy[close_col],
                             fastk_period=fastk_period, fastd_period=fastd_period)
    df_copy['Fast_Stochastic_K_monthly'] = fastk
    df_copy['Fast_Stochastic_D_monthly'] = fastd

    # Slow Stochasticを計算
    slowk, slowd = ta.STOCH(df_copy[high_col], df_copy[low_col], df_copy[close_col],
                            fastk_period=slowk_period, slowd_period=slowd_period)
    df_copy['Slow_Stochastic_K_monthly'] = slowk
    df_copy['Slow_Stochastic_D_monthly'] = slowd

    return df_copy

# 月毎にリサンプリングして各種データを取る
df_monthly = df.resample('M').agg({'High':'max', 'Low':'min', 'Close':'last'})
df_monthly.fillna(method='ffill', inplace=True)

# 月毎のストキャスティクスを計算
df_monthly_with_stochastics = add_monthly_stochastics(df_monthly)

# 新しいカラムに月の最後の日付をセット
df_monthly_with_stochastics['Month_last_day'] = df_monthly_with_stochastics.index

# 月の最後の日付を取得
month_last_day = df.resample('M').last().index

# 新しいカラムに月の最後の日付をセット
df['Month_last_day'] = df.index.to_series().apply(lambda x: month_last_day[month_last_day >= x][0])

# 元の日毎のデータに月毎のストキャスティクスをマージ
df = pd.merge(df, df_monthly_with_stochastics[['Month_last_day', 'Fast_Stochastic_K_monthly', 'Fast_Stochastic_D_monthly', 'Slow_Stochastic_K_monthly', 'Slow_Stochastic_D_monthly']], on='Month_last_day', how='left')

# インデックスを再設定
df.index = original_index

# Month_last_dayカラムを削除
df.drop('Month_last_day', axis=1, inplace=True)

from scipy.stats import spearmanr

# RCIの計算
def calc_rci(df, column, window):
    if len(df) < window:
        return np.nan
    ranks = df[column].rank()
    return 100 * (1 - 6 * sum((ranks - np.arange(window) - 1)**2) / (window * (window**2 - 1)))

# DataFrameにRCIを追加
def add_rci(df, column, window, suffix):
    rci_values = []
    for i in range(len(df)):
        if i < window:
            rci_values.append(np.nan)
            continue
        rci = calc_rci(df.iloc[i-window:i, :], column, window)
        rci_values.append(rci)
    df[f'RCI_{suffix}'] = rci_values

# 日単位
add_rci(df, 'Close', 9, '9_day')
add_rci(df, 'Close', 26, '26_day')

# 週単位（週5日のマーケットを仮定）
add_rci(df, 'Close', 9 * 5, '9_week')
add_rci(df, 'Close', 26 * 5, '26_week')

# 月単位（月平均21取引日を仮定）
add_rci(df, 'Close', 9 * 21, '9_month')
add_rci(df, 'Close', 26 * 21, '26_month')

# ATRを追加
df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

# CMFを追加
df['CMF'] = ta.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)

# ROCを追加
df['ROC_10'] = ta.ROC(df['Close'], timeperiod=10)
df['ROC_14'] = ta.ROC(df['Close'], timeperiod=14)
df['ROC_25'] = ta.ROC(df['Close'], timeperiod=25)

# CCIを追加
df['CCI'] = ta.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)

# PASR（パラボリック）を追加
df['PSAR'] = ta.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)

# OBVを追加
df['OBV'] = ta.OBV(df['Close'], df['Volume'])

#ラグ変数を追加
"""df['Lag_1year_Open'] = df['Open'].shift(200)
df['Lag_1year_High'] = df['High'].shift(200)
df['Lag_1year_Low'] = df['Low'].shift(200)
df['Lag_1year_Close'] = df['Close'].shift(200)
df['Lag_1year_Volume'] = df['Volume'].shift(200)
df['Lag_1year_Body'] = df['Body'].shift(200)

df['Lag_2year_Open'] = df['Open'].shift(200*2)
df['Lag_2year_High'] = df['High'].shift(200*2)
df['Lag_2year_Low'] = df['Low'].shift(200*2)
df['Lag_2year_Close'] = df['Close'].shift(200*2)
df['Lag_2year_Volume'] = df['Volume'].shift(200*2)
df['Lag_2year_Body'] = df['Body'].shift(200*2)

df['Lag_3year_Open'] = df['Open'].shift(200*3)
df['Lag_3year_High'] = df['High'].shift(200*3)
df['Lag_3year_Low'] = df['Low'].shift(200*3)
df['Lag_3year_Close'] = df['Close'].shift(200*3)
df['Lag_3year_Volume'] = df['Volume'].shift(200*3)
df['Lag_3year_Body'] = df['Body'].shift(200*3)

df['Lag_4year_Open'] = df['Open'].shift(200*4)
df['Lag_4year_High'] = df['High'].shift(200*4)
df['Lag_4year_Low'] = df['Low'].shift(200*4)
df['Lag_4year_Close'] = df['Close'].shift(200*4)
df['Lag_4year_Volume'] = df['Volume'].shift(200*4)
df['Lag_4year_Body'] = df['Body'].shift(200*4)

df['Lag_5year_Open'] = df['Open'].shift(200*5)
df['Lag_5year_High'] = df['High'].shift(200*5)
df['Lag_5year_Low'] = df['Low'].shift(200*5)
df['Lag_5year_Close'] = df['Close'].shift(200*5)
df['Lag_5year_Volume'] = df['Volume'].shift(200*5)
df['Lag_5year_Body'] = df['Body'].shift(200*5)"""
