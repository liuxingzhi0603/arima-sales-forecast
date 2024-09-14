import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 准备你提供的数据
data = {
    'Date': ['2011/1', '2011/2', '2011/3', '2011/4', '2012/1', '2012/2', '2012/3', '2012/4', 
             '2013/1', '2013/2', '2013/3', '2013/4', '2014/1', '2014/2', '2014/3', '2014/4', 
             '2015/1', '2015/2', '2015/3', '2015/4', '2016/1', '2016/2', '2016/3', '2016/4'],
    'Sales': [7, 6, 4, 6, 8, 7, 6, 8, 9, 8, 6, 9, 9, 10, 9, 12, 13, 12, 10, 11, np.nan, np.nan, np.nan, np.nan],
    'Trend': [5.073, 5.446, 5.819, 6.192, 6.565, 6.931, 7.311, 7.684, 8.057, 8.430, 8.803, 9.176, 
              9.549, 9.922, 10.295, 10.668, 11.041, 11.414, 11.787, 12.160, 12.533, 12.906, 13.297, 13.652]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Streamlit 应用程序
st.title('销售数据的 ARIMA 模型预测')

# 创建滑块来调整 ARIMA 模型的参数
p = st.slider('ARIMA(p)', 0, 5, value=1)
d = st.slider('ARIMA(d)', 0, 2, value=1)
q = st.slider('ARIMA(q)', 0, 5, value=1)

# 填充缺失值，以进行模型拟合
df['Sales'] = df['Sales'].fillna(method='ffill')  # 这里使用前向填充进行缺失值填充

# 构建 ARIMA 模型
model = ARIMA(df['Sales'], order=(p, d, q))
model_fit = model.fit()

# 预测未来4个周期
forecast_steps = 4
forecast = model_fit.forecast(steps=forecast_steps)

# 创建预测数据框
forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=3), periods=forecast_steps, freq='Q')
forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)

# 绘制结果
st.subheader('销售预测')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Sales'], label='历史销售', marker='o')
ax.plot(forecast_df.index, forecast_df['Forecast'], label='预测销售', marker='o', linestyle='--', color='red')
ax.set_title(f'ARIMA({p}, {d}, {q}) 销售预测')
ax.set_xlabel('日期')
ax.set_ylabel('销售量')
ax.legend()
ax.grid(True)
st.pyplot(fig)
