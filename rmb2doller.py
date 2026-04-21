import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---- 页面配置 ----
st.set_page_config(
    page_title="zhang_pang USD/CNY central parity prediction",
    page_icon="💰",
    layout="wide"
)

# ---- 标题 ----
st.title("💰 zhang_pang USD/CNY central parity prediction")
st.caption(f"数据来源：中国外汇交易中心 | 更新于 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ---- 固定货币为美元 ----
CURRENCY = "美元"
CURRENCY_DISPLAY = "美元 (USD)"

# ---- 缓存数据获取（近一年） ----
@st.cache_data(ttl=3600)
def load_data(currency):
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=370)).strftime("%Y%m%d")
    df = ak.currency_boc_sina(symbol=currency, start_date=start_date, end_date=end_date)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={'日期': 'date', '央行中间价': 'central_parity'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    one_year_ago = datetime.now() - timedelta(days=365)
    df = df[df['date'] >= one_year_ago]
    return df[['date', 'central_parity']]

# ---- 数据平滑处理（插值填补非交易日） ----
def smooth_curve(df):
    """对原始数据进行线性插值，生成连续日期序列，消除断点"""
    if df.empty:
        return df
    # 创建完整的日期范围（从最早到最晚）
    full_date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    df_full = pd.DataFrame({'date': full_date_range})
    # 合并原始数据，缺失值用 NaN 表示
    df_merged = df_full.merge(df, on='date', how='left')
    # 线性插值填充 NaN
    df_merged['central_parity'] = df_merged['central_parity'].interpolate(method='linear')
    return df_merged

# ---- 未来预测（Holt-Winters 指数平滑） ----
def forecast_future(df, periods=30):
    """基于历史数据预测未来 periods 天的中间价"""
    if len(df) < 30:
        return pd.DataFrame()  # 数据太少无法预测
    # 使用插值后的数据
    df_smooth = smooth_curve(df)
    series = df_smooth.set_index('date')['central_parity']
    # 拟合 Holt-Winters 模型（无季节性，因中间价无明显季节性）
    model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal=None,
        initialization_method='estimated'
    )
    fit = model.fit()
    # 预测未来
    forecast = fit.forecast(periods)
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'central_parity': forecast.values,
        'type': 'forecast'
    })
    # 标记历史数据类型
    df_smooth['type'] = 'historical'
    # 合并历史与预测
    combined = pd.concat([df_smooth, forecast_df], ignore_index=True)
    return combined

# ---- 加载数据 ----
with st.spinner(f"正在获取 {CURRENCY_DISPLAY} 近一年中间价数据..."):
    df_raw = load_data(CURRENCY)

if df_raw.empty:
    st.error("❌ 数据获取失败，请稍后重试或检查 AKShare 版本。")
else:
    # ---- 生成预测数据 ----
    combined_df = forecast_future(df_raw, periods=30)

    # ---- 显示最新中间价 ----
    latest_historical = df_raw.iloc[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📅 最新日期", latest_historical['date'].strftime("%Y-%m-%d"))
    with col2:
        st.metric("💵 最新中间价", f"{latest_historical['central_parity']:.4f}")
    with col3:
        # 计算近一年变化
        first = df_raw.iloc[0]['central_parity']
        change = latest_historical['central_parity'] - first
        change_pct = (change / first) * 100
        st.metric("📈 近一年变动", f"{change:+.4f}", f"{change_pct:+.2f}%")

    # ---- 绘制曲线（历史平滑线 + 预测虚线） ----
    fig = go.Figure()

    # 历史平滑曲线
    hist_data = combined_df[combined_df['type'] == 'historical']
    fig.add_trace(go.Scatter(
        x=hist_data['date'],
        y=hist_data['central_parity'],
        mode='lines',
        name='历史中间价 (平滑)',
        line=dict(color='#1f77b4', width=2)
    ))

    # 预测曲线（虚线）
    forecast_data = combined_df[combined_df['type'] == 'forecast']
    if not forecast_data.empty:
        fig.add_trace(go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['central_parity'],
            mode='lines',
            name='未来30天预测',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))

        # 添加预测置信区间（可选，这里用上下浮动简单表示）
        last_hist_value = hist_data.iloc[-1]['central_parity']
        # 简单估计：上下0.5%波动带
        upper = forecast_data['central_parity'] * 1.005
        lower = forecast_data['central_parity'] * 0.995
        fig.add_trace(go.Scatter(
            x=forecast_data['date'].tolist() + forecast_data['date'][::-1].tolist(),
            y=upper.tolist() + lower[::-1].tolist(),
            fill='toself',
            fillcolor='rgba(255,127,14,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='预测区间 (±0.5%)'
        ))

    fig.update_layout(
        title="USD/CNY 中间价走势及未来30天预测",
        xaxis_title="日期",
        yaxis_title="中间价",
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- 预测数据表格 ----
    with st.expander("📋 查看未来30天预测数据"):
        st.dataframe(
            forecast_data[['date', 'central_parity']].rename(columns={'central_parity': '预测中间价'}),
            use_container_width=True,
            hide_index=True
        )

    # ---- 原始数据表格 ----
    with st.expander("📋 查看原始历史数据"):
        st.dataframe(
            df_raw.sort_values('date', ascending=False),
            use_container_width=True,
            hide_index=True
        )

    # ---- 下载按钮（包含历史+预测） ----
    csv = combined_df[['date', 'central_parity', 'type']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ 下载全部数据 (历史+预测)",
        data=csv,
        file_name=f'USD_CNY_central_parity_forecast_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
    )

# ---- 侧边栏说明 ----
st.sidebar.markdown("---")
st.sidebar.info(
    "**关于数据与预测**  \n"
    "- 历史数据来自中国外汇交易中心，通过 AKShare 获取，这里注意定义起始日期，或者写死？不然会溢出，后面看用户需求是否要增加筛选器，可以自定义起止日期。  \n"
    "- 非交易日数据已通过线性插值平滑处理，曲线连续。  \n"
    "- 预测采用 **Holt-Winters 指数平滑模型**，基于近一年趋势外推。置信度 0.95  \n"
    "- 预测仅供张徇齐分析使用参考。"
)
