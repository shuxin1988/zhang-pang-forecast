import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
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

# ---- 侧边栏：日期范围选择 ----
st.sidebar.header("📅 数据范围设置")
# 默认结束日期为今天，开始日期为一年前
default_end = datetime.now().date()
default_start = default_end - timedelta(days=365)

date_range = st.sidebar.date_input(
    "选择起止日期",
    value=(default_start, default_end),
    min_value=datetime(2005, 1, 1).date(),   # 最早可查数据约在2005年
    max_value=default_end,
    help="选择数据的历史范围，预测将基于此范围重新计算"
)

# 处理用户输入的日期范围
if len(date_range) == 2:
    start_date_user, end_date_user = date_range
    start_date_user = datetime.combine(start_date_user, datetime.min.time())
    end_date_user = datetime.combine(end_date_user, datetime.min.time())
else:
    # 如果未选择完整范围，使用默认值
    start_date_user = datetime.combine(default_start, datetime.min.time())
    end_date_user = datetime.combine(default_end, datetime.min.time())

# 确保结束日期不晚于今天
if end_date_user.date() > datetime.now().date():
    end_date_user = datetime.now()
    st.sidebar.warning("结束日期不能超过今天，已自动调整为当前日期。")

# 确保开始日期不晚于结束日期
if start_date_user >= end_date_user:
    start_date_user = end_date_user - timedelta(days=365)
    st.sidebar.warning("起始日期不能晚于结束日期，已自动调整为结束日期前一年。")

# 计算用于显示的天数
days_selected = (end_date_user - start_date_user).days

# ---- 缓存数据获取（根据用户选择的日期范围） ----
@st.cache_data(ttl=3600)
def load_data(currency, start_date, end_date):
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    df = ak.currency_boc_sina(symbol=currency, start_date=start_str, end_date=end_str)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={'日期': 'date', '央行中间价': 'central_parity'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df[['date', 'central_parity']]

# ---- 数据平滑处理（插值填补非交易日） ----
def smooth_curve(df):
    if df.empty or len(df) < 2:
        return df
    full_date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    df_full = pd.DataFrame({'date': full_date_range})
    df_merged = df_full.merge(df, on='date', how='left')
    df_merged['central_parity'] = df_merged['central_parity'].interpolate(method='linear')
    return df_merged

# ---- 未来预测（Holt-Winters 指数平滑） ----
def forecast_future(df, periods=30):
    if len(df) < 30:
        return pd.DataFrame()
    df_smooth = smooth_curve(df)
    series = df_smooth.set_index('date')['central_parity']
    model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal=None,
        initialization_method='estimated'
    )
    fit = model.fit()
    forecast = fit.forecast(periods)
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'central_parity': forecast.values,
        'type': 'forecast'
    })
    df_smooth['type'] = 'historical'
    combined = pd.concat([df_smooth, forecast_df], ignore_index=True)
    return combined

# ---- 加载数据 ----
with st.spinner(f"正在获取 {CURRENCY_DISPLAY} {start_date_user.strftime('%Y-%m-%d')} 至 {end_date_user.strftime('%Y-%m-%d')} 的中间价数据..."):
    df_raw = load_data(CURRENCY, start_date_user, end_date_user)

if df_raw.empty:
    st.error("❌ 所选日期范围内无数据，请调整起止日期后重试。")
else:
    # ---- 生成预测数据 ----
    combined_df = forecast_future(df_raw, periods=30)

    # ---- 显示最新中间价（基于所选范围的最后一天） ----
    latest_historical = df_raw.iloc[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📅 最新日期", latest_historical['date'].strftime("%Y-%m-%d"))
    with col2:
        st.metric("💵 最新中间价", f"{latest_historical['central_parity']:.4f}")
    with col3:
        # 计算所选范围内的变动
        first = df_raw.iloc[0]['central_parity']
        change = latest_historical['central_parity'] - first
        change_pct = (change / first) * 100
        st.metric("📈 区间变动", f"{change:+.4f}", f"{change_pct:+.2f}%")

    # ---- 绘制曲线（历史平滑线 + 预测虚线） ----
    fig = go.Figure()

    hist_data = combined_df[combined_df['type'] == 'historical']
    fig.add_trace(go.Scatter(
        x=hist_data['date'],
        y=hist_data['central_parity'],
        mode='lines',
        name='历史中间价 (平滑)',
        line=dict(color='#1f77b4', width=2)
    ))

    forecast_data = combined_df[combined_df['type'] == 'forecast']
    if not forecast_data.empty:
        fig.add_trace(go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['central_parity'],
            mode='lines',
            name='未来30天预测',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))

        # 预测置信区间（上下0.5%波动）
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
        title=f"USD/CNY 中间价走势及未来30天预测（数据范围：{start_date_user.strftime('%Y-%m-%d')} 至 {end_date_user.strftime('%Y-%m-%d')}）",
        xaxis_title="日期",
        yaxis_title="中间价",
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, width='stretch')

    # ---- 预测数据表格 ----
    with st.expander("📋 查看未来30天预测数据"):
        st.dataframe(
            forecast_data[['date', 'central_parity']].rename(columns={'central_parity': '预测中间价'}),
            width='stretch',
            hide_index=True
        )

    # ---- 原始数据表格 ----
    with st.expander("📋 查看原始历史数据"):
        st.dataframe(
            df_raw.sort_values('date', ascending=False),
            width='stretch',
            hide_index=True
        )

    # ---- 下载按钮 ----
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
    "- 历史数据来自中国外汇交易中心，通过 AKShare 获取。  \n"
    "- 非交易日数据已通过线性插值平滑处理，曲线连续。  \n"
    "- 预测采用 **Holt-Winters 指数平滑模型**，基于所选历史范围外推，置信度参考 ±0.5%。  \n"
    "- 预测仅供张徇齐分析使用参考。"
)
