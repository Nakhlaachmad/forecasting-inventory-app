import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 1. Konfigurasi Halaman & Tema Monochrome via CSS
st.set_page_config(page_title="Demand Forecast - Mono", layout="wide")

st.markdown("""
    <style>
    /* Mengubah latar belakang dan teks menjadi monokrom */
    .stApp { background-color: #ffffff; }
    .stMetric { 
        background-color: #f8f9fa; 
        border: 1px solid #dee2e6; 
        border-radius: 5px; 
    }
    h1, h2, h3 { color: #212529; }
    /* Tombol dan Sidebar */
    .stButton>button { 
        background-color: #212529; 
        color: white; 
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Fungsi Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('DATASET 2018.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    return df

df_raw = load_data()

# 3. SIDEBAR
st.sidebar.title("🌑 Dashboard Control")
categories = sorted(df_raw['Category'].unique())
selected_category = st.sidebar.selectbox("Kategori", ["Semua"] + categories)

sub_cats = sorted(df_raw['Sub-Category'].unique()) if selected_category == "Semua" \
           else sorted(df_raw[df_raw['Category'] == selected_category]['Sub-Category'].unique())

selected_subs = st.sidebar.multiselect("Sub-Kategori", options=sub_cats, default=sub_cats[:2])

forecast_period = st.sidebar.select_slider("Periode Prediksi (Hari)", options=[30, 60, 90, 180], value=90)
safety_stock_ratio = st.sidebar.slider("Safety Stock (%)", 0, 50, 20) / 100

# 4. PEMROSESAN & MODELING
df_filtered = df_raw.copy()
if selected_category != "Semua":
    df_filtered = df_filtered[df_filtered['Category'] == selected_category]
if selected_subs:
    df_filtered = df_filtered[df_filtered['Sub-Category'].isin(selected_subs)]

df_prophet = df_filtered.groupby('Order Date')['Sales'].sum().reset_index()
df_prophet.columns = ['ds', 'y']

m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m.fit(df_prophet)
future = m.make_future_dataframe(periods=forecast_period)
forecast = m.predict(future)

# 5. UI UTAMA
st.title("🔘 Demand & Inventory Planner")
st.caption("Analisis Prediksi Berbasis Data Historis 2018")

# Metrics (Black & White style)
m1, m2, m3 = st.columns(3)
m1.metric("Total Sales", f"${df_filtered['Sales'].sum():,.0f}")
m2.metric("Avg Daily", f"${df_prophet['y'].mean():,.2f}")
m3.metric("Safety Stock", f"{int(safety_stock_ratio*100)}%")

tab1, tab2, tab3 = st.tabs(["📊 Forecast Trend", "🗓️ Seasonality", "📋 Stock List"])

with tab1:
    st.subheader("Tren Prediksi (Monochrome)")
    fig = go.Figure()
    # Confidence Interval (Abu-abu sangat muda)
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(200,200,200,0.2)', name='Confidence Range'))
    
    # Data Aktual (Hitam)
    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name="Aktual", line=dict(color='#000000', width=1.5)))
    
    # Prediksi (Abu-abu Tua Putus-putus)
    fig.add_trace(go.Scatter(x=forecast['ds'].tail(forecast_period), y=forecast['yhat'].tail(forecast_period), 
                             name="Prediksi", line=dict(color='#6c757d', width=3, dash='dot')))
    
    fig.update_layout(template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        # Weekly Pattern (Grayscale)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_weekly = forecast.copy()
        df_weekly['Day'] = df_weekly['ds'].dt.day_name()
        weekly_pattern = df_weekly.groupby('Day')['weekly'].mean().reindex(day_order).reset_index()
        
        fig_week = px.line(weekly_pattern, x='Day', y='weekly', title="Pola Mingguan", markers=True, color_discrete_sequence=['#343a40'])
        fig_week.update_layout(template="plotly_white")
        st.plotly_chart(fig_week, use_container_width=True)

    with c2:
        # Yearly Pattern (Grayscale Bar)
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        df_yearly = forecast.copy()
        df_yearly['Month'] = df_yearly['ds'].dt.month_name()
        yearly_pattern = df_yearly.groupby('Month')['yearly'].mean().reindex(month_order).reset_index()
        
        fig_month = px.bar(yearly_pattern, x='Month', y='yearly', title="Pola Bulanan", color_discrete_sequence=['#adb5bd'])
        fig_month.update_layout(template="plotly_white")
        st.plotly_chart(fig_month, use_container_width=True)

with tab3:
    st.subheader("Rekomendasi Inventori")
    df_future = forecast[forecast['ds'] > df_raw['Order Date'].max()].copy()
    df_future['Bulan'] = df_future['ds'].dt.strftime('%Y-%m')
    
    inv_table = df_future.groupby('Bulan').agg({'yhat': 'sum'}).reset_index()
    inv_table.columns = ['Bulan', 'Estimasi Permintaan']
    inv_table['Safety Stock'] = inv_table['Estimasi Permintaan'] * safety_stock_ratio
    inv_table['Stok Ideal'] = inv_table['Estimasi Permintaan'] + inv_table['Safety Stock']
    
    # Styling Tabel Monochrome
    st.table(inv_table.style.format(precision=2))

st.divider()
st.caption("Data Intelligence Dashboard | Minimalist Edition")