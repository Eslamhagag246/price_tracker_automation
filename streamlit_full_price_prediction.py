import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import os

from supabase import create_client

SUPABASE_URL = "https://ryiqzurrmvaftbnpiopx.supabase.co"
SUPABASE_KEY = "sb_publishable_b7_MsHFtjFmSRbS5Ek_y6w_9QaWtM4x"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_data(ttl=600)
def load_data_from_db(device_type):
     try:

        if device_type == "Tablets":
            table = "tablets_prices"
        else:
            table = "mobile_prices"

        response = supabase.table(table).select("*").execute()

        if response.data:
            df = pd.DataFrame(response.data)

            # Convert timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Create date column expected by model
            df["date"] = df["timestamp"].dt.date
            df["date"] = pd.to_datetime(df["date"])

            # Clean price column
            df["price"] = df["price"].astype(str)
            df["price"] = df["price"].str.replace("EGP", "", regex=False)
            df["price"] = df["price"].str.replace(",", "", regex=False)
            df["price"] = pd.to_numeric(df["price"], errors="coerce")

            # Product key
            df["product_key"] = (
                df["name"].str.lower().str.strip() + " " +
                df["website"].str.lower() + " " +
                df["ram_gb"].astype(str) + " " +
                df["storage_gb"].astype(str)
            )

            return df

        return None

    except Exception as e:
        st.sidebar.warning(f"Database connection failed: {str(e)}")
        return None

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Price Tracker - Tablets & Mobiles",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
# IMPORT MODELS
# ═══════════════════════════════════════════════════════════
MODELS_LOADED = {'tablet': False, 'mobile': False}
tablet_model = None
mobile_model = None

try:
    from tablet_model_newVersion import (
        load_and_preprocess_data as load_tablet_data_func,
        forecast_product as forecast_tablet_func,
        load_global_model as load_tablet_model
    )
    # Try to load pre-trained model
    try:
        tablet_model = load_tablet_model()
        MODELS_LOADED['tablet'] = True
    except:
        st.sidebar.warning("⚠️ Tablet model not trained yet")
except ImportError as e:
    st.error(f"❌ Error importing tablet_model_newVersion.py: {str(e)}")

try:
    from mobile_model_newVersion import (
        load_and_preprocess_data as load_mobile_data_func,
        forecast_product as forecast_mobile_func,
        load_global_model as load_mobile_model
    )
    # Try to load pre-trained model
    try:
        mobile_model = load_mobile_model()
        MODELS_LOADED['mobile'] = True
    except:
        st.sidebar.warning("⚠️ Mobile model not trained yet")
except ImportError as e:
    st.error(f"❌ Error importing mobile_model_newVersion.py: {str(e)}")

# ═══════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }

.main .block-container {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}

h1 {
    color: #667eea;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

h2, h3 {
    color: #4a5568;
    font-weight: 600;
}

.stSelectbox label, .stMultiSelect label {
    font-weight: 600;
    color: #2d3748;
}

div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
    color: #667eea;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 2rem;
    font-weight: 600;
    transition: all 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.device-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    margin: 0.2rem;
}

.badge-tablet {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.badge-mobile {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
}

.stat-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    margin: 0.5rem 0;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}

.stat-label {
    font-size: 0.85rem;
    opacity: 0.9;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_data(device_type):
    """Load data based on device type"""
    if device_type == "Tablets":
        filepath = 'tablets_cleaned_continuous.csv'
        load_func = load_tablet_data_func
    else:
        filepath = 'mobile_cleaned_70K.csv'
        load_func = load_mobile_data_func
    
    try:
        df = load_func(filepath)
        return df, filepath
    except FileNotFoundError:
        st.error(f"❌ File not found: {filepath}")
        return None, filepath
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, filepath


def create_forecast_chart(result, device_type):
    """Create beautiful forecast chart"""
    pdf = result['pdf']
    forecast_dates = result['forecast_dates']
    forecast_prices = result['forecast_prices']
    mae = result['mae']
    
    # Colors
    if device_type == "Tablets":
        color_main = '#667eea'
        color_forecast = '#f093fb'
    else:
        color_main = '#f5576c'
        color_forecast = '#feca57'
    
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=pdf['date'],
        y=pdf['price'],
        mode='lines+markers',
        name='Historical Price',
        line=dict(color=color_main, width=3),
        marker=dict(size=6, color=color_main),
        hovertemplate='<b>%{x}</b><br>EGP %{y:,.0f}<extra></extra>'
    ))
    
    # Rolling average
    if 'rolling_avg_7' in pdf.columns:
        fig.add_trace(go.Scatter(
            x=pdf['date'],
            y=pdf['rolling_avg_7'],
            mode='lines',
            name='7-Day Average',
            line=dict(color=color_main, width=2, dash='dot'),
            opacity=0.6,
            hovertemplate='<b>%{x}</b><br>Avg: EGP %{y:,.0f}<extra></extra>'
        ))
    
    # Connection to forecast
    last_hist_date = pdf['date'].iloc[-1]
    last_hist_price = pdf['price'].iloc[-1]
    
    fig.add_trace(go.Scatter(
        x=[last_hist_date, forecast_dates[0]],
        y=[last_hist_price, forecast_prices[0]],
        mode='lines',
        line=dict(color='gray', width=2, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_prices,
        mode='lines+markers',
        name='7-Day Forecast',
        line=dict(color=color_forecast, width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond', color=color_forecast),
        hovertemplate='<b>%{x}</b><br>Forecast: EGP %{y:,.0f}<extra></extra>'
    ))
    
    # Confidence band
    upper = [p + mae for p in forecast_prices]
    lower = [max(0, p - mae) for p in forecast_prices]
    
    fig.add_trace(go.Scatter(
        x=forecast_dates + forecast_dates[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor=f'rgba(240, 147, 251, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Today marker
    today = pd.Timestamp.today().normalize()
    today_str = today.strftime('%Y-%m-%d')
    
    fig.add_shape(
        type="line",
        x0=today_str, x1=today_str,
        y0=0, y1=1,
        yref='paper',
        line=dict(color="gray", width=2, dash="dot")
    )
    
    fig.add_annotation(
        x=today_str, y=1,
        yref='paper',
        text="Today",
        showarrow=False,
        font=dict(color="gray", size=11),
        yshift=10
    )
    
    # Layout
    fig.update_layout(
        title={
            'text': '📈 Price History & 7-Day Forecast',
            'font': {'size': 20, 'color': '#2d3748', 'family': 'Inter'}
        },
        xaxis_title='Date',
        yaxis_title='Price (EGP)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter', size=12, color='#4a5568'),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e2e8f0',
            borderwidth=1
        ),
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f7fafc')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f7fafc')
    
    return fig


# ═══════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════

st.title("📱 Price Tracker Pro")
st.markdown("**Track & Forecast Prices for Tablets & Mobile Phones**")
st.markdown("---")

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎯 Select Device Type")
    
    device_type = st.radio(
        "Choose category:",
        options=["Tablets", "Mobile Phones"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Check if model is loaded
    model_key = 'tablet' if device_type == "Tablets" else 'mobile'
    
    if MODELS_LOADED[model_key]:
        st.success(f"✅ {device_type} model loaded")
    else:
        st.error(f"❌ {device_type} model not found")
        st.info("Please train the model first by running the model file")
        st.code(f"python {'tablet' if model_key == 'tablet' else 'mobile'}_model_newVersion.py")
        st.stop()
    
    # Load data
    df, filepath = load_data(device_type)
             
    if df is not None:
        st.markdown("### 📊 Dataset Info")
        st.metric("Total Products", f"{df['product_key'].nunique():,}")
        st.metric("Data Points", f"{len(df):,}")
        
        last_update = df['date'].max()
        st.markdown(f"**Last Updated:** {last_update.strftime('%b %d, %Y')}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(f"""
    Currently showing: **{device_type}**
    
    Data file: `{filepath}`
    
    Model: **Global Linear Regression**
    """)

if df is None:
    st.stop()

# ═══════════════════════════════════════════════════════════
# 📊 MARKET INSIGHTS (PRICE MOVEMENTS)
# ═══════════════════════════════════════════════════════════

st.markdown("## 📊 Market Insights")

def generate_market_insights(df):

    insights_df = df.copy()

    # Sort by product & date
    insights_df = insights_df.sort_values(['product_key', 'date'])

    # Get last 2 prices for each product
    insights_df['prev_price'] = insights_df.groupby('product_key')['price'].shift(1)

    # Price change
    insights_df['price_change'] = insights_df['price'] - insights_df['prev_price']
    insights_df['pct_change'] = (insights_df['price_change'] / insights_df['prev_price']) * 100

    # Keep only latest record per product
    latest = insights_df.groupby('product_key').tail(1)

    latest = latest.dropna(subset=['pct_change'])

    # Top movers
    top_increase = latest.sort_values('pct_change', ascending=False).head(5)
    top_decrease = latest.sort_values('pct_change', ascending=True).head(5)

    return top_increase, top_decrease


if df is not None and len(df) > 0:

    top_up, top_down = generate_market_insights(df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Top Price Increases")
        if len(top_up) > 0:
            st.dataframe(
                top_up[['name', 'brand', 'price', 'pct_change']]
                .rename(columns={
                    'name': 'Product',
                    'brand': 'Brand',
                    'price': 'Current Price',
                    'pct_change': '% Change'
                }),
                use_container_width=True
            )
        else:
            st.info("No increase data available")

    with col2:
        st.markdown("### 📉 Top Price Drops")
        if len(top_down) > 0:
            st.dataframe(
                top_down[['name', 'brand', 'price', 'pct_change']]
                .rename(columns={
                    'name': 'Product',
                    'brand': 'Brand',
                    'price': 'Current Price',
                    'pct_change': '% Change'
                }),
                use_container_width=True
            )
        else:
            st.info("No decrease data available")
# ═══════════════════════════════════════════════════════════
# FILTERS
# ═══════════════════════════════════════════════════════════

st.markdown("### 🔍 Search & Filter Products")

search_term = st.text_input(
    "🔎 Search by product name",
    placeholder="e.g., iPad, Galaxy, iPhone...",
    help="Search for products by name"
)

if search_term:
    filtered_df = df[df['name'].str.contains(search_term, case=False, na=False)]
else:
    filtered_df = df.copy()

# Filters
col1, col2, col3, col4 = st.columns(4)

with col1:
    brands = sorted(filtered_df['brand'].unique())
    selected_brands = st.multiselect("🏷️ Brand", brands, default=[])

with col2:
    websites = sorted(filtered_df['website'].unique())
    selected_websites = st.multiselect("🛒 Website", websites, default=[])

with col3:
    rams = sorted(filtered_df['ram_gb'].unique())
    selected_rams = st.multiselect("💾 RAM (GB)", rams, default=[])

with col4:
    storages = sorted(filtered_df['storage_gb'].unique())
    selected_storages = st.multiselect("💿 Storage (GB)", storages, default=[])

# Apply filters
if selected_brands:
    filtered_df = filtered_df[filtered_df['brand'].isin(selected_brands)]
if selected_websites:
    filtered_df = filtered_df[filtered_df['website'].isin(selected_websites)]
if selected_rams:
    filtered_df = filtered_df[filtered_df['ram_gb'].isin(selected_rams)]
if selected_storages:
    filtered_df = filtered_df[filtered_df['storage_gb'].isin(selected_storages)]

# Active filters
active_filters = []
if search_term:
    active_filters.append(f"Search: {search_term}")
if selected_brands:
    active_filters.append(f"Brands: {', '.join(selected_brands)}")
if selected_websites:
    active_filters.append(f"Websites: {', '.join(selected_websites)}")
if selected_rams:
    active_filters.append(f"RAM: {', '.join(map(str, selected_rams))}GB")
if selected_storages:
    active_filters.append(f"Storage: {', '.join(map(str, selected_storages))}GB")

if active_filters:
    badge_class = 'badge-tablet' if device_type == "Tablets" else 'badge-mobile'
    badges_html = ''.join([f'<span class="device-badge {badge_class}">{f}</span>' for f in active_filters])
    st.markdown(f"**Active Filters:** {badges_html}", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════
# PRODUCT SELECTION
# ═══════════════════════════════════════════════════════════

if filtered_df.empty:
    st.warning("⚠️ No products found. Try different filters.")
    st.stop()

product_groups = filtered_df.groupby('product_key').agg({
    'name': 'first',
    'brand': 'first',
    'website': 'first',
    'ram_gb': 'first',
    'storage_gb': 'first',
    'price': 'count'
}).reset_index()

product_groups.columns = ['product_key', 'name', 'brand', 'website', 'ram_gb', 'storage_gb', 'n_obs']
product_groups = product_groups.sort_values('n_obs', ascending=False)

st.markdown(f"**Found {len(product_groups)} products**")

selected_product = st.selectbox(
    f"📱 Select a {device_type[:-1].lower()}",
    options=product_groups['product_key'].tolist(),
    format_func=lambda x: (
        f"{product_groups[product_groups['product_key']==x]['name'].values[0]} | "
        f"{product_groups[product_groups['product_key']==x]['ram_gb'].values[0]}GB + "
        f"{product_groups[product_groups['product_key']==x]['storage_gb'].values[0]}GB | "
        f"{product_groups[product_groups['product_key']==x]['website'].values[0].upper()} | "
        f"({product_groups[product_groups['product_key']==x]['n_obs'].values[0]} observations)"
    ),
    help="Select a product to see price forecast"
)

# ═══════════════════════════════════════════════════════════
# FORECAST
# ═══════════════════════════════════════════════════════════

st.markdown("---")

product_df = df[df['product_key'] == selected_product].copy()
product_info = product_groups[product_groups['product_key'] == selected_product].iloc[0]

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"## 📱 {product_info['name']}")
with col2:
    badge_class = 'badge-tablet' if device_type == "Tablets" else 'badge-mobile'
    st.markdown(f'<span class="device-badge {badge_class}">{device_type[:-1]}</span>', unsafe_allow_html=True)

# Specs
spec_col1, spec_col2, spec_col3, spec_col4 = st.columns(4)
spec_col1.metric("🏷️ Brand", product_info['brand'].title())
spec_col2.metric("💾 RAM", f"{product_info['ram_gb']}GB")
spec_col3.metric("💿 Storage", f"{product_info['storage_gb']}GB")
spec_col4.metric("🛒 Website", product_info['website'].upper())

st.markdown("---")

# Generate forecast
with st.spinner("🤖 Generating AI forecast..."):
    try:
        # Use appropriate model and function
        if device_type == "Tablets":
            result = forecast_tablet_func(product_df, days_ahead=7, model=tablet_model)
        else:
            result = forecast_mobile_func(product_df, days_ahead=7, model=mobile_model)
    except Exception as e:
        st.error(f"❌ Error generating forecast: {str(e)}")
        st.code(str(e))
        st.stop()

# Stats
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Current Price</div>
        <div class="stat-value">EGP {result['last_price']:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with stat_col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">7-Day Forecast</div>
        <div class="stat-value">EGP {result['forecast_prices'][-1]:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with stat_col3:
    change = result['forecast_prices'][-1] - result['last_price']
    change_pct = (change / result['last_price']) * 100
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Expected Change</div>
        <div class="stat-value">{change:+,.0f} EGP</div>
        <div style="font-size:0.9rem; margin-top:0.3rem;">({change_pct:+.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with stat_col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Confidence</div>
        <div class="stat-value">{result['confidence']}</div>
        <div style="font-size:0.9rem; margin-top:0.3rem;">({result['n_obs']} days tracked)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Chart
st.plotly_chart(create_forecast_chart(result, device_type), use_container_width=True)

# Forecast table
st.markdown("### 📅 7-Day Forecast Breakdown")

forecast_table = pd.DataFrame({
    'Date': [d.strftime('%A, %B %d') for d in result['forecast_dates']],
    'Forecasted Price': [f"EGP {p:,.0f}" for p in result['forecast_prices']],
    'Lower Bound': [f"EGP {max(0, p - result['mae']):,.0f}" for p in result['forecast_prices']],
    'Upper Bound': [f"EGP {(p + result['mae']):,.0f}" for p in result['forecast_prices']]
})

st.dataframe(forecast_table, use_container_width=True, hide_index=True)

# Stats
st.markdown("---")
st.markdown("### 📊 Price Statistics")

stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
stats_col1.metric("📉 Minimum Price", f"EGP {result['min_price']:,.0f}")
stats_col2.metric("📊 Average Price", f"EGP {result['avg_price']:,.0f}")
stats_col3.metric("📈 Maximum Price", f"EGP {result['max_price']:,.0f}")
stats_col4.metric("🎯 Model Accuracy (MAE)", f"±{result['mae']:,.0f} EGP")

# URL
if 'URL' in product_df.columns:
    url = product_df['URL'].iloc[-1]
    if url and str(url) != 'nan':
        st.markdown(f"[🔗 View on {product_info['website'].upper()}]({url})")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; font-size: 0.9rem; padding: 1rem;'>
    <p>📱 Price Tracker Pro - Powered by Global Linear Regression</p>
    <p>One model trained on ALL products for better generalization</p>
</div>
""", unsafe_allow_html=True)
