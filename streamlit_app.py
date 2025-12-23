import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Pagina configuratie
st.set_page_config(
    page_title="Walmart Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titel
st.title("📊 Walmart Sales Dashboard")
st.markdown("Interactief dashboard voor Walmart verkoopsgegevens")

# Data laden
@st.cache_data
def load_data():
    df = pd.read_csv('Walmart.csv')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    return df

df = load_data()

# Sidebar voor filters
st.sidebar.header("🔍 Filters")

# Date filter
date_range = st.sidebar.date_input(
    "Selecteer datumrange",
    value=(df['transaction_date'].min().date(), df['transaction_date'].max().date()),
    min_value=df['transaction_date'].min().date(),
    max_value=df['transaction_date'].max().date()
)

# Category filter
categories = st.sidebar.multiselect(
    "Selecteer categorie(s)",
    options=df['category'].unique(),
    default=df['category'].unique()
)

# Store location filter
locations = st.sidebar.multiselect(
    "Selecteer winkellocatie(s)",
    options=df['store_location'].unique(),
    default=df['store_location'].unique()
)

# Customer loyalty level filter
loyalty_levels = st.sidebar.multiselect(
    "Selecteer loyaliteitsniveau",
    options=df['customer_loyalty_level'].unique(),
    default=df['customer_loyalty_level'].unique()
)

# Filter data
filtered_df = df[
    (df['transaction_date'].dt.date >= date_range[0]) &
    (df['transaction_date'].dt.date <= date_range[1]) &
    (df['category'].isin(categories)) &
    (df['store_location'].isin(locations)) &
    (df['customer_loyalty_level'].isin(loyalty_levels))
]

# KPI's (Key Performance Indicators)
st.subheader("📈 Belangrijkste metingen")
col1, col2, col3, col4 = st.columns(4)

total_sales = (filtered_df['quantity_sold'] * filtered_df['unit_price']).sum()
total_transactions = len(filtered_df)
avg_transaction_value = total_sales / total_transactions if total_transactions > 0 else 0
total_items_sold = filtered_df['quantity_sold'].sum()

# Bereken vorige periode voor vergelijking
prev_period_df = df[
    (df['transaction_date'].dt.date >= date_range[0] - pd.Timedelta(days=30)) &
    (df['transaction_date'].dt.date < date_range[0]) &
    (df['category'].isin(categories)) &
    (df['store_location'].isin(locations)) &
    (df['customer_loyalty_level'].isin(loyalty_levels))
]
prev_sales = (prev_period_df['quantity_sold'] * prev_period_df['unit_price']).sum()
sales_growth = ((total_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0

with col1:
    st.metric("💰 Totale Omzet", f"€{total_sales:,.2f}", f"{sales_growth:+.1f}% vs vorige maand")

with col2:
    st.metric("🛍️ Aantal Transacties", f"{total_transactions:,}")

with col3:
    st.metric("📦 Totale Verkochte Items", f"{total_items_sold:,}")

with col4:
    st.metric("💵 Gemiddelde Transactiewaarde", f"€{avg_transaction_value:,.2f}")

st.divider()

# Visualisaties
col1, col2 = st.columns(2)

# Verkoop per categorie
with col1:
    st.subheader("Verkoop per Categorie")
    sales_by_category = filtered_df.groupby('category').apply(
        lambda x: (x['quantity_sold'] * x['unit_price']).sum()
    ).sort_values(ascending=False)
    
    fig_category = px.bar(
        x=sales_by_category.index,
        y=sales_by_category.values,
        labels={'x': 'Categorie', 'y': 'Omzet (€)'},
        color=sales_by_category.values,
        color_continuous_scale='Blues'
    )
    fig_category.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_category, width='stretch')

# Verkoop per locatie
with col2:
    st.subheader("Verkoop per Winkellocatie")
    sales_by_location = filtered_df.groupby('store_location').apply(
        lambda x: (x['quantity_sold'] * x['unit_price']).sum()
    ).sort_values(ascending=False)
    
    fig_location = px.bar(
        x=sales_by_location.index,
        y=sales_by_location.values,
        labels={'x': 'Locatie', 'y': 'Omzet (€)'},
        color=sales_by_location.values,
        color_continuous_scale='Greens'
    )
    fig_location.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_location, width='stretch')

# Verkoop over tijd
st.subheader("📅 Verkoop over Tijd")
daily_sales = filtered_df.groupby(filtered_df['transaction_date'].dt.date).apply(
    lambda x: (x['quantity_sold'] * x['unit_price']).sum()
)

fig_timeline = px.line(
    x=daily_sales.index,
    y=daily_sales.values,
    labels={'x': 'Datum', 'y': 'Omzet (€)'},
    markers=True
)
fig_timeline.update_layout(height=400, hovermode='x unified')
st.plotly_chart(fig_timeline, width='stretch')

# Promoties analyse
col1, col2 = st.columns(2)

with col1:
    st.subheader("Impact van Promoties")
    promo_impact = filtered_df.groupby('promotion_applied').apply(
        lambda x: (x['quantity_sold'] * x['unit_price']).sum()
    )
    
    fig_promo = px.pie(
        values=promo_impact.values,
        names=['Geen Promotie', 'Met Promotie'] if len(promo_impact) > 1 else ['Geen Promotie'],
        hole=0.4,
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    fig_promo.update_layout(height=400)
    st.plotly_chart(fig_promo, width='stretch')

# Loyaliteitsniveau analyse
with col2:
    st.subheader("Verkoop per Loyaliteitsniveau")
    loyalty_sales = filtered_df.groupby('customer_loyalty_level').apply(
        lambda x: (x['quantity_sold'] * x['unit_price']).sum()
    ).sort_values(ascending=False)
    
    fig_loyalty = px.bar(
        x=loyalty_sales.index,
        y=loyalty_sales.values,
        labels={'x': 'Loyaliteitsniveau', 'y': 'Omzet (€)'},
        color=loyalty_sales.values,
        color_continuous_scale='Purples'
    )
    fig_loyalty.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_loyalty, width='stretch')

# Geavanceerde analyse
st.divider()
st.subheader(" Verdere Analyse")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Producten")
    top_products = filtered_df.groupby('product_name').apply(
        lambda x: (x['quantity_sold'] * x['unit_price']).sum()
    ).sort_values(ascending=False).head(10)
    
    fig_products = px.bar(
        x=top_products.values,
        y=top_products.index,
        orientation='h',
        labels={'x': 'Omzet (€)', 'y': 'Product'},
        color=top_products.values,
        color_continuous_scale='Viridis'
    )
    fig_products.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_products, width='stretch')

with col2:
    st.subheader("Betalingsmethode Verdeling")
    payment_method = filtered_df['payment_method'].value_counts()
    
    fig_payment = px.pie(
        values=payment_method.values,
        names=payment_method.index,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_payment.update_layout(height=400)
    st.plotly_chart(fig_payment, width='stretch')

# 🎯 NIEUWE SECTIE: Voorspellingen & Inzichten
st.divider()
st.subheader("🔮 Voorspellingen & Business Inzichten")

# Voorspelling volgende maand
st.subheader("📈 Omzet Voorspelling (Volgende Maand)")

# Bereken dagelijkse verkoop voor forecasting
daily_sales_full = df.groupby(df['transaction_date'].dt.date).apply(
    lambda x: (x['quantity_sold'] * x['unit_price']).sum()
).reset_index()
daily_sales_full.columns = ['date', 'sales']
daily_sales_full['date'] = pd.to_datetime(daily_sales_full['date'])
daily_sales_full = daily_sales_full.sort_values('date')

# Verbeterde forecasting met moving averages
if len(daily_sales_full) > 30:
    # Gebruik laatste 60 dagen voor meer context
    recent_data = daily_sales_full.tail(60).copy()
    recent_data['days'] = range(len(recent_data))

    # Bereken 7-daagse moving average voor smoothing
    recent_data['sales_smoothed'] = recent_data['sales'].rolling(window=7, center=True).mean()

    # Fill NaN values aan de randen
    recent_data['sales_smoothed'] = recent_data['sales_smoothed'].fillna(method='bfill').fillna(method='ffill')

    # Gebruik smoothed data voor regressie
    X = recent_data[['days']].values
    y = recent_data['sales_smoothed'].values

    model = LinearRegression()
    model.fit(X, y)

    # Voorspel volgende 30 dagen
    future_days = np.array(range(60, 90)).reshape(-1, 1)
    predictions = model.predict(future_days)

    # Alternatieve methode: Gemiddelde van laatste 4 weken
    last_4_weeks = recent_data.tail(28)['sales'].values
    avg_last_4_weeks = np.mean(last_4_weeks)

    # Gebruik gewogen gemiddelde: 70% model, 30% recent gemiddelde
    final_predictions = 0.7 * predictions + 0.3 * avg_last_4_weeks

    # Visualiseer met beide lijnen
    fig_forecast = go.Figure()

    # Originele dagelijkse data
    fig_forecast.add_trace(go.Scatter(
        x=recent_data['date'], y=recent_data['sales'],
        mode='markers', name='Dagelijkse Omzet',
        marker=dict(color='lightblue', size=4, opacity=0.6)
    ))

    # Smoothed trend lijn
    fig_forecast.add_trace(go.Scatter(
        x=recent_data['date'], y=recent_data['sales_smoothed'],
        mode='lines', name='7-daagse Gemiddelde',
        line=dict(color='blue', width=2)
    ))

    # Voorspelling
    future_dates = pd.date_range(start=recent_data['date'].max() + pd.Timedelta(days=1), periods=30)
    fig_forecast.add_trace(go.Scatter(
        x=future_dates, y=final_predictions,
        mode='lines+markers', name='Voorspelling (Gecombineerd Model)',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6)
    ))

    fig_forecast.update_layout(
        title="Verbeterde Omzet Voorspelling - Smoothed Trend + Historisch Gemiddelde",
        xaxis_title="Datum", yaxis_title="Omzet (€)",
        height=400, hovermode='x unified'
    )
    st.plotly_chart(fig_forecast, width='stretch')

    # Voorspellingsinzicht
    avg_predicted = final_predictions.mean()
    current_avg = recent_data['sales'].tail(30).mean()
    growth_rate = ((avg_predicted - current_avg) / current_avg) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Gemiddelde Voorspelde Omzet", f"€{avg_predicted:,.0f}")
    with col2:
        st.metric("📈 Verwachte Groei", f"{growth_rate:+.1f}%")
    with col3:
        st.metric("🎯 Voorspellingsmethode", "Historische data")

    

# 📋 Business Aanbevelingen
st.subheader("💡 Business Aanbevelingen")

recommendations = []

# Analyseer welke categorieën het beste presteren
category_performance = filtered_df.groupby('category').apply(
    lambda x: (x['quantity_sold'] * x['unit_price']).sum()
).sort_values(ascending=False)

top_category = category_performance.index[0]
top_category_sales = category_performance.iloc[0]

recommendations.append(f"🚀 **{top_category}** is je best presterende categorie (€{top_category_sales:,.0f} omzet). Overweeg meer voorraad en marketing hierin te investeren.")

# Analyseer loyaliteitsprogramma
loyalty_analysis = filtered_df.groupby('customer_loyalty_level').apply(
    lambda x: (x['quantity_sold'] * x['unit_price']).sum()
).sort_values(ascending=False)

if 'Platinum' in loyalty_analysis.index:
    platinum_sales = loyalty_analysis['Platinum']
    recommendations.append(f"👑 **Platinum leden** genereren €{platinum_sales:,.0f}. Beloon hen met exclusieve aanbiedingen om retentie te verhogen.")

# Analyseer promotie effectiviteit
promo_effect = filtered_df.groupby('promotion_applied').apply(
    lambda x: (x['quantity_sold'] * x['unit_price']).sum()
)

if len(promo_effect) > 1:
    promo_sales = promo_effect.get(True, 0)
    no_promo_sales = promo_effect.get(False, 0)
    if promo_sales > no_promo_sales:
        promo_lift = ((promo_sales - no_promo_sales) / no_promo_sales) * 100
        recommendations.append(f"🎯 **Promoties werken!** Ze verhogen omzet met {promo_lift:.1f}%. Overweeg meer promotionele campagnes.")

# Analyseer voorraadbeheer
stock_analysis = filtered_df.groupby('product_name').agg({
    'quantity_sold': 'sum',
    'inventory_level': 'mean',
    'reorder_point': 'mean'
}).reset_index()

low_stock_products = stock_analysis[stock_analysis['inventory_level'] <= stock_analysis['reorder_point']]
if len(low_stock_products) > 0:
    recommendations.append(f"📦 **{len(low_stock_products)} producten** zijn bijna uitverkocht. Bestel bij voor levering niet stagneert.")

# Toon aanbevelingen
for rec in recommendations[:5]:  # Toon max 5 aanbevelingen
    st.info(rec)

#  Customer Segmentation
st.subheader("👥 Klantsegmentatie")

# Simple K-means clustering gebaseerd op inkomen en leeftijd
customer_data = filtered_df[['customer_age', 'customer_income']].drop_duplicates()
customer_data = customer_data.dropna()

if len(customer_data) > 10:
    # Normaliseer data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(customer_data)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer_data['segment'] = kmeans.fit_predict(scaled_data)
    
    # Visualiseer segmenten
    fig_segments = px.scatter(
        customer_data, x='customer_age', y='customer_income',
        color='segment', color_continuous_scale='viridis',
        labels={'customer_age': 'Leeftijd', 'customer_income': 'Inkomen (€)'},
        title='Klantsegmenten gebaseerd op Leeftijd & Inkomen'
    )
    fig_segments.update_layout(height=400)
    st.plotly_chart(fig_segments, width='stretch')
    
    # Segment inzichten
    segment_summary = customer_data.groupby('segment').agg({
        'customer_age': 'mean',
        'customer_income': 'mean',
        'segment': 'count'
    }).rename(columns={'segment': 'aantal_klanten'})
    
    st.subheader("Segment Kenmerken")
    for idx, row in segment_summary.iterrows():
        st.write(f"**Segment {idx+1}:** {row['aantal_klanten']} klanten, "
                f"Gem. leeftijd: {row['customer_age']:.0f}, "
                f"Gem. inkomen: €{row['customer_income']:,.0f}")

# Detailtabel
st.divider()
st.subheader("📋 Transactiedetails")

# Sortering opties
sort_column = st.selectbox(
    "Sorteer op:",
    options=['transaction_date', 'unit_price', 'quantity_sold', 'customer_income']
)

display_df = filtered_df[[
    'transaction_id', 'customer_id', 'product_name', 'category', 
    'quantity_sold', 'unit_price', 'transaction_date', 'store_location',
    'customer_loyalty_level', 'payment_method'
]].sort_values(by=sort_column, ascending=False).head(100)

st.dataframe(display_df, width='stretch')

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; margin-top: 50px;'>
    
    <p>Data gesorteerd van {0} tot {1}</p>
</div>
""".format(df['transaction_date'].min().date(), df['transaction_date'].max().date()), 
unsafe_allow_html=True)
