import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Page config
st.set_page_config(
    page_title="RTO Prediction Dashboard",
    page_icon="📦",
    layout="wide"
)

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/meesho_features.csv')
    return df

@st.cache_resource
def load_model_artifacts():
    model = joblib.load('models/production/model.pkl')
    with open('models/production/metadata.json', 'r') as f:
        metadata = json.load(f)
    return model, metadata

df = load_data()
model, metadata = load_model_artifacts()

# Title
st.title("📦 RTO Prediction Dashboard")
st.markdown("**Return-to-Origin (RTO) Analytics for E-commerce Logistics**")

# Sidebar
st.sidebar.header("Filters")
selected_states = st.sidebar.multiselect(
    "Select States",
    options=df['delivery_state'].unique(),
    default=df['delivery_state'].unique()[:5]
)

# Filter data
if selected_states:
    df_filtered = df[df['delivery_state'].isin(selected_states)]
else:
    df_filtered = df

# Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Orders", len(df_filtered))
with col2:
    rto_rate = df_filtered['is_rto'].mean() * 100
    st.metric("RTO Rate", f"{rto_rate:.1f}%")
with col3:
    avg_price = df_filtered['final_price'].mean()
    st.metric("Avg Order Value", f"₹{avg_price:.0f}")
with col4:
    st.metric("Model Accuracy", f"{metadata['performance']['accuracy']*100:.1f}%")

st.divider()

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Geographic Analysis", "🎯 Model Performance", "🔍 Predictions"])

with tab1:
    st.header("RTO Distribution Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RTO distribution pie chart
        rto_counts = df_filtered['is_rto'].value_counts()
        fig_pie = px.pie(
            values=rto_counts.values,
            names=['Delivered', 'RTO'],
            title="Order Status Distribution",
            color_discrete_sequence=['#00CC96', '#EF553B']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # RTO by day of week
        dow_rto = df_filtered.groupby('day_of_week')['is_rto'].agg(['count', 'mean'])
        dow_rto['rto_rate'] = dow_rto['mean'] * 100
        dow_rto.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig_dow = px.bar(
            dow_rto,
            x=dow_rto.index,
            y='rto_rate',
            title="RTO Rate by Day of Week",
            labels={'x': 'Day', 'rto_rate': 'RTO Rate (%)'},
            color='rto_rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_dow, use_container_width=True)
    
    # Price analysis
    st.subheader("Price Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_price = px.box(
            df_filtered,
            x='is_rto',
            y='final_price',
            title="Order Value Distribution: Delivered vs RTO",
            labels={'is_rto': 'Status', 'final_price': 'Order Value (₹)'},
            color='is_rto',
            color_discrete_map={0: '#00CC96', 1: '#EF553B'}
        )
        fig_price.update_xaxes(ticktext=['Delivered', 'RTO'], tickvals=[0, 1])
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        fig_shipping = px.box(
            df_filtered,
            x='is_rto',
            y='shipping_charges_total',
            title="Shipping Charges: Delivered vs RTO",
            labels={'is_rto': 'Status', 'shipping_charges_total': 'Shipping (₹)'},
            color='is_rto',
            color_discrete_map={0: '#00CC96', 1: '#EF553B'}
        )
        fig_shipping.update_xaxes(ticktext=['Delivered', 'RTO'], tickvals=[0, 1])
        st.plotly_chart(fig_shipping, use_container_width=True)

with tab2:
    st.header("Geographic RTO Patterns")
    
    # State-level analysis
    state_rto = df_filtered.groupby('delivery_state').agg({
        'is_rto': ['count', 'sum', 'mean']
    }).round(3)
    state_rto.columns = ['Total Orders', 'RTO Count', 'RTO Rate']
    state_rto['RTO Rate'] = (state_rto['RTO Rate'] * 100).round(1)
    state_rto = state_rto.sort_values('RTO Rate', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_state = px.bar(
            state_rto.head(15),
            x=state_rto.head(15).index,
            y='RTO Rate',
            title="Top 15 States by RTO Rate",
            labels={'x': 'State', 'RTO Rate': 'RTO Rate (%)'},
            color='RTO Rate',
            color_continuous_scale='Reds'
        )
        fig_state.update_xaxes(tickangle=45)
        st.plotly_chart(fig_state, use_container_width=True)
    
    with col2:
        fig_volume = px.scatter(
            state_rto,
            x='Total Orders',
            y='RTO Rate',
            size='RTO Count',
            hover_name=state_rto.index,
            title="State Volume vs RTO Rate",
            labels={'Total Orders': 'Order Volume', 'RTO Rate': 'RTO Rate (%)'}
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Metro vs Non-Metro
    st.subheader("Metro vs Non-Metro Cities")
    metro_rto = df_filtered.groupby('is_metro')['is_rto'].agg(['count', 'mean'])
    metro_rto['rto_rate'] = metro_rto['mean'] * 100
    metro_rto.index = ['Non-Metro', 'Metro']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_metro = px.bar(
            metro_rto,
            x=metro_rto.index,
            y='rto_rate',
            title="RTO Rate: Metro vs Non-Metro",
            labels={'x': 'City Type', 'rto_rate': 'RTO Rate (%)'},
            color='rto_rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_metro, use_container_width=True)
    
    with col2:
        st.dataframe(
            state_rto.head(10),
            use_container_width=True,
            height=400
        )

with tab3:
    st.header("Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    perf = metadata['performance']
    
    with col1:
        st.metric("Accuracy", f"{perf['accuracy']:.2%}")
        st.metric("Precision", f"{perf['precision']:.2%}")
    
    with col2:
        st.metric("Recall", f"{perf['recall']:.2%}")
        st.metric("F1-Score", f"{perf['f1_score']:.2%}")
    
    with col3:
        st.metric("ROC-AUC", f"{perf['roc_auc']:.2%}")
        st.metric("PR-AUC", f"{perf['pr_auc']:.2%}")
    
    st.divider()
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = np.array(perf['confusion_matrix'])
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Delivered', 'Predicted RTO'],
        y=['Actual Delivered', 'Actual RTO'],
        text=cm,
        texttemplate="%{text}",
        colorscale='Blues'
    ))
    fig_cm.update_layout(title="Confusion Matrix", height=400)
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Feature Importance
    if metadata['feature_importance']:
        st.subheader("Top 15 Feature Importance")
        
        feature_cols = [
            'day_of_week', 'day_of_month', 'month', 'is_weekend', 
            'is_month_start', 'is_month_end', 'is_metro', 'pin_rto_rate', 
            'state_rto_rate', 'pin_order_count', 'state_order_count',
            'quantity', 'meesho_price', 'final_price', 'shipping_charges_total',
            'price_per_unit', 'discount_amount', 'discount_pct', 
            'shipping_to_price_ratio', 'has_valid_pincode', 'has_state',
            'address_quality_score', 'product_rto_rate', 'product_order_count',
            'product_length', 'price_category_encoded', 'state_clean_encoded'
        ]
        
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': metadata['feature_importance']
        }).sort_values('Importance', ascending=False).head(15)
        
        fig_imp = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance (Top 15)",
            color='Importance',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_imp, use_container_width=True)

with tab4:
    st.header("🔍 Sample Predictions")
    
    # Show sample predictions
    sample_size = st.slider("Number of samples to display", 5, 20, 10)
    sample_df = df_filtered.sample(min(sample_size, len(df_filtered)))
    
    display_cols = ['sub_order_no', 'delivery_state', 'final_price', 
                    'is_metro', 'is_rto', 'pin_rto_rate', 'state_rto_rate']
    
    st.dataframe(
        sample_df[display_cols],
        use_container_width=True
    )
    
    # Model insights
    st.subheader("Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Highest Risk States:**
        {', '.join(state_rto.head(3).index.tolist())}
        """)
    
    with col2:
        weekend_rto = df_filtered[df_filtered['is_weekend']==1]['is_rto'].mean()
        weekday_rto = df_filtered[df_filtered['is_weekend']==0]['is_rto'].mean()
        st.warning(f"""
        **Weekend Risk:** {weekend_rto:.1%} vs Weekday: {weekday_rto:.1%}
        """)

# Footer
st.divider()
st.caption("RTO Prediction Dashboard | Built with Streamlit & MLflow")