
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from plotly.subplots import make_subplots
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/df_final.csv', parse_dates=['date'])
    # Infer channel columns
    channels = ['fb', 'gg', 'tt']
    return df, channels

def load_campaign_data():
    """Load campaign-level data for drill-down analysis"""
    df_fb = pd.read_csv('data/Facebook.csv', parse_dates=['date'])
    df_gg = pd.read_csv('data/Google.csv', parse_dates=['date'])
    df_tt = pd.read_csv('data/TikTok.csv', parse_dates=['date'])
    
    # Standardize column names
    for df in [df_fb, df_gg, df_tt]:
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('#', 'num').str.replace(r'[^a-z0-9_]', '', regex=True)
        df.rename(columns={'impression': 'impressions', 'attributed_revenue': 'revenue'}, inplace=True)
    
    # Add channel identifier
    df_fb['channel'] = 'Facebook'
    df_gg['channel'] = 'Google'
    df_tt['channel'] = 'TikTok'
    
    # Combine all campaign data
    campaign_df = pd.concat([df_fb, df_gg, df_tt], ignore_index=True)
    return campaign_df

df, channels = load_data()
campaign_df = load_campaign_data()

# Sidebar filters
st.sidebar.header('Filters')
min_date, max_date = df['date'].min(), df['date'].max()
date_range = st.sidebar.date_input('Date range', [min_date, max_date], min_value=min_date, max_value=max_date)
selected_channels = st.sidebar.multiselect('Channels', ['Facebook', 'Google', 'TikTok'], default=['Facebook', 'Google', 'TikTok'])
channel_map = {'Facebook': 'fb', 'Google': 'gg', 'TikTok': 'tt'}
active_channels = [channel_map[c] for c in selected_channels]

# Filter data
mask = (df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))
df_filt = df.loc[mask].copy()

# High-level KPIs
total_revenue = df_filt['total_revenue'].sum()
gross_profit = df_filt['gross_profit'].sum() if 'gross_profit' in df_filt.columns else np.nan
orders = df_filt['orders'].sum() if 'orders' in df_filt.columns else np.nan
total_spend = df_filt['total_spend'].sum()
roas = total_revenue / total_spend if total_spend > 0 else np.nan

st.title('Marketing Intelligence Dashboard')

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric('Total Revenue', f"${total_revenue:,.0f}")
col2.metric('Gross Profit', f"${gross_profit:,.0f}" if not np.isnan(gross_profit) else '-')
col3.metric('Orders', f"{orders:,.0f}" if not np.isnan(orders) else '-')
col4.metric('Marketing Spend', f"${total_spend:,.0f}")
col5.metric('ROAS', f"{roas:.2f}" if not np.isnan(roas) else '-')

st.markdown('---')

# Daily trend charts
st.subheader('Daily Trends')
trend_cols = st.multiselect('Show trends for:', ['Revenue', 'Orders', 'Spend'], default=['Revenue', 'Orders', 'Spend'])
trend_map = {'Revenue': 'total_revenue', 'Orders': 'orders', 'Spend': 'total_spend'}
fig = px.line(df_filt, x='date', y=[trend_map[c] for c in trend_cols if trend_map[c] in df_filt.columns], labels={'value': 'Value', 'date': 'Date', 'variable': 'Metric'})
st.plotly_chart(fig, width="stretch")

# Channel-level breakdown
st.subheader('Channel Breakdown')
breakdown_metric = st.selectbox('Metric', ['Spend', 'Revenue'])
metric_map = {'Spend': 'spend', 'Revenue': 'revenue'}
data = []
for ch in active_channels:
    ch_label = [k for k, v in channel_map.items() if v == ch][0]
    spend = df_filt[f'{ch}_spend'].sum() if f'{ch}_spend' in df_filt.columns else 0
    revenue = df_filt[f'{ch}_revenue'].sum() if f'{ch}_revenue' in df_filt.columns else 0
    data.append({'Channel': ch_label, 'Spend': spend, 'Revenue': revenue})
df_break = pd.DataFrame(data)
fig2 = px.bar(df_break, x='Channel', y=breakdown_metric, color='Channel', barmode='group', text_auto=True)
st.plotly_chart(fig2, width="stretch")

# Performance metrics by channel
st.subheader('Channel Performance Metrics')
st.markdown('---')

# Advanced Analytics Section
st.header('🔬 Advanced Analytics')

# Create tabs for different advanced features
tab1, tab2, tab3, tab4 = st.tabs(['📈 Forecasting', '📊 Seasonality', '🎯 Attribution', '🔍 Campaign Drill-down'])

with tab1:
    st.subheader('Revenue Forecasting')
    
    # Simple forecasting using linear regression
    forecast_days = st.slider('Forecast Days', 7, 30, 14)
    
    # Prepare data for forecasting
    df_forecast = df_filt.copy()
    df_forecast['day_num'] = range(len(df_forecast))
    
    # Train model on total revenue
    X = df_forecast[['day_num']].values
    y = df_forecast['total_revenue'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate predictions
    future_days = np.arange(len(df_forecast), len(df_forecast) + forecast_days).reshape(-1, 1)
    future_dates = pd.date_range(df_forecast['date'].max() + timedelta(days=1), periods=forecast_days)
    predictions = model.predict(future_days)
    
    # Create forecast chart
    fig_forecast = go.Figure()
    
    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=df_forecast['date'], 
        y=df_forecast['total_revenue'],
        name='Historical Revenue',
        line=dict(color='blue')
    ))
    
    # Forecast
    fig_forecast.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    fig_forecast.update_layout(
        title='Revenue Forecast',
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        height=400
    )
    
    st.plotly_chart(fig_forecast, width="stretch")
    
    # Forecast metrics
    col1, col2, col3 = st.columns(3)
    col1.metric('Forecasted Total Revenue', f"${predictions.sum():,.0f}")
    col2.metric('Average Daily Revenue', f"${predictions.mean():,.0f}")
    col3.metric('Model R²', f"{r2_score(y, model.predict(X)):.3f}")

with tab2:
    st.subheader('Seasonality Analysis')
    
    # Add day of week analysis
    df_season = df_filt.copy()
    df_season['day_of_week'] = df_season['date'].dt.day_name()
    df_season['week_num'] = df_season['date'].dt.isocalendar().week
    
    # Day of week performance
    dow_perf = df_season.groupby('day_of_week').agg({
        'total_revenue': 'mean',
        'total_spend': 'mean',
        'roas': 'mean'
    }).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    fig_dow = px.bar(
        x=dow_perf.index, 
        y=dow_perf['total_revenue'],
        title='Average Revenue by Day of Week',
        labels={'x': 'Day of Week', 'y': 'Average Revenue ($)'}
    )
    st.plotly_chart(fig_dow, width="stretch")
    
    # Weekly trends
    weekly_trend = df_season.groupby('week_num').agg({
        'total_revenue': 'sum',
        'total_spend': 'sum'
    })
    
    fig_weekly = px.line(
        weekly_trend, 
        x=weekly_trend.index, 
        y=['total_revenue', 'total_spend'],
        title='Weekly Trends',
        labels={'value': 'Amount ($)', 'week_num': 'Week Number'}
    )
    st.plotly_chart(fig_weekly, width="stretch")

with tab3:
    st.subheader('Attribution Modeling')
    
    # Multi-touch attribution analysis
    st.write("**Channel Contribution Analysis**")
    
    # Calculate channel correlations with business outcomes
    channel_cols = [col for col in df_filt.columns if any(ch in col for ch in ['fb', 'gg', 'tt']) and 'spend' in col]
    
    correlations = {}
    for col in channel_cols:
        if df_filt[col].sum() > 0:  # Only include channels with spend
            corr_revenue = df_filt[col].corr(df_filt['business_total_revenue'])
            corr_orders = df_filt[col].corr(df_filt['business_orders']) if 'business_orders' in df_filt.columns else 0
            correlations[col.replace('_spend', '').upper()] = {
                'Revenue Correlation': corr_revenue,
                'Orders Correlation': corr_orders
            }
    
    if correlations:
        corr_df = pd.DataFrame(correlations).T
        st.dataframe(corr_df.style.format('{:.3f}'), width="stretch")
    
    # Attribution weights based on last 7 days impact
    st.write("**7-Day Attribution Windows**")
    attribution_results = {}
    
    for lag in range(1, 8):
        for channel in ['fb', 'gg', 'tt']:
            spend_col = f'{channel}_spend'
            if spend_col in df_filt.columns:
                lagged_spend = df_filt[spend_col].shift(lag)
                if not lagged_spend.isna().all():
                    corr = lagged_spend.corr(df_filt['business_total_revenue'])
                    attribution_results[f'{channel.upper()}_lag_{lag}'] = corr
    
    # Show top attribution windows
    if attribution_results:
        top_attribution = sorted(attribution_results.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        attr_df = pd.DataFrame(top_attribution, columns=['Channel_Lag', 'Correlation'])
        
        fig_attr = px.bar(
            attr_df, 
            x='Channel_Lag', 
            y='Correlation',
            title='Top Attribution Windows (Channel Impact on Revenue)',
            color='Correlation',
            color_continuous_scale='RdYlBu'
        )
        fig_attr.update_xaxes(tickangle=45)
        st.plotly_chart(fig_attr, width="stretch")

with tab4:
    st.subheader('Campaign-Level Analysis')
    
    # Campaign performance drill-down
    selected_channel = st.selectbox('Select Channel for Drill-down', ['Facebook', 'Google', 'TikTok'])
    
    # Filter campaign data
    channel_campaigns = campaign_df[campaign_df['channel'] == selected_channel].copy()
    
    if not channel_campaigns.empty:
        # Date range filter for campaigns
        campaign_date_range = st.date_input(
            'Campaign Date Range', 
            [channel_campaigns['date'].min(), channel_campaigns['date'].max()],
            min_value=channel_campaigns['date'].min(),
            max_value=channel_campaigns['date'].max()
        )
        
        # Filter by date
        if len(campaign_date_range) == 2:
            mask = (channel_campaigns['date'] >= pd.to_datetime(campaign_date_range[0])) & \
                   (channel_campaigns['date'] <= pd.to_datetime(campaign_date_range[1]))
            channel_campaigns = channel_campaigns.loc[mask]
        
        # Campaign performance summary
        campaign_summary = channel_campaigns.groupby('campaign').agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'revenue': 'sum'
        }).round(2)
        
        # Calculate campaign metrics
        campaign_summary['CTR'] = (campaign_summary['clicks'] / campaign_summary['impressions']).round(4)
        campaign_summary['CPC'] = (campaign_summary['spend'] / campaign_summary['clicks']).round(2)
        campaign_summary['ROAS'] = (campaign_summary['revenue'] / campaign_summary['spend']).round(2)
        
        # Sort by ROAS
        campaign_summary = campaign_summary.sort_values('ROAS', ascending=False)
        
        st.write(f"**Top Campaigns - {selected_channel}**")
        st.dataframe(
            campaign_summary.head(10).style.format({
                'spend': '${:,.2f}',
                'impressions': '{:,.0f}',
                'clicks': '{:,.0f}',
                'revenue': '${:,.2f}',
                'CTR': '{:.2%}',
                'CPC': '${:.2f}',
                'ROAS': '{:.2f}x'
            }),
            width="stretch"
        )
        
        # Campaign performance visualization
        top_campaigns = campaign_summary.head(10)
        
        fig_campaigns = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Spend vs Revenue', 'ROAS by Campaign', 'CTR Distribution', 'CPC Distribution'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Spend vs Revenue
        fig_campaigns.add_trace(
            go.Scatter(x=top_campaigns['spend'], y=top_campaigns['revenue'], 
                      mode='markers', name='Campaigns', text=top_campaigns.index),
            row=1, col=1
        )
        
        # ROAS by campaign
        fig_campaigns.add_trace(
            go.Bar(x=top_campaigns.index[:5], y=top_campaigns['ROAS'][:5], name='ROAS'),
            row=1, col=2
        )
        
        # CTR distribution
        fig_campaigns.add_trace(
            go.Histogram(x=campaign_summary['CTR'], name='CTR', nbinsx=10),
            row=2, col=1
        )
        
        # CPC distribution
        fig_campaigns.add_trace(
            go.Histogram(x=campaign_summary['CPC'], name='CPC', nbinsx=10),
            row=2, col=2
        )
        
        fig_campaigns.update_layout(height=600, showlegend=False)
        fig_campaigns.update_xaxes(title_text="Spend ($)", row=1, col=1)
        fig_campaigns.update_yaxes(title_text="Revenue ($)", row=1, col=1)
        fig_campaigns.update_xaxes(title_text="Campaign", row=1, col=2, tickangle=45)
        fig_campaigns.update_yaxes(title_text="ROAS", row=1, col=2)
        fig_campaigns.update_xaxes(title_text="CTR", row=2, col=1)
        fig_campaigns.update_xaxes(title_text="CPC ($)", row=2, col=2)
        
        st.plotly_chart(fig_campaigns, width="stretch")
        
        # Tactical insights
        st.write("**📋 Tactical Insights**")
        
        # Best performing tactic
        tactic_perf = channel_campaigns.groupby('tactic').agg({
            'revenue': 'sum',
            'spend': 'sum'
        })
        tactic_perf['ROAS'] = tactic_perf['revenue'] / tactic_perf['spend']
        best_tactic = tactic_perf['ROAS'].idxmax()
        
        col1, col2, col3 = st.columns(3)
        col1.metric('Best Performing Tactic', best_tactic)
        col2.metric('Best Tactic ROAS', f"{tactic_perf.loc[best_tactic, 'ROAS']:.2f}x")
        col3.metric('Total Campaigns', len(campaign_summary))

st.markdown('---')
# Channel performance metrics
perf_metrics = []
for ch in active_channels:
    ch_label = [k for k, v in channel_map.items() if v == ch][0]
    imp = df_filt[f'{ch}_impressions'].sum() if f'{ch}_impressions' in df_filt.columns else 0
    clk = df_filt[f'{ch}_clicks'].sum() if f'{ch}_clicks' in df_filt.columns else 0
    spend = df_filt[f'{ch}_spend'].sum() if f'{ch}_spend' in df_filt.columns else 0
    revenue = df_filt[f'{ch}_revenue'].sum() if f'{ch}_revenue' in df_filt.columns else 0
    ctr = clk / imp if imp > 0 else np.nan
    cpc = spend / clk if clk > 0 else np.nan
    roas = revenue / spend if spend > 0 else np.nan
    perf_metrics.append({'Channel': ch_label, 'CTR': ctr, 'CPC': cpc, 'ROAS': roas})
df_perf = pd.DataFrame(perf_metrics)
st.dataframe(df_perf.style.format({'CTR': '{:.2%}', 'CPC': '${:.2f}', 'ROAS': '{:.2f}'}), width="stretch")

# Correlation/lag view
st.subheader('Correlation & Lag Analysis')
st.markdown('Correlation between marketing spend and business outcomes (e.g., revenue, orders).')
outcome = st.selectbox('Business Outcome', ['Revenue', 'Orders'])
outcome_col = 'total_revenue' if outcome == 'Revenue' else 'orders'
if outcome_col in df_filt.columns:
    corr = df_filt['total_spend'].corr(df_filt[outcome_col])
    st.write(f"Correlation (same day): {corr:.2f}")
    # Lagged correlation
    max_lag = 14
    lags = range(0, max_lag+1)
    lag_corrs = [df_filt['total_spend'].corr(df_filt[outcome_col].shift(-lag)) for lag in lags]
    fig3 = px.line(x=list(lags), y=lag_corrs, labels={'x': 'Lag (days)', 'y': 'Correlation'}, title='Lagged Correlation: Spend vs Outcome')
    st.plotly_chart(fig3, width="stretch")

st.markdown('---')
st.caption('Minimal dashboard. Data updates with sidebar filters.')
