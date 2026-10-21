"""
Visualization & UI Utilities for DemandWise
============================================
Generates clean, professional Plotly charts adhering to the SaaS design palette:
- Primary accent: Deep Green (#15803D)
- Warning: Amber (#D97706)
- Alert / Surge: Crimson (#DC2626)
- Slate / Text: Charcoal (#1F2937)
- Background: Soft White (#FFFFFF / #F9FAFB)
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def get_base_layout(title: str = "", height: int = 350) -> Dict[str, Any]:
    """Returns clean, minimal layout options for Plotly figures."""
    return dict(
        title=dict(text=title, font=dict(family=FONT_FAMILY, size=15, color="#1F2937", weight="bold")),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family=FONT_FAMILY, color="#4B5563", size=12),
        margin=dict(l=40, r=30, t=50, b=40),
        height=height,
        xaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            linecolor="#E5E7EB",
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            linecolor="#E5E7EB",
            tickfont=dict(size=11)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        )
    )


def plot_store_sales_trend(
    df: pd.DataFrame,
    store_name: str,
    days: int = 60
) -> go.Figure:
    """Renders recent overall store daily sales trend with 7-day moving average."""
    sub = df[df['store_name'] == store_name].copy()
    daily = sub.groupby('date', as_index=False)['sales'].sum().sort_values('date')
    
    if days and len(daily) > days:
        daily = daily.tail(days)
        
    daily['ma7'] = daily['sales'].rolling(7, min_periods=1).mean()
    
    fig = go.Figure()
    
    # Daily bars or muted line
    fig.add_trace(go.Bar(
        x=daily['date'],
        y=daily['sales'],
        name="Daily Sales Units",
        marker=dict(color="#E5E7EB"),
        opacity=0.75,
        hovertemplate="%{x|%b %d, %Y}<br>Sales: %{y:,.0f} units<extra></extra>"
    ))
    
    # 7-day trendline
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['ma7'],
        name="7-Day Moving Avg",
        mode="lines",
        line=dict(color="#15803D", width=2.5),
        hovertemplate="%{x|%b %d, %Y}<br>7-Day Avg: %{y:,.1f} units<extra></extra>"
    ))
    
    layout = get_base_layout(f"Daily Sales Trend — {store_name} (Last {days} Days)", height=320)
    fig.update_layout(layout)
    return fig


def plot_category_sales_explorer(
    df: pd.DataFrame,
    store_name: str,
    category: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> go.Figure:
    """Renders daily sales line chart and 7-day moving average for selected category."""
    sub = df[(df['store_name'] == store_name) & (df['category'] == category)].copy()
    sub['date'] = pd.to_datetime(sub['date'])
    sub = sub.sort_values('date')
    
    if start_date:
        sub = sub[sub['date'] >= pd.to_datetime(start_date)]
    if end_date:
        sub = sub[sub['date'] <= pd.to_datetime(end_date)]
        
    sub['ma7'] = sub['sales'].rolling(7, min_periods=1).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sub['date'],
        y=sub['sales'],
        name="Daily Sales",
        mode="lines",
        line=dict(color="#9CA3AF", width=1.5),
        hovertemplate="%{x|%b %d, %Y}<br>Sales: %{y:,.0f} units<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=sub['date'],
        y=sub['ma7'],
        name="7-Day Moving Avg",
        mode="lines",
        line=dict(color="#15803D", width=2.5),
        hovertemplate="%{x|%b %d, %Y}<br>7-Day Avg: %{y:,.1f} units<extra></extra>"
    ))
    
    layout = get_base_layout(f"Historical Sales — {category} ({store_name})", height=350)
    fig.update_layout(layout)
    return fig


def plot_category_breakdown_bar(df: pd.DataFrame, store_name: str) -> go.Figure:
    """Renders horizontal bar chart of total category sales for a store."""
    sub = df[df['store_name'] == store_name].copy()
    cat_totals = sub.groupby('category', as_index=False)['sales'].sum()
    cat_totals = cat_totals.sort_values('sales', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=cat_totals['sales'],
        y=cat_totals['category'],
        orientation='h',
        marker=dict(color="#15803D"),
        hovertemplate="%{y}<br>Total Sales: %{x:,.0f} units<extra></extra>"
    ))
    
    layout = get_base_layout(f"Total Sales by Category — {store_name}", height=320)
    layout['xaxis']['title'] = "Units Sold"
    fig.update_layout(layout)
    return fig


def plot_revenue_trend(df: pd.DataFrame, store_name: str) -> go.Figure:
    """Renders total daily revenue over time for a store."""
    sub = df[df['store_name'] == store_name].copy()
    daily_rev = sub.groupby('date', as_index=False)['revenue'].sum().sort_values('date')
    daily_rev['ma7'] = daily_rev['revenue'].rolling(7, min_periods=1).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_rev['date'],
        y=daily_rev['revenue'],
        name="Daily Revenue ($)",
        marker=dict(color="#E5E7EB"),
        hovertemplate="%{x|%b %d, %Y}<br>Revenue: $%{y:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=daily_rev['date'],
        y=daily_rev['ma7'],
        name="7-Day Avg Revenue ($)",
        line=dict(color="#0F766E", width=2.5),
        hovertemplate="%{x|%b %d, %Y}<br>7-Day Avg: $%{y:,.2f}<extra></extra>"
    ))
    
    layout = get_base_layout(f"Daily Revenue Trend — {store_name}", height=320)
    fig.update_layout(layout)
    return fig


def plot_promotion_comparison(df: pd.DataFrame, store_name: str, category: Optional[str] = None) -> go.Figure:
    """Compares average daily sales between promotional and non-promotional days."""
    sub = df[df['store_name'] == store_name].copy()
    if category:
        sub = sub[sub['category'] == category].copy()
        
    promo_stats = sub.groupby('promotion')['sales'].mean().reset_index()
    promo_stats['label'] = promo_stats['promotion'].map({0: 'Regular Days', 1: 'Promotional Days'})
    
    fig = go.Figure(go.Bar(
        x=promo_stats['label'],
        y=promo_stats['sales'],
        marker=dict(color=['#9CA3AF', '#15803D']),
        text=promo_stats['sales'].round(1),
        textposition='outside',
        hovertemplate="%{x}<br>Average Sales: %{y:,.1f} units/day<extra></extra>"
    ))
    
    title_suffix = f" ({category})" if category else ""
    layout = get_base_layout(f"Promotion Impact on Daily Sales — {store_name}{title_suffix}", height=300)
    layout['yaxis']['title'] = "Average Units Sold / Day"
    fig.update_layout(layout)
    return fig


def plot_confusion_matrix_chart(cm_data: Dict[str, Any]) -> go.Figure:
    """Renders a 2x2 confusion matrix heatmap."""
    matrix = cm_data.get('matrix', [[0, 0], [0, 0]])
    labels = ["No Surge (0)", "Surge (1)"]
    
    text = [
        [f"True Negative (TN)<br><b>{matrix[0][0]:,}</b>", f"False Positive (FP)<br><b>{matrix[0][1]:,}</b>"],
        [f"False Negative (FN)<br><b>{matrix[1][0]:,}</b>", f"True Positive (TP)<br><b>{matrix[1][1]:,}</b>"]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(family=FONT_FAMILY, size=13, color="#1F2937"),
        colorscale=[[0, "#F3F4F6"], [0.5, "#D1FAE5"], [1, "#15803D"]],
        showscale=False
    ))
    
    layout = get_base_layout("Test Set Confusion Matrix", height=320)
    layout['xaxis']['title'] = "Predicted Class"
    layout['yaxis']['title'] = "Actual Class"
    layout['yaxis']['autorange'] = "reversed"
    fig.update_layout(layout)
    return fig


def plot_roc_curve_chart(roc_dict: Dict[str, Any], auc_score: Optional[float] = None) -> go.Figure:
    """Renders ROC curve."""
    fig = go.Figure()
    
    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name="Random Classifier",
        mode="lines",
        line=dict(color="#D1D5DB", dash="dash")
    ))
    
    if roc_dict and 'fpr' in roc_dict:
        auc_label = f" (AUC = {auc_score:.4f})" if auc_score is not None else ""
        fig.add_trace(go.Scatter(
            x=roc_dict['fpr'],
            y=roc_dict['tpr'],
            name=f"Logistic Regression{auc_label}",
            mode="lines",
            line=dict(color="#15803D", width=2.5)
        ))
        
    layout = get_base_layout("ROC Curve (Receiver Operating Characteristic)", height=320)
    layout['xaxis']['title'] = "False Positive Rate (1 - Specificity)"
    layout['yaxis']['title'] = "True Positive Rate (Recall)"
    fig.update_layout(layout)
    return fig


def plot_feature_importance_chart(coefficients: List[Dict[str, Any]], top_n: int = 10) -> go.Figure:
    """Renders horizontal bar chart of top positive and negative coefficients."""
    if not coefficients:
        return go.Figure()
        
    # Get top positive and negative
    pos = [c for c in coefficients if c['coefficient'] > 0][:top_n // 2]
    neg = [c for c in coefficients if c['coefficient'] < 0][:top_n // 2]
    combined = sorted(pos + neg, key=lambda x: x['coefficient'])
    
    features = [c['feature'] for c in combined]
    values = [c['coefficient'] for c in combined]
    colors = ["#15803D" if v > 0 else "#DC2626" for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(color=colors),
        hovertemplate="%{y}<br>Beta Coefficient: %{x:+.4f}<extra></extra>"
    ))
    
    layout = get_base_layout("Standardized Logistic Regression Coefficients", height=350)
    layout['xaxis']['title'] = "Coefficient Value (Log-Odds Impact)"
    fig.update_layout(layout)
    return fig


def plot_chronological_split_diagram(split_info: Dict[str, Any]) -> go.Figure:
    """Renders interactive timeline showing chronological splits and 7-day purge gaps."""
    fig = go.Figure()
    
    # Train
    fig.add_trace(go.Bar(
        y=["Data Splits"],
        x=[pd.to_datetime(split_info['train_end']) - pd.to_datetime(split_info['train_start'])],
        base=[split_info['train_start']],
        orientation='h',
        name="Training Core",
        marker=dict(color="#15803D"),
        text=[f"Train ({split_info.get('train_rows', 0):,} rows)"],
        textposition="inside",
        hoverinfo="text",
        hovertext=f"Train Core: {split_info['train_start']} to {split_info['train_end']}"
    ))
    
    # Val Purge
    fig.add_trace(go.Bar(
        y=["Data Splits"],
        x=[pd.Timedelta(days=7)],
        base=[split_info['val_purge_start']],
        orientation='h',
        name="7-Day Purge Gap",
        marker=dict(color="#9CA3AF", pattern_shape="/"),
        text=["Purge Gap"],
        textposition="inside",
        hoverinfo="text",
        hovertext="7-day buffer preventing train targets from looking into validation window"
    ))
    
    # Validation
    fig.add_trace(go.Bar(
        y=["Data Splits"],
        x=[pd.to_datetime(split_info['val_end']) - pd.to_datetime(split_info['val_start'])],
        base=[split_info['val_start']],
        orientation='h',
        name="Validation Slice (Threshold Tuning)",
        marker=dict(color="#0F766E"),
        text=[f"Val ({split_info.get('val_rows', 0):,} rows)"],
        textposition="inside",
        hoverinfo="text",
        hovertext=f"Validation Slice: {split_info['val_start']} to {split_info['val_end']}"
    ))
    
    # Test Purge
    fig.add_trace(go.Bar(
        y=["Data Splits"],
        x=[pd.Timedelta(days=7)],
        base=[split_info['test_purge_start']],
        orientation='h',
        name="7-Day Purge Gap",
        marker=dict(color="#9CA3AF", pattern_shape="/"),
        showlegend=False,
        text=["Purge Gap"],
        textposition="inside",
        hoverinfo="text",
        hovertext="7-day buffer preventing validation targets from looking into test window"
    ))
    
    # Test
    fig.add_trace(go.Bar(
        y=["Data Splits"],
        x=[pd.to_datetime(split_info['test_end']) - pd.to_datetime(split_info['test_start'])],
        base=[split_info['test_start']],
        orientation='h',
        name="Holdout Test Set (Unbiased Eval)",
        marker=dict(color="#2563EB"),
        text=[f"Test ({split_info.get('test_rows', 0):,} rows)"],
        textposition="inside",
        hoverinfo="text",
        hovertext=f"Holdout Test: {split_info['test_start']} to {split_info['test_end']}"
    ))
    
    layout = get_base_layout("Leakage-Safe Chronological Splitting Strategy", height=240)
    layout['barmode'] = 'overlay'
    layout['yaxis']['showticklabels'] = False
    fig.update_layout(layout)
    return fig
