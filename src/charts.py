"""Shared Plotly chart builders — consistent styling across all pages."""
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from src.st_config import PLOTLY_LAYOUT, C_ACCENT, C_DANGER, C_WARN, C_INFO, COLORWAY, C_TEXT, C_MUTED


def _apply_layout(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    return fig


def create_gauge(value: int, title: str, max_val: int = 5) -> go.Figure:
    """Semi-circular gauge for ordinal prediction (1-5)."""
    color = C_ACCENT if value >= 4 else C_WARN if value >= 3 else C_DANGER
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(font=dict(size=48, color=color)),
        gauge=dict(
            axis=dict(range=[1, max_val], tickwidth=1, tickcolor=C_MUTED,
                      tickvals=[1, 2, 3, 4, 5], tickfont=dict(size=11, color=C_MUTED)),
            bar=dict(color=color, thickness=0.3),
            bgcolor="rgba(255,255,255,0.03)",
            borderwidth=0,
            steps=[
                dict(range=[1, 2], color="rgba(255,107,107,0.12)"),
                dict(range=[2, 3], color="rgba(255,184,77,0.08)"),
                dict(range=[3, 4], color="rgba(255,184,77,0.05)"),
                dict(range=[4, 5], color="rgba(0,212,170,0.08)"),
            ],
        ),
    ))
    layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('margin',)}
    fig.update_layout(
        **layout,
        height=220,
        margin=dict(l=30, r=30, t=10, b=10),
    )
    return fig


def create_proba_bar(proba: dict[int, float]) -> go.Figure:
    """Horizontal stacked bar showing class probability distribution."""
    classes = sorted(proba.keys())
    vals = [proba[c] for c in classes]
    colors = [C_DANGER, C_WARN, "#E8ECF1", C_INFO, C_ACCENT][:len(classes)]

    fig = go.Figure()
    cumulative = 0
    for i, (cls, val) in enumerate(zip(classes, vals)):
        fig.add_trace(go.Bar(
            x=[val], y=[""], orientation="h",
            name=f"Class {cls}",
            marker_color=colors[i] if i < len(colors) else C_MUTED,
            text=f"{val:.0%}" if val > 0.08 else "",
            textposition="inside",
            textfont=dict(size=11),
            hovertemplate=f"Class {cls}: {val:.1%}<extra></extra>",
        ))
    layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis', 'yaxis', 'margin')}
    fig.update_layout(
        **layout,
        barmode="stack",
        height=50,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False),
    )
    return fig


def create_radar(data: dict[str, list[float]], categories: list[str], title: str = "") -> go.Figure:
    """Radar/spider chart overlaying multiple series."""
    fig = go.Figure()
    for i, (name, values) in enumerate(data.items()):
        vals = list(values) + [values[0]]  # close the polygon
        cats = list(categories) + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name=name,
            line=dict(color=COLORWAY[i % len(COLORWAY)], width=2),
            fillcolor=f"rgba({_hex_to_rgb(COLORWAY[i % len(COLORWAY)])}, 0.1)",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 5],
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=10, color=C_MUTED),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=11, color=C_TEXT),
            ),
        ),
        height=420,
        showlegend=True,
        legend=dict(
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def create_heatmap(
    z: list[list[float]],
    x_labels: list[str],
    y_labels: list[str],
    colorscale: str = "Teal",
    zmin: float | None = None,
    zmax: float | None = None,
    fmt: str = ".2f",
) -> go.Figure:
    """Annotated heatmap with custom dark styling."""
    text = [[f"{v:{fmt}}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        text=text, texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.3f}<extra></extra>",
        colorbar=dict(tickfont=dict(color=C_MUTED)),
    ))
    layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis', 'yaxis')}
    fig.update_layout(
        **layout,
        height=max(300, len(y_labels) * 45 + 80),
        xaxis=dict(tickfont=dict(size=11), gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed", gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def create_horizontal_bar(
    labels: list[str],
    values: list[float],
    color: str = C_ACCENT,
    height: int = 400,
) -> go.Figure:
    """Horizontal bar chart for feature importance."""
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=color,
        marker_opacity=[0.5 + 0.5 * (v / max(values)) if max(values) > 0 else 0.7 for v in values],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis', 'yaxis')}
    fig.update_layout(
        **layout,
        height=height,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def create_bar_chart(
    x: list, y: list, color: str = C_ACCENT, labels: dict | None = None, height: int = 350
) -> go.Figure:
    """Simple vertical bar chart."""
    fig = go.Figure(go.Bar(
        x=x, y=y,
        marker_color=color,
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    if labels:
        fig.update_layout(xaxis_title=labels.get("x"), yaxis_title=labels.get("y"))
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex to comma-separated RGB string."""
    h = hex_color.lstrip("#")
    return ", ".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))
