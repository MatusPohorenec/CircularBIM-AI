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


def create_grouped_bar(
    data: dict[str, list[float]],
    categories: list[str],
    title: str = "",
    height: int = 400,
    ci_low: dict[str, list[float]] | None = None,
    ci_high: dict[str, list[float]] | None = None,
) -> go.Figure:
    """Grouped bar chart for model comparison across metrics.

    Parameters
    ----------
    ci_low, ci_high : optional dicts keyed by model name with lists of
        lower/upper 95 % CI bounds. When supplied, error bars are drawn.
    """
    fig = go.Figure()
    for i, (name, values) in enumerate(data.items()):
        error_y = None
        if ci_low and ci_high and name in ci_low and name in ci_high:
            error_y = dict(
                type="data",
                symmetric=False,
                array=[h - v for v, h in zip(values, ci_high[name])],
                arrayminus=[v - lo for v, lo in zip(values, ci_low[name])],
                color="rgba(255,255,255,0.35)",
                thickness=1.5,
                width=4,
            )
        fig.add_trace(go.Bar(
            name=name,
            x=categories,
            y=values,
            marker_color=COLORWAY[i % len(COLORWAY)],
            error_y=error_y,
            hovertemplate=f"{name}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('yaxis',)}
    fig.update_layout(
        **layout,
        barmode="group",
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            range=[0, 1.05],
            gridcolor="rgba(255,255,255,0.05)",
            dtick=0.2,
        ),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14)))
    return fig


def create_cv_strip_plot(
    cv_df,
    metric: str = "macro_f1",
    height: int = 380,
) -> go.Figure:
    """Strip plot (jittered dot plot) of per-fold metric values by model.

    Parameters
    ----------
    cv_df : DataFrame with columns 'model', 'fold', and the chosen metric.
    """
    import pandas as _pd
    models = cv_df["model"].unique().tolist()
    fig = go.Figure()
    for i, model in enumerate(models):
        subset = cv_df[cv_df["model"] == model]
        vals = subset[metric].values
        fig.add_trace(go.Box(
            y=vals,
            name=model,
            boxpoints="all",
            jitter=0.45,
            pointpos=0,
            marker=dict(
                color=COLORWAY[i % len(COLORWAY)],
                size=6,
                opacity=0.8,
            ),
            line=dict(color=COLORWAY[i % len(COLORWAY)]),
            fillcolor="rgba(0,0,0,0)",
            hovertemplate=f"{model}<br>{metric}: %{{y:.4f}}<extra></extra>",
        ))
    layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis', 'yaxis')}
    fig.update_layout(
        **layout,
        height=height,
        showlegend=False,
        yaxis=dict(
            title=metric.replace("_", " ").title(),
            gridcolor="rgba(255,255,255,0.05)",
        ),
        xaxis=dict(
            tickfont=dict(size=10),
        ),
    )
    return fig


def create_model_heatmap(
    models: list[str],
    metrics: list[str],
    values: list[list[float]],
    height: int = 350,
) -> go.Figure:
    """Heatmap of models (rows) x metrics (cols)."""
    text = [[f"{v:.4f}" if v is not None else "\u2014" for v in row] for row in values]
    fig = go.Figure(go.Heatmap(
        z=values,
        x=metrics,
        y=models,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorscale=[
            [0, "rgba(255,107,107,0.3)"],
            [0.5, "rgba(255,184,77,0.3)"],
            [1.0, "rgba(0,212,170,0.6)"],
        ],
        zmin=0, zmax=1,
        hovertemplate="Model: %{y}<br>Metric: %{x}<br>Value: %{z:.4f}<extra></extra>",
        colorbar=dict(tickfont=dict(color=C_MUTED), len=0.8),
    ))
    layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis', 'yaxis')}
    fig.update_layout(
        **layout,
        height=max(height, len(models) * 40 + 80),
        xaxis=dict(tickfont=dict(size=11), side="top"),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
    )
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex to comma-separated RGB string."""
    h = hex_color.lstrip("#")
    return ", ".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))
