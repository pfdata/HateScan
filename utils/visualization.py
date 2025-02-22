# Third-party imports
import numpy as np
import plotly.graph_objects as go


def create_radar_plot(categories: list, values: list, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values, theta=categories, fill="toself", name="Selected Features"
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-6, 6])
        ),
        showlegend=False,
        title=title,
    )
    return fig


def create_waveform_plot(
    time: np.ndarray, amplitude: np.ndarray, title: str
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=amplitude, mode="lines", name="Waveform"))

    fig.update_layout(
        title=title, xaxis_title="Time (s)", yaxis_title="Amplitude", showlegend=False
    )
    return fig
