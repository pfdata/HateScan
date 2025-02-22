from dash import dcc, html
from utils.file_utils import get_file_list
from components.style import (
    colors,
    button_style_disabled,
    button_style_active,
    dropdown_style,
)


def create_left_sidebar():
    file_list = get_file_list()
    filtered_file_list = [file for file in file_list if file.endswith(".wav")]

    return html.Div(
        [
            html.H2(
                [
                    html.Span("Hate", style={"color": colors["accent"]}),
                    html.Span(
                        " Speech Detection",
                        style={"color": colors["primary"]},
                    ),
                ],
                style={
                    "marginBottom": "20px",
                    "fontSize": "24px",
                    "fontWeight": "bold",
                },
            ),
            html.H3(
                "Upload Files",
                style={
                    "marginBottom": "10px",
                    "color": colors["text"],
                    "fontSize": "18px",
                },
            ),
            dcc.Upload(
                id="upload-data",
                children=html.Div(
                    [
                        html.I(
                            className="fas fa-cloud-upload-alt",
                            style={
                                "fontSize": "32px",
                                "marginBottom": "15px",
                                "color": colors["button"],
                            },
                        ),
                        html.Div(
                            "Drag and Drop or",
                            style={"fontSize": "16px", "marginBottom": "5px"},
                        ),
                        html.A(
                            "Select Files",
                            style={
                                "color": colors["button"],
                                "textDecoration": "underline",
                                "fontSize": "16px",
                                "fontWeight": "bold",
                                "marginBottom": "10px",
                            },
                        ),
                        html.P(
                            "Supported: .wav, .mp4, .txt",
                            style={
                                "fontSize": "13px",
                                "color": colors["text"],
                                "opacity": "0.8",
                                "margin": "0",
                            },
                        ),
                    ]
                ),
                style={
                    "width": "100%",
                    "height": "150px",
                    "borderWidth": "2px",
                    "borderStyle": "dashed",
                    "borderRadius": "12px",
                    "textAlign": "center",
                    "marginBottom": "25px",
                    "borderColor": colors["button"],
                    "backgroundColor": "#f8f9fa",
                    "display": "flex",
                    "flexDirection": "column",
                    "justifyContent": "center",
                    "alignItems": "center",
                    "cursor": "pointer",
                    "transition": "all 0.3s ease",
                    "&:hover": {
                        "borderColor": colors["accent"],
                        "backgroundColor": "#ffffff",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
                    },
                },
                multiple=True,
                accept=".wav,.mp4,.txt",
            ),
            dcc.Store(id="extracted-acoustic-features-store", data={}),
            dcc.Store(id="transcription-store", data={}),
            dcc.Store(id="linguistic-features-store", data={}),
            html.H4(
                "Select Files for Processing",
                style={
                    "marginBottom": "15px",
                    "color": colors["text"],
                    "fontSize": "16px",
                },
            ),
            dcc.Dropdown(
                id="file-dropdown",
                options=[{"label": file, "value": file} for file in filtered_file_list],
                placeholder="Select your audio files",
                multi=True,
                style=dropdown_style,
            ),
            dcc.Dropdown(
                id="transcript-dropdown",
                options=[
                    {"label": file, "value": file}
                    for file in file_list
                    if file.endswith(".txt")
                ],
                placeholder="Select your transcripts",
                multi=True,
                style=dropdown_style,
            ),
            html.Div(
                [
                    html.Button(
                        "Process Acoustic Features",
                        id="process-acoustic-btn",
                        n_clicks=0,
                        disabled=True,
                        style=button_style_disabled,
                    ),
                    html.Button(
                        "Process Linguistic Features",
                        id="process-linguistic-btn",
                        n_clicks=0,
                        disabled=True,
                        style=button_style_disabled,
                    ),
                ],
                style={"marginTop": "15px"},
            ),
            html.Div(
                [
                    dcc.Loading(
                        id="loading-output-message",
                        type="default",
                        children=html.Div(
                            id="output-message",
                            style={"fontSize": "18px", "color": colors["accent"]},
                        ),
                    ),
                ],
                style={
                    "marginTop": "20px",
                    "padding": "15px",
                    "backgroundColor": "white",
                    "borderRadius": "8px",
                    "border": f'1px solid {colors["accent"]}',
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                },
            ),
            html.Div(
                [
                    html.H4(
                        "Model Predictions",
                        style={
                            "marginBottom": "15px",
                            "color": colors["text"],
                            "fontSize": "18px",
                        },
                    ),
                    html.Div(
                        [
                            html.H5(
                                "Acoustic Model:",
                                style={"color": colors["text"], "fontSize": "16px"},
                            ),
                            dcc.Loading(
                                id="loading-acoustic-prediction",
                                type="default",
                                children=html.Div(
                                    id="acoustic-prediction",
                                    style={"marginBottom": "10px", "fontSize": "14px"},
                                ),
                            ),
                            html.Button(
                                "Predict Acoustic Model",
                                id="predict-acoustic-btn",
                                n_clicks=0,
                                style=button_style_active,
                            ),
                        ],
                        id="acoustic-prediction-block",
                        style={"display": "none", "marginBottom": "20px"},
                    ),
                    html.Div(
                        [
                            html.H5(
                                "Linguistic Model:",
                                style={"color": colors["text"], "fontSize": "16px"},
                            ),
                            dcc.Loading(
                                id="loading-linguistic-prediction",
                                type="default",
                                children=html.Div(
                                    id="linguistic-prediction",
                                    style={"marginBottom": "10px", "fontSize": "14px"},
                                ),
                            ),
                            html.Button(
                                "Predict Linguistic Model",
                                id="predict-linguistic-btn",
                                n_clicks=0,
                                style=button_style_active,
                            ),
                        ],
                        id="linguistic-prediction-block",
                        style={"display": "none", "marginBottom": "20px"},
                    ),
                    html.Div(
                        [
                            html.H5(
                                "Multimodal Classification:",
                                style={"color": colors["text"], "fontSize": "16px"},
                            ),
                            dcc.Loading(
                                id="loading-multimodal-prediction",
                                type="default",
                                children=html.Div(
                                    id="multimodal-prediction",
                                    style={"marginBottom": "10px", "fontSize": "14px"},
                                ),
                            ),
                            html.Button(
                                "Predict Multimodal",
                                id="predict-multimodal-btn",
                                n_clicks=0,
                                style=button_style_active,
                            ),
                        ],
                        id="multimodal-prediction-block",
                        style={"display": "none"},
                    ),
                ],
                style={
                    "padding": "20px",
                    "backgroundColor": "white",
                    "border": f'1px solid {colors["primary"]}',
                    "borderRadius": "8px",
                    "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
                },
            ),
        ],
        style={
            "width": "300px",
            "display": "inline-block",
            "verticalAlign": "top",
            "padding": "20px",
            "backgroundColor": colors["background"],
            "height": "100vh",
            "position": "fixed",
            "left": "0",
            "top": "0",
            "overflowY": "auto",
            "zIndex": "1000",
            "boxShadow": "2px 0 10px rgba(0,0,0,0.1)",
        },
    )
