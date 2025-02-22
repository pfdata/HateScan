from dash import html, dcc


def create_right_sidebar():
    return html.Div(
        id="feature-sidebar",
        children=[
            html.H2("Features", style={"margin-bottom": "20px"}),
            html.H3("Acoustic Functional Features", style={"margin-bottom": "10px"}),
            html.Div(
                [
                    html.Button(
                        "Select All",
                        id="select-all-features",
                        n_clicks=0,
                        style={"margin-right": "10px"},
                    ),
                    html.Button("Clear", id="clear-features", n_clicks=0),
                ],
                style={"margin-top": "10px"},
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="acoustic-feature-selection",
                        options=[
                            {"label": "MFCC1-13", "value": "mfcc_all"},
                            {"label": "Pitch (F0)", "value": "pitch"},
                            {"label": "Loudness", "value": "loudness"},
                            {"label": "Spectral", "value": "spectral"},
                            {"label": "Voice Quality", "value": "voice_quality"},
                            {"label": "HNR & Formants", "value": "hnr_formants"},
                            {"label": "Formants (F1, F2, F3)", "value": "formants"},
                            {"label": "MFCCs", "value": "mfcc"},
                            {"label": "Timing & Segmentation", "value": "timing"},
                            {
                                "label": "Unvoiced Dynamics",
                                "value": "unvoiced_dynamics",
                            },
                            {"label": "Intensity", "value": "intensity"},
                        ],
                        value=[],
                        labelStyle={"display": "block"},
                    ),
                ],
                style={
                    "maxHeight": "150px",
                    "overflowY": "scroll",
                    "margin-bottom": "20px",
                },
            ),
            html.H3("Low-Level Descriptors (LLD)", style={"margin-bottom": "10px"}),
            html.Div(
                [
                    html.Button(
                        "Select All",
                        id="select-all-lld",
                        n_clicks=0,
                        style={"margin-right": "10px"},
                    ),
                    html.Button("Clear", id="clear-lld", n_clicks=0),
                ],
                style={"margin-top": "10px"},
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="lld-acoustic-feature-selection",
                        options=[
                            {"label": "MFCCs", "value": "mfcc_lld"},
                            {"label": "Vocal Dynamics", "value": "vocal_dynamics"},
                            {"label": "Loudness", "value": "loudness_lld"},
                            # {"label": "Pitch", "value": "pitch_lld"},
                            {"label": "Voice Quality", "value": "voice_quality_lld"},
                            {"label": "Harmonics", "value": "harmonics"},
                            {"label": "Formants", "value": "formants_lld"},
                        ],
                        value=[],
                        labelStyle={"display": "block"},
                    ),
                ],
                style={
                    "maxHeight": "150px",
                    "overflowY": "scroll",
                    "margin-bottom": "20px",
                },
            ),
            html.H3("Linguistic Features", style={"margin-bottom": "10px"}),
            html.Div(
                [
                    html.Button(
                        "Select All",
                        id="select-all-linguistic",
                        n_clicks=0,
                        style={"margin-right": "10px"},
                    ),
                    html.Button("Clear", id="clear-linguistic", n_clicks=0),
                ],
                style={"margin-top": "10px"},
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="linguistic-feature-selection",
                        options=[
                            {"label": "Part-Of-Speech", "value": "pos"},
                            {"label": "LIWC", "value": "liwc"},
                            {"label": "TF-IDF", "value": "tf-idf"},
                        ],
                        value=[],
                        labelStyle={"display": "block"},
                    ),
                ],
                style={
                    "maxHeight": "150px",
                    "overflowY": "scroll",
                    "margin-bottom": "20px",
                },
            ),
        ],
        style={
            "width": "20%",
            "display": "none",
            "verticalAlign": "top",
            "padding": "20px",
            "backgroundColor": "#f0f0f0",
            "height": "100vh",
            "position": "fixed",
            "right": "0",
            "top": "0",
            "zIndex": "1000",
            "boxShadow": "2px 0 5px rgba(0,0,0,0.1)",
        },
    )
