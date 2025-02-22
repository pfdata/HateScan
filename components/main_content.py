from dash import dcc, html, dash_table
from utils.file_utils import get_file_list
from components.style import colors


def create_main_content():
    file_list = get_file_list()

    return html.Div(
        children=[
            html.Div(
                children=[
                    html.H3("Audio Visualization", style={"color": colors["text"]}),
                    dcc.Tabs(id="file-tabs", value=None, children=[]),
                    html.Audio(
                        id="audio-player",
                        controls=True,
                        src="",
                        hidden=True,
                        style={"width": "100%", "marginBottom": "20px"},
                    ),
                    dcc.Loading(
                        id="loading-audio-content",
                        type="default",
                        children=html.Div(
                            children=[
                                html.Div(
                                    dash_table.DataTable(
                                        id="functional-features-table",
                                        style_table={
                                            "display": "none",
                                            "width": "100%",
                                            "maxHeight": "400px",
                                            "overflowY": "auto",
                                            "border": f"1px solid {colors['primary']}",
                                        },
                                        style_cell={
                                            "textAlign": "left",
                                            "padding": "8px",
                                            "whiteSpace": "normal",
                                            "fontSize": "14px",
                                        },
                                        style_header={
                                            "backgroundColor": colors["background"],
                                            "fontWeight": "bold",
                                            "color": colors["text"],
                                        },
                                        columns=[
                                            {"name": "Feature", "id": "Feature"},
                                            {"name": "Raw Value", "id": "Raw Value"},
                                            {
                                                "name": "Hate Speech Mean",
                                                "id": "Hate Speech Mean",
                                            },
                                            {
                                                "name": "Non-Hate Speech Mean",
                                                "id": "Non-Hate Speech Mean",
                                            },
                                        ],
                                        data=[],
                                    ),
                                    style={"width": "100%", "marginTop": "20px"},
                                ),
                                html.Div(
                                    dcc.Graph(
                                        id="functional-features-graph-bar",
                                        style={"display": "none"},
                                    ),
                                    style={"width": "100%", "marginTop": "20px"},
                                ),
                                html.Div(
                                    dcc.Graph(
                                        id="functional-features-graph-radar",
                                        style={"display": "none"},
                                    ),
                                    style={"width": "100%", "marginTop": "20px"},
                                ),
                                html.Div(
                                    dcc.Graph(
                                        id="lld-features-graph",
                                        style={"display": "none"},
                                    ),
                                    style={"width": "100%", "marginTop": "20px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "alignItems": "center",
                                "marginTop": "20px",
                            },
                        ),
                    ),
                ],
                style={"marginBottom": "50px"},
            ),
            html.Div(
                children=[
                    html.H3(
                        "Transcript Visualization", style={"color": colors["text"]}
                    ),
                    dcc.Tabs(id="transcript-tabs", value=None, children=[]),
                    dcc.Loading(
                        id="loading-transcript-content",
                        type="default",
                        children=html.Div(
                            children=[
                                html.Div(
                                    id="transcript-box",
                                    children="Transcript will be displayed here.",
                                    hidden=True,
                                    style={
                                        "width": "100%",
                                        "height": "150px",
                                        "border": f"1px solid {colors['primary']}",
                                        "padding": "12px",
                                        "overflowY": "auto",
                                        "backgroundColor": colors["background"],
                                        "borderRadius": "5px",
                                        "fontSize": "14px",
                                    },
                                ),
                                html.Div(
                                    dash_table.DataTable(
                                        id="linguistic-features-table",
                                        columns=[
                                            {"name": "Feature", "id": "Feature"},
                                            {"name": "Raw Value", "id": "Raw Value"},
                                            {
                                                "name": "Hate Speech Mean",
                                                "id": "Hate Speech Mean",
                                            },
                                            {
                                                "name": "Non-Hate Speech Mean",
                                                "id": "Non-Hate Speech Mean",
                                            },
                                        ],
                                        data=[],
                                        style_table={"display": "none"},
                                    ),
                                    style={"marginTop": "20px"},
                                ),
                                html.Div(
                                    dcc.Graph(
                                        id="linguistic-features-graph-bar",
                                        style={"display": "none"},
                                    ),
                                    style={"width": "100%", "marginTop": "20px"},
                                ),
                                html.Div(
                                    dcc.Graph(
                                        id="linguistic-features-graph-radar",
                                        style={"display": "none"},
                                    ),
                                    style={"width": "100%", "marginTop": "20px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "alignItems": "center",
                                "marginTop": "20px",
                            },
                        ),
                    ),
                ],
                style={"marginBottom": "50px"},
            ),
        ],
        style={
            "width": "60%",
            "display": "block",
            "padding": "50px",
            "marginLeft": "20%",
            "marginRight": "10%",
            "backgroundColor": "white",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
            "borderRadius": "8px",
        },
    )
