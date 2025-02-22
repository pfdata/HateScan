import base64
import os
from pathlib import Path
import dash
from dash import dcc
from dash.dependencies import Input, Output, State
from speech_feature_extractor.audio_utils import process_audio
from utils.common_paths import output_dir
from utils.file_utils import get_file_list

from components.style import (
    button_style_disabled,
    button_style_active,
)


def register_callbacks(app):
    def save_uploaded_file(contents, filename):
        content_type, content_string = contents.split(",")

        decoded = base64.b64decode(content_string)

        file_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(decoded)

        if filename.endswith(".mp4") or filename.endswith(".wav"):
            hate_media_paths = [file_path]
            process_audio(
                hate_media_paths, output_dir, new_sample_rate=16000, new_channels=1
            )
        return filename

    @app.callback(
        [
            Output("file-dropdown", "options"),
            Output("file-dropdown", "value"),
            Output("transcript-dropdown", "options"),
        ],
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
    )
    def update_file_and_transcript_lists(contents, filename):
        if contents is not None and filename is not None:
            if not isinstance(contents, list):
                contents = [contents]
            if not isinstance(filename, list):
                filename = [filename]

            for content, name in zip(contents, filename):
                save_uploaded_file(content, name)

        files = get_file_list()

        filtered_file_list = [file for file in files if file.endswith(".wav")]
        file_dropdown_options = [
            {"label": file, "value": file} for file in filtered_file_list
        ]

        filtered_transcript_list = [file for file in files if file.endswith(".txt")]
        transcript_dropdown_options = [
            {"label": file, "value": file} for file in filtered_transcript_list
        ]

        file_dropdown_value = (
            next(
                (name for name in filename if name.endswith(".wav")),
                dash.no_update,
            )
            if filename
            else dash.no_update
        )

        return file_dropdown_options, file_dropdown_value, transcript_dropdown_options

    @app.callback(
        [Output("audio-player", "src"), Output("audio-player", "hidden")],
        [Input("file-tabs", "value"), Input("file-dropdown", "value")],
    )
    def update_audio_source(selected_file_tab, selected_files_dropdown):
        if not selected_files_dropdown or not selected_file_tab:
            return no_update, True

        return f"/files/{selected_file_tab}", False

    @app.callback(
        Output("transcript-box", "children"),
        Output("transcript-box", "hidden"),
        Input("transcript-tabs", "value"),
        Input("transcript-dropdown", "value"),
        State("transcription-store", "data"),
    )
    def display_transcriptions(
        selected_file_tab, transcripts_dropdown, transcription_data
    ):
        if not selected_file_tab or not transcripts_dropdown:
            return (
                "No file selected. Please select a transcript in the file dropdown.",
                True,
            )
        try:
            transcript_path = Path(output_dir) / selected_file_tab
            with open(str(transcript_path), "r") as transcript:
                file_data = transcript.readlines()
                transcription_data[selected_file_tab] = file_data
            return file_data, False
        except Exception as e:
            print(
                f"An error occurred while processing the transcription data: {str(e)}"
            )
            return (
                "An unexpected error occurred while displaying the transcription. Please try again later.",
                False,
            )

    from dash import no_update

    @app.callback(
        Output("process-acoustic-btn", "style"),
        Output("process-acoustic-btn", "disabled"),
        Output("output-message", "children"),
        Output("acoustic-prediction", "children"),
        Output("predict-acoustic-btn", "style"),
        Input("file-dropdown", "value"),
    )
    def update_acoustic_button_style(selected_files):
        if selected_files:
            return (
                button_style_active,
                False,
                no_update,
                no_update,
                button_style_active,
            )
        else:
            return (
                button_style_disabled,
                True,
                "",
                "",
                {"display": "none"},
            )

    @app.callback(
        Output("extracted-acoustic-features-store", "data", allow_duplicate=True),
        Output("transcription-store", "data", allow_duplicate=True),
        Output("linguistic-features-store", "data", allow_duplicate=True),
        Output("acoustic-prediction", "children", allow_duplicate=True),
        Output("linguistic-prediction", "children", allow_duplicate=True),
        Output("multimodal-prediction", "children", allow_duplicate=True),
        Input("file-dropdown", "value"),
        Input("transcript-dropdown", "value"),
        prevent_initial_call=True,
    )
    def clear_stored_features(audio_files, transcript_files):
        if not audio_files and not transcript_files:
            return {}, {}, {}, "", "", ""
        return no_update, no_update, no_update, no_update, no_update, no_update

    @app.callback(
        Output("process-linguistic-btn", "style"),
        Output("process-linguistic-btn", "disabled"),
        Output("linguistic-prediction", "children"),
        Output("predict-linguistic-btn", "style"),
        Input("transcript-dropdown", "value"),
    )
    def update_linguistic_button_style(selected_files):
        if selected_files:
            return (
                button_style_active,
                False,
                no_update,
                button_style_active,
            )
        else:
            return (
                button_style_disabled,
                True,
                "",
                {"display": "none"},
            )

    @app.callback(Output("file-tabs", "children"), Input("file-dropdown", "value"))
    def update_tabs(selected_files):
        if not selected_files:
            return []

        return [dcc.Tab(label=file, value=file) for file in selected_files]

    @app.callback(
        Output("transcript-tabs", "children"),
        Output("transcript-tabs", "value"),
        Input("transcript-dropdown", "value"),
    )
    def update_transcript_tabs(selected_files):
        if not selected_files:
            return [], ""

        return [dcc.Tab(label=file, value=file) for file in selected_files], ""

    @app.callback(
        Output("feature-sidebar", "style"),
        [
            Input("extracted-acoustic-features-store", "data"),
            Input("transcription-store", "data"),
            Input("file-dropdown", "value"),
            Input("transcript-dropdown", "value"),
        ],
    )
    def toggle_sidebar(
        acoustic_features, transcription, selected_file, selected_transcript
    ):
        has_acoustic_features = False
        if acoustic_features and selected_file:
            if isinstance(selected_file, list):
                has_acoustic_features = any(
                    file in acoustic_features for file in selected_file
                )
            else:
                has_acoustic_features = selected_file in acoustic_features

        has_transcription = False
        if transcription and selected_transcript:
            if isinstance(selected_transcript, list):
                has_transcription = any(
                    transcript in transcription for transcript in selected_transcript
                )
            else:
                has_transcription = selected_transcript in transcription

        if has_acoustic_features or has_transcription:
            return {
                "width": "10%",
                "display": "inline-block",
                "verticalAlign": "top",
                "padding": "20px",
                "backgroundColor": "#f0f0f0",
                "height": "100vh",
                "position": "fixed",
                "right": "0",
                "top": "0",
                "zIndex": "1000",
                "boxShadow": "2px 0 5px rgba(0,0,0,0.1)",
            }

        return {"display": "none"}
