import time
from pathlib import Path
from dash.dependencies import Input, Output, State
from dash.long_callback import DiskcacheLongCallbackManager
from utils.features import (
    extract_liwc_features,
    extract_pos_features,
    initialize_liwc_extractor,
    initialize_pos_tagger,
    normalize_transcript,
    extract_mfcc,
    extract_genf_acoustic_features,
    extract_ngram_tfidf_dataframe,
)
import speech_feature_extractor.opensmile_extractor as opensmile_extractor
from utils.common_paths import output_dir


def register_audio_callbacks(app):
    def process_acoustic_files(selected_files, n_mfcc=13):
        results = {}
        durations = {}

        for selected_file in selected_files:
            start_time = time.time()
            file_key = Path(selected_file).name
            feature_dict = {}

            try:
                opensmile_features = extract_genf_acoustic_features(
                    selected_file,
                    opensmile_extractor=opensmile_extractor,
                )

                clean_open_smile_dict = {
                    key: list(value.values())[0]
                    for key, value in opensmile_features.to_dict().items()
                }
                feature_dict.update(clean_open_smile_dict)

                mfcc_features = extract_mfcc(selected_file, n_mfcc=n_mfcc)
                feature_dict.update(mfcc_features)

            except Exception as e:
                feature_dict["Error"] = str(e)
                print(f"Error processing file {selected_file}: {e}")

            finally:
                end_time = time.time()
                durations[file_key] = round(end_time - start_time, 2)
                results[file_key] = feature_dict

        message = (
            f"All acoustic features for {len(selected_files)} files processed successfully.\n"
            "Durations (in seconds) for each file:\n"
            + "\n".join(f"{file}: {duration}s" for file, duration in durations.items())
        )
        return message, results

    def process_linguistic_files(selected_files, transcriptions):
        linguistic_features_data = {}
        durations = {}
        error_messages = []

        nlp_model = initialize_pos_tagger()
        liwc_extractor = initialize_liwc_extractor()

        for selected_file in selected_files:
            start_time = time.time()
            try:
                transcription = transcriptions.get(selected_file, "")
                if not transcription:
                    raise ValueError(
                        f"No transcription available for file: {selected_file}"
                    )
                transcription = normalize_transcript(transcription)

                pos_features = extract_pos_features(transcription, nlp_model)
                liwc_features = extract_liwc_features(transcription, liwc_extractor)
                ngram_tfidf = extract_ngram_tfidf_dataframe(transcription)
                features = {
                    **pos_features,
                    **liwc_features,
                    **ngram_tfidf,
                }
                linguistic_features_data[selected_file] = features

                end_time = time.time()
                durations[selected_file] = round(end_time - start_time, 2)

            except Exception as e:
                end_time = time.time()
                durations[selected_file] = round(end_time - start_time, 2)
                error_messages.append(f"Error processing {selected_file}: {str(e)}")
                linguistic_features_data[selected_file] = f"Error: {str(e)}"

        if error_messages:
            message = (
                f"Some linguistic features were not processed successfully:\n"
                + "\n".join(error_messages)
            )
        else:
            message = (
                f"All linguistic features for {len(selected_files)} files processed successfully.\n"
                "Durations (in seconds) for each file:\n"
                + "\n".join(
                    f"{file}: {duration}s" for file, duration in durations.items()
                )
            )

        return message, linguistic_features_data

    @app.callback(
        Output("output-message", "children", allow_duplicate=True),
        Output("transcription-store", "data", allow_duplicate=True),
        Output("linguistic-prediction-block", "style", allow_duplicate=True),
        Input("process-linguistic-btn", "n_clicks"),
        State("transcript-dropdown", "value"),
        State("output-message", "children"),
        prevent_initial_call=True,
    )
    def process_linguistic_features(n_clicks, selected_files, current_message):
        if n_clicks == 0 or not selected_files:
            return (
                "Please select one or more files before clicking 'Process Linguistic Features'.",
                {},
                {"display": "none"},
            )
        if not isinstance(selected_files, list):
            selected_files = [selected_files]

        transcriptions = {}
        for file in selected_files:
            try:
                file_path = Path(output_dir) / file
                if file_path.exists() and file_path.is_file():
                    with open(file_path, "r", encoding="utf-8") as f:
                        transcriptions[file] = f.read()
                else:
                    transcriptions[file] = (
                        f"Error: File '{file}' does not exist in the output folder."
                    )
            except Exception as e:
                transcriptions[file] = (
                    f"Error: Unable to read file '{file}' due to: {str(e)}"
                )

        message, transcription_data = process_linguistic_files(
            selected_files, transcriptions
        )

        # Append new message to existing message
        if current_message and isinstance(current_message, str):
            combined_message = (
                f"{current_message}\n\n[Linguistic Processing]\n{message}"
            )
        else:
            combined_message = f"[Linguistic Processing]\n{message}\n"

        return combined_message, transcription_data, {"display": "block"}

    @app.callback(
        Output("output-message", "children", allow_duplicate=True),
        Output("extracted-acoustic-features-store", "data", allow_duplicate=True),
        Output("acoustic-prediction-block", "style", allow_duplicate=True),
        Input("process-acoustic-btn", "n_clicks"),
        State("file-dropdown", "value"),
        State("output-message", "children"),
        prevent_initial_call=True,
    )
    def process_acoustic_features(n_clicks, selected_files, current_message):
        if n_clicks == 0 or not selected_files:
            return (
                "Please select a file before clicking 'Process Acoustic Features'.",
                {},
                {"display": "none"},
            )

        if not isinstance(selected_files, list):
            selected_files = [selected_files]

        messages, acoustic_features_data = process_acoustic_files(selected_files)

        # Append new message to existing message
        if current_message and isinstance(current_message, str):
            combined_message = f"{current_message}\n\n[Acoustic Processing]\n{messages}"
        else:
            combined_message = f"[Acoustic Processing]\n{messages}"

        return combined_message, acoustic_features_data, {"display": "block"}

    @app.callback(
        Output("multimodal-prediction-block", "style"),
        [
            Input("transcription-store", "data"),
            Input("extracted-acoustic-features-store", "data"),
        ],
    )
    def toggle_multimodal_block(transcription_data, acoustic_data):
        if transcription_data and acoustic_data:
            return {"display": "block"}
        return {"display": "none"}
