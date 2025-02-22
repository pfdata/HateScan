import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from dash import html
from dash.dependencies import Input, Output, State
import torch
from utils.common_paths import (
    model_path,
    best_acoustic_model,
    best_linguistic_model,
    mutimodal,
)
from utils.features import predict_sample


def register_audio_callbacks(app):
    @app.callback(
        Output("acoustic-prediction", "children", allow_duplicate=True),
        Input("predict-acoustic-btn", "n_clicks"),
        State("extracted-acoustic-features-store", "data"),
        prevent_initial_call=True,
    )
    def predict_acoustic_model(n_clicks, features_data):
        if n_clicks > 0:
            if not features_data:
                return "No features available. Please extract features first."

            try:
                model_fullpath = Path("models") / best_acoustic_model
                with open(model_fullpath, "rb") as file:
                    model = pickle.load(file)

                if hasattr(model, "feature_names"):
                    expected_features = model.feature_names
                else:
                    return "Model error: Expected features are not defined in the model file."
                if hasattr(model, "scaler"):
                    scaler = model.scaler
                else:
                    return "Model error: Scaler no defined in the model file."
                predictions_messages = []

                for file_name, file_features in features_data.items():
                    flat_features = {
                        key: (value["0"] if isinstance(value, dict) else value)
                        for key, value in file_features.items()
                    }

                    features_df = pd.DataFrame([flat_features])
                    missing_features = [
                        feat
                        for feat in expected_features
                        if feat not in features_df.columns
                    ]

                    if missing_features:
                        predictions_messages.append(
                            html.Div(
                                [
                                    html.P(
                                        f"Feature Mismatch Error in '{file_name}':",
                                        style={"fontWeight": "bold"},
                                    ),
                                    html.Ul(
                                        [
                                            html.Li(
                                                f"Missing features: {', '.join(missing_features)}"
                                            )
                                        ],
                                        style={"marginLeft": "20px"},
                                    ),
                                ]
                            )
                        )
                        continue

                    if hasattr(model, "predict"):
                        try:
                            X_scaled = scaler.fit_transform(
                                features_df[expected_features]
                            )
                            X_scaled_df = pd.DataFrame(
                                X_scaled, columns=expected_features
                            )
                            prediction = model.predict(X_scaled_df)
                            if prediction[0] == 1:
                                prediction_text = "Hate Speech Detected"
                                text_color = "red"
                            else:
                                prediction_text = "No Hate Speech Detected"
                                text_color = "green"
                            predictions_messages.append(
                                html.P(
                                    f"Prediction for '{file_name}': {prediction_text}",
                                    style={
                                        "fontWeight": "bold",
                                        "marginLeft": "20px",
                                        "color": text_color,
                                    },
                                )
                            )
                        except Exception as pred_error:
                            predictions_messages.append(
                                html.P(
                                    f"Prediction Error for '{file_name}': {str(pred_error)}",
                                    style={"color": "red"},
                                )
                            )
                    else:
                        predictions_messages.append(
                            html.P(
                                "Model Error: The loaded object is not a scikit-learn model with a 'predict' method.",
                                style={"color": "red"},
                            )
                        )

                return html.Div(predictions_messages)

            except Exception as e:
                return f"Prediction Error: {str(e)}"

        return "Click the button to predict using the acoustic model."

    @app.callback(
        Output("linguistic-prediction", "children", allow_duplicate=True),
        Input("predict-linguistic-btn", "n_clicks"),
        State("transcription-store", "data"),
        prevent_initial_call=True,
    )
    def predict_linguistic_model(n_clicks, linguistic_features_data):
        if n_clicks > 0:
            if not linguistic_features_data:
                return (
                    "No linguistic features available. Please extract features first."
                )

            try:
                if isinstance(linguistic_features_data, dict):
                    linguistic_features_list = []
                    file_keys = []
                    for file_key, feature_dict in linguistic_features_data.items():
                        linguistic_features_list.append(feature_dict)
                        file_keys.append(file_key)
                    linguistic_features = pd.DataFrame(linguistic_features_list)
                else:
                    return "Invalid data format in linguistic features store."

                model_fullpath = model_path / best_linguistic_model
                with open(model_fullpath, "rb") as file:
                    model = pickle.load(file)

                model_feature_names = getattr(model, "feature_names", None)
                if hasattr(model, "scaler"):
                    scaler = model.scaler
                else:
                    return "Model error: Scaler no defined in the model file."
                if model_feature_names is None:
                    return "The model does not have a 'feature_names' attribute."

                missing_features = set(model_feature_names) - set(
                    linguistic_features.columns
                )
                if missing_features:
                    return f"Feature Mismatch Error: The following features are missing: {missing_features}"

                linguistic_features = linguistic_features[model_feature_names]
                linguistic_features.index = file_keys
                X_scaled = scaler.fit_transform(linguistic_features)
                X_scaled_df = pd.DataFrame(X_scaled, columns=model_feature_names)
                predictions = model.predict(X_scaled_df)

                predictions_messages = [
                    html.P(
                        f"File '{file_key}': {'Hate Speech Detected' if prediction == 1 else 'No Hate Speech Detected'}",
                        style={
                            "fontWeight": "bold",
                            "marginLeft": "20px",
                            "color": "red" if prediction == 1 else "green",
                        },
                    )
                    for file_key, prediction in zip(
                        linguistic_features.index, predictions
                    )
                ]

                return html.Div(predictions_messages)

            except Exception as e:
                return f"Prediction Error: {str(e)}"

        return "Click the button to predict using the linguistic model."

    # @app.callback(
    #     Output("multimodal-prediction", "children"),
    #     Input("predict-multimodal-btn", "n_clicks"),
    #     State("extracted-acoustic-features-store", "data"),
    #     State("transcription-store", "data"),
    #     prevent_initial_call=True,
    # )
    # def predict_multimodal_model(
    #     n_clicks, acoustic_features_data, linguistic_features_data
    # ):
    #     if n_clicks > 0:
    #         if not acoustic_features_data:
    #             return "No acoustic features available. Please extract features first."
    #         if not linguistic_features_data:
    #             return (
    #                 "No linguistic features available. Please extract features first."
    #             )

    #         try:
    #             model_fullpath = model_path / mutimodal
    #             if not model_fullpath.exists():
    #                 return "Prediction Error: Model file is missing."

    #             with open(model_fullpath, "rb") as file:
    #                 model = pickle.load(file)

    #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #             model.to(device)
    #             model.eval()

    #             prediction_divs = []

    #             for (acoustic_file, acoustic_feature_dict), (
    #                 linguistic_file,
    #                 linguistic_feature_dict,
    #             ) in zip(
    #                 acoustic_features_data.items(), linguistic_features_data.items()
    #             ):

    #                 prediction_label, probability = predict_sample(
    #                     model, acoustic_feature_dict, linguistic_feature_dict
    #                 )

    #                 if probability > 0.5:
    #                     prediction_text = "Hate Speech Detected"
    #                     if probability > 0.9:
    #                         prediction_text += " (High Confidence)"
    #                     elif probability > 0.7:
    #                         prediction_text += " (Medium Confidence)"
    #                     else:
    #                         prediction_text += " (Low Confidence)"
    #                     text_color = "red"
    #                 else:
    #                     prediction_text = "No Hate Speech Detected"
    #                     if probability < 0.1:
    #                         prediction_text += " (High Confidence)"
    #                     elif probability < 0.3:
    #                         prediction_text += " (Medium Confidence)"
    #                     else:
    #                         prediction_text += " (Low Confidence)"
    #                     text_color = "green"

    #                 prediction_divs.append(
    #                     html.Div(
    #                         [
    #                             html.P(
    #                                 f"Acoustic File: {acoustic_file}",
    #                                 style={"fontWeight": "bold"},
    #                             ),
    #                             html.P(
    #                                 f"Linguistic File: {linguistic_file}",
    #                                 style={"fontWeight": "bold"},
    #                             ),
    #                             html.P(
    #                                 f"Multimodal Prediction: {prediction_text}",
    #                                 style={
    #                                     "fontWeight": "bold",
    #                                     "marginLeft": "20px",
    #                                     "color": text_color,
    #                                 },
    #                             ),
    #                             html.Hr(),
    #                         ]
    #                     )
    #                 )

    #             return html.Div(prediction_divs)

    #         except Exception as e:
    #             return f"Prediction Error: {str(e)}"

    #     return "Click the button to predict using the multimodal model."

    # @app.callback(
    #     Output("multimodal-prediction", "children"),
    #     Input("predict-multimodal-btn", "n_clicks"),
    #     State("extracted-acoustic-features-store", "data"),
    #     State("transcription-store", "data"),
    #     prevent_initial_call=True,
    # )
    # def predict_multimodal_model(
    #     n_clicks,
    #     acoustic_features_data,
    #     linguistic_features_data,
    # ):
    #     if n_clicks > 0:
    #         if not acoustic_features_data:
    #             return "No acoustic features available. Please extract features first."
    #         if not linguistic_features_data:
    #             return (
    #                 "No linguistic features available. Please extract features first."
    #             )

    #         try:
    #             # Load the pretrained models if not already loaded
    #             best_acoustic_fullpath = model_path / best_acoustic_model
    #             with open(best_acoustic_fullpath, "rb") as file:
    #                 acoustic_ensemble = pickle.load(file)

    #             best_linguistic_fullpath = model_path / best_linguistic_model
    #             with open(best_linguistic_fullpath, "rb") as file:
    #                 linguistic_ensemble = pickle.load(file)

    #             prediction_divs = []

    #             for (acoustic_file, acoustic_feature_dict), (
    #                 linguistic_file,
    #                 linguistic_feature_dict,
    #             ) in zip(
    #                 acoustic_features_data.items(), linguistic_features_data.items()
    #             ):
    #                 # Filter acoustic features to match model's expected features
    #                 acoustic_features = np.array(
    #                     [
    #                         acoustic_feature_dict[feature]
    #                         for feature in acoustic_ensemble.feature_names
    #                         if feature in acoustic_feature_dict
    #                     ]
    #                 ).reshape(1, -1)

    #                 # Filter linguistic features to match model's expected features
    #                 linguistic_features = np.array(
    #                     [
    #                         linguistic_feature_dict[feature]
    #                         for feature in linguistic_ensemble.feature_names
    #                         if feature in linguistic_feature_dict
    #                     ]
    #                 ).reshape(1, -1)

    #                 # Check dimensions before prediction
    #                 if acoustic_features.shape[1] != len(
    #                     acoustic_ensemble.feature_names
    #                 ):
    #                     return f"Acoustic feature mismatch: Model expects {len(acoustic_ensemble.feature_names)} features, but got {acoustic_features.shape[1]}."
    #                 if linguistic_features.shape[1] != len(
    #                     linguistic_ensemble.feature_names
    #                 ):
    #                     return f"Linguistic feature mismatch: Model expects {len(linguistic_ensemble.feature_names)} features, but got {linguistic_features.shape[1]}."

    #                 # Predict using the acoustic and linguistic models
    #                 acoustic_prediction = acoustic_ensemble.predict(acoustic_features)[
    #                     0
    #                 ]
    #                 linguistic_prediction = linguistic_ensemble.predict(
    #                     linguistic_features
    #                 )[0]

    #                 # Perform majority voting
    #                 votes = np.bincount([acoustic_prediction, linguistic_prediction])
    #                 multimodal_prediction = np.argmax(votes)

    #                 if multimodal_prediction == 1:
    #                     prediction_text = "Hate Speech Detected"
    #                     text_color = "red"
    #                 else:
    #                     prediction_text = "No Hate Speech Detected"
    #                     text_color = "green"

    #                 prediction_divs.append(
    #                     html.Div(
    #                         [
    #                             html.P(
    #                                 f"Acoustic File: {acoustic_file}",
    #                                 style={"fontWeight": "bold"},
    #                             ),
    #                             html.P(
    #                                 f"Linguistic File: {linguistic_file}",
    #                                 style={"fontWeight": "bold"},
    #                             ),
    #                             html.P(
    #                                 f"Multimodal Prediction: {prediction_text}",
    #                                 style={
    #                                     "fontWeight": "bold",
    #                                     "marginLeft": "20px",
    #                                     "color": text_color,
    #                                 },
    #                             ),
    #                             html.Hr(),
    #                         ]
    #                     )
    #                 )

    #             return html.Div(prediction_divs)

    #         except Exception as e:
    #             return f"Prediction Error: {str(e)}"

    #     return "Click the button to predict using the multimodal model."
    @app.callback(
        Output("multimodal-prediction", "children"),
        Input("predict-multimodal-btn", "n_clicks"),
        State("extracted-acoustic-features-store", "data"),
        State("transcription-store", "data"),
        prevent_initial_call=True,
    )
    def predict_multimodal_model(
        n_clicks,
        acoustic_features_data,
        linguistic_features_data,
    ):
        if n_clicks > 0:
            if not acoustic_features_data:
                return "No acoustic features available. Please extract features first."
            if not linguistic_features_data:
                return (
                    "No linguistic features available. Please extract features first."
                )

            try:
                # Load the pretrained models if not already loaded
                best_acoustic_fullpath = model_path / best_acoustic_model
                with open(best_acoustic_fullpath, "rb") as file:
                    acoustic_ensemble = pickle.load(file)

                best_linguistic_fullpath = model_path / best_linguistic_model
                with open(best_linguistic_fullpath, "rb") as file:
                    linguistic_ensemble = pickle.load(file)

                prediction_divs = []

                for (acoustic_file, acoustic_feature_dict), (
                    linguistic_file,
                    linguistic_feature_dict,
                ) in zip(
                    acoustic_features_data.items(), linguistic_features_data.items()
                ):
                    # Filter acoustic features to match model's expected features
                    acoustic_features = np.array(
                        [
                            acoustic_feature_dict[feature]
                            for feature in acoustic_ensemble.feature_names
                            if feature in acoustic_feature_dict
                        ]
                    ).reshape(1, -1)

                    # Filter linguistic features to match model's expected features
                    linguistic_features = np.array(
                        [
                            linguistic_feature_dict[feature]
                            for feature in linguistic_ensemble.feature_names
                            if feature in linguistic_feature_dict
                        ]
                    ).reshape(1, -1)

                    # Check dimensions before prediction
                    if acoustic_features.shape[1] != len(
                        acoustic_ensemble.feature_names
                    ):
                        return f"Acoustic feature mismatch: Model expects {len(acoustic_ensemble.feature_names)} features, but got {acoustic_features.shape[1]}."
                    if linguistic_features.shape[1] != len(
                        linguistic_ensemble.feature_names
                    ):
                        return f"Linguistic feature mismatch: Model expects {len(linguistic_ensemble.feature_names)} features, but got {linguistic_features.shape[1]}."

                    # Predict probabilities using the acoustic and linguistic models
                    acoustic_probs = acoustic_ensemble.predict_proba(acoustic_features)[
                        0
                    ]
                    linguistic_probs = linguistic_ensemble.predict_proba(
                        linguistic_features
                    )[0]

                    # Perform soft voting: average the probabilities
                    combined_probs = (acoustic_probs + linguistic_probs) / 2
                    multimodal_prediction = np.argmax(combined_probs)

                    # Prepare prediction text
                    if multimodal_prediction == 1:
                        prediction_text = "Hate Speech Detected"
                        text_color = "red"
                    else:
                        prediction_text = "No Hate Speech Detected"
                        text_color = "green"

                    prediction_divs.append(
                        html.Div(
                            [
                                html.P(
                                    f"Acoustic File: {acoustic_file}",
                                    style={"fontWeight": "bold"},
                                ),
                                html.P(
                                    f"Linguistic File: {linguistic_file}",
                                    style={"fontWeight": "bold"},
                                ),
                                html.P(
                                    f"Multimodal Prediction: {prediction_text}",
                                    style={
                                        "fontWeight": "bold",
                                        "marginLeft": "20px",
                                        "color": text_color,
                                    },
                                ),
                                html.Hr(),
                            ]
                        )
                    )

                return html.Div(prediction_divs)

            except Exception as e:
                return f"Prediction Error: {str(e)}"

        return "Click the button to predict using the multimodal model."
