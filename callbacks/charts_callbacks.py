import dash
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import opensmile
import speech_feature_extractor.opensmile_extractor as opensmile_extractor
import pickle
import numpy as np

from utils.common_paths import (
    output_dir,
    model_path,
    best_linguistic_model,
    best_acoustic_model,
    linguistic_stats,
    acoustic_stats,
)
from utils.features import (
    FEATURE_GROUPS,
    generate_bar_plot,
    generate_radar_plot,
    process_features,
    select_features,
    expand_features,
)


def register_callbacks(app):

    @app.callback(
        Output("functional-features-graph-radar", "figure"),
        Output("functional-features-graph-radar", "style"),
        Input("file-tabs", "value"),
        Input("extracted-acoustic-features-store", "data"),
        Input("acoustic-feature-selection", "value"),
    )
    def update_functional_features_radar(
        selected_file, features_data, selected_features
    ):
        if not selected_file or not selected_features or not features_data:
            return go.Figure(), {"display": "none"}

        try:
            stats = pd.read_csv(acoustic_stats)

            required_columns = {"feature", "label", "mean", "std"}
            if not required_columns.issubset(stats.columns):
                print(
                    f"Missing columns in acoustic stats: {required_columns - set(stats.columns)}"
                )
                return go.Figure(), {"display": "none"}

            expanded_features = expand_features(selected_features, FEATURE_GROUPS)
            if not expanded_features:
                print("Expanded features are empty or invalid.")
                return go.Figure(), {"display": "none"}

            summary_df = stats[stats["feature"].isin(expanded_features)]
            if summary_df.empty:
                print("No matching features found in acoustic stats.")
                return go.Figure(), {"display": "none"}

            acoustic_features = features_data.get(selected_file, {})
            filtered_values = {
                key: acoustic_features.get(key, 0) for key in expanded_features
            }

            features_list = list(filtered_values.keys())
            raw_values = list(filtered_values.values())
            all_means_values = (
                summary_df.loc[summary_df["label"] == "All"]
                .set_index("feature")
                .reindex(features_list)["mean"]
                .fillna(0)
                .tolist()
            )
            all_std_values = [
                std if std != 0 else 1e-6
                for std in summary_df.loc[summary_df["label"] == "All"]
                .set_index("feature")
                .reindex(features_list)["std"]
                .fillna(0)
                .tolist()
            ]
            hate_means_values = (
                summary_df.loc[summary_df["label"] == "Hate"]
                .set_index("feature")
                .reindex(features_list)["mean"]
                .fillna(0)
                .tolist()
            )
            non_hate_means_values = (
                summary_df.loc[summary_df["label"] == "Non Hate"]
                .set_index("feature")
                .reindex(features_list)["mean"]
                .fillna(0)
                .tolist()
            )

            selected_file_values = [
                (raw - mean) / std
                for raw, mean, std in zip(raw_values, all_means_values, all_std_values)
            ]
            hate_means_normalized = [
                (hate - mean) / std
                for hate, mean, std in zip(
                    hate_means_values, all_means_values, all_std_values
                )
            ]
            non_hate_means_normalized = [
                (non_hate - mean) / std
                for non_hate, mean, std in zip(
                    non_hate_means_values, all_means_values, all_std_values
                )
            ]

            fig = generate_radar_plot(
                selected_file,
                selected_file_values,
                features_list,
                hate_means_normalized,
                non_hate_means_normalized,
            )

            style = {
                "display": "block",
                "maxHeight": "600px",
                "overflowY": "auto",
                "width": "100%",
                "border": "1px solid lightgrey",
            }

            return fig, style
        except Exception as e:
            print(f"Error updating functional features radar: {e}")
            return go.Figure(), {"display": "none"}

    @app.callback(
        Output("functional-features-graph-bar", "figure"),
        Output("functional-features-graph-bar", "style"),
        Input("file-tabs", "value"),
        Input("extracted-acoustic-features-store", "data"),
        Input("acoustic-feature-selection", "value"),
    )
    def update_functional_features_bar(selected_file, features_data, selected_features):
        if not selected_file or not selected_features or not features_data:
            return go.Figure(), {"display": "none"}

        try:
            stats = pd.read_csv(acoustic_stats)

            required_columns = {"feature", "label", "mean", "std"}
            if not required_columns.issubset(stats.columns):
                print(
                    f"Missing columns in acoustic stats: {required_columns - set(stats.columns)}"
                )
                return go.Figure(), {"display": "none"}

            expanded_features = expand_features(selected_features, FEATURE_GROUPS)
            if not expanded_features:
                print("Expanded features are empty or invalid.")
                return go.Figure(), {"display": "none"}

            summary_df = stats[stats["feature"].isin(expanded_features)]
            if summary_df.empty:
                print("No matching features found in acoustic stats.")
                return go.Figure(), {"display": "none"}

            acoustic_features = features_data.get(selected_file, {})
            filtered_values = {
                key: acoustic_features.get(key, 0) for key in expanded_features
            }

            features_list = list(filtered_values.keys())
            raw_values = list(filtered_values.values())
            all_means_values = (
                summary_df.loc[summary_df["label"] == "All"]
                .set_index("feature")
                .reindex(features_list)["mean"]
                .fillna(0)
                .tolist()
            )
            all_std_values = [
                std if std != 0 else 1e-6
                for std in summary_df.loc[summary_df["label"] == "All"]
                .set_index("feature")
                .reindex(features_list)["std"]
                .fillna(0)
                .tolist()
            ]
            hate_means_values = (
                summary_df.loc[summary_df["label"] == "Hate"]
                .set_index("feature")
                .reindex(features_list)["mean"]
                .fillna(0)
                .tolist()
            )
            non_hate_means_values = (
                summary_df.loc[summary_df["label"] == "Non Hate"]
                .set_index("feature")
                .reindex(features_list)["mean"]
                .fillna(0)
                .tolist()
            )

            selected_file_values = [
                (raw - mean) / std
                for raw, mean, std in zip(raw_values, all_means_values, all_std_values)
            ]
            hate_means_normalized = [
                (hate - mean) / std
                for hate, mean, std in zip(
                    hate_means_values, all_means_values, all_std_values
                )
            ]
            non_hate_means_normalized = [
                (non_hate - mean) / std
                for non_hate, mean, std in zip(
                    non_hate_means_values, all_means_values, all_std_values
                )
            ]

            fig = generate_bar_plot(
                selected_file,
                selected_file_values,
                features_list,
                hate_means_normalized,
                non_hate_means_normalized,
            )

            style = {
                "display": "block",
                "maxHeight": "600px",
                "overflowY": "auto",
                "width": "100%",
                "border": "1px solid lightgrey",
            }

            return fig, style
        except Exception as e:
            print(f"Error updating functional features bar: {e}")
            return go.Figure(), {"display": "none"}

    @app.callback(
        Output("functional-features-table", "data"),
        Output("functional-features-table", "style_table"),
        Input("file-tabs", "value"),
        Input("acoustic-feature-selection", "value"),
        Input("extracted-acoustic-features-store", "data"),
    )
    def update_functional_features_table(
        selected_file, selected_features, features_data
    ):
        if not selected_file or not selected_features or not features_data:
            return [], {"display": "none"}

        try:
            stats = pd.read_csv(acoustic_stats)

            required_columns = {"feature", "label", "mean"}
            if not required_columns.issubset(stats.columns):
                print(
                    f"Missing columns in acoustic stats: {required_columns - set(stats.columns)}"
                )
                return [], {"display": "none"}

            expanded_features = expand_features(selected_features, FEATURE_GROUPS)
            if not expanded_features:
                print("Expanded features are empty or invalid.")
                return [], {"display": "none"}

            summary_df = stats[stats["feature"].isin(expanded_features)]
            if summary_df.empty:
                print("No matching features found in acoustic stats.")
                return [], {"display": "none"}

            acoustic_features = features_data.get(selected_file, {})
            filtered_values = {
                key: acoustic_features.get(key, None) for key in expanded_features
            }

            features_list = list(filtered_values.keys())
            raw_values = list(filtered_values.values())
            hate_means_values = (
                summary_df.loc[summary_df["label"] == "Hate"]
                .set_index("feature")
                .reindex(features_list)["mean"]
                .fillna(0)
                .tolist()
            )
            non_hate_means_values = (
                summary_df.loc[summary_df["label"] == "Non Hate"]
                .set_index("feature")
                .reindex(features_list)["mean"]
                .fillna(0)
                .tolist()
            )

            functional_df = pd.DataFrame(
                {
                    "Feature": features_list,
                    "Raw Value": raw_values,
                    "Hate Speech Mean": hate_means_values,
                    "Non-Hate Speech Mean": non_hate_means_values,
                }
            )
            functional_df = functional_df.map(
                lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
            )

            table_data = functional_df.to_dict("records")

            table_style = {
                "display": "block",
                "maxHeight": "400px",
                "overflowY": "auto",
                "width": "100%",
                "border": "1px solid lightgrey",
            }

            return table_data, table_style

        except Exception as e:
            print(f"Error updating functional features table: {e}")
            return [], {"display": "none"}

    @app.callback(
        Output("lld-features-graph", "figure"),
        Output("lld-features-graph", "style"),
        [Input("file-tabs", "value"), Input("lld-acoustic-feature-selection", "value")],
    )
    def update_lld_features(selected_file, selected_feature_group):
        if (
            selected_file is None
            or selected_feature_group is None
            or len(selected_feature_group) == 0
        ):
            return go.Figure(), {"display": "none"}

        file_path = f"{output_dir}/{selected_file}"

        features: pd.DataFrame = opensmile_extractor.extract(
            file_path,
            opensmile_set=opensmile.FeatureSet.eGeMAPSv02,
            opensmile_feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

        if features.empty:
            print("Warning: features empty in the LLD extraction")
            return go.Figure(), {"display": "none"}

        selected_features_df = select_features(features, selected_feature_group)

        feature_list = []
        for group in selected_feature_group:
            feature_list += FEATURE_GROUPS.get(group, [])

        feature_list = [
            feature
            for feature in feature_list
            if feature in selected_features_df.columns
        ]
        if len(feature_list) == 0:
            print(
                f"Warning: No valid features found for groups {selected_feature_group}"
            )
            return go.Figure(), {"display": "none"}

        hop_size_in_seconds = 0.01
        time_in_seconds = [
            i * hop_size_in_seconds for i in range(len(selected_features_df))
        ]

        selected_features_df = selected_features_df.copy()
        selected_features_df["time_in_seconds"] = time_in_seconds

        fig = go.Figure()

        for feature in feature_list:
            fig.add_trace(
                go.Scatter(
                    x=selected_features_df["time_in_seconds"],
                    y=selected_features_df[feature],
                    mode="lines",
                    name=feature,
                )
            )

        fig.update_layout(
            title=f"Selected Features Over Time",
            xaxis=dict(
                title="Time (seconds)",
                rangeslider=dict(
                    visible=True,
                    thickness=0.1,
                ),
            ),
            yaxis_title="Feature Value",
            legend_title="Features",
            hovermode="x unified",
        )

        return fig, {"display": "block"}

    @app.callback(
        Output("acoustic-feature-selection", "value"),
        [Input("select-all-features", "n_clicks"), Input("clear-features", "n_clicks")],
        [State("acoustic-feature-selection", "options")],
    )
    def update_functional_features_list(select_all_clicks, clear_clicks, options):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "select-all-features":
            return [option["value"] for option in options]
        elif button_id == "clear-features":
            return []

    @app.callback(
        Output("lld-acoustic-feature-selection", "value"),
        [Input("select-all-lld", "n_clicks"), Input("clear-lld", "n_clicks")],
        [State("lld-acoustic-feature-selection", "options")],
    )
    def update_lld_features_list(select_all_clicks, clear_clicks, options):
        ctx = dash.callback_context

        if not ctx.triggered:
            return dash.no_update
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "select-all-lld":
            return [option["value"] for option in options]
        elif button_id == "clear-lld":
            return []

    @app.callback(
        Output("linguistic-feature-selection", "value"),
        [
            Input("select-all-linguistic", "n_clicks"),
            Input("clear-linguistic", "n_clicks"),
        ],
        [State("linguistic-feature-selection", "options")],
    )
    def update_linguistic_features_list(select_all_clicks, clear_clicks, options):
        ctx = dash.callback_context

        if not ctx.triggered:
            return dash.no_update
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "select-all-linguistic":
            return [option["value"] for option in options]
        elif button_id == "clear-linguistic":
            return []

    @app.callback(
        Output("linguistic-features-table", "data"),
        Output("linguistic-features-table", "style_table"),
        Input("transcript-tabs", "value"),
        Input("linguistic-feature-selection", "value"),
        State("transcription-store", "data"),
    )
    def update_linguistic_features_table(
        selected_transcript, selected_features, transcription_data
    ):
        if not selected_transcript or not selected_features or not transcription_data:
            return [], {"display": "none"}

        try:
            stats = pd.read_csv(linguistic_stats)

            required_columns = {"feature", "label", "mean"}
            if not required_columns.issubset(stats.columns):
                print(
                    f"Missing columns in linguistic stats: {required_columns - set(stats.columns)}"
                )
                return [], {"display": "none"}

            expanded_features = expand_features(selected_features, FEATURE_GROUPS)
            if not expanded_features:
                print("Expanded features are empty or invalid.")
                return [], {"display": "none"}

            summary_df = stats[stats["feature"].isin(expanded_features)]
            if summary_df.empty:
                print("No matching features found in linguistic stats.")
                return [], {"display": "none"}

            hate_means = summary_df.loc[
                summary_df["label"] == "Hate", ["feature", "mean"]
            ].set_index("feature")
            non_hate_means = summary_df.loc[
                summary_df["label"] == "Non Hate", ["feature", "mean"]
            ].set_index("feature")

            linguistic_features = transcription_data.get(selected_transcript, {})
            filtered_values = {
                key: linguistic_features.get(key, None) for key in expanded_features
            }

            features_list = list(filtered_values.keys())
            raw_values = list(filtered_values.values())
            hate_means_values = (
                hate_means.reindex(features_list)["mean"].fillna(0).tolist()
            )
            non_hate_means_values = (
                non_hate_means.reindex(features_list)["mean"].fillna(0).tolist()
            )

            if not features_list:
                print(f"No data for selected transcript: {selected_transcript}")
                return [], {"display": "none"}

            linguistic_df = pd.DataFrame(
                {
                    "Feature": features_list,
                    "Raw Value": raw_values,
                    "Hate Speech Mean": hate_means_values,
                    "Non-Hate Speech Mean": non_hate_means_values,
                }
            )
            linguistic_df = linguistic_df.map(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
            )

            table_data = linguistic_df.to_dict("records")

            table_style = {
                "display": "block",
                "maxHeight": "400px",
                "overflowY": "auto",
                "width": "100%",
                "border": "1px solid lightgrey",
            }

            return table_data, table_style

        except Exception as e:
            print(f"[L-Table] Unexpected error: {e}")
            return [], {"display": "none"}

    @app.callback(
        Output("linguistic-features-graph-bar", "figure"),
        Output("linguistic-features-graph-bar", "style"),
        Input("transcript-tabs", "value"),
        State("transcription-store", "data"),
        Input("linguistic-feature-selection", "value"),
    )
    def update_linguistic_features_bar(
        selected_transcript, transcription_data, selected_features
    ):
        if not transcription_data or not selected_transcript or not selected_features:
            return go.Figure(), {"display": "none"}

        try:
            stats = pd.read_csv(linguistic_stats)

            required_columns = {"feature", "label", "mean", "std"}
            if not required_columns.issubset(stats.columns):
                print(
                    f"Missing columns in linguistic stats: {required_columns - set(stats.columns)}"
                )
                return go.Figure(), {"display": "none"}

            expanded_features = expand_features(selected_features, FEATURE_GROUPS)
            if not expanded_features:
                print("Expanded features are empty or invalid.")
                return go.Figure(), {"display": "none"}

            summary_df = stats[stats["feature"].isin(expanded_features)]
            if summary_df.empty:
                print("No matching features found in linguistic stats.")
                return go.Figure(), {"display": "none"}

            linguistic_features = transcription_data.get(selected_transcript, {})
            filtered_values = {
                key: linguistic_features.get(key, 0) for key in expanded_features
            }

            features_list = list(filtered_values.keys())
            raw_values = list(filtered_values.values())
            all_means_values = (
                summary_df.loc[summary_df["label"] == "All"]
                .set_index("feature")
                .reindex(features_list)["mean"]
                .fillna(0)
                .tolist()
            )
            all_std_values = [
                std if std != 0 else 1e-6
                for std in summary_df.loc[summary_df["label"] == "All"]
                .set_index("feature")
                .reindex(features_list)["std"]
                .fillna(0)
                .tolist()
            ]
            hate_means_values = (
                summary_df.loc[summary_df["label"] == "Hate"]
                .set_index("feature")
                .reindex(features_list)["mean"]
                .fillna(0)
                .tolist()
            )
            non_hate_means_values = (
                summary_df.loc[summary_df["label"] == "Non Hate"]
                .set_index("feature")
                .reindex(features_list)["mean"]
                .fillna(0)
                .tolist()
            )

            selected_file_values = [
                (raw - mean) / std
                for raw, mean, std in zip(raw_values, all_means_values, all_std_values)
            ]
            hate_means_normalized = [
                (hate - mean) / std
                for hate, mean, std in zip(
                    hate_means_values, all_means_values, all_std_values
                )
            ]
            non_hate_means_normalized = [
                (non_hate - mean) / std
                for non_hate, mean, std in zip(
                    non_hate_means_values, all_means_values, all_std_values
                )
            ]

            selected_file_values = np.nan_to_num(selected_file_values)
            hate_means_normalized = np.nan_to_num(hate_means_normalized)
            non_hate_means_normalized = np.nan_to_num(non_hate_means_normalized)

            fig = generate_bar_plot(
                selected_transcript,
                selected_file_values,
                features_list,
                hate_means_normalized,
                non_hate_means_normalized,
            )

            style = {"display": "block"}
            return fig, style

        except ValueError as ve:
            print(f"ValueError: {ve}")
            return go.Figure(), {"display": "none"}
        except KeyError as ke:
            print(f"KeyError: {ke}")
            return go.Figure(), {"display": "none"}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return go.Figure(), {"display": "none"}

    @app.callback(
        Output("linguistic-features-graph-radar", "figure"),
        Output("linguistic-features-graph-radar", "style"),
        Input("transcript-tabs", "value"),
        State("transcription-store", "data"),
        Input("linguistic-feature-selection", "value"),
    )
    def update_linguistic_features_radar(
        selected_transcript, transcription_data, selected_features
    ):
        if not transcription_data or not selected_transcript or not selected_features:
            return go.Figure(), {"display": "none"}

        try:
            stats = pd.read_csv(linguistic_stats)

            required_columns = {"feature", "label", "mean", "std"}
            if not required_columns.issubset(stats.columns):
                print(
                    f"Missing columns in linguistic stats: {required_columns - set(stats.columns)}"
                )
                return go.Figure(), {"display": "none"}

            expanded_features = expand_features(selected_features, FEATURE_GROUPS)
            if not expanded_features:
                print("Expanded features are empty or invalid.")
                return go.Figure(), {"display": "none"}

            summary_df = stats[stats["feature"].isin(expanded_features)]
            if summary_df.empty:
                print("No matching features found in linguistic stats.")
                return go.Figure(), {"display": "none"}

            linguistic_features = transcription_data.get(selected_transcript, {})
            filtered_values = {
                key: linguistic_features.get(key, 0) for key in expanded_features
            }

            features_list = list(filtered_values.keys())
            raw_values = list(filtered_values.values())
            all_means_values = (
                summary_df.loc[summary_df["label"] == "All"]
                .set_index("feature")
                .reindex(features_list)["mean"]
                .fillna(0)
                .tolist()
            )
            all_std_values = [
                std if std != 0 else 1e-6
                for std in summary_df.loc[summary_df["label"] == "All"]
                .set_index("feature")
                .reindex(features_list)["std"]
                .fillna(0)
                .tolist()
            ]
            hate_means_values = (
                summary_df.loc[summary_df["label"] == "Hate"]
                .set_index("feature")
                .reindex(features_list)["mean"]
                .fillna(0)
                .tolist()
            )
            non_hate_means_values = (
                summary_df.loc[summary_df["label"] == "Non Hate"]
                .set_index("feature")
                .reindex(features_list)["mean"]
                .fillna(0)
                .tolist()
            )

            selected_file_values = [
                (raw - mean) / std
                for raw, mean, std in zip(raw_values, all_means_values, all_std_values)
            ]
            hate_means_normalized = [
                (hate - mean) / std
                for hate, mean, std in zip(
                    hate_means_values, all_means_values, all_std_values
                )
            ]
            non_hate_means_normalized = [
                (non_hate - mean) / std
                for non_hate, mean, std in zip(
                    non_hate_means_values, all_means_values, all_std_values
                )
            ]

            selected_file_values = np.nan_to_num(selected_file_values)
            hate_means_normalized = np.nan_to_num(hate_means_normalized)
            non_hate_means_normalized = np.nan_to_num(non_hate_means_normalized)

            fig = generate_radar_plot(
                selected_transcript,
                selected_file_values,
                features_list,
                hate_means_normalized,
                non_hate_means_normalized,
            )

            style = {"display": "block"}
            return fig, style

        except Exception as e:
            print(f"Error updating linguistic features radar: {e}")
            return go.Figure(), {"display": "none"}
