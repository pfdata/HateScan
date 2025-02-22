# Third-party imports
import numpy as np
import opensmile
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_waveform(y: np.ndarray, sr: int):
    """
    Generate time array for audio waveform visualization.

    Parameters:
    - y (np.ndarray): Audio signal array
    - sr (int): Sampling rate in Hz

    Returns:
    - np.ndarray: Time array corresponding to the audio samples
    """
    time = np.arange(len(y)) / sr
    return time


def check_feature_compatibility(features: pd.DataFrame, model) -> bool:
    """
    Check if the extracted features are compatible with the given acoustic model.

    Parameters:
    - features (pd.DataFrame): The extracted features to be checked.
    - model: The model to check compatibility against. Assumes the model has `.n_features_in_`
             or a similar attribute to check expected feature count.

    Returns:
    - bool: True if compatible, otherwise False.
    """
    if features.isna().any().any():
        print("Feature compatibility check failed: Found NaN values.")
        return False

    if not np.issubdtype(features.values.dtype, np.floating):
        print(
            "Feature compatibility check failed: Features are not of a floating point type."
        )
        return False

    if features.shape[1] != model.n_features_in_:
        print(
            f"Feature compatibility check failed: Expected {model.n_features_in_} features, but got {features.shape[1]}."
        )
        return False

    print("Feature compatibility check passed: Features are compatible with the model.")
    return True


def extract_transcription_data(selected_file, transcriber, output_folder):
    """
    Extract transcription data from an audio file using a transcriber.

    Parameters:
    - selected_file (str): The name of the selected audio file.
    - transcriber (object): Transcriber instance with `transcribe_audio` and `normalize_transcripts` methods.
    - output_folder (str): Folder where the audio files are stored.

    Returns:
    - dict: Transcription data as a dictionary or error message.
    """
    if not selected_file:
        return {
            "transcript": "Please select a file from the dropdown to see the transcription."
        }

    file_path = f"{output_folder}/{selected_file}"

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    try:
        transcript_df = transcriber.transcribe_audio(
            audio_paths=file_path, language="english", device=device
        )
        n_transcript_df = transcriber.normalize_transcripts(transcript_df)

        transcription_data = n_transcript_df.to_dict(orient="records")
    except FileNotFoundError:
        transcription_data = {
            "transcript": f"Error - The file {selected_file} could not be found on the server."
        }
    except Exception as e:
        print(f"An error occurred during transcription: {str(e)}")
        transcription_data = {
            "transcript": "An unexpected error occurred while processing the file. Please try again later."
        }

    return transcription_data


# Third-party imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_neural_network(X_train, y_train):
    model = MLPClassifier(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_ensemble(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    model = VotingClassifier(estimators=[("rf", rf), ("gb", gb)], voting="soft")
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    model = SVC(probability=True, kernel="rbf", random_state=42)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def prepare_linguistic_data(
    merged_df,
) -> tuple[pd.DataFrame, pd.Series, list[str], StandardScaler, pd.DataFrame, pd.Series]:
    label_mapping = {"Hate": 1, "Non Hate": 0}
    merged_df["bool_label"] = merged_df["label"].map(label_mapping)

    print("\nPercentage Distribution:")
    print(merged_df["label"].value_counts(normalize=True) * 100)

    columns_to_drop = [
        "label",
        "bool_label",
        "Unnamed: 0",
        "file_name",
        "file_key",
        "video_file_name",
        "hate_snippet",
        "target",
        "transcript",
        "language",
        "normalized_transcript",
    ]

    feature_columns = merged_df.columns.difference(columns_to_drop)
    X = merged_df[feature_columns]
    y = merged_df["bool_label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

    feature_names = feature_columns.tolist()

    return X_scaled_df, y, feature_names, scaler, X


def prepare_acoustic_data(
    merged_df,
) -> tuple[pd.DataFrame, pd.Series, list[str], StandardScaler, pd.DataFrame, pd.Series]:
    label_mapping = {"Hate": 1, "Non Hate": 0}
    merged_df["bool_label"] = merged_df["label"].map(label_mapping)

    columns_to_drop = [
        "label",
        "bool_label",
        "video_file_name",
        "hate_snippet",
        "target",
    ]

    feature_columns = merged_df.columns.difference(columns_to_drop)
    X = merged_df[feature_columns]
    y = merged_df["bool_label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

    return X_scaled_df, y, feature_columns.tolist(), scaler, X
