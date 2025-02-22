# Standard library imports
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pickle

# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# Project-specific imports
from utils.audio import (
    check_feature_compatibility,
    generate_waveform,
)
from utils.common_paths import (
    annotations_file,
    data_dir,
    output_dir,
    speech_feature_extractor,
    tmp_dir,
)
from utils.features import (
    FEATURE_GROUPS,
    load_features_from_dict,
    normalize_features,
    normalize_precomputed_means,
    select_features,
    select_stats_features,
)
from utils.file_utils import get_file_list
from utils.visualization import create_radar_plot, create_waveform_plot

sample_hate_audio_path = (
    Path(__file__).parent / "test_data" / "sample_hate_16kHz_mono.wav"
)


def test_generate_waveform():
    y = np.array([1, 2, 3, 4, 5])
    sr = 1000

    expected_time = np.array([0.0, 0.001, 0.002, 0.003, 0.004])

    result = generate_waveform(y, sr)

    np.testing.assert_array_almost_equal(result, expected_time)


def test_common_paths():
    assert (
        speech_feature_extractor.exists()
    ), "Speech feature extractor path does not exist"
    assert data_dir.exists(), "Data directory does not exist"
    assert output_dir.exists(), "Output directory does not exist"
    assert tmp_dir.exists(), "Temporary directory does not exist"

    assert annotations_file.exists(), "Annotations file does not exist"


def test_load_features_from_dict():
    features_dict = {
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "file": ["a", "b", "c"],
    }
    result = load_features_from_dict(features_dict)
    assert isinstance(result, pd.DataFrame)
    assert "file" not in result.columns


def test_select_features():
    columns = FEATURE_GROUPS["pitch"] + FEATURE_GROUPS["loudness"]
    data = {col.replace(".", "_"): [1, 2] for col in columns}
    df = pd.DataFrame(data)

    selected = select_features(df, ["pitch", "loudness"])

    assert selected.shape[1] == len(columns)


def test_select_stats_features():
    df = pd.DataFrame(
        {
            "Feature": ["F0semitoneFrom27_5Hz_sma3nz_amean", "loudness_sma3_amean"],
            "value": [1, 2],
        }
    )
    selected = select_stats_features(df, ["pitch", "loudness"])
    assert selected.shape[0] == 2


def test_normalize_features():
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    normalized = normalize_features(df)
    assert normalized.shape == df.shape


def test_normalize_precomputed_means():
    df = pd.DataFrame(
        {"Feature": ["F0semitoneFrom27_5Hz_sma3nz_amean", "loudness_sma3_amean"]}
    )
    all_means = [1, 2]
    hate_means = [3, 4]
    non_hate_means = [5, 6]
    all_norm, hate_norm, non_hate_norm = normalize_precomputed_means(
        df, all_means, hate_means, non_hate_means
    )
    assert len(all_norm) == len(hate_norm) == len(non_hate_norm) == 2


def test_get_file_list_no_files(monkeypatch):
    monkeypatch.setattr(os, "listdir", lambda _: [])
    result = get_file_list()
    assert result == ["No files available"]


def test_get_file_list_with_files(monkeypatch):
    monkeypatch.setattr(os, "listdir", lambda _: ["file1.txt", "file2.txt"])
    result = get_file_list()
    assert result == ["file1.txt", "file2.txt"]


def test_get_file_list_folder_not_found(monkeypatch):
    monkeypatch.setattr(
        os, "listdir", lambda _: (_ for _ in ()).throw(FileNotFoundError)
    )
    result = get_file_list()
    assert result == ["Output folder not found"]


def test_create_radar_plot():
    categories = ["A", "B", "C"]
    values = [1, 2, 3]
    title = "Radar Test"

    fig = create_radar_plot(categories, values, title)

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == title
    assert len(fig.data) == 1
    assert list(fig.data[0].r) == values


def test_create_waveform_plot():
    time = np.array([0, 1, 2, 3])
    amplitude = np.array([1, -1, 0.5, -0.5])
    title = "Waveform Test"

    fig = create_waveform_plot(time, amplitude, title)

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == title
    assert fig.data[0].x.tolist() == time.tolist()
    assert fig.data[0].y.tolist() == amplitude.tolist()
