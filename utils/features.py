# Standard library imports
import sqlite3
import string
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import librosa
import plotly.graph_objects as go
import spacy
import opensmile
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import warnings
import torch
import torch.nn as nn

# Project-specific imports
import speech_feature_extractor.opensmile_extractor as opensmile_extractor
from speech_feature_extractor.liwc_extractor import LIWCExtractor
from utils.common_paths import (
    tfidf_vectorizer_path,
    output_dir,
    top_50_tfidf_features_path,
    best_acoustic_model,
    best_linguistic_model,
    model_path,
)

# Feature groups
FEATURE_GROUPS = {
    "pitch": [
        "F0semitoneFrom27.5Hz_sma3nz_amean",
        "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
        "F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
        "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
        "F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
        "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
        "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
        "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
        "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
        "F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope",
    ],
    "loudness": [
        "loudness_sma3_amean",
        "loudness_sma3_stddevNorm",
        "loudness_sma3_percentile20.0",
        "loudness_sma3_percentile50.0",
        "loudness_sma3_percentile80.0",
        "loudness_sma3_pctlrange0-2",
        "loudness_sma3_meanRisingSlope",
        "loudness_sma3_stddevRisingSlope",
        "loudness_sma3_meanFallingSlope",
        "loudness_sma3_stddevFallingSlope",
    ],
    "spectral": ["spectralFlux_sma3_amean", "spectralFlux_sma3_stddevNorm"],
    "voice_quality": [
        "jitterLocal_sma3nz_amean",
        "jitterLocal_sma3nz_stddevNorm",
        "shimmerLocaldB_sma3nz_amean",
        "shimmerLocaldB_sma3nz_stddevNorm",
    ],
    "hnr_formants": [
        "HNRdBACF_sma3nz_amean",
        "HNRdBACF_sma3nz_stddevNorm",
        "logRelF0-H1-H2_sma3nz_amean",
        "logRelF0-H1-H2_sma3nz_stddevNorm",
        "logRelF0-H1-A3_sma3nz_amean",
        "logRelF0-H1-A3_sma3nz_stddevNorm",
    ],
    "formants": [
        "F1frequency_sma3nz_amean",
        "F1frequency_sma3nz_stddevNorm",
        "F1bandwidth_sma3nz_amean",
        "F1bandwidth_sma3nz_stddevNorm",
        "F1amplitudeLogRelF0_sma3nz_amean",
        "F1amplitudeLogRelF0_sma3nz_stddevNorm",
        "F2frequency_sma3nz_amean",
        "F2frequency_sma3nz_stddevNorm",
        "F2bandwidth_sma3nz_amean",
        "F2bandwidth_sma3nz_stddevNorm",
        "F2amplitudeLogRelF0_sma3nz_amean",
        "F2amplitudeLogRelF0_sma3nz_stddevNorm",
        "F3frequency_sma3nz_amean",
        "F3frequency_sma3nz_stddevNorm",
        "F3bandwidth_sma3nz_amean",
        "F3bandwidth_sma3nz_stddevNorm",
        "F3amplitudeLogRelF0_sma3nz_amean",
        "F3amplitudeLogRelF0_sma3nz_stddevNorm",
    ],
    "mfcc": [
        "mfcc1_sma3_amean",
        "mfcc1_sma3_stddevNorm",
        "mfcc2_sma3_amean",
        "mfcc2_sma3_stddevNorm",
        "mfcc3_sma3_amean",
        "mfcc3_sma3_stddevNorm",
        "mfcc4_sma3_amean",
        "mfcc4_sma3_stddevNorm",
    ],
    "timing": [
        "VoicedSegmentsPerSec",
        "MeanVoicedSegmentLengthSec",
        "StddevVoicedSegmentLengthSec",
        "MeanUnvoicedSegmentLength",
        "StddevUnvoicedSegmentLength",
    ],
    "vocal_dynamics": [
        "alphaRatioV_sma3nz_amean",
        "alphaRatioV_sma3nz_stddevNorm",
        "hammarbergIndexV_sma3nz_amean",
        "hammarbergIndexV_sma3nz_stddevNorm",
        "slopeV0-500_sma3nz_amean",
        "slopeV0-500_sma3nz_stddevNorm",
        "slopeV500-1500_sma3nz_amean",
        "slopeV500-1500_sma3nz_stddevNorm",
        "spectralFluxV_sma3nz_amean",
        "spectralFluxV_sma3nz_stddevNorm",
        "mfcc1V_sma3nz_amean",
        "mfcc1V_sma3nz_stddevNorm",
        "mfcc2V_sma3nz_amean",
        "mfcc2V_sma3nz_stddevNorm",
        "mfcc3V_sma3nz_amean",
        "mfcc3V_sma3nz_stddevNorm",
        "mfcc4V_sma3nz_amean",
        "mfcc4V_sma3nz_stddevNorm",
    ],
    "unvoiced_dynamics": [
        "alphaRatioUV_sma3nz_amean",
        "hammarbergIndexUV_sma3nz_amean",
        "slopeUV0-500_sma3nz_amean",
        "slopeUV500-1500_sma3nz_amean",
        "spectralFluxUV_sma3nz_amean",
    ],
    "intensity": ["loudnessPeaksPerSec", "equivalentSoundLevel_dBp"],
    "top10": [
        "loudness_sma3_meanRisingSlope",
        "loudness_sma3_pctlrange0-2",
        "loudness_sma3_percentile80_0",
        "slopeUV0-500_sma3nz_amean",
        "mfcc3_sma3_amean",
        "equivalentSoundLevel_dBp",
        "mfcc4V_sma3nz_stddevNorm",
        "loudness_sma3_stddevRisingSlope",
        "spectralFluxUV_sma3nz_amean",
        "slopeV0-500_sma3nz_amean",
    ],
    "vocal_dynamics": [
        "alphaRatio_sma3",
        "hammarbergIndex_sma3",
        "slope0-500_sma3",
        "slope500-1500_sma3",
        "spectralFlux_sma3",
    ],
    "loudness_lld": ["Loudness_sma3"],
    "mfcc_lld": ["mfcc1_sma3", "mfcc2_sma3", "mfcc3_sma3", "mfcc4_sma3"],
    # "pitch_lld": ["F0semitoneFrom27.5Hz_sma3nz"],
    "voice_quality_lld": [
        "jitterLocal_sma3nz",
        "shimmerLocaldB_sma3nz",
        "HNRdBACF_sma3nz",
    ],
    "harmonics": ["logRelF0-H1-H2_sma3nz", "logRelF0-H1-A3_sma3nz"],
    "formants_lld": [
        "F1frequency_sma3nz",
        "F1bandwidth_sma3nz",
        "F1amplitudeLogRelF0_sma3nz",
        "F2frequency_sma3nz",
        "F2bandwidth_sma3nz",
        "F2amplitudeLogRelF0_sma3nz",
        "F3frequency_sma3nz",
        "F3bandwidth_sma3nz",
        "F3amplitudeLogRelF0_sma3nz",
    ],
    "mfcc_all": [f"MFCC_{i+1}" for i in range(13)],
    "pos": [
        "POS_$",
        "POS_''",
        "POS_,",
        "POS_-LRB-",
        "POS_-RRB-",
        "POS_.",
        "POS_:",
        "POS_ADD",
        "POS_AFX",
        "POS_CC",
        "POS_CD",
        "POS_DT",
        "POS_EX",
        "POS_FW",
        "POS_HYPH",
        "POS_IN",
        "POS_JJ",
        "POS_JJR",
        "POS_JJS",
        "POS_LS",
        "POS_MD",
        "POS_NFP",
        "POS_NN",
        "POS_NNP",
        "POS_NNPS",
        "POS_NNS",
        "POS_PDT",
        "POS_POS",
        "POS_PRP",
        "POS_PRP$",
        "POS_RB",
        "POS_RBR",
        "POS_RBS",
        "POS_RP",
        "POS_SYM",
        "POS_TO",
        "POS_UH",
        "POS_VB",
        "POS_VBD",
        "POS_VBG",
        "POS_VBN",
        "POS_VBP",
        "POS_VBZ",
        "POS_WDT",
        "POS_WP",
        "POS_WP$",
        "POS_WRB",
        "POS_XX",
        "POS_``",
    ],
    "liwc": [
        "funct",
        "pronoun",
        "ppron",
        "i",
        "we",
        "you",
        "shehe",
        "they",
        "ipron",
        "article",
        "verb",
        "auxverb",
        "past",
        "present",
        "future",
        "adverb",
        "preps",
        "conj",
        "negate",
        "quant",
        "number",
        "swear",
        "social",
        "family",
        "friend",
        "humans",
        "affect",
        "posemo",
        "negemo",
        "anx",
        "anger",
        "sad",
        "cogmech",
        "insight",
        "cause",
        "discrep",
        "tentat",
        "certain",
        "inhib",
        "incl",
        "excl",
        "percept",
        "see",
        "hear",
        "feel",
        "bio",
        "body",
        "health",
        "sexual",
        "ingest",
        "relativ",
        "motion",
        "space",
        "time",
        "work",
        "achieve",
        "leisure",
        "home",
        "money",
        "relig",
        "death",
        "assent",
        "nonfl",
        "filler",
        "total_words",
        "six_letter_words",
        "dict_words",
        "punctuation",
        "emoticons",
    ],
    "ngrams": [],
    "tf-idf": [
        "to",
        "and",
        "of",
        "in",
        "that",
        "nigger",
        "is",
        "it",
        "oh",
        "for",
        "this",
        "im",
        "my",
        "we",
        "on",
        "me",
        "he",
        "they",
        "are",
        "your",
        "was",
        "what",
        "be",
        "dont",
        "know",
        "like",
        "its",
        "not",
        "so",
        "all",
        "no",
        "with",
        "have",
        "up",
        "but",
        "get",
        "youre",
        "fucking",
        "out",
        "just",
        "do",
        "white",
        "fuck",
        "people",
        "about",
        "in the",
        "go",
        "if",
        "yeah",
        "were",
    ],
}


def load_features_from_dict(features_dict: dict) -> pd.DataFrame:
    try:
        features_df = pd.DataFrame.from_dict(features_dict, orient="index").T
    except Exception as e:
        print(f"Error loading features from dict: {e}")
    if "file" in features_df.columns:
        features_df = features_df.drop(columns=["file"])
    return features_df


def load_lld_features(file_name: str) -> pd.DataFrame:
    file_path = f"{output_dir}/{file_name}"
    features = opensmile_extractor.extract(
        file_path,
        opensmile_set=opensmile.FeatureSet.eGeMAPSv02,
        opensmile_feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    return features.drop(columns=["file"])


def select_features(features: pd.DataFrame, selected_groups: list) -> pd.DataFrame:
    selected_columns = []
    for feature_group in selected_groups:
        selected_columns.extend(FEATURE_GROUPS.get(feature_group, []))
    selected_columns = [col.replace(".", "_") for col in selected_columns]
    features.columns = [col.replace(".", "_") for col in features.columns]
    return features[selected_columns]


def select_stats_features(
    features: pd.DataFrame, selected_groups: list
) -> pd.DataFrame:
    selected_columns = []
    for feature_group in selected_groups:
        selected_columns.extend(FEATURE_GROUPS.get(feature_group, []))
    selected_columns = [col.replace(".", "_") for col in selected_columns]
    feature_names = features["Feature"].str.replace(".", "_", regex=False)
    return features[feature_names.isin(selected_columns)]


def normalize_features(selected_features_df: pd.DataFrame, scaler=None) -> pd.DataFrame:
    if len(selected_features_df) == 1 or scaler is None:
        return selected_features_df
    else:
        normalized_features = pd.DataFrame(
            scaler.transform(selected_features_df), columns=selected_features_df.columns
        )
        return normalized_features


def normalize_precomputed_means(
    filtered_stats: pd.DataFrame,
    all_means: list,
    hate_means: list,
    non_hate_means: list,
):
    means_df = pd.DataFrame(
        {
            "All_Mean": all_means,
            "Hate_Mean": hate_means,
            "Non_Hate_Mean": non_hate_means,
        },
        index=filtered_stats["Feature"],
    )

    scaler = StandardScaler()

    normalized_means = pd.DataFrame(
        scaler.fit_transform(means_df), columns=means_df.columns
    )

    all_means_normalized = normalized_means["All_Mean"].values
    hate_means_normalized = normalized_means["Hate_Mean"].values
    non_hate_means_normalized = normalized_means["Non_Hate_Mean"].values

    return all_means_normalized, hate_means_normalized, non_hate_means_normalized


def generate_radar_plot(
    selected_file,
    selected_file_values,
    categories,
    hate_means_normalized,
    non_hate_means_normalized,
):
    import pandas as pd
    import plotly.graph_objs as go

    data = pd.DataFrame(
        {
            "Feature": categories,
            "Selected File": selected_file_values,
            "Hate Mean": hate_means_normalized,
            "Non Hate Mean": non_hate_means_normalized,
        }
    )

    data = data.sort_values(by="Selected File", ascending=False)

    categories_sorted = data["Feature"].tolist()
    selected_file_values_sorted = data["Selected File"].tolist()
    hate_means_normalized_sorted = data["Hate Mean"].tolist()
    non_hate_means_normalized_sorted = data["Non Hate Mean"].tolist()

    categories_sorted += [categories_sorted[0]]
    selected_file_values_sorted += [selected_file_values_sorted[0]]
    hate_means_normalized_sorted += [hate_means_normalized_sorted[0]]
    non_hate_means_normalized_sorted += [non_hate_means_normalized_sorted[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=selected_file_values_sorted,
            theta=categories_sorted,
            fill="toself",
            name=f"Selected File: {selected_file}",
            marker_color="blue",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=hate_means_normalized_sorted,
            theta=categories_sorted,
            fill="toself",
            name="Hate Mean (Standardized)",
            marker_color="red",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=non_hate_means_normalized_sorted,
            theta=categories_sorted,
            fill="toself",
            name="Non Hate Mean (Standardized)",
            marker_color="green",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
        showlegend=True,
        title="Standardized Functional Features Radar Plot (Ordered by Z-Score)",
    )

    return fig


def generate_bar_plot(
    selected_file,
    selected_file_values,
    categories,
    hate_means_normalized,
    non_hate_means_normalized,
    num_features=None,
):
    unique_data = {
        category: (v1, v2, v3)
        for category, v1, v2, v3 in zip(
            categories,
            selected_file_values,
            hate_means_normalized,
            non_hate_means_normalized,
        )
    }
    categories = list(unique_data.keys())
    selected_file_values, hate_means_normalized, non_hate_means_normalized = zip(
        *unique_data.values()
    )

    if num_features is not None:
        categories = categories[:num_features]
        selected_file_values = selected_file_values[:num_features]
        hate_means_normalized = hate_means_normalized[:num_features]
        non_hate_means_normalized = non_hate_means_normalized[:num_features]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=categories, y=selected_file_values, name=f"Selected File: {selected_file}"
        )
    )

    fig.add_trace(
        go.Bar(x=categories, y=hate_means_normalized, name="Hate Mean (Standardized)")
    )
    fig.add_trace(
        go.Bar(
            x=categories,
            y=non_hate_means_normalized,
            name="Non Hate Mean (Standardized)",
        )
    )

    fig.update_layout(
        barmode="group",
        title="Standardized Functional Features Comparison",
        xaxis=dict(
            title="Features",
            tickangle=-45,
            automargin=True,
            rangeslider=dict(
                visible=True,
                thickness=0.1,
                bgcolor="lightgray",
                bordercolor="black",
                borderwidth=1,
            ),
        ),
        yaxis=dict(title="Z-Score"),
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
    )

    return fig


def process_features(selected_file, features_data, selected_features, database_dir):
    if (
        features_data is None
        or not selected_features
        or selected_file not in features_data
    ):
        return None, {"display": "none"}

    file_data = features_data[selected_file]

    features = load_features_from_dict(file_data)
    selected_features_df = select_features(features, selected_features)

    selected_feature_names = [
        f
        for feature in selected_features
        for f in (FEATURE_GROUPS[feature] if feature in FEATURE_GROUPS else [feature])
    ]
    stats_df = query_feature_statistics(selected_feature_names, database_dir)
    all_means, all_stds = extract_mean_and_std_values(
        stats_df, "All", selected_feature_names
    )
    hate_means, hate_stds = extract_mean_and_std_values(
        stats_df, "Hate", selected_feature_names
    )
    non_hate_means, non_hate_stds = extract_mean_and_std_values(
        stats_df, "Non Hate", selected_feature_names
    )

    categories = selected_features_df.columns.tolist()
    selected_feature_raw = selected_features_df.iloc[0].values

    selected_file_values = (selected_feature_raw - all_means) / all_stds
    hate_means_normalized = (hate_means - all_means) / all_stds
    non_hate_means_normalized = (non_hate_means - all_means) / all_stds

    selected_file_values = np.nan_to_num(selected_file_values)
    hate_means_normalized = np.nan_to_num(hate_means_normalized)
    non_hate_means_normalized = np.nan_to_num(non_hate_means_normalized)

    return (
        selected_file_values,
        categories,
        hate_means_normalized,
        non_hate_means_normalized,
        selected_feature_raw,
        hate_means,
        non_hate_means,
    ), {"display": "block"}


def initialize_liwc_extractor(language="English"):
    script_dir = Path(__file__).resolve().parent

    project_root = script_dir.parent

    liwc_path = (
        project_root
        / "csl-speech-feature-extractor"
        / "speech_feature_extractor"
        / "data"
        / f"LIWC_{language}.dic"
    )
    return LIWCExtractor(liwc_path)


def initialize_pos_tagger():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Model 'en_core_web_sm' not found. Downloading it now...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    return nlp


def extract_pos_features(text, nlp_model):
    doc = nlp_model(text)
    pos_tags = list(nlp_model.get_pipe("tagger").labels)
    total_tokens = len(doc)
    tag_counts = {tag: 0 for tag in pos_tags}
    for token in doc:
        if token.tag_ in tag_counts:
            tag_counts[token.tag_] += 1
    pos_freqs = {
        f"POS_{tag}": count / total_tokens if total_tokens > 0 else 0
        for tag, count in tag_counts.items()
    }
    return pos_freqs


def extract_liwc_features(text, liwc_extractor):
    try:
        if not text or not isinstance(text, str) or text.strip() == "":
            warnings.warn(
                "Input text is empty or invalid. Skipping LIWC feature extraction."
            )
            return {}

        liwc_features = liwc_extractor.extract_from_transcript(text)

        if not liwc_features or all(value == 0 for value in liwc_features):
            warnings.warn("LIWC feature extraction returned empty or zero values.")
            return {}

        feature_names = liwc_extractor.get_category_names()
        liwc_features_dict = dict(zip(feature_names, liwc_features))

        return liwc_features_dict

    except Exception as e:
        warnings.warn(f"Error during LIWC feature extraction: {e}")
        return {}


def normalize_transcript(transcript, lowercase=True, remove_punctuation=True):
    if lowercase:
        transcript = transcript.lower()
    if remove_punctuation:
        transcript = transcript.translate(str.maketrans("", "", string.punctuation))
    return transcript


def extract_mfcc(audio_path, n_mfcc=13):
    try:
        full_file_path = f"{output_dir}/{audio_path}"

        y, sr = librosa.load(full_file_path, sr=None)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        mfcc_mean = mfccs.mean(axis=1)

        mfcc_df = pd.DataFrame(
            [mfcc_mean], columns=[f"MFCC_{i+1}" for i in range(n_mfcc)]
        )
        mfcc_dict = mfcc_df.iloc[0].to_dict()

        return mfcc_dict

    except Exception as e:
        print(f"[Extracting MFCC] Error loading file {audio_path}: {e}")
        return None


def extract_genf_acoustic_features(audio_path, opensmile_extractor):
    try:
        full_file_path = f"{output_dir}/{audio_path}"

        features = opensmile_extractor.extract(
            full_file_path,
            opensmile_set=opensmile.FeatureSet.eGeMAPSv02,
            opensmile_feature_level=opensmile.FeatureLevel.Functionals,
        )

        if "file" in features.columns:
            features.drop(columns=["file"], inplace=True)

        features = features.astype("float64")

        return features
    except Exception as e:
        print(f"[Extracting egemaps v2] Error loading file {audio_path}: {e}")
        return None


def extract_ngram_tfidf_dataframe(transcript, ngram_range=(1, 3)):
    try:
        if not transcript or not isinstance(transcript, str):
            raise ValueError("Input transcript must be a non-empty string.")

        with open(tfidf_vectorizer_path, "rb") as vec_file:
            vectorizer = pickle.load(vec_file)

        with open(top_50_tfidf_features_path, "rb") as features_file:
            top_50_features = pickle.load(features_file)

        tfidf_matrix = vectorizer.transform([transcript])
        ngram_feature_names = vectorizer.get_feature_names_out()

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=ngram_feature_names,
        )
        tfidf_dict = tfidf_df.iloc[0].to_dict()

        top_50_tfidf_dict = {
            feature: tfidf_dict.get(feature, 0.0) for feature in top_50_features
        }

        return top_50_tfidf_dict

    except Exception as e:
        print(f"Error extracting TF-IDF features: {e}")
        return None


def expand_features(selected_features, feature_groups):
    if not selected_features:
        raise ValueError("[L-Bar] No features selected.")

    expanded_features = [
        sub_feature
        for feature in selected_features
        for sub_feature in feature_groups.get(feature, [feature])
    ]
    return expanded_features


class FusionModel(nn.Module):
    def __init__(
        self,
        acoustic_dim,
        linguistic_dim,
        hidden_dim=128,
        scaler_acoustic=None,
        scaler_linguistic=None,
        acoustic_cols=None,
        linguistic_cols=None,
    ):
        super().__init__()
        self.acoustic_fc = nn.Linear(acoustic_dim, hidden_dim)
        self.linguistic_fc = nn.Linear(linguistic_dim, hidden_dim)
        self.fusion_fc = nn.Linear(2 * hidden_dim, 1)
        self.relu = nn.ReLU()

        self.scaler_acoustic = scaler_acoustic
        self.scaler_linguistic = scaler_linguistic
        self.acoustic_cols = acoustic_cols
        self.linguistic_cols = linguistic_cols

    def forward(self, acoustic_feats, linguistic_feats):
        a_out = self.relu(self.acoustic_fc(acoustic_feats))
        l_out = self.relu(self.linguistic_fc(linguistic_feats))
        fused = torch.cat([a_out, l_out], dim=1)
        logits = self.fusion_fc(fused).squeeze(1)
        return logits


def preprocess_for_prediction(
    acoustic_input: dict,
    linguistic_input: dict,
    model: FusionModel,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acoustic_dict_filtered = {
        col: acoustic_input[col] for col in model.acoustic_cols if col in acoustic_input
    }
    linguistic_dict_filtered = {
        col: linguistic_input[col]
        for col in model.linguistic_cols
        if col in linguistic_input
    }

    acoustic_array = np.array(
        [acoustic_dict_filtered[col] for col in model.acoustic_cols], dtype=np.float32
    )
    linguistic_array = np.array(
        [linguistic_dict_filtered[col] for col in model.linguistic_cols],
        dtype=np.float32,
    )

    if model.scaler_acoustic is not None:
        acoustic_array = model.scaler_acoustic.transform(acoustic_array.reshape(1, -1))
    if model.scaler_linguistic is not None:
        linguistic_array = model.scaler_linguistic.transform(
            linguistic_array.reshape(1, -1)
        )

    acoustic_tensor = torch.tensor(acoustic_array, dtype=torch.float32, device=device)
    linguistic_tensor = torch.tensor(
        linguistic_array, dtype=torch.float32, device=device
    )

    return acoustic_tensor, linguistic_tensor


def predict_sample(model: FusionModel, acoustic_input: dict, linguistic_input: dict):
    acoustic_tensor, linguistic_tensor = preprocess_for_prediction(
        acoustic_input, linguistic_input, model
    )

    model.eval()
    with torch.no_grad():
        logits = model(acoustic_tensor, linguistic_tensor)
        prob = torch.sigmoid(logits).item()
        prediction = 1 if prob > 0.5 else 0

    return prediction, prob
