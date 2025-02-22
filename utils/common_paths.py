from pathlib import Path

current_script_path = Path().resolve()
base_dir = current_script_path

speech_feature_extractor = (
    base_dir / "csl-speech-feature-extractor" / "speech_feature_extractor"
)
data_dir = base_dir / "data"
output_dir = base_dir / "output"
tmp_dir = base_dir / "tmp"
model_path = base_dir / "models"

data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
tmp_dir.mkdir(parents=True, exist_ok=True)
model_path.mkdir(parents=True, exist_ok=True)
speech_feature_extractor.parent.mkdir(parents=True, exist_ok=True)

annotations_file = data_dir / "hatemm_annotation.csv"
linguistic_stats = data_dir / "linguistic_stats.csv"
acoustic_stats = data_dir / "acoustic_stats.csv"

tfidf_vectorizer_path = model_path / "tfidf_vectorizer.pkl"
count_vectorizer_path = model_path / "count_vectorizer.pkl"
top_50_tfidf_features_path = model_path / "tfidf_top_50_features.pkl"

best_linguistic_model = "linguistic_ensemble_features_166.pkl"
best_acoustic_model = "acoustic_random_forest_features_101.pkl"
mutimodal = "fusion_model.pkl"

app_name = "HateScan"
app_version = "1.0.0"
author_name = "Pablo Freitas Santos"
contact_email = "pablo2@uni-bremen.de"

__all__ = [
    "speech_feature_extractor",
    "data_dir",
    "annotations_file",
    "output_dir",
    "tmp_dir",
    "base_dir",
    "model_path",
    "tfidf_vectorizer_path",
    "count_vectorizer_path",
    "truncated_svd_path",
    "top_50_tfidf_features_path",
    "top_50_ngram_features_path",
    "best_linguistic_model",
    "best_acoustic_model",
    "acoustic_stats",
    "linguistic_stats",
]
