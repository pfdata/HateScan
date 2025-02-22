import opensmile
import pandas as pd


def extract(
    audio_paths,
    output_path=None,
    opensmile_set=opensmile.FeatureSet.ComParE_2016,
    opensmile_feature_level=opensmile.FeatureLevel.Functionals,
):
    """
    Extract opensmile acoustic features from one or multiple audio files and optionally save them to a CSV file.

    Args:
        audio_paths (str or list of str): Path(s) to the audio file(s) from which features will be extracted.
        output_path (str, optional): Path to save the extracted features to a CSV file. If None, the features won't be saved.
        opensmile_set (opensmile.FeatureSet): The set of features to be used for extraction.
        opensmile_feature_level (opensmile.FeatureLevel): The level of features to be extracted.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted opensmile features from all provided audio files.
    """
    smile = opensmile.Smile(
        feature_set=opensmile_set, feature_level=opensmile_feature_level
    )

    if isinstance(audio_paths, str):
        audio_paths = [audio_paths]

    all_features = []

    for audio_path in audio_paths:
        try:
            opensmile_features = smile.process_file(audio_path)
            opensmile_features = opensmile_features.reset_index()
            opensmile_features = opensmile_features.drop(columns=["start", "end"])
            all_features.append(opensmile_features)
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")
            continue

    if all_features:
        concatenated_features = pd.concat(all_features, ignore_index=True)
    else:
        concatenated_features = pd.DataFrame()

    if output_path is not None:
        concatenated_features.to_csv(output_path, index=False)

    return concatenated_features
