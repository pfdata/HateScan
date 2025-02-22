# Standard library imports
import ast
import os
from pathlib import Path
from typing import List

# Third-party imports
import pandas as pd
from pandas import DataFrame
from pydub import AudioSegment

# Constants
AUDIO_SUFFIX = "_audio_16kHz_mono.wav"  # Standard suffix for the processed files


def gather_media_paths(directory: Path, extensions: List[str]) -> List[Path]:
    """
    Gathers all media files with specified extensions from a given directory.

    Parameters:
    directory (Path): The directory to search for media files.
    extensions (List[str]): A list of file extensions to look for (e.g., ['.mp4']).

    Returns:
    List[Path]: A list of paths to the media files found.
    """
    return [path for path in directory.iterdir() if path.suffix in extensions]


def load_annotation_file(csv_file):
    """Loads the annotation CSV file into a pandas DataFrame."""
    return pd.read_csv(csv_file)


def parse_hate_snippet(snippet_str):
    """Parses the hate snippet string into a list of time intervals."""
    if pd.isna(snippet_str) or snippet_str == "":
        return []
    try:
        return ast.literal_eval(snippet_str)
    except (SyntaxError, ValueError):
        return []


def get_processed_audio_file(video_file_name):
    """Converts a video file name to the corresponding audio file name."""
    actual_filename = os.path.splitext(video_file_name)[0]
    processed_audio_file = f"{actual_filename}{AUDIO_SUFFIX}"
    return processed_audio_file


def extract_and_merge_snippets(audio_metadata):
    """
    Extracts the hate snippets from the audio file and merges them into a single audio file.

    Parameters:
    audio_metadata (dict): A dictionary containing metadata about the audio file.

    Returns:
    AudioSegment: An AudioSegment object containing the merged snippets, or None if no snippets are found.
    """
    # Load the audio file
    audio_file_path = audio_metadata["file_path"]
    audio = AudioSegment.from_wav(audio_file_path)

    # Get the hate snippets from metadata
    hate_snippets = audio_metadata.get("hate_snippets", [])

    # If there are no hate snippets, return None or an empty audio
    if not hate_snippets:
        print(f"No hate snippets found in {audio_metadata['audio_file']}.")
        return None  # or you could return AudioSegment.empty() to return an empty audio file

    # Initialize an empty audio segment to hold the merged snippets
    merged_audio = AudioSegment.empty()

    # Extract and merge the hate snippets
    for snippet in hate_snippets:
        start_time_str, end_time_str = snippet
        start_time = time_to_milliseconds(start_time_str)
        end_time = time_to_milliseconds(end_time_str)

        # Extract the snippet from the audio
        snippet_audio = audio[start_time:end_time]

        # Append the snippet to the merged audio
        merged_audio += snippet_audio

    # Return the merged audio
    return merged_audio


def time_to_milliseconds(time_str):
    """
    Converts a time string (in format 'hh:mm:ss' or 'mm:ss') into milliseconds.

    Parameters:
    time_str (str): The time string to convert.

    Returns:
    int: Time in milliseconds.
    """
    parts = time_str.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
    elif len(parts) == 2:
        hours = 0
        minutes = int(parts[0])
        seconds = int(parts[1])
    else:
        raise ValueError("Invalid time format")

    return (hours * 3600 + minutes * 60 + seconds) * 1000


def save_merged_snippets(merged_audio, output_file):
    """
    Saves the merged audio snippets to a file.

    Parameters:
    merged_audio (AudioSegment): The merged audio snippets.
    output_file (str): The path to the output file where the merged snippets will be saved.
    """
    if merged_audio is not None:
        merged_audio.export(output_file, format="wav")
        print(f"Merged audio saved to {output_file}")
    else:
        print(f"No audio to save for {output_file}")
