import whisper
import os
import mimetypes
import string
import pandas as pd

def is_audio_file(filepath):
    """
    Determines if the specified file is an audio file based on its MIME type.

    Args:
        filepath (str): Path to the file to check.

    Returns:
        bool: True if the file is an audio file, False otherwise.
    """
    if not os.path.isfile(filepath):
        return False

    mime_type, _ = mimetypes.guess_type(filepath)
    return mime_type and mime_type.startswith('audio')

def transcribe_audio(audio_paths, language='german', whisper_model='medium', device='cuda', output_path=None):
    """
    Transcribes one or multiple audio files to text using OpenAI's Whisper model and returns a DataFrame.
    Optionally saves the DataFrame to a specified path.

    Args:
        audio_paths (str or list of str): Path(s) to the audio file(s) to transcribe.
        language (str, optional): Language hint for the transcription model.
        whisper_model (str, optional): The Whisper model size to use. Defaults to 'medium'.
        device (str, optional): The device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.
        output_path (str, optional): Path to save the DataFrame. If None, the DataFrame is not saved.

    Returns:
        pd.DataFrame: A DataFrame with columns ['audio_path', 'transcript'].

    Raises:
        Exception: If the provided audio_path is not an audio file.
    """
    # Ensure audio_paths is a list
    if isinstance(audio_paths, str):
        audio_paths = [audio_paths]

    # Initialize list to store transcriptions
    transcriptions = []

    # Load the Whisper model
    model = whisper.load_model(whisper_model, device=device)

    # Iterate over audio files and transcribe each
    for audio_path in audio_paths:
        if not is_audio_file(audio_path):
            raise Exception(f'Audio path {audio_path} is not a audio file.')

        # Perform the transcription
        result = model.transcribe(audio_path, language=language)

        # Append the file name and transcript to the list
        transcriptions.append([audio_path, result["text"]])

    # Create a DataFrame from the transcriptions list
    transcription_df = pd.DataFrame(transcriptions, columns=['audio_path', 'transcript'])

    # Save the DataFrame if output_path is provided
    if output_path is not None:
        transcription_df.to_csv(output_path, index=False)

    return transcription_df

def normalize_transcripts(df, lowercase=True, remove_punctuation=True, output_path=None):
    """
    Normalizes the 'transcript' column in a DataFrame by converting to lowercase
    and optionally removing punctuation for each transcript. Optionally saves the
    DataFrame to a specified path.

    Args:
        df (pd.DataFrame): DataFrame containing the 'transcript' column to normalize.
        lowercase (bool, optional): Whether to convert the transcripts to lowercase. Defaults to True.
        remove_punctuation (bool, optional): Whether to remove punctuation from the transcripts. Defaults to False.
        output_path (str, optional): Path to save the DataFrame. If None, the DataFrame is not saved.

    Returns:
        pd.DataFrame: The DataFrame with normalized transcripts.
    """
    def normalize_transcript(transcript):
        """
        Helper function to normalize a single transcript.
        """
        if lowercase:
            transcript = transcript.lower()
        if remove_punctuation:
            transcript = transcript.translate(str.maketrans('', '', string.punctuation))
        return transcript
    
    # Apply the normalization to each transcript in the DataFrame
    df['transcript'] = df['transcript'].apply(normalize_transcript)
    
    # Save the DataFrame if an output path is provided
    if output_path is not None:
        df.to_csv(output_path, index=False)
    
    return df
