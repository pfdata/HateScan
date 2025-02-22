import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import pandas as pd
import os

class SpeakerSeparator:
    """
    This class provides methods for speaker diarization, splitting audio by speaker,
    and finding the longest audio file among a list of files. It utilizes the `pyannote.audio`
    library for diarization.
    """

    def __init__(self, auth_token, device='cuda'):
        """
        Initializes the SpeakerSeparator with a pre-trained diarization model and sets the computation device.

        Args:
            auth_token (str): Authentication token for accessing the pre-trained model.
            device (str): The computation device ('cuda' or 'cpu') for model inference.
        """
        self.auth_token = auth_token
        self.device = device
        # Load the pre-trained diarization model
        self.pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=self.auth_token)        
        # Move the pipeline to the specified device
        self.pipeline.to(torch.device(self.device))

    def batch_diarize(self, audio_paths, output_dir, speakers=1):
        """
        Performs speaker diarization on multiple audio files and saves the outputs in the specified directory.

        Args:
            audio_paths (list of str): Paths to the input audio files.
            output_dir (str): Directory where the diarization outputs in RTTM format will be saved.
            speakers (int, optional): The expected number of speakers in the audio. Defaults to 1.

        Returns:
            list of str: Paths to the RTTM files generated for each audio file.
        """
        diarization_paths = []
        for audio_path, output_path in zip(audio_paths, output_dir):
            diarization_paths.append(self.diarize(audio_path, output_dir, speakers))
        return diarization_paths

    def diarize(self, audio_path, output_dir, speakers=1):
        """
        Performs speaker diarization on a single audio file and saves the result in RTTM format.

        Args:
            audio_path (str): Path to the input audio file.
            output_dir (str): Directory to save the diarization output in RTTM format.
            speakers (int, optional): The expected number of speakers in the audio. Defaults to 1.

        Returns:
            str: Path to the generated RTTM file.
        """
        # Apply the pipeline to the audio file
        diarization = self.pipeline(audio_path, num_speakers=speakers)

        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/{os.path.splitext(os.path.basename(audio_path))[0]}_diarization.rttm'

        # Save the diarization result in RTTM format
        with open(output_path, "w") as rttm_file:
            diarization.write_rttm(rttm_file)
            return output_path

    def batch_split_audio_by_speaker(self, audio_paths, rttm_paths, output_dir):
        """
        Splits multiple audio files into separate files for each speaker based on RTTM files.

        Args:
            audio_paths (list of str): Paths to the original audio files.
            rttm_paths (list of str): Paths to the corresponding RTTM files with diarization information.
            output_dir (str): Directory where the split audio files for each speaker will be saved.

        Returns:
            list of list of str: Nested list containing paths to the audio files created for each speaker.
        """
        list_of_speaker_files = []
        for audio_path, rttm_path in zip(audio_paths, rttm_paths):
            list_of_speaker_files.append(self.split_audio_by_speaker(audio_path, rttm_path, output_dir))
        return list_of_speaker_files

    def split_audio_by_speaker(self, audio_path, rttm_path, output_dir):
        """
        Splits a single audio file into separate files for each speaker based on an RTTM file.

        Args:
            audio_path (str): Path to the original audio file.
            rttm_path (str): Path to the RTTM file with diarization information.
            output_dir (str): Directory to save the split audio files for each speaker.

        Returns:
            list of str: Paths to the audio files created for each speaker.
        """
        audio = AudioSegment.from_file(audio_path)
        columns = ['Type', 'FileID', 'ChannelNumber', 'StartTime', 'Duration', 'Orthography', 'SpeakerType', 'SpeakerID', 'Confidence', 'SignalLookahead']
        rttm_df = pd.read_csv(rttm_path, sep=' ', names=columns, usecols=['FileID', 'StartTime', 'Duration', 'SpeakerID'])
        
        speaker_files = []

        for speaker_id, group in rttm_df.groupby('SpeakerID'):
            speaker_audio = AudioSegment.silent(duration=0)  # Initialize a silent segment
            
            for _, row in group.iterrows():
                start_time_ms, end_time_ms = int(row['StartTime'] * 1000), int((row['StartTime'] + row['Duration']) * 1000) # int conversion was at the wrong place
                segment = audio[start_time_ms:end_time_ms]
                speaker_audio += segment  # Concatenate segments for the current speaker
            
            output_path = f"{output_dir}/{row['FileID']}_{speaker_id.lower()}.wav"
            speaker_audio.export(output_path, format="wav")
            speaker_files.append(output_path)

        return speaker_files

    def batch_find_longest_audio_file(self, list_of_audio_paths):
        """
        Finds the longest audio file in each sublist of audio file paths.

        Args:
            list_of_audio_paths (list of list of str): Nested list where each sublist contains paths to audio files.

        Returns:
            list of str: List of paths to the longest audio file in each sublist.
        """
        longest_audio_file_tuple = []
        for audio_paths in list_of_audio_paths:
            longest_audio_file_tuple.append(self.find_longest_audio_file(audio_paths))
        return longest_audio_file_tuple

    def find_longest_audio_file(self, audio_paths):
        """
        Determines the longest audio file from a list of audio file paths.

        Args:
            audio_paths (list of str): Paths to the audio files to be compared.

        Returns:
            str: Path to the longest audio file.
        """
        longest_duration, longest_audio_path = 0, None

        for path in audio_paths:
            audio = AudioSegment.from_file(path)
            duration = len(audio) / 1000.0  # Convert milliseconds to seconds
            
            if duration > longest_duration:
                longest_duration, longest_audio_path = duration, path

        return longest_audio_path