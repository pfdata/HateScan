import os
from pydub import AudioSegment

def process_audio(audio_paths, output_dir, new_sample_rate=16000, new_channels=1):
    """
    Processes one or multiple audio files by resampling to a new sample rate and changing the number of channels.
    Returns the paths of the processed audio files.

    Args:
        audio_paths (str or list of str): Path(s) to the input audio file(s).
        output_dir (str): Directory where the processed audio files will be saved.
        new_sample_rate (int): The new sample rate in Hz.
        new_channels (int): The number of channels for the output file (1 for mono, 2 for stereo).

    Returns:
        list of str: Paths to the processed audio files.

    Raises:
        ValueError: If an unsupported number of channels is specified.
    """
    if isinstance(audio_paths, str):
        audio_paths = [audio_paths]  # Convert to list if only one path is provided

    processed_paths = []  # Store paths of processed audio files

    for audio_path in audio_paths:
        # Load the audio file
        audio = AudioSegment.from_file(audio_path)
        
        # Resample the audio
        resampled_audio = audio.set_frame_rate(new_sample_rate)
        
        # Change the number of channels
        if new_channels not in [1, 2]:
            raise ValueError("Unsupported number of channels. Only mono (1) and stereo (2) are supported.")
        processed_audio = resampled_audio.set_channels(new_channels)

        # Create output_path
        os.makedirs(output_dir, exist_ok=True)      
        output_file_name = f"{os.path.splitext(os.path.basename(audio_path))[0]}_{int(new_sample_rate/1000)}kHz_{'mono' if new_channels == 1 else 'stereo'}.wav"
        output_path = os.path.join(output_dir, output_file_name)
        
        # Export the processed audio
        processed_audio.export(output_path, format="wav")
        processed_paths.append(output_path)

    return processed_paths
