import os
from utils.common_paths import output_dir

def get_file_list():
    """
    Retrieve the list of files in the output folder.

    Returns:
        list: A list of file names in the output folder, or a message indicating
              the folder is empty or not found.
    """
    try:
        files = os.listdir(output_dir)
        return files if files else ["No files available"]
    except FileNotFoundError:
        return ["Output folder not found"]
