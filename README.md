# HateScan

## Overview

HateScan is an interactive dashboard designed to detect hate speech from audio and text files. The dashboard allows users to upload media, analyze speech characteristics, and visualize extracted features in real-time. The system utilizes OpenSmile to extract both functional and low-level descriptors (LLD) features.

## Prerequisites

To set up and run this project, you need:

- **uv**: Install it using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
- **Git**: Required to initialize the speech feature extractor submodule. Access to the private CSL GitLab repository is required; configure your GitLab credentials before initializing it.
- **Python 3.15**: uv can download this version automatically when creating the environment.
- **FFmpeg**: Required by Whisper and pydub to decode audio. Install it using your operating system's package manager and ensure `ffmpeg` is on your `PATH`.

## Installation

Run the following commands from the repository root:

1. Initialize the speech feature extractor submodule:

   ```bash
   git submodule update --init --recursive
   ```

   The submodule URL currently references `CI_PERSONAL_ACCESS_TOKEN`; ensure your GitLab authentication is configured for your environment. Do not commit credentials.

2. Create a virtual environment:

   ```bash
   uv venv --python 3.15
   ```

3. Compile the dependencies for your local checkout:

   ```bash
   uv pip compile --python 3.15 --output-file requirements.txt requirements.in
   ```

   The requirements include the speech feature extractor as an editable dependency using a relative path. Recompile from the repository root when changing dependencies.

4. Install the dependencies into `.venv`:

   ```bash
   uv pip install -r requirements.txt
   ```

This repository uses requirements files rather than a `pyproject.toml` or `uv.lock`, so use `uv pip` for dependency management. After editing `requirements.in`, repeat the compile and install commands above.

## Running the Application

Start the app from the repository root using the virtual environment:

```bash
uv run --no-project python app.py
```

Open your browser and navigate to <http://127.0.0.1:8050/>.

## Authors

- Pablo Freitas [pablo2@uni-bremen.de](pablo2@uni-bremen.de)

## Generating Documentation

1. Install the documentation dependencies from the repository root:

   ```bash
   uv pip install -r requirements_docs.txt
   ```

2. Build the documentation:

   ```bash
   uv run --no-project python -m sphinx -b html docs/source docs/build/html
   ```

3. Open `docs/build/html/index.html` in your web browser:

   ```bash
   # macOS
   open docs/build/html/index.html

   # Linux
   xdg-open docs/build/html/index.html

   # On Windows
   start docs/build/html/index.html
   ```

The documentation provides detailed information about the project's modules, classes and functions with examples of usage.

## Acknowledgments

- HateMM Database
- Rathi Rammohan and Jordan Behrendt

## License

This project is licensed under the GNU General Public License v3.0.

### GNU General Public License Version 3, 29 June 2007

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.en.html) for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
