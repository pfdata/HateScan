# HateScan

## Overview

HateScan is an interactive dashboard designed to detect hate speech from audio and text files. The dashboard allows users to upload media, analyze speech characteristics, and visualize extracted features in real-time. The system utilizes OpenSmile to extract both functional and low-level descriptors (LLD) features.

## Pre-requirements

To set up and run this project, ensure the following tools are installed:

- **Python**: Install Python 3.x (the project has been tested with `>=3.8`).
- **pip**: Ensure Python's package installer (`pip`) is installed.
- **pip-tools**: Used for managing dependency versions.

### Installation Steps

1. Install `pip-tools`:
   ```bash
   pip install pip-tools
   ```
2. Compile the dependencies into requirements.txt:

   ```bash
   pip-compile requirements.in
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the app:

```bash
python app.py
```

2. Open your browser and navigate to: http://127.0.0.1:8050/

## Authors

- Pablo Freitas [pablo2@uni-bremne.de](pablo2@uni-bremen.de)

### Generating Documentation

1. Navigate to the docs directory:

   ```bash
   cd docs
   ```

2. Build the documentation:

   ```bash
   make html
   ```

3. View the documentation by opening `docs/build/html/index.html` in your web browser:

   ```bash
   # On Linux/Mac
   open build/html/index.html

   # On Windows
   start build/html/index.html
   ```

The documentation provides detailed information about the project's modules, classes and functions with examples of usage.

## Acknolegments

- HateMM Database
- Rathi Rammohan and Jordan Behrendt

## License

This project is licensed under the GNU General Public License v3.0.

### GNU General Public License Version 3, 29 June 2007

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.en.html) for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
