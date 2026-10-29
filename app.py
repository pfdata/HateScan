# Third-party imports
import dash
from flask import send_from_directory
import torch.multiprocessing as mp

# Local imports
from callbacks import (
    audio_callbacks,
    charts_callbacks,
    file_callbacks,
    prediction_callbacks,
)
from components.init import create_layout
from utils.common_paths import output_dir
from utils.features import FusionModel

app = dash.Dash(__name__)

server = app.server

app.layout = create_layout()

charts_callbacks.register_callbacks(app)
file_callbacks.register_callbacks(app)
audio_callbacks.register_audio_callbacks(app)
prediction_callbacks.register_audio_callbacks(app)


@server.route("/files/<path:filename>")
def serve_file(filename):
    return send_from_directory(output_dir, filename)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    app.run(debug=False)
