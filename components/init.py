# Third-party imports
from dash import html

# Project-specific imports
from components.footer import create_footer
from components.left_sidebar import create_left_sidebar
from components.main_content import create_main_content
from components.right_sidebar import create_right_sidebar


def create_layout():
    return html.Div(
        [
            create_left_sidebar(),
            create_main_content(),
            create_right_sidebar(),
            create_footer(),
        ]
    )
