from dash import html
from utils.common_paths import app_name, app_version, author_name, contact_email
from components.style import footer_style, footer_content_style, footer_text_style


def create_footer():
    return html.Footer(
        children=[
            html.Div(
                [
                    html.Span(
                        f"{app_name} v{app_version}",
                        style=footer_text_style,
                    ),
                    html.Span(
                        f"Developed by {author_name}",
                        style=footer_text_style,
                    ),
                    html.Span(
                        f"Email: {contact_email}",
                        style=footer_text_style,
                    ),
                ],
                style=footer_content_style,
            )
        ],
        style=footer_style,
    )
