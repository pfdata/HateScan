footer_style = {
    "textAlign": "center",
    "padding": "5px",
    "backgroundColor": "#f1f1f1",
    "position": "fixed",
    "left": "0",
    "bottom": "0",
    "width": "100%",
    "fontSize": "12px",
}

footer_content_style = {
    "display": "flex",
    "justifyContent": "center",
    "alignItems": "center",
}

footer_text_style = {"margin": "0 10px", "fontSize": "12px"}

colors = {
    "primary": "#000000",  # Black for main elements
    "accent": "#E74C3C",  # Red for emphasis
    "background": "#F5F5F5",  # Light gray background
    "text": "#000000",  # Black text
    "button": "#E74C3C",  # Red buttons
    "disabled": "#CCCCCC",  # Gray for disabled state
}

button_style_base = {
    "padding": "10px 15px",
    "border": "none",
    "borderRadius": "5px",
    "fontSize": "14px",
    "fontWeight": "500",
    "cursor": "pointer",
    "transition": "all 0.3s ease",
    "width": "100%",
    "marginBottom": "10px",
}

button_style_disabled = {
    **button_style_base,
    "backgroundColor": colors["disabled"],
    "color": "#666666",
    "cursor": "not-allowed",
}

button_style_active = {
    **button_style_base,
    "backgroundColor": colors["button"],
    "color": "white",
    "&:hover": {
        "backgroundColor": "#C0392B",  # Darker red on hover
    },
}

dropdown_style = {
    "width": "100%",
    "marginBottom": "15px",
    "backgroundColor": "white",
    "color": colors["text"],
    "border": f"1px solid {colors['accent']}",  # Changed to accent color
    "borderRadius": "8px",  # Increased border radius
    "boxShadow": "0 2px 4px rgba(0,0,0,0.15)",  # Enhanced shadow
    "padding": "8px",  # Added padding
    "fontSize": "18px",  # Added font size
    "fontWeight": "500",  # Added font weight
    "cursor": "pointer",  # Added cursor style
    "transition": "all 0.2s ease",  # Added transition
    "&:hover": {
        "borderColor": colors["button"],
        "boxShadow": "0 3px 6px rgba(0,0,0,0.2)",
    },
}

# Additional styles from left_sidebar.py
heading_style = {
    "marginBottom": "20px",
    "fontSize": "24px",
    "fontWeight": "bold",
}

upload_heading_style = {
    "marginBottom": "10px",
    "color": colors["text"],
    "fontSize": "18px",
}

upload_icon_style = {
    "fontSize": "32px",
    "marginBottom": "15px",
    "color": colors["button"],
}

upload_text_style = {
    "fontSize": "16px",
    "marginBottom": "5px",
}

upload_link_style = {
    "color": colors["button"],
    "textDecoration": "underline",
    "fontSize": "16px",
    "fontWeight": "bold",
    "marginBottom": "10px",
}

upload_support_text_style = {
    "fontSize": "13px",
    "color": colors["text"],
    "opacity": "0.8",
    "margin": "0",
}

upload_container_style = {
    "width": "100%",
    "height": "150px",
    "borderWidth": "2px",
    "borderStyle": "dashed",
    "borderRadius": "12px",
    "textAlign": "center",
    "marginBottom": "25px",
    "borderColor": colors["button"],
    "backgroundColor": "#f8f9fa",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "center",
    "alignItems": "center",
    "cursor": "pointer",
    "transition": "all 0.3s ease",
    "&:hover": {
        "borderColor": colors["accent"],
        "backgroundColor": "#ffffff",
        "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
    },
}

file_select_heading_style = {
    "marginBottom": "15px",
    "color": colors["text"],
    "fontSize": "16px",
}

output_message_style = {
    "fontSize": "14px",
    "color": colors["accent"],
}

output_container_style = {
    "marginTop": "20px",
    "padding": "15px",
    "backgroundColor": "white",
    "borderRadius": "8px",
    "border": f'1px solid {colors["accent"]}',
    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
}

prediction_heading_style = {
    "marginBottom": "15px",
    "color": colors["text"],
    "fontSize": "18px",
}

model_heading_style = {
    "color": colors["text"],
    "fontSize": "16px",
}

prediction_text_style = {
    "marginBottom": "10px",
    "fontSize": "14px",
}

prediction_container_style = {
    "padding": "20px",
    "backgroundColor": "white",
    "border": f'1px solid {colors["primary"]}',
    "borderRadius": "8px",
    "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
}

sidebar_container_style = {
    "width": "300px",
    "display": "inline-block",
    "verticalAlign": "top",
    "padding": "20px",
    "backgroundColor": colors["background"],
    "height": "100vh",
    "position": "fixed",
    "left": "0",
    "top": "0",
    "overflowY": "auto",
    "zIndex": "1000",
    "boxShadow": "2px 0 10px rgba(0,0,0,0.1)",
}
