from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


def image_file_to_data_url(path: str | Path) -> str:
    """Read a local image file and return a browser-displayable data URL."""
    path = Path(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

