"""
Blueprint icon cache for ARLO.

Icons are the arctracker.io item renders (256 px PNG with transparency),
downloaded once on first use into the icon cache directory and never
committed.
"""

import re
from pathlib import Path

import requests

from .config import get_settings
from .config import logger

ICON_URL = "https://cdn.arctracker.io/items/v2/{item_id}.png"
ICON_TIMEOUT = 15
USER_AGENT = "ARLO/1.0 (Arc Raiders Loot Overlay - https://github.com/Soygen/ARLO)"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_ITEM_ID = re.compile(r"[a-z0-9_]+")


class IconDownloadError(RuntimeError):
    """A blueprint icon could not be downloaded or saved."""


def blueprint_icon_path(item_id: str, cache_dir: Path | None = None) -> Path:
    """Path of the cached icon for an arctracker item id, downloading it once.

    cache_dir defaults to the configured icon_cache_dir. Raises
    IconDownloadError when the id is malformed, the download fails or the
    answer is not a PNG.
    """
    if not _ITEM_ID.fullmatch(item_id or ""):
        msg = f"Invalid blueprint item id {item_id!r}"
        raise IconDownloadError(msg)
    cache_dir = Path(cache_dir or get_settings().icon_cache_dir)
    path = cache_dir / f"{item_id}.png"
    if path.is_file():
        return path

    url = ICON_URL.format(item_id=item_id)
    try:
        resp = requests.get(
            url, timeout=ICON_TIMEOUT, headers={"User-Agent": USER_AGENT}
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        msg = f"Could not download the icon for {item_id} from {url}: {e}"
        raise IconDownloadError(msg) from e
    if not resp.content.startswith(PNG_SIGNATURE):
        msg = f"The icon for {item_id} from {url} is not a PNG"
        raise IconDownloadError(msg)

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        partial.write_bytes(resp.content)
        partial.replace(path)
    except OSError as e:
        msg = f"Could not save the icon for {item_id} to {path}: {e}"
        raise IconDownloadError(msg) from e
    logger.debug(f"Cached blueprint icon {path}")
    return path
