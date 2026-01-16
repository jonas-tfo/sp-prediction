"""Model weights download and caching."""

import urllib.request
from pathlib import Path

from .config import Config


def get_cache_dir() -> Path:
    """Get the cache directory for storing model weights."""
    cache_dir = Path.home() / ".cache" / "sandwichsp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_weights_path() -> Path:
    """Get the path to the cached weights file."""
    return get_cache_dir() / Config.WEIGHTS_FILENAME


def download_weights(url: str = None, force: bool = False) -> Path:
    """Download model weights if not already cached.

    Args:
        url: URL to download from. Defaults to Config.WEIGHTS_URL.
        force: If True, re-download even if cached.

    Returns:
        Path to the weights file.

    Raises:
        RuntimeError: If download fails.
    """
    if url is None:
        url = Config.WEIGHTS_URL

    weights_path = get_weights_path()

    if weights_path.exists() and not force:
        return weights_path

    print(f"Downloading model weights from {url}...")
    print(f"Saving to {weights_path}")

    try:
        urllib.request.urlretrieve(url, weights_path, _download_progress)
        print("\nDownload complete.")
    except Exception as e:
        if weights_path.exists():
            weights_path.unlink()
        raise RuntimeError(f"Failed to download weights: {e}") from e

    return weights_path


def _download_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Progress callback for urllib.request.urlretrieve."""
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\rProgress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)


def ensure_weights(url: str = None) -> Path:
    """Ensure weights are available, downloading if necessary.

    Args:
        url: Optional custom URL for weights.

    Returns:
        Path to the weights file.
    """
    weights_path = get_weights_path()

    if weights_path.exists():
        return weights_path

    return download_weights(url=url)
