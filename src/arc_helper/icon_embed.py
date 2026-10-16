"""
Image-embedding match for item icon identity (phase 3, step 3).

A slot crop and every catalog icon go through the same crop step as the hash
baseline (icon_match.art_picture, then icon_match.art_square: the art box on a
flat backdrop, padded to a square). The square is embedded with DINOv2-small
(Apache-2.0, Meta) running on the CPU through onnxruntime, and the crop is
matched to the catalog by cosine similarity.

The model is the ONNX export published as onnx-community/dinov2-small on
Hugging Face, pinned to one revision and checked by SHA-256. It is downloaded
on first use into the model cache (cache/models/dinov2-small/ under the app
folder, gitignored), never committed. The export has one output, the last
layer's tokens after the final layer norm. The embedding pools them as
POOLING says (see Embedder) and is L2 normalised.

onnxruntime is the optional 'icons' extra and is imported only when the first
picture is embedded. Gallery embeddings are cached to disk as an .npz keyed by
the model revision and every icon file's size and modification time, so a
gallery is embedded once and rebuilt only when an icon changes.
"""

import hashlib
import threading
import time
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import requests
from PIL import Image

from .config import APP_DIR
from .config import logger
from .icon_match import TOP_N
from .icon_match import art_picture
from .icon_match import art_square
from .icon_match import icon_region

MODEL_REPO = "onnx-community/dinov2-small"
MODEL_REVISION = "8b1f705a3a7f6f062f6bdd21986c1583d3ef105d"
MODEL_FILE = "onnx/model.onnx"
MODEL_SHA256 = "f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}"
MODEL_DIR = APP_DIR / "cache" / "models" / "dinov2-small"
INPUT_SIDE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BATCH = 16
# "cls": the class token. "cls+mean": the class token and the mean of the
# patch tokens, each L2 normalised, side by side.
POOLINGS = ("cls", "cls+mean")
POOLING = "cls"
DOWNLOAD_CHUNK = 1 << 20


class EmbedCandidate(NamedTuple):
    """One ranked gallery entry: its item id, display name and cosine similarity."""

    item_id: str
    name: str
    similarity: float


class ModelUnavailableError(RuntimeError):
    """The model file is missing and cannot be downloaded, or fails its checksum."""


def model_path(model_dir: Path = MODEL_DIR, *, download: bool = True) -> Path:
    """Return the cached ONNX model, downloading and checking it the first time."""
    path = model_dir / f"model-{MODEL_REVISION[:12]}.onnx"
    if path.is_file():
        return path
    if not download:
        msg = f"no DINOv2 model at {path}"
        raise ModelUnavailableError(msg)
    model_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    digest = hashlib.sha256()
    logger.info("Downloading the icon embedding model from %s", MODEL_URL)
    try:
        with requests.get(MODEL_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with partial.open("wb") as f:
                for chunk in resp.iter_content(DOWNLOAD_CHUNK):
                    digest.update(chunk)
                    f.write(chunk)
    except requests.RequestException as e:
        partial.unlink(missing_ok=True)
        msg = f"cannot download the DINOv2 model: {e}"
        raise ModelUnavailableError(msg) from e
    if digest.hexdigest() != MODEL_SHA256:
        partial.unlink(missing_ok=True)
        msg = f"DINOv2 model checksum mismatch ({digest.hexdigest()})"
        raise ModelUnavailableError(msg)
    partial.replace(path)
    return path


class Embedder:
    """DINOv2-small on onnxruntime, loaded on the first embed() call."""

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        *,
        download: bool = True,
        pooling: str = POOLING,
    ):
        if pooling not in POOLINGS:
            msg = f"unknown pooling {pooling!r}, use one of {POOLINGS}"
            raise ValueError(msg)
        self._model_dir = model_dir
        self._download = download
        self.pooling = pooling
        self._session = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def _load(self):
        with self._load_lock:
            return self._load_locked()

    def _load_locked(self):
        if self._session is None:
            path = model_path(self._model_dir, download=self._download)
            import onnxruntime as ort  # optional 'icons' extra

            options = ort.SessionOptions()
            options.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(path),
                options,
                providers=["CPUExecutionProvider"],
            )
        return self._session

    def embed(self, squares: Sequence[np.ndarray]) -> np.ndarray:
        """Return one L2-normalised row per RGB square picture (any side)."""
        if not squares:
            return np.zeros((0, 0), dtype=np.float32)
        session = self._load()
        rows = []
        for start in range(0, len(squares), BATCH):
            batch = np.stack([_prepare(s) for s in squares[start : start + BATCH]])
            (hidden,) = session.run(["last_hidden_state"], {"pixel_values": batch})
            rows.append(self._pool(hidden))
        vectors = np.concatenate(rows).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def warm(self) -> None:
        """Load the model and run one blank embed, so a later embed() is quick."""
        self.embed([np.zeros((INPUT_SIDE, INPUT_SIDE, 3), dtype=np.uint8)])

    def _pool(self, hidden: np.ndarray) -> np.ndarray:
        cls = _unit(hidden[:, 0])
        if self.pooling == "cls":
            return cls
        return np.hstack([cls, _unit(hidden[:, 1:].mean(axis=1))])


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _prepare(square: np.ndarray) -> np.ndarray:
    """Resize an RGB square to INPUT_SIDE and normalise it as DINOv2 expects."""
    if square.shape[:2] != (INPUT_SIDE, INPUT_SIDE):
        resized = Image.fromarray(square).resize(
            (INPUT_SIDE, INPUT_SIDE), Image.Resampling.BICUBIC
        )
        square = np.asarray(resized)
    pixels = (square.astype(np.float32) / 255.0 - MEAN) / STD
    return pixels.transpose(2, 0, 1)


def crop_square(crop: np.ndarray) -> np.ndarray:
    """The model's view of a slot crop: art_picture() then art_square()."""
    return art_square(art_picture(crop), INPUT_SIDE)


def icon_square(icon: Image.Image) -> np.ndarray:
    """The model's view of a catalog icon, through the same art_square() step."""
    return art_square(icon_region(icon), INPUT_SIDE)


@dataclass(frozen=True)
class EmbedGallery:
    """The catalog icon embeddings, one L2-normalised row per item."""

    item_ids: tuple[str, ...]
    names: tuple[str, ...]
    vectors: np.ndarray

    def __len__(self) -> int:
        return len(self.item_ids)


def _gallery_key(icons: Mapping[str, tuple[str, Path]], pooling: str) -> str:
    digest = hashlib.sha256(f"{MODEL_REVISION}|{pooling}".encode())
    for item_id, (name, path) in sorted(icons.items()):
        stat = path.stat()
        digest.update(f"{item_id}|{name}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def build_gallery(
    icons: Mapping[str, tuple[str, Path]],
    embedder: Embedder,
    cache_file: Path | None = None,
) -> EmbedGallery:
    """Embed every catalog icon. icons maps item id to (display name, PNG path).

    With cache_file, a cached gallery for exactly these icon files is loaded
    instead of embedding, and a fresh one is written there.
    """
    ids = tuple(icons)
    names = tuple(icons[i][0] for i in ids)
    if not ids:
        return EmbedGallery((), (), np.zeros((0, 0), dtype=np.float32))
    key = _gallery_key(icons, embedder.pooling)
    if cache_file is not None and cache_file.is_file():
        with np.load(cache_file) as cached:
            if str(cached["key"]) == key:
                order = {item_id: n for n, item_id in enumerate(cached["ids"])}
                rows = cached["vectors"][[order[i] for i in ids]]
                return EmbedGallery(ids, names, rows)
    squares = []
    for item_id in ids:
        with Image.open(icons[item_id][1]) as image:
            squares.append(icon_square(image))
    vectors = embedder.embed(squares)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as f:
            np.savez(f, key=np.array(key), ids=np.array(ids), vectors=vectors)
    return EmbedGallery(ids, names, vectors)


_default: Embedder | None = None


def default_embedder() -> Embedder:
    """The shared Embedder on MODEL_DIR, created on first use."""
    global _default  # noqa: PLW0603 - one lazily loaded model per process
    if _default is None:
        _default = Embedder()
    return _default


def match(
    crop: np.ndarray,
    gallery: EmbedGallery,
    embedder: Embedder | None = None,
    top: int = TOP_N,
) -> list[EmbedCandidate]:
    """Return the top catalog items for a slot crop, most similar first.

    crop is an RGB array of the slot inside its glow ring. Without an
    embedder the shared default_embedder() is used.
    """
    if len(gallery) == 0:
        return []
    vector = (embedder or default_embedder()).embed([crop_square(crop)])[0]
    similarity = gallery.vectors @ vector
    order = np.argsort(-similarity, kind="stable")[:top]
    return [
        EmbedCandidate(gallery.item_ids[i], gallery.names[i], float(similarity[i]))
        for i in order
    ]


def timed_match(
    crop: np.ndarray,
    gallery: EmbedGallery,
    embedder: Embedder | None = None,
    top: int = TOP_N,
) -> tuple[list[EmbedCandidate], float]:
    """match() with its duration in seconds."""
    start = time.perf_counter()
    found = match(crop, gallery, embedder, top)
    return found, time.perf_counter() - start
