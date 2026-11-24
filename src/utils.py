"""
Module: src.utils
Description: Utility helpers for artifact serialization, reproducibility, logging,
             hashing, and plotting support.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np


def get_logger(name: str = "SAL_EC") -> logging.Logger:
    """
    Returns a structured console and file logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


logger = get_logger("SAL_EC_UTILS")


def set_seed(seed: int = 42) -> None:
    """
    Sets deterministic random seeds across standard random, numpy, and PyTorch (if present).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_artifact(obj: Any, filepath: Union[str, Path]) -> Path:
    """
    Serializes a Python object (model, pipeline, transformer) to disk using joblib.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved artifact to: %s", path)
    return path


def load_artifact(filepath: Union[str, Path]) -> Any:
    """
    Loads a serialized joblib artifact from disk.
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found at: {path}")
    obj = joblib.load(path)
    logger.info("Loaded artifact from: %s", path)
    return obj


def save_json(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2) -> Path:
    """
    Serializes a dictionary to formatted JSON.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(val: Any) -> Any:
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, (np.floating, float)):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, dict):
            return {k: _sanitize(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_sanitize(v) for v in val]
        return str(val)

    clean_data = _sanitize(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=indent)
    logger.info("Saved JSON metrics to: %s", path)
    return path


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a JSON file as a dictionary.
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_plot(fig: Any, filepath: Union[str, Path], dpi: int = 300) -> Path:
    """
    Saves a Matplotlib or Plotly figure to disk.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Matplotlib figure
    if hasattr(fig, "savefig"):
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved Matplotlib figure to: %s", path)
    # Plotly figure
    elif hasattr(fig, "write_image"):
        try:
            fig.write_image(str(path))
            logger.info("Saved Plotly figure to: %s", path)
        except Exception:
            # Fallback to HTML if kaleido is not installed
            html_path = path.with_suffix(".html")
            fig.write_html(str(html_path))
            logger.info("Saved Plotly figure as HTML to: %s", html_path)
            return html_path
    elif hasattr(fig, "write_html"):
        fig.write_html(str(path))
        logger.info("Saved interactive HTML chart to: %s", path)
    else:
        raise TypeError(f"Unsupported figure type: {type(fig)}")

    return path


def hash_identifier(identifier: str, salt: str = "sal_ec_2026") -> str:
    """
    Generates a deterministic SHA-256 pseudonym hash for student identifiers.
    Note: Cryptographic hashing provides pseudonymization, not absolute anonymization.
    """
    combined = f"{identifier}_{salt}".encode("utf-8")
    return hashlib.sha256(combined).hexdigest()[:12].upper()
