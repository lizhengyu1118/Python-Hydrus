# -*- coding: utf-8 -*-
"""
Lightweight caching utilities for visualization scripts.

Each cache is tied to the CSV it represents and stores a companion metadata
JSON file that records a hash of the data-defining parameters. Subsequent runs
can compare current parameters to the stored hash to decide whether the cached
CSV is still valid.
"""

from __future__ import annotations

import json
import hashlib
import os
from typing import Any, Dict, Optional

import pandas as pd

META_SUFFIX = ".meta.json"


def _normalize_param(value: Any) -> Any:
    """Convert arbitrary objects into JSON-serializable primitives."""
    if isinstance(value, dict):
        return {str(k): _normalize_param(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_param(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return value


def _build_signature(script_name: str, params: Dict[str, Any]) -> str:
    normalized = {
        "script": script_name,
        "params": _normalize_param(params or {})
    }
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _meta_path(csv_path: str) -> str:
    return csv_path + META_SUFFIX


def load_cached_dataframe(csv_path: str, script_name: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Return DataFrame if both CSV and metadata hash match current params."""
    if not os.path.exists(csv_path):
        return None

    metadata_path = _meta_path(csv_path)
    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        return None

    current_sig = _build_signature(script_name, params)
    if meta.get("signature") != current_sig:
        return None

    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def save_dataframe_with_signature(df: pd.DataFrame, csv_path: str, script_name: str, params: Dict[str, Any]) -> None:
    """Persist DataFrame and companion metadata signature."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    metadata = {
        "signature": _build_signature(script_name, params),
        "script": script_name,
        "params": _normalize_param(params or {}),
        "csv_path": os.path.abspath(csv_path)
    }
    metadata_path = _meta_path(csv_path)
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
