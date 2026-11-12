"""PyCam ``rois.json`` files: finding them, reading trap centres, and writing lattices / pasted coordinates (no Qt).

Camera-pixel centres are converted to focal-plane positions with ``ROI_JSON_UM_PER_CAMERA_PIXEL`` around the image
centre (ImgResX/2, ImgResY/2); axial ``z`` is not stored in the file.
"""

from __future__ import annotations

import json
import os

import numpy as np

from gui.parameters import param
from gui.constants import ROI_JSON_UM_PER_CAMERA_PIXEL, SYSTEM_MAGNIFICATION, TWEEZER_ROIS_DIR


def _um_per_px() -> float:
    um_per_px = float(ROI_JSON_UM_PER_CAMERA_PIXEL)
    if um_per_px <= 0:
        um_per_px = 1.0
    return um_per_px


def resolveRoisPath(base: str | None, tweezer_dir: str = TWEEZER_ROIS_DIR) -> str | None:
    """Resolve path to PyCam ``rois.json`` (``Tweezer_ROIs/`` or an existing file path)."""
    base = (base or '').strip()
    if not base:
        return os.path.join(tweezer_dir, 'rois.json')
    if os.path.isfile(base):
        return base
    cand = os.path.join(tweezer_dir, base)
    if os.path.isfile(cand):
        return cand
    if not base.lower().endswith('.json'):
        p_json = os.path.join(tweezer_dir, base + '.json')
        if os.path.isfile(p_json):
            return p_json
    return None


def readTrapCoordsM(path: str, verbose: bool = False) -> np.ndarray | None:
    """Trap positions (N, 3) in metres (x, y, z = 0) from the ``centers`` of a ``rois.json``; None if it has none."""
    with open(path, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    centers = doc.get('centers')
    if not isinstance(centers, list) or len(centers) < 1:
        if verbose:
            print('rois.json has no valid "centers" array.')
        return None

    coords_m = centersToCoordsM(centers)
    if coords_m is None and verbose:
        print('rois.json "centers" produced no valid points.')
    return coords_m


def centersToCoordsM(centers: list) -> np.ndarray | None:
    """Camera-pixel centres [[x, y], ...] → trap positions (N, 3) in metres (x, y, z = 0); None if none are valid."""
    um_per_px = _um_per_px()
    cx = 0.5 * float(param.ImgResX)
    cy = 0.5 * float(param.ImgResY)

    rows = []
    for c in centers:
        if not isinstance(c, (list, tuple)) or len(c) < 2:
            continue
        px_x, px_y = float(c[0]), float(c[1])
        x_um = (px_x - cx) * um_per_px
        y_um = (px_y - cy) * um_per_px
        rows.append((x_um * 1e-6, y_um * 1e-6, 0.0))
    if not rows:
        return None

    return np.asarray(rows, dtype=np.float64)


def trapsPosToCentersPx(traps_pos: np.ndarray) -> np.ndarray:
    """Trap positions ``traps_pos`` (N, 4): index, y, x, z [m] → (N, 2) camera-pixel centres [x, y]; inverse of centersToCoordsM."""
    um_per_px = _um_per_px()
    cx = 0.5 * float(param.ImgResX)
    cy = 0.5 * float(param.ImgResY)
    x_um = traps_pos[:, 2] * 1e6
    y_um = traps_pos[:, 1] * 1e6
    x_px = x_um / um_per_px + cx
    y_px = y_um / um_per_px + cy
    return np.column_stack([x_px, y_px])


def readCentersPx(path: str) -> np.ndarray:
    """(N, 2) camera-pixel centres of a ``rois.json`` (raises OSError / ValueError / TypeError if unreadable)."""
    with open(path, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    return np.asarray([c[:2] for c in doc.get('centers', []) if isinstance(c, (list, tuple)) and len(c) >= 2], dtype=np.float64)


def latticeCenters(arr_m: np.ndarray, sx_um: float, sy_um: float) -> tuple[list[list[float]], int]:
    """Array coordinates (m) → sub-pixel camera centres around (ImgResX/2, ImgResY/2), and the ROI side in pixels."""
    um_per_px = _um_per_px()
    cx = 0.5 * float(param.ImgResX)
    cy = 0.5 * float(param.ImgResY)
    x_um = arr_m[:, 0] * 1e6
    y_um = arr_m[:, 1] * 1e6
    centers = [[round(float(x_um[i] / um_per_px + cx), 4), round(float(y_um[i] / um_per_px + cy), 4)] for i in range(x_um.shape[0])]
    side = max(3, int(round(min(float(sx_um), float(sy_um)) / um_per_px)))
    return centers, side


def pastedCenters(coords: np.ndarray, in_um: bool) -> tuple[list[list[float]], int]:
    """Pasted (N, 2|3) coordinates → camera centres (µm focal plane converted like the array generator) and ROI side."""
    xy = coords[:, :2]
    if in_um:
        um_per_px = _um_per_px()
        center = np.array([0.5 * float(param.ImgResX), 0.5 * float(param.ImgResY)])
        xy = xy * float(SYSTEM_MAGNIFICATION) / um_per_px + center
    centers = [[round(float(x), 4), round(float(y), 4)] for x, y in xy]

    if xy.shape[0] > 1:
        d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
        np.fill_diagonal(d, np.inf)
        side = max(3, int(round(d.min())))
    else:
        side = 3
    return centers, side


def writeRoisJson(path: str, centers: list[list[float]], roi_side_px: int, angle_deg: float = 0.0) -> None:
    """Write a PyCam ``rois.json`` (schema version 2) with pixel ``centers`` (int or float)."""
    doc = {
        'format_version': 2,
        'centers': centers,
        'roi_sizes': [int(roi_side_px)],
        'x_com': 0.0,
        'y_com': 0.0,
        'angle_deg': float(angle_deg),
        'rescale': 1.0,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write('\n')
