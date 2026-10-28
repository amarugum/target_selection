from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk, ImageDraw
except ImportError as exc:
    raise RuntimeError("Pillow is required. Install with: pip install pillow") from exc


TARGET_CLASSES = [
    "Typical",
    "Oblique-right",
    "Oblique-left",
    "Elongated",
    "Wide & short (flat-bottom)",
    "Rounded-bottom",
    "Pointed-bottom",
    "Near-parallel sidewall",
]

CLASS_TO_BOTTOM = {
    "Typical": "circle",
    "Oblique-right": "circle",
    "Oblique-left": "circle",
    "Elongated": "circle",
    "Wide & short (flat-bottom)": "flat",
    "Rounded-bottom": "ellipse",
    "Pointed-bottom": "pointed",
    "Near-parallel sidewall": "circle",
}

BOTTOM_LABELS = {
    "circle": "Circular arc",
    "ellipse": "Elliptical arc",
    "flat": "Flat",
    "pointed": "Pointed / V",
}

# Muted colors for up to three observed masks.
SOURCE_COLORS = [
    (82, 126, 173),
    (109, 154, 111),
    (183, 132, 87),
]


@dataclass
class ControlPoint:
    name: str
    x: float
    y: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive GUI for geometric idealization of observed target masks."
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mask", type=Path, action="append", default=[],
                   help="Optional observed mask/target.npy. Repeat up to 3 times.")
    p.add_argument("--channel", type=int, default=1,
                   help="Channel used when an input .npy is CxHxW.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--target-id", default="")
    p.add_argument("--target-class", choices=TARGET_CLASSES, default="Typical")
    p.add_argument(
        "--candidate-csv",
        type=Path,
        default=None,
        help=(
            "Optional final_candidates.csv from target_selection_gui. "
            "Rows are loaded from source_target_path and can be grouped into the 8 ideal target classes."
        ),
    )
    p.add_argument("--canvas-size", type=int, default=760)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mask(path: Path, channel: int, threshold: float) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float32)
        if arr.ndim == 3:
            if not 0 <= channel < arr.shape[0]:
                raise ValueError(
                    f"Channel {channel} unavailable in {path}; array shape={arr.shape}"
                )
            arr = arr[channel]
        elif arr.ndim != 2:
            raise ValueError(f"Expected 2-D or CxHxW array, got {arr.shape}: {path}")
        return (arr >= threshold).astype(np.uint8)

    if suffix in {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}:
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        return (arr >= 128).astype(np.uint8)

    raise ValueError(f"Unsupported mask format: {path}")


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    p = np.pad(m, 1, mode="constant", constant_values=False)
    up = p[:-2, 1:-1]
    dn = p[2:, 1:-1]
    lf = p[1:-1, :-2]
    rt = p[1:-1, 2:]
    interior = m & up & dn & lf & rt
    return (m & ~interior).astype(np.uint8)


def translate_x(mask: np.ndarray, dx: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask)
    if dx == 0:
        return mask.copy()
    if dx > 0:
        if dx < w:
            out[:, dx:] = mask[:, :w-dx]
    else:
        d = -dx
        if d < w:
            out[:, :w-d] = mask[:, d:]
    return out


def opening_center_x(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (mask.shape[1] - 1) / 2.0

    ymin, ymax = int(ys.min()), int(ys.max())
    depth = max(1, ymax - ymin + 1)
    y1 = min(ymax, int(round(ymin + 0.18 * depth)))

    centers = []
    for y in range(ymin, y1 + 1):
        xx = np.flatnonzero(mask[y])
        if xx.size:
            centers.append((float(xx.min()) + float(xx.max())) / 2.0)
    if centers:
        return float(np.median(centers))
    return float(np.mean(xs))


def mask_metrics(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {
            "area_px": 0,
            "area_fraction": 0.0,
            "centroid_x_px": float("nan"),
            "centroid_y_px": float("nan"),
            "opening_width_px": 0.0,
            "depth_px": 0.0,
        }

    ymin, ymax = int(ys.min()), int(ys.max())
    depth = ymax - ymin + 1
    widths = []
    for y in range(ymin, min(ymax, ymin + max(2, int(0.12 * depth))) + 1):
        xx = np.flatnonzero(mask[y])
        if xx.size:
            widths.append(float(xx.max() - xx.min() + 1))
    return {
        "area_px": int(mask.sum()),
        "area_fraction": float(mask.mean()),
        "centroid_x_px": float(xs.mean()),
        "centroid_y_px": float(ys.mean()),
        "opening_width_px": float(np.median(widths)) if widths else 0.0,
        "depth_px": float(depth),
    }


def iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    u = np.logical_or(aa, bb).sum()
    return 1.0 if u == 0 else float(np.logical_and(aa, bb).sum()) / float(u)


def circle_arc_through_points(
    b: np.ndarray, c: np.ndarray, d: np.ndarray, n: int = 120
) -> np.ndarray:
    """
    Return circular arc from B to D that passes through C.
    Falls back to a quadratic Bezier if the three points are nearly collinear.
    """
    x1, y1 = b
    x2, y2 = c
    x3, y3 = d
    den = 2.0 * (
        x1 * (y2 - y3)
        + x2 * (y3 - y1)
        + x3 * (y1 - y2)
    )
    if abs(den) < 1e-8:
        t = np.linspace(0, 1, n)
        pts = (
            ((1 - t) ** 2)[:, None] * b
            + (2 * (1 - t) * t)[:, None] * c
            + (t ** 2)[:, None] * d
        )
        return pts

    ux = (
        (x1*x1 + y1*y1) * (y2 - y3)
        + (x2*x2 + y2*y2) * (y3 - y1)
        + (x3*x3 + y3*y3) * (y1 - y2)
    ) / den
    uy = (
        (x1*x1 + y1*y1) * (x3 - x2)
        + (x2*x2 + y2*y2) * (x1 - x3)
        + (x3*x3 + y3*y3) * (x2 - x1)
    ) / den

    center = np.array([ux, uy], dtype=float)
    radius = float(np.linalg.norm(b - center))
    if radius < 1e-8:
        return np.vstack([b, d])

    def ang(p):
        v = p - center
        return math.atan2(v[1], v[0])

    a0 = ang(b)
    ac = ang(c)
    a1 = ang(d)

    def norm2pi(a):
        return a % (2 * math.pi)

    a0n, acn, a1n = map(norm2pi, (a0, ac, a1))
    ccw_total = (a1n - a0n) % (2 * math.pi)
    ccw_to_c = (acn - a0n) % (2 * math.pi)

    if ccw_to_c <= ccw_total + 1e-9:
        theta = a0 + np.linspace(0, ccw_total, n)
    else:
        cw_total = (a0n - a1n) % (2 * math.pi)
        theta = a0 - np.linspace(0, cw_total, n)

    return center + radius * np.column_stack([np.cos(theta), np.sin(theta)])


def ellipse_arc_from_shoulders_apex(
    b: np.ndarray, c: np.ndarray, d: np.ndarray, n: int = 120
) -> tuple[np.ndarray, np.ndarray]:
    """
    Semi-ellipse with B and D as chord endpoints and C controlling sag/depth.
    The major axis lies along B-D. C is projected onto the perpendicular axis.
    Returns (arc, effective_apex).
    """
    chord = d - b
    L = float(np.linalg.norm(chord))
    if L < 1e-8:
        return np.vstack([b, c, d]), c.copy()

    u = chord / L
    m = (b + d) / 2.0
    nvec = np.array([-u[1], u[0]], dtype=float)
    if np.dot(c - m, nvec) < 0:
        nvec = -nvec

    a = L / 2.0
    bsemi = abs(float(np.dot(c - m, nvec)))
    bsemi = max(bsemi, 1.0)

    theta = np.linspace(math.pi, 0.0, n)
    arc = (
        m
        + (a * np.cos(theta))[:, None] * u
        + (bsemi * np.sin(theta))[:, None] * nvec
    )
    apex = m + bsemi * nvec
    return arc, apex



def unit_vector(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(2, dtype=float)
    return v / n


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    ua = unit_vector(a)
    ub = unit_vector(b)
    if float(np.linalg.norm(ua)) < 1e-12 or float(np.linalg.norm(ub)) < 1e-12:
        return float("nan")
    c = float(np.clip(abs(np.dot(ua, ub)), 0.0, 1.0))
    return math.degrees(math.acos(c))


def circle_from_three_points(b: np.ndarray, c: np.ndarray, d: np.ndarray):
    x1, y1 = b
    x2, y2 = c
    x3, y3 = d
    den = 2.0 * (
        x1 * (y2 - y3)
        + x2 * (y3 - y1)
        + x3 * (y1 - y2)
    )
    if abs(den) < 1e-10:
        return None
    ux = (
        (x1*x1 + y1*y1) * (y2 - y3)
        + (x2*x2 + y2*y2) * (y3 - y1)
        + (x3*x3 + y3*y3) * (y1 - y2)
    ) / den
    uy = (
        (x1*x1 + y1*y1) * (x3 - x2)
        + (x2*x2 + y2*y2) * (x1 - x3)
        + (x3*x3 + y3*y3) * (x2 - x1)
    ) / den
    center = np.array([ux, uy], dtype=float)
    radius = float(np.linalg.norm(b - center))
    return center, radius


def line_inward_normal(top: np.ndarray, bottom: np.ndarray, side: str) -> np.ndarray:
    d = unit_vector(bottom - top)
    if side == "left":
        n = np.array([d[1], -d[0]], dtype=float)  # right side in image coordinates
    else:
        n = np.array([-d[1], d[0]], dtype=float)  # left side in image coordinates
    return unit_vector(n)


def tangent_circle_to_sidewalls_through_apex(
    tl: np.ndarray,
    tr: np.ndarray,
    ba: np.ndarray,
    old_bl: np.ndarray,
    old_br: np.ndarray,
):
    """Exact circle tangent to both sidewall lines and passing through BA.

    Returns (center, radius, qL, qR) or None.  qL/qR are tangency points.
    The candidate closest to the existing BL/BR is selected.
    """
    # Sidewall directions use the current lower handles to define the lines.
    nL = line_inward_normal(tl, old_bl, "left")
    nR = line_inward_normal(tr, old_br, "right")
    if np.linalg.norm(nL) < 1e-12 or np.linalg.norm(nR) < 1e-12:
        return None

    cL = -float(np.dot(nL, tl))
    cR = -float(np.dot(nR, tr))
    a = nL - nR
    b = cR - cL
    aa = float(np.dot(a, a))
    if aa < 1e-12:
        return None

    c0 = (b / aa) * a
    d = unit_vector(np.array([-a[1], a[0]], dtype=float))
    alpha = float(np.dot(nL, c0) + cL)
    beta = float(np.dot(nL, d))
    u = c0 - ba

    # ||u + t d||^2 = (alpha + beta t)^2
    A = 1.0 - beta * beta
    B = 2.0 * (float(np.dot(u, d)) - alpha * beta)
    C = float(np.dot(u, u)) - alpha * alpha

    roots = []
    if abs(A) < 1e-12:
        if abs(B) < 1e-12:
            return None
        roots = [-C / B]
    else:
        disc = B * B - 4.0 * A * C
        if disc < -1e-8:
            return None
        disc = max(0.0, disc)
        sd = math.sqrt(disc)
        roots = [(-B + sd) / (2.0 * A), (-B - sd) / (2.0 * A)]

    candidates = []
    for t in roots:
        center = c0 + t * d
        r = alpha + beta * t
        if r <= 1e-6:
            continue
        qL = center - r * nL
        qR = center - r * nR
        # Prefer the solution whose tangency points are near the current shoulders
        # and whose center is not below the bottom apex.
        score = float(np.linalg.norm(qL - old_bl) + np.linalg.norm(qR - old_br))
        if center[1] > ba[1]:
            score += 1000.0 + 10.0 * (center[1] - ba[1])
        if qL[1] < tl[1] - 5 or qR[1] < tr[1] - 5:
            score += 500.0
        candidates.append((score, center, float(r), qL, qR))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, center, r, qL, qR = candidates[0]
    return center, r, qL, qR


def conic_tangent_ellipse(
    bl: np.ndarray,
    ba: np.ndarray,
    br: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
    n: int = 180,
):
    """General ellipse through BL/BA/BR tangent to both sidewall directions.

    The conic is solved linearly from 3 point constraints + 2 tangent constraints.
    Returns (arc, coeffs) or None if the conic is not a real ellipse.
    Coefficients are [A,B,C,D,E,F] for Ax^2+Bxy+Cy^2+Dx+Ey+F=0.
    """
    def point_row(p):
        x, y = map(float, p)
        return [x*x, x*y, y*y, x, y, 1.0]

    def tangent_row(p, v):
        x, y = map(float, p)
        vx, vy = map(float, unit_vector(v))
        # gradient F dot tangent = 0
        # Fx = 2Ax + By + D; Fy = Bx + 2Cy + E
        return [
            2*x*vx,
            y*vx + x*vy,
            2*y*vy,
            vx,
            vy,
            0.0,
        ]

    M = np.asarray([
        point_row(bl),
        point_row(ba),
        point_row(br),
        tangent_row(bl, left_tangent),
        tangent_row(br, right_tangent),
    ], dtype=float)

    try:
        _, _, vh = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        return None
    coeff = vh[-1].astype(float)
    if not np.all(np.isfinite(coeff)):
        return None
    scale = float(np.linalg.norm(coeff[:3]))
    if scale < 1e-12:
        return None
    coeff /= scale
    A, B, C, D, E, F = coeff

    Q = np.array([[A, B/2.0], [B/2.0, C]], dtype=float)
    dvec = np.array([D, E], dtype=float)
    evals = np.linalg.eigvalsh(Q)
    if evals[0] * evals[1] <= 0:
        # Multiply by -1 cannot change indefiniteness.
        return None
    if evals[0] < 0 and evals[1] < 0:
        coeff = -coeff
        A, B, C, D, E, F = coeff
        Q = -Q
        dvec = -dvec

    try:
        center = -0.5 * np.linalg.solve(Q, dvec)
    except np.linalg.LinAlgError:
        return None

    Fc = float(center @ Q @ center + dvec @ center + F)
    if Fc >= -1e-10:
        return None

    evals, evecs = np.linalg.eigh(Q)
    if np.any(evals <= 1e-12):
        return None
    axes = np.sqrt((-Fc) / evals)
    if not np.all(np.isfinite(axes)) or np.any(axes <= 0):
        return None

    def param_angle(p):
        z = evecs.T @ (p - center)
        return math.atan2(z[1] / axes[1], z[0] / axes[0])

    a0 = param_angle(bl)
    ac = param_angle(ba)
    a1 = param_angle(br)

    def norm2pi(a):
        return a % (2 * math.pi)

    a0n, acn, a1n = map(norm2pi, (a0, ac, a1))
    ccw_total = (a1n - a0n) % (2 * math.pi)
    ccw_to_c = (acn - a0n) % (2 * math.pi)
    if ccw_to_c <= ccw_total + 1e-8:
        theta = a0 + np.linspace(0.0, ccw_total, n)
    else:
        cw_total = (a0n - a1n) % (2 * math.pi)
        theta = a0 - np.linspace(0.0, cw_total, n)

    local = np.column_stack([axes[0]*np.cos(theta), axes[1]*np.sin(theta)])
    arc = center + local @ evecs.T

    # Numerical sanity check that BA lies near the sampled branch.
    if float(np.min(np.linalg.norm(arc - ba[None, :], axis=1))) > 5.0:
        return None
    return arc, coeff


def tangent_mismatch_deg(
    points: dict[str, ControlPoint],
    bottom_type: str,
    tangent_ellipse: bool = False,
) -> tuple[float, float]:
    tl = np.array([points["TL"].x, points["TL"].y], dtype=float)
    bl = np.array([points["BL"].x, points["BL"].y], dtype=float)
    ba = np.array([points["BA"].x, points["BA"].y], dtype=float)
    br = np.array([points["BR"].x, points["BR"].y], dtype=float)
    tr = np.array([points["TR"].x, points["TR"].y], dtype=float)
    vL = bl - tl
    vR = br - tr

    if bottom_type == "circle":
        geo = circle_from_three_points(bl, ba, br)
        if geo is None:
            return float("nan"), float("nan")
        center, _ = geo
        # Circle tangent is perpendicular to radius.
        tL = np.array([-(bl-center)[1], (bl-center)[0]])
        tR = np.array([-(br-center)[1], (br-center)[0]])
        return angle_between_deg(vL, tL), angle_between_deg(vR, tR)

    if bottom_type == "ellipse":
        if tangent_ellipse:
            return 0.0, 0.0
        chord = br - bl
        if np.linalg.norm(chord) < 1e-12:
            return float("nan"), float("nan")
        # Standard half-ellipse endpoint tangents are perpendicular to its chord.
        t = np.array([-chord[1], chord[0]], dtype=float)
        return angle_between_deg(vL, t), angle_between_deg(vR, t)

    return float("nan"), float("nan")


def rasterize_polygon(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    pil = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(pil)
    xy = [(float(x), float(y)) for x, y in points]
    draw.polygon(xy, fill=255)
    return (np.asarray(pil, dtype=np.uint8) > 0).astype(np.uint8)


def target_polygon(
    points: dict[str, ControlPoint],
    bottom_type: str,
    shape: tuple[int, int],
    tangent_ellipse: bool = False,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    tl = np.array([points["TL"].x, points["TL"].y], dtype=float)
    bl = np.array([points["BL"].x, points["BL"].y], dtype=float)
    ba = np.array([points["BA"].x, points["BA"].y], dtype=float)
    br = np.array([points["BR"].x, points["BR"].y], dtype=float)
    tr = np.array([points["TR"].x, points["TR"].y], dtype=float)

    conic_coeffs = None
    if bottom_type == "circle":
        bottom = circle_arc_through_points(bl, ba, br)
        effective_apex = ba.copy()
    elif bottom_type == "ellipse":
        if tangent_ellipse:
            solved = conic_tangent_ellipse(bl, ba, br, bl - tl, br - tr)
            if solved is None:
                bottom, effective_apex = ellipse_arc_from_shoulders_apex(bl, ba, br)
            else:
                bottom, conic_coeffs = solved
                effective_apex = ba.copy()
        else:
            bottom, effective_apex = ellipse_arc_from_shoulders_apex(bl, ba, br)
    elif bottom_type == "flat":
        bottom = np.linspace(bl, br, 80)
        effective_apex = (bl + br) / 2.0
    elif bottom_type == "pointed":
        bottom = np.vstack([
            np.linspace(bl, ba, 60, endpoint=False),
            np.linspace(ba, br, 60),
        ])
        effective_apex = ba.copy()
    else:
        raise ValueError(bottom_type)

    poly = np.vstack([
        tl,
        bl,
        bottom[1:-1] if len(bottom) > 2 else bottom,
        br,
        tr,
    ])
    mask = rasterize_polygon(poly, shape)
    return mask, effective_apex, conic_coeffs



def normalize_comment(value: str) -> str:
    return str(value).strip().replace(" ", "").lower()


def candidate_target_class(category: str, comment: str) -> Optional[str]:
    """Map the review CSV's category/comment fields to the eight final target classes."""
    cat = str(category).strip()
    com = normalize_comment(comment)

    if cat == "Typical":
        return "Typical"
    if cat == "Elongated":
        return "Elongated"
    if cat == "Oblique":
        if com in {"右", "right", "r"}:
            return "Oblique-right"
        if com in {"左", "left", "l"}:
            return "Oblique-left"
        return None
    if cat == "Wide & short":
        return "Wide & short (flat-bottom)"
    if cat == "Other":
        if com in {"丸底", "rounded", "round", "rounded-bottom"}:
            return "Rounded-bottom"
        if com in {"とんがり", "尖り", "pointed", "pointed-bottom", "triangle", "triangular"}:
            return "Pointed-bottom"
        if com in {"低テーパー", "低taper", "low-taper", "lowtaper", "near-parallel", "parallel"}:
            return "Near-parallel sidewall"
    return None


def read_candidate_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return []

    required = {"hole_uid", "source_target_path"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Candidate CSV is missing required columns {sorted(missing)}: {path}"
        )

    out = []
    for r in rows:
        src = str(r.get("source_target_path", "")).strip()
        if not src:
            continue
        category = str(r.get("category", "")).strip()
        comment = str(r.get("comment", "")).strip()
        mapped = candidate_target_class(category, comment)
        out.append({
            "hole_uid": str(r.get("hole_uid", "")).strip(),
            "pulse_number": str(r.get("pulse_number", "")).strip(),
            "category": category,
            "comment": comment,
            "mapped_target_class": mapped or "",
            "source_target_path": src,
        })
    return out




def round_pixel(value: float) -> float:
    """Round a non-negative image coordinate to the nearest integer pixel."""
    return float(int(math.floor(float(value) + 0.5)))



class IdealTargetDesigner(tk.Tk):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.output_dir = args.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.title("Ideal Target Designer")
        self.geometry("1680x920")
        self.minsize(1450, 820)

        self.mask_paths: list[Path] = []
        self.raw_masks: list[np.ndarray] = []
        self.display_masks: list[np.ndarray] = []
        self.shape: Optional[tuple[int, int]] = None

        self.candidate_csv_path: Optional[Path] = (
            args.candidate_csv.resolve() if args.candidate_csv is not None else None
        )
        self.candidate_rows: list[dict] = []
        self.group_var = tk.StringVar(value="Typical")

        self.points: dict[str, ControlPoint] = {}
        self.drag_name: Optional[str] = None
        self.last_active_name: Optional[str] = None
        self.active_point_var = tk.StringVar(
            value="Active point: —   |   x=0 at image center, y=0 at image top"
        )

        self.target_id_var = tk.StringVar(value=args.target_id or "T_IDEAL_01")
        self.class_var = tk.StringVar(value=args.target_class)
        self.bottom_var = tk.StringVar(value=CLASS_TO_BOTTOM[args.target_class])
        self.opacity_var = tk.DoubleVar(value=0.16)
        self.align_sources_var = tk.BooleanVar(value=True)
        self.lock_center_var = tk.BooleanVar(value=False)
        self.center_ref_var = tk.StringVar(value="Ideal centroid")
        self.show_fill_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Load 1–3 observed masks.")
        self.metrics_var = tk.StringVar(value="")

        # Geometric constraint controls.
        self.axis_snap_var = tk.BooleanVar(value=True)
        self.right_angle_snap_var = tk.BooleanVar(value=True)
        self.snap_tolerance_deg_var = tk.DoubleVar(value=7.5)
        self.tangent_ellipse_var = tk.BooleanVar(value=False)
        self.last_tangent_ellipse_coeffs: Optional[np.ndarray] = None

        self.display_size = int(args.canvas_size)
        self.photo_ref = None

        self._build_ui()

        if self.candidate_csv_path is not None:
            self.load_candidate_csv(self.candidate_csv_path, show_errors=True)

        for p in args.mask[:3]:
            self._add_mask_path(p.resolve(), redraw=False)

        if self.raw_masks:
            self.auto_initialize()
        else:
            self.render()

    # ---------------- UI ----------------
    def _build_ui(self):
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)

        # Main image canvas stays on the left.  The controls are intentionally
        # widened toward the left and arranged in two columns to avoid a long
        # vertical settings strip.
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=8)

        right_outer = ttk.Frame(main, width=760)
        right_outer.pack(side="right", fill="both", padx=(4, 8), pady=8)
        right_outer.pack_propagate(False)

        # Keep vertical scrolling only as a fallback for smaller displays.
        right_canvas = tk.Canvas(
            right_outer,
            width=740,
            highlightthickness=0,
        )
        right_scroll = ttk.Scrollbar(
            right_outer,
            orient="vertical",
            command=right_canvas.yview,
        )
        right_canvas.configure(yscrollcommand=right_scroll.set)
        right_scroll.pack(side="right", fill="y")
        right_canvas.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(right_canvas)
        right_window = right_canvas.create_window(
            (0, 0),
            window=right,
            anchor="nw",
        )
        right.bind(
            "<Configure>",
            lambda e: right_canvas.configure(
                scrollregion=right_canvas.bbox("all")
            ),
        )
        right_canvas.bind(
            "<Configure>",
            lambda e: right_canvas.itemconfigure(
                right_window,
                width=e.width,
            ),
        )

        # Two equal-width control columns.
        col_left = ttk.Frame(right)
        col_right = ttk.Frame(right)
        col_left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        col_right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right.columnconfigure(0, weight=1, uniform="control")
        right.columnconfigure(1, weight=1, uniform="control")

        self.canvas = tk.Canvas(
            left,
            width=self.display_size,
            height=self.display_size,
            background="black",
            highlightthickness=1,
            highlightbackground="#888888",
        )
        self.canvas.pack(anchor="center")
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        ttk.Label(
            left,
            textvariable=self.active_point_var,
            anchor="center",
            font=("Consolas", 11, "bold"),
        ).pack(fill="x", pady=(5, 0))

        ttk.Label(
            left,
            textvariable=self.status_var,
            anchor="w",
        ).pack(fill="x", pady=(3, 0))

        # ================================================================
        # LEFT CONTROL COLUMN
        # ================================================================

        # Target identity
        box = ttk.LabelFrame(col_left, text="Target")
        box.pack(fill="x", pady=(0, 8))
        ttk.Label(box, text="Target ID").grid(
            row=0, column=0, sticky="w", padx=6, pady=4
        )
        ttk.Entry(
            box,
            textvariable=self.target_id_var,
            width=25,
        ).grid(
            row=0, column=1, sticky="ew", padx=6, pady=4
        )
        ttk.Label(box, text="Class").grid(
            row=1, column=0, sticky="w", padx=6, pady=4
        )
        class_cb = ttk.Combobox(
            box,
            textvariable=self.class_var,
            values=TARGET_CLASSES,
            state="readonly",
            width=23,
        )
        class_cb.grid(
            row=1, column=1, sticky="ew", padx=6, pady=4
        )
        class_cb.bind("<<ComboboxSelected>>", self.on_class_changed)
        box.columnconfigure(1, weight=1)

        # Candidate CSV
        box = ttk.LabelFrame(col_left, text="Candidate CSV")
        box.pack(fill="x", pady=(0, 8))

        self.candidate_csv_label = ttk.Label(
            box,
            text="No candidate CSV loaded",
            wraplength=330,
        )
        self.candidate_csv_label.pack(
            fill="x", padx=6, pady=(6, 4)
        )

        row = ttk.Frame(box)
        row.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Button(
            row,
            text="Load final_candidates.csv…",
            command=self.load_candidate_csv_dialog,
        ).pack(side="left")

        ttk.Label(row, text="Group").pack(
            side="left", padx=(8, 4)
        )
        group_cb = ttk.Combobox(
            row,
            textvariable=self.group_var,
            values=TARGET_CLASSES,
            state="readonly",
            width=20,
        )
        group_cb.pack(side="left", fill="x", expand=True)

        ttk.Button(
            box,
            text="Load this group from CSV",
            command=self.load_candidate_group,
        ).pack(fill="x", padx=6, pady=(0, 4))

        self.candidate_tree = ttk.Treeview(
            box,
            columns=("hole", "category", "comment", "mapped"),
            show="headings",
            height=5,
            selectmode="extended",
        )
        self.candidate_tree.heading("hole", text="hole")
        self.candidate_tree.heading("category", text="category")
        self.candidate_tree.heading("comment", text="comment")
        self.candidate_tree.heading("mapped", text="ideal class")
        self.candidate_tree.column(
            "hole", width=55, anchor="center", stretch=False
        )
        self.candidate_tree.column("category", width=82)
        self.candidate_tree.column("comment", width=68)
        self.candidate_tree.column("mapped", width=120)
        self.candidate_tree.pack(
            fill="x", padx=6, pady=(0, 4)
        )

        ttk.Button(
            box,
            text="Load selected rows (max 3)",
            command=self.load_selected_candidate_rows,
        ).pack(fill="x", padx=6, pady=(0, 6))

        # Source masks
        box = ttk.LabelFrame(col_left, text="Observed masks (max 3)")
        box.pack(fill="x", pady=(0, 8))
        self.source_list = tk.Listbox(box, height=3)
        self.source_list.pack(
            fill="x", padx=6, pady=(6, 4)
        )
        btns = ttk.Frame(box)
        btns.pack(fill="x", padx=6, pady=(0, 5))
        ttk.Button(
            btns,
            text="Add masks…",
            command=self.add_masks_dialog,
        ).pack(side="left")
        ttk.Button(
            btns,
            text="Remove selected",
            command=self.remove_selected_mask,
        ).pack(side="left", padx=4)
        ttk.Button(
            btns,
            text="Clear",
            command=self.clear_masks,
        ).pack(side="left")

        ttk.Checkbutton(
            box,
            text="Align observed masks to image X-center",
            variable=self.align_sources_var,
            command=self.on_alignment_changed,
        ).pack(anchor="w", padx=6, pady=(0, 3))

        op = ttk.Frame(box)
        op.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(op, text="Overlay opacity").pack(side="left")
        ttk.Scale(
            op,
            from_=0.05,
            to=0.40,
            variable=self.opacity_var,
            orient="horizontal",
            command=lambda _=None: self.render(),
        ).pack(side="left", fill="x", expand=True, padx=6)

        # X centering moved to the left column because it belongs to source /
        # placement setup rather than geometric primitive editing.
        box = ttk.LabelFrame(col_left, text="X centering")
        box.pack(fill="x", pady=(0, 8))
        ttk.Label(box, text="Reference").grid(
            row=0, column=0, sticky="w", padx=6, pady=4
        )
        ref = ttk.Combobox(
            box,
            textvariable=self.center_ref_var,
            values=["Ideal centroid", "Opening midpoint"],
            state="readonly",
            width=20,
        )
        ref.grid(
            row=0, column=1, sticky="ew", padx=6, pady=4
        )
        ttk.Button(
            box,
            text="Center ideal target X",
            command=self.center_ideal_x,
        ).grid(
            row=1, column=0, columnspan=2,
            sticky="ew", padx=6, pady=4
        )
        ttk.Checkbutton(
            box,
            text="Auto-center X while editing (TL/TR remain free)",
            variable=self.lock_center_var,
        ).grid(
            row=2, column=0, columnspan=2,
            sticky="w", padx=6, pady=(2, 2)
        )
        ttk.Label(
            box,
            text=(
                "TL/TR X positions remain independent. Centering only "
                "translates the complete ideal target."
            ),
            wraplength=330,
            foreground="#555555",
        ).grid(
            row=3, column=0, columnspan=2,
            sticky="w", padx=6, pady=(0, 6)
        )
        box.columnconfigure(1, weight=1)

        # ================================================================
        # RIGHT CONTROL COLUMN
        # ================================================================

        # Geometry
        box = ttk.LabelFrame(col_right, text="Geometry")
        box.pack(fill="x", pady=(0, 8))
        ttk.Label(box, text="Bottom").grid(
            row=0, column=0, sticky="w", padx=6, pady=4
        )
        bottom_cb = ttk.Combobox(
            box,
            textvariable=self.bottom_var,
            values=["circle", "ellipse", "flat", "pointed"],
            state="readonly",
            width=18,
        )
        bottom_cb.grid(
            row=0, column=1, sticky="ew", padx=6, pady=4
        )
        bottom_cb.bind(
            "<<ComboboxSelected>>",
            self.on_bottom_changed,
        )

        ttk.Checkbutton(
            box,
            text="Show ideal-mask fill",
            variable=self.show_fill_var,
            command=self.render,
        ).grid(
            row=1, column=0, columnspan=2,
            sticky="w", padx=6, pady=3
        )

        ttk.Button(
            box,
            text="Auto initialize from masks",
            command=self.auto_initialize,
        ).grid(
            row=2, column=0, columnspan=2,
            sticky="ew", padx=6, pady=3
        )
        ttk.Button(
            box,
            text="Level opening",
            command=self.level_opening,
        ).grid(
            row=3, column=0, sticky="ew", padx=6, pady=3
        )
        ttk.Button(
            box,
            text="Parallelize sidewalls",
            command=self.parallelize_sidewalls,
        ).grid(
            row=3, column=1, sticky="ew", padx=6, pady=3
        )
        box.columnconfigure(0, weight=1)
        box.columnconfigure(1, weight=1)

        # Snapping / geometric constraints
        box = ttk.LabelFrame(col_right, text="Snapping / constraints")
        box.pack(fill="x", pady=(0, 8))

        ttk.Checkbutton(
            box,
            text="Snap straight segments to X/Y while dragging",
            variable=self.axis_snap_var,
        ).grid(
            row=0, column=0, columnspan=2,
            sticky="w", padx=6, pady=2
        )
        ttk.Checkbutton(
            box,
            text="Snap sidewalls to 90° from opening",
            variable=self.right_angle_snap_var,
        ).grid(
            row=1, column=0, columnspan=2,
            sticky="w", padx=6, pady=2
        )

        ttk.Label(
            box,
            text="Angular tolerance",
        ).grid(
            row=2, column=0, sticky="w", padx=6, pady=3
        )
        ttk.Spinbox(
            box,
            from_=1.0,
            to=20.0,
            increment=0.5,
            textvariable=self.snap_tolerance_deg_var,
            width=8,
        ).grid(
            row=2, column=1, sticky="w", padx=6, pady=3
        )

        ttk.Button(
            box,
            text="Opening → X",
            command=self.snap_opening_x,
        ).grid(
            row=3, column=0, sticky="ew", padx=6, pady=3
        )
        ttk.Button(
            box,
            text="Shoulders → X",
            command=self.snap_shoulders_x,
        ).grid(
            row=3, column=1, sticky="ew", padx=6, pady=3
        )
        ttk.Button(
            box,
            text="Left wall → Y",
            command=lambda: self.snap_wall_y("left"),
        ).grid(
            row=4, column=0, sticky="ew", padx=6, pady=3
        )
        ttk.Button(
            box,
            text="Right wall → Y",
            command=lambda: self.snap_wall_y("right"),
        ).grid(
            row=4, column=1, sticky="ew", padx=6, pady=3
        )
        ttk.Button(
            box,
            text="Sidewalls ⟂ opening",
            command=self.orthogonalize_sidewalls_to_opening,
        ).grid(
            row=5, column=0, columnspan=2,
            sticky="ew", padx=6, pady=3
        )
        ttk.Button(
            box,
            text="Enforce arc ↔ sidewall tangency",
            command=self.enforce_arc_tangency,
        ).grid(
            row=6, column=0, columnspan=2,
            sticky="ew", padx=6, pady=(3, 6)
        )
        box.columnconfigure(0, weight=1)
        box.columnconfigure(1, weight=1)

        # Metrics
        box = ttk.LabelFrame(col_right, text="Live metrics")
        box.pack(fill="x", pady=(0, 8))
        ttk.Label(
            box,
            textvariable=self.metrics_var,
            justify="left",
            font=("Consolas", 9),
        ).pack(fill="x", padx=6, pady=6)

        # Save / output
        box = ttk.LabelFrame(col_right, text="Output")
        box.pack(fill="x", pady=(0, 8))
        ttk.Button(
            box,
            text="Save ideal target",
            command=self.save_target,
        ).pack(fill="x", padx=6, pady=(6, 3))
        ttk.Button(
            box,
            text="Open output folder",
            command=self.open_output_dir,
        ).pack(fill="x", padx=6, pady=(3, 6))

        help_text = (
            "Drag handles:\n"
            "TL/TR = opening edges (independent X movement)\n"
            "BL/BR = bottom-arc shoulders\n"
            "BA = bottom apex\n\n"
            "Rounded-bottom uses an ellipse.\n"
            "Near-parallel sidewall uses a circular bottom arc."
        )
        help_box = ttk.LabelFrame(col_right, text="Handle guide")
        help_box.pack(fill="x", pady=(0, 8))
        ttk.Label(
            help_box,
            text=help_text,
            justify="left",
        ).pack(fill="x", padx=6, pady=6)

    # ---------------- Candidate CSV ----------------
    def load_candidate_csv_dialog(self):
        path = filedialog.askopenfilename(
            title="Select final_candidates.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.load_candidate_csv(Path(path).resolve(), show_errors=True)

    def load_candidate_csv(self, path: Path, show_errors: bool = False):
        try:
            rows = read_candidate_csv(path)
        except Exception as exc:
            if show_errors:
                messagebox.showerror("Candidate CSV", str(exc), parent=self)
            return

        self.candidate_csv_path = path
        self.candidate_rows = rows
        self.candidate_csv_label.configure(
            text=f"{path.name}  ({len(rows)} candidates)"
        )

        for item in self.candidate_tree.get_children():
            self.candidate_tree.delete(item)

        for i, r in enumerate(rows):
            self.candidate_tree.insert(
                "",
                "end",
                iid=str(i),
                values=(
                    r["hole_uid"],
                    r["category"],
                    r["comment"],
                    r["mapped_target_class"],
                ),
            )

        self.status_var.set(
            f"Candidate CSV loaded: {len(rows)} rows. "
            "Choose a group or select up to 3 rows."
        )

    def clear_loaded_masks_only(self):
        self.source_list.delete(0, "end")
        self.mask_paths.clear()
        self.raw_masks.clear()
        self.display_masks.clear()
        self.points = {}
        self.shape = None

    def load_candidate_group(self):
        if not self.candidate_rows:
            messagebox.showwarning(
                "Candidate CSV",
                "Load final_candidates.csv first.",
                parent=self,
            )
            return

        target_class = self.group_var.get()
        matches = [
            r for r in self.candidate_rows
            if r["mapped_target_class"] == target_class
        ]
        if not matches:
            messagebox.showwarning(
                "Candidate group",
                f"No rows map to {target_class}.",
                parent=self,
            )
            return
        if len(matches) > 3:
            messagebox.showwarning(
                "Candidate group",
                f"{target_class} has {len(matches)} rows. "
                "Select up to 3 rows manually in the table.",
                parent=self,
            )
            return

        self._load_candidate_records(matches, target_class)

    def load_selected_candidate_rows(self):
        if not self.candidate_rows:
            messagebox.showwarning(
                "Candidate CSV",
                "Load final_candidates.csv first.",
                parent=self,
            )
            return

        selected = self.candidate_tree.selection()
        if not selected:
            messagebox.showwarning(
                "Candidate CSV",
                "Select 1–3 rows in the table.",
                parent=self,
            )
            return
        if len(selected) > 3:
            messagebox.showwarning(
                "Candidate CSV",
                "Select at most 3 rows.",
                parent=self,
            )
            return

        records = [self.candidate_rows[int(i)] for i in selected]
        mapped = {r["mapped_target_class"] for r in records if r["mapped_target_class"]}
        target_class = next(iter(mapped)) if len(mapped) == 1 else None
        self._load_candidate_records(records, target_class)

    def _load_candidate_records(self, records: list[dict], target_class: Optional[str]):
        missing = []
        paths = []
        for r in records:
            p = Path(r["source_target_path"])
            if not p.exists():
                missing.append(str(p))
            else:
                paths.append(p)

        if missing:
            messagebox.showerror(
                "Missing target.npy",
                "The following source_target_path entries do not exist on this PC:\n\n"
                + "\n".join(missing),
                parent=self,
            )
            return

        self.clear_loaded_masks_only()
        try:
            for p in paths:
                self._add_mask_path(p.resolve(), redraw=False)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc), parent=self)
            return

        if target_class:
            self.class_var.set(target_class)
            self.group_var.set(target_class)
            self.bottom_var.set(CLASS_TO_BOTTOM[target_class])

        # Generate a useful default ID from class. User can still edit it.
        if target_class:
            self.target_id_var.set(
                "T_" + target_class.upper()
                .replace(" & ", "_")
                .replace(" ", "_")
                .replace("-", "_")
                .replace("(", "")
                .replace(")", "")
            )

        self.rebuild_display_masks()
        self.auto_initialize()

        hole_text = ", ".join(r["hole_uid"] for r in records)
        self.status_var.set(
            f"Loaded from candidate CSV: {hole_text}"
            + (f" -> {target_class}" if target_class else "")
        )

    # ---------------- Masks ----------------
    def add_masks_dialog(self):
        paths = filedialog.askopenfilenames(
            title="Select up to 3 observed masks",
            filetypes=[
                ("Mask files", "*.npy *.png *.tif *.tiff *.bmp *.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return

        for s in paths:
            if len(self.mask_paths) >= 3:
                messagebox.showinfo(
                    "Observed masks",
                    "Maximum of 3 observed masks can be loaded.",
                    parent=self,
                )
                break
            try:
                self._add_mask_path(Path(s).resolve(), redraw=False)
            except Exception as exc:
                messagebox.showerror("Load error", str(exc), parent=self)

        if self.raw_masks:
            self.auto_initialize()
        else:
            self.render()

    def _add_mask_path(self, path: Path, redraw: bool = True):
        if path in self.mask_paths:
            return
        mask = load_mask(path, self.args.channel, self.args.threshold)
        if self.shape is None:
            self.shape = mask.shape
        elif mask.shape != self.shape:
            raise ValueError(
                f"All masks must have the same shape. "
                f"Expected {self.shape}, got {mask.shape}: {path}"
            )

        self.mask_paths.append(path)
        self.raw_masks.append(mask)
        self.source_list.insert("end", path.name)
        self.rebuild_display_masks()
        if redraw:
            self.render()

    def remove_selected_mask(self):
        sel = self.source_list.curselection()
        if not sel:
            return
        i = int(sel[0])
        self.source_list.delete(i)
        self.mask_paths.pop(i)
        self.raw_masks.pop(i)
        if not self.raw_masks:
            self.shape = None
            self.points = {}
        self.rebuild_display_masks()
        self.render()

    def clear_masks(self):
        self.source_list.delete(0, "end")
        self.mask_paths.clear()
        self.raw_masks.clear()
        self.display_masks.clear()
        self.points = {}
        self.shape = None
        self.render()

    def on_alignment_changed(self):
        self.rebuild_display_masks()
        if self.raw_masks:
            self.auto_initialize()
        else:
            self.render()

    def rebuild_display_masks(self):
        self.display_masks = []
        if not self.raw_masks:
            return
        h, w = self.raw_masks[0].shape
        xmid = (w - 1) / 2.0
        for m in self.raw_masks:
            if self.align_sources_var.get():
                dx = int(round(xmid - opening_center_x(m)))
                self.display_masks.append(translate_x(m, dx))
            else:
                self.display_masks.append(m.copy())

    # ---------------- Geometry ----------------
    def snap_control_points_to_pixels(self):
        """Force every editable control point onto integer image coordinates."""
        if not self.points or self.shape is None:
            return
        h, w = self.shape
        for p in self.points.values():
            p.x = float(np.clip(round_pixel(p.x), 0, w - 1))
            p.y = float(np.clip(round_pixel(p.y), 0, h - 1))

    def on_bottom_changed(self, _event=None):
        self.tangent_ellipse_var.set(False)
        self.last_tangent_ellipse_coeffs = None
        self.render()

    def on_class_changed(self, _event=None):
        cls = self.class_var.get()
        self.bottom_var.set(CLASS_TO_BOTTOM[cls])
        self.tangent_ellipse_var.set(False)
        self.last_tangent_ellipse_coeffs = None
        if cls == "Near-parallel sidewall" and self.points:
            self.parallelize_sidewalls()
        else:
            self.render()

    def auto_initialize(self):
        if not self.display_masks:
            return
        h, w = self.display_masks[0].shape
        agg = (np.mean(np.stack(self.display_masks, axis=0), axis=0) >= 0.5).astype(np.uint8)
        ys, xs = np.nonzero(agg)
        if len(xs) == 0:
            return

        ymin, ymax = int(ys.min()), int(ys.max())
        depth = max(1, ymax - ymin + 1)

        def edge_at_fraction(frac: float):
            target_y = int(round(ymin + frac * (depth - 1)))
            order = sorted(
                range(ymin, ymax + 1),
                key=lambda yy: abs(yy - target_y),
            )
            for yy in order:
                xx = np.flatnonzero(agg[yy])
                if xx.size:
                    return float(xx.min()), float(xx.max()), float(yy)
            return float(xs.min()), float(xs.max()), float(target_y)

        l0, r0, y0 = edge_at_fraction(0.08)
        l1, r1, y1 = edge_at_fraction(0.72)

        deepest = ys >= max(ymin, ymax - max(2, int(0.04 * depth)))
        bx = float(np.median(xs[deepest])) if np.any(deepest) else float(np.mean(xs))
        by = float(ymax)

        self.points = {
            "TL": ControlPoint("TL", l0, y0),
            "BL": ControlPoint("BL", l1, y1),
            "BA": ControlPoint("BA", bx, by),
            "BR": ControlPoint("BR", r1, y1),
            "TR": ControlPoint("TR", r0, y0),
        }

        if self.bottom_var.get() == "flat":
            flat_y = float(np.percentile(ys, 96))
            self.points["BL"].y = flat_y
            self.points["BR"].y = flat_y
            self.points["BA"].y = flat_y
            self.points["BA"].x = (
                self.points["BL"].x + self.points["BR"].x
            ) / 2.0

        self.snap_control_points_to_pixels()
        if self.lock_center_var.get():
            self.center_ideal_x()
        else:
            self.render()

    def level_opening(self):
        if not self.points:
            return
        y = round_pixel((self.points["TL"].y + self.points["TR"].y) / 2.0)
        self.points["TL"].y = y
        self.points["TR"].y = y
        self.snap_control_points_to_pixels()
        self.render()

    def parallelize_sidewalls(self):
        if not self.points:
            return
        tl = np.array([self.points["TL"].x, self.points["TL"].y])
        bl = np.array([self.points["BL"].x, self.points["BL"].y])
        tr = np.array([self.points["TR"].x, self.points["TR"].y])
        br = np.array([self.points["BR"].x, self.points["BR"].y])

        vl = bl - tl
        vr = br - tr
        # Average the horizontal drift per unit depth while preserving each
        # side's vertical span.
        sl = vl[0] / vl[1] if abs(vl[1]) > 1e-9 else 0.0
        sr = vr[0] / vr[1] if abs(vr[1]) > 1e-9 else 0.0
        s = (sl + sr) / 2.0

        self.points["BL"].x = self.points["TL"].x + s * (
            self.points["BL"].y - self.points["TL"].y
        )
        self.points["BR"].x = self.points["TR"].x + s * (
            self.points["BR"].y - self.points["TR"].y
        )
        self.snap_control_points_to_pixels()
        self.render()

    def snap_opening_x(self):
        if not self.points:
            return
        y = round_pixel((self.points["TL"].y + self.points["TR"].y) / 2.0)
        self.points["TL"].y = y
        self.points["TR"].y = y
        self.snap_control_points_to_pixels()
        self.render()

    def snap_shoulders_x(self):
        if not self.points:
            return
        y = round_pixel((self.points["BL"].y + self.points["BR"].y) / 2.0)
        self.points["BL"].y = y
        self.points["BR"].y = y
        if self.bottom_var.get() == "flat":
            self.points["BA"].y = y
        self.snap_control_points_to_pixels()
        self.tangent_ellipse_var.set(False)
        self.render()

    def snap_wall_y(self, side: str):
        if not self.points:
            return
        if side == "left":
            x = round_pixel((self.points["TL"].x + self.points["BL"].x) / 2.0)
            self.points["TL"].x = x
            self.points["BL"].x = x
        else:
            x = round_pixel((self.points["TR"].x + self.points["BR"].x) / 2.0)
            self.points["TR"].x = x
            self.points["BR"].x = x
        self.snap_control_points_to_pixels()
        self.tangent_ellipse_var.set(False)
        if self.lock_center_var.get():
            self.center_ideal_x()
        else:
            self.render()

    def orthogonalize_sidewalls_to_opening(self):
        """Make both straight sidewalls exactly perpendicular to the opening line."""
        if not self.points:
            return
        tl = np.array([self.points["TL"].x, self.points["TL"].y], dtype=float)
        tr = np.array([self.points["TR"].x, self.points["TR"].y], dtype=float)
        bl = np.array([self.points["BL"].x, self.points["BL"].y], dtype=float)
        br = np.array([self.points["BR"].x, self.points["BR"].y], dtype=float)
        u = unit_vector(tr - tl)
        if np.linalg.norm(u) < 1e-12:
            return
        n = np.array([-u[1], u[0]], dtype=float)
        if n[1] < 0:
            n = -n
        len_l = max(1.0, float(np.linalg.norm(bl - tl)))
        len_r = max(1.0, float(np.linalg.norm(br - tr)))
        bl2 = tl + n * len_l
        br2 = tr + n * len_r
        self.points["BL"].x, self.points["BL"].y = map(float, bl2)
        self.points["BR"].x, self.points["BR"].y = map(float, br2)
        self.snap_control_points_to_pixels()
        self.tangent_ellipse_var.set(False)
        if self.lock_center_var.get():
            self.center_ideal_x()
        else:
            self.render()

    def enforce_arc_tangency(self):
        if not self.points:
            return
        bt = self.bottom_var.get()
        if bt not in {"circle", "ellipse"}:
            messagebox.showinfo(
                "Tangency",
                "Arc-line tangency is available for circular and elliptical bottoms.",
                parent=self,
            )
            return

        tl = np.array([self.points["TL"].x, self.points["TL"].y], dtype=float)
        tr = np.array([self.points["TR"].x, self.points["TR"].y], dtype=float)
        bl = np.array([self.points["BL"].x, self.points["BL"].y], dtype=float)
        ba = np.array([self.points["BA"].x, self.points["BA"].y], dtype=float)
        br = np.array([self.points["BR"].x, self.points["BR"].y], dtype=float)

        if bt == "circle":
            solved = tangent_circle_to_sidewalls_through_apex(tl, tr, ba, bl, br)
            if solved is None:
                messagebox.showwarning(
                    "Tangency",
                    "Could not construct a stable circle tangent to both sidewalls "
                    "and passing through BA. Adjust the sidewalls or BA and retry.",
                    parent=self,
                )
                return
            _center, _radius, ql, qr = solved
            self.points["BL"].x, self.points["BL"].y = map(float, ql)
            self.points["BR"].x, self.points["BR"].y = map(float, qr)
            # Editable control points are pixel-indexed.  The continuous
            # tangency solution is therefore rounded to the nearest pixels.
            self.snap_control_points_to_pixels()
            self.tangent_ellipse_var.set(False)
            self.status_var.set(
                "Circular bottom: tangency solved, then BL/BR snapped to nearest "
                "integer pixels. Check the residual tangent error in Live metrics."
            )
        else:
            self.snap_control_points_to_pixels()
            bl = np.array([self.points["BL"].x, self.points["BL"].y], dtype=float)
            ba = np.array([self.points["BA"].x, self.points["BA"].y], dtype=float)
            br = np.array([self.points["BR"].x, self.points["BR"].y], dtype=float)
            tl = np.array([self.points["TL"].x, self.points["TL"].y], dtype=float)
            tr = np.array([self.points["TR"].x, self.points["TR"].y], dtype=float)
            solved = conic_tangent_ellipse(bl, ba, br, bl - tl, br - tr)
            if solved is None:
                messagebox.showwarning(
                    "Tangency",
                    "The current three bottom points and sidewall directions do not "
                    "produce a stable real ellipse. Adjust BL/BA/BR or sidewalls and retry.",
                    parent=self,
                )
                return
            _arc, coeff = solved
            self.tangent_ellipse_var.set(True)
            self.last_tangent_ellipse_coeffs = coeff
            self.status_var.set(
                "Elliptical bottom: exact conic ellipse constrained tangent to both sidewalls."
            )

        if self.lock_center_var.get():
            self.center_ideal_x()
        else:
            self.render()

    def straight_neighbors(self, name: str) -> list[str]:
        bt = self.bottom_var.get()
        mapping = {
            "TL": ["TR", "BL"],
            "TR": ["TL", "BR"],
            "BL": ["TL"],
            "BR": ["TR"],
            "BA": [],
        }
        if bt == "flat":
            mapping["BL"].append("BR")
            mapping["BR"].append("BL")
        elif bt == "pointed":
            mapping["BL"].append("BA")
            mapping["BR"].append("BA")
            mapping["BA"] = ["BL", "BR"]
        return mapping.get(name, [])

    def apply_axis_snap(self, moved_name: str):
        if not self.axis_snap_var.get() or not self.points:
            return
        p = self.points[moved_name]
        tol = math.radians(float(self.snap_tolerance_deg_var.get()))
        tan_tol = math.tan(tol)
        best_x = None
        best_x_score = float("inf")
        best_y = None
        best_y_score = float("inf")
        for nb_name in self.straight_neighbors(moved_name):
            q = self.points[nb_name]
            dx = p.x - q.x
            dy = p.y - q.y
            if abs(dy) > 1e-9:
                score = abs(dx) / abs(dy)
                if score <= tan_tol and score < best_x_score:
                    best_x_score = score
                    best_x = q.x
            if abs(dx) > 1e-9:
                score = abs(dy) / abs(dx)
                if score <= tan_tol and score < best_y_score:
                    best_y_score = score
                    best_y = q.y
        if best_x is not None:
            p.x = float(best_x)
        if best_y is not None:
            p.y = float(best_y)

    def apply_right_angle_snap(self, moved_name: str):
        if not self.right_angle_snap_var.get() or not self.points:
            return
        tol = float(self.snap_tolerance_deg_var.get())
        tl = np.array([self.points["TL"].x, self.points["TL"].y], dtype=float)
        tr = np.array([self.points["TR"].x, self.points["TR"].y], dtype=float)
        opening = tr - tl
        if np.linalg.norm(opening) < 1e-12:
            return
        u = unit_vector(opening)
        n = np.array([-u[1], u[0]], dtype=float)
        if n[1] < 0:
            n = -n

        if moved_name == "BL":
            p0 = tl
            p = np.array([self.points["BL"].x, self.points["BL"].y], dtype=float)
            if abs(90.0 - angle_between_deg(opening, p - p0)) <= tol:
                length = float(np.dot(p - p0, n))
                if length < 0:
                    length = abs(length)
                p2 = p0 + n * max(1.0, length)
                self.points["BL"].x, self.points["BL"].y = map(float, p2)
                self.snap_control_points_to_pixels()
        elif moved_name == "BR":
            p0 = tr
            p = np.array([self.points["BR"].x, self.points["BR"].y], dtype=float)
            if abs(90.0 - angle_between_deg(opening, p - p0)) <= tol:
                length = float(np.dot(p - p0, n))
                if length < 0:
                    length = abs(length)
                p2 = p0 + n * max(1.0, length)
                self.points["BR"].x, self.points["BR"].y = map(float, p2)
                self.snap_control_points_to_pixels()

    def current_ideal_mask(self) -> Optional[np.ndarray]:
        if not self.points or self.shape is None:
            return None
        mask, _, coeffs = target_polygon(
            self.points,
            self.bottom_var.get(),
            self.shape,
            tangent_ellipse=bool(self.tangent_ellipse_var.get()),
        )
        self.last_tangent_ellipse_coeffs = coeffs
        return mask

    def center_ideal_x(self):
        """Translate the complete ideal target in X without changing its shape.

        This is intentionally not a symmetry operation: TL and TR retain their
        independently chosen spacing relative to the rest of the target.
        """
        if not self.points or self.shape is None:
            return
        h, w = self.shape
        # Use an actual pixel index as the center.  For an even-width image
        # (e.g. 384 px), this intentionally uses x = 192 rather than 191.5.
        desired = float(w // 2)

        if self.center_ref_var.get() == "Ideal centroid":
            m = self.current_ideal_mask()
            ys, xs = np.nonzero(m) if m is not None else ([], [])
            current = float(xs.mean()) if len(xs) else (
                self.points["TL"].x + self.points["TR"].x
            ) / 2.0
        else:
            current = (
                self.points["TL"].x + self.points["TR"].x
            ) / 2.0

        # Pixel-indexed target placement: translate only by an integer number
        # of pixels.  This preserves the current shape exactly on the pixel grid.
        dx = round_pixel(desired - current) if desired - current >= 0 else -round_pixel(current - desired)

        # Clip the shift so no control point leaves the image, then keep it integer.
        xs0 = [p.x for p in self.points.values()]
        dx = max(dx, -min(xs0))
        dx = min(dx, (w - 1) - max(xs0))
        dx = float(int(round(dx)))

        for p in self.points.values():
            p.x += dx
        self.snap_control_points_to_pixels()
        self.render()

    def normalize_bottom_constraints(self, moved_name: str):
        if not self.points:
            return
        bt = self.bottom_var.get()

        if bt == "flat":
            if moved_name in {"BL", "BR", "BA"}:
                y = self.points[moved_name].y
                self.points["BL"].y = y
                self.points["BR"].y = y
                self.points["BA"].y = y
                self.points["BA"].x = (
                    self.points["BL"].x + self.points["BR"].x
                ) / 2.0

        elif bt == "ellipse":
            # BA must lie on the perpendicular bisector of BL-BR so the ellipse
            # passes through the visible apex handle exactly.
            b = np.array([self.points["BL"].x, self.points["BL"].y], dtype=float)
            d = np.array([self.points["BR"].x, self.points["BR"].y], dtype=float)
            c = np.array([self.points["BA"].x, self.points["BA"].y], dtype=float)
            chord = d - b
            L = float(np.linalg.norm(chord))
            if L > 1e-8:
                u = chord / L
                m = (b + d) / 2.0
                nvec = np.array([-u[1], u[0]], dtype=float)
                if np.dot(c - m, nvec) < 0:
                    nvec = -nvec
                sag = max(1.0, float(np.dot(c - m, nvec)))
                c2 = m + sag * nvec
                self.points["BA"].x = float(c2[0])
                self.points["BA"].y = float(c2[1])

        self.snap_control_points_to_pixels()

    # ---------------- Canvas interaction ----------------
    def image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        if self.shape is None:
            return x, y
        h, w = self.shape
        s = min(self.display_size / w, self.display_size / h)
        ox = (self.display_size - w * s) / 2.0
        oy = (self.display_size - h * s) / 2.0
        return ox + x * s, oy + y * s

    def canvas_to_image(self, cx: float, cy: float) -> tuple[float, float]:
        if self.shape is None:
            return cx, cy
        h, w = self.shape
        s = min(self.display_size / w, self.display_size / h)
        ox = (self.display_size - w * s) / 2.0
        oy = (self.display_size - h * s) / 2.0
        x = (cx - ox) / s
        y = (cy - oy) / s
        return x, y

    def image_center_x_px(self) -> int:
        if self.shape is None:
            return 0
        return int(self.shape[1] // 2)

    def centered_x_px(self, x_abs: float) -> int:
        """X coordinate with image-center pixel defined as x=0."""
        return int(round(float(x_abs))) - self.image_center_x_px()

    def update_active_point_label(self, name: Optional[str] = None):
        if name is None:
            name = self.drag_name or self.last_active_name
        if name is None or name not in self.points or self.shape is None:
            self.active_point_var.set(
                "Active point: —   |   x=0 at image center, y=0 at image top"
            )
            return
        p = self.points[name]
        x_rel = self.centered_x_px(p.x)
        y_top = int(round(p.y))
        self.active_point_var.set(
            f"Active point: {name}   x={x_rel:+d} px   y={y_top:d} px"
        )

    def nearest_handle(self, cx: float, cy: float, max_px: float = 18.0) -> Optional[str]:
        best = None
        best_d = max_px
        for name, p in self.points.items():
            px, py = self.image_to_canvas(p.x, p.y)
            d = math.hypot(cx - px, cy - py)
            if d < best_d:
                best_d = d
                best = name
        return best

    def on_press(self, event):
        self.drag_name = self.nearest_handle(event.x, event.y)
        if self.drag_name is not None:
            self.last_active_name = self.drag_name
            self.update_active_point_label(self.drag_name)
            self.render()

    def on_drag(self, event):
        if self.drag_name is None or self.shape is None:
            return
        x, y = self.canvas_to_image(event.x, event.y)
        h, w = self.shape
        p = self.points[self.drag_name]
        p.x = float(np.clip(round_pixel(x), 0, w - 1))
        p.y = float(np.clip(round_pixel(y), 0, h - 1))

        self.normalize_bottom_constraints(self.drag_name)
        self.apply_axis_snap(self.drag_name)
        self.apply_right_angle_snap(self.drag_name)
        self.snap_control_points_to_pixels()
        self.last_active_name = self.drag_name
        self.update_active_point_label(self.drag_name)
        # Any manual geometry edit invalidates an earlier tangent-ellipse solve.
        if self.bottom_var.get() == "ellipse":
            self.tangent_ellipse_var.set(False)
            self.last_tangent_ellipse_coeffs = None
        if self.lock_center_var.get() and self.drag_name not in {"TL", "TR"}:
            # Keep the overall target centered during non-opening edits.
            # TL and TR are deliberately excluded so either opening edge can be
            # moved independently in X without the opposite edge being shifted.
            self.center_ideal_x()
        else:
            self.render()

    def on_release(self, _event):
        if self.drag_name is not None:
            self.last_active_name = self.drag_name
            self.update_active_point_label(self.drag_name)
        self.drag_name = None
        self.render()

    # ---------------- Rendering ----------------
    def compose_preview(self) -> Image.Image:
        S = self.display_size
        bg = Image.new("RGBA", (S, S), (0, 0, 0, 255))
        if self.shape is None:
            draw = ImageDraw.Draw(bg)
            draw.text((25, 25), "Load 1–3 observed masks", fill=(220, 220, 220, 255))
            return bg.convert("RGB")

        h, w = self.shape
        scale = min(S / w, S / h)
        rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        ox, oy = (S - rw) // 2, (S - rh) // 2

        alpha_fill = int(round(255 * float(self.opacity_var.get())))
        for i, m in enumerate(self.display_masks):
            color = SOURCE_COLORS[i % len(SOURCE_COLORS)]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., :3] = color
            rgba[..., 3] = m.astype(np.uint8) * alpha_fill
            layer = Image.fromarray(rgba, mode="RGBA").resize(
                (rw, rh), Image.Resampling.NEAREST
            )
            bg.alpha_composite(layer, (ox, oy))

            # Slightly stronger boundary to distinguish overlapping masks.
            bd = boundary_mask(m)
            brgba = np.zeros((h, w, 4), dtype=np.uint8)
            brgba[..., :3] = color
            brgba[..., 3] = bd.astype(np.uint8) * min(220, alpha_fill + 100)
            blayer = Image.fromarray(brgba, mode="RGBA").resize(
                (rw, rh), Image.Resampling.NEAREST
            )
            bg.alpha_composite(blayer, (ox, oy))

        ideal = self.current_ideal_mask()
        if ideal is not None:
            if self.show_fill_var.get():
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[..., :3] = (255, 255, 255)
                rgba[..., 3] = ideal.astype(np.uint8) * 35
                layer = Image.fromarray(rgba, mode="RGBA").resize(
                    (rw, rh), Image.Resampling.NEAREST
                )
                bg.alpha_composite(layer, (ox, oy))

            bd = boundary_mask(ideal)
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., :3] = (255, 255, 255)
            rgba[..., 3] = bd.astype(np.uint8) * 255
            layer = Image.fromarray(rgba, mode="RGBA").resize(
                (rw, rh), Image.Resampling.NEAREST
            )
            bg.alpha_composite(layer, (ox, oy))

        draw = ImageDraw.Draw(bg)
        # Coordinate reference: the actual center pixel is x=0.
        # For 384 px width this is absolute x=192.
        cx_abs = w // 2
        cx = ox + cx_abs * scale
        draw.line((cx, oy, cx, oy + rh), fill=(120, 120, 120, 160), width=1)
        draw.text((cx + 5, oy + 5), "x=0", fill=(170, 170, 170, 220))

        # Geometry segments / handles.
        if self.points:
            order = ["TL", "BL", "BA", "BR", "TR"]
            for a, b in zip(order[:-1], order[1:]):
                xa, ya = self.image_to_canvas(self.points[a].x, self.points[a].y)
                xb, yb = self.image_to_canvas(self.points[b].x, self.points[b].y)
                draw.line((xa, ya, xb, yb), fill=(230, 230, 230, 100), width=1)

            for name, p in self.points.items():
                x, y = self.image_to_canvas(p.x, p.y)
                r = 7
                draw.ellipse((x-r, y-r, x+r, y+r), fill=(255, 255, 255, 255))
                draw.ellipse((x-r+2, y-r+2, x+r-2, y+r-2), fill=(30, 30, 30, 255))
                draw.text((x + 9, y - 10), name, fill=(255, 255, 255, 255))

            if self.drag_name is not None and self.drag_name in self.points:
                p = self.points[self.drag_name]
                x, y = self.image_to_canvas(p.x, p.y)
                x_rel = self.centered_x_px(p.x)
                y_top = int(round(p.y))
                label = f"{self.drag_name}  x={x_rel:+d}  y={y_top:d} px"
                # Small dark backing box keeps the live coordinate readable
                # over white masks.
                tx, ty = x + 18, y + 14
                bbox = draw.textbbox((tx, ty), label)
                pad = 4
                draw.rectangle(
                    (bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad),
                    fill=(0, 0, 0, 210),
                    outline=(220, 220, 220, 220),
                )
                draw.text((tx, ty), label, fill=(255, 255, 255, 255))

        return bg.convert("RGB")

    def update_metrics(self):
        ideal = self.current_ideal_mask()
        if ideal is None:
            self.metrics_var.set("")
            return
        met = mask_metrics(ideal)
        tl_y = float(self.points["TL"].y) if self.points else float("nan")
        tr_y = float(self.points["TR"].y) if self.points else float("nan")
        opening_y_mean = (tl_y + tr_y) / 2.0

        centroid_x_centered = met["centroid_x_px"] - self.image_center_x_px()
        lines = [
            f"area       : {100*met['area_fraction']:.2f} %",
            f"opening    : {met['opening_width_px']:.1f} px",
            f"depth      : {met['depth_px']:.1f} px",
            f"centroid x : {centroid_x_centered:+.2f} px (center=0)",
            f"TL y/top   : {int(round(tl_y))} px",
            f"TR y/top   : {int(round(tr_y))} px",
            f"opening y  : {opening_y_mean:.1f} px",
        ]
        active_name = self.drag_name or self.last_active_name
        if active_name is not None and active_name in self.points:
            ap = self.points[active_name]
            lines.append(
                f"active {active_name}: x={self.centered_x_px(ap.x):+d} px, "
                f"y={int(round(ap.y))} px"
            )
        for i, src in enumerate(self.display_masks, 1):
            lines.append(f"IoU vs obs{i} : {iou(ideal, src):.3f}")

        if self.points:
            tl = np.array([self.points["TL"].x, self.points["TL"].y], dtype=float)
            tr = np.array([self.points["TR"].x, self.points["TR"].y], dtype=float)
            bl = np.array([self.points["BL"].x, self.points["BL"].y], dtype=float)
            br = np.array([self.points["BR"].x, self.points["BR"].y], dtype=float)
            opening = tr - tl
            aL = angle_between_deg(opening, bl - tl)
            aR = angle_between_deg(opening, br - tr)
            lines.append(f"opening⊥L/R: {aL:.1f}° / {aR:.1f}°")
            tmL, tmR = tangent_mismatch_deg(
                self.points,
                self.bottom_var.get(),
                tangent_ellipse=bool(self.tangent_ellipse_var.get()),
            )
            if np.isfinite(tmL) and np.isfinite(tmR):
                lines.append(f"arc tangent err: {tmL:.2f}° / {tmR:.2f}°")
            if self.bottom_var.get() == "ellipse" and self.tangent_ellipse_var.get():
                lines.append("ellipse mode : tangent-constrained")
        self.metrics_var.set("\n".join(lines))

    def render(self):
        self.update_active_point_label()
        im = self.compose_preview()
        self.photo_ref = ImageTk.PhotoImage(im)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo_ref, anchor="nw")
        self.update_metrics()

        if self.mask_paths:
            self.status_var.set(
                f"{len(self.mask_paths)} observed mask(s) loaded. "
                f"Drag TL/BL/BA/BR/TR to idealize the contour."
            )
        else:
            self.status_var.set("Load 1–3 observed masks.")

    # ---------------- Save / provenance ----------------
    def save_target(self):
        if not self.raw_masks or self.shape is None or not self.points:
            messagebox.showwarning(
                "Save ideal target",
                "Load observed masks and define the ideal geometry first.",
                parent=self,
            )
            return

        target_id = self.target_id_var.get().strip()
        if not target_id:
            messagebox.showwarning("Target ID", "Enter a Target ID.", parent=self)
            return

        ideal = self.current_ideal_mask()
        if ideal is None:
            return

        target_dir = self.output_dir / target_id
        target_dir.mkdir(parents=True, exist_ok=True)

        npy_path = target_dir / "desired_target_mask.npy"
        png_path = target_dir / "desired_target_mask.png"
        overlay_path = target_dir / "idealization_overlay.png"
        json_path = target_dir / "ideal_target.json"
        csv_path = target_dir / "ideal_target_parameters.csv"

        np.save(npy_path, ideal.astype(np.float32), allow_pickle=False)
        Image.fromarray((ideal * 255).astype(np.uint8), mode="L").save(png_path)
        self.compose_preview().save(overlay_path)

        ideal_met = mask_metrics(ideal)
        source_entries = []
        for path, raw, shown in zip(self.mask_paths, self.raw_masks, self.display_masks):
            source_entries.append({
                "path": str(path),
                "sha256": sha256_file(path),
                "raw_opening_center_x_px": opening_center_x(raw),
                "display_opening_center_x_px": opening_center_x(shown),
                "iou_to_ideal_display_aligned": iou(ideal, shown),
                "metrics_raw": mask_metrics(raw),
            })

        payload = {
            "schema_version": "1.1",
            "designer_version": "v4",
            "target_id": target_id,
            "target_class": self.class_var.get(),
            "idealization_term": "geometric idealization",
            "control_point_grid": "integer_pixel",
            "bottom_type": self.bottom_var.get(),
            "bottom_definition": BOTTOM_LABELS[self.bottom_var.get()],
            "image_shape_yx": list(self.shape),
            "channel": int(self.args.channel),
            "threshold": float(self.args.threshold),
            "source_alignment_for_display": bool(self.align_sources_var.get()),
            "x_centering": {
                "locked": bool(self.lock_center_var.get()),
                "reference": self.center_ref_var.get(),
                "image_center_x_px": (self.shape[1] - 1) / 2.0,
            },
            "control_points_px": {
                k: {
                    "x_abs": int(round(v.x)),
                    "x_centered": int(round(v.x)) - self.image_center_x_px(),
                    "y_from_top": int(round(v.y)),
                }
                for k, v in self.points.items()
            },
            "coordinate_convention": {
                "x_zero_abs_px": self.image_center_x_px(),
                "x_positive_direction": "right",
                "y_zero_abs_px": 0,
                "y_positive_direction": "down",
            },
            "opening_position_from_top_px": {
                "TL_y_px": float(self.points["TL"].y),
                "TR_y_px": float(self.points["TR"].y),
                "mean_y_px": float(
                    (self.points["TL"].y + self.points["TR"].y) / 2.0
                ),
                "is_horizontal": bool(
                    abs(self.points["TL"].y - self.points["TR"].y) < 1e-6
                ),
            },
            "geometry_constraints": {
                "axis_snap_enabled": bool(self.axis_snap_var.get()),
                "right_angle_snap_enabled": bool(self.right_angle_snap_var.get()),
                "snap_tolerance_deg": float(self.snap_tolerance_deg_var.get()),
                "tangent_constrained_ellipse": bool(self.tangent_ellipse_var.get()),
                "tangent_ellipse_conic_coefficients_ABCDEF": (
                    self.last_tangent_ellipse_coeffs.tolist()
                    if self.last_tangent_ellipse_coeffs is not None else None
                ),
            },
            "ideal_metrics": ideal_met,
            "sources": source_entries,
            "files": {
                "desired_target_mask_npy": npy_path.name,
                "desired_target_mask_png": png_path.name,
                "overlay_preview": overlay_path.name,
            },
            "note": (
                "Observed masks are preserved unchanged. The desired target is a "
                "reproducible geometric idealization defined by the stored control points."
            ),
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        flat = {
            "target_id": target_id,
            "target_class": self.class_var.get(),
            "bottom_type": self.bottom_var.get(),
            "area_fraction": ideal_met["area_fraction"],
            "opening_width_px": ideal_met["opening_width_px"],
            "depth_px": ideal_met["depth_px"],
            "centroid_x_px": ideal_met["centroid_x_px"],
            "opening_TL_y_from_top_px": float(self.points["TL"].y),
            "opening_TR_y_from_top_px": float(self.points["TR"].y),
            "opening_mean_y_from_top_px": float(
                (self.points["TL"].y + self.points["TR"].y) / 2.0
            ),
            "source_count": len(self.mask_paths),
        }
        for name in ["TL", "BL", "BA", "BR", "TR"]:
            x_abs = int(round(self.points[name].x))
            y_abs = int(round(self.points[name].y))
            flat[f"{name}_x_px"] = x_abs
            flat[f"{name}_x_centered_px"] = x_abs - self.image_center_x_px()
            flat[f"{name}_y_px"] = y_abs

        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat))
            w.writeheader()
            w.writerow(flat)

        self.update_registry(flat, npy_path, json_path)
        messagebox.showinfo(
            "Saved",
            f"Saved ideal target:\n{target_dir}",
            parent=self,
        )

    def update_registry(self, flat: dict, mask_path: Path, json_path: Path):
        reg = self.output_dir / "ideal_target_registry.csv"
        fieldnames = [
            "target_id", "target_class", "bottom_type",
            "area_fraction", "opening_width_px", "depth_px",
            "centroid_x_px",
            "opening_TL_y_from_top_px",
            "opening_TR_y_from_top_px",
            "opening_mean_y_from_top_px",
            "source_count",
            "desired_mask_path", "parameter_json_path",
        ]
        row = {
            "target_id": flat["target_id"],
            "target_class": flat["target_class"],
            "bottom_type": flat["bottom_type"],
            "area_fraction": flat["area_fraction"],
            "opening_width_px": flat["opening_width_px"],
            "depth_px": flat["depth_px"],
            "centroid_x_px": flat["centroid_x_px"],
            "opening_TL_y_from_top_px": flat["opening_TL_y_from_top_px"],
            "opening_TR_y_from_top_px": flat["opening_TR_y_from_top_px"],
            "opening_mean_y_from_top_px": flat["opening_mean_y_from_top_px"],
            "source_count": flat["source_count"],
            "desired_mask_path": str(mask_path.resolve()),
            "parameter_json_path": str(json_path.resolve()),
        }

        rows = []
        if reg.exists():
            with reg.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            rows = [r for r in rows if r.get("target_id") != row["target_id"]]
        rows.append(row)

        with reg.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    def open_output_dir(self):
        try:
            import os
            os.startfile(str(self.output_dir))
        except Exception:
            messagebox.showinfo(
                "Output folder",
                str(self.output_dir),
                parent=self,
            )


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.mask = [p.resolve() for p in args.mask]
    if args.candidate_csv is not None:
        args.candidate_csv = args.candidate_csv.resolve()
    app = IdealTargetDesigner(args)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
