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
) -> tuple[np.ndarray, np.ndarray]:
    tl = np.array([points["TL"].x, points["TL"].y], dtype=float)
    bl = np.array([points["BL"].x, points["BL"].y], dtype=float)
    ba = np.array([points["BA"].x, points["BA"].y], dtype=float)
    br = np.array([points["BR"].x, points["BR"].y], dtype=float)
    tr = np.array([points["TR"].x, points["TR"].y], dtype=float)

    if bottom_type == "circle":
        bottom = circle_arc_through_points(bl, ba, br)
        effective_apex = ba.copy()
    elif bottom_type == "ellipse":
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
    return mask, effective_apex


class IdealTargetDesigner(tk.Tk):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.output_dir = args.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.title("Ideal Target Designer")
        self.geometry("1280x900")
        self.minsize(1180, 820)

        self.mask_paths: list[Path] = []
        self.raw_masks: list[np.ndarray] = []
        self.display_masks: list[np.ndarray] = []
        self.shape: Optional[tuple[int, int]] = None

        self.points: dict[str, ControlPoint] = {}
        self.drag_name: Optional[str] = None

        self.target_id_var = tk.StringVar(value=args.target_id or "T_IDEAL_01")
        self.class_var = tk.StringVar(value=args.target_class)
        self.bottom_var = tk.StringVar(value=CLASS_TO_BOTTOM[args.target_class])
        self.opacity_var = tk.DoubleVar(value=0.16)
        self.align_sources_var = tk.BooleanVar(value=True)
        self.lock_center_var = tk.BooleanVar(value=True)
        self.center_ref_var = tk.StringVar(value="Opening midpoint")
        self.show_fill_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Load 1–3 observed masks.")
        self.metrics_var = tk.StringVar(value="")

        self.display_size = int(args.canvas_size)
        self.photo_ref = None

        self._build_ui()

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

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=8)

        right = ttk.Frame(main, width=365)
        right.pack(side="right", fill="y", padx=(4, 8), pady=8)

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
            textvariable=self.status_var,
            anchor="w",
        ).pack(fill="x", pady=(5, 0))

        # Target identity
        box = ttk.LabelFrame(right, text="Target")
        box.pack(fill="x", pady=(0, 8))
        ttk.Label(box, text="Target ID").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(box, textvariable=self.target_id_var, width=28).grid(
            row=0, column=1, sticky="ew", padx=6, pady=4
        )
        ttk.Label(box, text="Class").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        class_cb = ttk.Combobox(
            box, textvariable=self.class_var, values=TARGET_CLASSES,
            state="readonly", width=25
        )
        class_cb.grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        class_cb.bind("<<ComboboxSelected>>", self.on_class_changed)
        box.columnconfigure(1, weight=1)

        # Source masks
        box = ttk.LabelFrame(right, text="Observed masks (max 3)")
        box.pack(fill="x", pady=(0, 8))
        self.source_list = tk.Listbox(box, height=4)
        self.source_list.pack(fill="x", padx=6, pady=(6, 4))
        btns = ttk.Frame(box)
        btns.pack(fill="x", padx=6, pady=(0, 5))
        ttk.Button(btns, text="Add masks…", command=self.add_masks_dialog).pack(side="left")
        ttk.Button(btns, text="Remove selected", command=self.remove_selected_mask).pack(
            side="left", padx=4
        )
        ttk.Button(btns, text="Clear", command=self.clear_masks).pack(side="left")

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
            op, from_=0.05, to=0.40, variable=self.opacity_var,
            orient="horizontal", command=lambda _=None: self.render()
        ).pack(side="left", fill="x", expand=True, padx=6)

        # Geometry
        box = ttk.LabelFrame(right, text="Geometry")
        box.pack(fill="x", pady=(0, 8))
        ttk.Label(box, text="Bottom").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        bottom_cb = ttk.Combobox(
            box,
            textvariable=self.bottom_var,
            values=["circle", "ellipse", "flat", "pointed"],
            state="readonly",
            width=18,
        )
        bottom_cb.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        bottom_cb.bind("<<ComboboxSelected>>", lambda e: self.render())

        ttk.Checkbutton(
            box,
            text="Show ideal-mask fill",
            variable=self.show_fill_var,
            command=self.render,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=3)

        ttk.Button(box, text="Auto initialize from masks", command=self.auto_initialize).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=3
        )
        ttk.Button(box, text="Level opening", command=self.level_opening).grid(
            row=3, column=0, sticky="ew", padx=6, pady=3
        )
        ttk.Button(box, text="Parallelize sidewalls", command=self.parallelize_sidewalls).grid(
            row=3, column=1, sticky="ew", padx=6, pady=3
        )
        box.columnconfigure(0, weight=1)
        box.columnconfigure(1, weight=1)

        # X centering
        box = ttk.LabelFrame(right, text="X centering")
        box.pack(fill="x", pady=(0, 8))
        ttk.Label(box, text="Reference").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ref = ttk.Combobox(
            box,
            textvariable=self.center_ref_var,
            values=["Opening midpoint", "Ideal centroid"],
            state="readonly",
            width=20,
        )
        ref.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(box, text="Center ideal target X", command=self.center_ideal_x).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=4
        )
        ttk.Checkbutton(
            box,
            text="Lock ideal target to X-center while editing",
            variable=self.lock_center_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 6))
        box.columnconfigure(1, weight=1)

        # Metrics
        box = ttk.LabelFrame(right, text="Live metrics")
        box.pack(fill="x", pady=(0, 8))
        ttk.Label(
            box,
            textvariable=self.metrics_var,
            justify="left",
            font=("Consolas", 9),
        ).pack(fill="x", padx=6, pady=6)

        # Save
        box = ttk.LabelFrame(right, text="Output")
        box.pack(fill="x")
        ttk.Button(box, text="Save ideal target", command=self.save_target).pack(
            fill="x", padx=6, pady=(6, 3)
        )
        ttk.Button(box, text="Open output folder", command=self.open_output_dir).pack(
            fill="x", padx=6, pady=(3, 6)
        )

        help_text = (
            "Drag handles:\n"
            "TL/TR = opening edges\n"
            "BL/BR = bottom-arc shoulders\n"
            "BA = bottom apex\n\n"
            "Rounded-bottom uses an ellipse.\n"
            "Near-parallel sidewall uses a circular bottom arc."
        )
        ttk.Label(right, text=help_text, justify="left").pack(fill="x", pady=(8, 0))

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
    def on_class_changed(self, _event=None):
        cls = self.class_var.get()
        self.bottom_var.set(CLASS_TO_BOTTOM[cls])
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

        if self.lock_center_var.get():
            self.center_ideal_x()
        else:
            self.render()

    def level_opening(self):
        if not self.points:
            return
        y = (self.points["TL"].y + self.points["TR"].y) / 2.0
        self.points["TL"].y = y
        self.points["TR"].y = y
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
        self.render()

    def current_ideal_mask(self) -> Optional[np.ndarray]:
        if not self.points or self.shape is None:
            return None
        mask, _ = target_polygon(self.points, self.bottom_var.get(), self.shape)
        return mask

    def center_ideal_x(self):
        if not self.points or self.shape is None:
            return
        h, w = self.shape
        desired = (w - 1) / 2.0

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

        dx = desired - current

        # Clip the shift so no control point leaves the image.
        xs0 = [p.x for p in self.points.values()]
        dx = max(dx, -min(xs0))
        dx = min(dx, (w - 1) - max(xs0))

        for p in self.points.values():
            p.x += dx
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

    def on_drag(self, event):
        if self.drag_name is None or self.shape is None:
            return
        x, y = self.canvas_to_image(event.x, event.y)
        h, w = self.shape
        p = self.points[self.drag_name]
        p.x = float(np.clip(x, 0, w - 1))
        p.y = float(np.clip(y, 0, h - 1))

        self.normalize_bottom_constraints(self.drag_name)
        if self.lock_center_var.get():
            # Center only in X; Y geometry is untouched.
            self.center_ideal_x()
        else:
            self.render()

    def on_release(self, _event):
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
        # Image center line.
        cx = ox + ((w - 1) / 2.0) * scale
        draw.line((cx, oy, cx, oy + rh), fill=(120, 120, 120, 160), width=1)

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

        return bg.convert("RGB")

    def update_metrics(self):
        ideal = self.current_ideal_mask()
        if ideal is None:
            self.metrics_var.set("")
            return
        met = mask_metrics(ideal)
        lines = [
            f"area       : {100*met['area_fraction']:.2f} %",
            f"opening    : {met['opening_width_px']:.1f} px",
            f"depth      : {met['depth_px']:.1f} px",
            f"centroid x : {met['centroid_x_px']:.2f} px",
        ]
        for i, src in enumerate(self.display_masks, 1):
            lines.append(f"IoU vs obs{i} : {iou(ideal, src):.3f}")
        self.metrics_var.set("\n".join(lines))

    def render(self):
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
            "schema_version": "1.0",
            "target_id": target_id,
            "target_class": self.class_var.get(),
            "idealization_term": "geometric idealization",
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
                k: {"x": float(v.x), "y": float(v.y)}
                for k, v in self.points.items()
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
            "source_count": len(self.mask_paths),
        }
        for name in ["TL", "BL", "BA", "BR", "TR"]:
            flat[f"{name}_x_px"] = self.points[name].x
            flat[f"{name}_y_px"] = self.points[name].y

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
            "centroid_x_px", "source_count",
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
    app = IdealTargetDesigner(args)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
