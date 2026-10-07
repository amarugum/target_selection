from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw

CATEGORIES = ["", "Typical", "Elongated", "Oblique", "Wide & short", "Other"]


def parse_args():
    p = argparse.ArgumentParser(description="GUI for staged visual target selection.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pulse", type=int, default=50)
    p.add_argument("--channel", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--exclude-csv", type=Path, default=None)
    p.add_argument("--exclude-hole-column", default="hole_uid")
    p.add_argument("--seed", type=int, default=260910)
    p.add_argument("--per-page", type=int, default=20)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--thumb-size", type=int, default=165)
    p.add_argument("--cluster-run", type=Path, default=None)
    p.add_argument("--cluster-k", type=int, default=4)
    p.add_argument("--iou-csv", type=Path, default=None)
    return p.parse_args()


def load_excluded(path: Optional[Path], column: str) -> set[str]:
    if path is None:
        return set()
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"{column!r} not found in {path}")
    return set(df[column].dropna().astype(str).str.strip())


def scan_samples(root: Path, pulse: int, excluded: set[str]) -> list[dict]:
    rows = []
    for d in sorted(p for p in root.glob("h*") if p.is_dir()):
        if d.name in excluded:
            continue
        target = d / f"p{pulse:04d}" / "target.npy"
        if target.exists():
            rows.append({
                "hole_uid": d.name,
                "pulse_number": pulse,
                "source_target_path": str(target.resolve()),
            })
    return rows


def load_mask(path: Path, channel: int, threshold: float) -> np.ndarray:
    a = np.asarray(np.load(path), dtype=np.float32)
    if a.ndim == 3:
        if not 0 <= channel < a.shape[0]:
            raise ValueError(f"channel={channel} unavailable: {path}, shape={a.shape}")
        a = a[channel]
    elif a.ndim != 2:
        raise ValueError(f"Unexpected target shape {a.shape}: {path}")
    return (a >= threshold).astype(np.uint8)


def descriptors(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return dict(area_fraction=0.0, width_px=0, height_px=0, aspect_h_over_w=0.0, center_shift_px=0.0)
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    width, height = xmax - xmin + 1, ymax - ymin + 1
    rows = []
    for y in range(ymin, ymax + 1):
        xx = np.flatnonzero(mask[y])
        if xx.size:
            rows.append((y, float(xx.mean())))
    rr = np.asarray(rows)
    shift = 0.0
    if len(rr) >= 6:
        yn = (rr[:, 0] - rr[:, 0].min()) / max(1.0, rr[:, 0].max() - rr[:, 0].min())
        top = rr[(yn >= 0.15) & (yn <= 0.35)]
        bot = rr[(yn >= 0.65) & (yn <= 0.85)]
        if len(top) and len(bot):
            shift = float(bot[:, 1].mean() - top[:, 1].mean())
    return {
        "area_fraction": float(mask.mean()),
        "width_px": width,
        "height_px": height,
        "aspect_h_over_w": float(height / max(1, width)),
        "center_shift_px": shift,
    }


def cluster_metadata(run: Optional[Path]) -> dict[int, dict[str, dict]]:
    out = {}
    if run is None or not run.exists():
        return out
    for kd in sorted(run.glob("K*")):
        if not kd.is_dir():
            continue
        try:
            k = int(kd.name[1:])
        except ValueError:
            continue
        p = kd / "cluster_assignments.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        m = {}
        for _, r in df.iterrows():
            uid = str(r.get("hole_uid", "")).strip()
            if uid:
                m[uid] = {
                    "cluster_id": r.get("cluster_id", ""),
                    "distance_percentile": r.get("distance_percentile", ""),
                    "is_nearest": r.get("is_nearest", ""),
                }
        out[k] = m
    return out


def iou_metadata(path: Optional[Path]) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        uid = str(r.get("source_hole_uid", "")).strip()
        if uid:
            out[uid] = str(r.get("target_id", ""))
    return out



def _as01(value) -> int:
    """Robustly parse CSV values such as 1/0, True/False, yes/no."""
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "selected"}:
        return 1
    if s in {"0", "false", "no", "n", "", "nan", "none"}:
        return 0
    try:
        return 1 if float(s) != 0 else 0
    except Exception:
        return 0


@dataclass
class Row:
    hole_uid: str
    pulse_number: int
    source_target_path: str
    display_order: int
    round1_selected: int = 0
    category: str = ""
    comment: str = ""
    round2_selected: int = 0
    final_candidate: int = 0


class Store:
    fields = list(Row.__dataclass_fields__.keys())

    def __init__(self, outdir: Path):
        self.outdir = outdir.resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.csv = self.outdir / "target_selection_review.csv"
        self.session = self.outdir / "target_selection_session.json"

    def load_or_init(self, samples: list[dict], seed: int, meta: dict) -> list[Row]:
        by = {s["hole_uid"]: s for s in samples}
        if self.csv.exists():
            df = pd.read_csv(self.csv, keep_default_na=False)
            rows = []
            seen = set()
            for _, r in df.iterrows():
                uid = str(r["hole_uid"])
                if uid not in by:
                    continue
                seen.add(uid)
                rows.append(Row(
                    hole_uid=uid,
                    pulse_number=int(r["pulse_number"]),
                    source_target_path=by[uid]["source_target_path"],
                    display_order=int(r["display_order"]),
                    round1_selected=_as01(r.get("round1_selected", 0)),
                    category=str(r["category"]),
                    comment=str(r["comment"]),
                    round2_selected=_as01(r.get("round2_selected", 0)),
                    final_candidate=_as01(r.get("final_candidate", 0)),
                ))
            next_order = max([x.display_order for x in rows], default=-1) + 1
            for uid in sorted(set(by) - seen):
                s = by[uid]
                rows.append(Row(uid, int(s["pulse_number"]), s["source_target_path"], next_order))
                next_order += 1
            rows.sort(key=lambda x: x.display_order)
            self.save(rows)
            return rows

        order = list(range(len(samples)))
        random.Random(seed).shuffle(order)
        rows = []
        for i, idx in enumerate(order):
            s = samples[idx]
            rows.append(Row(s["hole_uid"], int(s["pulse_number"]), s["source_target_path"], i))
        self.save(rows)
        self.session.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return rows

    def save(self, rows: list[Row]):
        tmp = self.csv.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writeheader()
            for r in sorted(rows, key=lambda x: x.display_order):
                w.writerow(asdict(r))
        tmp.replace(self.csv)


class Zoom(tk.Toplevel):
    def __init__(self, parent, row: Row, mask: np.ndarray):
        super().__init__(parent)
        self.title(row.hole_uid)
        im = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize((700, 700), Image.Resampling.NEAREST)
        self.photo = ImageTk.PhotoImage(im)
        ttk.Label(self, image=self.photo).pack(padx=10, pady=10)
        ttk.Label(self, text=f"{row.hole_uid}  p{row.pulse_number:04d}", font=("Segoe UI", 13, "bold")).pack(pady=(0, 10))


class App(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.title("Target Selection Review")
        self.geometry("1500x980")
        self.minsize(1200, 780)

        self.excluded = load_excluded(args.exclude_csv, args.exclude_hole_column)
        samples = scan_samples(args.data_root.resolve(), args.pulse, self.excluded)
        if not samples:
            raise RuntimeError("No samples found")

        self.store = Store(args.output_dir)
        session_meta = {
            "seed": args.seed,
            "pulse": args.pulse,
            "channel": args.channel,
            "threshold": args.threshold,
            "valid_samples": len(samples),
            "excluded_samples": len(self.excluded),
            "note": "Round-1 display order is fixed and persisted in target_selection_review.csv",
        }
        self.rows = self.store.load_or_init(samples, args.seed, session_meta)
        self.cmeta = cluster_metadata(args.cluster_run)
        self.imeta = iou_metadata(args.iou_csv)
        ks = sorted(self.cmeta)

        self.mode = tk.StringVar(value="Round 1")
        self.catfilter = tk.StringVar(value="All")
        self.reveal = tk.BooleanVar(value=False)
        self.k = tk.IntVar(value=args.cluster_k if args.cluster_k in ks else (ks[0] if ks else args.cluster_k))
        self.page = 0
        self.photos = []
        self.mask_cache = {}
        self.desc_cache = {}
        self.info = tk.StringVar()

        self._build(ks)
        self.refresh()
        self.bind("<Left>", lambda e: self.prev())
        self.bind("<Right>", lambda e: self.next())
        self.bind("<Control-s>", lambda e: self.save())

    def _build(self, ks):
        bar = ttk.Frame(self); bar.pack(fill="x", padx=8, pady=6)
        ttk.Label(bar, text="Mode").pack(side="left")
        cb = ttk.Combobox(bar, textvariable=self.mode, values=["Round 1", "Round 2", "Round 3"], state="readonly", width=10)
        cb.pack(side="left", padx=(4, 12)); cb.bind("<<ComboboxSelected>>", lambda e: self.change_mode())
        ttk.Label(bar, text="Category").pack(side="left")
        cf = ttk.Combobox(bar, textvariable=self.catfilter, values=["All"] + CATEGORIES[1:], state="readonly", width=15)
        cf.pack(side="left", padx=(4, 12)); cf.bind("<<ComboboxSelected>>", lambda e: self.reset())
        ttk.Checkbutton(bar, text="Reveal metadata", variable=self.reveal, command=self.refresh).pack(side="left", padx=(0, 12))
        if ks:
            ttk.Label(bar, text="Cluster K").pack(side="left")
            kc = ttk.Combobox(bar, textvariable=self.k, values=ks, state="readonly", width=5)
            kc.pack(side="left", padx=(4, 12)); kc.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        ttk.Button(bar, text="Export shortlist", command=self.export).pack(side="right", padx=3)
        ttk.Button(bar, text="Start Round 3", command=lambda: self.goto_round("Round 3")).pack(side="right", padx=3)
        ttk.Button(bar, text="Start Round 2", command=lambda: self.goto_round("Round 2")).pack(side="right", padx=3)
        ttk.Button(bar, text="Save", command=self.save).pack(side="right", padx=8)
        ttk.Button(bar, text="Next", command=self.next).pack(side="right", padx=3)
        ttk.Button(bar, text="Prev", command=self.prev).pack(side="right", padx=3)

        ttk.Label(self, textvariable=self.info).pack(fill="x", padx=8, pady=(0, 4))

        self.canvas = tk.Canvas(self, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.canvas.pack(side="left", fill="both", expand=True)
        self.grid = ttk.Frame(self.canvas)
        self.win = self.canvas.create_window((0, 0), window=self.grid, anchor="nw")
        self.grid.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.win, width=e.width))
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))

    def current_rows(self):
        if self.mode.get() == "Round 1":
            xs = list(self.rows)
        elif self.mode.get() == "Round 2":
            # Round 2 input is exactly the Round 1 shortlist.
            xs = [r for r in self.rows if _as01(r.round1_selected) == 1]
        else:
            # Round 3 input is exactly the Round 2 shortlist.
            xs = [r for r in self.rows if _as01(r.round2_selected) == 1]

        if self.catfilter.get() != "All":
            xs = [r for r in xs if r.category == self.catfilter.get()]
        return sorted(xs, key=lambda x: x.display_order)

    def mask(self, r):
        if r.hole_uid not in self.mask_cache:
            self.mask_cache[r.hole_uid] = load_mask(Path(r.source_target_path), self.args.channel, self.args.threshold)
        return self.mask_cache[r.hole_uid]

    def desc(self, r):
        if r.hole_uid not in self.desc_cache:
            self.desc_cache[r.hole_uid] = descriptors(self.mask(r))
        return self.desc_cache[r.hole_uid]

    def selected(self, r):
        return bool(r.round1_selected if self.mode.get() == "Round 1" else r.round2_selected if self.mode.get() == "Round 2" else r.final_candidate)

    def toggle(self, r):
        if self.mode.get() == "Round 1":
            r.round1_selected = 0 if r.round1_selected else 1
            if not r.round1_selected:
                r.round2_selected = 0; r.final_candidate = 0
        elif self.mode.get() == "Round 2":
            r.round2_selected = 0 if r.round2_selected else 1
            if not r.round2_selected:
                r.final_candidate = 0
        else:
            r.final_candidate = 0 if r.final_candidate else 1
        self.store.save(self.rows); self.refresh()

    def setcat(self, r, v):
        r.category = v; self.store.save(self.rows); self.refresh()

    def comment(self, r):
        v = simpledialog.askstring("Comment", f"Comment for {r.hole_uid}", initialvalue=r.comment, parent=self)
        if v is not None:
            r.comment = v.strip(); self.store.save(self.rows); self.refresh()

    def refresh(self):
        for c in self.grid.winfo_children():
            c.destroy()
        self.photos = []

        # Recount from state every refresh so the UI and CSV cannot silently diverge.
        r1 = sum(_as01(x.round1_selected) for x in self.rows)
        r2 = sum(_as01(x.round2_selected) for x in self.rows)
        fin = sum(_as01(x.final_candidate) for x in self.rows)

        xs = self.current_rows()
        npage = max(1, math.ceil(len(xs) / self.args.per_page))
        self.page = min(max(self.page, 0), npage - 1)
        page_items = xs[self.page*self.args.per_page:(self.page+1)*self.args.per_page]

        if not xs:
            if self.mode.get() == "Round 2":
                msg = (
                    "Round 2 has no input candidates.\n\n"
                    f"Round 1 selected count = {r1}\n"
                    "Return to Round 1 and select candidates by clicking the images.\n"
                    "Also make sure the Category filter is set to All."
                )
            elif self.mode.get() == "Round 3":
                msg = (
                    "Round 3 has no input candidates.\n\n"
                    f"Round 2 selected count = {r2}\n"
                    "Return to Round 2 and select candidates by clicking the images.\n"
                    "Also make sure the Category filter is set to All."
                )
            else:
                msg = "No valid samples are available."
            ttk.Label(
                self.grid,
                text=msg,
                justify="center",
                font=("Segoe UI", 14),
            ).grid(row=0, column=0, padx=60, pady=80)
        else:
            for i, r in enumerate(page_items):
                self.card(r, i // self.args.cols, i % self.args.cols)

        if self.mode.get() == "Round 1":
            input_txt = f"input={len(self.rows)}"
        elif self.mode.get() == "Round 2":
            input_txt = f"input from R1={r1}"
        else:
            input_txt = f"input from R2={r2}"

        self.info.set(
            f"{self.mode.get()} | {input_txt} | page {self.page+1}/{npage} | "
            f"shown {len(page_items)}/{len(xs)} | R1={r1} R2={r2} Final={fin} | "
            f"valid={len(self.rows)} excluded={len(self.excluded)}"
        )
        self.canvas.yview_moveto(0)

    def card(self, r, rr, cc):
        sel = self.selected(r)
        frame = tk.Frame(self.grid, bd=3 if sel else 1, relief="solid", background="#d7ffd7" if sel else "#eeeeee")
        frame.grid(row=rr, column=cc, padx=5, pady=5, sticky="nsew"); self.grid.grid_columnconfigure(cc, weight=1)
        im = Image.fromarray((self.mask(r)*255).astype(np.uint8), mode="L")
        im.thumbnail((self.args.thumb_size, self.args.thumb_size), Image.Resampling.NEAREST)
        ph = ImageTk.PhotoImage(im); self.photos.append(ph)
        b = tk.Button(frame, image=ph, bd=0, command=lambda x=r:self.toggle(x), cursor="hand2")
        b.pack(padx=4, pady=(4,2)); b.bind("<Double-Button-1>", lambda e,x=r:Zoom(self,x,self.mask(x)))
        ttk.Label(frame, text=f"{r.hole_uid}{'  ✓' if sel else ''}", font=("Segoe UI",10,"bold")).pack()
        cat = ttk.Combobox(frame, values=CATEGORIES, state="readonly", width=15); cat.set(r.category); cat.pack(pady=2)
        cat.bind("<<ComboboxSelected>>", lambda e,x=r,c=cat:self.setcat(x,c.get()))
        ttk.Button(frame, text="Comment" + (" *" if r.comment else ""), command=lambda x=r:self.comment(x)).pack(pady=(1,3))
        # Round 1 is intentionally blind even when reveal is checked.
        if self.reveal.get() and self.mode.get() != "Round 1":
            d = self.desc(r); lines = [f"area={100*d['area_fraction']:.1f}% H/W={d['aspect_h_over_w']:.2f}", f"shift={d['center_shift_px']:+.1f}px"]
            cm = self.cmeta.get(self.k.get(), {}).get(r.hole_uid)
            if cm:
                try: pct = f"{100*float(cm['distance_percentile']):.0f}%"
                except Exception: pct = str(cm['distance_percentile'])
                lines.append(f"K{self.k.get()}: C{cm['cluster_id']} dist-pct={pct}")
            if r.hole_uid in self.imeta:
                lines.append(f"IoU: {self.imeta[r.hole_uid]}")
            ttk.Label(frame, text="\n".join(lines), justify="left", font=("Consolas",8)).pack(padx=4,pady=(0,4))

    def change_mode(self):
        # A category filter from the previous round is a common reason for an
        # apparently empty next round. Always start a new round with All.
        self.catfilter.set("All")
        self.page = 0
        self.store.save(self.rows)
        self.refresh()

    def goto_round(self, mode: str):
        if mode == "Round 2":
            n = sum(_as01(x.round1_selected) for x in self.rows)
            if n == 0:
                messagebox.showwarning(
                    "Round 2",
                    "No Round 1 candidates are selected yet.",
                    parent=self,
                )
                return
        if mode == "Round 3":
            n = sum(_as01(x.round2_selected) for x in self.rows)
            if n == 0:
                messagebox.showwarning(
                    "Round 3",
                    "No Round 2 candidates are selected yet.",
                    parent=self,
                )
                return
        self.mode.set(mode)
        self.change_mode()

    def reset(self):
        self.page = 0
        self.refresh()
    def prev(self):
        if self.page > 0: self.page -= 1; self.refresh()
    def next(self):
        xs = self.current_rows(); npage = max(1, math.ceil(len(xs)/self.args.per_page))
        if self.page < npage-1: self.page += 1; self.refresh()
    def save(self): self.store.save(self.rows); messagebox.showinfo("Saved", str(self.store.csv), parent=self)

    def export(self):
        out = self.store.outdir / "exports"; out.mkdir(exist_ok=True)
        df = pd.DataFrame([asdict(r) for r in self.rows])
        df[df.round1_selected == 1].to_csv(out/"round1_shortlist.csv", index=False, encoding="utf-8-sig")
        df[df.round2_selected == 1].to_csv(out/"round2_shortlist.csv", index=False, encoding="utf-8-sig")
        df[df.final_candidate == 1].to_csv(out/"final_candidates.csv", index=False, encoding="utf-8-sig")
        finals = [r for r in self.rows if r.final_candidate]
        if finals: self.contact(finals, out/"final_candidates.png")
        messagebox.showinfo("Export", f"Exported to\n{out}", parent=self)

    def contact(self, rows, out):
        cols = min(5, max(1, len(rows))); nrows = math.ceil(len(rows)/cols); tile=220; lh=45
        canvas = Image.new("RGB", (cols*tile, nrows*(tile+lh)), "white"); draw = ImageDraw.Draw(canvas)
        for i,r in enumerate(rows):
            rr,cc=divmod(i,cols); x,y=cc*tile,rr*(tile+lh)
            im=Image.fromarray((self.mask(r)*255).astype(np.uint8),mode="L").convert("RGB"); im.thumbnail((tile-10,tile-10),Image.Resampling.NEAREST)
            bg=Image.new("RGB",(tile,tile),"black"); bg.paste(im,((tile-im.width)//2,(tile-im.height)//2)); canvas.paste(bg,(x,y))
            draw.text((x+5,y+tile+5),f"{r.hole_uid}  {r.category}",fill="black")
        canvas.save(out,dpi=(160,160))


def main():
    a = parse_args()
    for name in ["data_root","output_dir","exclude_csv","cluster_run","iou_csv"]:
        v = getattr(a,name)
        if v is not None: setattr(a,name,v.resolve())
    app = App(a); app.mainloop(); return 0

if __name__ == "__main__":
    raise SystemExit(main())
