#!/usr/bin/env python3
"""
NOVA - Numerical Omnitechnique Visualization and Analysis
One GUI for FTIR, Raman, UV-Vis, NMR, PXRD and generic XY (e.g. dissolution) data.

Run:      python nova.py
Requires: numpy pandas matplotlib   (optional: scipy for Savitzky-Golay + peak finding,
          openpyxl for .xlsx).  tkinter ships with python.org Python.

Files: .csv .txt .dat .xy .tsv .jdx .dx .xlsx .xls
  - column 1 = x, every further column = one spectrum (UV-Vis time series etc.)
  - header/junk lines are skipped automatically
  - XY mode + "col 3 = error bars": columns are read as (y, err) pairs
Blanks: load the blank like any spectrum, tick "Is blank", then assign it to a sample
(or to all samples) -> processed = sample - scale * blank (blank interpolated onto sample x).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser, simpledialog

try:
    from scipy.signal import savgol_filter, find_peaks
except ImportError:
    savgol_filter = find_peaks = None
try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from scipy.interpolate import PchipInterpolator
except ImportError:
    sparse = spsolve = PchipInterpolator = None

# technique: (x label, y label, reverse x, default normalisation, peak decimals)
TECH = {
    "FTIR":   ("Wavenumber (cm$^{-1}$)", "Transmittance (%)", True,  "None", 0),
    "Raman":  ("Raman shift (cm$^{-1}$)", "Intensity (a.u.)", False, "None", 0),
    "UV-Vis": ("Wavelength (nm)", "Absorbance", False, "None", 0),
    "NMR":    ("Chemical shift $\\delta$ (ppm)", "Intensity (a.u.)", True, "None", 2),
    "PXRD":   ("2$\\theta$ (degrees)", "Intensity (a.u.)", False, "Max", 1),
    "XY":     ("Time (min)", "Cumulative Dissolution (%)", False, "None", 2),
}
PALETTE = ["#000000", "#d62728", "#006ED5", "#05be39", "#DB0BA0", "#E69F00", "#2ca02c", "#9467bd"]
MARKERS = "osv^D<>p"
# time (3rd dimension) is entered manually by the user, stored in seconds
UNIT = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}
RADS = ["CuKa", "CoKa", "MoKa", "FeKa", "CrKa", "AgKa", "Custom (set λ)"]
REFPAL = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#17becf"]
REFMODES = ["Sub-panels", "Tick marks under pattern", "Lines through pattern"]
HINT = {"PXRD": "PXRD: load the pattern, then 'Add PXRD reference...' to import .cif (needs pymatgen) or .xy stick lists.",
        "UV-Vis": "UV-Vis: load ONE file - wavelength in column 1, then one absorbance column per sample "
                  "(header = sample name). Set times manually in 'Time (3rd axis)'."}
TIME_MODES = ("3D", "Heatmap")
GRID_STYLES = ("Surface", "Wireframe", "Surface + contour", "Scatter points")
STYLES_3D = ("Waterfall (lines)", "Filled waterfall") + GRID_STYLES


# UI look
BG, NAVY, ACCENT, INK, MUTED, LINE = "#f5f7fb", "#0b1f3a", "#2f6fed", "#1c2536", "#4a5568", "#d9dfeb"
FONT = "Helvetica Neue" if sys.platform == "darwin" else ("Segoe UI" if sys.platform.startswith("win") else "DejaVu Sans")
NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


class Dataset:
    def __init__(self, x, y, label, color, kind="data", yerr=None):
        self.x, self.y, self.yerr = np.asarray(x, float), np.asarray(y, float), yerr
        self.label, self.color, self.kind = label, color, kind
        self.visible, self.is_blank = True, False
        self.xshift, self.yscale, self.bscale, self.blank = 0.0, 1.0, 1.0, None
        self.t = None  # time in seconds (3rd dimension)
        self.cif, self.hkl, self.ckey = None, None, None  # CIF-derived reference info
        self.marks = []  # manual peak labels: [x, text]


# ----------------------------------------------------------------- file reading
MISSING = {"", "nan", "na", "n/a", "null", "-"}


def read_any(path):
    """-> x (n,), Y (n,k), names [k or None].
    Standard layout: column 1 = x (e.g. wavelength), every other column = one sample (header = sample name).
    Tolerates header/junk lines, empty cells (-> NaN) and numeric headers (e.g. times)."""
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        rows = [["" if pd.isna(v) else v for v in r] for r in pd.read_excel(p, header=None).itertuples(index=False)]
    else:
        def split(ln):
            ln = ln.rstrip("\r\n")
            if "\t" in ln:
                return [t.strip() for t in ln.split("\t")]
            if "," in ln or ";" in ln:
                return [t.strip() for t in re.split(r"[,;]", ln)]
            return ln.split()
        rows = [split(ln) for ln in p.read_bytes().decode("utf-8-sig", errors="replace").splitlines()]
    ok = lambda t: str(t).strip().lower() in MISSING or NUM.match(str(t).strip())
    isnum = lambda r: len(r) >= 2 and NUM.match(str(r[0]).strip()) and all(ok(t) for t in r[1:])
    data = [r for r in rows if isnum(r)]
    if not data:
        raise ValueError("No numeric data found. Expected: column 1 = x, further columns = samples.")
    lens = [len(r) for r in data]
    k = max(set(lens), key=lens.count)
    first = next(i for i, r in enumerate(rows) if isnum(r) and len(r) == k)
    f = lambda t: float(t) if NUM.match(str(t).strip()) else np.nan
    A = np.array([[f(t) for t in r] for r in data if len(r) == k])
    names = [None] * (k - 1)
    hdr = [str(t).strip() for t in rows[first - 1]] if first > 0 else []
    if len(hdr) == k:
        names = [h or None for h in hdr[1:]]
    if A.shape[0] > 3:  # numeric header row (e.g. 0, 5, 10, 15 min)?
        dx = np.median(np.abs(np.diff(A[1:, 0])))
        if dx > 0 and abs(A[1, 0] - A[0, 0]) > 10 * dx:
            names = [f"{v:g}" if np.isfinite(v) else None for v in A[0, 1:]]
            A = A[1:]
    order = np.argsort(A[:, 0])
    return A[order, 0], A[order, 1:], names


def cif_pattern(path, wavelength):
    """Simulated powder pattern from a .cif -> (2theta, intensity 0-100, hkl labels, label). Needs pymatgen."""
    import warnings
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = Structure.from_file(path)
        pat = XRDCalculator(wavelength=wavelength).get_pattern(st, scaled=True, two_theta_range=(0, 150))
    bar = lambda i: str(i) if i >= 0 else "\\bar{%d}" % abs(i)
    hkl = ["$(" + "".join(bar(int(i)) for i in h[0]["hkl"]) + ")$" for h in pat.hkls]
    return np.array(pat.x, float), np.array(pat.y, float), hkl, f"{Path(path).stem} ({st.composition.reduced_formula})"


def fmt_hkl(t):
    """'111' / '(1 -1 0)' / '-1,1,0' -> mathtext '$(1\\bar{1}0)$'. Anything else is shown as typed."""
    t = t.strip()
    if t.startswith("$"):
        return t
    core = t.strip("() ")
    if re.fullmatch(r"[-\d\s,]+", core):
        idx = re.findall(r"-?\d+", core) if re.search(r"[\s,]", core) else re.findall(r"-?\d", core)
        if len(idx) in (3, 4):
            return "$(" + "".join(("\\bar{%d}" % abs(int(i))) if int(i) < 0 else str(int(i)) for i in idx) + ")$"
    return t


def tag(ax, text, xy, color, G, below=False, xycoords="data"):
    ax.annotate(text, xy, xycoords=xycoords, xytext=(0, -2 if below else 2), textcoords="offset points",
                rotation=90, ha="center", va="top" if below else "bottom", fontsize=max(G["font"] - 5, 6), color=color)


# ----------------------------------------------------------------- processing
def modpoly(x, y, n, it=50):
    xs, yy = (x - x.mean()) / (np.ptp(x) or 1), y.copy()
    for _ in range(it):
        yy = np.minimum(yy, np.polyval(np.polyfit(xs, yy, n), xs))
    return np.polyval(np.polyfit(xs, yy, n), xs)


BG_MODES = ["None", "Offset to min", "Constant at x₀", "Linear (endpoints)", "Polynomial", "Rubberband",
            "SNIP", "ALS (asymmetric LS)", "Anchor points"]


def rubberband(x, y):
    """Lower convex hull of the spectrum (x ascending)."""
    h = []
    for i in range(len(x)):
        while len(h) >= 2 and (y[h[-1]] - y[h[-2]]) * (x[i] - x[h[-2]]) >= (y[i] - y[h[-2]]) * (x[h[-1]] - x[h[-2]]):
            h.pop()
        h.append(i)
    return np.interp(x, x[h], y[h])


def snip(y, iters):
    """SNIP clipping baseline (good for XRD / Raman)."""
    y0 = y.min()
    w = np.log(np.log(np.sqrt(y - y0 + 1) + 1) + 1)
    for p in range(1, int(iters) + 1):
        if 2 * p >= len(w): break
        a = w.copy()
        a[p:-p] = np.minimum(w[p:-p], (w[:-2 * p] + w[2 * p:]) / 2)
        w = a
    return (np.exp(np.exp(w) - 1) - 1) ** 2 - 1 + y0


def als(x, y, lam, p, it=10):
    """Asymmetric least squares (Eilers & Boelens). Falls back to a polynomial without scipy."""
    if sparse is None:
        return modpoly(x, y, 5)
    n = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2))
    DD, w = lam * D.dot(D.T), np.ones(n)
    for _ in range(it):
        z = spsolve((sparse.spdiags(w, 0, n, n) + DD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def baseline(x, y, G):
    """Background estimate for one spectrum, according to the chosen method."""
    m, sg = G["bgmode"], (-1.0 if G["bgdown"] else 1.0)
    yy = sg * y
    if m == "None":
        return np.zeros_like(y)
    if m == "Offset to min":
        b = np.full_like(yy, yy.min())
    elif m.startswith("Constant"):
        if G["bgx0"] is None:
            raise ValueError("Background: enter a reference x\u2080 value.")
        b = np.full_like(yy, yy[np.argmin(np.abs(x - G["bgx0"]))])
    elif m.startswith("Linear"):
        k = max(1, min(3, len(x) // 10))
        b = np.polyval(np.polyfit([x[:k].mean(), x[-k:].mean()], [yy[:k].mean(), yy[-k:].mean()], 1), x)
    elif m == "Polynomial":
        b = modpoly(x, yy, int(G["poly"]))
    elif m == "Rubberband":
        b = rubberband(x, yy)
    elif m == "SNIP":
        b = snip(yy, G["snip_it"])
    elif m.startswith("ALS"):
        b = als(x, yy, 10 ** G["als_lam"], G["als_p"])
    else:  # Anchor points
        a = sorted(float(t) for t in re.split(r"[,;\s]+", G["anchors"].strip()) if NUM.match(t))
        if len(a) < 2:
            raise ValueError("Background: enter at least 2 anchor x-values (comma-separated).")
        idx = [int(np.argmin(np.abs(x - v))) for v in a]
        ax_, ay_ = x[idx], np.array([yy[max(i - 2, 0):i + 3].mean() for i in idx])
        keep = np.concatenate([[True], np.diff(ax_) > 0])
        b = (PchipInterpolator(ax_[keep], ay_[keep])(x) if PchipInterpolator and keep.sum() > 2
             else np.interp(x, ax_[keep], ay_[keep]))
    return sg * b


def pick_peaks(y, prom, minima=False):
    s = -y if minima else y
    if find_peaks:
        return find_peaks(s, prominence=prom)[0]
    i = np.where((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]) & (s[1:-1] - s.min() > prom))[0] + 1
    return i


def process(d, G):
    x, y, e = d.x + d.xshift, d.y.copy(), d.yerr
    b = d.blank
    if b is not None and b is not d:
        y = y - d.bscale * np.interp(x, b.x + b.xshift, b.y)
    if G["convert"].startswith("%T"):
        y = -np.log10(np.clip(y, 1e-6, None) / 100)
    elif G["convert"].startswith("Abs"):
        y = 100 * 10 ** (-y)
    m = np.ones(len(x), bool)
    if G["xmin"] is not None: m &= x >= G["xmin"]
    if G["xmax"] is not None: m &= x <= G["xmax"]
    if m.sum() < 3:
        raise ValueError(f"X-range leaves <3 points for '{d.label}'.")
    x, y = x[m], y[m]
    e = None if e is None else np.asarray(e)[m]
    y = y - baseline(x, y, G)
    if G["bgclip"]:
        y = np.clip(y, 0, None)
    w = int(G["smooth"])
    if w >= 3:
        w += (w % 2 == 0)
        w = min(w, len(y) - (len(y) % 2 == 0))
        y = savgol_filter(y, w, min(3, w - 1)) if savgol_filter else np.convolve(y, np.ones(w) / w, "same")
    if G["deriv"]:
        y = np.gradient(y, x)
    nm = G["norm"]
    if nm == "Max":
        y = y / (np.abs(y).max() or 1)
    elif nm == "Min-max":
        y = (y - y.min()) / (np.ptp(y) or 1)
    elif nm == "Area":
        y = y / (abs(np.sum(np.abs(np.diff(x)) * (y[:-1] + y[1:]) / 2)) or 1)
    y = y * d.yscale
    return x, y, e


# ----------------------------------------------------------------- GUI
class Nova:
    def __init__(self, root):
        self.root, self.ds, self._job, self._loading = root, [], None, False
        self.curves, self.main_ax, self._view = [], None, None
        root.title("NOVA - Numerical Omnitechnique Visualization and Analysis")
        root.geometry("1380x860")
        self.vars = {}
        self._style()
        self._build()
        self.tech_changed()

    def _style(self):
        s = ttk.Style(); s.theme_use("clam")
        s.configure(".", background=BG, foreground=INK, font=(FONT, 10))
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=MUTED)
        s.configure("TLabelframe", background=BG, bordercolor=LINE, relief="solid", borderwidth=1)
        s.configure("TLabelframe.Label", background=BG, foreground=NAVY, font=(FONT, 10, "bold"))
        s.configure("TButton", background="#e6eaf2", foreground=INK, borderwidth=0, padding=(10, 6), focuscolor="#e6eaf2")
        s.map("TButton", background=[("active", "#d5dcea"), ("pressed", "#c6cfe2")])
        s.configure("Accent.TButton", background=ACCENT, foreground="white", font=(FONT, 10, "bold"), focuscolor=ACCENT)
        s.map("Accent.TButton", background=[("active", "#1f5bd6"), ("pressed", "#1a4fbd")])
        s.configure("TEntry", fieldbackground="white", bordercolor=LINE, lightcolor=LINE, darkcolor=LINE, padding=4)
        s.configure("TCombobox", fieldbackground="white", background="white", bordercolor=LINE,
                    lightcolor=LINE, darkcolor=LINE, arrowcolor=NAVY, padding=4)
        s.map("TCombobox", fieldbackground=[("readonly", "white")], foreground=[("readonly", INK)],
              selectbackground=[("readonly", "white")], selectforeground=[("readonly", INK)])
        s.configure("TCheckbutton", background=BG, foreground=INK)
        s.map("TCheckbutton", background=[("active", BG)])
        s.configure("Treeview", background="white", fieldbackground="white", rowheight=24, borderwidth=0)
        s.map("Treeview", background=[("selected", "#dbe6fd")], foreground=[("selected", NAVY)])
        s.configure("Treeview.Heading", background="#e6eaf2", foreground=NAVY, relief="flat", font=(FONT, 9, "bold"))
        s.configure("Vertical.TScrollbar", background="#cfd6e4", troughcolor=BG, borderwidth=0, arrowsize=12)
        self.root.configure(bg=BG)

    # ---- helpers
    def V(self, name, kind, val):
        v = {"s": tk.StringVar, "b": tk.BooleanVar}[kind](value=val)
        v.trace_add("write", self.sched)
        self.vars[name] = v
        return v

    def sched(self, *_):
        if self._loading:
            return
        if self._job:
            self.root.after_cancel(self._job)
        self._job = self.root.after(150, self.draw)

    def status(self, msg, err=False):
        self.lbl.config(text=msg, fg="#b00020" if err else INK)

    def row(self, parent, text, widget_fn):
        f = ttk.Frame(parent); f.pack(fill="x", pady=1)
        ttk.Label(f, text=text, width=15).pack(side="left")
        w = widget_fn(f); w.pack(side="left", fill="x", expand=True)
        return w

    def entry(self, p, text, name, val, cmd=None):
        v = cmd or self.V(name, "s", val)
        return self.row(p, text, lambda f: ttk.Entry(f, textvariable=v, width=10))

    def combo(self, p, text, name, vals, val):
        v = self.V(name, "s", val)
        return self.row(p, text, lambda f: ttk.Combobox(f, textvariable=v, values=vals, state="readonly", width=18))

    def check(self, p, text, name, val=False):
        ttk.Checkbutton(p, text=text, variable=self.V(name, "b", val)).pack(anchor="w")

    def box(self, p, title):
        f = ttk.LabelFrame(p, text=title, padding=5); f.pack(fill="x", padx=4, pady=3)
        return f

    # ---- layout
    def _build(self):
        head = tk.Frame(self.root, bg=NAVY)
        head.pack(side="top", fill="x")
        tk.Label(head, text="NOVA", font=(FONT, 24, "bold"), fg="white", bg=NAVY).pack(side="left", padx=(20, 14), pady=10)
        tk.Frame(head, width=1, bg="#3a4f73").pack(side="left", fill="y", pady=14)
        tk.Label(head, text="Numerical Omnitechnique Visualization and Analysis",
                 font=(FONT, 12), fg="#aebfdc", bg=NAVY).pack(side="left", padx=14)
        tk.Frame(self.root, height=3, bg=ACCENT).pack(side="top", fill="x")
        left = ttk.Frame(self.root); left.pack(side="left", fill="y")
        cv = tk.Canvas(left, width=345, highlightthickness=0, bg=BG)
        sb = ttk.Scrollbar(left, command=cv.yview)
        cv.configure(yscrollcommand=sb.set); sb.pack(side="right", fill="y"); cv.pack(side="left", fill="y")
        p = ttk.Frame(cv); cv.create_window((0, 0), window=p, anchor="nw")
        p.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        div = 1 if sys.platform == "darwin" else 120
        cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(int(-e.delta / div) or (-1 if e.delta > 0 else 1), "units"))

        b = self.box(p, "1. Technique & data")
        v = self.V("tech", "s", "FTIR"); v.trace_add("write", lambda *_: self.tech_changed())
        self.row(b, "Technique", lambda f: ttk.Combobox(f, textvariable=v, values=list(TECH), state="readonly", width=18))
        self.check(b, "XY: columns are (y, error) pairs", "errcol")
        bf = ttk.Frame(b); bf.pack(fill="x", pady=2)
        ttk.Button(bf, text="Load data...", style="Accent.TButton", command=self.load).pack(side="left", expand=True, fill="x")
        ttk.Button(bf, text="Add PXRD reference...", command=lambda: self.load(ref=True)).pack(side="left", expand=True, fill="x")

        b = self.box(p, "2. Datasets (double-click = show/hide)")
        self.tree = ttk.Treeview(b, columns=("v", "name", "role"), show="headings", height=7, selectmode="browse")
        for c, t, w in (("v", "", 25), ("name", "Name", 190), ("role", "Role", 60)):
            self.tree.heading(c, text=t); self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.tree.bind("<Double-1>", self.toggle_vis)
        bf = ttk.Frame(b); bf.pack(fill="x", pady=2)
        for t, c in (("↑", lambda: self.move(-1)), ("↓", lambda: self.move(1)), ("Colour", self.pick_color),
                     ("Remove", self.remove), ("Clear", self.clear)):
            ttk.Button(bf, text=t, width=7, command=c).pack(side="left")

        b = self.box(p, "3. Selected dataset")
        self.sel = {k: tk.StringVar() for k in ("label", "xshift", "yscale", "bscale", "blank", "time")}
        self.sel_isblank = tk.BooleanVar()
        for k, t in (("label", "Label"), ("xshift", "X shift"), ("yscale", "Y scale"), ("bscale", "Blank scale"), ("time", "Time (3rd axis)")):
            self.row(b, t, lambda f, k=k: ttk.Entry(f, textvariable=self.sel[k], width=10))
        self.blank_cb = self.row(b, "Subtract blank", lambda f: ttk.Combobox(f, textvariable=self.sel["blank"], state="readonly", width=18))
        ttk.Checkbutton(b, text="Is blank (hidden unless 'show blanks')", variable=self.sel_isblank).pack(anchor="w")
        for v in list(self.sel.values()) + [self.sel_isblank]:
            v.trace_add("write", self.sel_changed)
        ttk.Button(b, text="Use selected as blank for ALL samples", command=self.blank_all).pack(fill="x")

        b = self.box(p, "4. Processing (all datasets)")
        self.combo(b, "Convert", "convert", ["None", "%T → Absorbance", "Absorbance → %T"], "None")
        self.entry(b, "Smooth window", "smooth", "0")
        self.combo(b, "Normalise", "norm", ["None", "Max", "Min-max", "Area"], "None")
        self.check(b, "1st derivative", "deriv")
        self.entry(b, "X min (crop)", "xmin", "")
        self.entry(b, "X max (crop)", "xmax", "")

        b = self.box(p, "5. Background subtraction")
        self.combo(b, "Method", "bgmode", BG_MODES, "None")
        self.check(b, "Peaks point down (e.g. %T dips)", "bgdown")
        self.entry(b, "Poly order", "poly", "3")
        self.entry(b, "ALS λ (log10)", "als_lam", "6")
        self.entry(b, "ALS asymmetry", "als_p", "0.01")
        self.entry(b, "SNIP iterations", "snip_it", "40")
        self.entry(b, "Anchor x-values", "anchors", "")
        self.entry(b, "Reference x₀", "bgx0", "")
        self.check(b, "Clip negatives to 0", "bgclip")
        ttk.Button(b, text="Preview background...", command=self.preview_bg).pack(fill="x", pady=2)

        b = self.box(p, "6. Peaks")
        self.check(b, "Label peaks", "peaks")
        self.check(b, "Find minima (FTIR %T dips)", "minima")
        self.entry(b, "Prominence (%)", "prom", "5")

        b = self.box(p, "7. Manual peak labels")
        self.check(b, "Click plot to label peaks", "labelmode")
        ttk.Label(b, text="Left-click a peak: add / edit label.  Right-click a label: remove.\n"
                          "Type 111 or -1 1 0 for hkl; other text is shown as typed.", wraplength=300).pack(anchor="w")
        self.entry(b, "Rotation (°)", "markrot", "90")
        self.entry(b, "Match tol. (x units)", "mtol", "0.3")
        self.check(b, "Leader lines", "leader", True)
        bf = ttk.Frame(b); bf.pack(fill="x", pady=2)
        ttk.Button(bf, text="Auto-assign from refs", command=self.auto_assign).pack(side="left", expand=True, fill="x")
        bf = ttk.Frame(b); bf.pack(fill="x")
        ttk.Button(bf, text="Clear (selected)", command=lambda: self.clear_marks(False)).pack(side="left", expand=True, fill="x")
        ttk.Button(bf, text="Clear all", command=lambda: self.clear_marks(True)).pack(side="left", expand=True, fill="x")

        b = self.box(p, "8. XRD references (.cif / .xy)")
        self.combo(b, "Radiation", "rad", RADS, "CuKa")
        self.entry(b, "Custom λ (Å)", "lam", "1.5406")
        self.combo(b, "Show as", "refmode", REFMODES, "Sub-panels")
        self.entry(b, "Show ≥ (% int.)", "refmin", "1")
        self.check(b, "Label hkl on reflections", "hkl", True)
        self.entry(b, "Label ≥ (% int.)", "hklmin", "15")

        b = self.box(p, "9. Time (3rd axis)")
        self.check(b, "Use time as 3rd axis", "use_time", True)
        self.combo(b, "Time unit", "tunit", list(UNIT), "min")
        self.vars["tunit"].trace_add("write", lambda *_: self.on_select())
        self.entry(b, "Time label", "tlabel", "")
        self.entry(b, "Elevation", "elev", "25"); self.entry(b, "Azimuth", "azim", "-60")
        ttk.Button(b, text="Set times for all traces...", command=self.set_times).pack(fill="x", pady=2)

        b = self.box(p, "10. Display")
        self.combo(b, "Mode", "mode", ["Overlay", "Stacked", "3D", "Heatmap"], "Overlay")
        self.combo(b, "3D style", "style3d", list(STYLES_3D), "Waterfall (lines)")
        self.entry(b, "Stack offset", "offset", "1.2")
        self.combo(b, "Colours", "cmap", ["Custom colours", "viridis", "plasma", "coolwarm", "turbo", "Blues", "magma"], "Custom colours")
        self.entry(b, "Line width", "lw", "1.3")
        self.entry(b, "Font size", "font", "12")
        self.entry(b, "X label", "xlabel", ""); self.entry(b, "Y label", "ylabel", "")
        for n, t, d in (("reverse", "Reverse x-axis", False), ("legend", "Legend", True), ("inline", "Inline labels (stacked)", True),
                        ("box", "Full box frame", True), ("hide_y", "Hide y ticks", False),
                        ("markers", "Markers (XY / error bars)", False), ("show_blanks", "Show blanks", False)):
            self.check(b, t, n, d)

        b = self.box(p, "11. Export")
        self.entry(b, "DPI", "dpi", "600"); self.entry(b, "Width (in)", "fw", "7"); self.entry(b, "Height (in)", "fh", "5")
        self.check(b, "Transparent background", "transp")
        bf = ttk.Frame(b); bf.pack(fill="x")
        ttk.Button(bf, text="Save figure...", style="Accent.TButton", command=self.save).pack(side="left", expand=True, fill="x")
        ttk.Button(bf, text="Export processed CSV...", command=self.export_csv).pack(side="left", expand=True, fill="x")

        right = ttk.Frame(self.root); right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor="white")
        self.cv = FigureCanvasTkAgg(self.fig, master=right)
        self.toolbar = NavigationToolbar2Tk(self.cv, right)
        self.toolbar.pack(side="top", fill="x")
        self.cv.mpl_connect("button_press_event", self.on_click)
        self.cv.get_tk_widget().pack(fill="both", expand=True)
        self.lbl = tk.Label(right, text="Choose a technique and load data.", anchor="w", bg="white", padx=10, pady=5, font=(FONT, 10))
        self.lbl.pack(fill="x")

    # ---- technique presets
    def tech_changed(self):
        xl, yl, rev, norm, _ = TECH[self.vars["tech"].get()]
        self._loading = True
        for k, v in (("xlabel", xl), ("ylabel", yl), ("reverse", rev), ("norm", norm)):
            self.vars[k].set(v)
        self._loading = False
        self.sched()
        self.status(HINT.get(self.vars["tech"].get(), "Load data: column 1 = x, further columns = spectra (header = name)."))

    # ---- loading
    def load(self, ref=False):
        ft = [("Reference patterns", "*.cif *.xy *.txt *.csv *.dat"), ("All files", "*.*")] if ref else [("All files", "*.*")]
        paths = filedialog.askopenfilenames(title="Select reference files (.cif / .xy)" if ref else "Select data files", filetypes=ft)
        for path in paths:
            if ref and Path(path).suffix.lower() == ".cif":
                self._add_cif(path)
                continue
            try:
                x, Y, names = read_any(path)
            except Exception as e:
                messagebox.showerror("Load error", f"{Path(path).name}:\n{e}")
                continue
            stem, k = Path(path).stem, Y.shape[1]
            col = lambda: PALETTE[len([d for d in self.ds if d.kind == "data"]) % len(PALETTE)]
            if ref:
                self._add(x, Y[:, 0], stem, self.ref_color(), "ref")
            elif self.vars["tech"].get() == "XY" and self.vars["errcol"].get() and k >= 2:
                for j in range(0, k - 1, 2):
                    self._add(x, Y[:, j], names[j] or stem, col(), yerr=Y[:, j + 1])
            else:
                for j in range(k):
                    self._add(x, Y[:, j], (names[j] or stem) if k > 1 else stem, col())
        self.refresh()
        self.sched()

    def _add(self, x, y, label, color, kind="data", yerr=None):
        m = np.isfinite(y) & (np.isfinite(yerr) if yerr is not None else True)
        if m.sum() < 3:
            return None
        names, lab, n = {d.label for d in self.ds}, label, 2
        while lab in names:
            lab, n = f"{label} ({n})", n + 1
        d = Dataset(x[m], y[m], lab, color, kind, None if yerr is None else yerr[m])
        self.ds.append(d)
        return d

    def ref_color(self):
        return REFPAL[len([d for d in self.ds if d.kind == "ref"]) % len(REFPAL)]

    def wavelength(self, G):
        return (G["lam"] or 1.5406) if G["rad"].startswith("Custom") else G["rad"]

    def _add_cif(self, path):
        try:
            x, y, hkl, label = cif_pattern(path, self.wavelength(self.G()))
        except ImportError:
            return messagebox.showerror("pymatgen needed", "Reading .cif files needs pymatgen:\n\n    pip install pymatgen\n\n"
                                        "(.xy / .csv stick lists work without it.)")
        except Exception as e:
            return messagebox.showerror("CIF error", f"{Path(path).name}:\n{e}")
        d = self._add(x, y, label, self.ref_color(), "ref")
        if d:
            d.cif, d.hkl, d.ckey = path, hkl, self.wavelength(self.G())
            self.status(f"{d.label}: {len(x)} reflections simulated ({d.ckey}).")

    def ref_sticks(self, d, G):
        """-> (2theta, rel. intensity 0-1, hkl labels or None), filtered by the minimum-intensity setting."""
        if d.cif:
            lam = self.wavelength(G)
            if d.ckey != lam:
                try:
                    d.x, d.y, d.hkl, _ = cif_pattern(d.cif, lam)
                    d.ckey = lam
                except Exception as e:
                    self.status(f"{d.label}: could not recompute for this radiation ({e})", True)
        x, y = d.x + d.xshift, d.y / (d.y.max() or 1)
        m = y * 100 >= G["refmin"]
        return x[m], y[m], ([h for h, k in zip(d.hkl, m) if k] if d.hkl and len(d.hkl) == len(m) else None)

    def refs_on_main(self, ax, refs, G):
        """References drawn on the main axis: tick rows below the pattern, or dotted lines through it."""
        if G["refmode"].startswith("Tick"):
            y0, y1 = ax.get_ylim(); span = y1 - y0
            top0 = y0 - .02 * span
            for k, r in enumerate(refs):
                x, y, hk = self.ref_sticks(r, G)
                low = top0 - (k + 1) * .09 * span
                h = (.02 + .05 * y) * span
                ax.vlines(x, low, low + h, colors=r.color, lw=1.2)
                ax.text(.01, low + .045 * span, r.label, transform=ax.get_yaxis_transform(), color=r.color,
                        fontsize=max(G["font"] - 3, 7), fontweight="bold", va="center")
                if G["hkl"] and hk:
                    for xi, hi, hh, yi in zip(x, h, hk, y):
                        if yi * 100 >= G["hklmin"]: tag(ax, hh, (xi, low + hi), r.color, G)
            ax.set_ylim(top0 - len(refs) * .09 * span - .02 * span, y1)
            ax.set_yticks([t for t in ax.get_yticks() if y0 - 1e-9 <= t <= y1])
        else:
            for k, r in enumerate(refs):
                x, y, hk = self.ref_sticks(r, G)
                ax.vlines(x, 0, 1, transform=ax.get_xaxis_transform(), colors=r.color, linestyles=":", lw=.9, alpha=.8)
                ax.text(.99, .98 - k * .055, r.label, transform=ax.transAxes, color=r.color, ha="right", va="top",
                        fontsize=max(G["font"] - 3, 7), fontweight="bold")
                if G["hkl"] and hk:
                    for xi, hh, yi in zip(x, hk, y):
                        if yi * 100 >= G["hklmin"]:
                            tag(ax, hh, (xi, 1), r.color, G, below=True, xycoords=ax.get_xaxis_transform())

    def refresh(self, keep=None):
        self.tree.delete(*self.tree.get_children())
        for i, d in enumerate(self.ds):
            role = "ref" if d.kind == "ref" else ("blank" if d.is_blank else "sample")
            self.tree.insert("", "end", iid=str(i), values=("☑" if d.visible else "☐", d.label, role))
        self.blank_cb["values"] = ["(none)"] + [d.label for d in self.ds if d.kind == "data"]
        if keep is not None and keep < len(self.ds):
            self.tree.selection_set(str(keep))

    def cur(self):
        s = self.tree.selection()
        return (int(s[0]), self.ds[int(s[0])]) if s else (None, None)

    def on_select(self, _=None):
        i, d = self.cur()
        if d is None:
            return
        self._loading = True
        self.sel["label"].set(d.label); self.sel["xshift"].set(str(d.xshift)); self.sel["yscale"].set(str(d.yscale))
        self.sel["bscale"].set(str(d.bscale)); self.sel["blank"].set(d.blank.label if d.blank else "(none)")
        self.sel_isblank.set(d.is_blank)
        self.sel["time"].set("" if d.t is None else f"{d.t / UNIT[self.vars['tunit'].get()]:g}")
        self._loading = False

    def sel_changed(self, *_):
        if self._loading:
            return
        i, d = self.cur()
        if d is None:
            return
        f = lambda s, dflt: (float(s) if re.match(r"^\s*[-+]?\d*\.?\d+([eE][-+]?\d+)?\s*$", s) else dflt)
        d.label = self.sel["label"].get() or d.label
        d.xshift, d.yscale = f(self.sel["xshift"].get(), d.xshift), f(self.sel["yscale"].get(), d.yscale)
        d.bscale, d.is_blank = f(self.sel["bscale"].get(), d.bscale), self.sel_isblank.get()
        tt = self.sel["time"].get().strip()
        if tt == "":
            d.t = None
        elif re.match(r"^[-+]?\d*\.?\d+([eE][-+]?\d+)?$", tt):
            d.t = float(tt) * UNIT[self.vars["tunit"].get()]
        name = self.sel["blank"].get()
        d.blank = next((o for o in self.ds if o.label == name and o is not d and o.kind == "data"), None)
        self._loading = True
        self.refresh(i)
        self._loading = False
        self.sched()

    def blank_all(self):
        i, d = self.cur()
        if d is None or d.kind != "data":
            return self.status("Select a dataset to use as the blank.", True)
        d.is_blank = True
        for o in self.ds:
            if o.kind == "data" and not o.is_blank:
                o.blank = d
        self.refresh(i); self.on_select(); self.sched()

    def toggle_vis(self, e):
        iid = self.tree.identify_row(e.y)
        if iid:
            self.ds[int(iid)].visible ^= True
            self.refresh(int(iid)); self.sched()

    def move(self, dlt):
        i, d = self.cur()
        if d is None or not 0 <= i + dlt < len(self.ds):
            return
        self.ds[i], self.ds[i + dlt] = self.ds[i + dlt], self.ds[i]
        self.refresh(i + dlt); self.sched()

    def pick_color(self):
        i, d = self.cur()
        if d is None:
            return
        c = colorchooser.askcolor(color=d.color)[1]
        if c:
            d.color = c; self.vars["cmap"].set("Custom colours"); self.sched()

    def remove(self):
        i, d = self.cur()
        if d is None:
            return
        for o in self.ds:
            if o.blank is d: o.blank = None
        self.ds.pop(i); self.refresh(); self.sched()

    def clear(self):
        self.ds.clear(); self.refresh(); self.sched()

    # ---- settings
    def G(self):
        G = {k: v.get() for k, v in self.vars.items()}
        n = lambda s: float(s) if re.match(r"^\s*[-+]?\d*\.?\d+([eE][-+]?\d+)?\s*$", str(s)) else None
        for k in ("xmin", "xmax", "bgx0"):
            G[k] = n(G[k])
        for k, dflt in (("poly", 3), ("smooth", 0), ("offset", 1.2), ("lw", 1.3), ("font", 12), ("prom", 5), ("elev", 25), ("azim", -60), ("lam", 1.5406), ("refmin", 1), ("hklmin", 15),
                       ("als_lam", 6), ("als_p", 0.01), ("snip_it", 40), ("markrot", 90), ("mtol", 0.3)):
            G[k] = n(G[k]) if n(G[k]) is not None else dflt
        return G

    # ---- drawing
    def draw(self):
        self._job = None
        f, G = self.fig, self.G()
        f.clear()
        self.curves, self.main_ax = [], None
        matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                                    "font.size": G["font"], "xtick.direction": "in", "ytick.direction": "in"})
        data = [d for d in self.ds if d.kind == "data" and d.visible and (G["show_blanks"] or not d.is_blank)]
        refs = [d for d in self.ds if d.kind == "ref" and d.visible]
        if not data:
            f.text(.5, .5, "Load data to begin", ha="center", color="grey"); self.cv.draw_idle(); return
        try:
            cur = [(d,) + process(d, G) for d in data]
        except Exception as e:
            self.status(str(e), True); return
        n, mode = len(cur), G["mode"]
        cm = None if G["cmap"] == "Custom colours" else matplotlib.colormaps[G["cmap"]]
        Ut = UNIT[G["tunit"]]
        havet = bool(G["use_time"]) and all(c[0].t is not None for c in cur)
        tvals = [c[0].t / Ut for c in cur] if havet else list(range(n))
        t_lo, t_hi = min(tvals), max(tvals)
        colors = [d.color if cm is None else cm((tv - t_lo) / ((t_hi - t_lo) or 1)) for tv, (d, *_r) in zip(tvals, cur)]
        dec = TECH[G["tech"]][4]
        lw = G["lw"]

        if mode in TIME_MODES:
            return self.draw_time(f, G, cur, colors)

        subp = bool(refs) and G["refmode"] == "Sub-panels"
        if subp:
            gs = GridSpec(1 + len(refs), 1, height_ratios=[4] + [.8] * len(refs), hspace=.1, figure=f)
            ax = f.add_subplot(gs[0])
        else:
            ax = f.add_subplot(111)
        self.main_ax = ax
        stacked = mode == "Stacked"
        step = G["offset"] * max(np.ptp(y) for _, _, y, _ in cur) if stacked else 0
        for i, (d, x, y, e) in enumerate(cur):
            y0 = y
            if stacked:
                y = y - y.min() + i * step
            c = colors[i]
            if e is not None or G["markers"]:
                ax.errorbar(x, y, yerr=e, fmt="-" + MARKERS[i % len(MARKERS)], color=c, capsize=4, ms=6,
                            mfc="white", mew=1.3, lw=lw, label=d.label)
            else:
                ax.plot(x, y, color=c, lw=lw, label=d.label)
            self.curves.append((d, x, y0, y))
            for mx, txt in d.marks:
                if x.min() <= mx <= x.max():
                    ax.annotate(fmt_hkl(txt), (mx, float(np.interp(mx, x, y))), xytext=(0, 8), textcoords="offset points",
                                rotation=G["markrot"], ha="center", va="bottom", fontsize=max(G["font"] - 3, 7), color=c,
                                arrowprops=dict(arrowstyle="-", lw=.7, color=c, shrinkA=0, shrinkB=2) if G["leader"] else None)
            if stacked and G["inline"]:
                ax.text(0.02, y.max() + .02 * step, d.label, transform=ax.get_yaxis_transform(),
                        va="bottom", color=c, fontsize=G["font"] - 2)
            if G["peaks"]:
                s = y - (i * step if stacked else 0)
                for k in pick_peaks(s, G["prom"] / 100 * (np.ptp(s) or 1), G["minima"]):
                    ax.annotate(f"{x[k]:.{dec}f}", (x[k], y[k]), xytext=(0, 4 if not G["minima"] else -4),
                                textcoords="offset points", rotation=90, ha="center",
                                va="bottom" if not G["minima"] else "top", fontsize=max(G["font"] - 5, 6), color=c)
        xs = np.concatenate([c[1] for c in cur])
        lo, hi = (G["xmin"] if G["xmin"] is not None else xs.min()), (G["xmax"] if G["xmax"] is not None else xs.max())
        ax.set_xlim((hi, lo) if G["reverse"] else (lo, hi))
        ax.margins(y=.08)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_ylabel(G["ylabel"])
        if G["hide_y"] or stacked: ax.set_yticks([])
        if refs and not subp:
            self.refs_on_main(ax, refs, G)
        if G["legend"] and not (stacked and G["inline"]): ax.legend(frameon=False)
        if cm is not None and havet and n > 1:
            f.colorbar(ScalarMappable(Normalize(t_lo, t_hi), cm), ax=ax, pad=.02,
                       label=G["tlabel"] or f"Time ({G['tunit']})")
        axes = [ax]
        if subp:
            ax.tick_params(labelbottom=False)
            for k, r in enumerate(refs):
                ra = f.add_subplot(gs[k + 1], sharex=ax)
                rx, ry, hk = self.ref_sticks(r, G)
                ra.vlines(rx, 0, ry, colors=r.color, lw=1.5); ra.set_yticks([])
                ra.set_ylim(0, 1.6 if (G["hkl"] and hk) else 1.2)
                if G["hkl"] and hk:
                    for xi, yi, hh in zip(rx, ry, hk):
                        if yi * 100 >= G["hklmin"]: tag(ra, hh, (xi, yi), r.color, G)
                ra.text(.98, .9, r.label, transform=ra.transAxes, fontsize=max(G["font"] - 3, 7), fontweight="bold",
                        va="top", ha="right", bbox=dict(fc="white", alpha=.7, ec="none"))
                if k < len(refs) - 1: ra.tick_params(labelbottom=False)
                axes.append(ra)
        axes[-1].set_xlabel(G["xlabel"])
        if self._view:  # keep the user's zoom after adding/removing a label
            ax.set_xlim(*self._view[0]); ax.set_ylim(*self._view[1])
            self._view = None
        for a in axes:
            for nme, s in a.spines.items():
                s.set_visible(G["box"] or nme in ("left", "bottom"))
        f.tight_layout()
        self.cv.draw_idle()
        self.status(f"{n} trace(s) plotted | {G['tech']} | {mode}")

    def preview_bg(self):
        """Pop-up: selected sample's signal, fitted background and corrected result."""
        i, d = self.cur()
        if d is None or d.kind != "data":
            return self.status("Select a sample in the dataset list to preview its background.", True)
        win = tk.Toplevel(self.root)
        win.title(f"Background preview - {d.label}"); win.configure(bg=BG); win.geometry("760x600")
        fig = Figure(figsize=(7, 5), dpi=100, facecolor="white")
        cv = FigureCanvasTkAgg(fig, master=win)
        NavigationToolbar2Tk(cv, win).pack(side="top", fill="x")

        def render():
            G = self.G()
            x, y, _ = process(d, dict(G, bgmode="None", bgclip=False, smooth=0, deriv=False, norm="None"))
            fig.clear()
            try:
                base = baseline(x, y, G)
            except Exception as e:
                fig.text(.5, .5, str(e), ha="center", color="#b00020"); cv.draw_idle(); return
            a1 = fig.add_subplot(211); a2 = fig.add_subplot(212, sharex=a1)
            a1.plot(x, y, color=INK, lw=1.2, label="Signal"); a1.plot(x, base, "--", color="#d62728", lw=1.5, label="Background")
            a1.legend(frameon=False)
            a2.plot(x, y - base, color=ACCENT, lw=1.2, label="Corrected"); a2.axhline(0, color="grey", lw=.6)
            a2.legend(frameon=False); a2.set_xlabel(G["xlabel"])
            if G["reverse"]: a1.invert_xaxis()
            fig.tight_layout(); cv.draw_idle()

        ttk.Button(win, text="Refresh with current settings", command=render).pack(side="bottom", fill="x", padx=8, pady=6)
        cv.get_tk_widget().pack(fill="both", expand=True)
        render()

    # ---- manual peak labels
    def _keep(self):
        if self.main_ax is not None:
            self._view = (self.main_ax.get_xlim(), self.main_ax.get_ylim())
        self.sched()

    def suggest_hkl(self, px, G):
        """hkl text (readable form) of the nearest visible reference reflection within tolerance."""
        best = None
        for r in (d for d in self.ds if d.kind == "ref" and d.visible):
            x, y, hk = self.ref_sticks(r, G)
            if hk is None or len(x) == 0:
                continue
            k = int(np.argmin(np.abs(x - px)))
            if abs(x[k] - px) <= G["mtol"] and (best is None or abs(x[k] - px) < best[0]):
                best = (abs(x[k] - px), re.sub(r"\\bar\{(\d+)\}", r"-\1", hk[k]).strip("$"))
        return best[1] if best else ""

    def on_click(self, ev):
        ax = self.main_ax
        if (not self.vars["labelmode"].get() or self.toolbar.mode != "" or ax is None or ev.inaxes is not ax
                or ev.xdata is None or not self.curves):
            return
        G = self.G()
        xl, yl = ax.get_xlim(), ax.get_ylim()
        xr, yr = abs(xl[1] - xl[0]), abs(yl[1] - yl[0])
        hit = None
        for d, x, y0, yp in self.curves:  # clicked on an existing label?
            for m in d.marks:
                if abs(m[0] - ev.xdata) < .012 * xr and x.min() <= m[0] <= x.max():
                    if abs(float(np.interp(m[0], x, yp)) + .06 * yr - ev.ydata) < .12 * yr:
                        hit = (d, m)
        if hit:
            d, m = hit
            if ev.button in (2, 3):
                d.marks.remove(m); return self._keep()
            t = simpledialog.askstring("Edit label", f"Label for peak at {m[0]:.2f} (blank = remove):",
                                       initialvalue=m[1], parent=self.root)
            if t is None:
                return
            if t.strip(): m[1] = t.strip()
            else: d.marks.remove(m)
            return self._keep()
        if ev.button != 1:
            return
        best = None  # trace closest to the click, then snap to its local peak
        for d, x, y0, yp in self.curves:
            if x.min() <= ev.xdata <= x.max():
                dist = abs(float(np.interp(ev.xdata, x, yp)) - ev.ydata)
                if best is None or dist < best[0]:
                    best = (dist, d, x, y0)
        if best is None:
            return
        _, d, x, y0 = best
        msk = np.abs(x - ev.xdata) <= .015 * xr
        if not msk.any():
            return
        xs, ys = x[msk], y0[msk]
        px = float(xs[np.argmin(ys) if G["minima"] else np.argmax(ys)])
        dec = max(TECH[G["tech"]][4], 2)
        t = simpledialog.askstring("Label peak", f"Peak at {px:.{dec}f}\nReflection (e.g. 111 or -1 1 0):",
                                   initialvalue=self.suggest_hkl(px, G), parent=self.root)
        if t and t.strip():
            d.marks.append([px, t.strip()]); self._keep()

    def auto_assign(self):
        if not self.curves:
            return self.status("Plot a pattern first.", True)
        G, n = self.G(), 0
        for d, x, y0, _yp in self.curves:
            for k in pick_peaks(y0, G["prom"] / 100 * (np.ptp(y0) or 1), G["minima"]):
                h = self.suggest_hkl(x[k], G)
                if h and not any(abs(m[0] - x[k]) < G["mtol"] for m in d.marks):
                    d.marks.append([float(x[k]), h]); n += 1
        self._keep()
        self.status(f"Assigned {n} label(s) from the references. Click a label to edit, right-click to remove.")

    def clear_marks(self, everything):
        i, d = self.cur()
        for o in (self.ds if everything else ([d] if d else [])):
            o.marks.clear()
        self._keep()

    def set_times(self):
        """Dialog: type a time for every trace (list order), or auto-fill start + step."""
        data = [d for d in self.ds if d.kind == "data" and not d.is_blank]
        if not data:
            return self.status("Load data first.", True)
        u = self.vars["tunit"].get(); U = UNIT[u]
        win = tk.Toplevel(self.root)
        win.title(f"Set times ({u})"); win.configure(bg=BG); win.transient(self.root); win.geometry("400x520")
        top = ttk.Frame(win, padding=8); top.pack(fill="x")
        st, sp = tk.StringVar(value="0"), tk.StringVar(value="10")
        for t, v in (("Start", st), ("Step", sp)):
            ttk.Label(top, text=t).pack(side="left"); ttk.Entry(top, textvariable=v, width=7).pack(side="left", padx=(3, 8))
        wrap = ttk.Frame(win); wrap.pack(fill="both", expand=True, padx=8)
        cv = tk.Canvas(wrap, bg=BG, highlightthickness=0); sb = ttk.Scrollbar(wrap, command=cv.yview)
        cv.configure(yscrollcommand=sb.set); sb.pack(side="right", fill="y"); cv.pack(side="left", fill="both", expand=True)
        body = ttk.Frame(cv); cv.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        rows = []
        for d in data:
            v = tk.StringVar(value="" if d.t is None else f"{d.t / U:g}")
            r = ttk.Frame(body); r.pack(fill="x", pady=1)
            ttk.Label(r, text=d.label[:30], width=30).pack(side="left")
            ttk.Entry(r, textvariable=v, width=9).pack(side="left")
            ttk.Label(r, text=u).pack(side="left", padx=4)
            rows.append((d, v))

        def fill():
            try:
                a, b = float(st.get()), float(sp.get())
            except ValueError:
                return
            for k, (_d, v) in enumerate(rows):
                v.set(f"{a + k * b:g}")

        def apply():
            for d, v in rows:
                t = v.get().strip()
                if t == "":
                    d.t = None
                else:
                    try: d.t = float(t) * U
                    except ValueError: pass
            win.destroy(); self.on_select(); self.sched()

        ttk.Button(top, text="Auto-fill", command=fill).pack(side="left")
        ttk.Button(win, text="Apply", style="Accent.TButton", command=apply).pack(fill="x", padx=8, pady=8)

    def draw_time(self, f, G, cur, colors):
        """3D (several styles) or heatmap with time (or trace index) as the 3rd axis."""
        mode, style, lw, U = G["mode"], G["style3d"], G["lw"], UNIT[G["tunit"]]
        have = bool(G["use_time"]) and all(c[0].t is not None for c in cur)
        order = sorted(range(len(cur)), key=lambda i: cur[i][0].t) if have else list(range(len(cur)))
        cur, colors = [cur[i] for i in order], [colors[i] for i in order]
        tv = np.array([c[0].t / U for c in cur]) if have else np.arange(len(cur), dtype=float)
        tlab = G["tlabel"] or (f"Time ({G['tunit']})" if have else "Trace #")
        grid = mode == "Heatmap" or style in GRID_STYLES
        if grid and len(cur) < 2:
            return self.status(f"{'Heatmap' if mode == 'Heatmap' else style} needs at least 2 visible traces.", True)
        cm = "viridis" if G["cmap"] == "Custom colours" else G["cmap"]
        if grid:
            xg = np.linspace(max(c[1].min() for c in cur), min(c[1].max() for c in cur), max(len(c[1]) for c in cur))
            Z = np.array([np.interp(xg, c[1], c[2]) for c in cur])
        if mode == "Heatmap":
            ax = f.add_subplot(111)
            f.colorbar(ax.pcolormesh(xg, tv, Z, shading="auto", cmap=cm), ax=ax, label=G["ylabel"])
        else:
            ax, m = f.add_subplot(111, projection="3d"), None
            if style == "Waterfall (lines)":
                for i, (d, x, y, _e) in enumerate(cur):
                    ax.plot(x, np.full_like(x, tv[i]), y, color=colors[i], lw=lw)
            elif style == "Filled waterfall":
                zmin = min(c[2].min() for c in cur)
                for i, (d, x, y, _e) in enumerate(cur):
                    verts = [[(x[0], zmin), *zip(x, y), (x[-1], zmin)]]
                    ax.add_collection3d(PolyCollection(verts, facecolors=[to_rgba(colors[i], .55)],
                                                       edgecolors=[colors[i]], linewidths=lw), zs=tv[i], zdir="y")
                pad = (np.ptp(tv) or 1) * .05
                ax.set_xlim(min(c[1].min() for c in cur), max(c[1].max() for c in cur))
                ax.set_ylim(tv.min() - pad, tv.max() + pad); ax.set_zlim(zmin, max(c[2].max() for c in cur))
            else:
                X, T = np.meshgrid(xg, tv)
                if style == "Surface":
                    m = ax.plot_surface(X, T, Z, cmap=cm, linewidth=0, antialiased=True)
                elif style == "Wireframe":
                    ax.plot_wireframe(X, T, Z, color="#1f3a63", linewidth=.6, rstride=1, cstride=max(1, len(xg) // 80))
                elif style == "Surface + contour":
                    m = ax.plot_surface(X, T, Z, cmap=cm, alpha=.85, linewidth=0)
                    ax.contourf(X, T, Z, zdir="z", offset=Z.min(), cmap=cm, alpha=.8)
                else:
                    k = max(1, Z.size // 6000)
                    m = ax.scatter(X.ravel()[::k], T.ravel()[::k], Z.ravel()[::k], c=Z.ravel()[::k], cmap=cm, s=4)
            if m is not None:
                f.colorbar(m, ax=ax, shrink=.6, pad=.1, label=G["ylabel"])
            ax.set_zlabel(G["ylabel"])
            if not have:
                ax.set_yticks(tv); ax.set_yticklabels([c[0].label for c in cur], fontsize=G["font"] - 4)
            ax.view_init(elev=G["elev"], azim=G["azim"])
        ax.set_xlabel(G["xlabel"]); ax.set_ylabel(tlab)
        if G["reverse"]: ax.invert_xaxis()
        try: f.tight_layout()
        except Exception: pass
        self.cv.draw_idle()
        self.status(f"{len(cur)} traces | {'Heatmap' if mode == 'Heatmap' else '3D: ' + style} | "
                    f"3rd axis: {tlab if have else 'trace order (set times in the Time section)'}")

    # ---- export
    def save(self):
        if not self.ds:
            return self.status("Nothing to save.", True)
        path = filedialog.asksaveasfilename(defaultextension=".pdf", initialfile="nova_figure",
                                            filetypes=[("PDF", "*.pdf"), ("PNG", "*.png"), ("SVG", "*.svg"),
                                                       ("TIFF", "*.tiff"), ("JPEG", "*.jpg")])
        if not path:
            return
        G, old = self.G(), self.fig.get_size_inches().copy()
        try:
            self.fig.set_size_inches(float(G["fw"]), float(G["fh"]))
            self.fig.savefig(path, dpi=float(G["dpi"]), bbox_inches="tight", transparent=bool(G["transp"]))
            self.status(f"Saved {path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
        finally:
            self.fig.set_size_inches(old); self.cv.draw_idle()

    def export_csv(self):
        G = self.G()
        data = [d for d in self.ds if d.kind == "data" and d.visible and (G["show_blanks"] or not d.is_blank)]
        if not data:
            return self.status("No visible data to export.", True)
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="nova_processed")
        if not path:
            return
        cols = {}
        for d in data:
            x, y, _ = process(d, G)
            cols[f"{d.label} x"], cols[f"{d.label} y"] = pd.Series(x), pd.Series(y)
        pd.DataFrame(cols).to_csv(path, index=False)
        self.status(f"Exported {path}")


if __name__ == "__main__":
    root = tk.Tk()
    Nova(root)
    root.lift()
    root.mainloop()
