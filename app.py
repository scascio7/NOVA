"""NOVA core window: dataset manager, preprocessing, visualisation, export. Techniques and features plug in around it."""
from nova.common import *
from nova.dataset import Dataset
from nova.fileio import read_any
from nova.processing import process, pick_peaks
from nova.registry import REGISTRY, discover
from nova.features.background import BackgroundMixin
from nova.features.peaks import PeakMixin, fmt_hkl
from nova.features.timeaxis import TimeMixin, TIME_MODES, STYLES_3D

ALWAYS = {"data", "datasets", "props", "preproc", "viz", "export"}


class Core:
    SECTIONS = {
        "data": (10, "Data", "build_data"),
        "datasets": (20, "Dataset Manager (double-click = show/hide)", "build_datasets"),
        "props": (30, "Dataset Properties", "build_props"),
        "preproc": (40, "Preprocessing", "build_preproc"),
        "viz": (110, "Visualization", "build_viz"),
        "export": (120, "Export", "build_export"),
    }

    def __init__(self, root):
        self.root, self.ds, self._job, self._loading = root, [], None, False
        self.curves, self.main_ax, self._view = [], None, None
        root.title("NOVA - Numerical Omnitechnique Visualization and Analysis")
        root.geometry("1380x860")
        self.vars = {}
        self.active, self.sections = set(), {}
        self._style()
        self._build()
        self.tech_changed()

    def _style(self):
        s = ttk.Style(); s.theme_use("clam")
        s.configure(".", background=BG, foreground=INK, font=(FONT, 11))
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=MUTED)
        s.configure("TLabelframe", background=BG, bordercolor=LINE, relief="solid", borderwidth=1)
        s.configure("TLabelframe.Label", background=BG, foreground=NAVY, font=(FONT, 11, "bold"))
        s.configure("TButton", background="#e6eaf2", foreground=INK, borderwidth=0, padding=(10, 6), focuscolor="#e6eaf2")
        s.map("TButton", background=[("active", "#d5dcea"), ("pressed", "#c6cfe2")])
        s.configure("Accent.TButton", background=ACCENT, foreground="white", font=(FONT, 11, "bold"), focuscolor=ACCENT)
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
        f = ttk.Frame(parent); f.pack(fill="x", pady=2)
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

    def load(self, ref=False):
        ft = [("Supported data", "*.csv *.txt *.dat *.xy *.tsv *.jdx *.dx *.jdf *.xlsx *.xls"), ("All files", "*.*")
        ]
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
        for klass in type(self).__mro__:  # sections come from the core and every mixed-in feature / technique
            for sid, (order, title, fn) in klass.__dict__.get("SECTIONS", {}).items():
                fr = ttk.LabelFrame(p, text=title, padding=5)
                self.sections[sid] = (order, title, fr)
                getattr(self, fn)(fr)
        right = ttk.Frame(self.root); right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor="white")
        self.cv = FigureCanvasTkAgg(self.fig, master=right)
        self.toolbar = NavigationToolbar2Tk(self.cv, right)
        self.toolbar.pack(side="top", fill="x")
        self.cv.mpl_connect("button_press_event", self.on_click)
        self.cv.get_tk_widget().pack(fill="both", expand=True)
        self.lbl = tk.Label(right, text="Choose a technique and load data.", anchor="w", bg="white", padx=10, pady=5, font=(FONT, 11))
        self.lbl.pack(fill="x")

    def tech_changed(self):
        t = REGISTRY[self.vars["tech"].get()]
        self._loading = True
        preset = {"xlabel": t.xlabel, "ylabel": t.ylabel, "reverse": t.reverse, "norm": t.norm,
                  "bgdown": False, "minima": False, **t.defaults}
        for k, v in preset.items():
            if k in self.vars:
                self.vars[k].set(v)
        self.mode_cb["values"] = list(t.modes)
        if self.vars["mode"].get() not in t.modes:
            self.vars["mode"].set("Overlay")
        self._loading = False
        self.apply_sections(ALWAYS | set(t.sections))
        self.sched()
        self.status(t.hint)

    def on(self, sid):
        return sid in self.active

    def apply_sections(self, active):
        """Show only the side-panel sections that belong to the selected technique (numbered in order)."""
        self.active = set(active)
        for _o, _t, fr in self.sections.values():
            fr.pack_forget()
        n = 0
        for sid, (_o, title, fr) in sorted(self.sections.items(), key=lambda kv: kv[1][0]):
            if sid in self.active:
                n += 1
                fr.configure(text=f"{n}. {title}")
                fr.pack(fill="x", padx=4, pady=2)

    def make_axes(self, f, G):
        return f.add_subplot(111), None

    def decorate_axes(self, f, ax, ctx, G):
        return []

    def build_data(self, b):
        v = self.V("tech", "s", next(iter(REGISTRY)))
        v.trace_add("write", lambda *_: self.tech_changed())
        self.row(b, "Technique", lambda f: ttk.Combobox(f, textvariable=v, values=list(REGISTRY), state="readonly", width=18))
        ttk.Button(b, text="Load data...", style="Accent.TButton", command=self.load).pack(fill="x", pady=2)

    def build_datasets(self, b):
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

    def build_props(self, b):
        self.sel = {k: tk.StringVar() for k in ("label", "xshift", "yscale", "bscale", "blank", "time")}
        self.sel_isblank = tk.BooleanVar()
        for k, t in (("label", "Label"), ("xshift", "X shift"), ("yscale", "Y scale"), ("bscale", "Background scale"), ("time", "Time (3rd axis)")):
            self.row(b, t, lambda f, k=k: ttk.Entry(f, textvariable=self.sel[k], width=10))
        self.blank_cb = self.row(b, "Background spectrum", lambda f: ttk.Combobox(f, textvariable=self.sel["blank"], state="readonly", width=18))
        ttk.Checkbutton(b, text="Is background (hidden unless 'show blanks')", variable=self.sel_isblank).pack(anchor="w")
        for v in list(self.sel.values()) + [self.sel_isblank]:
            v.trace_add("write", self.sel_changed)
        ttk.Button(b, text="Use selected as blank for ALL samples", command=self.blank_all).pack(fill="x")

    def build_preproc(self, b):
        self.combo(b, "Convert", "convert", ["None", "%T → Absorbance", "Absorbance → %T"], "None")
        self.entry(b, "Smooth window", "smooth", "0")
        self.combo(b, "Normalise", "norm", ["None", "Max", "Min-max", "Area"], "None")
        self.check(b, "1st derivative", "deriv")
        self.entry(b, "X min (crop)", "xmin", "")
        self.entry(b, "X max (crop)", "xmax", "")

    def build_viz(self, b):
        self.mode_cb = self.combo(b, "Mode", "mode", ["Overlay", "Stacked", "3D", "Heatmap"], "Overlay")
        self.combo(b, "3D style", "style3d", list(STYLES_3D), "Waterfall (lines)")
        self.entry(b, "Stack offset", "offset", "1.2")
        self.combo(b, "Colours", "cmap", ["Custom colours", "viridis", "plasma", "coolwarm", "turbo", "Blues", "magma"], "Custom colours")
        self.entry(b, "Line width", "lw", "1.3")
        self.entry(b, "Font size", "font", "12")
        self.entry(b, "X label", "xlabel", ""); self.entry(b, "Y label", "ylabel", "")
        for n, t, d in (("reverse", "Reverse x-axis", False), ("legend", "Legend", True), ("inline", "Inline labels (stacked)", True),
                        ("box", "Full box frame", True), ("hide_y", "Hide y ticks", False),
                        ("show_blanks", "Show blanks", False)):
            self.check(b, t, n, d)

    def build_export(self, b):
        self.entry(b, "DPI", "dpi", "600"); self.entry(b, "Width (in)", "fw", "7"); self.entry(b, "Height (in)", "fh", "5")
        self.check(b, "Transparent background", "transp")
        bf = ttk.Frame(b); bf.pack(fill="x")
        ttk.Button(bf, text="Save figure...", style="Accent.TButton", command=self.save).pack(side="left", expand=True, fill="x")
        ttk.Button(bf, text="Export processed CSV...", command=self.export_csv).pack(side="left", expand=True, fill="x")

    def G(self):
        G = {k: v.get() for k, v in self.vars.items()}
        n = lambda s: float(s) if re.match(r"^\s*[-+]?\d*\.?\d+([eE][-+]?\d+)?\s*$", str(s)) else None
        for k in ("xmin", "xmax", "bgx0"):
            G[k] = n(G[k])
        for k, dflt in (("poly", 3), ("smooth", 0), ("offset", 1.2), ("lw", 1.3), ("font", 12), ("prom", 5), ("elev", 25), ("azim", -60), ("lam", 1.5406), ("refmin", 1), ("hklmin", 15),
                       ("als_lam", 6), ("als_p", 0.01), ("snip_it", 40), ("markrot", 90), ("mtol", 0.3)):
            G[k] = n(G[k]) if n(G[k]) is not None else dflt
        t = REGISTRY[G["tech"]]  # features of hidden sections are switched off
        if not self.on("baseline"): G["bgmode"] = "None"
        if not self.on("peaks"): G["peaks"] = False
        if not self.on("time"): G["use_time"] = False
        if not self.on("xyopts"): G["markers"] = False
        if G["mode"] not in t.modes: G["mode"] = "Overlay"
        return G

    def draw(self):
        self._job = None
        f, G = self.fig, self.G()
        f.clear()
        self.curves, self.main_ax = [], None
        matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                                    "font.size": G["font"], "xtick.direction": "in", "ytick.direction": "in"})
        data = [d for d in self.ds if d.kind == "data" and d.visible and (G["show_blanks"] or not d.is_blank)]
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
        dec = REGISTRY[G["tech"]].decimals
        lw = G["lw"]

        if mode in TIME_MODES:
            return self.draw_time(f, G, cur, colors)

        ax, ctx = self.make_axes(f, G)
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
            for mx, txt in (d.marks if self.on("labels") else []):
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
        extra = self.decorate_axes(f, ax, ctx, G)
        if G["legend"] and not (stacked and G["inline"]): ax.legend(frameon=False)
        if cm is not None and havet and n > 1:
            f.colorbar(ScalarMappable(Normalize(t_lo, t_hi), cm), ax=ax, pad=.02,
                       label=G["tlabel"] or f"Time ({G['tunit']})")
        axes = [ax] + extra
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


def create_app_class():
    """Discover technique plugins and compose the GUI class from the core, shared features and technique mixins."""
    discover()
    mixins = [t.mixin for t in REGISTRY.values() if t.mixin]
    return type("Nova", tuple(mixins) + (BackgroundMixin, PeakMixin, TimeMixin, Core), {})
