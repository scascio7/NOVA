"""PXRD: reference phases (.cif via pymatgen, or .xy sticks), hkl labelling, radiation selection."""
from nova.common import *
from nova.registry import Technique, register
from nova.processing import pick_peaks
from nova.dataset import Dataset

RADS = ["CuKa", "CoKa", "MoKa", "FeKa", "CrKa", "AgKa", "Custom (set λ)"]


REFPAL = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#17becf"]


REFMODES = ["Sub-panels", "Tick marks under pattern", "Lines through pattern"]


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


def tag(ax, text, xy, color, G, below=False, xycoords="data"):
    ax.annotate(text, xy, xycoords=xycoords, xytext=(0, -2 if below else 2), textcoords="offset points",
                rotation=90, ha="center", va="top" if below else "bottom", fontsize=max(G["font"] - 5, 6), color=color)


class PXRDMixin:
    SECTIONS = {"refs": (90, "Add Reference Phases (.cif / .xy)", "build_refs")}

    def build_refs(self, b):
        ttk.Button(b, text="Add reference phase (.cif / .xy)...", style="Accent.TButton",
                   command=lambda: self.load(ref=True)).pack(fill="x", pady=2)
        ttk.Button(b, text="Auto-assign hkl to detected peaks", command=self.auto_assign).pack(fill="x", pady=2)
        self.combo(b, "Radiation", "rad", RADS, "CuKa")
        self.entry(b, "Custom λ (Å)", "lam", "1.5406")
        self.combo(b, "Show as", "refmode", REFMODES, "Sub-panels")
        self.entry(b, "Show ≥ (% int.)", "refmin", "1")
        self.check(b, "Label hkl on reflections", "hkl", True)
        self.entry(b, "Label ≥ (% int.)", "hklmin", "15")

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

    def visible_refs(self, G):
        return [d for d in self.ds if d.kind == "ref" and d.visible] if self.on("refs") else []

    def make_axes(self, f, G):
        refs = self.visible_refs(G)
        if refs and G["refmode"] == "Sub-panels":
            gs = GridSpec(1 + len(refs), 1, height_ratios=[4] + [.8] * len(refs), hspace=.1, figure=f)
            return f.add_subplot(gs[0]), gs
        return super().make_axes(f, G)

    def decorate_axes(self, f, ax, gs, G):
        refs = self.visible_refs(G)
        if not refs:
            return super().decorate_axes(f, ax, gs, G)
        if G["refmode"] != "Sub-panels":
            self.refs_on_main(ax, refs, G)
            return []
        axes = []
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
        return axes


@register
class PXRD(Technique):
    name, order = "PXRD", 50
    xlabel = "2$\\theta$ (degrees)"
    ylabel = "Intensity (a.u.)"
    reverse = False
    norm = "Max"
    decimals = 1
    sections = frozenset({"baseline", "peaks", "labels", "refs", "time"})
    mixin = PXRDMixin
    hint = "PXRD: load the pattern, then 'Add reference phase' to import .cif (needs pymatgen) or .xy stick lists."
