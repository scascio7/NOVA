"""NMR: ppm axis (reversed), solvent / reference-peak calibration."""
from nova.common import *
from nova.registry import Technique, register
from nova.processing import pick_peaks

NMR_SOLVENTS = {
    "CDCl3": 7.26,
    "DMSO-d6": 2.50,
    "CD3OD": 3.31,
    "Acetone-d6": 2.05,
    "D2O": 4.79,
    "THF-d8": 1.72,
    "C6D6": 7.16,
    "DMF-d7": 2.75,
}


class NMRMixin:
    nmr_reference_mode = False
    SECTIONS = {"nmr": (70, "NMR Options", "build_nmr")}

    def build_nmr(self, b):
        self.check(
            b,
            "Deuterated solvent used",
            "nmr_deut",
            True
        )

        self.combo(
            b,
            "Solvent",
            "nmr_solvent",
            list(NMR_SOLVENTS),
            "CDCl3"
        )

        ttk.Button(
            b,
            text="Auto Reference to Solvent",
            command=self.auto_reference_solvent
        ).pack(fill="x", pady=2)

        ttk.Button(
            b,
            text="Select Reference Peak",
            command=self.start_nmr_reference
        ).pack(fill="x", pady=2)

    def on_click(self, ev):
        if self.nmr_reference_mode:
            return self.reference_click(ev)
        return super().on_click(ev)

    def start_nmr_reference(self):
        if self.vars["tech"].get() != "NMR":
            return self.status(
                "Switch to NMR mode first.",
                True
            )
        self.nmr_reference_mode = True
        self.status(
            "Click a peak to assign a chemical shift."
        )

    def auto_reference_solvent(self):
        i, d = self.cur()
        if d is None:
            return
        solvent = self.vars["nmr_solvent"].get()
        target_ppm = NMR_SOLVENTS[solvent]
        idx = np.argmax(d.y)
        observed_ppm = d.x[idx]
        d.ppm_offset += target_ppm - observed_ppm
        self.sched()
        self.status(
            f"Referenced to {solvent} ({target_ppm:.2f} ppm)"
        )

    def reference_click(self, ev):
        if ev.xdata is None:
            return
        if not self.curves:
            return
        closest = None
        for d, x, y0, yp in self.curves:
            if x.min() <= ev.xdata <= x.max():
                dist = abs(
                    float(np.interp(ev.xdata, x, yp))
                    - ev.ydata
                )
                if closest is None or dist < closest[0]:
                    closest = (dist, d, x, y0)
        if closest is None:
            return
        _, d, x, y0 = closest
        peaks = pick_peaks(
            y0,
            0.01 * np.ptp(y0)
        )
        if len(peaks) == 0:
            return
        nearest = peaks[
            np.argmin(
                np.abs(x[peaks] - ev.xdata)
            )
        ]
        observed_ppm = x[nearest]
        assigned_ppm = simpledialog.askfloat(
            "Reference Peak",
            f"Observed peak = {observed_ppm:.4f} ppm\n\nAssign to:"
        )
        if assigned_ppm is None:
            return
        shift = assigned_ppm - observed_ppm
        d.ppm_offset += shift
        d.nmr_reference_peak = observed_ppm
        d.nmr_reference_ppm = assigned_ppm
        self.nmr_reference_mode = False
        self.status(
            f"Shifted spectrum by {shift:.4f} ppm"
        )
        self.sched()



@register
class NMR(Technique):
    name, order = "NMR", 40
    xlabel = "Chemical shift $\\delta$ (ppm)"
    ylabel = "Intensity (a.u.)"
    reverse = True
    norm = "None"
    decimals = 2
    sections = frozenset({"baseline", "peaks", "nmr", "labels", "time"})
    mixin = NMRMixin
    hint = "NMR: load the spectrum, then use NMR Options to reference it to the solvent peak or a chosen peak."
