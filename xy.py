"""Generic XY plots (e.g. dissolution profiles with error bars and markers)."""
from nova.common import *
from nova.registry import Technique, register


class XYMixin:
    SECTIONS = {"xyopts": (95, "XY Options", "build_xy")}

    def build_xy(self, b):
        self.check(b, "Columns are (y, error) pairs", "errcol")
        self.check(b, "Markers", "markers", True)


@register
class XY(Technique):
    name, order = "XY", 60
    xlabel = "Time (min)"
    ylabel = "Cumulative Dissolution (%)"
    reverse = False
    norm = "None"
    decimals = 2
    sections = frozenset({"xyopts"})
    modes = ("Overlay",)
    mixin = XYMixin
    hint = "XY: column 1 = x; tick (y, error) pairs for error bars. Plain overlay plot."
