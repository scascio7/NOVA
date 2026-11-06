"""Raman: Raman shift (cm^-1), intensity."""
from nova.registry import Technique, register


@register
class Raman(Technique):
    name, order = "Raman", 20
    xlabel = "Raman shift (cm$^{-1}$)"
    ylabel = "Intensity (a.u.)"
    reverse = False
    norm = "None"
    decimals = 0
    sections = frozenset({"baseline", "peaks", "labels", "time"})
