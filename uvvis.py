"""UV-Vis: one file, wavelength in column 1 then one absorbance column per sample."""
from nova.registry import Technique, register


@register
class UVVis(Technique):
    name, order = "UV-Vis", 30
    xlabel = "Wavelength (nm)"
    ylabel = "Absorbance"
    reverse = False
    norm = "None"
    decimals = 0
    sections = frozenset({"baseline", "peaks", "labels", "time"})
    hint = ("UV-Vis: load ONE file - wavelength in column 1, then one absorbance column per sample "
            "(header = sample name). Set times manually in the Time Series section.")
