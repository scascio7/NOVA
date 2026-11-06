"""FTIR: wavenumber axis reversed, %T aware (peaks point down)."""
from nova.registry import Technique, register


@register
class FTIR(Technique):
    name, order = "FTIR", 10
    xlabel = "Wavenumber (cm$^{-1}$)"
    ylabel = "Transmittance (%)"
    reverse = True
    norm = "None"
    decimals = 0
    sections = frozenset({"baseline", "peaks", "labels", "time"})
    defaults = {"bgdown": True, "minima": True}
    hint = ("FTIR: column 1 = wavenumber. %T dips are treated as peaks by default; "
            "use Preprocessing > Convert for absorbance.")
