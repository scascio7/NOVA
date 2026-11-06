"""Technique repository.

Every file in nova/techniques/ defines one Technique subclass decorated with @register.  The GUI start-up code
(nova.app.create_app_class) imports them all, lists them in the technique menu, shows only the side-panel
sections each technique asks for, and mixes in any technique-specific code (`mixin`).
"""
import importlib
import pkgutil

REGISTRY = {}


class Technique:
    name = ""                    # menu name
    order = 99                   # position in the menu
    xlabel, ylabel = "X", "Y"
    reverse = False              # reverse x-axis by default
    norm = "None"                # default normalisation
    decimals = 2                 # decimals for auto peak labels
    hint = "Load data: column 1 = x, further columns = spectra (header = name)."
    # optional sections: baseline, peaks, labels, time, nmr, refs, xyopts  (core sections are always shown)
    sections = frozenset()
    modes = ("Overlay", "Stacked", "3D", "Heatmap")
    defaults = {}                # extra GUI variable presets applied when selected, e.g. {"bgdown": True}
    mixin = None                 # class with technique-specific methods / UI sections (optional)


def register(cls):
    REGISTRY[cls.name] = cls()
    return cls


def discover():
    from nova import techniques
    for m in pkgutil.iter_modules(techniques.__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"nova.techniques.{m.name}")
    items = sorted(REGISTRY.items(), key=lambda kv: kv[1].order)
    REGISTRY.clear(); REGISTRY.update(items)
    return REGISTRY
