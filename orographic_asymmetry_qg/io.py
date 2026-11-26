"""Output paths, caches, CSV and NetCDF files, figures and reports."""

import csv
import datetime
import os

import numpy as np
from scipy.io import netcdf_file

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT = os.path.join(ROOT, "outputs")
DIRS = {k: os.path.join(OUT, k) for k in
        ("figures", "animations", "data", "netcdf", "reports", "cache")}

AUTHORS = "Faiz R. Fajary, Sandy H. S. Herho"


def path(kind, name):
    """Absolute path of an output file, creating its folder."""
    d = DIRS[kind]
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def save_cache(name, **arrays):
    """Store arrays in ``outputs/cache/<name>.npz``."""
    np.savez_compressed(path("cache", name + ".npz"), **arrays)


def load_cache(name):
    """Load ``outputs/cache/<name>.npz`` as a dict of arrays."""
    fn = path("cache", name + ".npz")
    if not os.path.exists(fn):
        raise FileNotFoundError(fn + " is missing; run the compute scripts "
                                "first (scripts/run_all.py)")
    with np.load(fn, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def has_cache(name):
    """True if the cache file exists."""
    return os.path.exists(path("cache", name + ".npz"))


def write_csv(name, header, rows):
    """Write a CSV table to ``outputs/data``."""
    with open(path("data", name + ".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([("%.6g" % v) if isinstance(v, (float, np.floating))
                        else v for v in r])


def write_netcdf(name, dims, variables, title, attrs=None):
    """Write a self-describing NetCDF-3 file to ``outputs/netcdf``.

    Parameters
    ----------
    name : str
        File stem.
    dims : dict
        Dimension name to coordinate ``(values, units, long_name)``, or
        to an integer size for a dimension without a coordinate.
    variables : dict
        Variable name to ``(dims, values, units, long_name)``.
    title : str
        Global ``title`` attribute.
    attrs : dict, optional
        Further global attributes (scalars or strings).
    """
    fn = path("netcdf", name + ".nc")
    with netcdf_file(fn, "w", version=2) as f:
        f.title = title
        f.Conventions = "CF-1.8"
        f.authors = AUTHORS
        f.institution = ("Atmospheric Science Research Group and Applied "
                         "Geology Research Group, Bandung Institute of "
                         "Technology")
        f.source = ("orographic_asymmetry_qg (idealized model, no "
                    "observations)")
        f.history = "created %s" % datetime.date.today().isoformat()
        for k, v in (attrs or {}).items():
            setattr(f, k, v)
        for dname, spec in dims.items():
            if isinstance(spec, (int, np.integer)):
                f.createDimension(dname, int(spec))
                continue
            vals, units, lname = spec
            vals = np.asarray(vals, dtype="f8")
            f.createDimension(dname, vals.size)
            v = f.createVariable(dname, "f8", (dname,))
            v[:] = vals
            v.units, v.long_name = units, lname
        for vname, (vd, vals, units, lname) in variables.items():
            v = f.createVariable(vname, "f8", tuple(vd))
            v[:] = np.asarray(vals, dtype="f8")
            v.units, v.long_name = units, lname
    return fn


def save_figure(fig, name):
    """Save a figure as PDF and as a 600 dpi PNG."""
    fig.savefig(path("figures", name + ".pdf"))
    fig.savefig(path("figures", name + ".png"), dpi=600)


def write_report(name, lines):
    """Write a plain-text report to ``outputs/reports``."""
    with open(path("reports", name + ".txt"), "w") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")
