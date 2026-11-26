"""Rendering helpers for the animations.

Every field, parcel and number drawn comes from the model: the
streamfunction and potential vorticity, the orography, the parcels
(advected by the model velocity) and the instruments. Maps carry
distance axes in km and color bars in physical units; the mountain is
drawn as shaded relief with a hypsometric tint, labeled height contours
and a summit marker, so its position and size can be read directly.
The dark background, the relief shading, the glow of the parcel trails
and the color maps are presentation only.
"""

import os
import subprocess

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LightSource, LinearSegmentedColormap, to_rgba

BG = "#060b14"
PANEL = "#0b1422"
TEXT = "#e9eef5"
DIM = "#8a9bb3"
ACCENT = "#ffb547"
WEST = "#58a6ff"
EAST = "#ff7b72"

# Diverging map for streamfunction and PV anomalies (cyclonic blue,
# anticyclonic amber), designed for a dark background.
PV = LinearSegmentedColormap.from_list(
    "pv_dark", ["#1b6ef3", "#1a3f7a", "#101a2b", "#7a4a12", "#f3a21b"])
TRAIL = LinearSegmentedColormap.from_list(
    "trail", ["#2b5cff", "#38d0ff", "#e8fbff"])
TERRAIN = LinearSegmentedColormap.from_list(
    "terrain_dark", ["#0b1422", "#2b3a4f", "#7d8a9c", "#e7ecf2"])


# Hypsometric tint for the relief (lowland green to snow), as on
# physical maps.
HYPSO = LinearSegmentedColormap.from_list(
    "hypso", ["#3f6b3a", "#7d8f4e", "#b8a46a", "#9c7a54", "#8a7a72",
              "#e9e6e1"])


def relief(h, dx, dy, h_min=60.0, azdeg=315, altdeg=40, exag=40.0):
    """RGBA image of the mountain as tinted, shaded relief.

    Parameters
    ----------
    h : ndarray (ny, nx)
        Height in metres.
    dx, dy : float
        Grid spacing in metres.
    h_min : float
        Heights below this fade to transparent, so the flow field shows
        around the mountain.
    exag : float
        Vertical exaggeration of the shading (presentation only).

    Returns
    -------
    ndarray (ny, nx, 4)
    """
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    hmax = max(float(h.max()), 1.0)
    rgb = ls.shade(h, cmap=HYPSO, vert_exag=exag, dx=dx, dy=dy,
                   blend_mode="soft", vmin=0.0, vmax=hmax)
    a = np.clip((h - h_min) / (4 * h_min), 0.0, 1.0) ** 0.7
    rgb[..., 3] = 0.92 * a
    return rgb


def mountain_marks(ax, X, Y, h, levels=(500, 1000, 1500, 2000),
                   color="#f4efe6", label=True, fontsize=7.5):
    """Labeled height contours (m) and a summit marker with its height.

    ``X, Y`` are the plotting coordinates (for example km) of ``h``.
    """
    cs = ax.contour(X, Y, h, levels=list(levels), colors=color,
                    linewidths=0.6, alpha=0.85, zorder=6)
    if label:
        ax.clabel(cs, fmt=lambda v: "%d m" % v, fontsize=fontsize - 1.5,
                  inline_spacing=2, colors=color)
    j = np.unravel_index(np.argmax(h), h.shape)
    ax.plot(X[j], Y[j], marker="^", ms=7, mfc="#ffffff", mec="#2b2b2b",
            mew=0.8, zorder=7)
    ax.annotate("summit %.1f km" % (h[j] / 1e3), (X[j], Y[j]),
                xytext=(8, 8), textcoords="offset points", color="#ffffff",
                fontsize=fontsize, fontweight="bold", zorder=8)
    return cs


def map_axes(ax, xlabel=r"$x$ (km)", ylabel=r"$y$ (km)", fontsize=8.5):
    """Dark map axes with distance labels and outward ticks."""
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color("#3a4a60")
        s.set_linewidth(0.8)
    ax.tick_params(colors=DIM, labelsize=fontsize - 1, direction="out",
                   length=3)
    ax.set_xlabel(xlabel, color=DIM, fontsize=fontsize, labelpad=2)
    ax.set_ylabel(ylabel, color=DIM, fontsize=fontsize, labelpad=2)


def unit_colorbar(fig, mappable, rect, label, fontsize=8.5,
                  orientation="vertical"):
    """A color bar on its own axes with a units label, dark styled."""
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation)
    cb.outline.set_edgecolor("#3a4a60")
    cb.ax.tick_params(colors=DIM, labelsize=fontsize - 1.5)
    cb.set_label(label, color=DIM, fontsize=fontsize)
    return cb


class TerrainView:
    """An oblique, lit 3D view of the model domain.

    The mountain is drawn as a surface with true heights (km) and a
    vertical exaggeration set by the box aspect; a field (streamfunction
    or PV) is draped over it as color, lit by the relief, and blended
    toward a hypsometric tint on the mountain so the landform stays
    visible. Distances are in km.

    Parameters
    ----------
    ax : Axes3D
    X, Y : ndarray (ny, nx)
        Coordinates in km.
    h : ndarray (ny, nx)
        Height in m.
    cmap, vmin, vmax
        Color map and limits of the draped field.
    z_aspect : float
        Height of the box relative to its meridional extent.
    """

    def __init__(self, ax, X, Y, h, cmap, vmin, vmax, z_aspect=0.45,
                 elev=32, azim=-58, tint=0.75, zoom=1.3):
        self.ax, self.X, self.Y = ax, X, Y
        self.hkm = h / 1e3
        self.cmap, self.vmin, self.vmax = cmap, vmin, vmax
        self.ls = LightSource(azdeg=300, altdeg=40)
        hypso = HYPSO(np.clip(h / max(h.max(), 1.0), 0, 1))[..., :3]
        w = np.clip((h - 80.0) / 500.0, 0, 1)[..., None] * tint
        self.hypso, self.w = hypso, w
        self.dx = (X[0, 1] - X[0, 0]) * 1e3
        self.dy = (Y[1, 0] - Y[0, 0]) * 1e3
        ax.set_facecolor(BG)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.04, 0.08, 0.13, 1.0))
            axis._axinfo["grid"]["color"] = (0.23, 0.29, 0.38, 0.5)
            axis._axinfo["grid"]["linewidth"] = 0.4
            axis.line.set_color("#3a4a60")
        ax.tick_params(colors=DIM, labelsize=7, pad=0)
        ax.computed_zorder = False
        ax.view_init(elev=elev, azim=azim)
        lx = X.max() - X.min()
        ly = Y.max() - Y.min()
        ax.set_box_aspect((lx / ly, 1.0, z_aspect), zoom=zoom)
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(0, max(self.hkm.max() * 1.15, 0.1))
        ax.set_xlabel(r"$x$ (km)", color=DIM, fontsize=9, labelpad=4)
        ax.set_ylabel(r"$y$ (km)", color=DIM, fontsize=9, labelpad=2)
        ax.set_zlabel(r"$h$ (km)", color=DIM, fontsize=9, labelpad=0)
        self.surf = None

    def colors(self, field):
        """Lit facecolors of the draped field."""
        from matplotlib.colors import Normalize
        rgb = self.cmap(Normalize(self.vmin, self.vmax)(field))[..., :3]
        rgb = (1 - self.w) * rgb + self.w * self.hypso
        lit = self.ls.shade_rgb(rgb, self.hkm * 1e3, vert_exag=25.0,
                                dx=self.dx, dy=self.dy, blend_mode="soft")
        return np.clip(lit, 0, 1)

    def draw(self, field):
        """Draw (or recolor) the surface with ``field`` draped on it."""
        fc = self.colors(field)
        if self.surf is None:
            self.surf = self.ax.plot_surface(
                self.X, self.Y, self.hkm, facecolors=fc, rstride=1,
                cstride=1, linewidth=0, antialiased=False, shade=False,
                zorder=1)
        else:
            # one face per grid cell, colored by its lower-left corner
            self.surf.set_facecolor(fc[:-1, :-1].reshape(-1, 3))

    def height_at(self, x, y):
        """Terrain height (km) at points by bilinear interpolation."""
        from scipy.interpolate import RegularGridInterpolator
        if not hasattr(self, "_interp"):
            self._interp = RegularGridInterpolator(
                (self.Y[:, 0], self.X[0]), self.hkm, bounds_error=False,
                fill_value=0.0)
        return self._interp(np.column_stack([np.ravel(y), np.ravel(x)]))

    def contours(self, levels=(500, 1000, 1500, 2000), color="#f4efe6",
                 summit_m=None):
        """Height contours (m) drawn on the terrain, and the summit.

        ``summit_m`` labels the true peak height when the grid does not
        sample the summit exactly.
        """
        import matplotlib.pyplot as plt
        tmp = plt.figure()
        cs = tmp.add_subplot().contour(self.X, self.Y, self.hkm * 1e3,
                                       levels=list(levels))
        for lev, segs in zip(cs.levels, cs.allsegs):
            for sg in segs:
                z = np.full(len(sg), lev / 1e3 + 0.01)
                self.ax.plot(sg[:, 0], sg[:, 1], z, color=color, lw=0.6,
                             alpha=0.8, zorder=4)
        plt.close(tmp)
        j = np.unravel_index(np.argmax(self.hkm), self.hkm.shape)
        x, y, z = self.X[j], self.Y[j], self.hkm[j]
        self.ax.plot([x, x], [y, y], [z, z * 1.12], color="w", lw=1.0,
                     zorder=6)
        zl = z if summit_m is None else summit_m / 1e3
        self.ax.text(x, y, z * 1.14, "%.1f km" % zl, color="w", fontsize=8,
                     fontweight="bold", zorder=7, ha="center")


def hillshade(eta, dx, dy, azdeg=315, altdeg=35):
    """RGBA hill shading of the orography (presentation only)."""
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    z = eta / (np.abs(eta).max() + 1e-12)
    return ls.shade(z, cmap=TERRAIN, vert_exag=6.0, dx=dx, dy=dy,
                    blend_mode="soft")


class Trails:
    """Fading parcel trails drawn as line segments.

    Parameters
    ----------
    ax : Axes
    n : int
        Number of parcels.
    length : int
        Number of stored positions per trail.
    period : tuple
        ``(Lx, Ly)``: segments that jump across a periodic boundary are
        hidden.
    color : str or None
        Fixed color; ``None`` colors by trail age with ``TRAIL``.
    """

    def __init__(self, ax, n, length, period, color=None, lw=1.1,
                 alpha=0.9, zorder=5):
        self.ax, self.n, self.len = ax, n, length
        self.lx, self.ly = period
        self.hist = []
        self.lc = LineCollection([], linewidths=lw, zorder=zorder,
                                 capstyle="round")
        self.color, self.alpha = color, alpha
        ax.add_collection(self.lc)
        self.heads = ax.scatter([], [], s=4, c="#ffffff", lw=0,
                                zorder=zorder + 1, alpha=0.9)

    def push(self, x, y):
        """Add the current (unwrapped) positions."""
        self.hist.append(np.array([np.mod(x, self.lx), np.mod(y, self.ly)]))
        if len(self.hist) > self.len:
            self.hist.pop(0)
        self._draw()

    def _draw(self):
        h = np.array(self.hist)          # (m, 2, n)
        if h.shape[0] < 2:
            return
        p0, p1 = h[:-1], h[1:]
        jump = ((np.abs(p1[:, 0] - p0[:, 0]) > 0.5 * self.lx)
                | (np.abs(p1[:, 1] - p0[:, 1]) > 0.5 * self.ly))
        seg = np.stack([p0, p1], axis=1)    # (m-1, 2pts, 2, n)
        seg = np.moveaxis(seg, 3, 1)        # (m-1, n, 2pts, 2)
        age = np.linspace(0.05, 1.0, h.shape[0] - 1)[:, None] * \
            np.ones((1, self.n))
        ok = ~jump
        segs = seg[ok]
        a = age[ok]
        if self.color is None:
            cols = TRAIL(a)
        else:
            cols = np.tile(to_rgba(self.color), (a.size, 1))
        cols[:, 3] = self.alpha * a ** 1.5
        self.lc.set_segments(segs)
        self.lc.set_color(cols)
        self.heads.set_offsets(h[-1].T)


def save_animation(anim, mp4_path, gif_path, fps=30, gif_fps=10,
                   gif_width=560):
    """Write an MP4 with FFmpeg and a palette-optimized GIF from it."""
    anim.save(mp4_path, fps=fps, dpi=100,
              extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                          "-preset", "veryfast", "-crf", "20"])
    vf = ("fps=%d,scale=%d:-1:flags=lanczos,split[s0][s1];"
          "[s0]palettegen=max_colors=160[p];[s1][p]paletteuse="
          "dither=bayer:bayer_scale=4" % (gif_fps, gif_width))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4_path,
                    "-vf", vf, gif_path], check=True)


def render(fig, frame, n_frames, name, fps=30, gif_fps=10, gif_width=560):
    """Encode an animation, or preview a few frames as PNG.

    If the environment variable ``QG_PREVIEW`` holds comma-separated
    frame numbers, only those frames are drawn and saved as
    ``outputs/animations/<name>_<frame>.png``.
    """
    from matplotlib.animation import FuncAnimation

    from .io import path
    sel = os.environ.get("QG_PREVIEW")
    if sel:
        wanted = sorted(int(k) for k in sel.split(","))
        for k in range(max(wanted) + 1):
            frame(k)
            if k in wanted:
                fig.savefig(path("animations", "%s_%04d.png" % (name, k)),
                            dpi=100, facecolor=fig.get_facecolor())
        return
    anim = FuncAnimation(fig, frame, frames=n_frames, blit=False)
    save_animation(anim, path("animations", name + ".mp4"),
                   path("animations", name + ".gif"), fps=fps,
                   gif_fps=gif_fps, gif_width=gif_width)


def dark_axes(ax):
    """Style an axes for the dark animations."""
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color("#26354a")
    ax.tick_params(colors=DIM, labelsize=8)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
