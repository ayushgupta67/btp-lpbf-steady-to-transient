"""3D figures of the geometry part (tts_outputs/fig3d_*.png).   python tts_fig3d.py"""
import csv
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import tts_data as td

g = td.geometry(td.ROOT / "part_1" / "SS")
vox = g["vox"]  # (z, y, x) for the 10 part layers
ez, ey, ex = vox.shape
part = np.transpose(vox, (2, 1, 0))  # matplotlib voxels want (x, y, z)
sub = np.ones((ex, ey, 3), bool)     # substrate, 3 layers below the part
TEAL, GREY = "#1F7A8C", "#C9CCD3"


def draw(ax, filled_part, colours=None, cut=None, sub_on=True):
    full = np.concatenate([sub & sub_on, filled_part], axis=2)
    fc = np.empty(full.shape, dtype=object)
    fc[...] = GREY
    if colours is None:
        fc[:, :, 3:][filled_part] = TEAL
    else:
        fc[:, :, 3:] = colours
    if cut is not None:
        full = full & cut
    ax.voxels(full, facecolors=fc, edgecolor=(1, 1, 1, 0.25), linewidth=0.3)
    ax.set_box_aspect((ex, ey, ez + 3))
    ax.set_axis_off()


# 1) shape: full view + cut-away through the small hole (remove y >= 16)
fig = plt.figure(figsize=(12, 5.2))
for k, (title, cut, elev, azim) in enumerate((("whole part (substrate grey)", None, 28, -55),
                                               ("top view (lighter = higher layer)", None, 90, -90))):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    if k == 1:  # top view: shade by height so the openings show their depth
        shades = matplotlib.colormaps["GnBu"](np.linspace(0.95, 0.35, ez))
        col = np.empty(part.shape, dtype=object)
        for z in range(ez):
            col[:, :, z] = colors.to_hex(shades[z])
        draw(ax, part, colours=col)
    else:
        draw(ax, part)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=13)
fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0, wspace=0)
fig.savefig(td.OUT / "fig3d_part.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# 2) peaks in 3D, coloured per element (cut open to show the inside)
val = {"ss_peak": np.full((ex, ey, ez), np.nan), "tr_peak": np.full((ex, ey, ez), np.nan)}
for r in csv.DictReader(open(td.OUT / "elements.csv")):
    for key in val:
        val[key][int(r["ix"]), int(r["iy"]), int(r["layer"]) - 1] = float(r[key])
fig = plt.figure(figsize=(12.5, 5.4))
for k, (key, title) in enumerate((("ss_peak", "steady peak per element (cut at y = 15 mm)"), ("tr_peak", "transient peak per element (cut at y = 15 mm)"))):
    v = val[key]
    lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
    cmap = matplotlib.colormaps["inferno"]
    col = np.empty(v.shape, dtype=object)
    norm = colors.Normalize(lo, hi)
    for idx in np.ndindex(v.shape):
        col[idx] = cmap(norm(v[idx])) if np.isfinite(v[idx]) else GREY
    c = np.ones((ex, ey, ez + 3), bool); c[:, :15, :] = False
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    draw(ax, part, colours=col, cut=c)
    ax.view_init(elev=24, azim=-58)
    ax.set_title(title, fontsize=13)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.0)
    cb.set_label("°C", fontsize=10)
fig.subplots_adjust(left=0, right=0.97, top=0.93, bottom=0, wspace=0.05)
fig.text(0.02, 0.10, "dark band = steady-script bug:\nlayer 2 sits on the only 'cold' layer\n(wrong local-domain centre), not physics",
         fontsize=10.5, color="#C8102E", fontweight="bold", va="bottom")
fig.patches.append(matplotlib.patches.FancyArrowPatch((0.09, 0.215), (0.115, 0.37), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#C8102E"))
fig.savefig(td.OUT / "fig3d_peaks.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("ok")
