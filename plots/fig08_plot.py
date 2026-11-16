"""
Figure 8. Synthesis of the bottom boundary layer and of the supply of SACW at the shelf
break of the northern South Brazil Bight, drawn as a block diagram.

The block spans an idealised line of the array from the coast to the upper slope, and a
stretch of the coast holding one crest (upwelling phase) and one trough (downwelling
phase) of a coastal trapped wave travelling toward the northeast. The along-shore
distance is not to scale and the depth is exaggerated.

Southwest end face, cut through the crest
    the SACW below the 20 C isotherm (blue) beneath the warmer water (cream), the 20 C
    isotherm lifted by the wave (bold) above its position before the uplift (dashed),
    the uplift across the shelf (purple arrows), and at the shelf break the stratified
    bottom boundary layer with its upslope transport (red).
Shelf and slope
    the 20 C isotherm as a surface (translucent blue), lifted beneath the crest and
    lowered beneath the trough, with its intersection with the bottom (dark blue line),
    the front of the SACW, moving toward the coast and back; the bottom boundary layer at the shelf break, stratified where the interior flow runs
    southwest and the Ekman transport is upslope (blue) and mixed where it runs northeast
    and the transport is downslope (orange), with its transport across the isobaths
    (red); the interior flow (black arrow) and the near-bottom flow (white arrow) in each
    state, the near-bottom arrow drawn with the attenuation and the veering measured at
    FUB3 (shorter and aligned under upslope transport, longer and veered under downslope
    transport).
Sea surface
    the Brazil Current toward the southwest over the slope and the coastal trapped wave
    toward the northeast.

The attenuation and veering of each state set the near-bottom arrows and are read from
paper_v3/tables/fig07_numbers.csv when it holds them (keys FUB3_gain_up, FUB3_gain_dn,
FUB3_veer_up, FUB3_veer_dn), otherwise from the values of the manuscript.

Reads paper_v3/tables/fig07_numbers.csv. Writes paper_v3/figures/fig08_synthesis.png
and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow
from matplotlib.legend_handler import HandlerPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import PchipInterpolator

import config_v3 as V
import style as S

FIGSIZE = (7.48, 5.0)

# ------------------------------------------------------------- geometry of the block
L_ALONG = 300.0                     # along-shore length drawn (km, not to scale)
L_OFF = 120.0                       # offshore extent from the coast (km)
L_LAND = 14.0                       # land drawn behind the coast (km)
Z_BOX = 265.0                       # depth of the block (m), the slope below is cut
Z_TOP = 55.0                        # room above the sea surface for the land (m)
O_NODES = [0, 6, 20, 40, 66, 90, 104, 112, 120, 128, 140]
D_NODES = [0, 22, 50, 70, 100, 125, 165, 200, 265, 330, 420]
DEPTH = PchipInterpolator(O_NODES, D_NODES)
O_BREAK = 112.0                     # the 200 m isobath (km from the coast)

# ------------------------------------------------- the 20 C isotherm and the wave
Z20 = 80.0                          # depth of the 20 C isotherm at the 200 m isobath at rest (m)
TILT = 0.20                         # m km-1, the isotherm deepens offshore beneath the current
AMP = 24.0                          # uplift of the isotherm on the shelf by the wave (m)
WAVELENGTH = 1.35 * L_ALONG
X_CREST = 0.08 * L_ALONG            # crest near the southwest end face
TRAP_KM = 30.0                      # offshore decay of the wave beyond the shelf break
LAYER = (150.0, Z_BOX - 8.0)        # depths of the slope where the bottom boundary layer is drawn
H_LAYER = 28.0                      # thickness of the layer as drawn on the end face (m)

# ------------------------------------------------------------------- view and look
ELEV, AZIM = 30.0, -140.0           # camera, offshore and to the southwest of the block
BOX_ASPECT = (2.05, 1.35, 1.20)     # drawn lengths along the coast, across it and in depth
ZOOM = 1.00
DATA_RANGE = np.array([L_ALONG, L_OFF + L_LAND, Z_BOX + Z_TOP])
TO_BOX = np.array(BOX_ASPECT) / DATA_RANGE          # data units to box units, per axis
LIGHT = (-105.0, 48.0)              # azimuth and altitude of the light (degrees)
SEAFLOOR = 'lightgray'
LAND = 'tan'
SEA = 'lightblue'
WARM = 'oldlace'                    # water above the 20 C isotherm on the end face
SACW = 'lightblue'                  # water below it
ISOSURF = 'royalblue'
FRONT = 'navy'
CAPPED = 'cornflowerblue'           # stratified layer under upslope transport
MIXED = 'navajowhite'               # mixed layer under downslope transport
INTERIOR = 'black'                  # interior flow, filled arrow
NEARBOTTOM = 'white'                # near-bottom flow, open arrow
CURRENT_FILL = 'paleturquoise'
CURRENT_EDGE = 'teal'
EDGE = 'dimgray'
LABEL = 7.6                         # size of the labels written in the block

f_num = V.DIR_TAB / 'fig07_numbers.csv'
num = pd.read_csv(f_num).set_index('key') if f_num.exists() else pd.DataFrame(columns=['value'])


def n(key, default):
    return float(num.loc[key, 'value']) if key in num.index and np.isfinite(num.loc[key, 'value']) else default


GAIN = {'up': n('FUB3_gain_up', 0.54), 'dn': n('FUB3_gain_dn', 0.72)}
VEER = {'up': n('FUB3_veer_up', -4.0), 'dn': n('FUB3_veer_dn', -18.0)}


def eta(x, o):
    """Uplift of the isotherm by the wave (m, positive upward), uniform over the shelf and
    decaying offshore of the shelf break."""
    trap = np.where(o <= O_BREAK, 1.0, np.exp(-(o - O_BREAK) / TRAP_KM))
    return AMP * np.cos(2.0 * np.pi * (x - X_CREST) / WAVELENGTH) * trap


def z20(x, o, wave=True):
    """Depth of the 20 C isotherm (m, positive down)."""
    return Z20 + TILT * (o - O_BREAK) - (eta(x, o) if wave else 0.0)


def arrow(ax, pts, width, plane, colour, edge=None, head=0.30, head_width=2.3, zorder=10, alpha=1.0):
    """Flat arrow along the polyline pts (N by 3 in data units, x along the coast, y the
    minus offshore distance, z elevation), lying in the plane of normal plane, its width
    and head measured in units of the drawn box so that every arrow looks alike."""
    P = np.asarray(pts, float) * TO_BOX
    t = np.gradient(P, axis=0)
    side = np.cross(t, np.asarray(plane, float))
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    k = max(1, int(round((1.0 - head) * (len(P) - 1))))
    left = P[:k + 1] + 0.5 * width * side[:k + 1]
    right = P[:k + 1] - 0.5 * width * side[:k + 1]
    poly = np.vstack([left, P[k] + 0.5 * head_width * width * side[k], P[-1],
                      P[k] - 0.5 * head_width * width * side[k], right[::-1]]) / TO_BOX
    ax.add_collection3d(Poly3DCollection([poly], facecolor=colour, edgecolor=edge or colour, linewidth=0.6,
                                         alpha=alpha, zorder=zorder))


def shaded(colour, X, Y, Z):
    """Uniform colour lit from above the viewer, the relief measured in units of the drawn box."""
    xb, yb, zb = X * TO_BOX[0], Y * TO_BOX[1], Z * TO_BOX[2]
    dz_x = np.gradient(zb, axis=1) / np.gradient(xb, axis=1)
    dz_y = np.gradient(zb, axis=0) / np.gradient(yb, axis=0)
    normal = np.dstack([-dz_x, -dz_y, np.ones_like(zb)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    az, alt = np.radians(LIGHT[0]), np.radians(LIGHT[1])
    light = np.array([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)])
    lit = 0.58 + 0.42 * np.clip(normal @ light, 0.0, 1.0)
    return np.clip(lit[..., None] * np.array(to_rgb(colour)), 0.0, 1.0)


# --------------------------------------------------------------------- the figure
fig = plt.figure(figsize=FIGSIZE)
gs = gridspec.GridSpec(1, 1, figure=fig, left=-0.08, right=1.06, top=1.09, bottom=-0.03)
ax = fig.add_subplot(gs[0, 0], projection='3d', computed_zorder=False)

# plot coordinates: x along the coast toward the northeast (km), y minus the offshore
# distance so that the sea lies toward the viewer (km), z elevation (m)
x1 = np.linspace(0.0, L_ALONG, 150)

# ------------------------------------------------------------------ land and seafloor
o_land = np.linspace(-L_LAND, 0.0, 16)
XL, OL = np.meshgrid(x1, o_land)
ZL = Z_TOP * 0.9 * (1.0 - np.exp(OL / 4.0)) * (1.0 + 0.22 * np.sin(XL / 19.0) * np.cos(XL / 37.0))
ax.plot_surface(XL, -OL, ZL, facecolors=shaded(LAND, XL, -OL, ZL), shade=False, rstride=1, cstride=1, linewidth=0,
                antialiased=False, rasterized=True, zorder=1)
o_sea = np.linspace(0.0, L_OFF, 140)
XS, OS = np.meshgrid(x1, o_sea)
ZS = -np.minimum(DEPTH(OS), Z_BOX)
ax.plot_surface(XS, -OS, ZS, facecolors=shaded(SEAFLOOR, XS, -OS, ZS), shade=False, rstride=1, cstride=1,
                linewidth=0, antialiased=False, rasterized=True, zorder=1)
ax.plot(x1, np.zeros_like(x1), np.zeros_like(x1), color=EDGE, lw=0.9, zorder=2)                 # coastline

# ------------------------------------------------ bottom boundary layer at the shelf break
o_lay = np.linspace(float(np.interp(LAYER[0], D_NODES, O_NODES)), float(np.interp(LAYER[1], D_NODES, O_NODES)), 24)
XB, OB = np.meshgrid(x1, o_lay)
ZB = -np.minimum(DEPTH(OB), Z_BOX) + 2.0
upslope = eta(XB, O_BREAK) > 0.0                  # interior flow toward the southwest beneath the crest
fc = np.where(upslope[..., None], np.array(to_rgb(CAPPED)), np.array(to_rgb(MIXED)))
ax.plot_surface(XB, -OB, ZB, facecolors=fc, shade=False, rstride=1, cstride=1, linewidth=0, antialiased=False,
                rasterized=True, zorder=3)
x_zero = X_CREST + 0.25 * WAVELENGTH               # the layer changes state where the interior flow reverses
ax.plot(np.full_like(o_lay, x_zero), -o_lay, ZB[:, 0], color=EDGE, lw=0.6, zorder=3)

# transport of the layer across the isobaths, upslope beneath the crest, downslope beneath the trough
o_path = np.linspace(123.0, 103.0, 30)
for xa, up in ((0.39 * L_ALONG, True), (0.53 * L_ALONG, False), (0.92 * L_ALONG, False)):
    path = o_path if up else o_path[::-1]
    pts = np.c_[np.full_like(path, xa), -path, -np.minimum(DEPTH(path), Z_BOX) + 8.0]
    arrow(ax, pts, 0.045, (1.0, 0.0, 0.0), S.SYN_TRANSPORT, edge='darkred', zorder=4)

# interior flow and near-bottom flow above the 200 m isobath in each state
LENGTH = 52.0
for xc, key in ((0.29 * L_ALONG, 'up'), (0.68 * L_ALONG, 'dn')):
    sgn = -1.0 if key == 'up' else 1.0             # southwest beneath the crest, northeast beneath the trough
    ax.plot([xc, xc], [-O_BREAK, -O_BREAK], [-120.0, -198.0], color=EDGE, lw=0.6, ls=':', zorder=5)
    s_ = np.linspace(0.0, 1.0, 16) * LENGTH
    arrow(ax, np.c_[xc + sgn * s_, np.full_like(s_, -O_BREAK), np.full_like(s_, -155.0)], 0.045,
          (0.0, 0.0, 1.0), INTERIOR, edge='black', zorder=5)
    th = np.radians(VEER[key])                     # clockwise negative, seen from above
    ux, uo = sgn * np.cos(th), sgn * np.sin(th)     # uo positive onshore
    sb = s_ * GAIN[key]
    arrow(ax, np.c_[xc + sb * ux, -O_BREAK + sb * uo, np.full_like(sb, -194.0)], 0.045,
          (0.0, 0.0, 1.0), NEARBOTTOM, edge='black', zorder=5)

# ------------------------------------------------------ the 20 C isotherm as a surface
o_fine = np.linspace(0.0, L_OFF, 800)
o_int = np.empty_like(x1)
for i, xv in enumerate(x1):
    o_int[i] = o_fine[np.argmax(z20(xv, o_fine) < DEPTH(o_fine))]    # where the isotherm meets the bottom
O_ISO = O_BREAK + 4.0                              # the surface is drawn over the shelf
sg = np.linspace(0.0, 1.0, 50)
XI = np.repeat(x1[None, :], sg.size, axis=0)
OI = o_int[None, :] + sg[:, None] * (O_ISO - o_int[None, :])
ZI = -z20(XI, OI)
ax.plot_surface(XI, -OI, ZI, color=ISOSURF, alpha=0.22, shade=False, rstride=1, cstride=1, linewidth=0,
                antialiased=True, rasterized=True, zorder=6)
for o_line in np.r_[np.arange(30.0, O_ISO - 1.0, 20.0), O_ISO]:    # lines of constant offshore distance
    xs_ = x1[o_int < o_line - 1.0]
    ax.plot(xs_, np.full_like(xs_, -o_line), -z20(xs_, o_line), color=FRONT, lw=0.45, alpha=0.5, zorder=6)
ax.plot(x1, -o_int, -z20(x1, o_int) + 1.0, color=FRONT, lw=1.8, zorder=6)    # front of the SACW on the bottom

# --------------------------------------------- southwest end face, the crest of the wave
o_face = np.linspace(0.0, L_OFF, 300)
h_face = np.minimum(DEPTH(o_face), Z_BOX)
z_lift = np.minimum(z20(0.0, o_face), h_face)     # the lifted isotherm, on the bottom where it meets it
warm = np.vstack([np.c_[np.zeros_like(o_face), -o_face, np.zeros_like(o_face)],
                  np.c_[np.zeros_like(o_face), -o_face[::-1], -z_lift[::-1]]])
cold = np.vstack([np.c_[np.zeros_like(o_face), -o_face, -z_lift],
                  np.c_[np.zeros_like(o_face), -o_face[::-1], -h_face[::-1]]])
ax.add_collection3d(Poly3DCollection([warm], facecolor=WARM, edgecolor='none', zorder=7))
ax.add_collection3d(Poly3DCollection([cold], facecolor=SACW, edgecolor='none', zorder=7))
ax.plot(np.zeros_like(o_face), -o_face, -h_face, color=EDGE, lw=0.9, zorder=8)
inside = z20(0.0, o_face) < h_face - 0.5
ax.plot(np.zeros_like(o_face[inside]), -o_face[inside], -z20(0.0, o_face[inside]), color='black', lw=1.8, zorder=9)
rest = z20(0.0, o_face, wave=False)
inside = rest < h_face - 0.5
ax.plot(np.zeros_like(o_face[inside]), -o_face[inside], -rest[inside], color='black', lw=1.0, ls=(0, (3, 2)), zorder=9)
for om in (30.0, 62.0, 94.0):                      # uplift across the whole shelf
    zz = np.linspace(-float(z20(0.0, om, wave=False)), -float(z20(0.0, om)), 12)
    arrow(ax, np.c_[np.zeros_like(zz), np.full_like(zz, -om), zz], 0.032, (1.0, 0.0, 0.0), S.SYN_WAVE, head=0.42,
          head_width=2.4, zorder=12)
# stratified layer and its upslope transport on the face
o_l = o_face[(DEPTH(o_face) > LAYER[0]) & (DEPTH(o_face) < LAYER[1])]
poly = np.vstack([np.c_[np.zeros_like(o_l), -o_l, -DEPTH(o_l)],
                  np.c_[np.zeros_like(o_l), -o_l[::-1], -DEPTH(o_l[::-1]) + H_LAYER]])
ax.add_collection3d(Poly3DCollection([poly], facecolor=CAPPED, edgecolor='steelblue', lw=0.8, alpha=0.9, zorder=9))
p_face = np.linspace(o_l.max() - 1.0, o_l.min() + 1.0, 30)
arrow(ax, np.c_[np.zeros_like(p_face), -p_face, -DEPTH(p_face) + 0.45 * H_LAYER], 0.04, (1.0, 0.0, 0.0),
      S.SYN_TRANSPORT, edge='darkred', zorder=10)
ax.text(0.0, -58.0, -105.0, 'SACW', fontsize=LABEL, fontweight='bold', color='steelblue', ha='center', va='center',
        zorder=14)
ax.text(0.0, -47.0, -float(z20(0.0, 47.0)) + 6.0, '20 °C', fontsize=LABEL, fontweight='bold', ha='center',
        va='bottom', zorder=14)

# ----------------------------------------------------------- the box and the surface
xs_sea, os_sea = np.meshgrid(np.array([0.0, L_ALONG]), np.array([0.0, L_OFF]))
ax.plot_surface(xs_sea, -os_sea, np.zeros_like(xs_sea), color=SEA, alpha=0.20, shade=False, linewidth=0, zorder=12)
xs_off, zs_off = np.meshgrid(np.array([0.0, L_ALONG]), np.array([-Z_BOX, 0.0]))       # offshore face of the water
ax.plot_surface(xs_off, np.full_like(xs_off, -L_OFF), zs_off, color=SEA, alpha=0.12, shade=False, linewidth=0,
                zorder=12)
for (xa, oa), (xb, ob) in (((0, 0), (0, L_OFF)), ((0, L_OFF), (L_ALONG, L_OFF)), ((L_ALONG, L_OFF), (L_ALONG, 0))):
    ax.plot([xa, xb], [-oa, -ob], [0, 0], color=EDGE, lw=0.7, zorder=13)
for xa in (0.0, L_ALONG):
    ax.plot([xa, xa], [-L_OFF, -L_OFF], [0, -Z_BOX], color=EDGE, lw=0.7, zorder=13)
ax.plot(x1, np.full_like(x1, -L_OFF), np.full_like(x1, -Z_BOX), color=EDGE, lw=0.7, zorder=13)

# Brazil Current over the slope, toward the southwest
xs = np.linspace(L_ALONG - 6.0, 0.30 * L_ALONG, 30)
arrow(ax, np.c_[xs, np.full_like(xs, -L_OFF + 6.0), np.zeros_like(xs)], 0.08, (0.0, 0.0, 1.0), CURRENT_FILL,
      edge=CURRENT_EDGE, head=0.14, head_width=1.7, alpha=0.9, zorder=13)
ax.text(0.78 * L_ALONG, -L_OFF + 6.0, 8.0, 'Brazil Current', fontsize=LABEL, ha='center', va='bottom', zorder=14)
# coastal trapped wave, toward the northeast over the shelf
xs = np.linspace(0.30 * L_ALONG, 0.62 * L_ALONG, 20)
arrow(ax, np.c_[xs, np.full_like(xs, -42.0), np.full_like(xs, 2.0)], 0.07, (0.0, 0.0, 1.0), S.SYN_WAVE,
      head=0.26, head_width=2.1, zorder=13)
ax.text(0.40 * L_ALONG, -10.0, 4.0, 'coastal trapped wave', fontsize=LABEL, color=S.SYN_WAVE, fontweight='bold',
        ha='center', va='bottom', zorder=14)
# the two phases of the wave and the two states of the layer
for xa, word in ((X_CREST + 0.06 * L_ALONG, 'upwelling phase'), (X_CREST + 0.5 * WAVELENGTH, 'downwelling phase')):
    ax.text(xa, 6.0, Z_TOP + 4.0, word, fontsize=LABEL, fontweight='bold', ha='center', va='bottom', zorder=14)
ax.text(0.18 * L_ALONG, -L_OFF - 4.0, -232.0, 'stratified layer', fontsize=LABEL, color='midnightblue',
        ha='center', va='center', zorder=14)
ax.text(0.80 * L_ALONG, -L_OFF - 4.0, -232.0, 'mixed layer', fontsize=LABEL, color='saddlebrown', ha='center',
        va='center', zorder=14)
ax.text(2.0, 4.0, Z_TOP * 0.55, 'SW', fontsize=LABEL, fontweight='bold', ha='right', zorder=14)
ax.text(L_ALONG - 2.0, 4.0, Z_TOP * 0.55, 'NE', fontsize=LABEL, fontweight='bold', ha='left', zorder=14)

# ---------------------------------------------------------------------------- view
ax.set_xlim(0.0, L_ALONG)
ax.set_ylim(-L_OFF, L_LAND)
ax.set_zlim(-Z_BOX, Z_TOP)
ax.set_box_aspect(BOX_ASPECT, zoom=ZOOM)
ax.view_init(elev=ELEV, azim=AZIM)
ax.set_proj_type('persp', focal_length=0.55)
ax.set_axis_off()

# --------------------------------------------------- legend in the lower left corner
def legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return FancyArrow(0.0, 0.5 * height, width, 0.0, width=0.42 * height, head_width=1.05 * height,
                      head_length=0.38 * width, length_includes_head=True)


h = [FancyArrow(0, 0, 1, 0, fc=INTERIOR, ec='black', lw=0.6),
     FancyArrow(0, 0, 1, 0, fc=NEARBOTTOM, ec='black', lw=0.6),
     FancyArrow(0, 0, 1, 0, fc=S.SYN_TRANSPORT, ec='darkred', lw=0.6),
     FancyArrow(0, 0, 1, 0, fc=S.SYN_WAVE, ec=S.SYN_WAVE, lw=0.6),
     Line2D([], [], color=FRONT, lw=1.8), Line2D([], [], color='black', lw=1.0, ls=(0, (3, 2)))]
lab = ['interior flow', 'near-bottom flow', 'transport of the bottom\nboundary layer', 'uplift by the wave',
       'front of the SACW\non the bottom', '20 °C isotherm\nbefore the uplift']
fig.legend(h, lab, loc='lower left', bbox_to_anchor=(0.012, 0.02), ncol=1, frameon=False, fontsize=S.FS_SMALL,
           handlelength=2.0, handleheight=0.9, handletextpad=0.5, labelspacing=0.55,
           handler_map={FancyArrow: HandlerPatch(patch_func=legend_arrow)})

fig.savefig(V.DIR_FIG / 'fig08_synthesis.pdf', dpi=400)
S.save(fig, V.DIR_FIG / 'fig08_synthesis.png')
