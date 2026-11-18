"""
Figure 8. Synthesis of the supply of SACW to the shelf break of the northern South
Brazil Bight, a schematic drawn entirely in code. Two block diagrams of the shelf and
the upper slope seen from the southwest end, the coast on the left

    blocks[0, 0]   the downwelling phase of a coastal trapped wave, the isotherms
                   lowered, the transport of the bottom boundary layer downslope
                   and the layer mixed
    blocks[0, 1]   the upwelling phase two to three days later, the isotherms lifted
                   across the shelf, the transport upslope and the layer stratified

and below them three panels on the mechanisms, numbered as the markers in the blocks

    mech[0, 0]   1 the wave forced by winds farther south moves the isotherms
                   across the whole shelf
    mech[0, 1]   2 the buoyancy arrest, the lifted dense water caps the layer and
                   weakens the near-bottom flow only under upslope transport
    mech[0, 2]   3 the net supply, the near-bottom flow turns onshore when the
                   water is coldest

The top face of each block is the sea surface (the coastal trapped wave, the Brazil
Current and one of its cyclonic meanders) and the front face is a section across the
shelf (the 20 C isotherm over the SACW, the bottom boundary layer at the shelf break,
the interior and the near-bottom flow and the transport across the isobaths). Depth
and distance are schematic.

The flow symbols of the sections and the vectors of the second panel are scaled with
the attenuation and the veering measured at FUB3, read from
paper_v3/tables/fig07_numbers.csv when it holds them (keys FUB3_gain_up, FUB3_gain_dn,
FUB3_veer_up, FUB3_veer_dn), otherwise from the values of the manuscript.

Every colour is a named colour, the legend sits below the panels and no figure text
has a comma.

Reads paper_v3/tables/fig07_numbers.csv. Writes paper_v3/figures/fig08_synthesis.png
and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import PchipInterpolator

import config_v3 as V
import style as S

# =====================================================================
# CONFIGURATION
# =====================================================================

FIGSIZE = (7.48, 5.55)
VIEW = dict(elev=23.0, azim=-61.0)      # seen from offshore of the southwest end
BOX_ASPECT = (315.0, 190.0, 200.0)      # x, y and z lengths of the drawn box
ZOOM = 1.18
BLOCK_TOP, BLOCK_BOTTOM = 0.975, 0.42   # figure fractions of the two blocks
MECH_TOP, MECH_BOTTOM = 0.335, 0.105    # figure fractions of the three mechanism panels
MECH_WIDTHS = (1.0, 1.35, 1.0)          # relative widths, the arrest panel holds two states
REVERSAL_Y = 0.875                       # height of the reversal arrow, fraction of the block row

# geometry, x across the shelf (km, coast at 0), y along it toward the northeast (km),
# z schematic (surface 0, bottom of the section -1)
X_LAND, X_OFF = -25.0, 290.0
Y_SW, Y_NE = 0.0, 190.0
Z_FLOOR = -1.08                         # base of the block
Z_LAND = 0.04                           # height of the land
Z_TOP = 0.10                            # top of the drawn box
X_NODES = [0, 40, 90, 130, 150, 168, 195, 218, 245, 290]
D_NODES = [0.035, 0.08, 0.14, 0.20, 0.25, 0.37, 0.68, 0.90, 1.0, 1.0]
DEPTH = PchipInterpolator(X_NODES, D_NODES)
X_BREAK = 150.0                         # the shelf break
X_BC = 236.0                            # core of the Brazil Current
X_COLUMN = 181.0                        # where the interior and the near-bottom flow are drawn
XC, YC, RC = 218.0, 112.0, 25.0         # cyclonic meander, centre and radius

# the isotherms and the wave
Z_ISO = (-0.13, -0.20, -0.28, -0.36, -0.45, -0.55, -0.66, -0.78, -0.90)   # the first is the 20 C isotherm
UPLIFT = 0.06                          # displacement of the isotherms by the wave on the shelf
X_LAYER = (132.0, 208.0)                 # reach of the bottom boundary layer drawn at the shelf break
H_LAYER = {'up': 0.085, 'down': 0.115}  # thickness of the stratified and of the mixed layer
D_BEND = 26.0                           # distance the isotherms are carried up the slope in the layer (km)

# named colours, one per quantity
C_WAVE = 'rebeccapurple'                # coastal trapped wave and its uplift
C_TRANSPORT = 'firebrick'               # transport of the bottom boundary layer across the isobaths
C_BC = 'midnightblue'                   # Brazil Current
C_INTERIOR = 'black'                    # interior flow
C_NEAR = 'white'                        # near-bottom flow
C_T20 = 'black'                         # 20 C isotherm, unlike the dark blue of the Brazil Current
C_REST = 'black'                        # 20 C isotherm before the wave
C_ISO = 'slategrey'                     # isotherms of the SACW
C_SURFACE = 'lightcyan'
C_FRONT = dict(warm='lightcyan', sacw='lightskyblue')
C_SIDE = dict(warm='paleturquoise', sacw='cornflowerblue')
C_STRAT = 'royalblue'                   # stratified layer under upslope transport
C_MIXED = 'gold'                        # mixed layer under downslope transport
C_EARTH, C_EARTH_SIDE, C_LAND = 'tan', 'burlywood', 'darkseagreen'
C_CYC = 'mistyrose'
C_WIND = 'dimgrey'
C_TEXT, C_NOTE = 'black', 'dimgrey'
PHASE_COLOUR = {'down': 'chocolate', 'up': 'mediumblue'}

FS = dict(small=S.FS_SMALL, label=S.FS + 0.5, title=S.FS + 2.5)

# what changes between the two blocks
PHASES = {
    'down': dict(title='Downwelling phase', sign=-1.0, layer='mixed\nlayer', toward_reader=False),
    'up': dict(title='Upwelling phase', sign=+1.0, layer='stratified\nlayer', toward_reader=True),
}

# the numbered markers that tie the blocks to the mechanism panels
MARKERS = {1: (58.0, 132.0, 0.0), 2: (140.0, 0.0, -0.62), 3: (62.0, 0.0, -0.23)}

# drawing order, the faces first and what lies on them after
Z_FACE, Z_ON_TOP, Z_SIDE, Z_FRONT, Z_ON_FRONT, Z_TEXT = 1, 3, 5, 6, 8, 20

# attenuation and veering of the near-bottom flow at FUB3
f_num = V.DIR_TAB / 'fig07_numbers.csv'
num = pd.read_csv(f_num).set_index('key') if f_num.exists() else pd.DataFrame(columns=['value'])


def n(key, default):
    return float(num.loc[key, 'value']) if key in num.index and np.isfinite(num.loc[key, 'value']) else default


GAIN = {'up': n('FUB3_gain_up', 0.54), 'down': n('FUB3_gain_dn', 0.72)}
VEER = {'up': n('FUB3_veer_up', -4.0), 'down': n('FUB3_veer_dn', -18.0)}


# =====================================================================
# GEOMETRY
# =====================================================================

def depth(x):
    """Schematic water depth across the shelf and the upper slope, positive down."""
    return DEPTH(np.clip(np.asarray(x, dtype=float), X_NODES[0], X_NODES[-1]))


def isotherm(x, z0, sign):
    """An isotherm of the section. It deepens offshore beneath the Brazil Current and
    the wave displaces it over the shelf, less so offshore of the shelf break."""
    x = np.asarray(x, dtype=float)
    deepening = -0.06 * 0.5 * (1.0 + np.tanh((x - X_BC) / 28.0))
    trapped = 0.30 + 0.70 * 0.5 * (1.0 - np.tanh((x - 235.0) / 30.0))
    return z0 + deepening + sign * UPLIFT * trapped


def in_layer(x, frac, key):
    """Points of the bottom boundary layer on the section, at the fraction frac of its
    thickness above the bottom at the distances x. The thickness is measured along the
    normal to the bottom as drawn, so the layer looks as thick on the steep upper slope
    as on the outer shelf, and it tapers toward both ends of its reach."""
    x = np.asarray(x, dtype=float)
    kx = BOX_ASPECT[0] / (X_OFF - X_LAND)           # drawn length of one unit of x
    kz = BOX_ASPECT[2] / (Z_TOP - Z_FLOOR)          # and of one unit of z
    s = np.clip((x - X_LAYER[0]) / (X_LAYER[1] - X_LAYER[0]), 0.0, 1.0)
    taper = np.where(s < 0.5, np.sin(np.pi * s) ** 0.5, np.sin(np.pi * s))
    tx, tz = kx * np.ones_like(x), -kz * DEPTH.derivative()(np.clip(x, X_NODES[0], X_NODES[-1]))
    d = frac * kz * H_LAYER[key] * taper / np.hypot(tx, tz)
    return x - d * tz / kx, -depth(x) + d * tx / kz


def layer_top(x, key):
    """Height of the top of the bottom boundary layer above the bottom, zero outside its reach."""
    xe, ze = in_layer(np.linspace(X_LAYER[0], X_LAYER[1], 400), 1.0, key)
    top = np.interp(x, np.maximum.accumulate(xe), ze, left=np.nan, right=np.nan)
    return np.nan_to_num(np.maximum(top + depth(x), 0.0), nan=0.0)


def isotherm_path(z0, key):
    """The line of an isotherm on the section. Where it meets the bottom boundary layer
    it bends up the slope under upslope transport (dense water lifted along the bottom,
    a stratified layer) and it falls to the bottom under downslope transport (light
    water pushed down the slope, a mixed layer)."""
    sign = PHASES[key]['sign']
    xs = np.linspace(0.0, X_OFF, 1400)
    z = isotherm(xs, z0, sign)
    hab = z + depth(xs)                         # height above the bottom
    above = hab > 0.0
    if not above.any():
        return np.array([]), np.array([])
    i0 = int(np.argmax(above))                  # where the undisturbed isotherm meets the bottom
    inside = above & (hab <= layer_top(xs, key))
    if not inside.any():
        return xs[i0:], z[i0:]
    ie = int(np.max(np.where(inside)[0])) + 1   # where it enters the layer
    xe, ze, he = xs[ie], z[ie], hab[ie]
    if key == 'down':
        return np.r_[xe, xs[ie:]], np.r_[-depth(xe), z[ie:]]
    s = np.linspace(1.0, 0.0, 40)
    xb = xe - D_BEND * s
    beta = (depth(xe) - depth(xe - D_BEND)) / D_BEND
    p = max(1.3, beta * D_BEND / he)            # joins the interior isotherm without a kink
    zb = -depth(xb) + he * (1.0 - s) ** p
    return np.r_[xb, xs[ie:]], np.r_[zb, z[ie:]]


# =====================================================================
# DRAWING TOOLS
# =====================================================================

class Arrow3D(FancyArrowPatch):
    """A flat arrow between two points of the block."""

    def __init__(self, p0, p1, **kw):
        super().__init__((0, 0), (0, 0), **kw)
        self._p0, self._p1 = p0, p1

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = proj3d.proj_transform(*zip(self._p0, self._p1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min(zs))


def face(ax, pts, colour, zorder, edge='none', lw=0.0, alpha=1.0):
    ax.add_collection3d(Poly3DCollection([pts], facecolors=colour, edgecolors=edge, linewidths=lw,
                                         alpha=alpha, zorder=zorder))


def head(ax, p0, p1, colour, zorder, size=7.0, lw=1.0):
    """An arrow head at p1 that points along p0 to p1."""
    ax.add_artist(Arrow3D(p0, p1, arrowstyle='-|>', mutation_scale=size, color=colour, lw=lw,
                          shrinkA=0, shrinkB=0, zorder=zorder))


def block_arrow(ax, p0, p1, colour, zorder, width=1.0, scale=11.0, edge='white'):
    """A filled arrow."""
    style = f'simple,head_length={0.95 * width:.2f},head_width={1.55 * width:.2f},tail_width={0.62 * width:.2f}'
    ax.add_artist(Arrow3D(p0, p1, arrowstyle=style, mutation_scale=scale, facecolor=colour, edgecolor=edge,
                          lw=0.5, shrinkA=0, shrinkB=0, zorder=zorder))


def arc(ax, cx, cy, r, th0, th1, colour, zorder, lw=1.2, z=0.0, size=6.5):
    """A circular arrow on a horizontal plane, clockwise where th1 < th0."""
    th = np.radians(np.linspace(th0, th1, 80))
    x, y = cx + r * np.cos(th), cy + r * np.sin(th)
    ax.plot(x[:-3], y[:-3], np.full(th.size - 3, z), color=colour, lw=lw, solid_capstyle='butt', zorder=zorder)
    head(ax, (x[-8], y[-8], z), (x[-1], y[-1], z), colour, zorder, size=size, lw=lw)


def along_bottom(ax, x0, x1, lift, colour, zorder, lw=2.6, size=11.0, key=None):
    """A thick arrow on the section that follows the bottom from x0 to x1, at the height
    lift above it or, with a key, at that fraction of the thickness of the layer."""
    x = np.linspace(x0, x1, 60)
    x, z = (x, -depth(x) + lift) if key is None else in_layer(x, lift, key)
    ax.plot(x[:-5], np.zeros(x.size - 5), z[:-5], color=colour, lw=lw, solid_capstyle='butt', zorder=zorder)
    head(ax, (x[-9], 0.0, z[-9]), (x[-1], 0.0, z[-1]), colour, zorder, size=size, lw=0.9 * lw)


def flow_symbol(ax, x, z, toward_reader, size, fill, ink, zorder, lw=0.9):
    """Flow through the section, a dot toward the reader (southwest) and a cross away
    from the reader (northeast)."""
    ax.scatter([x], [0.0], [z], s=size, facecolor=fill, edgecolor='black', linewidth=lw, depthshade=False,
               zorder=zorder)
    if toward_reader:
        ax.scatter([x], [0.0], [z], s=size / 9.0, color=ink, depthshade=False, zorder=zorder + 1)
    else:
        ax.scatter([x], [0.0], [z], s=0.42 * size, marker='x', color=ink, linewidth=lw, depthshade=False,
                   zorder=zorder + 1)


def marker(ax, k, x, y, z=None, size=None):
    """A numbered circle, in the block (3d) or in a mechanism panel (2d)."""
    kw = dict(ha='center', va='center', fontsize=size or FS['small'], fontweight='bold', color=C_TEXT,
              bbox=dict(boxstyle='circle,pad=0.22', fc='white', ec=C_TEXT, lw=0.8), zorder=Z_TEXT + 2)
    if z is None:
        ax.text(x, y, str(k), transform=ax.transAxes, **kw)
    else:
        ax.text(x, y, z, str(k), **kw)


# =====================================================================
# THE BLOCK
# =====================================================================

def draw_faces(ax, key):
    sign = PHASES[key]['sign']
    xs = np.linspace(0.0, X_OFF, 300)
    zb = -depth(xs)
    z20 = np.maximum(isotherm(xs, Z_ISO[0], sign), zb)
    z20_off = float(isotherm(X_OFF, Z_ISO[0], sign))
    # sea surface and land
    face(ax, [(0, Y_SW, 0), (X_OFF, Y_SW, 0), (X_OFF, Y_NE, 0), (0, Y_NE, 0)], C_SURFACE, Z_FACE,
         edge='lightslategrey', lw=0.4)
    face(ax, [(X_LAND, Y_SW, Z_LAND), (0, Y_SW, Z_LAND), (0, Y_NE, Z_LAND), (X_LAND, Y_NE, Z_LAND)],
         C_LAND, Z_FACE, edge='dimgrey', lw=0.4)
    face(ax, [(0, Y_SW, 0), (0, Y_NE, 0), (0, Y_NE, Z_LAND), (0, Y_SW, Z_LAND)], C_EARTH_SIDE, Z_FACE + 1,
         edge='dimgrey', lw=0.4)
    face(ax, [(X_LAND, Y_SW, Z_FLOOR), (0, Y_SW, Z_FLOOR), (0, Y_SW, Z_LAND), (X_LAND, Y_SW, Z_LAND)],
         C_EARTH, Z_FRONT, edge='dimgrey', lw=0.4)
    # the offshore face, the water above the 20 C isotherm, the SACW and the sea bed
    for z0, z1, c in ((0.0, z20_off, C_SIDE['warm']), (z20_off, -1.0, C_SIDE['sacw']), (-1.0, Z_FLOOR, C_EARTH_SIDE)):
        face(ax, [(X_OFF, Y_SW, z0), (X_OFF, Y_NE, z0), (X_OFF, Y_NE, z1), (X_OFF, Y_SW, z1)], c, Z_SIDE)
    ax.plot([X_OFF, X_OFF], [Y_SW, Y_NE], [z20_off, z20_off], color=C_T20, lw=1.6, zorder=Z_SIDE + 1)
    # the section, the same two waters cut by the sea bed
    for top, bot, c in ((np.zeros_like(xs), z20, C_FRONT['warm']), (z20, zb, C_FRONT['sacw'])):
        pts = [(x, Y_SW, z) for x, z in zip(xs, top)] + [(x, Y_SW, z) for x, z in zip(xs[::-1], bot[::-1])]
        face(ax, pts, c, Z_FRONT)
    pts = [(x, Y_SW, z) for x, z in zip(xs, zb)] + [(X_OFF, Y_SW, Z_FLOOR), (0, Y_SW, Z_FLOOR)]
    face(ax, pts, C_EARTH, Z_FRONT + 1.5, edge='dimgrey', lw=0.4)
    # edges of the block
    for xe, ye in ((X_OFF, Y_SW), (X_OFF, Y_NE)):
        ax.plot([xe, xe], [ye, ye], [0.0, Z_FLOOR], color='dimgrey', lw=0.5, zorder=Z_ON_FRONT)
    ax.plot([0, X_OFF], [Y_SW, Y_SW], [0, 0], color='dimgrey', lw=0.6, zorder=Z_ON_FRONT)
    ax.plot([X_OFF, X_OFF], [Y_SW, Y_NE], [0, 0], color='dimgrey', lw=0.6, zorder=Z_ON_FRONT)
    ax.plot([X_OFF, X_OFF], [Y_SW, Y_NE], [Z_FLOOR, Z_FLOOR], color='dimgrey', lw=0.5, zorder=Z_ON_FRONT)


def draw_surface(ax, labels):
    # the coastal trapped wave, toward the northeast along the coast
    block_arrow(ax, (26.0, 12.0, 0.0), (26.0, Y_NE - 12.0, 0.0), C_WAVE, Z_ON_TOP + 1, width=1.25)
    # a cyclonic meander of the Brazil Current (clockwise in the Southern Hemisphere)
    th = np.radians(np.linspace(0.0, 360.0, 120))
    face(ax, [(XC + RC * np.cos(t), YC + RC * np.sin(t), 0.0) for t in th], C_CYC, Z_ON_TOP - 1, edge=C_BC, lw=0.5)
    arc(ax, XC, YC, RC - 7.0, 250.0, -40.0, C_BC, Z_ON_TOP, lw=1.3)
    # the Brazil Current, toward the southwest over the slope and around the meander
    y = np.linspace(Y_NE - 5.0, 9.0, 160)
    x = X_BC + 42.0 * np.exp(-((y - YC) / 40.0) ** 2)
    ax.plot(x[:-4], y[:-4], np.zeros(y.size - 4), color=C_BC, lw=2.6, solid_capstyle='butt', zorder=Z_ON_TOP)
    for i in (24, 112, 159):
        head(ax, (x[i - 9], y[i - 9], 0.0), (x[i], y[i], 0.0), C_BC, Z_ON_TOP, size=10.0, lw=1.6)
    if labels:
        ax.text(X_LAND - 8.0, 112.0, Z_LAND, 'coast', ha='right', va='center', fontsize=FS['small'],
                color=C_TEXT, zorder=Z_TEXT)
        ax.text(48.0, 66.0, 0.0, 'coastal trapped wave', zdir=(0, 1, 0), ha='left', va='center',
                fontsize=FS['small'], color=C_TEXT, zorder=Z_TEXT)
        ax.text(XC - RC - 12.0, YC + 14.0, 0.0, 'cyclonic\nmeander', ha='right', va='center', linespacing=0.95, fontsize=FS['small'],
                color=C_TEXT, zorder=Z_TEXT)
        ax.text(X_BC - 18.0, 44.0, 0.0, 'Brazil\nCurrent', ha='right', va='center', fontsize=FS['small'],
                color=C_TEXT, linespacing=0.95, zorder=Z_TEXT)


def draw_section(ax, key, labels):
    ph = PHASES[key]
    sign = ph['sign']
    xs = np.linspace(0.0, X_OFF, 600)
    zb = -depth(xs)
    # the bottom boundary layer at the shelf break
    xl = np.linspace(X_LAYER[0], X_LAYER[1], 160)
    pts = [(x, 0.0, z) for x, z in zip(*in_layer(xl, 1.0, key))] + \
          [(x, 0.0, z) for x, z in zip(xl[::-1], -depth(xl[::-1]))]
    face(ax, pts, C_STRAT if key == 'up' else C_MIXED, Z_FRONT + 1, edge='black', lw=0.4,
         alpha=0.9)
    # the isotherms of the SACW, bent in the layer
    for z0 in Z_ISO[1:]:
        x, z = isotherm_path(z0, key)
        ax.plot(x, np.zeros_like(x), z, color=C_ISO, lw=0.7, zorder=Z_ON_FRONT - 1)
    # the 20 C isotherm, before the wave and displaced by it
    rest = isotherm(xs, Z_ISO[0], 0.0)
    ax.plot(xs, np.zeros_like(xs), np.where(rest > zb, rest, np.nan), color=C_REST, lw=0.9, ls=(0, (3, 2)),
            zorder=Z_ON_FRONT)
    now = isotherm(xs, Z_ISO[0], sign)
    ax.plot(xs, np.zeros_like(xs), np.where(now > zb, now, np.nan), color=C_T20, lw=2.0, zorder=Z_ON_FRONT)
    # the displacement by the wave across the shelf
    for x in ((128.0, 218.0) if key == 'up' else (218.0,)):
        z0, z1 = float(isotherm(x, Z_ISO[0], 0.0)), float(isotherm(x, Z_ISO[0], sign))
        block_arrow(ax, (x, 0.0, z0 - 0.04 * sign), (x, 0.0, z1 + 0.035 * sign), C_WAVE, Z_ON_FRONT + 3, width=0.75)
    # the Brazil Current toward the reader, weaker with depth
    for z, s in ((-0.05, 60.0), (-0.31, 30.0)):
        ax.scatter([X_BC + 14.0], [0.0], [z], s=s, facecolor='white', edgecolor=C_BC, linewidth=1.0, depthshade=False,
                   zorder=Z_ON_FRONT + 2)
        ax.scatter([X_BC + 14.0], [0.0], [z], s=s / 9.0, color=C_BC, depthshade=False, zorder=Z_ON_FRONT + 3)
    # the interior flow above the layer and the near-bottom flow inside it
    xn, zn = in_layer(X_COLUMN, 0.5, key)
    flow_symbol(ax, X_COLUMN + 21.0, -0.305, ph['toward_reader'], 52.0, C_INTERIOR, 'white', Z_ON_FRONT + 4)
    flow_symbol(ax, float(xn), float(zn), ph['toward_reader'], 52.0 * GAIN[key] ** 2, C_NEAR, 'black',
                Z_ON_FRONT + 4, lw=0.7)
    # the transport of the layer across the isobaths, at the shelf break and on the shelf
    if key == 'up':
        along_bottom(ax, 170.0, 139.0, 0.5, C_TRANSPORT, Z_ON_FRONT + 2, lw=2.0, size=9.5, key=key)
        along_bottom(ax, 100.0, 44.0, 0.022, C_TRANSPORT, Z_ON_FRONT + 2, lw=2.0, size=9.5, key=None)
    else:
        along_bottom(ax, 139.0, 168.0, 0.5, C_TRANSPORT, Z_ON_FRONT + 2, lw=2.0, size=9.5, key=key)
        along_bottom(ax, 52.0, 106.0, 0.025, C_TRANSPORT, Z_ON_FRONT + 2, lw=1.7, size=8.5, key=None)
    ax.text(MARKERS[2][0] - 13.0, 0.0, MARKERS[2][2], ph['layer'], ha='right', va='center', fontsize=FS['small'],
            color=C_TEXT, fontweight='bold', linespacing=0.95, zorder=Z_TEXT)
    if labels:
        ax.text(252.0, 0.0, -0.62, 'SACW', ha='center', va='center', fontsize=FS['label'], color=C_TEXT,
                fontweight='bold', zorder=Z_TEXT)


def draw_block(fig, cell, key, labels):
    ax = fig.add_subplot(cell, projection='3d', computed_zorder=False)
    ax.set_proj_type('ortho')
    ax.view_init(**VIEW)
    ax.set_xlim(X_LAND, X_OFF)
    ax.set_ylim(Y_SW, Y_NE)
    ax.set_zlim(Z_FLOOR, Z_TOP)
    ax.set_box_aspect(BOX_ASPECT, zoom=ZOOM)
    ax.set_axis_off()
    draw_faces(ax, key)
    draw_surface(ax, labels)
    draw_section(ax, key, labels)
    for k, (x, y, z) in MARKERS.items():
        marker(ax, k, x, y, z)
    for artist in ax.get_children():        # the block may reach beyond the square frame of the 3d axes
        artist.set_clip_on(False)
    return ax


# =====================================================================
# MECHANISM PANELS (2d, equal aspect)
# =====================================================================

def vec(ax, p0, p1, colour, lw=1.8, zorder=5, ms=10.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, color=colour, lw=lw, shrinkA=0,
                                 shrinkB=0, zorder=zorder))


def fat(ax, p0, p1, fill, edge, width=1.0, scale=9.0, zorder=5):
    """A filled arrow with an edge, which keeps a white arrow readable."""
    style = f'simple,head_length={0.95 * width:.2f},head_width={1.5 * width:.2f},tail_width={0.6 * width:.2f}'
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=scale, facecolor=fill, edgecolor=edge,
                                 lw=0.7, shrinkA=0, shrinkB=0, zorder=zorder))


def mech_title(ax, k, title, subtitle):
    """The numbered title and a short statement above a mechanism panel."""
    nl = subtitle.count('\n') + 1
    y_sub = 3.0
    y_tit = y_sub + nl * FS['small'] * 1.22 + 4.0
    kw = dict(xy=(0.0, 1.0), xycoords='axes fraction', textcoords='offset points', annotation_clip=False)
    ax.annotate(subtitle, xytext=(0.0, y_sub), ha='left', va='bottom', fontsize=FS['small'], color=C_TEXT,
                linespacing=1.05, **kw)
    ax.annotate(str(k), xytext=(5.0, y_tit + 0.45 * FS['label']), ha='center', va='center', fontsize=FS['small'],
                fontweight='bold', color=C_TEXT, bbox=dict(boxstyle='circle,pad=0.22', fc='white', ec=C_TEXT, lw=0.8),
                **kw)
    ax.annotate(title, xytext=(14.0, y_tit), ha='left', va='bottom', fontsize=FS['label'], fontweight='bold',
                color=C_TEXT, **kw)


def mech_frame(ax, xlim, zlim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect('equal')
    ax.set_anchor('N')                  # the three panels share their top edge
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('lightgrey')
        sp.set_linewidth(0.6)


def mech_uplift(ax):
    """A section along the coast over the shelf. The wave forced by the winds farther
    south travels to the northeast, the downwelling phase ahead of the upwelling
    phase, and the isotherms return to their depth once it has passed."""
    x = np.linspace(0.0, 1.8, 300)
    z_rest, amp = -0.44, 0.15
    z20 = z_rest + amp * np.cos(2 * np.pi * (x - 0.50) / 1.50)
    z_bed = -0.90
    ax.fill_between(x, 0.0, z20, color=C_FRONT['warm'], lw=0, zorder=0)
    ax.fill_between(x, z20, z_bed, color=C_FRONT['sacw'], lw=0, zorder=0)
    ax.fill_between(x, z_bed, -1.05, color=C_EARTH, lw=0, zorder=0)
    ax.plot([0.0, 1.8], [0.0, 0.0], color='dimgrey', lw=0.6, zorder=2)
    ax.plot([0.0, 1.8], [z_bed, z_bed], color='dimgrey', lw=0.4, zorder=2)
    ax.plot(x, np.full_like(x, z_rest), color=C_REST, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.plot(x, z20, color=C_T20, lw=2.0, zorder=3)
    # the displacement in each phase
    fat(ax, (0.50, z_rest + 0.01), (0.50, z_rest + amp + 0.045), C_WAVE, 'white', width=0.8)
    fat(ax, (1.25, z_rest - 0.01), (1.25, z_rest - amp - 0.045), C_WAVE, 'white', width=0.8)
    ax.text(0.50, -0.135, 'upwelling\nphase', ha='center', va='center', fontsize=FS['small'],
            color=PHASE_COLOUR['up'], fontweight='bold', linespacing=0.95, zorder=6)
    ax.text(1.25, -0.20, 'downwelling\nphase', ha='center', va='center', fontsize=FS['small'],
            color=PHASE_COLOUR['down'], fontweight='bold', linespacing=0.95, zorder=6)
    # the wave travels to the northeast
    fat(ax, (0.92, -0.775), (1.70, -0.775), C_WAVE, 'white', width=1.05)
    ax.text(0.90, -0.775, 'to the northeast', ha='right', va='center', fontsize=FS['small'], color=C_TEXT, zorder=6)
    # the wind that forces it
    for xw in (0.07, 0.25):
        vec(ax, (xw, 0.09), (xw + 0.15, 0.09), C_WIND, lw=1.2, ms=7.0)
    ax.text(0.46, 0.09, 'wind farther south', ha='left', va='center', fontsize=FS['small'], color=C_NOTE, zorder=6)
    ax.text(0.03, -0.975, 'SW', ha='left', va='center', fontsize=FS['small'], color=C_TEXT, zorder=6)
    ax.text(1.77, -0.975, 'NE', ha='right', va='center', fontsize=FS['small'], color=C_TEXT, zorder=6)
    ax.text(0.90, -0.975, 'shelf', ha='center', va='center', fontsize=FS['small'], color=C_TEXT, zorder=6)
    mech_frame(ax, (0.0, 1.8), (-1.05, 0.18))
    mech_title(ax, 1, 'Uplift by the wave', 'winds farther south force a wave that\nmoves the isotherms across the shelf')


def slope_sketch(ax, x0, x1, z0, z1, key):
    """The bottom boundary layer on the slope in one state, the coast to the left."""
    w, h = x1 - x0, z1 - z0
    hl = 0.20 if key == 'up' else 0.30                  # thickness of the layer, fraction of the box
    u = np.linspace(0.0, 1.0, 200)
    vb = 0.50 - 0.42 * u                                # the bottom
    X = lambda uu: x0 + w * np.asarray(uu)
    Zc = lambda vv: z0 + h * np.asarray(vv)
    ax.fill_between(X(u), Zc(vb), z1, color=C_FRONT['sacw'], lw=0, zorder=0)
    ax.fill_between(X(u), z0, Zc(vb), color=C_EARTH, lw=0, zorder=0)
    ax.fill_between(X(u), Zc(vb), Zc(np.minimum(vb + hl, 1.0)), color=C_STRAT if key == 'up' else C_MIXED, lw=0,
                    alpha=0.9, zorder=1)
    ax.plot(X(u), Zc(vb + hl), color='black', lw=0.4, zorder=2)
    ax.plot(X(u), Zc(vb), color='dimgrey', lw=0.5, zorder=2)
    du = 0.30                                           # reach of the bend up the slope
    for v0 in (0.86, 0.70, 0.54, 0.38):
        ue = (0.50 + hl - v0) / 0.42                    # where the isotherm enters the layer
        if ue <= 0.0:
            ax.plot(X([0.0, 1.0]), Zc([v0, v0]), color=C_ISO, lw=0.7, zorder=2)
            continue
        uu, vv = [1.0, ue], [v0, v0]
        if key == 'down':
            uu, vv = uu + [ue], vv + [0.50 - 0.42 * ue]
        else:
            s = np.linspace(0.0, 1.0, 30)
            ub = ue - du * s
            p = max(1.3, 0.42 * du / hl)
            uu, vv = uu + list(ub), vv + list(0.50 - 0.42 * ub + hl * (1.0 - s) ** p)
        uu, vv = np.array(uu), np.array(vv)
        keep = uu >= 0.0
        ax.plot(X(uu[keep]), Zc(vv[keep]), color=C_ISO, lw=0.7, zorder=2)
    # the transport of the layer across the isobaths
    ua, ub = (0.80, 0.30) if key == 'up' else (0.26, 0.76)
    lift = 0.5 * hl
    vec(ax, (float(X(ua)), float(Zc(0.50 - 0.42 * ua + lift))), (float(X(ub)), float(Zc(0.50 - 0.42 * ub + lift))),
        C_TRANSPORT, lw=2.2, ms=10.0, zorder=4)
    ax.plot([x0, x1, x1, x0, x0], [z0, z0, z1, z1, z0], color='lightgrey', lw=0.6, zorder=5)


def mech_arrest(ax):
    """The buoyancy arrest. Under upslope transport the dense water lifted along the
    bottom keeps the layer stratified and its thermal wind weakens and aligns the
    near-bottom flow. Under downslope transport the layer mixes and the near-bottom
    flow veers as an Ekman layer. The two vectors of each state are drawn with the
    attenuation and the veering measured at FUB3."""
    rows = (('up', -0.035, -0.585, 'upslope', 'stratified layer', 'weak and aligned'),
            ('down', -0.645, -1.195, 'downslope', 'mixed layer', 'Ekman veering'))
    cx, L, xt = 0.82, 0.38, 1.34
    for key, zt, zb, name, state, what in rows:
        slope_sketch(ax, 0.035, 0.71, zb, zt, key)
        cz = 0.5 * (zt + zb) + 0.02
        a = np.radians(VEER[key])
        fat(ax, (cx, cz), (cx + L, cz), C_INTERIOR, 'black', width=0.95, zorder=5)
        fat(ax, (cx, cz), (cx + GAIN[key] * L * np.cos(a), cz + GAIN[key] * L * np.sin(a)), C_NEAR, 'black',
            width=0.5, zorder=6)
        ax.text(xt, cz + 0.135, name, ha='left', va='center', fontsize=FS['label'], fontweight='bold',
                color=PHASE_COLOUR[key], zorder=6)
        ax.text(xt, cz, state, ha='left', va='center', fontsize=FS['small'], color=C_TEXT, zorder=6)
        ax.text(xt, cz - 0.125, what, ha='left', va='center', fontsize=FS['small'], color=C_TEXT,
                fontweight='bold', zorder=6)
    ax.plot([0.77, 2.40], [-0.615, -0.615], color='lightgrey', lw=0.6, zorder=2)
    ax.text(cx - 0.02, -0.035 - 0.105, 'interior', ha='left', va='center', fontsize=FS['small'], color=C_TEXT,
            zorder=6)
    ax.text(cx - 0.02, -0.585 + 0.085, 'near-bottom', ha='left', va='center', fontsize=FS['small'], color=C_TEXT,
            zorder=6)
    mech_frame(ax, (0.0, 2.43), (-1.23, 0.0))
    mech_title(ax, 2, 'Buoyancy arrest', 'under upslope transport the lifted dense water\ncaps the layer and slows the near-bottom flow')


def mech_supply(ax):
    """The net supply. The uplift alone returns the isotherms to their depth. The
    near-bottom flow turns offshore in the downwelling phase and onshore in the
    upwelling phase, when the water is coldest, so the correlated fluctuations
    leave SACW on the shelf."""
    t0, t1 = 0.06, 1.74
    tm = 0.5 * (t0 + t1)
    t = np.linspace(t0, t1, 300)
    wave = np.sin(2 * np.pi * (t - t0) / (t1 - t0))
    ax.fill_between([t0, tm], -0.40, 0.40, color=C_MIXED, alpha=0.22, lw=0, zorder=0)
    ax.fill_between([tm, t1], -0.40, 0.40, color=C_STRAT, alpha=0.16, lw=0, zorder=0)
    ax.plot([t0, t1], [0.0, 0.0], color='dimgrey', lw=0.6, zorder=1)
    ax.plot(t, 0.27 * wave, color=C_T20, lw=2.0, zorder=3)
    ax.plot(t, -0.27 * wave, color=C_TRANSPORT, lw=2.0, ls=(0, (4, 1.6)), zorder=3)
    q = 0.25 * (t1 - t0)
    ax.text(t0 + q, 0.335, 'warmer', ha='center', va='center', fontsize=FS['small'], color=C_T20, zorder=6)
    ax.text(t0 + q, -0.335, 'offshore', ha='center', va='center', fontsize=FS['small'], color=C_TRANSPORT, zorder=6)
    ax.text(tm + q, -0.335, 'coldest', ha='center', va='center', fontsize=FS['small'], color=C_T20,
            fontweight='bold', zorder=6)
    ax.text(tm + q, 0.335, 'onshore', ha='center', va='center', fontsize=FS['small'], color=C_TRANSPORT,
            fontweight='bold', zorder=6)
    ax.text(t0 + q, 0.50, 'downwelling\nphase', ha='center', va='center', fontsize=FS['small'], linespacing=0.95,
            color=PHASE_COLOUR['down'], fontweight='bold', zorder=6)
    ax.text(tm + q, 0.50, 'upwelling\nphase', ha='center', va='center', fontsize=FS['small'], linespacing=0.95,
            color=PHASE_COLOUR['up'], fontweight='bold', zorder=6)
    vec(ax, (0.62, -0.51), (1.18, -0.51), C_NOTE, lw=1.0, ms=7.0)
    ax.text(0.58, -0.51, 'time', ha='right', va='center', fontsize=FS['small'], color=C_NOTE, zorder=6)
    mech_frame(ax, (0.0, 1.8), (-0.615, 0.615))
    mech_title(ax, 3, 'Net supply of SACW', 'the near-bottom flow turns onshore\nwhen the water is coldest')


# =====================================================================
# FIGURE
# =====================================================================

fig = plt.figure(figsize=FIGSIZE)
blocks = GridSpec(1, 2, figure=fig, left=-0.012, right=0.988, top=BLOCK_TOP, bottom=BLOCK_BOTTOM, wspace=0.0)
for i, key in enumerate(('down', 'up')):
    draw_block(fig, blocks[0, i], key, labels=(i == 0))
    fig.text(0.25 + 0.5 * i, 0.992, PHASES[key]['title'], ha='center', va='top', fontsize=FS['title'],
             fontweight='bold', color=PHASE_COLOUR[key])

# the reversal, between the two blocks
yr = BLOCK_BOTTOM + REVERSAL_Y * (BLOCK_TOP - BLOCK_BOTTOM)
fig.add_artist(FancyArrowPatch((0.458, yr), (0.542, yr), transform=fig.transFigure, arrowstyle='-|>',
                               mutation_scale=12, color=C_WAVE, lw=1.4))
fig.text(0.5, yr + 0.010, 'the wave reverses', ha='center', va='bottom', fontsize=FS['small'], color=C_TEXT)
fig.text(0.5, yr - 0.010, 'two to three\ndays later', ha='center', va='top', fontsize=FS['small'], color=C_NOTE,
         linespacing=1.0)

# the mechanisms, numbered as the markers in the blocks
mech = GridSpec(1, 3, figure=fig, left=0.03, right=0.975, top=MECH_TOP, bottom=MECH_BOTTOM, wspace=0.12,
                width_ratios=MECH_WIDTHS)
mech_uplift(fig.add_subplot(mech[0, 0]))
mech_arrest(fig.add_subplot(mech[0, 1]))
mech_supply(fig.add_subplot(mech[0, 2]))

# one legend below the panels, outside the drawings
hs = [Line2D([], [], color=C_WAVE, lw=3.4, label='coastal trapped wave and its uplift'),
      Line2D([], [], color=C_TRANSPORT, lw=2.6, label='transport of the bottom boundary layer'),
      Line2D([], [], color=C_BC, lw=2.2, marker='o', ms=6, markerfacecolor='white', label='Brazil Current'),
      Line2D([], [], color=C_T20, lw=2.0, label='20 °C isotherm'),
      Line2D([], [], color=C_REST, lw=0.9, ls=(0, (3, 2)), label='20 °C isotherm before the wave'),
      Line2D([], [], color=C_ISO, lw=0.8, label='isotherms of the SACW'),
      Line2D([], [], ls='none', marker='o', ms=7, markerfacecolor=C_INTERIOR, markeredgecolor='black',
             label='interior flow'),
      Line2D([], [], ls='none', marker='o', ms=6, markerfacecolor=C_NEAR, markeredgecolor='black',
             label='near-bottom flow'),
      Line2D([], [], ls='none', marker=r'$\odot$', ms=7.5, color='black', markeredgewidth=0.3,
             label='toward the southwest'),
      Line2D([], [], ls='none', marker=r'$\otimes$', ms=7.5, color='black', markeredgewidth=0.3,
             label='toward the northeast'),
      Patch(facecolor=C_STRAT, alpha=0.9, edgecolor='black', lw=0.4, label='stratified layer'),
      Patch(facecolor=C_MIXED, alpha=0.9, edgecolor='black', lw=0.4, label='mixed layer')]
fig.legend(handles=hs, loc='lower center', bbox_to_anchor=(0.5, 0.004), ncol=4, frameon=False,
           fontsize=FS['small'], handlelength=1.9, columnspacing=1.0, handletextpad=0.5, labelspacing=0.35)

fig.savefig(V.DIR_FIG / 'fig08_synthesis.pdf', dpi=400)
S.save(fig, V.DIR_FIG / 'fig08_synthesis.png')
