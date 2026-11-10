"""
Figure 7. Synthesis of the shelf-break bottom boundary layer and the supply of central
water measured by the moorings.

(a) With the flow toward the southwest (out of the page, dot) the bottom Ekman transport
    is upslope, the isotherms near the bottom are lifted where they meet the slope, the
    layer is capped by a stronger stratification (shaded) and the near-bottom flow is
    attenuated with little veering.
(b) With the flow toward the northeast (into the page, cross) the transport is downslope,
    the isotherms near the bottom are pushed down over a mixed layer (hatched) and the
    near-bottom flow keeps more of the interior speed and veers as an Ekman layer.
(c) The along-shore wind stress toward the southwest (dot at the surface) forces a coastal
    trapped wave that lifts the isotherms of the whole section (dashed before, solid after)
    within a day at the 50, 100 and 200 m moorings, while the correlated
    fluctuations of the layer carry central water onshore at the shelf break (arrow in the
    layer).
The size of the flow symbols in (a) and (b) scales with the measured speed at FUB3, and
every number quoted is read from the tables of Figures 2 to 6.

Reads paper_v3/tables/fig07_numbers.csv. Writes paper_v3/figures/fig07_synthesis.png and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerTuple
from scipy.interpolate import PchipInterpolator

import config_v3 as V
import style as S

FIGSIZE = (7.48, 7.4)
TITLE = S.FS

# idealised section of a line of the array, 50, 100 and 200 m moorings at 0, 46 and 92 km
X_NODES = [-6, 0, 20, 46, 70, 85, 92, 100, 108, 118, 140]
D_NODES = [44, 50, 70, 100, 125, 165, 200, 280, 420, 560, 700]
DEPTH = PchipInterpolator(X_NODES, D_NODES)
X_MOOR = {'50 m': 0.0, '100 m': 46.0, '200 m': 92.0}
ISO_T = (22, 20, 18, 16, 14, 12)
ISO_Z200 = (35, 70, 105, 145, 190, 240)     # depth of each isotherm at the 200 m mooring
ISO_TILT = 0.35                              # m km-1, deeper offshore beneath the current
H_LAYER = 28.0
BEND = 22.0
BEND_KM = 9.0                                # along-slope scale of the displacement
BREAK = (150.0, 280.0)                       # depths of the slope where the layer is drawn
UPLIFT = 25.0

num = pd.read_csv(V.DIR_TAB / 'fig07_numbers.csv').set_index('key')


def n(key, default=np.nan):
    return float(num.loc[key, 'value']) if key in num.index else default


def isotherms(x, sign=0, shift=0.0):
    """Depth of each isotherm along x, lifted by shift and displaced smoothly around the point
    where it meets the slope, upward for sign -1 (upslope transport of colder water) and
    downward for sign +1 (downslope transport of warmer water), so that it meets the bottom
    upslope or downslope of its undisturbed intersection."""
    D = DEPTH(x)
    out = []
    for z0 in ISO_Z200:
        z = z0 + ISO_TILT * (x - 92.0) - shift
        cross = np.flatnonzero(np.diff(np.sign(D - z)) != 0)
        if sign and cross.size:
            xi = x[cross[-1]]
            z = z + sign * BEND * np.exp(-((x - xi) / BEND_KM) ** 2)
        out.append(np.where(z < D, z, np.nan))
    return out


def section(ax, x0, x1, zmax):
    x = np.linspace(x0, x1, 400)
    ax.fill_between(x, DEPTH(x), zmax + 50, color=S.BOTTOM_FILL, ec=S.BOTTOM_EDGE, lw=0.8, zorder=5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(zmax, -12)
    ax.axhline(0.0, color=S.GUIDE, lw=0.6, zorder=1)
    ax.set_xlabel('distance from the 50 m mooring (km)')
    ax.set_ylabel('depth (m)')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return x


def mooring(ax, x, levels):
    ax.plot([x, x], [0, float(DEPTH(x))], color=S.GUIDE, lw=0.7, ls=':', zorder=3)
    ax.plot([x] * len(levels), levels, ls='none', marker='s', ms=4.2, mfc='white', mec=S.INK, mew=0.7, zorder=7)


def flow(ax, x, z, towards_viewer, size):
    """A flow normal to the section, a dot for the page toward the viewer and a cross away."""
    ax.plot(x, z, marker='o', ms=size, mfc='white', mec=S.SYN_CURRENT, mew=1.3, ls='none', zorder=8)
    ax.plot(x, z, marker='.' if towards_viewer else 'x', ms=size * (0.55 if towards_viewer else 0.6),
            mec=S.SYN_CURRENT, mfc=S.SYN_CURRENT, mew=1.3, ls='none', zorder=9)


def layer_arrow(ax, xa, xb, colour=S.SYN_TRANSPORT, lw=2.4):
    xs = np.linspace(xa, xb, 30)
    zs = DEPTH(xs) - 0.45 * H_LAYER
    ax.plot(xs[:-2], zs[:-2], color=colour, lw=lw, solid_capstyle='butt', zorder=8)
    ax.annotate('', xy=(xs[-1], zs[-1]), xytext=(xs[-3], zs[-3]),
                arrowprops=dict(arrowstyle='-|>', color=colour, lw=lw, mutation_scale=12), zorder=8)


fig = plt.figure(figsize=FIGSIZE)
gs_top = gridspec.GridSpec(1, 2, figure=fig, left=0.085, right=0.985, top=0.915, bottom=0.615, wspace=0.24)
gs_bot = gridspec.GridSpec(1, 1, figure=fig, left=0.085, right=0.985, top=0.375, bottom=0.075)
ax_a = fig.add_subplot(gs_top[0, 0])
ax_b = fig.add_subplot(gs_top[0, 1])
ax_c = fig.add_subplot(gs_bot[0, 0])

# ---------------------------------------------- (a) and (b) the two states of the layer
for ax, sign, key in ((ax_a, -1, 'up'), (ax_b, +1, 'dn')):
    x = section(ax, 58.0, 128.0, 330.0)
    D = DEPTH(x)
    at_break = (D > BREAK[0]) & (D < BREAK[1])
    if key == 'up':
        ax.fill_between(x, D - H_LAYER, D, where=at_break, color=S.SYN_CAPPED, lw=0, zorder=2)
    else:
        ax.fill_between(x, D - 0.8 * H_LAYER, D, where=at_break, color=S.SYN_MIXED, ec=S.GUIDE,
                        hatch='////', lw=0, zorder=2)
    for T, z in zip(ISO_T, isotherms(x, sign)):
        if key == 'dn':
            z = np.where((z < D - 0.8 * H_LAYER) | ~at_break, z, np.nan)
        ax.plot(x, z, color=S.ISOTHERM, lw=0.8, zorder=4)
        k = np.flatnonzero(np.isfinite(z))
        if k.size:
            ax.text(x[k[-1]] - 0.5, z[k[-1]] - 3.0, f'{T} °C', fontsize=S.FS_SMALL, ha='right', va='bottom', zorder=6)
    mooring(ax, 92.0, [85, 155, 194])
    toward = key == 'up'                        # southwest, out of the page
    g = n(f'FUB3_gain_{key}', 0.6)
    flow(ax, 104.0, 150.0, toward, 13.0)
    flow(ax, 97.0, 188.0, toward, 13.0 * g)
    layer_arrow(ax, 95.5, 83.0) if key == 'up' else layer_arrow(ax, 83.0, 95.5)
    veer = n(f'FUB3_veer_{key}')
    head = 'Toward the southwest, upslope transport' if key == 'up' else 'Toward the northeast, downslope transport'
    body = f'attenuation {g:.2f} and veering {veer:+.0f}°'.replace('-', '\u2212')
    ax.set_title(head + '\n' + body, fontsize=TITLE, loc='left')


# ------------------------------------------------------ (c) the heave of the section
x = section(ax_c, -6.0, 128.0, 270.0)
for T, z0, z1 in zip(ISO_T, isotherms(x, 0, 0.0), isotherms(x, 0, UPLIFT)):
    ax_c.plot(x, z0, color=S.GUIDE, lw=0.8, ls='--', zorder=3)
    ax_c.plot(x, z1, color=S.ISOTHERM, lw=0.9, zorder=4)
lev = {'50 m': [10, 30, 44], '100 m': [23, 75, 94], '200 m': [23, 85, 155, 194]}
z18 = dict(zip(ISO_T, ISO_Z200))[18]
for name, xm in X_MOOR.items():
    mooring(ax_c, xm, lev[name])
    zb = min(z18 + ISO_TILT * (xm - 92.0), float(DEPTH(xm)) - 8.0)
    ax_c.annotate('', xy=(xm + 4.0, zb - UPLIFT), xytext=(xm + 4.0, zb),
                  arrowprops=dict(arrowstyle='-|>', color=S.SYN_WAVE, lw=1.8, mutation_scale=10), zorder=9)
    ax_c.text(xm, -5.0, name, fontsize=S.FS_SMALL, ha='center', va='bottom', zorder=9)
flow(ax_c, 22.0, -1.0, True, 8.0)               # along-shore stress toward the southwest
if n('FUB3_flux_eddy', 0.0) > 0:
    layer_arrow(ax_c, 95.5, 86.0, lw=1.6)
ax_c.set_ylim(270.0, -14.0)
lagtxt = f'lag of {n("lag_min_days"):.0f} to {n("lag_max_days"):.0f} day across {n("lag_distance_max_km"):.0f} km'
ax_c.set_title('Heave of the section by the coastal trapped wave\n' + lagtxt, fontsize=TITLE, loc='left')

# symbols of the sections, between the two rows and outside the data areas
ring = S.marker_handle('white', 'o', ms=8.0)
ring.set_markeredgecolor(S.SYN_CURRENT)
ring.set_markeredgewidth(1.3)
dot = S.marker_handle(S.SYN_CURRENT, '.', ms=5.0)
cross = S.marker_handle(S.SYN_CURRENT, 'x', ms=5.0)
cross.set_markeredgecolor(S.SYN_CURRENT)
cross.set_markeredgewidth(1.3)
dot.set_markeredgecolor(S.SYN_CURRENT)
h = [(ring, dot), (ring, cross), S.line_handle(S.SYN_TRANSPORT, lw=2.4), S.line_handle(S.SYN_WAVE, lw=1.8),
     S.patch_handle(S.SYN_CAPPED), S.patch_handle(S.SYN_MIXED, ec=S.GUIDE, hatch='////'),
     S.marker_handle(S.INK, 's', ms=4.6, open_=True), S.line_handle(S.GUIDE, '--', lw=0.8)]
lab = ['flow toward the southwest', 'flow toward the northeast', 'transport of the bottom layer',
       'uplift by the coastal trapped wave', 'stratified layer', 'mixed layer', 'current meter',
       'isotherm before the uplift']
fig.legend(h, lab, loc='center', bbox_to_anchor=(0.53, 0.475), ncol=3,
           handler_map={tuple: HandlerTuple(ndivide=1)}, columnspacing=1.3, frameon=False)

# panel letters beside the first line of each title
fig.canvas.draw()
inv = fig.transFigure.inverted()
for ax, letter in zip((ax_a, ax_b, ax_c), 'abc'):
    bb = inv.transform(ax.title.get_window_extent().get_points())
    fig.text(ax.get_position().x0 - 0.055, bb[1, 1], f'({letter})', fontsize=S.FS_PANEL, fontweight='bold', va='top')

fig.savefig(V.DIR_FIG / 'fig07_synthesis.pdf')
S.save(fig, V.DIR_FIG / 'fig07_synthesis.png')