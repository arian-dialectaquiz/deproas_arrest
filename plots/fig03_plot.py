"""
Figure 3. Response of the near-bottom flow to the interior flow at the edge of the
Brazil Current.

(a) and (b) mean along-isobath velocity (positive toward the northeast) and mean
    cross-isobath velocity (positive onshore) at the 200 m moorings. The shading is
    the 95 % interval from the effective number of independent samples, widened in
    (b) to the range of the mean when the frame is rotated by the spread of the
    isobath bearing.
(c) and (d) complex gain of the lower on the upper velocity of each pair at FUB3 and
    FCF3, drawn as a polar diagram with the attenuation as the distance from the
    origin and the veering as the angle (clockwise negative), under upslope (filled)
    and downslope (open) Ekman transport, for the frictional pair between 155 and
    194 m (circle, site colour) and the control pair between 85 and 155 m (triangle,
    grey). The translucent sector of each gain spans its 95 % interval in attenuation
    and veering from resampling whole states. The khaki sector is an unarrested Ekman
    layer over the roughness, the heights and the thickness of the layer, the black
    segment the gain of a fully arrested layer at the lowest meter under upslope
    transport, and the dashed arc a gain of one. FBG3 holds too few downslope hours
    at the frictional pair for two gains and is left out.
(e) and (f) speed at 194 m against the speed at 155 m at FUB3 and FCF3, the median
    (line) and the interquartile range (shading) in bins of the interior speed for
    each kind of state, with the unarrested Ekman layer (khaki), the speed of a fully
    arrested layer at the design and GEBCO heights of the meter (dotted) and equal
    speeds (dashed).

Reads paper_v3/tables/fig03_*.csv. Writes paper_v3/figures/fig03_response.png and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerTuple

import config_v3 as V
import style as S

FIGSIZE = (7.48, 8.4)
DEPTH_LIM = (212.0, 0.0)
SPEED_LIM = (0.0, 0.45)
GAIN_SITES = ('FUB3', 'FCF3')
SPEED_SITES = ('FUB3', 'FCF3')
PAIR_MARK = {'interior': '^', 'bottom': 'o'}
THETA = (-75.0, 30.0)               # veerings spanned by the gain diagrams
GAIN_YLIM = (-0.90, 0.62)


def read(name, **kw):
    p = V.DIR_TAB / f'{name}.csv'
    if p.exists() and p.stat().st_size > 5:
        return pd.read_csv(p, **kw)
    print(f'  {p.name} is missing, its elements are left out')
    return pd.DataFrame()


prof = read('fig03_profiles')
resp = read('fig03_response')
bins = read('fig03_bins')
ref = read('fig03_reference')
here = [s for s in V.SITES_200 if len(prof) and s in set(prof.site)]
pair_name = {f'{a} to {b} m': k for k, (a, b) in V.PAIRS.items()}
if len(resp):
    resp['kind'] = resp.pair.map(pair_name)
    resp = resp[resp.half == 'all']

# ----------------------------------------------------------------- layout
fig = plt.figure(figsize=FIGSIZE)
gs_top = gridspec.GridSpec(1, 4, figure=fig, left=0.085, right=0.995, top=0.925, bottom=0.615,
                           width_ratios=[0.56, 0.56, 1.0, 1.0], wspace=0.18)
gs_bot = gridspec.GridSpec(1, 2, figure=fig, left=0.085, right=0.985, top=0.525, bottom=0.17, wspace=0.18)
ax_u = fig.add_subplot(gs_top[0, 0])
ax_v = fig.add_subplot(gs_top[0, 1], sharey=ax_u)
ax_gain = [fig.add_subplot(gs_top[0, 2]), fig.add_subplot(gs_top[0, 3])]
ax_sp = [fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1])]

# ------------------------------------------------------ (a) and (b) profiles
for site in here:
    g = prof[prof.site == site].sort_values('level')
    g = g[np.isfinite(g.u)]
    col = S.SITE_COLOURS[site]
    for ax, key in ((ax_u, 'u'), (ax_v, 'v')):
        lo, hi = g[f'{key}_lo'].values, g[f'{key}_hi'].values
        if key == 'v' and 'v_frame_lo' in g:
            lo, hi = np.fmin(lo, g.v_frame_lo.values), np.fmax(hi, g.v_frame_hi.values)
        ax.fill_betweenx(g.level, lo, hi, color=col, alpha=S.BAND_ALPHA, lw=0, zorder=2)
        ax.plot(g[key], g.level, color=col, ls=S.SITE_LINESTYLES[site], marker=S.SITE_MARKERS[site], ms=5.4,
                mec='black', mew=0.5, lw=1.4, zorder=4)
for ax, lab, letter in ((ax_u, 'along-isobath\nvelocity (m s$^{-1}$)', 'a'),
                        (ax_v, 'cross-isobath\nvelocity (m s$^{-1}$)', 'b')):
    ax.axvline(0.0, color=S.INK, lw=0.7, zorder=1)
    ax.set_ylim(*DEPTH_LIM)
    S.tidy(ax, axis='both')
    ax.set_xlabel(lab)
    S.panel(ax, letter)
ax_u.set_ylabel('depth (m)')
plt.setp(ax_v.get_yticklabels(), visible=False)
ax_u.legend([S.line_handle(S.SITE_COLOURS[s], S.SITE_LINESTYLES[s], marker=S.SITE_MARKERS[s], ms=5.4)
             for s in here], here, ncol=3, loc='lower left', bbox_to_anchor=(-0.05, 1.06),
            columnspacing=1.0, handlelength=1.9, handletextpad=0.4)

# ------------------------------------------- (c) and (d) the complex gain in polar form
for k, (ax, site) in enumerate(zip(ax_gain, GAIN_SITES)):
    S.gain_plane(ax, theta=THETA, rmax=1.12, ylim=GAIN_YLIM)
    col = S.SITE_COLOURS[site]
    un = ref[(ref.site == site) & (ref.kind == 'unarrested')] if len(ref) else pd.DataFrame()
    if len(un):
        S.gain_sector(ax, un.gain.min(), un.gain.max(), un.veer.min(), un.veer.max(), color=S.UNARRESTED, lw=0,
                      zorder=1)
    g = resp[resp.site == site] if len(resp) else pd.DataFrame()
    ar = ref[(ref.site == site) & (ref.kind == 'arrested')] if len(ref) else pd.DataFrame()
    up_b = g[(g.kind == 'bottom') & (g.state == S.UP)] if len(g) else pd.DataFrame()
    if len(ar) and len(up_b) and up_b.speed_median.iloc[0] > 0:
        caps = np.clip(ar.cap.values / up_b.speed_median.iloc[0], 0.0, 1.0)
        ax.plot([caps.min(), caps.max()], [0.0, 0.0], color=S.ARRESTED, lw=4.0, solid_capstyle='butt', zorder=5)
    for _, r in (g.sort_values('kind', ascending=False).iterrows() if len(g) else []):
        filled = S.STATE_FILLED[r.state]
        c_ = col if r.kind == 'bottom' else S.CONTROL
        S.gain_sector(ax, r.gain_lo, r.gain_hi, r.veer_lo, r.veer_hi, color=c_, lw=0, zorder=2,
                      alpha=S.BAND_ALPHA if r.kind == 'bottom' else S.BAND_ALPHA_LIGHT)
        x, y = S.gain_xy(r.gain, r.veer)
        ax.plot(x, y, marker=PAIR_MARK[r.kind], ms=8.0 if r.kind == 'bottom' else 6.4,
                mfc=c_ if filled else 'white', mec='black' if filled else c_, mew=0.6 if filled else 1.6,
                ls='none', zorder=6 if r.kind == 'bottom' else 4)
    ax.text(0.5, 1.0, site, transform=ax.transAxes, ha='center', va='bottom', fontweight='bold')
    xl, yl = S.gain_xy(0.62, 30.0)
    ax.text(xl - 0.13, yl + 0.13, 'attenuation', ha='center', va='bottom', rotation=30.0, rotation_mode='anchor',
            fontsize=S.FS)
    xl, yl = S.gain_xy(1.25, -15.0)
    ax.text(xl, yl, 'veering', ha='center', va='center', rotation=-15.0 + 90.0, fontsize=S.FS)
    S.panel(ax, 'cd'[k], dx=0.04, dy=1.0)

# ------------------------------------------------- (e) and (f) the speed relation
xx = np.linspace(*SPEED_LIM, 101)
for k, (ax, site) in enumerate(zip(ax_sp, SPEED_SITES)):
    col = S.SITE_COLOURS[site]
    rr = ref[(ref.site == site) & (ref.kind == 'unarrested')] if len(ref) else pd.DataFrame()
    if len(rr):
        ax.fill_between(xx, rr.gain.min() * xx, rr.gain.max() * xx, color=S.UNARRESTED, lw=0, zorder=1)
    ar = ref[(ref.site == site) & (ref.kind == 'arrested')] if len(ref) else pd.DataFrame()
    for cap in (ar.cap if len(ar) else []):
        ax.plot(xx, np.minimum(xx, cap), color=S.ARRESTED, ls=':', lw=1.6, zorder=3)
    ax.plot(xx, xx, color=S.GUIDE, ls='--', lw=0.9, zorder=2)
    for state in (S.UP, S.DOWN):
        g = bins[(bins.site == site) & (bins.state == state)].sort_values('centre') if len(bins) else pd.DataFrame()
        if g.empty:
            continue
        filled = S.STATE_FILLED[state]
        ax.fill_between(g.centre, g.q1, g.q3, color=col, alpha=0.32 if filled else 0.13, lw=0, zorder=4)
        ax.plot(g.centre, g['median'], color=col, ls=S.STATE_LS[state], lw=1.8, marker=S.SITE_MARKERS[site],
                ms=5.4, mfc=col if filled else 'white', mec='black' if filled else col, mew=0.5 if filled else 1.3,
                zorder=5)
    ax.set_xlim(*SPEED_LIM)
    ax.set_ylim(*SPEED_LIM)
    ax.set_aspect('equal', adjustable='box')
    S.tidy(ax, axis='both')
    ax.set_xlabel(f'speed at {V.PAIRS["bottom"][0]} m (m s$^{{-1}}$)')
    ax.set_ylabel(f'speed at {V.PAIRS["bottom"][1]} m (m s$^{{-1}}$)' if k == 0 else '')
    ax.set_title(site, fontweight='bold')
    S.panel(ax, 'ef'[k])

# ------------------------------------------------------------------ legend
grey = S.LIGHT_GUIDE
band_up = S.patch_handle(grey, alpha=0.45)
band_dn = S.patch_handle(grey, alpha=0.18)
h = [S.marker_handle(grey, 'o', ms=7.0), S.marker_handle(grey, 'o', ms=7.0, open_=True),
     S.marker_handle(S.CONTROL, '^', ms=6.0), S.marker_handle(S.CONTROL, '^', ms=6.0, open_=True),
     S.patch_handle(grey, alpha=0.45), S.patch_handle(S.UNARRESTED),
     plt.Line2D([], [], color=S.ARRESTED, lw=3.5), S.line_handle(S.GUIDE, '--', lw=0.9),
     (band_up, S.line_handle(grey, S.STATE_LS[S.UP], marker='o', ms=5.4)),
     (band_dn, S.line_handle(grey, S.STATE_LS[S.DOWN], marker='o', ms=5.4, mfc='white')),
     S.line_handle(S.ARRESTED, ':', lw=1.6), S.line_handle(S.GUIDE, '--', lw=0.9)]
lab = ['upslope, 155 to 194 m', 'downslope, 155 to 194 m', 'upslope, 85 to 155 m', 'downslope, 85 to 155 m',
       '95 % interval', 'unarrested Ekman layer', 'arrested layer (c, d)', 'gain of one (c, d)',
       'upslope, median and quartiles', 'downslope, median and quartiles', 'arrested layer (e, f)',
       'equal speeds (e, f)']
fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, columnspacing=2.4, handlelength=2.4,
           handletextpad=0.6, handler_map={tuple: HandlerTuple(ndivide=None)},
           title='Ekman transport of the interior flow at 155 m', title_fontsize=S.FS_LEG)

fig.savefig(V.DIR_FIG / 'fig03_response.pdf')
S.save(fig, V.DIR_FIG / 'fig03_response.png')
