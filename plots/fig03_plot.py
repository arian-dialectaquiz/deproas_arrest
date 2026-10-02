"""
Figure 3. Response of the near-bottom flow to the interior flow of the Brazil
Current edge.

(a) and (b) mean along-isobath velocity (positive toward the northeast against
    the Brazil Current) and mean cross-isobath velocity (positive onshore) at
    the 200 m moorings, with the 95 % interval from the effective number of
    independent samples, and in (b) the range of the mean when the frame is
    rotated by the spread of the isobath bearing (bar).
(c) and (d) attenuation and veering (clockwise negative) of the complex gain of
    the lower on the upper level of each pair, for the states in which the flow
    at 155 m runs with the current (filled, upslope Ekman transport) and against
    it (open, downslope), with the 95 % interval from resampling whole states,
    the band of an unarrested Ekman layer over the roughness, the heights and
    the thickness of the layer, and the arrested layer.
(e) and (f) speed at the lowest meter against the speed at 155 m at FUB3 and
    FCF3, the hours in logarithmic shading and the median and quartiles in bins
    of the interior speed for each kind of state, with the same references.

Reads paper_v3/tables/fig03_*.csv and paper_v3/data/fig03_hist.nc.
Writes paper_v3/figures/fig03_response.png and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

import config_v3 as V
import style as S

FIGSIZE = (7.48, 8.6)
LEG = S.LEGEND_SIZE
DEPTH_LIM = (212.0, 0.0)
SPEED_LIM = (0.0, 0.5)
OFFSET = {('interior', 'with the current'): -0.30, ('interior', 'against the current'): -0.10,
          ('bottom', 'with the current'): 0.10, ('bottom', 'against the current'): 0.30}
PAIR_MARK = {'interior': '^', 'bottom': 'o'}
SPEED_SITES = ('FUB3', 'FCF3')


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
f_h = V.DIR_DATA / 'fig03_hist.nc'
hist = xr.open_dataset(f_h) if f_h.exists() else None
here = [s for s in V.SITES_200 if len(prof) and s in set(prof.site)]
pair_name = {f'{a} to {b} m': k for k, (a, b) in V.PAIRS.items()}
if len(resp):
    resp['kind'] = resp.pair.map(pair_name)
    resp = resp[resp.half == 'all']

# ----------------------------------------------------------------- layout
fig = plt.figure(figsize=FIGSIZE)
gs_top = gridspec.GridSpec(1, 4, figure=fig, left=0.075, right=0.985, top=0.895, bottom=0.635,
                           width_ratios=[0.72, 0.72, 1.1, 1.1], wspace=0.42)
gs_bot = gridspec.GridSpec(2, 2, figure=fig, left=0.075, right=0.985, top=0.535, bottom=0.02,
                           height_ratios=[1.0, 0.10], hspace=0.14, wspace=0.16)
ax_u = fig.add_subplot(gs_top[0, 0])
ax_v = fig.add_subplot(gs_top[0, 1], sharey=ax_u)
ax_g = fig.add_subplot(gs_top[0, 2])
ax_p = fig.add_subplot(gs_top[0, 3])
ax_sp = [fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1])]
ax_leg = fig.add_subplot(gs_bot[1, :])
ax_leg.axis('off')

# ------------------------------------------------------ (a) and (b) profiles
for site in here:
    g = prof[prof.site == site].sort_values('level')
    g = g[np.isfinite(g.u)]
    for ax, key in ((ax_u, 'u'), (ax_v, 'v')):
        if key == 'v':
            for _, r in g.iterrows():
                ax.plot([r.v_frame_lo, r.v_frame_hi], [r.level, r.level], color=S.SITE_COLOURS[site],
                        lw=5.0, alpha=0.30, solid_capstyle='butt', zorder=2)
        ax.errorbar(g[key], g.level, xerr=np.clip([g[key] - g[f'{key}_lo'], g[f'{key}_hi'] - g[key]], 0, None),
                    color=S.SITE_COLOURS[site], ls=S.SITE_LINESTYLES[site],
                    marker=S.SITE_MARKERS[site], ms=4.4, mec='black', mew=0.45, lw=1.1,
                    elinewidth=0.8, capsize=1.6, zorder=4)
for ax, lab, letter in ((ax_u, 'along-isobath\nvelocity (m s$^{-1}$)', 'a'),
                        (ax_v, 'cross-isobath\nvelocity (m s$^{-1}$)', 'b')):
    ax.axvline(0.0, color=S.INK, lw=0.6, zorder=1)
    ax.set_ylim(*DEPTH_LIM)
    S.tidy(ax, axis='both')
    ax.set_xlabel(lab)
    S.panel(ax, letter)
ax_u.set_ylabel('depth (m)')
plt.setp(ax_v.get_yticklabels(), visible=False)
h = [S.line_handle(S.SITE_COLOURS[s], S.SITE_LINESTYLES[s], marker=S.SITE_MARKERS[s], ms=4.4)
     for s in here] + [S.patch_handle(S.GUIDE, alpha=0.30)]
ax_u.legend(h, here + ['range over the frame'], ncol=2, loc='lower left',
            bbox_to_anchor=(0.0, 1.04), fontsize=LEG, columnspacing=1.0, handlelength=2.0)

# ---------------------------------------------------- (c) and (d) the response
for ax, key in ((ax_g, 'gain'), (ax_p, 'veer')):
    for i, site in enumerate(here):
        rr = ref[(ref.site == site) & (ref.kind == 'unarrested')] if len(ref) else pd.DataFrame()
        if len(rr):
            lo, hi = rr[key].min(), rr[key].max()
            ax.add_patch(Rectangle((i + 0.0, lo), 0.42, hi - lo, fc=S.UNARRESTED, ec='none', zorder=1))
        g = resp[resp.site == site] if len(resp) else pd.DataFrame()
        for _, r in g.iterrows():
            x = i + OFFSET[(r.kind, r.state)]
            filled = r.state == 'with the current'
            ax.plot([x, x], [r[f'{key}_lo'], r[f'{key}_hi']], color=S.INK, lw=0.9, zorder=3)
            ax.plot(x, r[key], marker=PAIR_MARK[r.kind], ms=5.6, mec='black', mew=0.6, ls='none',
                    mfc=S.SITE_COLOURS[site] if filled else 'white', zorder=4)
            if key == 'gain' and r.kind == 'bottom' and len(ref):
                ar = ref[(ref.site == site) & (ref.kind == 'arrested')]
                if len(ar) and r.speed_median > 0:
                    for cap in ar.cap:
                        ax.plot(x, min(1.0, cap / r.speed_median), marker='_', ms=9, mew=1.4,
                                color=S.ARRESTED, ls='none', zorder=5)
    ax.set_xticks(range(len(here)))
    ax.set_xticklabels(here)
    ax.set_xlim(-0.55, len(here) - 0.45)
    S.tidy(ax, axis='y')
ax_g.axhline(1.0, color=S.GUIDE, lw=0.8, ls='--', zorder=0)
ax_g.set_ylim(0.0, max(1.15, ax_g.get_ylim()[1]))
ax_g.set_ylabel('attenuation')
ax_p.axhline(0.0, color=S.ARRESTED, lw=1.0, ls=':', zorder=0)
ax_p.set_ylabel('veering (°)')
S.panel(ax_g, 'c')
S.panel(ax_p, 'd')
h = [S.marker_handle(S.GUIDE, PAIR_MARK['interior'], ms=5.6),
     S.marker_handle(S.GUIDE, PAIR_MARK['interior'], ms=5.6, open_=True),
     S.marker_handle(S.GUIDE, PAIR_MARK['bottom'], ms=5.6),
     S.marker_handle(S.GUIDE, PAIR_MARK['bottom'], ms=5.6, open_=True),
     S.patch_handle(S.UNARRESTED),
     plt.Line2D([], [], color=S.ARRESTED, ls=':', lw=1.0, marker='_', ms=9, mew=1.4)]
ip, bp = V.PAIRS['interior'], V.PAIRS['bottom']
lab = [f'{ip[0]} to {ip[1]} m with', f'{ip[0]} to {ip[1]} m against',
       f'{bp[0]} to {bp[1]} m with', f'{bp[0]} to {bp[1]} m against',
       'unarrested layer', 'arrested layer']
x_mid = 0.5 * (ax_g.get_position().x0 + ax_p.get_position().x1)
fig.legend(h, lab, ncol=3, loc='lower center', bbox_to_anchor=(x_mid, ax_g.get_position().y1 + 0.012),
           bbox_transform=fig.transFigure, fontsize=LEG - 0.4, columnspacing=1.0, handlelength=1.3,
           handletextpad=0.4, title='pair and flow at 155 m with or against the current',
           title_fontsize=LEG - 0.4)

# ------------------------------------------------- (e) and (f) the speed relation
xx = np.linspace(*SPEED_LIM, 101)
for k, (ax, site) in enumerate(zip(ax_sp, SPEED_SITES)):
    if hist is not None and site in hist.site.values:
        c = hist['count'].sel(site=site).sum('state').values.T
        X, Y = np.meshgrid(hist.interior.values, hist.bottom.values)
        ax.pcolormesh(X, Y, np.where(c > 0, c, np.nan), cmap='Greys', norm=LogNorm(vmin=1),
                      shading='nearest', alpha=0.55, zorder=1, rasterized=True)
    rr = ref[(ref.site == site) & (ref.kind == 'unarrested')] if len(ref) else pd.DataFrame()
    if len(rr):
        ax.fill_between(xx, rr.gain.min() * xx, rr.gain.max() * xx, color=S.UNARRESTED, alpha=0.75,
                        lw=0, zorder=2)
    ar = ref[(ref.site == site) & (ref.kind == 'arrested')] if len(ref) else pd.DataFrame()
    for cap in ar.cap:
        ax.plot(xx, np.minimum(xx, cap), color=S.ARRESTED, ls=':', lw=1.2, zorder=3)
    ax.plot(xx, xx, color=S.GUIDE, ls='--', lw=0.8, zorder=2)
    for state, filled, ls in (('with the current', True, '-'), ('against the current', False, '--')):
        g = bins[(bins.site == site) & (bins.state == state)].sort_values('centre') if len(bins) \
            else pd.DataFrame()
        if g.empty:
            continue
        ax.errorbar(g.centre, g['median'], yerr=[g['median'] - g.q1, g.q3 - g['median']],
                    color=S.SITE_COLOURS[site], ls=ls, lw=1.1, marker=S.SITE_MARKERS[site], ms=4.8,
                    mfc=S.SITE_COLOURS[site] if filled else 'white', mec='black', mew=0.5,
                    elinewidth=0.8, capsize=1.6, zorder=5)
    ax.set_xlim(*SPEED_LIM)
    ax.set_ylim(*SPEED_LIM)
    ax.set_aspect('equal', adjustable='box')
    S.tidy(ax, axis='both')
    ax.set_xlabel(f'speed at {V.PAIRS["bottom"][0]} m (m s$^{{-1}}$)')
    ax.set_ylabel(f'speed at {V.PAIRS["bottom"][1]} m (m s$^{{-1}}$)' if k == 0 else '')
    ax.set_title(site)
    S.panel(ax, 'ef'[k])
h = [S.line_handle(S.GUIDE, '-', marker='o', ms=4.8), S.line_handle(S.GUIDE, '--', marker='o', ms=4.8, mfc='white'),
     S.patch_handle('darkgray'), S.patch_handle(S.UNARRESTED),
     S.line_handle(S.ARRESTED, ':', lw=1.2), S.line_handle(S.GUIDE, '--', lw=0.8)]
lab = ['with the current (median and quartiles)', 'against the current', 'hours (logarithmic shading)',
       'unarrested Ekman layer', 'arrested layer (design and GEBCO heights)', 'equal speeds']
ax_leg.legend(h, lab, loc='center', ncol=3, fontsize=LEG, borderaxespad=0.0, handlelength=2.4,
              columnspacing=1.6)

fig.savefig(V.DIR_FIG / 'fig03_response.pdf')
S.save(fig, V.DIR_FIG / 'fig03_response.png')