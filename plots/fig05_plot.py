"""
Figure 5. What brings the central water to the 200 m isobath.

(a) first (solid, filled) and second (dashed, open) empirical orthogonal function of the
    daily temperature at the levels of each 200 m mooring, scaled by the square root of
    their variance, with the share of the variance each carries in the legend.
(b) variance of the near-bottom temperature split into the heave of the interior that
    reaches the lowest meter (filled) and the variability of the layer itself (hatched),
    T'_194 = beta T'_155 + r, with beta beneath each site.
(c) correlation between the uplift of the isotherms at 155 m and the hindcast of a
    first-mode coastal trapped wave forced by the along-shore wind stress on the 100 m
    isobath, against the phase speed at the best decay time of each site, with the 95 %
    level of phase randomised surrogates (dotted), the range of phase speeds chosen in
    campaigns left out (bars at the base) and the phase speeds of the bight in Dottori
    and Castro (2018) shaded.
(d) observed uplift at FUB3 and FCF3 over the longest campaign of each (black) and the
    uplift reconstructed from the coastal trapped wave (colour of the site). The
    reconstruction that adds the position of the Brazil Current axis lies on the same
    line and is left out.
(e) and (f) coherence of the uplift at FUB3 and FCF3, multiple on the coastal trapped
    wave and the position of the Brazil Current axis (black), and partial with the wave
    (solid) and with the position of the axis (dashed), each with the other removed, with
    the 95 % level of the partial coherence (dash-dot) and of the multiple (arrowhead on
    the right edge), and the synoptic band shaded.
(g) variance of the uplift explained by the coastal trapped wave (filled), by the position
    of the Brazil Current axis (hatched) and by both (cross-hatched), with the 95 %
    interval from resampling blocks of five days.

Reads paper_v3/tables/fig05_*.csv. Writes paper_v3/figures/fig05_uplift.png and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import NullFormatter

import config_v3 as V
import style as S

FIGSIZE = (7.48, 9.2)
LEG = S.LEGEND_SIZE - 0.6
SERIES_SITES = ('FUB3', 'FCF3')
COH_SITES = ('FUB3', 'FCF3')
PRED = (('wave', None), ('axis', '////'), ('both', 'xxxx'))
PARTIAL_LEVEL = (0, (5, 1.5, 1, 1.5))    # dash-dot, a style no other line of (e) and (f) uses
G_MAX = 60.0


def read(name, **kw):
    p = V.DIR_TAB / f'{name}.csv'
    if p.exists() and p.stat().st_size > 5:
        return pd.read_csv(p, **kw)
    print(f'  {p.name} is missing, its elements are left out')
    return pd.DataFrame()


modes = read('fig05_modes')
part = read('fig05_partition')
grid = read('fig05_ctw_grid')
ctw = read('fig05_ctw')
attr = read('fig05_attribution')
coh = read('fig05_coherence')
daily = read('fig05_daily', parse_dates=['day'])
sites = [s for s in V.SITES_200 if (len(ctw) and s in set(ctw.site)) or (len(modes) and s in set(modes.site))]

# ----------------------------------------------------------------- layout
fig = plt.figure(figsize=FIGSIZE)
gs_a = gridspec.GridSpec(1, 3, figure=fig, left=0.08, right=0.985, top=0.865, bottom=0.665, wspace=0.42,
                         width_ratios=[0.9, 0.75, 1.35])
gs_b = gridspec.GridSpec(1, 2, figure=fig, left=0.08, right=0.985, top=0.57, bottom=0.44, wspace=0.16)
gs_c = gridspec.GridSpec(1, 3, figure=fig, left=0.08, right=0.985, top=0.285, bottom=0.05, wspace=0.40)
ax_m = fig.add_subplot(gs_a[0, 0])
ax_p = fig.add_subplot(gs_a[0, 1])
ax_c = fig.add_subplot(gs_a[0, 2])
ax_ts = [fig.add_subplot(gs_b[0, 0]), fig.add_subplot(gs_b[0, 1])]
ax_coh = [fig.add_subplot(gs_c[0, 0]), fig.add_subplot(gs_c[0, 1])]
ax_g = fig.add_subplot(gs_c[0, 2])

# ---------------------------------------------------------------- (a) modes
h, lab = [], []
for site in sites:
    g = modes[modes.site == site] if len(modes) else pd.DataFrame()
    if g.empty:
        continue
    fr = []
    for m, ls in ((1, '-'), (2, '--')):
        q = g[g['mode'] == m].sort_values('level')
        ax_m.plot(q.amplitude, q.level, color=S.SITE_COLOURS[site], ls=ls, lw=1.2, marker=S.SITE_MARKERS[site],
                  ms=4.0, mfc=S.SITE_COLOURS[site] if m == 1 else 'white', mec='black', mew=0.45, zorder=3)
        fr.append(100 * q.fraction.iloc[0])
    h.append(S.line_handle(S.SITE_COLOURS[site], '-', marker=S.SITE_MARKERS[site], ms=4.0))
    lab.append(f'{site} {fr[0]:.0f} and {fr[1]:.0f} %')
h += [S.line_handle(S.GUIDE, '-', lw=1.2), S.line_handle(S.GUIDE, '--', lw=1.2)]
lab += ['first mode', 'second mode']
ax_m.axvline(0.0, color=S.INK, lw=0.6, zorder=1)
ax_m.set_ylim(210, 0)
S.tidy(ax_m, axis='both')
ax_m.set_xlabel('amplitude (°C)')
ax_m.set_ylabel('depth (m)')
S.panel(ax_m, 'a')
ax_m.legend(h, lab, loc='lower left', bbox_to_anchor=(0.0, 1.01), fontsize=LEG, ncol=1,
            title='share of the first and second mode', title_fontsize=LEG, alignment='left')

# -------------------------------------------------------- (b) the partition
W = 0.5
ticks = []
for i, site in enumerate(sites):
    r = part[part.site == site] if len(part) else pd.DataFrame()
    if r.empty or 'transmitted' not in r:
        ticks.append(site)
        continue
    r = r.iloc[0]
    ax_p.bar(i, r.transmitted, W, color=S.SITE_COLOURS[site], ec='black', lw=0.7, zorder=3)
    ax_p.bar(i, r.own, W, bottom=r.transmitted, color='white', ec=S.SITE_COLOURS[site], hatch='////', lw=0.7,
             zorder=3)
    ticks.append(f'{site}\nβ {r.beta:.2f}')
ax_p.axhline(1.0, color=S.GUIDE, lw=0.7, ls=':', zorder=2)
ax_p.set_xticks(range(len(sites)))
ax_p.set_xticklabels(ticks)
ax_p.set_xlim(-0.6, len(sites) - 0.4)
ax_p.set_ylim(0.0, 1.08)
S.tidy(ax_p, axis='y')
ax_p.set_ylabel('share of the variance of T$_{194}$')
S.panel(ax_p, 'b')
ax_p.legend([S.patch_handle(S.GUIDE, ec='black'), S.patch_handle('white', ec=S.GUIDE, hatch='////')],
            ['heave of the interior', 'the layer itself'], loc='lower left', bbox_to_anchor=(0.0, 1.01),
            fontsize=LEG)

# ------------------------------------------------- (c) the coastal trapped wave
ax_c.axvspan(*V.CTW_LIT_C, color=S.SEASON_RANGE, lw=0, zorder=0)
h, lab = [], []
for k, site in enumerate(sites):
    b = ctw[ctw.site == site] if len(ctw) else pd.DataFrame()
    if b.empty:
        continue
    b = b.iloc[0]
    g = grid[(grid.site == site) & (grid.tf_days == b.tf_days)].sort_values('c')
    ax_c.plot(g.c, g.r.abs(), color=S.SITE_COLOURS[site], ls=S.SITE_LINESTYLES[site], lw=1.2,
              marker=S.SITE_MARKERS[site], ms=3.6, mec='black', mew=0.4, zorder=3)
    ax_c.axhline(b.r_level, color=S.SITE_COLOURS[site], ls=':', lw=1.0, zorder=2)
    ax_c.plot([b.c_min_cv - 0.2, b.c_max_cv + 0.2], [0.015 + 0.02 * k] * 2, color=S.SITE_COLOURS[site], lw=3.0,
              solid_capstyle='butt', zorder=4)
    h.append(S.line_handle(S.SITE_COLOURS[site], S.SITE_LINESTYLES[site], marker=S.SITE_MARKERS[site], ms=3.6))
    lab.append(f'{site} ({b.tf_days:g} days)')
h += [S.line_handle(S.GUIDE, ':', lw=1.0), S.line_handle(S.GUIDE, lw=3.0), S.patch_handle(S.SEASON_RANGE)]
lab += ['95 % level of surrogates', 'chosen out of sample', 'Dottori and Castro (2018)']
ax_c.set_xlim(min(V.CTW_C) - 0.5, max(V.CTW_C) + 0.5)
ax_c.set_ylim(bottom=0.0)
S.tidy(ax_c, axis='both')
ax_c.set_xlabel('phase speed (m s$^{-1}$)')
ax_c.set_ylabel('correlation with the uplift')
S.panel(ax_c, 'c')
ax_c.legend(h, lab, loc='lower left', bbox_to_anchor=(-0.05, 1.01), fontsize=LEG - 0.2, ncol=2,
            columnspacing=0.8, handlelength=1.8)

# ------------------------------------------------ (d) one campaign per site
ylim = 0.0
for k, (ax, site) in enumerate(zip(ax_ts, SERIES_SITES)):
    g = daily[daily.site == site] if len(daily) else pd.DataFrame()
    col = S.SITE_COLOURS[site]
    ax.axhline(0.0, color=S.GUIDE, lw=0.5, zorder=1)
    r = ctw[ctw.site == site] if len(ctw) else pd.DataFrame()
    title = site
    if len(g):
        full = g.dropna(subset=['eta', 'eta_wave'])
        camp = full.groupby('campaign').size().idxmax() if len(full) else None
        q = g[g.campaign == camp] if camp is not None else pd.DataFrame()
        ax.plot(q.day, q.eta, color=S.INK, lw=1.3, zorder=3)
        ax.plot(q.day, q.eta_wave, color=col, lw=1.3, zorder=4)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        title = f'{site} campaign {camp}'
        ylim = max(ylim, np.nanmax(np.abs(q[['eta', 'eta_wave']].values)))
    if len(r):
        title += f' with a wave of {r.c.iloc[0]:.0f} m s$^{{-1}}$ and {r.tf_days.iloc[0]:g} days'
    ax.legend([S.line_handle(S.INK, lw=1.3), S.line_handle(col, lw=1.3)],
              ['observed', 'from the coastal trapped wave'], ncol=2,
              loc='lower left', bbox_to_anchor=(0.0, 1.0), fontsize=LEG + 0.1, title=title,
              title_fontsize=LEG + 0.1, alignment='left')
    S.tidy(ax, axis='both')
    ax.tick_params(axis='x', labelsize=6.0)
    ax.set_ylabel('uplift at 155 m (m)' if k == 0 else '')
S.panel(ax_ts[0], 'd', dy=1.02)
for ax in ax_ts:
    ax.set_ylim(-1.1 * ylim, 1.1 * ylim)


# ---------------------------------------------------- (e) and (f) coherence
for k, (ax, site) in enumerate(zip(ax_coh, COH_SITES)):
    g = coh[coh.site == site].sort_values('period_days') if len(coh) else pd.DataFrame()
    col = S.SITE_COLOURS[site]
    ax.axvspan(*V.SYNOPTIC_DAYS, color=S.SEASON_SHADE, lw=0, zorder=0)
    if len(g):
        ax.plot(g.period_days, g.multiple, color=S.INK, lw=1.3, zorder=4)
        ax.plot(g.period_days, g.wave_partial, color=col, lw=1.3, zorder=3)
        ax.plot(g.period_days, g.axis_partial, color=col, lw=1.1, ls='--', zorder=3)
        ax.axhline(g.level_partial.iloc[0], color=col, ls=PARTIAL_LEVEL, lw=1.4, zorder=2)
        ax.plot(1.0, g.level.iloc[0], marker='<', ms=6.5, color=S.INK, transform=ax.get_yaxis_transform(),
                clip_on=False, zorder=6)
        nseg = int(g.n_segments.iloc[0])
    ax.set_xscale('log')
    ax.set_xlim(2.0, V.COH_SEG_DAYS)
    ax.set_xticks([2, 5, 10, 20, 40])
    ax.set_xticklabels(['2', '5', '10', '20', '40'])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.0, 1.0)
    S.tidy(ax, axis='both')
    ax.set_xlabel('period (days)')
    ax.set_ylabel('coherence' if k == 0 else '')
    S.panel(ax, 'ef'[k], dy=1.02)
    h = [S.line_handle(S.INK, lw=1.3), S.line_handle(col, lw=1.3), S.line_handle(col, '--', lw=1.1),
         S.marker_handle(S.INK, '<', ms=6.0), S.line_handle(col, PARTIAL_LEVEL, lw=1.4)]
    lab = ['multiple', '95 % level of the multiple (right edge)', 'partial with the wave',
           'partial with the axis position', '95 % level of the partial']
    h = [h[0], h[3], h[1], h[2], h[4]]
    ax.legend(h, lab, loc='lower left', bbox_to_anchor=(-0.02, 1.01), fontsize=LEG - 0.4, ncol=1,
              handlelength=1.8, title=f'{site} over {nseg} segments of {V.COH_SEG_DAYS} days' if len(g) else site,
              title_fontsize=LEG - 0.3, alignment='left')

# ------------------------------------------------------- (g) attribution
W = 0.26
up = attr[attr.target == 'uplift'] if len(attr) else pd.DataFrame()
for i, site in enumerate(sites):
    for j, (pred, hatch) in enumerate(PRED):
        r = up[(up.site == site) & (up.predictor == pred)]
        if r.empty:
            continue
        r = r.iloc[0]
        x = i + (j - 1) * W
        ax_g.bar(x, 100 * r.explained, W * 0.9, color=S.SITE_COLOURS[site] if hatch is None else 'white',
                 ec='black' if hatch is None else S.SITE_COLOURS[site], hatch=hatch, lw=0.7, zorder=3)
        ax_g.plot([x, x], [100 * max(r.lo, 0.0), min(100 * r.hi, G_MAX)], color=S.INK, lw=0.8, zorder=4)
ax_g.set_xticks(range(len(sites)))
ax_g.set_xticklabels(sites)
ax_g.set_xlim(-0.6, len(sites) - 0.4)
ax_g.set_ylim(0.0, G_MAX)
S.tidy(ax_g, axis='y')
ax_g.set_ylabel('uplift explained (%)')
S.panel(ax_g, 'g', dy=1.02)
ax_g.legend([S.patch_handle(S.GUIDE, ec='black'), S.patch_handle('white', ec=S.GUIDE, hatch='////'),
             S.patch_handle('white', ec=S.GUIDE, hatch='xxxx')],
            ['coastal trapped wave', 'position of the current axis', 'both'],
            loc='lower left', bbox_to_anchor=(0.0, 1.01), fontsize=LEG, ncol=1)

fig.savefig(V.DIR_FIG / 'fig05_uplift.pdf')
S.save(fig, V.DIR_FIG / 'fig05_uplift.png')