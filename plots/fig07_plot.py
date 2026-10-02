"""
Figure 6. Path of the central water from the shelf break to the coast and the flux that
supplies it.

(a) and (b) change of the near-bottom temperature at the 200, 100 and 50 m moorings of the
    Ubatuba and Cabo Frio lines and of the coastal upwelling index of the line around the
    onset of a cold event at 200 m, referred to the days before the onset, with the mean
    over the events and its 95 % interval, and in the legend the share of the events that
    cool each inner site.
(c) and (d) the same onsets for the coastal trapped wave of Figure 5, the uplift of the
    isotherms at 155 m and the cross-isobath velocity at the lowest meter of the 200 m
    mooring, each in standard deviations of its record, with the 95 % interval. The wave
    is negative for along-shore stress toward the southwest and the velocity positive
    onshore.
(e) lag of the largest correlation between the near-bottom temperature at 200 m and at the
    inner moorings, and with the coastal index, against the cross-shelf distance from the
    200 m mooring, with its 95 % interval, filled when the correlation exceeds the 95 %
    level of phase randomised surrogates, with the lags of advection at the speeds written
    on each line and the lag of a coastal trapped wave across a line shaded. The moorings
    carry the colours and symbols of (a) and (b) and the coastal index a cross in the
    colour of its line.
(f) near-bottom flux of the deficit below 20 °C across the isobaths at every mooring with
    at least 60 days of record, the term of the mean flow (filled) and of the correlated
    fluctuations (hatched) with their 95 % intervals, and the range of the total over the
    frame of Figure 1 (bar beside), positive onshore.

Reads paper_v3/tables/fig06_*.csv. Writes paper_v3/figures/fig06_path.png and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config_v3 as V
import style as S

FIGSIZE = (7.48, 9.6)
LEG = S.LEGEND_SIZE - 0.6
ORDER = ['FUB3', 'FUB2', 'FUB1', 'FBG3', 'FCF3', 'FCF2', 'FCF1']
COAST_MARK = 'X'
CHAIN = (('coastal trapped wave', S.INK, '-', None),
         ('uplift at 155 m', None, '-', 'o'),
         ('cross-isobath velocity near the bottom', None, '--', None))


def read(name, **kw):
    p = V.DIR_TAB / f'{name}.csv'
    if p.exists() and p.stat().st_size > 5:
        return pd.read_csv(p, **kw)
    print(f'  {p.name} is missing, its elements are left out')
    return pd.DataFrame()


comp = read('fig06_composite')
reach = read('fig06_reach')
lags = read('fig06_lags')
flux = read('fig06_flux')

# ----------------------------------------------------------------- layout
fig = plt.figure(figsize=FIGSIZE)
kw = dict(left=0.08, right=0.985, wspace=0.18)
gs1 = gridspec.GridSpec(1, 2, figure=fig, top=0.915, bottom=0.715, **kw)
gs2 = gridspec.GridSpec(1, 2, figure=fig, top=0.615, bottom=0.415, **kw)
gs3 = gridspec.GridSpec(1, 2, figure=fig, top=0.335, bottom=0.055, left=0.08, right=0.985, wspace=0.22)
ax_line = [fig.add_subplot(gs1[0, 0]), fig.add_subplot(gs1[0, 1])]
ax_ch = [fig.add_subplot(gs2[0, 0]), fig.add_subplot(gs2[0, 1])]
ax_l = fig.add_subplot(gs3[0, 0])
ax_q = fig.add_subplot(gs3[0, 1])


def days_axis(ax, ylabel):
    ax.axvline(0.0, color=S.GUIDE, lw=0.7, zorder=1)
    ax.axhline(0.0, color=S.INK, lw=0.5, zorder=1)
    ax.set_xlim(*V.COMPOSITE_DAYS)
    S.tidy(ax, axis='both')
    ax.set_xlabel('days from the onset at 200 m')
    ax.set_ylabel(ylabel)


# ------------------------------------------------------ (a) and (b) the lines
ylo, yhi = 0.0, 0.0
for k, (ax, line) in enumerate(zip(ax_line, V.LINES)):
    members = V.LINES[line][::-1]                       # 200, 100 and 50 m
    h, lab = [], []
    g0 = comp[comp.line == line] if len(comp) else pd.DataFrame()
    for site in members:
        g = g0[g0.variable == site].sort_values('lag')
        if g.empty:
            continue
        col = S.SITE_COLOURS[site]
        ax.fill_between(g.lag, g.lo, g.hi, color=col, alpha=0.16, lw=0, zorder=2)
        ax.plot(g.lag, g['mean'], color=col, ls=S.SITE_LINESTYLES[site], lw=1.4, marker=S.SITE_MARKERS[site],
                ms=3.4, mec='black', mew=0.35, zorder=4)
        r = reach[(reach.site == site)] if len(reach) else pd.DataFrame()
        tag = f' ({int(r.events_reached.iloc[0])} of {int(r.events_seen.iloc[0])} events)' if len(r) else \
            f' ({int(g.n_events.iloc[0])} events)'
        h.append(S.line_handle(col, S.SITE_LINESTYLES[site], marker=S.SITE_MARKERS[site], ms=3.6))
        lab.append(site + tag)
        ylo, yhi = min(ylo, np.nanmin(g['mean'])), max(yhi, np.nanmax(g['mean']))
    g = g0[g0.variable == 'coastal index'].sort_values('lag')
    if len(g):
        ax.fill_between(g.lag, g.lo, g.hi, color=S.GUIDE, alpha=0.12, lw=0, zorder=1)
        ax.plot(g.lag, g['mean'], color=S.INK, ls=(0, (1, 1.2)), lw=1.4, zorder=3)
        h.append(S.line_handle(S.INK, (0, (1, 1.2)), lw=1.4))
        lab.append('coastal upwelling index')
    days_axis(ax, 'change of temperature (°C)' if k == 0 else '')
    ax.legend(h, lab, loc='lower left', bbox_to_anchor=(0.0, 1.0), fontsize=LEG, ncol=2, title=line,
              title_fontsize=LEG + 0.2, alignment='left', columnspacing=1.0)
    S.panel(ax, 'ab'[k], dy=1.02)
pad = 0.35 * (yhi - ylo) if yhi > ylo else 0.2
for ax in ax_line:
    ax.set_ylim(ylo - pad, yhi + pad)

# ------------------------------------------ (c) and (d) from the wave to the flow
ylim = 0.0
for k, (ax, line) in enumerate(zip(ax_ch, V.LINES)):
    outer = V.LINES[line][-1]
    col_site = S.SITE_COLOURS[outer]
    h, lab = [], []
    for name, col, ls, mk in CHAIN:
        g = comp[(comp.line == line) & (comp.variable == name)].sort_values('lag') if len(comp) else pd.DataFrame()
        if g.empty:
            continue
        c = col if col is not None else col_site
        ax.fill_between(g.lag, g.lo, g.hi, color=c, alpha=0.12, lw=0, zorder=2)
        ax.plot(g.lag, g['mean'], color=c, ls=ls, lw=1.4, marker=mk, ms=3.2, mec='black', mew=0.35, zorder=4)
        h.append(S.line_handle(c, ls, lw=1.4, marker=mk, ms=3.4))
        lab.append({'coastal trapped wave': 'coastal trapped wave', 'uplift at 155 m': 'uplift at 155 m',
                    'cross-isobath velocity near the bottom': 'cross-isobath velocity at 194 m'}[name])
        ylim = max(ylim, np.nanmax(np.abs(g['mean'])))
    days_axis(ax, 'standard deviations' if k == 0 else '')
    ax.legend(h, lab, loc='lower left', bbox_to_anchor=(0.0, 1.0), fontsize=LEG, ncol=2,
              title=f'events at {outer}', title_fontsize=LEG + 0.2, alignment='left', columnspacing=1.0)
    S.panel(ax, 'cd'[k], dy=1.02)
for ax in ax_ch:
    ax.set_ylim(-1.5 * ylim, 1.5 * ylim)

# ----------------------------------------------------- (e) lag against distance
xmax = 1.1 * float(lags.distance_km.max()) if len(lags) else 100.0
ytop = V.LAG_DAYS[1] + 0.5
xx = np.linspace(0.0, xmax, 50)
ax_l.axhspan(-0.5, 0.5, color=S.SEASON_RANGE, lw=0, zorder=0)
LABEL_AT = {0.02: 0.6, 0.05: 0.6, 0.10: 0.8}    # fraction of the line at which its speed is written
for sp in V.ADVECTIVE_SPEEDS:
    yy = xx * 1e3 / sp / 86400.0
    ax_l.plot(xx, yy, color=S.GUIDE, ls='--', lw=0.8, zorder=1)
    # the speed written where the line leaves the panel
    x_end = min(xmax, ytop * sp * 86400.0 / 1e3)
    y_end = x_end * 1e3 / sp / 86400.0
    ang = np.degrees(np.arctan2(y_end, x_end))
    f_ = LABEL_AT.get(sp, 0.6)
    ax_l.text(f_ * x_end, f_ * y_end, f'{100 * sp:.0f} cm s$^{{-1}}$', fontsize=6.2, color=S.GUIDE,
              ha='right', va='bottom', zorder=2)
for _, r in (lags.iterrows() if len(lags) else []):
    coast = r.site == 'coastal index'
    col = S.LINE_COLOURS[r.line] if coast else S.SITE_COLOURS[r.site]
    mk = COAST_MARK if coast else S.SITE_MARKERS[r.site]
    sig = r.r >= r.level
    ax_l.plot([r.distance_km, r.distance_km], [r.lag_lo, r.lag_hi], color=col, lw=1.0, zorder=3)
    ax_l.plot(r.distance_km, r.lag, marker=mk, ms=6.0, ls='none', mfc=col if sig else 'white',
              mec='black' if sig else col, mew=0.8, zorder=4)
ax_l.axhline(0.0, color=S.INK, lw=0.5, zorder=1)
ax_l.set_xlim(0.0, xmax)
ax_l.set_ylim(V.LAG_DAYS[0] - 3.0, ytop)          # room beneath the lags for the legend
S.tidy(ax_l, axis='both')
ax_l.set_xlabel('distance from the 200 m mooring (km)')
ax_l.set_ylabel('lag (days)')
ax_l.legend([S.line_handle(S.GUIDE, '--', lw=0.8), S.patch_handle(S.SEASON_RANGE),
             S.marker_handle(S.GUIDE, COAST_MARK, ms=5.5), S.marker_handle(S.GUIDE, 'o', ms=5.0, open_=True)],
            ['advection', 'coastal trapped wave', 'coastal index of each line', 'below the 95 % level'],
            loc='lower left', fontsize=LEG, ncol=1, framealpha=0.9)
S.panel(ax_l, 'e', dy=1.02)

# ------------------------------------------------ (f) near-bottom flux of central water
W = 0.30
fq = flux[(flux.T_c == V.SACW_T) & (flux.short == 0)] if len(flux) else pd.DataFrame()
sites = [s for s in ORDER if len(fq) and s in set(fq.site)]
for i, site in enumerate(sites):
    g = fq[fq.site == site]
    r0 = g[g.rotation == 0.0]
    if r0.empty:
        continue
    r0 = r0.iloc[0]
    col = S.SITE_COLOURS[site]
    for j, (key, hatch) in enumerate((('flux_mean', None), ('flux_eddy', '////'))):
        x = i + (j - 0.5) * W
        ax_q.bar(x, 100 * r0[key], W * 0.92, color=col if hatch is None else 'white',
                 ec='black' if hatch is None else col, hatch=hatch, lw=0.7, zorder=3)
        ax_q.plot([x, x], [100 * r0[f'{key}_lo'], 100 * r0[f'{key}_hi']], color=S.INK, lw=0.8, zorder=4)
    ax_q.plot([i + 0.42, i + 0.42], [100 * g.flux_total.min(), 100 * g.flux_total.max()], color=S.GUIDE, lw=2.4,
              solid_capstyle='butt', zorder=3)
lines_at = [i - 0.5 for i in range(1, len(sites)) if sites[i][:2] != sites[i - 1][:2] or sites[i][:3] != sites[i - 1][:3]]
for x in lines_at:
    ax_q.axvline(x, color=S.LIGHT_GUIDE, lw=0.6, ls=':', zorder=1)
ax_q.axhline(0.0, color=S.INK, lw=0.6, zorder=2)
ax_q.set_xticks(range(len(sites)))
ax_q.set_xticklabels(sites, fontsize=6.4)
ax_q.set_xlim(-0.6, len(sites) - 0.4)
S.tidy(ax_q, axis='y')
ax_q.set_ylabel('flux of the deficit below 20 °C (°C cm s$^{-1}$)')
ax_q.legend([S.patch_handle(S.GUIDE, ec='black'), S.patch_handle('white', ec=S.GUIDE, hatch='////'),
             S.line_handle(S.GUIDE, lw=2.4)], ['mean flow', 'correlated fluctuations', 'total over the frame'],
            loc='lower left', fontsize=LEG, ncol=1, framealpha=0.9,
            title='onshore positive', title_fontsize=LEG, alignment='left')
S.panel(ax_q, 'f', dy=1.02)

fig.savefig(V.DIR_FIG / 'fig06_path.pdf')
S.save(fig, V.DIR_FIG / 'fig06_path.png')