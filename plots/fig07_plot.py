"""
Figure 7. Path of the central water from the shelf break to the coast.

(a) and (b) change of the near-bottom temperature at the 200, 100 and 50 m moorings of the
    Ubatuba and Cabo Frio lines and of the coastal upwelling index of the line around the
    onset of a cold event at 200 m, referred to the days before the onset, the mean over
    the events (lines) with the 95 % interval of the 200 m mooring (shading), and in the
    legend the number of events that cool each inner mooring by 0.5 C within ten days
    out of those it recorded.
(c) and (d) the same onsets for the coastal trapped wave of Figure 5, the uplift of the
    isotherms at 155 m and the cross-isobath velocity at the lowest meter of the 200 m
    mooring, each in standard deviations of its record, with the 95 % interval. The wave
    is negative for along-shore stress toward the southwest and the velocity positive
    onshore.
(e) and (f) lagged correlation of the near-bottom temperature at each inner mooring of
    the line, and of the coastal index, with the near-bottom temperature at 200 m,
    positive lags when the inner series follows, with the largest correlation marked
    (filled when it exceeds the 95 % level of phase randomised surrogates). The narrow
    band around zero is the lag of a coastal trapped wave across the line, and the shaded
    range the lags of advection at 10 cm s-1 or slower from the 200 m mooring to the
    nearest inner mooring.
The near-bottom flux of the deficit below 20 C, the former panel (g), is drawn by
figS1_flux_plot.py for the Supporting Information.

Reads paper_v3/tables/fig06_*.csv. Writes paper_v3/figures/fig06_path.png and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config_v3 as V
import style as S

FIGSIZE = (7.48, 8.6)
COAST_LS = (0, (1, 1.2))
CHAIN = (('coastal trapped wave', S.INK, '-', 'coastal trapped wave'),
         ('uplift at 155 m', None, '-', 'uplift at 155 m'),
         ('cross-isobath velocity near the bottom', None, '--', 'cross-isobath velocity at 194 m'))
CTW_LAG_DAYS = 0.5                  # lag of a coastal trapped wave across a line, half width of the band
ADVECTIVE_MAX = 0.10                # m s-1, fastest near-bottom advection drawn


def read(name, **kw):
    p = V.DIR_TAB / f'{name}.csv'
    if p.exists() and p.stat().st_size > 5:
        return pd.read_csv(p, **kw)
    print(f'  {p.name} is missing, its elements are left out')
    return pd.DataFrame()


comp = read('fig06_composite')
reach = read('fig06_reach')
lags = read('fig06_lags')
curves = read('fig06_lagcurves')

# ----------------------------------------------------------------- layout
fig = plt.figure(figsize=FIGSIZE)
kw = dict(left=0.085, right=0.985, wspace=0.12)
gs1 = gridspec.GridSpec(1, 2, figure=fig, top=0.91, bottom=0.715, **kw)
gs2 = gridspec.GridSpec(1, 2, figure=fig, top=0.60, bottom=0.405, **kw)
gs3 = gridspec.GridSpec(1, 2, figure=fig, top=0.31, bottom=0.115, **kw)
ax_line = [fig.add_subplot(gs1[0, 0]), fig.add_subplot(gs1[0, 1])]
ax_ch = [fig.add_subplot(gs2[0, 0]), fig.add_subplot(gs2[0, 1])]
ax_l = [fig.add_subplot(gs3[0, 0]), fig.add_subplot(gs3[0, 1])]
ax_line[1].sharey(ax_line[0])
ax_ch[1].sharey(ax_ch[0])
ax_l[1].sharey(ax_l[0])


def days_axis(ax, ylabel, first):
    ax.axvline(0.0, color=S.GUIDE, lw=0.8, zorder=1)
    ax.axhline(0.0, color=S.INK, lw=0.6, zorder=1)
    ax.set_xlim(*V.COMPOSITE_DAYS)
    ax.set_xticks(np.arange(V.COMPOSITE_DAYS[0], V.COMPOSITE_DAYS[1] + 1, 4))
    S.tidy(ax, axis='both')
    ax.set_xlabel('days from the onset at 200 m')
    ax.set_ylabel(ylabel if first else '')
    if not first:
        plt.setp(ax.get_yticklabels(), visible=False)


# ------------------------------------------------------ (a) and (b) the lines
ylo, yhi = 0.0, 0.0
for k, (ax, line) in enumerate(zip(ax_line, V.LINES)):
    members = V.LINES[line][::-1]                       # 200, 100 and 50 m
    h, lab = [], []
    g0 = comp[comp.line == line] if len(comp) else pd.DataFrame()
    for site in members:
        g = g0[g0.variable == site].sort_values('lag') if len(g0) else pd.DataFrame()
        if g.empty:
            continue
        col = S.SITE_COLOURS[site]
        outer = site == members[0]
        if outer:                                       # the interval of the mooring that defines the events
            ax.fill_between(g.lag, g.lo, g.hi, color=col, alpha=S.BAND_ALPHA, lw=0, zorder=2)
        ax.plot(g.lag, g['mean'], color=col, ls=S.SITE_LINESTYLES[site], lw=2.0 if outer else 1.6, zorder=4)
        r = reach[(reach.site == site)] if len(reach) else pd.DataFrame()
        tag = f' ({int(r.events_reached.iloc[0])} of {int(r.events_seen.iloc[0])})' if len(r) else \
            f' ({int(g.n_events.iloc[0])} events)'
        h.append(S.line_handle(col, S.SITE_LINESTYLES[site], lw=2.0 if outer else 1.6))
        lab.append(site + tag)
        ylo, yhi = min(ylo, np.nanmin(g['mean'])), max(yhi, np.nanmax(g['mean']))
    g = g0[g0.variable == 'coastal index'].sort_values('lag') if len(g0) else pd.DataFrame()
    if len(g):
        ax.plot(g.lag, g['mean'], color=S.INK, ls=COAST_LS, lw=1.6, zorder=3)
        h.append(S.line_handle(S.INK, COAST_LS, lw=1.6))
        lab.append('coastal index')
    days_axis(ax, 'change of temperature (°C)', k == 0)
    ax.legend(h, lab, loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=2, columnspacing=1.0)
    ax.text(1.0, 1.02, line, transform=ax.transAxes, ha='right', va='bottom', fontweight='bold')
    S.panel(ax, 'ab'[k], dy=1.04)
pad = 0.25 * (yhi - ylo) if yhi > ylo else 0.2
ax_line[0].set_ylim(ylo - pad, yhi + pad)

# ------------------------------------------ (c) and (d) from the wave to the flow
ylim = 0.0
for k, (ax, line) in enumerate(zip(ax_ch, V.LINES)):
    outer = V.LINES[line][-1]
    col_site = S.SITE_COLOURS[outer]
    h, lab = [], []
    for name, col, ls, label in CHAIN:
        g = comp[(comp.line == line) & (comp.variable == name)].sort_values('lag') if len(comp) else pd.DataFrame()
        if g.empty:
            continue
        c = col if col is not None else col_site
        ax.fill_between(g.lag, g.lo, g.hi, color=c, alpha=S.BAND_ALPHA_LIGHT, lw=0, zorder=2)
        ax.plot(g.lag, g['mean'], color=c, ls=ls, lw=1.8, zorder=4)
        h.append(S.line_handle(c, ls, lw=1.8))
        lab.append(label)
        ylim = max(ylim, np.nanmax(np.abs(g['mean'])))
    days_axis(ax, 'standard deviations', k == 0)
    ax.text(1.0, 1.02, outer, transform=ax.transAxes, ha='right', va='bottom', fontweight='bold')
    S.panel(ax, 'cd'[k], dy=1.04)
ax_ch[0].set_ylim(-1.35 * ylim, 1.35 * ylim)
grey = S.LIGHT_GUIDE
fig.legend([S.line_handle(S.INK, '-', lw=1.8), S.line_handle(grey, '-', lw=1.8), S.line_handle(grey, '--', lw=1.8)],
           [c[3] for c in CHAIN], loc='lower center', bbox_to_anchor=(0.5, ax_ch[0].get_position().y1 + 0.022),
           ncol=3, columnspacing=1.6)

# ------------------------------------------------- (e) and (f) lagged correlations
for k, (ax, line) in enumerate(zip(ax_l, V.LINES)):
    gl = curves[curves.line == line] if len(curves) else pd.DataFrame()
    inner = [s for s in V.LINES[line][::-1][1:] if len(gl) and s in set(gl.site)]
    d_min = min(float(gl[gl.site == s].distance_km.iloc[0]) for s in inner) if inner else np.nan
    ax.axvspan(-CTW_LAG_DAYS, CTW_LAG_DAYS, color=S.SEASON_RANGE, lw=0, zorder=0)
    if np.isfinite(d_min):
        ax.axvspan(d_min * 1e3 / ADVECTIVE_MAX / 86400.0, V.LAG_DAYS[1] + 1, color='whitesmoke', lw=0, zorder=0)
    for site in inner + ['coastal index']:
        g = gl[gl.site == site].sort_values('lag')
        if g.empty:
            continue
        coast = site == 'coastal index'
        col = S.INK if coast else S.SITE_COLOURS[site]
        ls = COAST_LS if coast else S.SITE_LINESTYLES[site]
        ax.plot(g.lag, g.r, color=col, ls=ls, lw=1.6, zorder=3)
        j = int(np.nanargmax(g.r.values))
        sig = g.r.iloc[j] >= g.level.iloc[0]
        mk = 'X' if coast else S.SITE_MARKERS[site]
        ax.plot(g.lag.iloc[j], g.r.iloc[j], marker=mk, ms=7.0, mfc=col if sig else 'white',
                mec='black' if sig else col, mew=0.6 if sig else 1.4, ls='none', zorder=5)
    ax.axhline(0.0, color=S.INK, lw=0.6, zorder=1)
    ax.set_xlim(V.LAG_DAYS[0], V.LAG_DAYS[1])
    S.tidy(ax, axis='both')
    ax.set_xlabel('lag (days)')
    ax.set_ylabel('correlation with 200 m' if k == 0 else '')
    if k:
        plt.setp(ax.get_yticklabels(), visible=False)
    ax.text(1.0, 1.02, line, transform=ax.transAxes, ha='right', va='bottom', fontweight='bold')
    S.panel(ax, 'ef'[k], dy=1.04)
    h = [S.line_handle(S.SITE_COLOURS[s], S.SITE_LINESTYLES[s], lw=1.6) for s in inner]
    lab = list(inner)
    ax.legend(h, lab, loc='upper right', bbox_to_anchor=(1.0, 1.0), handlelength=1.8, framealpha=0.9,
              frameon=True, edgecolor='none')
lo_r = min(-0.2, float(curves.r.min()) - 0.05) if len(curves) else -0.2
ax_l[0].set_ylim(lo_r, 1.0)

# ------------------------------------------------------- legend of (e) and (f)
h = [S.marker_handle(S.LIGHT_GUIDE, 'o', ms=6.5), S.marker_handle(S.LIGHT_GUIDE, 'o', ms=6.5, open_=True),
     S.line_handle(S.INK, COAST_LS, lw=1.6), S.patch_handle(S.SEASON_RANGE), S.patch_handle('whitesmoke', ec='silver')]
lab = ['largest, above the 95 % level', 'largest, below the level', 'coastal index', 'coastal trapped wave',
       f'advection at {100 * ADVECTIVE_MAX:.0f} cm s$^{{-1}}$ or slower']
p0, p1 = ax_l[0].get_position(), ax_l[1].get_position()
fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5 * (p0.x0 + p1.x1), 0.0), ncol=3,
           columnspacing=1.0, handlelength=1.5, handletextpad=0.4)

fig.savefig(V.DIR_FIG / 'fig06_path.pdf')
S.save(fig, V.DIR_FIG / 'fig06_path.png')
