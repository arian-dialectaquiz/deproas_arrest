"""
Figure 5. What brings the central water to the 200 m isobath.

(a) first empirical orthogonal function of the daily temperature over the levels of
    each 200 m mooring, scaled by the square root of its variance, with the share of
    the variance it carries.
(b) variance of the near-bottom temperature split into the heave of the interior that
    reaches the lowest meter (filled) and the variability of the bottom layer itself
    (hatched), T'_194 = beta T'_155 + T'_L, with beta beneath each mooring.
(c) magnitude of the correlation between the uplift of the isotherms at 155 m and the
    hindcast of a first-mode coastal trapped wave forced by the along-shore wind stress
    on the 100 m isobath, against the phase speed at the best decay time of each
    mooring, with the 95 % level of phase randomised surrogates (dotted) and the phase
    speeds of the bight in Dottori and Castro (2018) (shading).
(d) uplift at 155 m at FUB3 over its longest campaign, observed (black) and
    reconstructed from the coastal trapped wave (colour).
(e) partial coherence at FUB3 of the uplift with the coastal trapped wave (solid) and
    with the position of the Brazil Current axis (dashed), each with the other input
    removed, with the 95 % level (dotted) and the synoptic band shaded.

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

FIGSIZE = (7.48, 6.6)
WAVE_SITES = ('FUB3', 'FCF3')       # moorings where the hindcast is significant
SERIES_SITE = 'FUB3'
COH_SITE = 'FUB3'


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
coh = read('fig05_coherence')
daily = read('fig05_daily', parse_dates=['day'])
mode_sites = [s for s in V.SITES_200 if len(modes) and s in set(modes.site)]
part_sites = [s for s in V.SITES_200 if len(part) and s in set(part.site)]
wave_sites = [s for s in WAVE_SITES if len(ctw) and s in set(ctw.site)]

# ----------------------------------------------------------------- layout
fig = plt.figure(figsize=FIGSIZE)
gs_a = gridspec.GridSpec(1, 3, figure=fig, left=0.085, right=0.985, top=0.885, bottom=0.565, wspace=0.42,
                         width_ratios=[0.85, 0.70, 1.25])
gs_b = gridspec.GridSpec(1, 2, figure=fig, left=0.085, right=0.985, top=0.385, bottom=0.085, wspace=0.26,
                         width_ratios=[1.75, 1.0])
ax_m = fig.add_subplot(gs_a[0, 0])
ax_p = fig.add_subplot(gs_a[0, 1])
ax_c = fig.add_subplot(gs_a[0, 2])
ax_ts = fig.add_subplot(gs_b[0, 0])
ax_coh = fig.add_subplot(gs_b[0, 1])

# ---------------------------------------------------------- (a) the first mode
h, lab = [], []
for site in mode_sites:
    q = modes[(modes.site == site) & (modes['mode'] == 1)].sort_values('level')
    if q.empty:
        continue
    col = S.SITE_COLOURS[site]
    ax_m.plot(q.amplitude, q.level, color=col, ls=S.SITE_LINESTYLES[site], lw=1.6, marker=S.SITE_MARKERS[site],
              ms=6.0, mec='black', mew=0.5, zorder=3)
    h.append(S.line_handle(col, S.SITE_LINESTYLES[site], marker=S.SITE_MARKERS[site], ms=6.0))
    lab.append(f'{site} ({100 * q.fraction.iloc[0]:.0f} %)')
ax_m.axvline(0.0, color=S.INK, lw=0.7, zorder=1)
ax_m.set_ylim(210, 0)
S.tidy(ax_m, axis='both')
ax_m.set_xlabel('amplitude (°C)')
ax_m.set_ylabel('depth (m)')
S.panel(ax_m, 'a')
ax_m.legend(h, lab, loc='lower left', bbox_to_anchor=(-0.04, 1.02), ncol=1, handlelength=2.0,
            title='first mode', alignment='left')

# -------------------------------------------------------- (b) the partition
W = 0.55
for i, site in enumerate(part_sites):
    r = part[part.site == site].iloc[0]
    col = S.SITE_COLOURS[site]
    ax_p.bar(i, 100 * r.transmitted, W, color=col, ec='black', lw=0.7, zorder=3)
    ax_p.bar(i, 100 * r.own, W, bottom=100 * r.transmitted, color='white', ec=col, hatch=S.DOWN_HATCH, lw=0.9,
             zorder=3)
ax_p.set_xticks(range(len(part_sites)))
ax_p.set_xticklabels([f'{s}\nβ = {float(part[part.site == s].beta.iloc[0]):.2f}' for s in part_sites])
ax_p.set_xlim(-0.6, len(part_sites) - 0.4)
ax_p.set_ylim(0.0, 100.0)
ax_p.tick_params(axis='x', length=0)
S.tidy(ax_p, axis='y')
ax_p.set_ylabel('near-bottom variance (%)')
S.panel(ax_p, 'b')
ax_p.legend([S.patch_handle(S.GUIDE, ec='black'), S.patch_handle('white', ec=S.GUIDE, hatch=S.DOWN_HATCH)],
            ['interior heave', 'bottom layer'], loc='lower left', bbox_to_anchor=(-0.08, 1.02), handlelength=1.6)

# ------------------------------------------------- (c) the coastal trapped wave
ax_c.axvspan(*V.CTW_LIT_C, color=S.SEASON_RANGE, lw=0, zorder=0)
h, lab = [], []
for site in wave_sites:
    b = ctw[ctw.site == site].iloc[0]
    g = grid[(grid.site == site) & (grid.tf_days == b.tf_days)].sort_values('c')
    col = S.SITE_COLOURS[site]
    ax_c.plot(g.c, g.r.abs(), color=col, ls=S.SITE_LINESTYLES[site], lw=1.6, marker=S.SITE_MARKERS[site],
              ms=5.4, mec='black', mew=0.5, zorder=3)
    ax_c.axhline(b.r_level, color=col, ls=':', lw=1.3, zorder=2)
    h.append(S.line_handle(col, S.SITE_LINESTYLES[site], marker=S.SITE_MARKERS[site], ms=5.4))
    lab.append(site)
h += [S.line_handle(S.GUIDE, ':', lw=1.3), S.patch_handle(S.SEASON_RANGE)]
lab += ['95 % level', 'Dottori and Castro (2018)']
ax_c.set_xlim(min(V.CTW_C) - 0.5, max(V.CTW_C) + 0.5)
ax_c.set_ylim(0.0, None)
S.tidy(ax_c, axis='both')
ax_c.set_xlabel('phase speed (m s$^{-1}$)')
ax_c.set_ylabel('correlation with the uplift')
S.panel(ax_c, 'c')
ax_c.legend(h, lab, loc='lower left', bbox_to_anchor=(-0.04, 1.02), ncol=2, columnspacing=1.0, handlelength=2.0)

# --------------------------------------------------- (d) the longest campaign
g = daily[daily.site == SERIES_SITE] if len(daily) else pd.DataFrame()
col = S.SITE_COLOURS[SERIES_SITE]
r = ctw[ctw.site == SERIES_SITE] if len(ctw) else pd.DataFrame()
if len(g):
    full = g.dropna(subset=['eta', 'eta_wave'])
    camp = full.groupby('campaign').size().idxmax() if len(full) else None
    q = g[g.campaign == camp].sort_values('day')
    ax_ts.axhline(0.0, color=S.GUIDE, lw=0.6, zorder=1)
    ax_ts.plot(q.day, q.eta, color=S.INK, lw=1.4, zorder=3)
    ax_ts.plot(q.day, q.eta_wave, color=col, lw=1.6, zorder=4)
    lim = 1.1 * np.nanmax(np.abs(q[['eta', 'eta_wave']].values))
    ax_ts.set_ylim(-lim, lim)
    ax_ts.set_xlim(q.day.min(), q.day.max())
    ax_ts.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    yr = sorted(set(q.day.dt.year))
    ax_ts.set_xlabel(' and '.join(str(y_) for y_ in yr))
wave_lab = 'coastal trapped wave'
if len(r):
    wave_lab += f' ({r.c.iloc[0]:.0f} m s$^{{-1}}$)'
ax_ts.legend([S.line_handle(S.INK, lw=1.4), S.line_handle(col, lw=1.6)], ['observed', wave_lab], ncol=2,
             loc='lower left', bbox_to_anchor=(-0.02, 1.0))
S.tidy(ax_ts, axis='both')
ax_ts.set_ylabel('uplift at 155 m (m)')
ax_ts.text(1.0, 1.02, SERIES_SITE, transform=ax_ts.transAxes, ha='right', va='bottom', fontweight='bold')
S.panel(ax_ts, 'd', dy=1.06)

# ------------------------------------------------------------ (e) coherence
g = coh[coh.site == COH_SITE].sort_values('period_days') if len(coh) else pd.DataFrame()
ax_coh.axvspan(*V.SYNOPTIC_DAYS, color=S.SEASON_SHADE, lw=0, zorder=0)
if len(g):
    ax_coh.plot(g.period_days, g.wave_partial, color=col, lw=1.7, zorder=3)
    ax_coh.plot(g.period_days, g.axis_partial, color=col, lw=1.4, ls='--', zorder=3)
    ax_coh.axhline(g.level_partial.iloc[0], color=S.GUIDE, ls=':', lw=1.4, zorder=2)
ax_coh.set_xscale('log')
ax_coh.set_xlim(2.0, V.COH_SEG_DAYS)
ax_coh.set_xticks([2, 5, 10, 20, 40])
ax_coh.set_xticklabels(['2', '5', '10', '20', '40'])
ax_coh.xaxis.set_minor_formatter(NullFormatter())
ax_coh.set_ylim(0.0, 1.0)
S.tidy(ax_coh, axis='both')
ax_coh.set_xlabel('period (days)')
ax_coh.set_ylabel('partial coherence')
ax_coh.legend([S.line_handle(col, lw=1.7), S.line_handle(col, '--', lw=1.4), S.line_handle(S.GUIDE, ':', lw=1.4)],
              ['with the wave', 'with the axis', '95 % level'], loc='lower left', bbox_to_anchor=(-0.04, 1.0),
              ncol=2, columnspacing=1.0, handlelength=1.8)
ax_coh.text(1.0, 1.02, COH_SITE, transform=ax_coh.transAxes, ha='right', va='bottom', fontweight='bold')
S.panel(ax_coh, 'e', dy=1.06)

fig.savefig(V.DIR_FIG / 'fig05_uplift.pdf')
S.save(fig, V.DIR_FIG / 'fig05_uplift.png')
"""
Figure 5. What brings the SACW to the 200 m isobath.

(a) first empirical orthogonal function of the daily temperature over the levels of
    each 200 m mooring, scaled by the square root of its variance, with its 95 %
    interval (shading) and the share of the variance it carries.
(b) variance of the near-bottom temperature split into the uplift of the interior that
    reaches the lowest meter (filled) and the variability of the bottom boundary layer
    itself (hatched), T'_194 = beta T'_155 + T'_L, with beta beneath each mooring and
    the 95 % interval of the split (whisker).
(c) magnitude of the correlation between the uplift of the isotherms at 155 m and the
    hindcast of a first-mode coastal trapped wave forced by the along-shore wind stress
    on the 100 m isobath, against the phase speed at the best decay time of each
    mooring, with its 95 % interval (shading), the phase speed chosen (ring), the
    95 % level of phase randomised surrogates (dotted) and the phase speeds of the
    bight in Dottori and Castro (2018) (shaded range).
(d) uplift at 155 m at FUB3 over its longest campaign, observed (black over grey) and
    reconstructed from the coastal trapped wave (colour with its 95 % interval), above
    the distance of the axis of the Brazil Current from the mooring over the same days,
    positive toward the mooring (dashed).
(e) dose and response, the median uplift in classes of equal size of the coastal
    trapped wave (solid, upwelling phase positive) and of the position of the axis
    (dashed, closer to the mooring positive), each in standard deviations, with 95 %
    intervals (shading).
(f) partial coherence of the uplift with the coastal trapped wave (solid) and with the
    position of the axis (dashed), each with the other input removed, with 95 %
    intervals (shading), the 95 % level (dotted) and the synoptic band (grey).

Every interval is drawn only when the tables hold it, so the script also runs on the
tables of the earlier fig05_compute.py, without the shading and without panel (e).

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

FIGSIZE = (7.48, 8.3)
WAVE_SITES = ('FUB3', 'FCF3')       # moorings where the hindcast is significant
SERIES_SITE = 'FUB3'
SHOW_AXIS = True                    # strip with the position of the axis beneath the series of panel (d)
OBSERVED_FILL = 'gainsboro'         # area between the observed uplift and zero


def read(name, **kw):
    p = V.DIR_TAB / f'{name}.csv'
    if p.exists() and p.stat().st_size > 5:
        return pd.read_csv(p, **kw)
    print(f'  {p.name} is missing, its elements are left out')
    return pd.DataFrame()


def magnitude_band(lo, hi):
    """Interval of the magnitude of a quantity from the interval of the quantity."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    crosses = np.sign(lo) != np.sign(hi)
    return np.where(crosses, 0.0, np.minimum(np.abs(lo), np.abs(hi))), np.maximum(np.abs(lo), np.abs(hi))


modes = read('fig05_modes')
part = read('fig05_partition')
grid = read('fig05_ctw_grid')
ctw = read('fig05_ctw')
coh = read('fig05_coherence')
resp = read('fig05_response')
daily = read('fig05_daily', parse_dates=['day'])
mode_sites = [s for s in V.SITES_200 if len(modes) and s in set(modes.site)]
part_sites = [s for s in V.SITES_200 if len(part) and s in set(part.site)]
wave_sites = [s for s in WAVE_SITES if len(ctw) and s in set(ctw.site)]

# ----------------------------------------------------------------- layout
fig = plt.figure(figsize=FIGSIZE)
gs_a = gridspec.GridSpec(1, 3, figure=fig, left=0.085, right=0.985, top=0.912, bottom=0.725, wspace=0.42,
                         width_ratios=[0.85, 0.70, 1.25])
gs_d = gridspec.GridSpec(2 if SHOW_AXIS else 1, 1, figure=fig, left=0.085, right=0.985, top=0.622, bottom=0.392,
                         hspace=0.10, height_ratios=[3.0, 1.0] if SHOW_AXIS else [1.0])
gs_e = gridspec.GridSpec(1, 2, figure=fig, left=0.085, right=0.490, top=0.252, bottom=0.058, wspace=0.13)
gs_f = gridspec.GridSpec(1, 2, figure=fig, left=0.580, right=0.985, top=0.252, bottom=0.058, wspace=0.13)
ax_m = fig.add_subplot(gs_a[0, 0])
ax_p = fig.add_subplot(gs_a[0, 1])
ax_c = fig.add_subplot(gs_a[0, 2])
ax_ts = fig.add_subplot(gs_d[0, 0])
ax_ax = fig.add_subplot(gs_d[1, 0], sharex=ax_ts) if SHOW_AXIS else None
ax_r = [fig.add_subplot(gs_e[0, 0])]
ax_r.append(fig.add_subplot(gs_e[0, 1], sharey=ax_r[0]))
ax_k = [fig.add_subplot(gs_f[0, 0])]
ax_k.append(fig.add_subplot(gs_f[0, 1], sharey=ax_k[0]))

# ---------------------------------------------------------- (a) the first mode
h, lab = [], []
for site in mode_sites:
    q = modes[(modes.site == site) & (modes['mode'] == 1)].sort_values('level')
    if q.empty:
        continue
    col = S.SITE_COLOURS[site]
    if 'amp_lo' in q:
        ax_m.fill_betweenx(q.level, q.amp_lo, q.amp_hi, color=col, alpha=S.BAND_ALPHA, lw=0, zorder=2)
    ax_m.plot(q.amplitude, q.level, color=col, ls=S.SITE_LINESTYLES[site], lw=1.6, marker=S.SITE_MARKERS[site],
              ms=6.0, mec='black', mew=0.5, zorder=3)
    h.append(S.line_handle(col, S.SITE_LINESTYLES[site], marker=S.SITE_MARKERS[site], ms=6.0))
    lab.append(f'{site} ({100 * q.fraction.iloc[0]:.0f} %)')
ax_m.axvline(0.0, color=S.INK, lw=0.7, zorder=1)
ax_m.set_ylim(210, 0)
S.tidy(ax_m, axis='both')
ax_m.set_xlabel('amplitude (°C)')
ax_m.set_ylabel('depth (m)')
S.panel(ax_m, 'a')
ax_m.legend(h, lab, loc='lower left', bbox_to_anchor=(-0.04, 1.02), ncol=1, handlelength=2.0,
            title='first mode', alignment='left')

# -------------------------------------------------------- (b) the partition
W = 0.55
for i, site in enumerate(part_sites):
    r = part[part.site == site].iloc[0]
    col = S.SITE_COLOURS[site]
    ax_p.bar(i, 100 * r.transmitted, W, color=col, ec='black', lw=0.7, zorder=3)
    ax_p.bar(i, 100 * r.own, W, bottom=100 * r.transmitted, color='white', ec=col, hatch=S.DOWN_HATCH, lw=0.9,
             zorder=3)
    if 'transmitted_lo' in part:
        ax_p.plot([i, i], [100 * r.transmitted_lo, 100 * r.transmitted_hi], color='white', lw=3.4,
                  solid_capstyle='butt', zorder=4)
        ax_p.errorbar(i, 100 * r.transmitted, yerr=[[100 * (r.transmitted - r.transmitted_lo)],
                                                    [100 * (r.transmitted_hi - r.transmitted)]],
                      color=S.INK, lw=1.3, capsize=3.5, capthick=1.3, zorder=5)
ax_p.set_xticks(range(len(part_sites)))
beta_fmt = 'β = {:.2f}' if len(part_sites) < 3 else 'β {:.2f}'
ax_p.set_xticklabels([f'{s}\n' + beta_fmt.format(float(part[part.site == s].beta.iloc[0])) for s in part_sites])
ax_p.set_xlim(-0.6, len(part_sites) - 0.4)
ax_p.set_ylim(0.0, 100.0)
ax_p.tick_params(axis='x', length=0)
S.tidy(ax_p, axis='y')
ax_p.set_ylabel('near-bottom variance (%)')
S.panel(ax_p, 'b')
ax_p.legend([S.patch_handle(S.GUIDE, ec='black'), S.patch_handle('white', ec=S.GUIDE, hatch=S.DOWN_HATCH)],
            ['interior uplift', 'bottom boundary layer'], loc='lower left', bbox_to_anchor=(-0.02, 1.02),
            handlelength=1.6)

# ------------------------------------------------- (c) the coastal trapped wave
ax_c.axvspan(*V.CTW_LIT_C, color=S.SEASON_RANGE, lw=0, zorder=0)
h, lab = [], []
for site in wave_sites:
    b = ctw[ctw.site == site].iloc[0]
    g = grid[(grid.site == site) & (grid.tf_days == b.tf_days)].sort_values('c')
    col = S.SITE_COLOURS[site]
    if 'r_lo' in g:
        lo, hi = magnitude_band(g.r_lo, g.r_hi)
        ax_c.fill_between(g.c, lo, hi, color=col, alpha=S.BAND_ALPHA, lw=0, zorder=2)
    ax_c.plot(g.c, g.r.abs(), color=col, ls=S.SITE_LINESTYLES[site], lw=1.6, marker=S.SITE_MARKERS[site],
              ms=5.4, mec='black', mew=0.5, zorder=3)
    ax_c.plot(b.c, abs(b.r), ls='none', marker='o', ms=11.0, mfc='none', mec=S.INK, mew=1.2, zorder=4)
    ax_c.axhline(b.r_level, color=col, ls=':', lw=1.3, zorder=2)
    h.append(S.line_handle(col, S.SITE_LINESTYLES[site], marker=S.SITE_MARKERS[site], ms=5.4))
    lab.append(site)
h += [S.line_handle(S.GUIDE, ':', lw=1.3), S.patch_handle(S.SEASON_RANGE)]
lab += ['95 % level', 'Dottori and Castro (2018)']
ax_c.set_xlim(min(V.CTW_C) - 0.5, max(V.CTW_C) + 0.5)
ax_c.set_ylim(0.0, 1.12 * ax_c.get_ylim()[1])
S.tidy(ax_c, axis='both')
ax_c.set_xlabel('phase speed (m s$^{-1}$)')
ax_c.set_ylabel('correlation with the uplift')
S.panel(ax_c, 'c')
ax_c.legend(h, lab, loc='lower left', bbox_to_anchor=(-0.04, 1.02), ncol=2, columnspacing=1.0, handlelength=2.0)

# --------------------------------------------------- (d) the longest campaign
g = daily[daily.site == SERIES_SITE] if len(daily) else pd.DataFrame()
col = S.SITE_COLOURS[SERIES_SITE]
r = ctw[ctw.site == SERIES_SITE] if len(ctw) else pd.DataFrame()
if len(g):
    full = g.dropna(subset=['eta', 'eta_wave'])
    camp = full.groupby('campaign').size().idxmax() if len(full) else None
    q = g[g.campaign == camp].sort_values('day')
    ax_ts.axhline(0.0, color=S.GUIDE, lw=0.6, zorder=1)
    ax_ts.fill_between(q.day, 0.0, q.eta, color=OBSERVED_FILL, lw=0, zorder=2)
    ax_ts.plot(q.day, q.eta, color=S.INK, lw=1.2, zorder=4)
    if 'eta_wave_lo' in q:
        ax_ts.fill_between(q.day, q.eta_wave_lo, q.eta_wave_hi, color=col, alpha=0.30, lw=0, zorder=3)
    ax_ts.plot(q.day, q.eta_wave, color=col, lw=1.9, zorder=5)
    lim = 1.08 * np.nanmax(np.abs(q[['eta', 'eta_wave']].values))
    ax_ts.set_ylim(-lim, lim)
    ax_ts.set_xlim(q.day.min(), q.day.max())
    if SHOW_AXIS:
        ax_ax.axhline(0.0, color=S.GUIDE, lw=0.6, zorder=1)
        ax_ax.fill_between(q.day, 0.0, q['axis'], color=col, alpha=S.BAND_ALPHA_LIGHT, lw=0, zorder=2)
        ax_ax.plot(q.day, q['axis'], color=col, lw=1.4, ls='--', zorder=3)
        lim = 1.15 * np.nanmax(np.abs(q['axis'].values))
        ax_ax.set_ylim(-lim, lim)
    last = ax_ax if SHOW_AXIS else ax_ts
    last.xaxis.set_major_locator(mdates.MonthLocator())
    last.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    last.set_xlabel(' and '.join(str(y_) for y_ in sorted(set(q.day.dt.year))))
wave_lab = 'coastal trapped wave'
if len(r):
    wave_lab += f' ({r.c.iloc[0]:.0f} m s$^{{-1}}$)'
h = [(S.patch_handle(OBSERVED_FILL), S.line_handle(S.INK, lw=1.2)),
     (S.patch_handle(col, alpha=0.30), S.line_handle(col, lw=1.9))]
lab = ['observed', wave_lab]
if SHOW_AXIS:
    h.append(S.line_handle(col, '--', lw=1.4))
    lab.append('axis of the Brazil Current')
ax_ts.legend(h, lab, ncol=3, loc='lower left', bbox_to_anchor=(-0.02, 1.0), columnspacing=1.6)
S.tidy(ax_ts, axis='both')
ax_ts.set_ylabel('uplift at 155 m (m)')
ax_ts.text(1.0, 1.02, SERIES_SITE, transform=ax_ts.transAxes, ha='right', va='bottom', fontweight='bold')
S.panel(ax_ts, 'd', dy=1.03)
if SHOW_AXIS:
    plt.setp(ax_ts.get_xticklabels(), visible=False)
    S.tidy(ax_ax, axis='both')
    ax_ax.set_ylabel('axis (km)')
    fig.align_ylabels([ax_ts, ax_ax])
    ax_ax.yaxis.set_major_locator(plt.MaxNLocator(3, symmetric=True))

# ------------------------------------------------------ (e) dose and response
for k, (site, ax) in enumerate(zip(WAVE_SITES, ax_r)):
    col = S.SITE_COLOURS[site]
    ax.axhline(0.0, color=S.GUIDE, lw=0.6, zorder=1)
    ax.axvline(0.0, color=S.GUIDE, lw=0.6, zorder=1)
    g = resp[resp.site == site] if len(resp) else pd.DataFrame()
    for name, ls, filled, alpha in (('axis', '--', False, S.BAND_ALPHA_LIGHT), ('wave', '-', True, S.BAND_ALPHA)):
        q = g[g.predictor == name].sort_values('x') if len(g) else pd.DataFrame()
        if q.empty:
            continue
        ax.fill_between(q.x, q.lo, q.hi, color=col, alpha=alpha, lw=0, zorder=2)
        ax.plot(q.x, q.eta, color=col, ls=ls, lw=1.7 if filled else 1.4, marker=S.SITE_MARKERS[site], ms=5.4,
                mfc=col if filled else 'white', mec='black' if filled else col, mew=0.5 if filled else 1.0,
                zorder=4 if filled else 3)
    ax.set_xlim(-2.1, 2.1)
    ax.set_xticks([-2, -1, 0, 1, 2])
    S.tidy(ax, axis='both')
    ax.set_title(site, fontweight='bold', pad=3.0)
ax_r[0].set_ylabel('uplift at 155 m (m)')
ax_r[0].set_xlabel('wave or axis (standard deviations)', x=1.065)
plt.setp(ax_r[1].get_yticklabels(), visible=False)
S.panel(ax_r[0], 'e', dy=1.05)
ax_r[0].legend([S.line_handle(S.INK, '-', lw=1.7, marker='o', ms=5.0),
                S.line_handle(S.INK, '--', lw=1.4, marker='o', ms=5.0, mfc='white')],
               ['coastal trapped wave (upwelling phase positive)', 'axis of the Brazil Current (closer positive)'],
               loc='lower left', bbox_to_anchor=(-0.04, 1.12), ncol=1, handlelength=2.4)

# ------------------------------------------------------------ (f) coherence
for k, (site, ax) in enumerate(zip(WAVE_SITES, ax_k)):
    col = S.SITE_COLOURS[site]
    g = coh[coh.site == site].sort_values('period_days') if len(coh) else pd.DataFrame()
    ax.axvspan(*V.SYNOPTIC_DAYS, color=S.SEASON_SHADE, lw=0, zorder=0)
    if len(g):
        if 'wave_partial_lo' in g:
            ax.fill_between(g.period_days, g.axis_partial_lo, g.axis_partial_hi, color=col, alpha=S.BAND_ALPHA_LIGHT,
                            lw=0, zorder=2)
            ax.fill_between(g.period_days, g.wave_partial_lo, g.wave_partial_hi, color=col, alpha=S.BAND_ALPHA,
                            lw=0, zorder=2)
        ax.plot(g.period_days, g.axis_partial, color=col, lw=1.4, ls='--', zorder=3)
        ax.plot(g.period_days, g.wave_partial, color=col, lw=1.8, zorder=4)
        ax.axhline(g.level_partial.iloc[0], color=S.GUIDE, ls=':', lw=1.4, zorder=3)
    ax.set_xscale('log')å
    ax.set_xlim(2.0, V.COH_SEG_DAYS)
    ax.set_xticks([2, 5, 10, 20, 40])
    ax.set_xticklabels(['2', '5', '10', '20', '40'])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.0, 1.0)
    S.tidy(ax, axis='both')
    ax.set_title(site, fontweight='bold', pad=3.0)
ax_k[0].set_ylabel('partial coherence')
ax_k[0].set_xlabel('period (days)', x=1.065)
plt.setp(ax_k[1].get_yticklabels(), visible=False)
S.panel(ax_k[0], 'f', dy=1.05)
ax_k[0].legend([S.line_handle(S.INK, lw=1.8), S.line_handle(S.INK, '--', lw=1.4), S.line_handle(S.GUIDE, ':', lw=1.4)
                S.patch_handle(S.SEASON_SHADE, ec='lightgray')],
               ['with the wave', 'with the axis', '95 % level', 'synoptic band'], loc='lower left',
               bbox_to_anchor=(-0.04, 1.12), ncol=2, columnspacing=1.2, handlelength=2.0)

fig.savefig(V.DIR_FIG / 'fig05_uplift.pdf')
S.save(fig, V.DIR_FIG / 'fig05_uplift.png')
