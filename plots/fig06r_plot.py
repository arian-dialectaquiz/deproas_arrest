"""
Figure 6. The arrest and the uplift under the season and the mesoscale activity.

(a) and (b) attenuation and veering (clockwise negative) of the complex gain of the
    bottom pair at each 200 m mooring, with the flow at 155 m running with the current
    (filled, upslope Ekman transport) and against it (open, downslope), for October to
    March (circle) and April to September (square), with the 95 % interval from
    resampling whole states, the value of the whole year (grey tick) and the band of an
    unarrested Ekman layer (shaded).
(c) buoyancy frequency of the layer between 155 and 194 m by kind of state and half,
    median (symbol), quartiles (thick bar) and fifth to ninety-fifth percentile (thin bar),
    with the share of unstable or mixed days above each group.
(d) duration of every state of the flow at 155 m against its mean eddy kinetic energy
    over the slope at FUB3 and FCF3, with the Spearman correlation in the legend.
(e) attenuation (upper) and veering (lower) of the bottom pair on the states of the lowest
    (triangle down) and the highest (triangle up) tercile of eddy kinetic energy, with the
    95 % interval and the band of an unarrested Ekman layer.
(f) correlation of the daily eddy kinetic energy with the uplift at 155 m, the position
    of the Brazil Current axis, the interior speed and the near-bottom flux of the deficit
    below 20 C carried by the fluctuations, with the 95 % level of phase randomised
    surrogates (dashes).
(g) wavelet power of the along-isobath velocity at 155 m at FUB3 over its longest
    campaign, with the 95 % level against red noise (contour) and the cone of influence
    (hatched).
(h) and (i) wavelet coherence of the uplift at 155 m with the coastal trapped wave (h)
    and with the eddy kinetic energy (i), with the 95 % level of red noise pairs (contour)
    and the cone of influence (hatched).

Reads paper_v3/tables/fig06r_*.csv, fig03_reference.csv and paper_v3/data/fig06r_wavelet.nc.
Writes paper_v3/figures/fig06_robustness.png and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, FuncFormatter

import config_v3 as V
import style as S

resp = pd.read_csv(V.DIR_TAB / 'fig06r_seasonal_response.csv')
buoy = pd.read_csv(V.DIR_TAB / 'fig06r_seasonal_buoyancy.csv')
st = pd.read_csv(V.DIR_TAB / 'fig06r_eke_states.csv')
dur = pd.read_csv(V.DIR_TAB / 'fig06r_eke_duration.csv')
terc = pd.read_csv(V.DIR_TAB / 'fig06r_eke_terciles.csv')
cor = pd.read_csv(V.DIR_TAB / 'fig06r_eke_correlation.csv')
ref = pd.read_csv(V.DIR_TAB / 'fig03_reference.csv')
wv = xr.open_dataset(V.DIR_DATA / 'fig06r_wavelet.nc')

SITES = [s for s in V.SITES_200 if s in set(resp.site)]
STATES = ('with the current', 'against the current')
HALF_MARKER = {'strat': 'o', 'mixed': 's'}
TERC_MARKER = {'lowest tercile': 'v', 'highest tercile': '^'}
X_HALF = {('with the current', 'strat'): -0.30, ('with the current', 'mixed'): -0.10,
          ('against the current', 'strat'): 0.10, ('against the current', 'mixed'): 0.30}
X_TERC = {('upslope', 'lowest tercile'): -0.30, ('upslope', 'highest tercile'): -0.10,
          ('downslope', 'lowest tercile'): 0.10, ('downslope', 'highest tercile'): 0.30}
VARS = ('uplift at 155 m', 'position of the axis', 'interior speed', 'flux by the fluctuations')
VAR_LABELS = ('Uplift', 'Axis position', 'Interior speed', 'Flux')
PERIOD_TICKS = (2, 4, 8, 16, 32, 64)
CMAP_POWER = S.truncated('cividis', 0.0, 1.0)
CMAP_COH = S.truncated('YlGnBu', 0.02, 0.95)
COI_HATCH = 'gainsboro'

fig = plt.figure(figsize=(7.2, 8.4))
gs = gridspec.GridSpec(3, 3, left=0.075, right=0.985, bottom=0.165, top=0.975, wspace=0.34, hspace=0.50)
ax = {k: fig.add_subplot(gs[r, c]) for k, (r, c) in zip('abcdfghi', [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2),
                                                                      (2, 0), (2, 1), (2, 2)])}
gs_e = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.12)
ax['e'] = fig.add_subplot(gs_e[0])
ax_ev = S.tidy(fig.add_subplot(gs_e[1], sharex=ax['e']), axis='y')
pg, ph, pi = ax['g'].get_position(), ax['h'].get_position(), ax['i'].get_position()
cax_p = fig.add_axes([pg.x0, pg.y0 - 0.058, pg.width, 0.010])
cax_c = fig.add_axes([ph.x0, ph.y0 - 0.058, pi.x1 - ph.x0, 0.010])
for k in 'abcdef':
    S.tidy(ax[k], axis='y')


def site_axis(a, sites):
    a.set_xticks(range(len(sites)))
    a.set_xticklabels(sites)
    a.set_xlim(-0.6, len(sites) - 0.4)


def unarrested(a, sites, key):
    for i, site in enumerate(sites):
        un = ref[(ref.site == site) & (ref.kind == 'unarrested')]
        if len(un):
            a.fill_between([i - 0.42, i + 0.42], un[key].min(), un[key].max(), color=S.UNARRESTED, lw=0,
                           zorder=1)


# ----------------------------------------------------------------- (a) and (b)
bottom = resp[resp.lower == V.PAIRS['bottom'][1]]
for k, key, lo, hi in (('a', 'gain', 'gain_lo', 'gain_hi'), ('b', 'veer', 'veer_lo', 'veer_hi')):
    a = ax[k]
    unarrested(a, SITES, key)
    for i, site in enumerate(SITES):
        g = bottom[bottom.site == site]
        for state in STATES:
            whole = g[(g.state == state) & (g.half == 'all')]
            if len(whole):
                a.plot([i + X_HALF[(state, 'strat')] - 0.06, i + X_HALF[(state, 'mixed')] + 0.06],
                       [whole[key].iloc[0]] * 2, color=S.LIGHT_GUIDE, lw=1.6, zorder=2, solid_capstyle='butt')
            for half in V.HALVES:
                r = g[(g.state == state) & (g.half == half)]
                if not len(r):
                    continue
                x, y = i + X_HALF[(state, half)], r[key].iloc[0]
                a.plot([x, x], [r[lo].iloc[0], r[hi].iloc[0]], color=S.SITE_COLOURS[site], lw=0.9, zorder=3)
                a.plot(x, y, marker=HALF_MARKER[half], ms=4.6, ls='none', mew=0.9, zorder=4,
                       mfc=S.SITE_COLOURS[site] if state == STATES[0] else 'white', mec=S.SITE_COLOURS[site])
    site_axis(a, SITES)
ax['a'].set_ylabel('Attenuation')
ax['b'].set_ylabel('Veering (°)')
ax['b'].axhline(0.0, color=S.GUIDE, lw=0.6, zorder=1)

# ------------------------------------------------------------------------- (c)
for i, site in enumerate(SITES):
    g = buoy[buoy.site == site]
    for state in STATES:
        for half in V.HALVES:
            r = g[(g.kind == state) & (g.half == half)]
            if not len(r):
                continue
            r = r.iloc[0]
            x = i + X_HALF[(state, half)]
            ax['c'].plot([x, x], [r.N_p5 * 1e3, r.N_p95 * 1e3], color=S.SITE_COLOURS[site], lw=0.7, zorder=2)
            ax['c'].plot([x, x], [r.N_q1 * 1e3, r.N_q3 * 1e3], color=S.SITE_COLOURS[site], lw=3.0,
                         solid_capstyle='butt', zorder=3)
            ax['c'].plot(x, r.N_median * 1e3, marker=HALF_MARKER[half], ms=4.6, ls='none', mew=0.9, zorder=4,
                         mfc=S.SITE_COLOURS[site] if state == STATES[0] else 'white', mec=S.SITE_COLOURS[site])
            ax['c'].text(x, 0.985, f'{r.unstable:.0f}', transform=ax['c'].get_xaxis_transform(), ha='center',
                         va='top', fontsize=5.4, color=S.INK)
site_axis(ax['c'], SITES)
ax['c'].set_ylim(0.0, ax['c'].get_ylim()[1] * 1.12)
ax['c'].set_ylabel('N (10$^{-3}$ s$^{-1}$)')

# ------------------------------------------------------------------------- (d)
S.tidy(ax['d'])
hd, ld = [], []
for site in [s for s in ('FUB3', 'FCF3') if s in set(st.site)]:
    g = st[st.site == site]
    for state in STATES:
        h = g[g.kind == state]
        ax['d'].plot(h.eke * 1e3, h.hours / 24.0, ls='none', marker=S.SITE_MARKERS[site], ms=3.0, mew=0.6,
                     mfc=S.SITE_COLOURS[site] if state == STATES[0] else 'white', mec=S.SITE_COLOURS[site])
    r = dur[(dur.site == site) & (dur.kind == 'all')]
    hd.append(S.marker_handle(S.SITE_COLOURS[site], S.SITE_MARKERS[site]))
    ld.append(f'ρ = {r.rho.iloc[0]:+.2f}' if len(r) else site)
ax['d'].set_xscale('log')
ax['d'].set_yscale('log')
for axis_ in (ax['d'].xaxis, ax['d'].yaxis):
    axis_.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    axis_.set_minor_formatter(FuncFormatter(lambda v, _: f'{v:g}' if f'{v:g}'[0] in '25' else ''))
ax['d'].set_xlabel('EKE (10$^{-3}$ m$^2$ s$^{-2}$)')
ax['d'].set_ylabel('State duration (days)')
ax['d'].legend(hd, ld, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6.4, borderaxespad=0.1,
               handletextpad=0.2, columnspacing=0.8)

# ------------------------------------------------------------------------- (e)
TS = [s for s in V.SITES_200 if s in set(terc.site)]
for a, word, key in ((ax['e'], 'gain', 'gain'), (ax_ev, 'veering', 'veer')):
    unarrested(a, TS, key)
    for i, site in enumerate(TS):
        for kind in ('upslope', 'downslope'):
            for grp in TERC_MARKER:
                r = terc[(terc.site == site) & (terc.group == grp) & (terc.quantity == f'{kind} {word}')]
                if not len(r):
                    continue
                r = r.iloc[0]
                x = i + X_TERC[(kind, grp)]
                a.plot([x, x], [r.lo, r.hi], color=S.SITE_COLOURS[site], lw=0.9, zorder=3)
                a.plot(x, r.value, marker=TERC_MARKER[grp], ms=4.4, ls='none', mew=0.9, zorder=4,
                       mfc=S.SITE_COLOURS[site] if kind == 'upslope' else 'white', mec=S.SITE_COLOURS[site])
site_axis(ax_ev, TS)
plt.setp(ax['e'].get_xticklabels(), visible=False)
ax['e'].set_ylabel('Attenuation')
ax_ev.set_ylabel('Veering (°)')
ax_ev.axhline(0.0, color=S.GUIDE, lw=0.6, zorder=1)

# ------------------------------------------------------------------------- (f)
CS = [s for s in V.SITES_200 if s in set(cor.site)]
width = 0.8 / max(len(CS), 1)
for k, site in enumerate(CS):
    g = cor[cor.site == site].set_index('variable')
    for j, var in enumerate(VARS):
        if var not in g.index or not np.isfinite(g.loc[var, 'r']):
            continue
        x = j - 0.4 + width * (k + 0.5)
        ax['f'].bar(x, g.loc[var, 'r'], width=width * 0.9, color=S.SITE_COLOURS[site], ec='black', lw=0.4)
        for sgn in (1, -1):
            ax['f'].plot([x - width * 0.45, x + width * 0.45], [sgn * g.loc[var, 'level']] * 2, color=S.INK,
                         lw=0.8, ls='--')
ax['f'].axhline(0.0, color=S.GUIDE, lw=0.6)
ax['f'].set_xticks(range(len(VARS)))
ax['f'].set_xticklabels(VAR_LABELS, rotation=25, ha='right')
ax['f'].set_ylabel('Correlation with EKE')

# ------------------------------------------------------------------- (g) to (i)
t = pd.DatetimeIndex(wv.time.values)
per = wv.period.values
coi = wv.coi.values


def wavelet_panel(a, field, level, cmap, vmin, vmax):
    m = a.pcolormesh(t, per, field, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
    a.contour(t, per, level, levels=[1.0], colors=S.INK, linewidths=0.6)
    a.fill_between(t, coi, per.max() * 1.2, facecolor='none', edgecolor=COI_HATCH, hatch='xxx', lw=0)
    a.plot(t, coi, color=S.GUIDE, lw=0.6)
    a.set_yscale('log', base=2)
    a.set_ylim(min(per.max(), 64.0), per.min())
    a.yaxis.set_major_locator(FixedLocator(PERIOD_TICKS))
    a.yaxis.set_major_formatter(FixedFormatter([str(p) for p in PERIOD_TICKS]))
    a.yaxis.set_minor_locator(NullLocator())
    a.xaxis.set_major_locator(mdates.MonthLocator())
    a.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    return m


power = wv['power_u155'].values
mp = wavelet_panel(ax['g'], np.log2(power), wv['signif_u155'].values, CMAP_POWER, -3, 3)
ax['g'].set_ylabel('Period (days)')
for k, name in (('h', 'wave'), ('i', 'eke')):
    if f'coh_{name}' not in wv:
        ax[k].set_axis_off()
        continue
    R2 = wv[f'coh_{name}'].values
    mc = wavelet_panel(ax[k], R2, R2 / wv[f'level_{name}'].values[:, None], CMAP_COH, 0.0, 1.0)
ax['g'].set_xlabel(f'{t[0].year}' if t[0].year == t[-1].year else f'{t[0].year} to {t[-1].year}')
for k in 'hi':
    ax[k].set_xlabel(ax['g'].get_xlabel())
cb = fig.colorbar(mp, cax=cax_p, orientation='horizontal', extend='both')
cb.set_label('log$_2$ of the normalised power')
cb = fig.colorbar(mc, cax=cax_c, orientation='horizontal')#, extend='both')
cb.set_label('Wavelet coherence')

for k in 'abcdefghi':
    S.panel(ax[k], k, dy=1.10 if k == 'd' else 1.02)

# ---------------------------------------------------------------------- legends
h1 = [S.marker_handle('gray', 'o'), S.marker_handle('gray', 's'), S.marker_handle(S.INK, 'o'),
      S.marker_handle(S.INK, 'o', open_=True), S.marker_handle(S.INK, 'v'), S.marker_handle(S.INK, '^'),
      S.line_handle(S.LIGHT_GUIDE, lw=1.6), S.patch_handle(S.UNARRESTED), S.line_handle(S.INK, ls='--', lw=0.8)]
l1 = [V.HALF_SHORT['strat'], V.HALF_SHORT['mixed'], 'With the current (upslope)', 'Against the current (downslope)',
      'Lowest EKE tercile', 'Highest EKE tercile', 'Whole year', 'Unarrested Ekman layer', '95% level']
h1 += [S.patch_handle(S.SITE_COLOURS[s], ec='black') for s in SITES]
l1 += SITES
fig.legend(h1, l1, loc='lower center', bbox_to_anchor=(0.52, 0.0), ncol=6, fontsize=6.6, columnspacing=1.0,
           handletextpad=0.4)

fig.savefig(V.DIR_FIG / 'fig06_robustness.pdf')
S.save(fig, V.DIR_FIG / 'fig06_robustness.png')