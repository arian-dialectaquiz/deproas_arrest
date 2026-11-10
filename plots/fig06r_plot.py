"""
Figure 6. The arrest and the uplift under the season and the mesoscale activity.

(a) asymmetry of the complex gain of the bottom pair at FUB3, the attenuation and the
    veering under upslope Ekman transport minus those under downslope transport, for
    the whole record (with its 95 % interval from resampling whole states, shading),
    for October to March (ONDJFM) and April to September (AMJJAS), and for the states
    in the lowest and the highest tercile of the eddy kinetic energy (EKE) over the
    slope. The shaded quadrant holds the weaker and less veered near-bottom flow under
    upslope transport that an arrest confined to that state gives.
(b) distribution of the daily squared buoyancy frequency of the layer between 155 and
    194 m at FUB3 and FCF3 in October to March (ONDJFM) and April to September (AMJJAS),
    drawn as in Figure 4a with the days sorted by the sign of the daily interior flow. The
    unstable or mixed days fill the bin below zero, without the percentages, which the text
    gives.
(c) correlation of the daily eddy kinetic energy over the slope with four daily series
    at each 200 m mooring, the uplift at 155 m, the offshore distance of the Brazil
    Current axis, the interior speed at 155 m and the near-bottom flux of the deficit
    below 20 C carried by the fluctuations. Values in bold on full colour exceed the
    95 % level of phase randomised surrogates, the others are faded.
(d) and (e) wavelet coherence of the uplift at 155 m with the coastal trapped wave (d)
    and with the eddy kinetic energy (e) over the longest campaign of FUB3, with the
    95 % level of red noise pairs (contour) and the cone of influence (hatched). The
    wavelet power of the velocity, which sets the synoptic and mesoscale bands, is
    drawn by figS2_wavelet_plot.py.

Reads paper_v3/tables/fig06r_*.csv, fig02_daily.csv, fig03_reference.csv and
paper_v3/data/fig06r_wavelet.nc. Writes paper_v3/figures/fig06_robustness.png and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

import config_v3 as V
import style as S

FIGSIZE = (7.48, 7.0)
GAIN_SITE = 'FUB3'
N2_SITES = ('FUB3', 'FCF3')
N2_SCALE = 1e-5
N2_STEP = 1.0
MIXED_BIN = (-1.7, -0.7)
HALF_SHORT = {'strat': 'ONDJFM', 'mixed': 'AMJJAS'}
# the subsets of (a), each with its marker, size and label
SUBSETS = (('all', 'o', 7.5, 'whole record'),
           ('strat', 's', 6.5, 'October to March'),
           ('mixed', 'D', 6.0, 'April to September'),
           ('lowest tercile', 'v', 7.0, 'lowest EKE tercile'),
           ('highest tercile', '^', 7.0, 'highest EKE tercile'))
SUBSET_SHORT = {'all': 'whole record', 'strat': 'ONDJFM', 'mixed': 'AMJJAS', 'lowest tercile': 'low EKE',
                'highest tercile': 'high EKE'}
VARS = (('uplift at 155 m', 'uplift', 1.0),
        ('position of the axis', 'axis offshore\ndistance', -1.0),     # stored positive toward the mooring
        ('interior speed', 'interior\nspeed', 1.0),
        ('flux by the fluctuations', 'flux by the\nfluctuations', 1.0))
PERIOD_TICKS = (2, 4, 8, 16, 32, 64)
CMAP_COH = S.truncated('YlGnBu', 0.02, 0.95)
CMAP_R = 'RdBu_r'


def read(name, **kw):
    p = V.DIR_TAB / f'{name}.csv'
    if p.exists() and p.stat().st_size > 5:
        return pd.read_csv(p, **kw)
    print(f'  {p.name} is missing, its elements are left out')
    return pd.DataFrame()


resp = read('fig06r_seasonal_response')
buoy = read('fig06r_seasonal_buoyancy')
terc = read('fig06r_eke_terciles')
cor = read('fig06r_eke_correlation')
ref = read('fig03_reference')
daily = read('fig02_daily', parse_dates=['day'])
f_wv = V.DIR_DATA / 'fig06r_wavelet.nc'
wv = xr.open_dataset(f_wv) if f_wv.exists() else None

# ----------------------------------------------------------------- layout
fig = plt.figure(figsize=FIGSIZE)
gs_top = gridspec.GridSpec(1, 3, figure=fig, left=0.095, right=0.99, top=0.90, bottom=0.56, wspace=0.50,
                           width_ratios=[1.05, 1.25, 0.80])
gs_bot = gridspec.GridSpec(1, 2, figure=fig, left=0.095, right=0.99, top=0.445, bottom=0.175, wspace=0.10)
ax_a = fig.add_subplot(gs_top[0, 0])
ax_b = fig.add_subplot(gs_top[0, 1])
ax_c = fig.add_subplot(gs_top[0, 2])
ax_w = [fig.add_subplot(gs_bot[0, k]) for k in range(2)]

# --------------------------------------------- (a) the asymmetry under every subset
col = S.SITE_COLOURS[GAIN_SITE]
bottom = resp[(resp.site == GAIN_SITE) & (resp.lower == V.PAIRS['bottom'][1])] if len(resp) else pd.DataFrame()
tt = terc[terc.site == GAIN_SITE] if len(terc) else pd.DataFrame()
asym = read('fig03_asymmetry')


def gain_of(subset, state):
    """Attenuation and veering of the bottom pair at the site for one subset and kind of state."""
    if subset in ('all', 'strat', 'mixed'):
        r = bottom[(bottom.half == subset) & (bottom.state == state)]
        return (float(r.gain.iloc[0]), float(r.veer.iloc[0])) if len(r) else None
    word = S.STATE_SHORT[state]
    g = tt[(tt.group == subset) & (tt.quantity == f'{word} gain')]
    v = tt[(tt.group == subset) & (tt.quantity == f'{word} veering')]
    return (float(g.value.iloc[0]), float(v.value.iloc[0])) if len(g) and len(v) else None


ax_a.add_patch(plt.Rectangle((-10.0, 0.0), 10.0, 1000.0, fc='whitesmoke', ec='none', zorder=0))
ax_a.axhline(0.0, color=S.INK, lw=0.7, zorder=1)
ax_a.axvline(0.0, color=S.INK, lw=0.7, zorder=1)
a0 = asym[(asym.site == GAIN_SITE) & (asym.pair == f'{V.PAIRS["bottom"][0]} to {V.PAIRS["bottom"][1]} m')
          & (asym.half == 'all')] if len(asym) else pd.DataFrame()
if len(a0):
    a0 = a0.iloc[0]
    ax_a.add_patch(plt.Rectangle((a0.gain_difference_lo, a0.veer_difference_lo),
                                 a0.gain_difference_hi - a0.gain_difference_lo,
                                 a0.veer_difference_hi - a0.veer_difference_lo, fc=col, alpha=S.BAND_ALPHA, lw=0,
                                 zorder=2))
pts = []
for subset, mk, ms, name in SUBSETS:
    up, dn = gain_of(subset, S.UP), gain_of(subset, S.DOWN)
    if up is None or dn is None:
        continue
    dg, dv = up[0] - dn[0], up[1] - dn[1]
    pts.append((dg, dv))
    ax_a.plot(dg, dv, marker=mk, ms=ms + 0.5, mfc=col, mec='black', mew=0.6, ls='none', zorder=5)
    ax_a.annotate(SUBSET_SHORT[subset], (dg, dv), xytext=(7, 0), textcoords='offset points', va='center',
                  fontsize=S.FS_SMALL, zorder=6)
if pts:
    P = np.array(pts)
    xr_ = max(0.32, 1.25 * np.abs(P[:, 0]).max())
    yr_ = max(30.0, 1.2 * np.abs(P[:, 1]).max())
    ax_a.set_xlim(-xr_, 0.5 * xr_)
    ax_a.set_ylim(-0.6 * yr_, yr_)
S.tidy(ax_a, axis='both')
ax_a.set_xlabel('attenuation, upslope minus downslope')
ax_a.set_ylabel('veering, upslope minus\ndownslope (°)')
ax_a.set_title(GAIN_SITE, fontweight='bold')
S.panel(ax_a, 'a')

# ------------------------------------------------ (b) N2 of the layer in each half
groups, colours, shares, ticks, pos = {}, {}, {}, [], []
x0 = 0.0
if len(daily):
    d = daily[daily.layer == 'lowest pair'] if 'layer' in daily else daily
    d = d.assign(half=[V.HALF_OF_MONTH[m] for m in d.day.dt.month],
                 kind=np.where(d.u_i <= -V.MIN_SPEED, S.UP, np.where(d.u_i >= V.MIN_SPEED, S.DOWN, '')))
    if {'T_top', 'T_bot'}.issubset(d.columns):
        d = d[np.isfinite(d.T_top) & np.isfinite(d.T_bot)]
    for site in N2_SITES:
        for half in V.HALVES:
            for state in (S.UP, S.DOWN):
                h = d[(d.site == site) & (d.half == half) & (d.kind == state)]
                if len(h) < 10:
                    continue
                groups[(x0, state)] = h.N.values.astype(float) ** 2 / N2_SCALE
                colours[(x0, state)] = S.SITE_COLOURS[site]
                b = buoy[(buoy.site == site) & (buoy.half == half) & (buoy.kind == state)] if len(buoy) else pd.DataFrame()
                if len(b):
                    shares[(x0, state)] = float(b.unstable.iloc[0])
            pos.append(x0)
            ticks.append(HALF_SHORT[half])
            x0 += 1.0
        x0 += 0.35
if groups:
    alln2 = np.concatenate([v[np.isfinite(v)] for v in groups.values()])
    n2_max = N2_STEP * np.ceil(np.percentile(alln2, 98) / N2_STEP)
    S.split_histograms(ax_b, groups, np.arange(0.0, n2_max + 1e-9, N2_STEP), colours, half_width=0.44,
                       mixed_bin=MIXED_BIN, labels=False)
    ax_b.set_ylim(MIXED_BIN[0] - 0.4, n2_max)
    yt = list(range(0, int(n2_max) + 1, 2))
    ax_b.set_yticks([0.5 * sum(MIXED_BIN)] + yt)
    ax_b.set_yticklabels(['$\\leq$0'] + [str(t) for t in yt])
ax_b.axhline(0.0, color=S.INK, lw=0.8, zorder=3)
ax_b.set_xticks(pos)
ax_b.set_xticklabels(ticks, fontsize=S.FS_TICK)
ax_b.tick_params(axis='x', length=0)
for k, site in enumerate(N2_SITES):
    if 2 * k + 1 < len(pos):
        ax_b.text(0.5 * (pos[2 * k] + pos[2 * k + 1]), 1.0, site, transform=ax_b.get_xaxis_transform(), ha='center',
                  va='bottom', fontweight='bold')
ax_b.set_xlim(pos[0] - 0.55 if pos else -0.5, pos[-1] + 0.55 if pos else 0.5)
S.tidy(ax_b, axis='y')
ax_b.set_ylabel('N$^2$ of the layer (10$^{-5}$ s$^{-2}$)')
S.panel(ax_b, 'b', dy=1.06)

# ------------------------------------------- (c) eddy kinetic energy and the daily series
cs = [s for s in V.SITES_200 if len(cor) and s in set(cor.site)]
R = np.full((len(VARS), len(cs)), np.nan)
SIG = np.zeros_like(R, dtype=bool)
for j, site in enumerate(cs):
    g = cor[cor.site == site].set_index('variable')
    for i, (var, _, sgn) in enumerate(VARS):
        if var in g.index and np.isfinite(g.loc[var, 'r']):
            R[i, j] = sgn * g.loc[var, 'r']
            SIG[i, j] = abs(g.loc[var, 'r']) > g.loc[var, 'level']
rgba = plt.get_cmap(CMAP_R)(Normalize(-0.6, 0.6)(np.nan_to_num(R)))
rgba[..., 3] = np.where(SIG, 0.85, 0.25)                 # the correlations below their level fade
rgba[~np.isfinite(R)] = (1, 1, 1, 0)
ax_c.imshow(rgba, aspect='auto')
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if np.isfinite(R[i, j]):
            ax_c.text(j, i, f'{R[i, j]:+.2f}'.replace('-', '−'), ha='center', va='center',
                      fontsize=S.FS_TICK + 0.4, fontweight='bold' if SIG[i, j] else 'normal',
                      color=S.INK if SIG[i, j] else S.GUIDE)
ax_c.set_xticks(range(len(cs)))
ax_c.set_xticklabels(cs)
ax_c.xaxis.tick_top()
ax_c.set_yticks(range(len(VARS)))
ax_c.set_yticklabels([v[1] for v in VARS])
ax_c.tick_params(length=0)
for sp in ax_c.spines.values():
    sp.set_visible(False)
ax_c.set_xticks(np.arange(-0.5, len(cs)), minor=True)
ax_c.set_yticks(np.arange(-0.5, len(VARS)), minor=True)
ax_c.grid(which='minor', color='white', lw=2.0)
ax_c.tick_params(which='minor', length=0)
ax_c.set_xlabel('correlation with the\neddy kinetic energy')
S.panel(ax_c, 'c', dx=-0.02, dy=1.06)

# ------------------------------------------------------------- (d) and (e) wavelet coherence
if wv is not None:
    t = pd.DatetimeIndex(wv.time.values)
    per = wv.period.values
    coi = wv.coi.values

    def wavelet_panel(a, field, level, cmap, vmin, vmax):
        m = a.pcolormesh(t, per, field, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
        a.contour(t, per, level, levels=[1.0], colors=S.INK, linewidths=0.9)
        a.fill_between(t, coi, per.max() * 1.2, facecolor='white', alpha=0.45, edgecolor='none', lw=0)
        a.fill_between(t, coi, per.max() * 1.2, facecolor='none', edgecolor='dimgray', hatch='xx', lw=0)
        a.plot(t, coi, color=S.INK, lw=0.8)
        a.set_yscale('log', base=2)
        a.set_ylim(min(per.max(), 64.0), per.min())
        a.yaxis.set_major_locator(FixedLocator(PERIOD_TICKS))
        a.yaxis.set_major_formatter(FixedFormatter([str(p) for p in PERIOD_TICKS]))
        a.yaxis.set_minor_locator(NullLocator())
        a.xaxis.set_major_locator(mdates.MonthLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        a.set_xlabel(f'{t[0].year}' if t[0].year == t[-1].year else f'{t[0].year} and {t[-1].year}')
        return m

    mc = None
    for k, (a, name, title) in enumerate(((ax_w[0], 'wave', 'uplift and coastal trapped wave'),
                                          (ax_w[1], 'eke', 'uplift and eddy kinetic energy'))):
        if f'coh_{name}' not in wv:
            a.set_axis_off()
            continue
        R2 = wv[f'coh_{name}'].values
        mc = wavelet_panel(a, R2, R2 / wv[f'level_{name}'].values[:, None], CMAP_COH, 0.0, 1.0)
        a.set_title(title, fontsize=S.FS)
        if k:
            plt.setp(a.get_yticklabels(), visible=False)
    ax_w[0].set_ylabel('period (days)')
    p0, p1 = (a.get_position() for a in ax_w)
    if mc is not None:
        cax_c = fig.add_axes([p0.x0 + 0.25 * p0.width, p0.y0 - 0.085, p1.x1 - p0.x0 - 0.5 * p0.width, 0.013])
        cb = fig.colorbar(mc, cax=cax_c, orientation='horizontal')
        cb.set_label('wavelet coherence')
for a, letter in zip(ax_w, 'de'):
    S.panel(a, letter, dy=1.03)

# ------------------------------------------------------------------ legend of (b)
grey = S.LIGHT_GUIDE
ax_b.legend([S.patch_handle(grey, ec='black'), S.patch_handle('white', ec=grey, hatch=S.DOWN_HATCH)],
            ['upslope', 'downslope'], loc='lower left', bbox_to_anchor=(-0.04, 1.08), ncol=2, handlelength=1.6,
            title='Ekman transport', title_fontsize=S.FS_LEG, alignment='left')

fig.savefig(V.DIR_FIG / 'fig06_robustness.pdf')
S.save(fig, V.DIR_FIG / 'fig06_robustness.png')
