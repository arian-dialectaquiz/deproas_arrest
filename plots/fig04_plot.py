"""
Figure 4. Buoyancy of the bottom layer and the time of the arrest.

(a) distribution of the daily squared buoyancy frequency of the layer between 155 and
    194 m at the 200 m moorings, on the days of upslope Ekman transport (left, filled)
    and of downslope transport (right, hatched), as the share of days in bins of
    0.5 x 10^-5 s^-2, with the median (black tick). The bin beneath zero gathers the
    days on which the layer is unstable or mixed, and its share is written beneath it.
(b) and (c) change of the temperature difference across the layer since each reversal
    of the flow at 155 m at FUB3 and FCF3, against the time since the reversal over the
    median canonical arrest time of the site, the median over the states (symbols) with its 95 % interval
    from resampling whole states (shading), and the same change in the layer model
    (black lines).
(d) and (e) attenuation of the near-bottom flow at FUB3 corrected to an interior speed
    of 0.15 m s-1, with the offset flow of Figure 3 removed, and its veering, from the
    hours with an interior speed between 0.10 and 0.30 m s-1, against the same scaled time, with the 95 % interval, the exponential decay fitted to the
    attenuation (lines in (d)), the decay that a full arrest over one canonical arrest
    time would give (dotted) and the unarrested Ekman layer (khaki).
The symbols along the upper edge of (b) to (e) mark, on the same scaled time, the
observed arrest time (the e-folding time of the decay in (d)), the wedge time, the
restoring time of the layer model and the median duration of the states of the site. The
hatched hours lie within one cutoff (40 h) of the low-pass filter from the reversal, where
the filtered records still carry the state before it. The intervals of (b) and (c) are
clipped to the range of the medians.

Reads paper_v3/tables/fig04_*.csv, fig02_summary.csv, fig02_daily.csv and
fig03_reference.csv. Writes paper_v3/figures/fig04_buoyancy.png and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import NullFormatter

import config_v3 as V
import style as S

FIGSIZE = (7.48, 7.9)
TIME_SITE = 'FUB3'
DRIFT_SITES = ('FUB3', 'FCF3')
N2_SCALE = 1e-5                     # s-2 of the axis of (a)
N2_STEP = 1.0                       # bin width in units of N2_SCALE
MIXED_BIN = (-1.7, -0.7)            # where the bin of the unstable or mixed days is drawn
HALF_WIDTH = 0.44                   # widest bin of all, in site spacings
# time scales marked along the upper edge of (b) to (e), on the scaled time
MARKS = (('arrest', 'D', 'black', 'observed arrest time'),
         ('wedge', 's', 'white', 'wedge time'),
         ('restoring', '^', 'darkgray', 'restoring time'),
         ('state', 'v', 'black', 'median state duration'))


def read(name, **kw):
    p = V.DIR_TAB / f'{name}.csv'
    if p.exists() and p.stat().st_size > 5:
        return pd.read_csv(p, **kw)
    print(f'  {p.name} is missing, its elements are left out')
    return pd.DataFrame()


def time_axis(ax, label=True):
    ax.set_xscale('log')
    ax.set_xlim(10 ** V.TIME_EDGES[0], 10 ** V.TIME_EDGES[1])
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(['0.1', '1', '10', '100'])
    ax.xaxis.set_minor_formatter(NullFormatter())
    if label:
        ax.set_xlabel('time since the reversal over the\nmedian canonical arrest time')


mix = read('fig04_mixing')
days4 = read('fig04_daily', parse_dates=['day'])
drift = read('fig04_drift')
mod = read('fig04_model')
summ2 = read('fig02_summary')
dec = read('fig04_decay')
tim = read('fig04_time')
ref = read('fig03_reference')
mod_i = mod[mod.forcing == 'interior'] if len(mod) and 'forcing' in mod else mod
if days4.empty:
    # the days classified by the sign of the daily interior flow until fig04_compute writes its table
    d2 = read('fig02_daily', parse_dates=['day'])
    if len(d2):
        print('  fig04_daily.csv missing, the days of (a) are classified by the daily interior flow')
        d2 = (d2[d2.layer == 'lowest pair'] if 'layer' in d2 else d2).copy()
        d2['kind'] = np.where(d2.u_i <= -V.MIN_SPEED, S.UP, np.where(d2.u_i >= V.MIN_SPEED, S.DOWN, ''))
        days4 = d2[['site', 'day', 'kind', 'N']]
n2_sites = [s for s in V.SITES_200 if len(days4) and s in set(days4.site)]


def tau_of(site):
    """Median canonical arrest time of the site (h), the unit of the scaled time."""
    r = summ2[(summ2.site == site) & (summ2.half == 'all')] if len(summ2) else pd.DataFrame()
    q = dec[(dec.site == site) & (dec.kind == S.UP)] if len(dec) else pd.DataFrame()
    if len(q) and np.isfinite(q.tau_median_h.iloc[0]):
        return float(q.tau_median_h.iloc[0])
    return float(r.tau_c_median_h.iloc[0]) if len(r) else np.nan


def scales(site):
    """Time scales of a site over its median canonical arrest time."""
    out = {}
    r = summ2[(summ2.site == site) & (summ2.half == 'all')] if len(summ2) else pd.DataFrame()
    q = dec[(dec.site == site) & (dec.kind == S.UP)] if len(dec) else pd.DataFrame()
    tau = tau_of(site)
    if not np.isfinite(tau) or tau <= 0:
        return out
    if len(q) and np.isfinite(q.x_e.iloc[0]):
        out['arrest'] = float(q.x_e.iloc[0])
    if len(r):
        out['wedge'] = float(r.tau_w_median_h.iloc[0]) / tau
        if 'state_median_h' in r:
            out['state'] = float(r.state_median_h.iloc[0]) / tau
    m_ = mod_i[mod_i.site == site] if len(mod_i) else pd.DataFrame()
    if len(m_):
        out['restoring'] = float(m_.tau_h.iloc[0]) / tau
    return out


def filter_memory(ax, site):
    """Hatch the hours that lie within one cutoff of the low-pass filter from the reversal, where
    the filtered velocity and temperature still carry the state before it."""
    tau = tau_of(site)
    if np.isfinite(tau) and tau > 0:
        ax.axvspan(10 ** V.TIME_EDGES[0], V.LP_CUTOFF_H / tau, facecolor='none', edgecolor='lightgray',
                   hatch='\\\\', lw=0, zorder=0.6)


def mark_scales(ax, site, keys=('arrest', 'wedge', 'restoring', 'state')):
    """Time scales of the site along the upper edge, raised one above the other where two
    lie within an eighth of a decade."""
    filter_memory(ax, site)
    sc = scales(site)
    placed = []
    for key, mk, fc, _ in sorted(MARKS, key=lambda m: sc.get(m[0], np.inf)):
        if key not in keys or key not in sc or not (10 ** V.TIME_EDGES[0] < sc[key] < 10 ** V.TIME_EDGES[1]):
            continue
        x = sc[key]
        level = 0
        while any(abs(np.log10(x / xp)) < 0.125 and lv == level for xp, lv in placed):
            level += 1
        placed.append((x, level))
        ax.axvline(x, color='silver', lw=0.8, zorder=0.5)
        ax.plot(x, 1.0 + 0.075 * level, marker=mk, ms=7.0, mfc=fc, mec='black', mew=0.8, ls='none', clip_on=False,
                transform=ax.get_xaxis_transform(), zorder=10)


# ----------------------------------------------------------------- layout
fig = plt.figure(figsize=FIGSIZE)
gs_top = gridspec.GridSpec(1, 3, figure=fig, left=0.085, right=0.985, top=0.94, bottom=0.615,
                           width_ratios=[1.3, 1.0, 1.0], wspace=0.30)
gs_bot = gridspec.GridSpec(1, 2, figure=fig, left=0.085, right=0.985, top=0.47, bottom=0.185, wspace=0.26)
ax_n = fig.add_subplot(gs_top[0, 0])
ax_dt = [fig.add_subplot(gs_top[0, 1]), fig.add_subplot(gs_top[0, 2])]
ax_g = fig.add_subplot(gs_bot[0, 0])
ax_p = fig.add_subplot(gs_bot[0, 1], sharex=ax_g)

# ------------------------------------------------- (a) distribution of N2 of the layer
groups, colours, shares = {}, {}, {}
for i, site in enumerate(n2_sites):
    for state in (S.UP, S.DOWN):
        h = days4[(days4.site == site) & (days4.kind == state)]
        if len(h) < 10:
            continue
        groups[(i, state)] = h.N.values.astype(float) ** 2 / N2_SCALE      # NaN where N2 <= 0
        colours[(i, state)] = S.SITE_COLOURS[site]
        r = mix[(mix.site == site) & (mix.kind == state)] if len(mix) else pd.DataFrame()
        if len(r):
            shares[(i, state)] = float(r.unstable.iloc[0])
alln2 = np.concatenate([v[np.isfinite(v)] for v in groups.values()]) if groups else np.array([1.0])
n2_max = N2_STEP * np.ceil(np.percentile(alln2, 98) / N2_STEP)
edges = np.arange(0.0, n2_max + 1e-9, N2_STEP)
S.split_histograms(ax_n, groups, edges, colours, half_width=HALF_WIDTH, mixed_bin=MIXED_BIN, shares=shares)
ax_n.axhline(0.0, color=S.INK, lw=0.8, zorder=3)
ax_n.set_xticks(range(len(n2_sites)))
ax_n.set_xticklabels(n2_sites)
ax_n.set_xlim(-0.55, len(n2_sites) - 0.45)
ax_n.set_ylim(MIXED_BIN[0] - 1.3, n2_max)
yt = [t for t in range(0, int(n2_max) + 1, 2)]
ax_n.set_yticks([0.5 * sum(MIXED_BIN)] + yt)
ax_n.set_yticklabels(['$\\leq$0'] + [str(t) for t in yt])
ax_n.tick_params(axis='x', length=0)
S.tidy(ax_n, axis='y')
ax_n.set_ylabel('N$^2$ of the layer (10$^{-5}$ s$^{-2}$)')
S.panel(ax_n, 'a')

# -------------------------------------------- (b) and (c) drift across the layer in time
for k, (ax, site) in enumerate(zip(ax_dt, DRIFT_SITES)):
    col = S.SITE_COLOURS[site]
    for state in (S.UP, S.DOWN):
        g = drift[(drift.site == site) & (drift.kind == state)].sort_values('x') if len(drift) else pd.DataFrame()
        if g.empty:
            continue
        filled = S.STATE_FILLED[state]
        ax.fill_between(g.x, g.dT_lo, g.dT_hi, color=col, alpha=S.BAND_ALPHA if filled else S.BAND_ALPHA_LIGHT,
                        lw=0, zorder=2)
        if 'dT_model' in g and g.dT_model.notna().any():
            ax.plot(g.x, g.dT_model, color=S.INK, ls=S.STATE_LS[state], lw=1.3, zorder=3)
        ax.plot(g.x, g.dT, color=col, ls='none', marker=S.SITE_MARKERS[site], ms=6.0,
                mfc=col if filled else 'white', mec='black' if filled else col, mew=0.5 if filled else 1.3, zorder=4)
    ax.axhline(0.0, color=S.INK, lw=0.6, zorder=1)
    g = drift[drift.site == site] if len(drift) else pd.DataFrame()
    if len(g):
        lim = 1.6 * np.nanmax(np.abs(np.r_[g.dT.values, g.dT_model.values]))
        ax.set_ylim(-lim, lim)
    time_axis(ax)
    S.tidy(ax, axis='both')
    mark_scales(ax, site, keys=('wedge', 'restoring', 'state'))
    ax.set_ylabel('change across the layer (°C)' if k == 0 else '')
    ax.set_title(site, fontweight='bold', pad=22.0)
    S.panel(ax, 'bc'[k], dy=1.06)

# --------------------------------------- (d) and (e) the near-bottom flow in time
xx = np.logspace(*V.TIME_EDGES[:2], 100)
col = S.SITE_COLOURS[TIME_SITE]
rr = ref[(ref.site == TIME_SITE) & (ref.kind == 'unarrested')] if len(ref) else pd.DataFrame()
ar = ref[(ref.site == TIME_SITE) & (ref.kind == 'arrested')] if len(ref) else pd.DataFrame()
for ax, key, lab_y in ((ax_g, 'gain', 'attenuation'), (ax_p, 'veer', 'veering (°)')):
    if len(rr):
        lo_, hi_ = rr[key].min(), rr[key].max()
        ax.axhspan(lo_, min(hi_, 1.2) if key == 'gain' else hi_, color=S.UNARRESTED, lw=0, zorder=0)
    for state in (S.UP, S.DOWN):
        g = tim[(tim.site == TIME_SITE) & (tim.kind == state)].sort_values('x') if len(tim) else pd.DataFrame()
        if g.empty:
            continue
        filled = S.STATE_FILLED[state]
        ax.fill_between(g.x, g[f'{key}_lo'], g[f'{key}_hi'], color=col,
                        alpha=S.BAND_ALPHA if filled else S.BAND_ALPHA_LIGHT, lw=0, zorder=2)
        ax.plot(g.x, g[key], color=col, ls='none' if key == 'gain' else S.STATE_LS[state], lw=1.2,
                marker=S.SITE_MARKERS[TIME_SITE], ms=6.0, mfc=col if filled else 'white',
                mec='black' if filled else col, mew=0.5 if filled else 1.3, zorder=4)
    if key == 'gain' and len(dec):
        for state in (S.UP, S.DOWN):
            q = dec[(dec.site == TIME_SITE) & (dec.kind == state)]
            if len(q) and np.isfinite(q.x_e.iloc[0]):
                q = q.iloc[0]
                ax.plot(xx, q.g_inf + (q.g_0 - q.g_inf) * np.exp(-xx / q.x_e), color=col, ls=S.STATE_LS[state],
                        lw=1.6, zorder=3)
    if key == 'gain' and len(rr) and len(ar):
        g_u = float(rr.gain.median())
        g_a = float(min(1.0, ar.cap.mean() / V.U_REF))
        ax.plot(xx, g_a + (g_u - g_a) * np.exp(-xx), color=S.ARRESTED, ls=':', lw=1.6, zorder=3)
        ax.set_ylim(0.0, 1.2)
    if key == 'veer':
        if len(rr):
            ax.plot(xx, float(rr.veer.median()) * np.exp(-xx), color=S.ARRESTED, ls=':', lw=1.6, zorder=3)
        ax.axhline(0.0, color=S.INK, lw=0.6, zorder=1)
    time_axis(ax)
    S.tidy(ax, axis='both')
    mark_scales(ax, TIME_SITE)
    ax.set_ylabel(f'{lab_y} at {V.U_REF:.2f} m s$^{{-1}}$')
    ax.set_title(TIME_SITE, fontweight='bold', pad=22.0)
S.panel(ax_g, 'd', dy=1.06)
S.panel(ax_p, 'e', dy=1.06)

# ------------------------------------------------------------------ legend
grey = S.LIGHT_GUIDE
h = [(S.patch_handle(grey), S.marker_handle(grey, 'o', ms=6.0)),
     (S.patch_handle('white', ec=grey, hatch=S.DOWN_HATCH), S.marker_handle(grey, 'o', ms=6.0, open_=True)),
     S.patch_handle(grey, alpha=0.45), S.patch_handle(S.UNARRESTED),
     S.line_handle(S.INK, '-', lw=1.3), S.line_handle(grey, '-', lw=1.6), S.line_handle(S.ARRESTED, ':', lw=1.6)]
h.append(S.patch_handle('white', ec='lightgray', hatch='\\\\'))
lab = ['upslope Ekman transport', 'downslope Ekman transport', '95 % interval', 'unarrested Ekman layer',
       'layer model (b, c)', 'fitted decay (d)', 'full arrest over one arrest time',
       f'within {V.LP_CUTOFF_H:.0f} h of the reversal']
for _, mk, fc, name in MARKS:
    hh = S.marker_handle(fc, mk, ms=6.5)
    hh.set_markeredgecolor('black')
    hh.set_markeredgewidth(0.8)
    h.append(hh)
    lab.append(name)
fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, columnspacing=2.6, handlelength=2.0,
           handletextpad=0.5, handler_map={tuple: HandlerTuple(ndivide=None)})

fig.savefig(V.DIR_FIG / 'fig04_buoyancy.pdf')
S.save(fig, V.DIR_FIG / 'fig04_buoyancy.png')
