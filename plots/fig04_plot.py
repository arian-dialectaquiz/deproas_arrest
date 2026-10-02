"""
Figure 4. Buoyancy of the bottom layer and the time of the arrest.

(a) buoyancy frequency of the layer between 155 and 194 m on the days with the
    current (filled, upslope Ekman transport) and against it (hatched,
    downslope), with the quartiles as the box, the fifth to the ninety-fifth
    percentile as the whisker and the median as the line, and the share of days
    on which the layer is unstable or mixed beneath each site.
(b) change of the temperature difference across the layer since each reversal of
    the flow at 155 m, against the time since the reversal over the arrest time,
    with the median over the states and its 95 % interval (symbols) and the same
    change in the layer model (line without symbols).
(c) time scales of the arrest at each mooring, the canonical arrest time
    (|f| s^2)^-1 and the wedge time 1 / (2 Cd N s^3) of Figure 2 (median and
    quartiles or the range of Cd), the e-folding time of the decay of the
    attenuation with the current in (d), the restoring time of the layer model
    and the time the interior flow keeps one sign, each with its interval (upper),
    and the share of the unarrested Ekman transport that builds the buoyancy of
    the layer, the arrested thickness over the thickness C_d / c of the layer
    model, for the transport upslope (filled) and downslope (open) with its 95 %
    interval (lower).
(d) and (e) attenuation of the near-bottom flow at FUB3 corrected to an interior
    speed of 0.15 m s-1, with the offset flow of Figure 3 removed, and its
    veering, against the time since the reversal over the arrest time, with the
    95 % interval from resampling whole states, the exponential decay fitted to
    the attenuation and the decay that the arrest predicts over one arrest time.

Reads paper_v3/tables/fig04_*.csv and fig03_reference.csv.
Writes paper_v3/figures/fig04_buoyancy.png and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter

import config_v3 as V
import style as S

FIGSIZE = (7.48, 7.4)
LEG = S.LEGEND_SIZE
KINDS = (('with the current', True, '-'), ('against the current', False, '--'))
TIME_SITE = 'FUB3'
DRIFT_SITES = ('FUB3', 'FCF3')


def read(name, **kw):
    p = V.DIR_TAB / f'{name}.csv'
    if p.exists() and p.stat().st_size > 5:
        return pd.read_csv(p, **kw)
    print(f'  {p.name} is missing, its elements are left out')
    return pd.DataFrame()


def time_axis(ax):
    ax.set_xscale('log')
    ax.set_xlim(10 ** V.TIME_EDGES[0], 10 ** V.TIME_EDGES[1])
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(['0.1', '1', '10', '100'])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('time since the reversal over the arrest time')


mix = read('fig04_mixing')
drift = read('fig04_drift')
mod = read('fig04_model')
summ2 = read('fig02_summary')
dec = read('fig04_decay')
tim = read('fig04_time')
ref = read('fig03_reference')
bins3 = read('fig03_bins')
mix_sites = [s for s in V.SITES_200 if len(mix) and s in set(mix.site)]
mod_i = mod[mod.forcing == 'interior'] if len(mod) and 'forcing' in mod else mod

# ----------------------------------------------------------------- layout
fig = plt.figure(figsize=FIGSIZE)
gs_top = gridspec.GridSpec(1, 3, figure=fig, left=0.08, right=0.985, top=0.905, bottom=0.575,
                           width_ratios=[0.95, 1.0, 1.15], wspace=0.40)
gs_bot = gridspec.GridSpec(2, 2, figure=fig, left=0.08, right=0.985, top=0.445, bottom=0.03,
                           height_ratios=[1.0, 0.14], wspace=0.28, hspace=0.62)
ax_n = fig.add_subplot(gs_top[0, 0])
ax_dt = fig.add_subplot(gs_top[0, 1])
gs_c = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_top[0, 2], hspace=0.55,
                                        height_ratios=[1.25, 1.0])
ax_bs = [fig.add_subplot(gs_c[0, 0]), fig.add_subplot(gs_c[1, 0])]
ax_g = fig.add_subplot(gs_bot[0, 0])
ax_p = fig.add_subplot(gs_bot[0, 1], sharex=ax_g)
ax_leg = fig.add_subplot(gs_bot[1, :])
ax_leg.axis('off')

# ---------------------------------------------- (a) stratification of the layer
W = 0.30
ticks = []
for i, site in enumerate(mix_sites):
    parts = []
    for j, (kind, filled, _) in enumerate(KINDS):
        r = mix[(mix.site == site) & (mix.kind == kind)]
        if r.empty:
            continue
        r = r.iloc[0]
        x = i + (j - 0.5) * (W + 0.06)
        sc = 1e3
        ax_n.plot([x, x], [r.N_p5 * sc, r.N_p95 * sc], color=S.INK, lw=0.7, zorder=2)
        ax_n.add_patch(Rectangle((x - W / 2, r.N_q1 * sc), W, (r.N_q3 - r.N_q1) * sc, zorder=3,
                                 fc=S.SITE_COLOURS[site] if filled else 'white',
                                 ec='black' if filled else S.SITE_COLOURS[site],
                                 hatch=None if filled else '////', lw=0.8))
        ax_n.plot([x - W / 2, x + W / 2], [r.N_median * sc] * 2, color=S.INK, lw=1.3, zorder=4)
        parts.append(f'{r.unstable:.0f}')
    ticks.append(f'{site}\n' + ' and '.join(parts) + ' %' if parts else site)
ax_n.set_xticks(range(len(mix_sites)))
ax_n.set_xticklabels(ticks, fontsize=6.0)
ax_n.set_xlim(-0.6, len(mix_sites) - 0.4)
ax_n.set_ylim(bottom=0.0)
S.tidy(ax_n, axis='y')
ax_n.set_ylabel('N of the layer (10$^{-3}$ s$^{-1}$)')
S.panel(ax_n, 'a')
ax_n.legend([S.patch_handle(S.GUIDE, ec='black'), S.patch_handle('white', ec=S.GUIDE, hatch='////')],
            ['with the current', 'against the current'], ncol=1, loc='lower left',
            bbox_to_anchor=(0.0, 1.01), fontsize=LEG - 0.3,
            title='unstable or mixed days beneath', title_fontsize=LEG - 0.5, alignment='left')

# -------------------------------------------- (b) drift across the layer in time
for site in [s for s in DRIFT_SITES if len(drift) and s in set(drift.site)]:
    for kind, filled, ls in KINDS:
        g = drift[(drift.site == site) & (drift.kind == kind)].sort_values('x')
        if g.empty:
            continue
        ax_dt.fill_between(g.x, g.dT_lo, g.dT_hi, color=S.SITE_COLOURS[site], alpha=0.15, lw=0, zorder=2)
        ax_dt.plot(g.x, g.dT, color=S.SITE_COLOURS[site], ls='none', marker=S.SITE_MARKERS[site],
                   ms=4.0, mfc=S.SITE_COLOURS[site] if filled else 'white', mec='black', mew=0.4, zorder=4)
        if 'dT_model' in g and g.dT_model.notna().any():
            ax_dt.plot(g.x, g.dT_model, color=S.SITE_COLOURS[site], ls=ls, lw=1.2, zorder=3)
ax_dt.axhline(0.0, color=S.INK, lw=0.6, zorder=1)
time_axis(ax_dt)
S.tidy(ax_dt, axis='both')
ax_dt.set_ylabel('change across the layer (°C)')
S.panel(ax_dt, 'b')

# ------------------------------------------------------- (c) the time scales
ax_ts, ax_ef = ax_bs
TS = (('canonical', 'o', S.INK, 'black', 'canonical'),
      ('wedge', 's', 'white', 'black', 'wedge'),
      ('decay', 'D', None, 'black', 'velocity decay'),
      ('restoring', '^', None, 'black', 'buoyancy restoring'),
      ('state', 'v', S.RETAINED_BAR, 'black', 'one sign'))
ts_sites = [s_ for s_ in V.SITES_200 if len(summ2) and s_ in set(summ2.site)]
allx = []
for i, site in enumerate(ts_sites):
    r = summ2[(summ2.site == site) & (summ2.half == 'all')]
    if r.empty:
        continue
    r = r.iloc[0]
    vals = {'canonical': (r.tau_c_median_h, r.tau_c_q1_h, r.tau_c_q3_h),
            'wedge': (r.tau_w_median_h, r.tau_w_lo_h, r.tau_w_hi_h),
            'state': (r.get('state_median_h', np.nan), r.get('state_q1_h', np.nan), r.get('state_q3_h', np.nan))}
    q = dec[(dec.site == site) & (dec.kind == 'with the current')] if len(dec) else pd.DataFrame()
    if len(q) and np.isfinite(q.t_e_h.iloc[0]):
        q = q.iloc[0]
        vals['decay'] = (q.t_e_h, q.x_e_lo * q.tau_median_h, q.x_e_hi * q.tau_median_h)
    m_ = mod_i[mod_i.site == site] if len(mod_i) else pd.DataFrame()
    if len(m_):
        m_ = m_.iloc[0]
        vals['restoring'] = (m_.tau_h, m_.tau_lo, m_.tau_hi)
    for j, (key, mk, fc, ec, _) in enumerate(TS):
        if key not in vals or not np.isfinite(vals[key][0]):
            continue
        med, lo, hi = vals[key]
        y = i + (j - 2) * 0.17
        ax_ts.plot([lo, hi], [y, y], color=S.INK, lw=0.8, zorder=3)
        ax_ts.plot(med, y, marker=mk, ms=4.4, mfc=S.SITE_COLOURS[site] if fc is None else fc, mec=ec, mew=0.6,
                   ls='none', zorder=4)
        allx += [lo, hi, med]
allx = np.asarray([v for v in allx if np.isfinite(v) and v > 0])
ax_ts.set_xscale('log')
if allx.size:
    ax_ts.set_xlim(max(0.3, 0.6 * allx.min()), min(3000.0, 1.6 * allx.max()))
xt = [t for t in (1, 3, 10, 30, 100, 300, 1000) if ax_ts.get_xlim()[0] <= t <= ax_ts.get_xlim()[1]]
ax_ts.set_xticks(xt)
ax_ts.set_xticklabels([f'{t:g}' for t in xt])
ax_ts.xaxis.set_minor_formatter(NullFormatter())
ax_ts.set_yticks(range(len(ts_sites)))
ax_ts.set_yticklabels(ts_sites)
ax_ts.set_ylim(len(ts_sites) - 0.5, -0.5)
S.tidy(ax_ts, axis='x')
ax_ts.set_xlabel('time (h)')
S.panel(ax_ts, 'c', dy=1.02)
ax_ts.legend([S.marker_handle(S.GUIDE if fc is None else fc, mk, ms=4.4, open_=(fc == 'white'))
              for _, mk, fc, _, _ in TS], [t[4] for t in TS], loc='lower left', bbox_to_anchor=(0.0, 1.04),
             ncol=2, fontsize=LEG - 0.9, handlelength=1.0, columnspacing=0.8, handletextpad=0.3)

# share of the unarrested Ekman transport that builds the buoyancy of the layer
ef_sites = [s_ for s_ in V.SITES_200 if len(mod_i) and s_ in set(mod_i.site)]
for i, site in enumerate(ef_sites):
    m_ = mod_i[mod_i.site == site].iloc[0]
    for j, (key, filled) in enumerate((('up', True), ('dn', False))):
        x = i + (j - 0.5) * 0.28
        val, lo, hi = m_[f'eff_{key}'], m_[f'eff_{key}_lo'], m_[f'eff_{key}_hi']
        if not np.isfinite(val) or val <= 0:
            continue
        ax_ef.plot([x, x], [max(lo, 1e-3), hi], color=S.INK, lw=0.8, zorder=3)
        ax_ef.plot(x, val, marker=S.SITE_MARKERS[site], ms=5.0, mfc=S.SITE_COLOURS[site] if filled else 'white',
                   mec='black' if filled else S.SITE_COLOURS[site], mew=0.8, ls='none', zorder=4)
ax_ef.axhline(1.0, color=S.GUIDE, ls=':', lw=1.0, zorder=1)
ax_ef.set_yscale('log')
ax_ef.set_ylim(5e-3, 2.0)
ax_ef.set_yticks([0.01, 0.1, 1.0])
ax_ef.set_yticklabels(['0.01', '0.1', '1'])
ax_ef.yaxis.set_minor_formatter(NullFormatter())
ax_ef.set_xticks(range(len(ef_sites)))
ax_ef.set_xticklabels(ef_sites)
ax_ef.set_xlim(-0.6, len(ef_sites) - 0.4)
S.tidy(ax_ef, axis='y')
ax_ef.set_ylabel('share of the\nEkman transport', fontsize=6.6)


# --------------------------------------- (d) and (e) the near-bottom flow in time
xx = np.logspace(*V.TIME_EDGES[:2], 100)
rr = ref[(ref.site == TIME_SITE) & (ref.kind == 'unarrested')] if len(ref) else pd.DataFrame()
ar = ref[(ref.site == TIME_SITE) & (ref.kind == 'arrested')] if len(ref) else pd.DataFrame()
g_up = tim[(tim.site == TIME_SITE) & (tim.kind == 'with the current')] if len(tim) else pd.DataFrame()
for ax, key, lab_y in ((ax_g, 'gain', 'attenuation'), (ax_p, 'veer', 'veering (°)')):
    if key == 'gain' and len(rr):
        ax.axhspan(rr.gain.min(), min(rr.gain.max(), 1.1), color=S.UNARRESTED, lw=0, zorder=0)
    if key == 'veer' and len(rr):
        ax.axhspan(rr.veer.min(), rr.veer.max(), color=S.UNARRESTED, lw=0, zorder=0)
    for kind, filled, ls in KINDS:
        g = tim[(tim.site == TIME_SITE) & (tim.kind == kind)].sort_values('x') if len(tim) else pd.DataFrame()
        if g.empty:
            continue
        ax.fill_between(g.x, g[f'{key}_lo'], g[f'{key}_hi'], color=S.SITE_COLOURS[TIME_SITE],
                        alpha=0.15 if filled else 0.08, lw=0, zorder=2)
        ax.plot(g.x, g[key], color=S.SITE_COLOURS[TIME_SITE], ls='none' if key == 'gain' else ls, lw=1.2,
                marker=S.SITE_MARKERS[TIME_SITE], ms=4.2, mfc=S.SITE_COLOURS[TIME_SITE] if filled else 'white',
                mec='black', mew=0.45, zorder=4)
    if key == 'gain' and len(dec):
        for kind, filled, ls in KINDS:
            q = dec[(dec.site == TIME_SITE) & (dec.kind == kind)]
            if len(q) and np.isfinite(q.x_e.iloc[0]):
                q = q.iloc[0]
                ax.plot(xx, q.g_inf + (q.g_0 - q.g_inf) * np.exp(-xx / q.x_e), color=S.SITE_COLOURS[TIME_SITE],
                        ls=ls, lw=0.8, alpha=0.7, zorder=3)
    if key == 'gain' and len(rr) and len(ar):
        g_u = float(rr.gain.median())
        g_a = float(min(1.0, ar.cap.mean() / V.U_REF))
        ax.plot(xx, g_a + (g_u - g_a) * np.exp(-xx), color=S.ARRESTED, ls=':', lw=1.4, zorder=3)
        ax.set_ylim(0.0, 1.15)
    if key == 'veer':
        if len(rr):
            ax.plot(xx, float(rr.veer.median()) * np.exp(-xx), color=S.ARRESTED, ls=':', lw=1.4, zorder=3)
        ax.axhline(0.0, color=S.INK, lw=0.5, zorder=1)
    time_axis(ax)
    S.tidy(ax, axis='both')
    ax.set_ylabel(lab_y)
    ax.set_title(f'{TIME_SITE} at an interior speed of {V.U_REF:.2f} m s$^{{-1}}$', fontsize=7.0, loc='left')
S.panel(ax_g, 'd', dy=1.06)
S.panel(ax_p, 'e', dy=1.06)
h = [S.line_handle(S.SITE_COLOURS[s], S.SITE_LINESTYLES[s], marker=S.SITE_MARKERS[s], ms=4.2)
     for s in DRIFT_SITES if len(drift) and s in set(drift.site)]
lab = [s for s in DRIFT_SITES if len(drift) and s in set(drift.site)]
h += [S.line_handle(S.GUIDE, '-', marker='o', ms=4.2), S.line_handle(S.GUIDE, '--', marker='o', ms=4.2, mfc='white'),
      S.line_handle(S.GUIDE, '-', lw=1.2), S.patch_handle(S.UNARRESTED), S.line_handle(S.ARRESTED, ':', lw=1.4)]
lab += ['with the current', 'against the current', 'layer model or fitted decay', 'unarrested Ekman layer',
        'arrest over one arrest time']
ax_leg.legend(h, lab, loc='center', ncol=4, fontsize=LEG - 0.4, borderaxespad=0.0, columnspacing=1.1,
              handlelength=2.2)

fig.savefig(V.DIR_FIG / 'fig04_buoyancy.pdf')
S.save(fig, V.DIR_FIG / 'fig04_buoyancy.png')