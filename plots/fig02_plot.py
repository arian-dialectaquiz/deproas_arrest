"""
Figure 2. Regime of the shelf break in space, in season and at the moorings.

(a) slope Burger number over the slope of the northern South Brazil Bight, from
    the buoyancy frequency over the 50 m above the bottom of the regional
    climatology averaged over the four seasons and the GEBCO bottom slope, with
    the unit contour, the 200 m isobath followed in (c) and (d) with its
    distance from Cabo Frio, and the 200 m moorings.
(b) daily slope Burger number at the 200 m moorings from the moored temperature
    at the two lowest levels, for October to March (ONDJFM, filled) and April to
    September (AMJJAS, hatched), with the quartiles as the box, the fifth to the
    ninety-fifth percentile as the whisker and the median as the line, and the
    same quantity from the climatology (the GLORYS12V1 comparison is in Table S1).
(c) slope Burger number along the 200 m isobath, the mean of the four seasons
    (line) and their range (shading), with the moored median and quartiles at
    the position of each mooring.
(d) bottom slope and buoyancy frequency along the same path, each over its
    geometric mean along the path, with the share of the variance of log s
    that each carries.
(e) monthly median (line) and interquartile range (shading) of the moored slope
    Burger number, October to March shaded.
(f) canonical arrest time and wedge time at each mooring, with the duration of the
    states of the interior flow, which is the time the layer is given, each as a bar
    over its interquartile range (the range of C_d for the wedge time) with its median.

Reads paper_v3/data/fig02_map.nc and paper_v3/tables/fig02_*.csv.
Writes paper_v3/figures/fig02_regime.png and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, NullFormatter
from matplotlib.legend_handler import HandlerTuple

import config_v3 as V
import inputs_v3 as I
import style as S

FIGSIZE = (7.48, 9.4)
HALO = [pe.withStroke(linewidth=1.8, foreground='white')]
LEG = S.LEGEND_SIZE
S_TICKS = (0.2, 0.5, 1.0, 2.0, 5.0)
PATH_MARK_KM = 200.0
MONTHS = 'JFMAMJJASOND'


def read(name, **kw):
    p = V.DIR_TAB / f'{name}.csv'
    if p.exists() and p.stat().st_size > 5:
        return pd.read_csv(p, **kw)
    print(f'  {p.name} is missing, its elements are left out')
    return pd.DataFrame()


def aspect(extent):
    lon0, lon1, lat0, lat1 = extent
    return (lon1 - lon0) * np.cos(np.deg2rad(0.5 * (lat0 + lat1))) / (lat1 - lat0)


def log_s_axis(ax, lim):
    ax.set_yscale('log')
    ax.set_ylim(*lim)
    ticks = [t for t in S_TICKS if lim[0] <= t <= lim[1]]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:g}' for t in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(1.0, color=S.INK, lw=0.7, zorder=1)


# ------------------------------------------------------------------- data
mp = xr.open_dataset(V.DIR_DATA / 'fig02_map.nc')
iso = read('fig02_isobath')
var = read('fig02_variance')
daily = read('fig02_daily', parse_dates=['day'])
summ = read('fig02_summary')
mon = read('fig02_monthly')
st = read('fig02_states')
sites = read('fig01_sites').set_index('site')
gebco, zname = I.open_gebco()
here = [s for s in V.SITES_200 if len(summ) and s in set(summ.site)]

vals = np.concatenate([summ[['s_p5', 's_p95']].values.ravel(),
                       iso.filter(like='s_').values.ravel()]) if len(summ) else np.array([0.2, 5])
vals = vals[np.isfinite(vals) & (vals > 0)]
S_LIM = (max(0.08, 0.8 * vals.min()), min(12.0, 1.25 * vals.max()))

# colormap of s, cold below one and warm above, pale around one
lev = np.asarray(V.S_LEVELS)
below, above = int((lev < 1).sum()), int((lev > 1).sum())
cols = list(plt.get_cmap(S.CMAP_DIVERGING)(np.linspace(0.08, 0.44, below))) + \
    list(plt.get_cmap(S.CMAP_DIVERGING)(np.linspace(0.58, 0.92, above)))
cmap_s = ListedColormap(cols)
cmap_s.set_under(plt.get_cmap(S.CMAP_DIVERGING)(0.02))
cmap_s.set_over(plt.get_cmap(S.CMAP_DIVERGING)(0.98))
norm_s = BoundaryNorm(lev, ncolors=cmap_s.N)

# ----------------------------------------------------------------- layout
fig = plt.figure(figsize=FIGSIZE)
gs_top = gridspec.GridSpec(1, 2, figure=fig, left=0.075, right=0.95, top=0.935, bottom=0.705,
                           width_ratios=[aspect(V.MAP_EXTENT), 0.95], wspace=0.24)
gs_mid = gridspec.GridSpec(2, 1, figure=fig, left=0.075, right=0.985, top=0.615, bottom=0.36,
                           hspace=0.30)
gs_bot = gridspec.GridSpec(1, 2, figure=fig, left=0.075, right=0.985, top=0.255, bottom=0.05,
                           width_ratios=[1.25, 1.0], wspace=0.28)
ax_map = fig.add_subplot(gs_top[0, 0])
ax_box = fig.add_subplot(gs_top[0, 1])
ax_s = fig.add_subplot(gs_mid[0, 0])
ax_r = fig.add_subplot(gs_mid[1, 0], sharex=ax_s)
ax_mon = fig.add_subplot(gs_bot[0, 0])
ax_t = fig.add_subplot(gs_bot[0, 1])

# --------------------------------------------------------------- (a) map
lon0, lon1, lat0, lat1 = V.MAP_EXTENT
sub = gebco[zname].sel(lon=slice(lon0 - 0.2, lon1 + 0.2), lat=slice(lat0 - 0.2, lat1 + 0.2))
blon, blat = np.meshgrid(sub.lon.values, sub.lat.values)
depth = -sub.values
sy = mp.s.sel(season='year')
pm = ax_map.pcolormesh(mp.lon, mp.lat, sy, cmap=cmap_s, norm=norm_s, shading='nearest', zorder=1,
                       rasterized=True)
if np.isfinite(sy.values).sum() > 4:
    ax_map.contour(mp.lon, mp.lat, sy, levels=[1.0], colors=S.INK, linewidths=0.9, zorder=3)
else:
    print('  no slope Burger number over the slope, (a) shows the geometry alone')
ax_map.contour(blon, blat, depth, levels=[1000], colors=S.ISOBATH, linewidths=0.45, zorder=2)
ax_map.contourf(blon, blat, depth, levels=[-1e4, 0.0], colors=[S.LAND], zorder=6)
ax_map.contour(blon, blat, depth, levels=[0.0], colors=S.COAST, linewidths=0.5, zorder=7)
if len(iso):
    ax_map.plot(iso.lon, iso.lat, color='white', lw=2.2, zorder=4)
    ax_map.plot(iso.lon, iso.lat, color=S.INK, lw=0.8, ls='--', zorder=5)
    for km in np.arange(0.0, iso.along_km.max(), PATH_MARK_KM):
        k = int(np.argmin(np.abs(iso.along_km - km)))
        ax_map.plot(iso.lon[k], iso.lat[k], marker='o', ms=2.6, color=S.INK, zorder=8)
        ax_map.annotate(f'{km:.0f}', (iso.lon[k], iso.lat[k]), xytext=(0, -9),
                        textcoords='offset points', ha='center', fontsize=S.FS_SMALL, zorder=8,
                        path_effects=HALO)
for site in here:
    r = sites.loc[site]
    S.ringed_marker(ax_map, r.lon, r.lat, S.SITE_MARKERS[site], S.SITE_COLOURS[site], ms=5.6,
                    zorder=10)
    ax_map.annotate(site, (r.lon, r.lat), xytext=V.LABEL_OFFSETS.get(site, (5, 4)),
                    textcoords='offset points', fontsize=S.FS_SMALL, zorder=12, path_effects=HALO)
ax_map.set_xlim(lon0, lon1)
ax_map.set_ylim(lat0, lat1)
S.geo_aspect(ax_map, 0.5 * (lat0 + lat1))
ax_map.xaxis.set_major_locator(MultipleLocator(1.0))
ax_map.yaxis.set_major_locator(MultipleLocator(1.0))
S.degree_ticks(ax_map)
S.panel(ax_map, 'a')
box = ax_map.inset_axes([0.012, 0.80, 0.40, 0.19], zorder=20)
box.set_xticks([])
box.set_yticks([])
box.set_facecolor('white')
for sp in box.spines.values():
    sp.set_linewidth(0.4)
    sp.set_edgecolor(S.LIGHT_GUIDE)
cax = box.inset_axes([0.06, 0.34, 0.88, 0.18])
cb = fig.colorbar(pm, cax=cax, orientation='horizontal', extend='both', ticks=lev[1::2])
cb.ax.set_xticklabels([f'{t:g}' for t in lev[1::2]])
cb.ax.set_title('slope Burger number', fontsize=S.FS_SMALL, pad=2.0)
cb.ax.tick_params(labelsize=S.FS_SMALL, length=1.8, pad=1.0)
cb.outline.set_linewidth(0.4)

# ----------------------------------------------- (b) distribution at the moorings
W = 0.30
for i, site in enumerate(here):
    for j, half in enumerate(V.HALVES):
        r = summ[(summ.site == site) & (summ.half == half)]
        if r.empty:
            continue
        r = r.iloc[0]
        x = i + (j - 0.5) * (W + 0.06)
        ax_box.plot([x, x], [r.s_p5, r.s_p95], color=S.INK, lw=0.7, zorder=2)
        ax_box.add_patch(Rectangle((x - W / 2, r.s_q1), W, r.s_q3 - r.s_q1, zorder=3,
                                   fc=S.SITE_COLOURS[site] if half == 'strat' else 'white',
                                   ec=S.SITE_COLOURS[site] if half == 'mixed' else 'black',
                                   hatch='////' if half == 'mixed' else None, lw=0.8))
        ax_box.plot([x - W / 2, x + W / 2], [r.s_median] * 2, color=S.INK, lw=1.3, zorder=4)
        # the GLORYS12V1 value is compared with the moored one in the Supporting Information (Table S1)
        if 's_clim' in r and np.isfinite(r.s_clim):
            ax_box.plot(x, r.s_clim, marker='X', ms=6.8, mfc=S.CLIM_MARK, mec='white', mew=0.7,
                        ls='none', zorder=6)
ax_box.set_xticks(range(len(here)))
tick = []
for site in here:
    r = summ[(summ.site == site) & (summ.half == 'all')]
    u = float(r.unstable_percent.iloc[0]) if len(r) and 'unstable_percent' in r else 0.0
    lab_ = site 
    if len(r) and r.layer.iloc[0] == 'upper pair':
        lab_ += f'\n{V.MOORED_LAYER_FALLBACK[0]} to {V.MOORED_LAYER_FALLBACK[1]} m'
    tick.append(lab_)
ax_box.set_xticklabels(tick, fontsize=S.FS_SMALL)
ax_box.set_xlim(-0.6, len(here) - 0.4)
log_s_axis(ax_box, S_LIM)
S.tidy(ax_box, axis='y')
ax_box.set_ylabel('slope Burger number')
S.panel(ax_box, 'b')
h = [S.patch_handle(S.GUIDE, ec='black'), S.patch_handle('white', ec=S.GUIDE, hatch='////'),
     S.marker_handle(S.CLIM_MARK, 'X', ms=6.0)]
ax_box.legend(h, ['ONDJFM', 'AMJJAS', 'climatology'],
              ncol=2, loc='lower center', bbox_to_anchor=(0.56, 1.01), columnspacing=1.0, handlelength=1.8)

# ----------------------------------------------------------- (c) along the path
seas = [c for c in (f's_{k}' for k in V.SEASONS) if c in iso]
if seas:
    ax_s.fill_between(iso.along_km, iso[seas].min(axis=1), iso[seas].max(axis=1),
                      color=S.SEASON_RANGE, lw=0, zorder=2)
ax_s.plot(iso.along_km, iso.s_year, color=S.INK, lw=1.2, zorder=3)
if iso.s_year.isna().all():
    print('  no slope Burger number along the path, (c) carries the moorings alone')
marks = iso.dropna(subset=['site']) if 'site' in iso else pd.DataFrame()
for _, m in marks.iterrows():
    r = summ[(summ.site == m.site) & (summ.half == 'all')]
    ax_s.axvline(m.along_km, color=S.SITE_COLOURS[m.site], lw=0.7, ls=':', zorder=1)
    ax_r.axvline(m.along_km, color=S.SITE_COLOURS[m.site], lw=0.7, ls=':', zorder=1)
    ax_s.annotate(m.site, (m.along_km, 1.0), xycoords=('data', 'axes fraction'), xytext=(3, -2),
                  textcoords='offset points', va='top', fontsize=S.FS_SMALL, path_effects=HALO)
    if len(r):
        r = r.iloc[0]
        ax_s.errorbar(m.along_km, r.s_median, yerr=[[r.s_median - r.s_q1], [r.s_q3 - r.s_median]],
                      marker=S.SITE_MARKERS[m.site], ms=6.0, mfc=S.SITE_COLOURS[m.site], mec='black',
                      mew=0.5, color='black', elinewidth=0.9, capsize=2.0, ls='none', zorder=6)
log_s_axis(ax_s, S_LIM)
ax_s.set_xlim(0.0, iso.along_km.max())
S.tidy(ax_s, axis='y')
ax_s.set_ylabel('slope Burger number')
plt.setp(ax_s.get_xticklabels(), visible=False)
S.panel(ax_s, 'c', dy=1.03)
h = [S.line_handle(S.INK, lw=1.2), S.patch_handle(S.SEASON_RANGE),
     S.line_handle('black', lw=0.9, marker='o', ms=5.4, mfc=S.GUIDE)]
ax_s.legend(h, ['climatology, mean of the four seasons'.replace(',', ''), 'range of the four seasons',
                'moorings (median and quartiles)'], ncol=3, loc='lower right',
            bbox_to_anchor=(1.0, 1.01), fontsize=LEG)

# ----------------------------------------------------- (d) what makes s change
frac = dict(zip(var.term, 100 * var.fraction_of_log_s)) if len(var) else {}
ok = iso[['alpha', 'N_year']].notna().all(axis=1)
if not ok.any():
    print('  no stratification along the path, (d) carries the slope alone')
ga = np.exp(np.mean(np.log(iso.alpha[ok]))) if ok.any() else np.exp(np.nanmean(np.log(iso.alpha)))
gn = np.exp(np.mean(np.log(iso.N_year[ok]))) if ok.any() else np.nan
ax_r.plot(iso.along_km, iso.alpha / ga, color=S.INK, lw=1.2, zorder=3)
ax_r.plot(iso.along_km, iso.N_year / gn, color=S.GUIDE, lw=1.2, ls='--', zorder=3)
ax_r.set_yscale('log')
rv = np.concatenate([(iso.alpha / ga)[ok], (iso.N_year / gn)[ok]]) if ok.any() else \
    (iso.alpha / np.exp(np.nanmean(np.log(iso.alpha)))).dropna().values
if rv.size:
    ax_r.set_ylim(0.8 * rv.min(), 1.25 * rv.max())
rt = [t for t in (0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0) if ax_r.get_ylim()[0] <= t <= ax_r.get_ylim()[1]]
ax_r.set_yticks(rt)
ax_r.set_yticklabels([f'{t:g}' for t in rt])
ax_r.yaxis.set_minor_formatter(NullFormatter())
ax_r.axhline(1.0, color=S.INK, lw=0.7, zorder=1)
S.tidy(ax_r, axis='y')
ax_r.set_ylabel('over the path mean')
ax_r.set_xlabel('distance along the 200 m isobath from Cabo Frio (km)')
S.panel(ax_r, 'd', dy=1.03)
h = [S.line_handle(S.INK, lw=1.2), S.line_handle(S.GUIDE, ls='--', lw=1.2)]
ax_r.legend(h, [f'bottom slope ({frac.get("slope", np.nan):.0f} % of the variance of log s)',
                f'buoyancy frequency ({frac.get("stratification", np.nan):.0f} %)'],
            ncol=2, loc='lower right', bbox_to_anchor=(1.0, 1.01), fontsize=LEG)

# --------------------------------------------------------- (e) cycle of the year
ax_mon.axvspan(0.5, 3.5, color=S.SEASON_SHADE, lw=0, zorder=0)
ax_mon.axvspan(9.5, 12.5, color=S.SEASON_SHADE, lw=0, zorder=0)
for k, site in enumerate(here):
    g = mon[mon.site == site].sort_values('month')
    if g.empty:
        continue
    col = S.SITE_COLOURS[site]
    ax_mon.fill_between(g.month, g.s_q1, g.s_q3, color=col, alpha=S.BAND_ALPHA, lw=0, zorder=2)
    ax_mon.plot(g.month, g.s_median, color=col, ls=S.SITE_LINESTYLES[site], marker=S.SITE_MARKERS[site], ms=5.0,
                mec='black', mew=0.5, lw=1.5, zorder=4)
log_s_axis(ax_mon, S_LIM)
ax_mon.set_xlim(0.5, 12.5)
ax_mon.set_xticks(range(1, 13))
ax_mon.set_xticklabels(list(MONTHS))
S.tidy(ax_mon, axis='y')
ax_mon.set_xlabel('month')
ax_mon.set_ylabel('slope Burger number')
S.panel(ax_mon, 'e', dy=1.03)
h = [S.line_handle(S.SITE_COLOURS[s], S.SITE_LINESTYLES[s], marker=S.SITE_MARKERS[s], ms=5.0)
     for s in here] + [S.patch_handle(S.SEASON_SHADE, ec=S.LIGHT_GUIDE)]
ax_mon.legend(h, here + [V.HALF_SHORT['strat']], ncol=4, loc='lower left',
              bbox_to_anchor=(0.0, 1.01), fontsize=LEG, columnspacing=1.0)

# ------------------------------------------------------- (f) the time scales
# each time scale as a bar over its interquartile range (the range of Cd for the wedge
# time) with its median marked, the symbols of the time scales of Figure 4
Q = (('tau_c', 'o', S.INK, 'gainsboro', 'canonical arrest time'),
     ('tau_w', 's', 'white', 'white', 'wedge time (range of C$_d$)'),
     ('state', 'v', S.INK, 'darkgray', 'state duration'))
allx = []
for i, site in enumerate(here):
    r = summ[(summ.site == site) & (summ.half == 'all')]
    if r.empty:
        continue
    r = r.iloc[0]
    rows = (('tau_c', r.tau_c_median_h, r.tau_c_q1_h, r.tau_c_q3_h),
            ('tau_w', r.tau_w_median_h, r.tau_w_lo_h, r.tau_w_hi_h),
            ('state', r.get('state_median_h', np.nan), r.get('state_q1_h', np.nan),
             r.get('state_q3_h', np.nan)))
    for j, (key, med, lo, hi) in enumerate(rows):
        if not np.isfinite(med):
            continue
        _, mk, fc, bar, _ = Q[j]
        y = i + (j - 1) * 0.27
        ax_t.barh(y, hi - lo, left=lo, height=0.20, color=bar, ec='black', lw=0.7, zorder=3)
        ax_t.plot(med, y, marker=mk, ms=6.4, mfc=fc, mec='black', mew=0.8, ls='none', zorder=4)
        allx += [lo, hi, med]
allx = np.asarray([x for x in allx if np.isfinite(x) and x > 0])
ax_t.set_xscale('log')
if allx.size:
    ax_t.set_xlim(0.5 * allx.min(), 2.0 * allx.max())
xt = [t for t in (0.3, 1, 3, 10, 30, 100, 300, 1000, 3000) if ax_t.get_xlim()[0] <= t <= ax_t.get_xlim()[1]]
ax_t.set_xticks(xt)
ax_t.set_xticklabels([f'{t:g}' for t in xt])
ax_t.xaxis.set_minor_formatter(NullFormatter())
ax_t.set_yticks(range(len(here)))
ax_t.set_yticklabels(here)
ax_t.set_ylim(len(here) - 0.5, -0.5)
ax_t.tick_params(axis='y', length=0)
S.tidy(ax_t, axis='x')
ax_t.set_xlabel('time (h)')
S.panel(ax_t, 'f', dy=1.03)
h = [(S.patch_handle(bar, ec='black'), S.marker_handle(fc, mk, ms=6.4, mec='black')) for _, mk, fc, bar, _ in Q]
for hh in h:
    hh[1].set_markeredgecolor('black')
ax_t.legend(h, [q[4] for q in Q], ncol=1, loc='lower left', bbox_to_anchor=(-0.02, 1.01),
            handlelength=1.8, handler_map={tuple: HandlerTuple(ndivide=None)})

fig.savefig(V.DIR_FIG / 'fig02_regime.pdf')
S.save(fig, V.DIR_FIG / 'fig02_regime.png')
