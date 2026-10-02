"""
Figure 1. Region, array and setting of the shelf break.

(a) northern South Brazil Bight with the GEBCO isobaths, the 200 m isobath of
    the moorings dashed, the eddy kinetic energy of the daily surface
    geostrophic velocity about its mean over the records (colour, over water
    deeper than 100 m) and the mean surface geostrophic circulation as
    streamlines whose width follows the speed, with the seven moorings and the
    Ubatuba and Cabo Frio lines, the keys of both fields over the continent.
(b) southeastern Brazil with the mean ERA5 wind stress of the same period, the
    extent of (a), the deep mooring FBC and the points of the 100 m isobath at
    27 and 30 S that carry the remote forcing of the coastal trapped waves.
(c) and (d) the Ubatuba and Cabo Frio lines, the GEBCO bottom, the mean
    isotherms every 2 C with the 20 C upper limit of the South Atlantic Central
    Water thicker, every current meter with the number of campaigns retained,
    and the median and quartiles of the daily position of the Brazil Current
    axis from the altimetry.
(e) calendar of the records, any level retained (light bar) and the levels the
    bottom layer analysis needs (dark bar, the lowest and interior levels
    together at 200 m and the lowest level at 100 and 50 m), with October to
    March shaded and the campaigns named above.

The figure is laid out with three gridspecs on fixed fractions of the page. The
widths of (a) and (b) follow from their geographic extents, so that both maps
share one height whatever extents the settings give.

Reads paper_v3/data/fig01_*.nc and paper_v3/tables/fig01_*.csv, and the GEBCO
grid. Writes paper_v3/figures/fig01_region.png and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator

import config_v3 as V
import inputs_v3 as I
import style as S

FIGSIZE = (7.48, 8.7)
HALO = [pe.withStroke(linewidth=1.8, foreground='white')]
SECTION_YLIM = (V.SECTION_ZMAX, -26.0)
AXIS_Y = -14.0                     # depth at which the axis of the current is drawn
LEG = S.LEGEND_SIZE


def read(name, **kw):
    p = V.DIR_TAB / f'{name}.csv'
    if p.exists() and p.stat().st_size > 5:
        return pd.read_csv(p, **kw)
    print(f'  {p.name} is missing, its elements are left out')
    return pd.DataFrame()


def aspect(extent):
    """Width over height of a map drawn at equal distances."""
    lon0, lon1, lat0, lat1 = extent
    return (lon1 - lon0) * np.cos(np.deg2rad(0.5 * (lat0 + lat1))) / (lat1 - lat0)


# ------------------------------------------------------------------- data
fld = xr.open_dataset(V.DIR_DATA / 'fig01_fields.nc')
f_wind = V.DIR_DATA / 'fig01_wind.nc'
wind = xr.open_dataset(f_wind) if f_wind.exists() else None
sites = read('fig01_sites').set_index('site')
inst = read('fig01_instruments')
sec = read('fig01_sections')
axis = read('fig01_axis')
tsec = read('fig01_tsection')
remote = read('fig01_remote')
inv = read('fig01_qc_inventory')
cal = read('fig01_calendar', parse_dates=['start', 'end'], dtype={'campaign': str})
gebco, zname = I.open_gebco()
deep = {}
if len(inv) and 'lat' in inv:
    for s_, g_ in inv[inv.site.isin(V.DEEP_SITES)].groupby('site'):
        deep[s_] = (float(np.nanmedian(g_.lon)), float(np.nanmedian(g_.lat)))

# ----------------------------------------------------------------- layout
fig = plt.figure(figsize=FIGSIZE)
gs_top = gridspec.GridSpec(1, 2, figure=fig, left=0.075, right=0.975, top=0.975, bottom=0.715,
                           width_ratios=[aspect(V.MAP_EXTENT), aspect(V.INSET_BOX)], wspace=0.10)
gs_mid = gridspec.GridSpec(1, 3, figure=fig, left=0.075, right=0.985, top=0.645, bottom=0.39,
                           width_ratios=[1.0, 1.0, 0.72], wspace=0.10)
gs_bot = gridspec.GridSpec(2, 1, figure=fig, left=0.075, right=0.985, top=0.29, bottom=0.022,
                           height_ratios=[1.0, 0.13], hspace=0.30)
ax_map = fig.add_subplot(gs_top[0, 0])
ax_ins = fig.add_subplot(gs_top[0, 1])
ax_ub = fig.add_subplot(gs_mid[0, 0])
ax_cf = fig.add_subplot(gs_mid[0, 1], sharey=ax_ub)
ax_leg = fig.add_subplot(gs_mid[0, 2])
ax_leg.axis('off')
ax_cal = fig.add_subplot(gs_bot[0, 0])
ax_leg_cal = fig.add_subplot(gs_bot[1, 0])
ax_leg_cal.axis('off')

# --------------------------------------------------------------- (a) map
lon0, lon1, lat0, lat1 = V.MAP_EXTENT
sub = gebco[zname].sel(lon=slice(lon0 - 0.2, lon1 + 0.2), lat=slice(lat0 - 0.2, lat1 + 0.2))
blon, blat = np.meshgrid(sub.lon.values, sub.lat.values)
depth = -sub.values
deep_water = fld.depth.values >= V.ALT_MIN_DEPTH
cmap = S.truncated(S.CMAP_EKE)
norm = BoundaryNorm(V.EKE_LEVELS, ncolors=cmap.N, clip=False)
cf = ax_map.contourf(fld.lon, fld.lat, np.where(deep_water, fld.eke.values, np.nan),
                     levels=V.EKE_LEVELS, cmap=cmap, norm=norm, extend='max', zorder=1)
cs = ax_map.contour(blon, blat, depth, levels=list(V.MAP_ISOBATHS), colors=S.ISOBATH,
                    linewidths=0.4, zorder=2)
ax_map.clabel(cs, fmt='%d', fontsize=5.0, inline_spacing=1.5)
ax_map.contour(blon, blat, depth, levels=[V.SHELF_BREAK], colors=S.SHELF_BREAK,
               linewidths=0.7, linestyles='--', zorder=3)
um = np.ma.masked_invalid(np.where(deep_water, fld.u_mean.values, np.nan))
vm = np.ma.masked_invalid(np.where(deep_water, fld.v_mean.values, np.nan))
spd = np.ma.hypot(um, vm).filled(0.0)
ax_map.streamplot(fld.lon.values, fld.lat.values, um, vm, density=V.STREAM_DENSITY,
                  color=S.STREAM, arrowsize=0.5, zorder=5,
                  linewidth=V.SPEED_LW[0] + (V.SPEED_LW[1] - V.SPEED_LW[0])
                  * np.clip(spd / V.SPEED_REF, 0.0, 1.0))
ax_map.contourf(blon, blat, depth, levels=[-1e4, 0.0], colors=[S.LAND], zorder=6)
ax_map.contour(blon, blat, depth, levels=[0.0], colors=S.COAST, linewidths=0.5, zorder=7)
for line, members in V.LINES.items():
    g = sites.loc[[s for s in members if s in sites.index]]
    ax_map.plot(g.lon, g.lat, color='white', lw=2.0, zorder=8)
    ax_map.plot(g.lon, g.lat, color=S.LINE_COLOURS[line], lw=1.0, ls=':', zorder=9)
for site, r in sites.iterrows():
    S.ringed_marker(ax_map, r.lon, r.lat, S.SITE_MARKERS[site], S.SITE_COLOURS[site], ms=4.6,
                    zorder=10)
    ax_map.annotate(site, (r.lon, r.lat), xytext=V.LABEL_OFFSETS.get(site, (5, 4)),
                    textcoords='offset points', fontsize=6.0, zorder=12, path_effects=HALO)
for name, (plon, plat) in V.PLACE_NAMES.items():
    off, ha = V.PLACE_OFFSETS.get(name, ((0, 5), 'center'))
    ax_map.annotate(name, (plon, plat), xytext=off, textcoords='offset points', ha=ha,
                    fontsize=6.0, style='italic', color=S.GUIDE, zorder=12)
ax_map.set_xlim(lon0, lon1)
ax_map.set_ylim(lat0, lat1)
S.geo_aspect(ax_map, 0.5 * (lat0 + lat1))
ax_map.xaxis.set_major_locator(MultipleLocator(1.0))
ax_map.yaxis.set_major_locator(MultipleLocator(1.0))
S.degree_ticks(ax_map)
S.panel(ax_map, 'a')

# keys of (a) over the continent, the colour of the eddy kinetic energy above
# and the width of the streamlines below
box = ax_map.inset_axes([0.012, 0.715, 0.40, 0.275], zorder=20)
box.set_xlim(0, 1)
box.set_ylim(0, 1)
box.set_xticks([])
box.set_yticks([])
box.set_facecolor('white')
for sp in box.spines.values():
    sp.set_linewidth(0.4)
    sp.set_edgecolor(S.LIGHT_GUIDE)
cax = box.inset_axes([0.06, 0.66, 0.88, 0.11])
cb = fig.colorbar(cf, cax=cax, orientation='horizontal', ticks=V.EKE_LEVELS[::2])
cb.ax.set_title('eddy kinetic energy (m$^2$ s$^{-2}$)', fontsize=6.3, pad=2.0)
cb.ax.tick_params(labelsize=5.8, length=1.8, pad=1.0)
cb.outline.set_linewidth(0.4)
for j, value in enumerate((0.15, V.SPEED_REF)):
    x0 = 0.06 + 0.46 * j
    box.plot([x0, x0 + 0.14], [0.13, 0.13], color=S.STREAM,
             lw=V.SPEED_LW[0] + (V.SPEED_LW[1] - V.SPEED_LW[0]) * value / V.SPEED_REF)
    box.text(x0 + 0.17, 0.13, f'{value:g} m s$^{{-1}}$', va='center', fontsize=5.9)
box.text(0.5, 0.25, 'mean surface geostrophic speed', ha='center', va='bottom', fontsize=6.3)

# ------------------------------------------------------ (b) wider setting
ilon0, ilon1, ilat0, ilat1 = V.INSET_BOX
isub = gebco[zname].sel(lon=slice(ilon0, ilon1), lat=slice(ilat0, ilat1))
stride = max(1, isub.sizes['lon'] // 500)
isub = isub.isel(lon=slice(None, None, stride), lat=slice(None, None, stride))
ilon, ilat = np.meshgrid(isub.lon.values, isub.lat.values)
idepth = -isub.values
ax_ins.contour(ilon, ilat, idepth, levels=[V.SHELF_BREAK], colors=S.ISOBATH, linewidths=0.45,
               zorder=2)
if wind is not None:
    k = V.WIND_STRIDE
    w = wind.isel(lat=slice(None, None, k), lon=slice(None, None, k))
    sea = w.depth.values > 0
    q = ax_ins.quiver(w.lon.values, w.lat.values, np.where(sea, w.taux.values, np.nan),
                      np.where(sea, w.tauy.values, np.nan), color=S.WIND,
                      scale=V.WIND_KEY / 0.20, scale_units='inches', width=0.005,
                      headwidth=3.5, headlength=4.0, zorder=4)
    ax_ins.quiverkey(q, 0.62, 1.035, V.WIND_KEY, f'{V.WIND_KEY:g} N m$^{{-2}}$', labelpos='E',
                     coordinates='axes', fontproperties={'size': 6.3}, labelsep=0.05)
ax_ins.contourf(ilon, ilat, idepth, levels=[-1e4, 0.0], colors=[S.LAND], zorder=5)
ax_ins.contour(ilon, ilat, idepth, levels=[0.0], colors=S.COAST, linewidths=0.45, zorder=6)
ax_ins.add_patch(plt.Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, fill=False,
                               ec=S.INK, lw=0.8, zorder=7))
for _, r in remote.iterrows():
    ax_ins.plot(r.lon, r.lat, marker='^', ms=5.2, mfc=S.INK, mec='white', mew=0.5, ls='none',
                zorder=8)
    ax_ins.annotate(f'{abs(r.lat):.0f}°S', (r.lon, r.lat), xytext=(6, -1),
                    textcoords='offset points', fontsize=6.3, va='center', zorder=8,
                    path_effects=HALO)
for name, (dlon, dlat) in deep.items():
    ax_ins.plot(dlon, dlat, marker=S.DEEP_MARKER, ms=5.6, mfc=S.DEEP_COLOUR, mec='white',
                mew=0.5, ls='none', zorder=8)
    ax_ins.annotate(name, (dlon, dlat), xytext=(0, -9), textcoords='offset points',
                    ha='center', fontsize=6.3, zorder=8, path_effects=HALO)
ax_ins.set_xlim(ilon0, ilon1)
ax_ins.set_ylim(ilat0, ilat1)
S.geo_aspect(ax_ins, 0.5 * (ilat0 + ilat1))
ax_ins.xaxis.set_major_locator(MultipleLocator(5.0))
ax_ins.yaxis.set_major_locator(MultipleLocator(4.0))
S.degree_ticks(ax_ins)
S.panel(ax_ins, 'b')

# ----------------------------------------------------- (c) and (d) sections
for k, (ax, line) in enumerate(((ax_ub, 'Ubatuba'), (ax_cf, 'Cabo Frio'))):
    g = sec[sec.line == line].sort_values('along_km')
    ax.fill_between(g.along_km, g.depth_m, 1e4, color=S.BOTTOM_FILL, lw=0, zorder=3)
    ax.plot(g.along_km, g.depth_m, color=S.BOTTOM_EDGE, lw=0.7, zorder=4)
    t = tsec[tsec.line == line] if len(tsec) else pd.DataFrame()
    if len(t):
        grid = t.pivot_table(index='z', columns='along_km', values='T')
        X, Z = np.meshgrid(grid.columns.values, grid.index.values)
        thin = [v for v in V.ISOTHERMS if v != V.TW_SACW]
        c1 = ax.contour(X, Z, grid.values, levels=thin, colors=S.ISOTHERM, linewidths=0.45,
                        zorder=2)
        c2 = ax.contour(X, Z, grid.values, levels=[V.TW_SACW], colors=S.ISOTHERM,
                        linewidths=1.2, zorder=2)
        for c in (c1, c2):
            ax.clabel(c, fmt=lambda v: f'{v:.0f}', fontsize=5.6, inline_spacing=1.5)
    for site in [s for s in V.LINES[line] if s in sites.index]:
        x = float(sites.loc[site, 'along_km'])
        bottom = float(np.interp(x, g.along_km, g.depth_m))
        ax.plot([x, x], [0.0, bottom], color=S.LIGHT_GUIDE, lw=0.5, ls=':', zorder=5)
        for _, r in inst[inst.site == site].iterrows():
            z = min(r.level, bottom - V.INSTRUMENT_CLEARANCE_M)
            done = r.n_campaigns > 0
            S.ringed_marker(ax, x, z, S.SITE_MARKERS[site],
                            S.SITE_COLOURS[site] if done else 'white', ms=4.6, zorder=7)
            ax.annotate(f'{r.n_campaigns:.0f}', (x, z), xytext=(6, 0),
                        textcoords='offset points', fontsize=5.8, va='center', color=S.GUIDE,
                        zorder=9, annotation_clip=False, path_effects=HALO)
        ax.annotate(site, (x, 0.0), xytext=(0, 4), textcoords='offset points', ha='center',
                    fontsize=6.0, annotation_clip=False, zorder=9, path_effects=HALO)
    a = axis[(axis.line == line) & (axis.half == 'all')] if len(axis) else pd.DataFrame()
    if len(a):
        a = a.iloc[0]
        ax.plot([a.axis_q1, a.axis_q3], [AXIS_Y, AXIS_Y], color=S.INK, lw=1.2, zorder=8)
        ax.plot(a.axis_median, AXIS_Y, marker='v', ms=5.0, color=S.INK, ls='none', zorder=9)
    ax.set_ylim(*SECTION_YLIM)
    ax.set_xlim(V.SECTION_KM[0], V.SECTION_KM[1])
    S.tidy(ax, axis='both')
    ax.set_xlabel('distance along the line (km)')
    ax.set_title(line, pad=10.0)
    S.panel(ax, 'cd'[k], dy=1.03)
ax_ub.set_ylabel('depth (m)')
plt.setp(ax_cf.get_yticklabels(), visible=False)

h = [S.marker_handle(S.SITE_COLOURS[s], S.SITE_MARKERS[s], ms=5.4) for s in V.SITES]
lab = list(V.SITES)
if deep:
    h.append(S.marker_handle(S.DEEP_COLOUR, S.DEEP_MARKER, ms=5.4))
    lab.append(' '.join(deep) + ' deep mooring')
if len(inst) and (inst.n_campaigns == 0).any():
    h.append(S.marker_handle('silver', 'o', ms=5.4, open_=True))
    lab.append('no record retained')
if len(tsec):
    h += [S.line_handle(S.ISOTHERM, lw=0.45), S.line_handle(S.ISOTHERM, lw=1.2)]
    lab += ['isotherm every 2 °C', f'{V.TW_SACW:.0f} °C isotherm']
h.append(S.line_handle(S.INK, lw=1.2, marker='v', ms=5.0))
lab.append('axis of the Brazil Current')
ax_leg.legend(h, lab, loc='center left', ncol=1, fontsize=LEG, handlelength=2.4,
              borderaxespad=0.0, labelspacing=0.5)

# --------------------------------------------------------- (e) calendar
order = [s for s in V.SITES_200 + V.SITES_100 + V.SITES_50 if len(cal) and s in set(cal.site)]
ypos = {s: i for i, s in enumerate(order)}
t0, t1 = pd.Timestamp(V.PERIOD[0]), pd.Timestamp(V.PERIOD[1])
for year in range(t0.year - 1, t1.year + 1):
    ax_cal.axvspan(pd.Timestamp(f'{year}-10-01'), pd.Timestamp(f'{year + 1}-04-01'),
                   color=S.SEASON_SHADE, lw=0, zorder=0)
for _, r in cal.iterrows():
    if r.site not in ypos:
        continue
    width = mdates.date2num(r.end) - mdates.date2num(r.start)
    if r.kind == 'any':
        ax_cal.barh(ypos[r.site], width, left=mdates.date2num(r.start), height=0.62,
                    color=S.RETAINED_BAR, lw=0, zorder=2)
    else:
        ax_cal.barh(ypos[r.site], width, left=mdates.date2num(r.start), height=0.28,
                    color=S.SITE_COLOURS[r.site], ec='black', lw=0.25, zorder=3)
if len(cal):
    for c, g in cal[cal.kind == 'any'].groupby('campaign'):
        mid = g.start.min() + (g.end.max() - g.start.min()) / 2
        ax_cal.annotate(f'C{c}', (mdates.date2num(mid), -0.55), ha='center', va='bottom',
                        fontsize=6.3, color=S.GUIDE, annotation_clip=False)
ax_cal.set_yticks(range(len(order)))
ax_cal.set_yticklabels(order, fontsize=6.8)
ax_cal.set_ylim(len(order) - 0.45, -0.95)
ax_cal.set_xlim(t0, t1)
ax_cal.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
ax_cal.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax_cal.tick_params(axis='x', labelsize=6.8)
S.tidy(ax_cal, axis='x')
S.panel(ax_cal, 'e', dy=1.05)
h = [S.patch_handle(S.RETAINED_BAR), S.patch_handle(S.GUIDE, ec='black'),
     S.patch_handle(S.SEASON_SHADE, ec=S.LIGHT_GUIDE)]
lab = ['any level retained', 'levels of the bottom layer analysis', 'October to March']
ax_leg_cal.legend(h, lab, loc='center', ncol=3, fontsize=LEG, borderaxespad=0.0,
                  handlelength=2.4, columnspacing=2.0)

fig.savefig(V.DIR_FIG / 'fig01_region.pdf')
S.save(fig, V.DIR_FIG / 'fig01_region.png')