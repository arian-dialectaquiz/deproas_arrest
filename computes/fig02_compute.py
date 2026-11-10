"""
Quantities of Figure 2, the regime of the shelf break.

The slope Burger number s = N alpha / |f| sets how fast and how far the bottom
boundary layer arrests (MacCready and Rhines 1991, Garrett et al. 1993, Brink
and Lentz 2010a), so this figure measures it in space, in season and at the
moorings, and sets its time scales against the time the interior flow keeps
one direction.

 1  slope        bottom slope and depth from the GEBCO gradient, block averaged
                 to SLOPE_RES_DEG and smoothed over SLOPE_SMOOTH_KM
 2  map          N over the BBL_N_THICKNESS metres above the bottom of every
                 point of the regional climatology (Sasaki et al. 2024), from
                 the potential density of TEOS-10, for each season and for the
                 year, and s over the slope between the 100 and 1000 m isobaths
 3  isobath      s, N and alpha along the 200 m isobath from Cabo Frio to the
                 southwest, and the split of the variance of log s along the
                 path into the parts of the slope, of the stratification and of
                 the Coriolis parameter, against the variance of the seasonal
                 change
 4  moorings     daily N and s at the 200 m sites from the moored temperature at
                 the two lowest levels, with the salinity from the temperature
                 and salinity relation of the central water fitted to the
                 climatology around each site, checked against the GLORYS12V1
                 reanalysis and the climatology over the same levels
 5  times        the canonical arrest time (|f| s^2)^-1, the wedge time
                 1 / (2 Cd N s^3) over the range of Cd, and the time the interior
                 flow keeps one sign, which is the time the layer is given

Outputs in paper_v3
    data/fig02_map.nc            N, alpha and s on the climatology grid
    tables/fig02_isobath.csv     the path along the 200 m isobath
    tables/fig02_variance.csv    split of the variance of log s along the path
    tables/fig02_ts.csv          temperature and salinity relation of every site
    tables/fig02_daily.csv       daily stratification and time scales at the sites
    tables/fig02_monthly.csv     monthly medians
    tables/fig02_states.csv      states of the interior flow
    tables/fig02_summary.csv     one row per site and half of the year
    logs/fig02_compute.txt

Created by Arian Dialectaquiz.
"""
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import config_v3 as V
import inputs_v3 as I
import physics_v3 as P

warnings.simplefilter('ignore', RuntimeWarning)
LOG = []
HOUR = 3600.0


def say(msg=''):
    print(msg)
    LOG.append(msg)


# ------------------------------------------------------------- 2 the map
def bottom_n2(ds, field, season):
    """N squared over the layer above the bottom of every column of one season. The bottom
    is the GEBCO depth, and a column is used when the climatology reaches within
    BOTTOM_MISMATCH_M of it. A climatology filled below the sea floor is accepted, since
    GEBCO then fixes the bottom, and a climatology that stops short of it is left out.
    The reason for every column left out is counted and reported."""
    T, S = ds['T'].values, ds['S'].values if 'S' in ds else None
    z = ds.z.values
    depth = field.depth.interp(lat=xr.DataArray(ds.lat.values, dims='lat'),
                               lon=xr.DataArray(ds.lon.values, dims='lon')).values
    out = np.full(T.shape[1:], np.nan)
    why = dict(outside_depth_range=0, no_profile=0, short_of_bottom=0, too_few_levels=0, used=0)
    for j in range(T.shape[1]):
        for i in range(T.shape[2]):
            h = depth[j, i]
            if not (V.MAP_DEPTH_RANGE[0] <= h <= V.MAP_DEPTH_RANGE[1]):
                why['outside_depth_range'] += 1
                continue
            ok = np.isfinite(T[:, j, i]) & (np.isfinite(S[:, j, i]) if S is not None else True)
            if ok.sum() < 3:
                why['no_profile'] += 1
                continue
            wet = z[ok].max()
            if wet < h - V.BOTTOM_MISMATCH_M:
                why['short_of_bottom'] += 1
                continue
            zb = min(h, wet)
            sal = S[ok, j, i] if S is not None else np.full(ok.sum(), V.S_CONSTANT)
            sig = P.sigma0(T[ok, j, i], sal, z[ok], ds.lon.values[i], ds.lat.values[j])
            out[j, i] = P.n2_fit(z[ok], sig, zb - V.BBL_N_THICKNESS, zb)
            if np.isfinite(out[j, i]):
                why['used'] += 1
            else:
                why['too_few_levels'] += 1
    say(f'    {season}: columns ' + ', '.join(f'{k.replace("_", " ")} {v}' for k, v in why.items()))
    return out


def maps(field):
    lon0, lon1, lat0, lat1 = V.MAP_EXTENT
    clim = I.open_climatology((lon0, lon1, lat0, lat1))
    seasons = [s for s in V.SEASONS if s in clim]
    if not seasons:
        raise FileNotFoundError('no season of the climatology could be read')
    ref = clim[seasons[0]]
    n2 = {}
    for s in seasons:
        ds = clim[s].interp(lat=ref.lat, lon=ref.lon) if s != seasons[0] else ref
        if 'S' not in ds:
            say(f'  {s}: no salinity, the constant {V.S_CONSTANT} bounds N from above')
        n2[s] = bottom_n2(ds, field, s)
        say(f'  {s}: N over the {V.BBL_N_THICKNESS:.0f} m above the bottom on '
            f'{int(np.isfinite(n2[s]).sum())} points, median {np.nanmedian(np.sqrt(np.maximum(n2[s], 0))):.2e} s-1')
    n2['year'] = np.nanmean(np.stack([n2[s] for s in seasons]), axis=0)
    alpha = field.alpha.interp(lat=ref.lat, lon=ref.lon).values
    f = np.abs(P.coriolis(ref.lat.values))[:, None]
    names = ['year'] + seasons
    N = np.stack([np.sqrt(np.where(n2[k] > 0, n2[k], np.nan)) for k in names])
    s_ = N * alpha[None] / f[None]
    out = xr.Dataset({'N': (('season', 'lat', 'lon'), N), 's': (('season', 'lat', 'lon'), s_),
                      'alpha': (('lat', 'lon'), alpha)},
                     coords={'season': names, 'lat': ref.lat.values, 'lon': ref.lon.values})
    out.attrs.update(eos=P.EOS, layer_m=V.BBL_N_THICKNESS, source=str(V.CLIM_DIR))
    sy = out.s.sel(season='year').values
    say(f'  s over the slope, median {np.nanmedian(sy):.2f}, above one on '
        f'{100 * np.nanmean(sy[np.isfinite(sy)] > 1):.0f} % of the points')
    if 'DJF' in names and 'JJA' in names:
        r = (out.s.sel(season='DJF') / out.s.sel(season='JJA')).values
        say(f'  summer over winter, median {np.nanmedian(r):.2f}, tenth and ninetieth percentiles '
            f'{np.nanpercentile(r, 10):.2f} and {np.nanpercentile(r, 90):.2f}')
    return out, clim


# ----------------------------------------------------------- 3 the isobath
def isobath(field, mp, geo):
    start = V.PLACE_NAMES['Cabo Frio']
    lo, la, along = P.isobath_path(field, float(V.SHELF_BREAK), V.MAP_EXTENT, start)
    alpha = field.alpha.interp(lat=xr.DataArray(la, dims='p'), lon=xr.DataArray(lo, dims='p')).values
    f = np.abs(P.coriolis(la))
    tab = pd.DataFrame(dict(along_km=along, lat=la, lon=lo, alpha=alpha, f=f))
    for k in mp.season.values:
        N = P.idw(la, lo, mp.lat.values, mp.lon.values, mp.N.sel(season=k).values, V.PATH_IDW_KM)
        tab[f'N_{k}'] = N
        tab[f's_{k}'] = N * alpha / f
    for site in V.SITES_200:
        if site in geo.index:
            d = P.distance_km(geo.loc[site, 'lat'], geo.loc[site, 'lon'], la, lo)
            k = int(np.argmin(d))
            tab.loc[k, 'site'] = site
            say(f'    {site} at {along[k]:.0f} km of the path, {d[k]:.1f} km from it'
                + ('' if d[k] <= 15.0 else ', check the extent of the path'))
    ok = tab[['s_year', 'N_year', 'alpha']].notna().all(axis=1)
    L = np.log(tab.loc[ok, ['s_year', 'N_year', 'alpha', 'f']])
    var_s = float(np.var(L.s_year))
    rows = [dict(term='slope', variance=float(np.var(L.alpha))),
            dict(term='stratification', variance=float(np.var(L.N_year))),
            dict(term='Coriolis', variance=float(np.var(L.f))),
            dict(term='covariance of slope and stratification',
                 variance=float(2 * np.cov(L.alpha, L.N_year)[0, 1]))]
    if 's_DJF' in tab and 's_JJA' in tab:
        r = np.log(tab.loc[ok, 's_DJF'] / tab.loc[ok, 's_JJA'])
        rows.append(dict(term='season', variance=float(np.nanvar(r))))
    var = pd.DataFrame(rows)
    var['fraction_of_log_s'] = var.variance / var_s
    say(f'  along {along[-1]:.0f} km of the isobath s is below one on '
        f'{100 * np.mean(tab.s_year[ok] < 1):.0f} % of the path, variance of log s {var_s:.3f}')
    for _, r in var.iterrows():
        say(f'    {r.term}: {100 * r.fraction_of_log_s:.0f} % of the variance of log s')
    return tab, var


# ---------------------------------------------------------- 4 the moorings
def ts_relation(clim, geo):
    rows = []
    for site in V.SITES_200:
        if site not in geo.index:
            continue
        T_, S_ = [], []
        for s, ds in clim.items():
            if 'S' not in ds:
                continue
            GLo, GLa = np.meshgrid(ds.lon.values, ds.lat.values)
            near = P.distance_km(geo.loc[site, 'lat'], geo.loc[site, 'lon'], GLa, GLo) <= V.TS_RADIUS_KM
            zsel = (ds.z.values >= V.TS_DEPTHS[0]) & (ds.z.values <= V.TS_DEPTHS[1])
            t = ds['T'].values[zsel][:, near].ravel()
            sa = ds['S'].values[zsel][:, near].ravel()
            ok = np.isfinite(t) & np.isfinite(sa)
            T_.append(t[ok])
            S_.append(sa[ok])
        if not T_ or sum(x.size for x in T_) < 10:
            rows.append(dict(site=site, a=V.S_CONSTANT, b=0.0, r2=np.nan, n=0,
                             note='constant salinity'))
            say(f'    {site}: no salinity near the site, the constant {V.S_CONSTANT} is used')
            continue
        t, sa = np.concatenate(T_), np.concatenate(S_)
        b, a = np.polyfit(t, sa, 1)
        r2 = 1.0 - np.var(sa - (a + b * t)) / np.var(sa)
        rows.append(dict(site=site, a=a, b=b, r2=r2, n=int(t.size), t_min=float(t.min()),
                         t_max=float(t.max()), note='climatology'))
        say(f'    {site}: S = {a:.3f} + {b:.4f} T over {t.min():.1f} to {t.max():.1f} degrees, '
            f'r2 {r2:.3f} from {t.size} values')
    return pd.DataFrame(rows).set_index('site')


def layer_of(d):
    """Lowest pair of levels whose temperatures overlap over at least ten days."""
    lv = list(d['level'])
    for pair, kind in ((V.MOORED_LAYER, 'lowest pair'), (V.MOORED_LAYER_FALLBACK, 'upper pair')):
        if pair[0] in lv and pair[1] in lv:
            both = np.isfinite(d['T'][:, lv.index(pair[0])]) & np.isfinite(d['T'][:, lv.index(pair[1])])
            if both.sum() * V.DT_H >= 240:
                return pair, kind
    return None, None


def moored_daily(lp, geo, ts):
    """Daily N and s at the 200 m sites, from the lowest pair of levels on the days it
    holds both temperatures and from the upper pair on the other days."""
    rows = []
    for site in V.SITES_200:
        if site not in lp or site not in geo.index:
            continue
        d = I.load_lp(lp[site], verbose=True)
        lv = list(d['level'])
        cols = {f'T{int(z)}': d['T'][:, lv.index(z)] for z in set(V.MOORED_LAYER + V.MOORED_LAYER_FALLBACK)
                if z in lv}
        if len(cols) < 2:
            say(f'    {site}: fewer than two levels with temperature')
            continue
        T = pd.DataFrame(cols, index=d['time'])
        T['u_i'] = d['u'][:, lv.index(V.INTERIOR_LEVEL[site])] if V.INTERIOR_LEVEL[site] in lv else np.nan
        T['campaign'] = d['campaign']
        T = T[T.campaign != '']
        g = T.groupby([T.campaign, T.index.floor('D')])
        day = g.mean(numeric_only=True)
        cnt = g.count()
        r = ts.loc[site]
        lat, lon = geo.loc[site, 'lat'], geo.loc[site, 'lon']
        alpha, f = geo.loc[site, 'slope'], abs(P.coriolis(lat))
        best = pd.DataFrame(index=day.index, columns=['N2', 'z_top', 'z_bot', 'layer', 'T_top', 'T_bot'],
                            dtype=object)
        for pair, kind in ((V.MOORED_LAYER_FALLBACK, 'upper pair'), (V.MOORED_LAYER, 'lowest pair')):
            a_, b_ = f'T{pair[0]}', f'T{pair[1]}'
            if a_ not in day or b_ not in day:
                continue
            full = (cnt[a_] >= V.MIN_DAY_HOURS) & (cnt[b_] >= V.MIN_DAY_HOURS)
            st_ = P.sigma0(day[a_], r.a + r.b * day[a_], pair[0], lon, lat)
            sb_ = P.sigma0(day[b_], r.a + r.b * day[b_], pair[1], lon, lat)
            n2 = pd.Series(P.n2_two_levels(st_, sb_, pair[0], pair[1]), index=day.index)
            use = full & n2.notna()
            best.loc[use, 'N2'] = n2[use]
            best.loc[use, 'z_top'], best.loc[use, 'z_bot'] = pair
            best.loc[use, 'layer'] = kind
            best.loc[use, 'T_top'], best.loc[use, 'T_bot'] = day.loc[use, a_], day.loc[use, b_]
        best = best.dropna(subset=['N2'])
        n2 = best.N2.astype(float).values
        N = np.sqrt(np.where(n2 > 0, n2, np.nan))
        s = N * alpha / f
        idx = best.index.to_frame(index=False)
        out = pd.DataFrame(dict(site=site, day=idx.iloc[:, 1].values, campaign=idx.iloc[:, 0].values,
                                layer=best.layer.values, z_top=best.z_top.astype(float).values,
                                z_bot=best.z_bot.astype(float).values,
                                T_top=best.T_top.astype(float).values,
                                T_bot=best.T_bot.astype(float).values, N=N, s=s,
                                u_i=day.loc[best.index, 'u_i'].values))
        out['tau_c_h'] = P.arrest_time_canonical(s, f) / HOUR
        for name, cd in (('tau_w_h', V.CD), ('tau_w_lo_h', V.CD_RANGE[1]), ('tau_w_hi_h', V.CD_RANGE[0])):
            out[name] = P.arrest_time_wedge(s, N, cd) / HOUR
        rows.append(out)
        kinds = out.layer.value_counts().to_dict()
        say(f'    {site}: {len(out)} days, ' + ' and '.join(f'{v} over the {k}' for k, v in kinds.items())
            + f', median N {np.nanmedian(N):.2e} s-1 and s {np.nanmedian(s):.2f}, negative N2 on '
            f'{100 * np.mean(n2 <= 0):.0f} % of the days')
    tab = pd.concat(rows, ignore_index=True)
    tab['half'] = I.halves_of(tab.day)
    tab['month'] = pd.DatetimeIndex(tab.day).month
    return tab


def glorys_daily(geo, daily):
    """Reanalysed N over the same levels at the nearest column deep enough."""
    files = I.glorys_files()
    if not files:
        say('  no GLORYS12V1 files')
        return pd.DataFrame()
    first = I.standardise(xr.open_dataset(files[0]))
    pick = {}
    for site in V.SITES_200:
        g = daily[daily.site == site]
        if g.empty:
            continue
        z_top, z_bot = (int(x) for x in g[['z_top', 'z_bot']].mode().iloc[0])
        lat, lon = geo.loc[site, 'lat'], geo.loc[site, 'lon']
        th = first['thetao'].isel(time=0)
        deep = th.depth.values[th.depth.values > z_bot]
        if deep.size == 0:
            continue
        wet = th.sel(depth=deep[0]).notnull().values
        GLo, GLa = np.meshgrid(th.lon.values, th.lat.values)
        d = P.distance_km(lat, lon, GLa, GLo)
        d = np.where(wet & (d <= V.GLORYS_SEARCH_KM), d, np.inf)
        if not np.isfinite(d).any():
            say(f'    {site}: no reanalysed column deeper than {z_bot} m within '
                f'{V.GLORYS_SEARCH_KM:.0f} km')
            continue
        j, i = np.unravel_index(np.argmin(d), d.shape)
        pick[site] = (j, i, z_top, z_bot, float(d[j, i]))
        say(f'    {site}: reanalysed column {d[j, i]:.1f} km from the site')
    first.close()
    rows = []
    for f_ in files:
        ds = I.standardise(xr.open_dataset(f_))
        if 'so' not in ds:
            say(f'  {f_.name} holds no salinity, the reanalysis is left out')
            return pd.DataFrame()
        ds = ds.sel(time=slice(*V.PERIOD))
        z = ds.depth.values
        for site, (j, i, z_top, z_bot, dist) in pick.items():
            th = ds['thetao'].isel(lat=j, lon=i).values
            so = ds['so'].isel(lat=j, lon=i).values
            lat, lon = float(ds.lat[j]), float(ds.lon[i])
            for k, t in enumerate(pd.DatetimeIndex(ds.time.values)):
                ok = np.isfinite(th[k]) & np.isfinite(so[k])
                if ok.sum() < 2:
                    continue
                tt = np.interp([z_top, z_bot], z[ok], th[k][ok])
                ss = np.interp([z_top, z_bot], z[ok], so[k][ok])
                sg = P.sigma0(tt, ss, [z_top, z_bot], lon, lat, potential=True)
                rows.append(dict(site=site, day=t.normalize(),
                                 N2_glorys=float(P.n2_two_levels(sg[0], sg[1], z_top, z_bot)),
                                 T_top_glorys=tt[0], T_bot_glorys=tt[1], distance_km=dist))
        ds.close()
    tab = pd.DataFrame(rows)
    tab['N_glorys'] = np.sqrt(np.where(tab.N2_glorys > 0, tab.N2_glorys, np.nan))
    return tab


def climatology_at_sites(clim, geo, daily):
    rows = []
    for site in V.SITES_200:
        g = daily[daily.site == site]
        if g.empty:
            continue
        z_top, z_bot = (int(x) for x in g[['z_top', 'z_bot']].mode().iloc[0])
        lat, lon = geo.loc[site, 'lat'], geo.loc[site, 'lon']
        for s, ds in clim.items():
            if s not in V.SEASONS:
                continue
            GLo, GLa = np.meshgrid(ds.lon.values, ds.lat.values)
            T3 = ds['T'].values
            deep = T3[np.searchsorted(ds.z.values, z_bot):].copy()
            wet = np.isfinite(deep).any(axis=0) if deep.size else np.zeros(GLa.shape, bool)
            d = np.where(wet, P.distance_km(lat, lon, GLa, GLo), np.inf)
            if not np.isfinite(d).any() or d.min() > V.TS_RADIUS_KM:
                continue
            j, i = np.unravel_index(np.argmin(d), d.shape)
            t = T3[:, j, i]
            sa = ds['S'].values[:, j, i] if 'S' in ds else np.full(t.size, V.S_CONSTANT)
            ok = np.isfinite(t) & np.isfinite(sa)
            tt = np.interp([z_top, z_bot], ds.z.values[ok], t[ok])
            ss = np.interp([z_top, z_bot], ds.z.values[ok], sa[ok])
            sg = P.sigma0(tt, ss, [z_top, z_bot], lon, lat)
            n2 = float(P.n2_two_levels(sg[0], sg[1], z_top, z_bot))
            rows.append(dict(site=site, season=s, N2=n2, distance_km=float(d[j, i])))
    tab = pd.DataFrame(rows)
    if len(tab):
        for site, g in tab.groupby('site'):
            rows.append(dict(site=site, season='year', N2=float(g.N2.mean()),
                             distance_km=float(g.distance_km.mean())))
        tab = pd.DataFrame(rows)
        tab['N'] = np.sqrt(np.where(tab.N2 > 0, tab.N2, np.nan))
    return tab


# ------------------------------------------------------------- 5 the states
def states(lp, daily):
    rows = []
    for site in V.SITES_200:
        if site not in lp:
            continue
        d = I.load_lp(lp[site])
        lv = list(d['level'])
        if V.INTERIOR_LEVEL[site] not in lv:
            continue
        u = d['u'][:, lv.index(V.INTERIOR_LEVEL[site])]
        dd = daily[daily.site == site].set_index('day')
        for a, b, sgn in P.runs_of_sign(u, d['campaign']):
            days = pd.DatetimeIndex(d['time'][a:b]).normalize().unique()
            tc = dd.tau_c_h.reindex(days).median() if len(dd) else np.nan
            tw = dd.tau_w_h.reindex(days).median() if len(dd) else np.nan
            hours = (b - a) * V.DT_H
            rows.append(dict(site=site, start=d['time'][a], end=d['time'][b - 1], hours=hours,
                             sign=sgn, upslope=int(sgn < 0), tau_c_h=tc, tau_w_h=tw,
                             gamma_c=hours / tc if tc else np.nan,
                             gamma_w=hours / tw if tw else np.nan))
    tab = pd.DataFrame(rows)
    for site, g in tab.groupby('site'):
        say(f'    {site}: {len(g)} states of at least {V.MIN_STATE_H} h, median '
            f'{g.hours.median() / 24:.1f} days ({g.hours.quantile(.25) / 24:.1f} to '
            f'{g.hours.quantile(.75) / 24:.1f}), {100 * g.upslope.mean():.0f} % upslope, '
            f'the time given over the canonical arrest time {g.gamma_c.median():.0f} and over '
            f'the wedge time {g.gamma_w.median():.0f}')
    return tab


# ------------------------------------------------------------------ main
def main():
    V.make_dirs()
    say('figure 2, the regime of the shelf break')
    say(f'  equation of state {P.EOS}')
    geo, _ = I.geometry(verbose=False)
    lp = I.lp_files()
    gebco, zname = I.open_gebco()

    say('\n 1 bottom slope')
    field = P.slope_field(gebco, zname, V.MAP_EXTENT)
    say(f'  GEBCO on {field.sizes["lat"]} by {field.sizes["lon"]} points of {V.SLOPE_RES_DEG} deg, '
        f'slope smoothed over {V.SLOPE_SMOOTH_KM:.0f} km')
    for site in V.SITES_200:
        if site in geo.index:
            a = float(field.alpha.interp(lat=geo.loc[site, 'lat'], lon=geo.loc[site, 'lon']))
            say(f'    {site}: {a:.4f} from the field and {geo.loc[site, "slope"]:.4f} in the table')

    say('\n 2 stratification above the bottom from the climatology')
    mp, clim = maps(field)
    out = V.DIR_DATA / 'fig02_map.nc'
    if out.exists():
        out.unlink()
    mp.to_netcdf(out)

    say('\n 3 the 200 m isobath')
    iso, var = isobath(field, mp, geo)
    iso.to_csv(V.DIR_TAB / 'fig02_isobath.csv', index=False)
    var.to_csv(V.DIR_TAB / 'fig02_variance.csv', index=False)

    say('\n 4 stratification at the moorings')
    ts = ts_relation(clim, geo)
    ts.to_csv(V.DIR_TAB / 'fig02_ts.csv')
    daily = moored_daily(lp, geo, ts)
    gl = glorys_daily(geo, daily)
    if len(gl):
        daily = daily.merge(gl, on=['site', 'day'], how='left')
        daily['s_glorys'] = daily.N_glorys * daily.site.map(geo.slope) / \
            np.abs(P.coriolis(daily.site.map(geo.lat)))
        for site, g in daily.groupby('site'):
            ok = g[['N', 'N_glorys']].notna().all(axis=1)
            if ok.sum() > 30:
                r = np.corrcoef(g.N[ok], g.N_glorys[ok])[0, 1]
                say(f'    {site}: reanalysed over moored N, median ratio '
                    f'{np.median(g.N_glorys[ok] / g.N[ok]):.2f}, correlation {r:+.2f} over '
                    f'{int(ok.sum())} days, bottom temperature bias '
                    f'{np.nanmedian(g.T_bot_glorys - g.T_bot):+.2f} degrees')
    cl = climatology_at_sites(clim, geo, daily)
    cl.to_csv(V.DIR_TAB / 'fig02_climatology_sites.csv', index=False)
    daily.to_csv(V.DIR_TAB / 'fig02_daily.csv', index=False)

    say('\n 5 time scales')
    st = states(lp, daily)
    st.to_csv(V.DIR_TAB / 'fig02_states.csv', index=False)

    rows, mon = [], []
    for site, g in daily.groupby('site'):
        low = g[g.layer == 'lowest pair']
        g = low if len(low) >= 30 else g           # one layer per site in the summaries
        for half in ('all',) + V.HALVES:
            h = g if half == 'all' else g[g.half == half]
            if h.s.notna().sum() < 10:
                continue
            q = np.nanpercentile(h.s, [5, 25, 50, 75, 95])      # stable days, the others counted
            r = dict(site=site, half=half, n_days=int(h.s.notna().sum()), layer=h.layer.mode().iloc[0],
                     s_p5=q[0], s_q1=q[1], s_median=q[2], s_q3=q[3], s_p95=q[4],
                     N_median=float(np.nanmedian(h.N)), above_one=100 * float(np.nanmean(h.s > 1)),
                     unstable_percent=100 * float(np.mean(~np.isfinite(h.N))),
                     tau_c_median_h=float(np.nanmedian(h.tau_c_h)),
                     tau_c_q1_h=float(np.nanpercentile(h.tau_c_h, 25)),
                     tau_c_q3_h=float(np.nanpercentile(h.tau_c_h, 75)),
                     tau_w_median_h=float(np.nanmedian(h.tau_w_h)),
                     tau_w_lo_h=float(np.nanmedian(h.tau_w_lo_h)),
                     tau_w_hi_h=float(np.nanmedian(h.tau_w_hi_h)))
            if 's_glorys' in h:
                r['s_glorys_median'] = float(np.nanmedian(h.s_glorys))
            c = cl[(cl.site == site)] if len(cl) else pd.DataFrame()
            if len(c):
                key = {'all': ['year'], 'strat': ['DJF', 'SON'], 'mixed': ['JJA', 'MAM']}[half]
                cc = c[c.season.isin(key)]
                if len(cc):
                    Nc = float(np.sqrt(np.nanmean(cc.N2)))
                    r['s_clim'] = Nc * geo.loc[site, 'slope'] / abs(P.coriolis(geo.loc[site, 'lat']))
            s_ = st[(st.site == site)] if len(st) else pd.DataFrame()
            if half != 'all' and len(s_):
                s_ = s_[I.halves_of(s_.start) == half]
            if len(s_):
                r.update(state_median_h=float(s_.hours.median()),
                         state_q1_h=float(s_.hours.quantile(.25)),
                         state_q3_h=float(s_.hours.quantile(.75)),
                         gamma_c_median=float(s_.gamma_c.median()))
            u = np.nanmedian(np.abs(h.u_i))
            ustar = np.sqrt(V.CD) * u
            delta = 0.4 * ustar / abs(P.coriolis(geo.loc[site, 'lat']))
            r['arrest_over_spindown'] = (delta / V.WATER_DEPTH_M[site]) / r['s_median'] ** 2
            rows.append(r)
            if half == 'all':
                say(f'    {site}: s {q[2]:.2f} ({q[1]:.2f} to {q[3]:.2f}), canonical arrest '
                    f'{r["tau_c_median_h"]:.1f} h, wedge {r["tau_w_median_h"]:.1f} h '
                    f'({r["tau_w_lo_h"]:.1f} to {r["tau_w_hi_h"]:.1f}), states of '
                    f'{r.get("state_median_h", np.nan) / 24:.1f} days, arrest over spin-down time '
                    f'{r["arrest_over_spindown"]:.3f}')
        for m, h in g.groupby('month'):
            if h.s.notna().sum() < 5:
                continue
            q = np.nanpercentile(h.s, [25, 50, 75])
            mon.append(dict(site=site, month=m, n_days=int(h.s.notna().sum()), s_q1=q[0],
                            s_median=q[1], s_q3=q[2],
                            s_glorys_median=float(np.nanmedian(h.s_glorys)) if 's_glorys' in h
                            else np.nan))
    pd.DataFrame(rows).to_csv(V.DIR_TAB / 'fig02_summary.csv', index=False)
    pd.DataFrame(mon).to_csv(V.DIR_TAB / 'fig02_monthly.csv', index=False)
    (V.DIR_LOG / 'fig02_compute.txt').write_text('\n'.join(LOG) + '\n')
    say(f'\ntables in {V.DIR_TAB}')


if __name__ == '__main__':
    main()