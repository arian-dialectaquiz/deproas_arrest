"""
Quantities of Figure 1, the array, its frame and its setting.

 1  geometry     position, GEBCO depth, bottom slope and isobath bearing of every
                 site and its distance along its line, with the bearing and the
                 slope refitted over several radii and the principal axis of the
                 subinertial flow at every level, which together bound the
                 uncertainty of the frame in which the cross-isobath velocity is
                 read, and the height of every current meter above the bottom
                 from the mooring design and from GEBCO
 2  inventory    every quality controlled record in data_qc with its position,
                 its depth attributes and its time span, which gives Table 1 and
                 locates the deep mooring FBC
 3  fields       mean surface geostrophic velocity of the sampling period and the
                 eddy kinetic energy of the departures from that mean, over water
                 deeper than ALT_MIN_DEPTH
 4  sections     along each line the GEBCO bottom, the mean velocity normal to the
                 line and the daily position of the axis of the Brazil Current,
                 its most negative normal velocity offshore of the shelf break
 5  wind         mean ERA5 wind stress over southeastern Brazil from the hourly
                 10 m wind with the drag coefficient of Large and Pond (1981), and
                 the points of the 100 m isobath at 27 and 30 S that carry the
                 remote forcing of the coastal trapped waves
 6  temperature  mean temperature along each line from the regional climatology,
                 else from the GLORYS12V1 reanalysis, for the isotherms of the
                 sections
 7  calendar     segments of every site with any level retained and with the
                 levels the bottom layer analysis needs, the lowest and the
                 interior level together at 200 m and the lowest level elsewhere

Outputs in paper_v3
    data/fig01_fields.nc         mean velocity, eddy kinetic energy and depth
    data/fig01_wind.nc           mean wind stress
    tables/fig01_sites.csv       Table 1
    tables/fig01_frame.csv       bearing and slope over the radii
    tables/fig01_axes.csv        principal axes of the flow at every level
    tables/fig01_heights.csv     heights of the current meters above the bottom
    tables/fig01_qc_inventory.csv
    tables/fig01_instruments.csv
    tables/fig01_sections.csv    bottom and normal velocity along the lines
    tables/fig01_axis_daily.csv  daily axis of the current
    tables/fig01_axis.csv        its median and quartiles
    tables/fig01_tsection.csv    mean temperature along the lines
    tables/fig01_remote.csv      remote wind points
    tables/fig01_calendar.csv    record segments
    logs/fig01_compute.txt

Created by Arian Dialectaquiz.
"""
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import config_v3 as V
import inputs_v3 as I

warnings.simplefilter('ignore', RuntimeWarning)
LOG = []


def say(msg=''):
    print(msg)
    LOG.append(msg)


def angle_diff(a, b):
    """Signed difference a minus b between two bearings, in -180 to 180 degrees."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def blocks(n, block, rng):
    """Indices of one moving block resample (Kuensch 1989), which carries the serial
    correlation of the series into the interval."""
    out, size = [], 0
    while size < n:
        s = int(rng.integers(0, max(n - block, 1)))
        b = np.arange(s, min(s + block, n))
        out.append(b)
        size += b.size
    return np.concatenate(out)[:n]


# ------------------------------------------------------------- 1 geometry
def geometry():
    geo, src = I.geometry()
    ln, _ = I.lines(geo)
    out = geo.reset_index().merge(ln[['site', 'line', 'along_km', 'cross_km']], on='site',
                                  how='left')
    out['line'] = out['line'].fillna('between the lines')
    say(f'  geometry of {len(out)} sites from {src}')
    return out.set_index('site'), ln


def frame(geo, gebco, zname):
    """Bearing, slope and depth of every site refitted over each radius."""
    rows = []
    for site in V.SITES:
        if site not in geo.index:
            continue
        r = geo.loc[site]
        row = dict(site=site, bearing_table=r.bearing, slope_table=r.slope, depth_table=r.depth)
        for rad in V.FRAME_RADII_KM:
            d, s, b = I.isobath_at(gebco, zname, r.lat, r.lon, rad)
            row.update({f'depth_{rad:g}km': d, f'slope_{rad:g}km': s, f'bearing_{rad:g}km': b})
        dev = angle_diff([row[f'bearing_{rad:g}km'] for rad in V.FRAME_RADII_KM], r.bearing)
        row['bearing_dev_max'] = float(np.nanmax(np.abs(dev)))
        rows.append(row)
        say(f'    {site}: bearing {r.bearing:5.1f} deg in the table, refitted '
            + ' '.join(f'{row[f"bearing_{rad:g}km"]:5.1f}' for rad in V.FRAME_RADII_KM)
            + f' over {V.FRAME_RADII_KM} km, slope '
            + ' '.join(f'{row[f"slope_{rad:g}km"]:.4f}' for rad in V.FRAME_RADII_KM))
    return pd.DataFrame(rows)


def principal_axes(lp, rng):
    """Principal axis of the subinertial flow at every level, in degrees counterclockwise
    from the along-isobath axis of the records, with a moving block interval. A flow
    that follows the isobaths gives an angle near zero, so the angle bounds the rotation
    that the frame may carry."""
    rows = []
    for site, path in lp.items():
        d = I.load_lp(path)
        for k, level in enumerate(d['level']):
            u, v = d['u'][:, k], d['v'][:, k]
            ok = np.isfinite(u) & np.isfinite(v) & (d['campaign'] != '')
            if ok.sum() < 240:
                continue
            uu, vv = u[ok] - u[ok].mean(), v[ok] - v[ok].mean()

            def axis_of(a, b):
                cuu, cvv, cuv = np.mean(a * a), np.mean(b * b), np.mean(a * b)
                th = 0.5 * np.degrees(np.arctan2(2.0 * cuv, cuu - cvv))
                lam = np.sqrt(np.maximum(np.linalg.eigvalsh([[cuu, cuv], [cuv, cvv]]), 0.0))
                return th, lam[1], lam[0]

            th, major, minor = axis_of(uu, vv)
            boot = []
            for _ in range(200):
                j = blocks(uu.size, int(24 * V.BOOT_BLOCK_DAYS / V.DT_H), rng)
                boot.append(axis_of(uu[j], vv[j])[0])
            dev = (np.asarray(boot) - th + 90.0) % 180.0 - 90.0
            lo, hi = th + np.percentile(dev, [2.5, 97.5])
            rows.append(dict(site=site, level=level, n_hours=int(ok.sum()), axis_deg=th,
                             axis_lo=lo, axis_hi=hi, minor_over_major=minor / major,
                             u_mean=float(u[ok].mean()), v_mean=float(v[ok].mean()),
                             mean_dir_deg=float(np.degrees(np.arctan2(v[ok].mean(),
                                                                       u[ok].mean())))))
            say(f'    {site} {level:4.0f} m: principal axis {th:+5.1f} deg ({lo:+5.1f} to '
                f'{hi:+5.1f}), minor over major {minor / major:.2f}')
    return pd.DataFrame(rows)


def heights(geo, inst):
    rows = []
    for _, r in inst.iterrows():
        if r.site not in geo.index:
            continue
        rows.append(dict(site=r.site, level=r.level,
                         hab_design=V.WATER_DEPTH_M[r.site] - r.level,
                         hab_gebco=float(geo.loc[r.site, 'depth']) - r.level))
    tab = pd.DataFrame(rows)
    for site in V.SITES_200:
        g = tab[(tab.site == site) & (tab.level == V.BOTTOM_LEVEL[site])]
        if len(g):
            say(f'    {site}: lowest meter {g.hab_design.iloc[0]:.0f} m above the bottom in '
                f'the design and {g.hab_gebco.iloc[0]:.0f} m over the GEBCO depth')
    return tab


def instruments(lp):
    """Levels of every site with the campaigns that hold data."""
    rows = []
    for site in V.SITES:
        have = {}
        if site in lp:
            d = I.load_lp(lp[site])
            for k, level in enumerate(d['level']):
                ok = np.isfinite(d['u'][:, k]) & (d['campaign'] != '')
                used = sorted({c for c in np.unique(d['campaign'][ok]) if c})
                have[float(level)] = (used, int(ok.sum()))
        for level in sorted(set(V.LEVELS[site]) | set(have)):
            used, hours = have.get(float(level), ([], 0))
            rows.append(dict(site=site, level=float(level), n_campaigns=len(used),
                             campaigns=' '.join(used), hours=hours))
    return pd.DataFrame(rows)


# ----------------------------------------------------------- 2 inventory
PATTERN = re.compile(r'DEPROAS_(?P<site>[A-Z0-9]+)_(?P<campaign>\d+)_(?P<level>\d{3,4})_qc\.nc$')


def _number(value):
    """First number held by an attribute, negative for a southern or western hemisphere."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    m = re.search(r'-?\d+(?:\.\d+)?', str(value))
    if not m:
        return np.nan
    x = float(m.group())
    if re.search(r'[SsWwOo]\b', str(value)) and x > 0:
        x = -x
    return x


def qc_inventory():
    rows = []
    for f in sorted(V.QC_DIR.glob('DEPROAS_*_qc.nc')):
        m = PATTERN.match(f.name)
        if not m:
            continue
        row = dict(file=f.name, site=m['site'], campaign=m['campaign'], level=int(m['level']))
        try:
            ds = xr.open_dataset(f)
            attrs = {str(k).lower(): v for k, v in ds.attrs.items()}
            for key, want in (('lat', ('lat',)), ('lon', ('lon',))):
                k = next((a for a in attrs if any(w in a for w in want)), None)
                row[key] = _number(attrs[k]) if k else np.nan
            depth_keys = [a for a in attrs if 'depth' in a]
            for a in depth_keys:
                row[f'attr_{re.sub(r"[^a-z0-9]+", "_", a).strip("_")}'] = _number(attrs[a])
            tname = next((n for n in ('time', 'TIME', 'date') if n in ds.variables), None)
            if tname is not None:
                t = pd.DatetimeIndex(ds[tname].values)
                row.update(start=t.min(), end=t.max(), n_samples=int(t.size))
            vname = next((n for n in ('VE', 'u', 'U', 'east', 'VEL_E') if n in ds.data_vars), None)
            if vname is not None:
                row['n_valid'] = int(np.isfinite(ds[vname].values).sum())
            ds.close()
        except Exception as e:                        # noqa: BLE001  reported and kept
            row['error'] = str(e)[:120]
        rows.append(row)
    tab = pd.DataFrame(rows)
    if len(tab):
        for site, g in tab.groupby('site'):
            say(f'    {site}: {len(g)} records, campaigns {" ".join(sorted(set(g.campaign)))}, '
                f'levels {" ".join(str(x) for x in sorted(set(g.level)))} m, position '
                f'{np.nanmedian(g.get("lat", np.nan)):.3f} {np.nanmedian(g.get("lon", np.nan)):.3f}')
    return tab


# -------------------------------------------------------------- 3 fields
def fields(alt, gebco, zname):
    lon0, lon1, lat0, lat1 = V.MAP_EXTENT
    sub = alt.sel(lon=slice(lon0 - 0.5, lon1 + 0.5), lat=slice(lat0 - 0.5, lat1 + 0.5))
    sub = sub.sel(time=slice(*V.PERIOD))
    u, v = sub['ugos'].load(), sub['vgos'].load()
    um, vm = u.mean('time'), v.mean('time')
    # The departures are taken from the mean of the sampling period, since the anomalies
    # of the product refer to a twenty year mean and would carry the mean of this period.
    eke = (0.5 * ((u - um) ** 2 + (v - vm) ** 2)).mean('time')
    depth = -gebco[zname].interp(lat=sub.lat, lon=sub.lon)
    out = xr.Dataset({'u_mean': um, 'v_mean': vm, 'eke': eke, 'depth': depth})
    out['n_days'] = int(sub.sizes['time'])
    out.attrs.update(period=f'{V.PERIOD[0]} to {V.PERIOD[1]}',
                     eke='departures of the daily velocity from the mean of the period')
    deep = depth >= V.ALT_MIN_DEPTH
    say(f'  {sub.sizes["time"]} days, eddy kinetic energy over water deeper than '
        f'{V.ALT_MIN_DEPTH:.0f} m of {float(eke.where(deep).mean()):.4f} m2 s-2 in the mean and '
        f'{float(eke.where(deep).max()):.4f} m2 s-2 at most')
    return out


# ------------------------------------------------------------ 4 sections
def sections(alt, lines_tab, gebco, zname):
    x = np.arange(V.SECTION_KM[0], V.SECTION_KM[1] + 1e-6, V.SECTION_STEP_KM)
    sub = alt.sel(time=slice(*V.PERIOD))
    t = pd.DatetimeIndex(sub.time.values)
    rows, daily = [], []
    for line, g in lines_tab.groupby('line'):
        r = g.iloc[0]
        unit = np.array([r.unit_east, r.unit_north])
        normal = np.array([-unit[1], unit[0]])        # positive with the shelf on the right
        c = np.cos(np.deg2rad(r.origin_lat))
        la = r.origin_lat + x * unit[1] / I.R_EARTH_KM
        lo = r.origin_lon + x * unit[0] / (I.R_EARTH_KM * c)
        pts = dict(lat=xr.DataArray(la, dims='s'), lon=xr.DataArray(lo, dims='s'))
        depth = -gebco[zname].interp(**pts).values
        un = (sub['ugos'].interp(**pts) * normal[0] + sub['vgos'].interp(**pts) * normal[1])
        un = un.load().values                              # time by distance
        rows.append(pd.DataFrame(dict(line=line, along_km=x, lat=la, lon=lo, depth_m=depth,
                                      v_mean=np.nanmean(un, axis=0), v_std=np.nanstd(un, axis=0))))
        offshore = depth >= V.SHELF_BREAK
        first = int(np.argmax(offshore)) if offshore.any() else 0
        search = (np.arange(x.size) >= first) & np.isfinite(un).any(axis=0)
        for k in range(t.size):
            w = np.where(search, un[k], np.nan)
            if not np.isfinite(w).any():
                continue
            j = int(np.nanargmin(w))
            if w[j] <= V.ALT_AXIS_MIN:
                daily.append(dict(line=line, time=t[k], axis_km=float(x[j]), axis_v=float(w[j]),
                                  axis_lat=float(la[j]), axis_lon=float(lo[j])))
        say(f'  {line}: shelf break at {x[first]:.0f} km, the axis of the current found on '
            f'{sum(d_["line"] == line for d_ in daily)} of {t.size} days')
    daily = pd.DataFrame(daily)
    summ = []
    if len(daily):
        daily['half'] = I.halves_of(daily.time)
        for line, g in daily.groupby('line'):
            for half in ('all',) + V.HALVES:
                h = g if half == 'all' else g[g.half == half]
                if h.empty:
                    continue
                q = np.percentile(h.axis_km, [25, 50, 75])
                summ.append(dict(line=line, half=half, n_days=len(h), axis_q1=q[0],
                                 axis_median=q[1], axis_q3=q[2],
                                 axis_v_median=float(h.axis_v.median())))
                if half == 'all':
                    say(f'    {line}: axis at {q[1]:.0f} km ({q[0]:.0f} to {q[2]:.0f}) with a '
                        f'median normal velocity of {h.axis_v.median():+.2f} m s-1')
    return pd.concat(rows, ignore_index=True), daily, pd.DataFrame(summ)


# ---------------------------------------------------------------- 5 wind
def drag_large_pond(speed):
    """Neutral drag coefficient of Large and Pond (1981), held at its low wind value below
    11 m s-1 and at its 25 m s-1 value above that speed."""
    s = np.clip(speed, 0.0, 25.0)
    return np.where(s < 11.0, 1.2e-3, (0.49 + 0.065 * s) * 1e-3)


def wind_mean(gebco, zname):
    fu = {re.search(r'(\d{4})', p.name).group(1): p for p in I.era5_files('u10')}
    fv = {re.search(r'(\d{4})', p.name).group(1): p for p in I.era5_files('v10')}
    years = sorted(set(fu) & set(fv))
    if not years:
        raise FileNotFoundError(f'no pair of era5_u10 and era5_v10 files in {V.ERA5_DIR}')
    lon0, lon1, lat0, lat1 = V.INSET_BOX
    acc_x = acc_y = None
    n = 0
    for y in years:
        du = I.standardise(xr.open_dataset(fu[y]))
        dv = I.standardise(xr.open_dataset(fv[y]))
        nu = 'u10' if 'u10' in du.data_vars else next(k for k in du.data_vars
                                                         if k.lower().startswith('u'))
        nv = 'v10' if 'v10' in dv.data_vars else next(k for k in dv.data_vars
                                                         if k.lower().startswith('v'))
        du = du[nu].sel(lon=slice(lon0, lon1), lat=slice(lat0, lat1), time=slice(*V.PERIOD))
        dv = dv[nv].sel(lon=slice(lon0, lon1), lat=slice(lat0, lat1), time=slice(*V.PERIOD))
        du, dv = xr.align(du, dv, join='inner')
        for a in range(0, du.sizes['time'], 744):
            uu = du.isel(time=slice(a, a + 744)).values.astype(float)
            vv = dv.isel(time=slice(a, a + 744)).values.astype(float)
            spd = np.hypot(uu, vv)
            cd = drag_large_pond(spd)
            tx = np.nansum(V.RHO_AIR * cd * spd * uu, axis=0)
            ty = np.nansum(V.RHO_AIR * cd * spd * vv, axis=0)
            acc_x = tx if acc_x is None else acc_x + tx
            acc_y = ty if acc_y is None else acc_y + ty
            n += uu.shape[0]
        lat, lon = du.lat.values, du.lon.values
    depth = -gebco[zname].interp(lat=xr.DataArray(lat, dims='lat'),
                                 lon=xr.DataArray(lon, dims='lon')).values
    out = xr.Dataset({'taux': (('lat', 'lon'), acc_x / n), 'tauy': (('lat', 'lon'), acc_y / n),
                      'depth': (('lat', 'lon'), depth)}, coords={'lat': lat, 'lon': lon})
    out.attrs.update(n_hours=n, years=' '.join(years), drag='Large and Pond (1981)')
    say(f'  wind stress from {n} time steps of {len(years)} years, mean magnitude over the ocean '
        f'{float(np.nanmean(np.hypot(out.taux, out.tauy).where(out.depth > 0))):.3f} N m-2')
    return out


def remote_points(gebco, zname):
    """Point of the REMOTE_ISOBATH on each latitude, found from offshore toward the coast
    so that coastal lagoons never enter the search."""
    rows = []
    for lat in V.REMOTE_WIND_LATS:
        row = gebco[zname].sel(lat=lat, method='nearest').sel(lon=slice(V.INSET_BOX[0], -38.0))
        depth, lon = -row.values.astype(float), row.lon.values
        shallow = np.flatnonzero(depth < V.REMOTE_ISOBATH)
        if shallow.size == 0 or shallow.max() + 1 >= lon.size:
            say(f'  no {V.REMOTE_ISOBATH:.0f} m isobath found at {lat} in the GEBCO grid')
            continue
        k = shallow.max() + 1
        rows.append(dict(lat=float(row.lat), lon=float(lon[k]), depth=float(depth[k])))
        say(f'  remote point at {float(row.lat):.2f} {float(lon[k]):.2f}, {depth[k]:.0f} m')
    return pd.DataFrame(rows)


# --------------------------------------------------------- 6 temperature
TEMP_NAMES = ('temp', 'temperature', 'thetao', 'TEMP', 'T', 't', 't_an', 'pot_temp', 'theta')
DEPTH_NAMES = ('depth', 'z', 'lev', 'level', 'deph', 'depth_m', 'Depth')


def unique_coords(da):
    """Every dimension coordinate kept once and in increasing order, since gridded
    products sometimes repeat a latitude, a longitude or a depth and interpolation
    refuses repeated values."""
    for d in da.dims:
        if d in da.coords and da[d].ndim == 1:
            _, idx = np.unique(da[d].values, return_index=True)
            if idx.size < da.sizes[d]:
                da = da.isel({d: np.sort(idx)})
            da = da.sortby(d)
    return da


def _temperature_of(ds, box):
    name = next((n for n in TEMP_NAMES if n in ds.data_vars), None)
    if name is None:
        name = next((n for n in ds.data_vars
                     if 'temperature' in str(ds[n].attrs.get('standard_name', '')).lower()
                     or 'temperature' in str(ds[n].attrs.get('long_name', '')).lower()), None)
    if name is None:
        return None, 'no temperature variable among ' + ' '.join(map(str, ds.data_vars))
    da = ds[name]
    zname = next((n for n in DEPTH_NAMES if n in da.dims), None)
    if zname is None or 'lat' not in da.dims or 'lon' not in da.dims:
        return None, f'{name} has the dimensions {da.dims}, not depth, latitude and longitude'
    da = da.rename({zname: 'z'})
    da = da.assign_coords(z=np.abs(da.z.values.astype(float)))
    da = unique_coords(da)
    for extra in [d for d in da.dims if d not in ('z', 'lat', 'lon')]:
        da = da.mean(extra)
    lon0, lon1, lat0, lat1 = box
    da = da.sel(lon=slice(lon0, lon1), lat=slice(lat0, lat1)).load()
    da = da.where((da > -5.0) & (da < 40.0))          # fill values of the land mask
    return da, name


def climatology_temperature(box):
    seasons = I.open_climatology(box)
    parts = [g['T'] for g in seasons.values() if np.isfinite(g['T'].values).any()]
    if not parts:
        return None, f'no readable temperature in the climatology of {V.CLIM_DIR}'
    ref = parts[0]
    mean = sum(p.interp(z=ref.z, lat=ref.lat, lon=ref.lon) for p in parts) / len(parts)
    return mean, (f'the mean of {len(parts)} seasons ({" ".join(seasons)}) of the climatology '
                  f'in {V.CLIM_DIR}')


def glorys_temperature(box):
    lon0, lon1, lat0, lat1 = box
    acc, n = None, 0
    for f in I.glorys_files():
        ds = I.standardise(xr.open_dataset(f))
        if 'thetao' not in ds:
            continue
        da = ds['thetao'].sel(lon=slice(lon0, lon1), lat=slice(lat0, lat1),
                              time=slice(*V.PERIOD)).rename({'depth': 'z'})
        if da.sizes['time'] == 0:
            continue
        da = unique_coords(da)
        s = da.sum('time', skipna=True).load()
        c = da.notnull().sum('time').load()
        acc = (s, c) if acc is None else (acc[0] + s, acc[1] + c)
        n += da.sizes['time']
    if acc is None:
        return None, 'no GLORYS12V1 temperature'
    return (acc[0] / acc[1].where(acc[1] > 0)), f'the GLORYS12V1 mean of {n} days'


def temperature_sections(sec):
    box = (float(sec.lon.min()) - 0.3, float(sec.lon.max()) + 0.3,
           float(sec.lat.min()) - 0.3, float(sec.lat.max()) + 0.3)
    sources = ['climatology', 'glorys'] if V.T_SECTION_SOURCE == 'climatology' else ['glorys']
    zi = np.arange(0.0, V.SECTION_ZMAX + 0.1, 2.0)
    for source in sources:
        try:
            T3, src = climatology_temperature(box) if source == 'climatology' \
                else glorys_temperature(box)
            if T3 is None:
                say(f'  {src}')
                continue
            T3 = unique_coords(T3.transpose('z', 'lat', 'lon'))
            rows = []
            for line, g in sec.groupby('line'):
                g = g.sort_values('along_km')
                Ti = T3.interp(lat=xr.DataArray(g.lat.values, dims='s'),
                               lon=xr.DataArray(g.lon.values, dims='s')).transpose('z', 's').values
                z = T3.z.values
                for j, (x, bottom) in enumerate(zip(g.along_km.values, g.depth_m.values)):
                    col = Ti[:, j]
                    ok = np.isfinite(col)
                    if ok.sum() < 2:
                        continue
                    prof = np.interp(zi, z[ok], col[ok], left=np.nan, right=np.nan)
                    prof[zi > bottom] = np.nan
                    rows.append(pd.DataFrame(dict(line=line, along_km=x, z=zi, T=prof)))
            if not rows:
                say(f'  {src} leaves no profile along the lines')
                continue
            say(f'  temperature of the sections from {src}')
            return pd.concat(rows, ignore_index=True), src
        except Exception as e:                        # noqa: BLE001  the next source is tried
            say(f'  the {source} failed for the sections, {e}')
    raise FileNotFoundError('no temperature product could be read for the sections')


# ------------------------------------------------------------ 7 calendar
def segments(t, mask):
    """Start, end and hours of every run of True, with gaps under GAP_MERGE_H joined."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    gap = np.diff(t[idx]) / np.timedelta64(1, 'h')
    cut = np.flatnonzero(gap > V.GAP_MERGE_H)
    out = []
    for a, b in zip(np.r_[0, cut + 1], np.r_[cut + 1, idx.size]):
        sel = idx[a:b]
        hours = sel.size * V.DT_H
        if hours >= V.MIN_SEGMENT_H:
            out.append((t[sel[0]], t[sel[-1]], hours))
    return out


def calendar(lp):
    rows = []
    for site, path in lp.items():
        d = I.load_lp(path)
        t, lv, camp = d['time'], list(d['level']), d['campaign']
        any_level = np.isfinite(d['u']).any(axis=1) & (camp != '')
        need = [V.BOTTOM_LEVEL[site], V.INTERIOR_LEVEL[site]] if site in V.SITES_200 \
            else [V.BOTTOM_LEVEL[site]]
        if all(n in lv for n in need):
            used = np.all([np.isfinite(d['u'][:, lv.index(n)]) for n in need], axis=0) & (camp != '')
        else:
            used = np.zeros(t.size, bool)
        for kind, mask in (('any', any_level), ('analysis', used)):
            for c in sorted({x for x in np.unique(camp) if x}):
                for a, b, h in segments(t.values, mask & (camp == c)):
                    rows.append(dict(site=site, kind=kind, campaign=c, start=a, end=b, hours=h))
        say(f'  {site}: {int(any_level.sum())} hours with any level and {int(used.sum())} with '
            f'the levels of the layer analysis')
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    V.make_dirs()
    rng = np.random.default_rng(101)
    say('figure 1, the array, its frame and its setting')
    geo, lines_tab = geometry()
    gebco, zname = I.open_gebco()
    lp = I.lp_files()

    say('\n 1 frame of the isobaths')
    fr = frame(geo, gebco, zname)
    fr.to_csv(V.DIR_TAB / 'fig01_frame.csv', index=False)
    ax_tab = principal_axes(lp, rng)
    ax_tab.to_csv(V.DIR_TAB / 'fig01_axes.csv', index=False)
    inst = instruments(lp)
    inst.to_csv(V.DIR_TAB / 'fig01_instruments.csv', index=False)
    heights(geo, inst).to_csv(V.DIR_TAB / 'fig01_heights.csv', index=False)

    say('\n 2 inventory of the quality controlled records')
    try:
        inv = qc_inventory()
        inv.to_csv(V.DIR_TAB / 'fig01_qc_inventory.csv', index=False)
    except Exception as e:                            # noqa: BLE001
        say(f'  inventory skipped, {e}')

    say('\n 3 altimetric fields')
    alt = I.open_altimetry()
    fld = fields(alt, gebco, zname)
    out = V.DIR_DATA / 'fig01_fields.nc'
    if out.exists():
        out.unlink()
    fld.to_netcdf(out)

    say('\n 4 sections and the axis of the current')
    sec, daily, summ = sections(alt, lines_tab, gebco, zname)
    sec.to_csv(V.DIR_TAB / 'fig01_sections.csv', index=False)
    daily.to_csv(V.DIR_TAB / 'fig01_axis_daily.csv', index=False)
    summ.to_csv(V.DIR_TAB / 'fig01_axis.csv', index=False)

    say('\n 5 wind stress and the remote points')
    try:
        wind = wind_mean(gebco, zname)
        out = V.DIR_DATA / 'fig01_wind.nc'
        if out.exists():
            out.unlink()
        wind.to_netcdf(out)
    except Exception as e:                            # noqa: BLE001
        say(f'  wind skipped, {e}')
    remote_points(gebco, zname).to_csv(V.DIR_TAB / 'fig01_remote.csv', index=False)

    say('\n 6 temperature along the lines')
    try:
        tsec, src = temperature_sections(sec)
        tsec.to_csv(V.DIR_TAB / 'fig01_tsection.csv', index=False)
    except Exception as e:                            # noqa: BLE001
        say(f'  temperature sections skipped, {e}')

    say('\n 7 calendar')
    cal = calendar(lp)
    cal.to_csv(V.DIR_TAB / 'fig01_calendar.csv', index=False)

    # Table 1, one row per site
    rows = []
    for site in V.SITES:
        if site not in geo.index:
            continue
        g = inst[inst.site == site]
        c = cal[(cal.site == site)] if len(cal) else pd.DataFrame(columns=['kind', 'hours', 'campaign'])
        r = geo.loc[site]
        rows.append(dict(site=site, line=r.line, lat=r.lat, lon=r.lon, depth=r.depth,
                         slope=r.slope, bearing=r.bearing, along_km=r.along_km,
                         levels=' '.join(f'{x:.0f}' for x in g.level),
                         campaigns=' '.join(sorted(set(c.campaign))) if len(c) else '',
                         hours_any=float(c[c.kind == 'any'].hours.sum()) if len(c) else 0.0,
                         hours_analysis=float(c[c.kind == 'analysis'].hours.sum()) if len(c) else 0.0))
    pd.DataFrame(rows).to_csv(V.DIR_TAB / 'fig01_sites.csv', index=False)
    (V.DIR_LOG / 'fig01_compute.txt').write_text('\n'.join(LOG) + '\n')
    say(f'\ntables in {V.DIR_TAB} and fields in {V.DIR_DATA}')


if __name__ == '__main__':
    main()