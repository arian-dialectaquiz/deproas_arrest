"""
Where the third version finds its inputs, and the readers every script shares.

The records of the first version, the tables of the exploratory steps and the
external fields live in different folders, so every script of this version asks
this module for them and never names a path. Each input is looked for in a list
of candidate folders and, when a table is missing, it is rebuilt from what is
available.

    geometry     site_geometry.csv, else external_site_bathymetry.csv, else
                 computed from the mooring positions and the GEBCO depth gradient
    lines        lines.csv, else built from the positions, with the offshore
                 direction of each line taken from its inner to its outer site
    records      <site>_lp.nc of the first version, the hourly low-passed series
                 already rotated into the frame of the local isobath
    fields       GEBCO, the DUACS altimetry, ERA5 and GLORYS12V1 readers

    python inputs_v3.py      report where every input was found

Created by Arian Dialectaquiz.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config_v3 as V

HERE = Path(__file__).resolve().parent
TABLE_DIRS = [V.OLD / 'tables', V.V2 / 'tables', V.ROOT / 'tables', V.OLD, V.ROOT]
DATA_DIRS = [V.OLD / 'data', V.V2 / 'data', V.ROOT / 'data_products', V.OLD, V.ROOT]
R_EARTH_KM = 111.2


# ------------------------------------------------------------- utilities
def find(name, dirs):
    for d in dirs:
        p = Path(d) / name
        if p.exists() and p.stat().st_size > 1:
            return p
    return None


def pick(cols, *keys):
    """First column whose name holds every key."""
    for c in cols:
        if all(k in str(c).lower() for k in keys):
            return c
    return None


def halves_of(times):
    return np.array([V.HALF_OF_MONTH[m] for m in pd.DatetimeIndex(times).month])


def distance_km(lat0, lon0, lat, lon):
    la = np.deg2rad(0.5 * (lat0 + np.asarray(lat)))
    return R_EARTH_KM * np.hypot(np.asarray(lat) - lat0, (np.asarray(lon) - lon0) * np.cos(la))


def standardise(ds):
    """Common names for the time, latitude and longitude of any product, longitudes in
    -180 to 180 and latitudes increasing."""
    ren = {k: v for k, v in (('valid_time', 'time'), ('latitude', 'lat'), ('longitude', 'lon'),
                             ('y', 'lat'), ('x', 'lon'))
           if k in ds.variables and v not in ds.variables}
    ds = ds.rename(ren)
    if 'expver' in ds.dims:
        ds = ds.max('expver', skipna=True)
    if 'lon' in ds.coords and ds.lon.ndim == 1 and float(ds.lon.max()) > 180.0:
        ds = ds.assign_coords(lon=(((ds.lon + 180.0) % 360.0) - 180.0)).sortby('lon')
    if 'lat' in ds.coords and ds.lat.ndim == 1:
        ds = ds.sortby('lat')
    return ds


# --------------------------------------------------------------- fields
def open_gebco():
    if V.BATHY_FILE is None:
        raise FileNotFoundError('BATHY_FILE is not defined in paper_old/config.py')
    ds = standardise(xr.open_dataset(V.BATHY_FILE))
    zname = next(n for n in ('elevation', 'z', 'Band1', 'topo', 'bathymetry') if n in ds.data_vars)
    return ds, zname


def open_altimetry():
    files = sorted(V.ALT_DIR.glob('duacs_l4_sbb_*.nc'))
    if not files:
        raise FileNotFoundError(f'no altimetry in {V.ALT_DIR}')
    parts = [standardise(xr.open_dataset(f)) for f in files]
    ds = xr.concat(parts, dim='time', data_vars='minimal', coords='minimal', compat='override',
                   join='outer')
    ds = ds.sortby('time')
    _, keep = np.unique(ds['time'].values, return_index=True)
    return ds.isel(time=np.sort(keep))


# ------------------------------------------------------------ climatology
CLIM_T = ('temp', 'temperature', 'TEMP', 'T', 't', 't_an', 'thetao', 'pot_temp', 'theta')
CLIM_S = ('salt', 'salinity', 'SALT', 'S', 's', 's_an', 'so', 'sal', 'psal')
LAT_NAMES = ('lat', 'latitude', 'LAT', 'Latitude', 'nav_lat', 'lat_rho', 'y')
LON_NAMES = ('lon', 'longitude', 'LON', 'Longitude', 'nav_lon', 'lon_rho', 'x')
DEPTH_NAMES = ('depth', 'z', 'lev', 'level', 'deph', 'depth_m', 'Depth', 'DEPTH')


def _pick(ds, names, key):
    name = next((n for n in names if n in ds.data_vars), None)
    if name is None:
        name = next((n for n in ds.data_vars
                     if key in str(ds[n].attrs.get('standard_name', '')).lower()
                     or key in str(ds[n].attrs.get('long_name', '')).lower()), None)
    return name


def unique_coords(da):
    """Every one-dimensional coordinate kept once and in increasing order, since gridded
    products sometimes repeat a latitude, a longitude or a depth."""
    for d in da.dims:
        if d in da.coords and da[d].ndim == 1:
            _, idx = np.unique(da[d].values, return_index=True)
            if idx.size < da.sizes[d]:
                da = da.isel({d: np.sort(idx)})
            da = da.sortby(d)
    return da


def _depth_values(ds, da, zdim):
    """Depths in metres of the levels of a field. The coordinate of the dimension is used
    when it holds depths, else a one-dimensional variable of the same length named as a
    depth, since a dimension without coordinate only carries the index of the level."""
    n = da.sizes[zdim]
    names = [zdim] + [m for m in DEPTH_NAMES + ('depth_levels', 'depths', 'zlev', 'z_levels',
                                                 'lev_depth', 'depth_t', 'st_ocean')
                      if m != zdim]
    for name in names:
        if name not in ds.variables or ds[name].ndim != 1 or ds[name].size != n:
            continue
        v = np.abs(np.asarray(ds[name].values, float))
        if not np.all(np.isfinite(v)):
            continue
        index = np.allclose(np.sort(v), np.arange(n)) or np.allclose(np.sort(v), np.arange(1, n + 1))
        if not index:
            return v, name
    raise KeyError(f'no depth values for the {n} levels of {zdim}, variables '
                   + ' '.join(map(str, ds.variables)))


def _on_regular_grid(da, lat, lon, box, z):
    """A field of dimensions depth, y and x with latitude and longitude of one or two
    dimensions, returned on one-dimensional latitude and longitude. Two-dimensional
    coordinates that are separable are collapsed and the others are interpolated
    linearly onto a regular grid of the median spacing."""
    zdim, ydim, xdim = da.dims
    if lat.ndim == 1 and lon.ndim == 1:
        la, lo = np.asarray(lat.values, float), np.asarray(lon.values, float)
        return xr.DataArray(da.values, dims=('z', 'lat', 'lon'),
                            coords={'z': z, 'lat': la, 'lon': lo})
    la = lat.transpose(ydim, xdim).values.astype(float)
    lo = lon.transpose(ydim, xdim).values.astype(float)
    if np.allclose(la, la[:, :1], atol=1e-5) and np.allclose(lo, lo[:1, :], atol=1e-5):
        return xr.DataArray(da.values, dims=('z', 'lat', 'lon'),
                            coords={'z': z, 'lat': la[:, 0], 'lon': lo[0, :]})
    from scipy.interpolate import griddata
    lon0, lon1, lat0, lat1 = box
    keep = (la >= lat0 - 0.5) & (la <= lat1 + 0.5) & (lo >= lon0 - 0.5) & (lo <= lon1 + 0.5)
    res = float(np.nanmedian(np.abs(np.diff(la, axis=0))))
    glat = np.arange(lat0 - 0.3, lat1 + 0.3 + 1e-9, res)
    glon = np.arange(lon0 - 0.3, lon1 + 0.3 + 1e-9, res)
    GLo, GLa = np.meshgrid(glon, glat)
    vals = da.values
    out = np.full((z.size, glat.size, glon.size), np.nan)
    for k in range(z.size):
        ok = keep & np.isfinite(vals[k])
        if ok.sum() > 10:
            out[k] = griddata((lo[ok], la[ok]), vals[k][ok], (GLo, GLa), method='linear')
    return xr.DataArray(out, dims=('z', 'lat', 'lon'), coords={'z': z, 'lat': glat, 'lon': glon})


def open_climatology(box, verbose=True):
    """Seasonal temperature and salinity of the regional climatology of Sasaki et al.
    (2024) on depth, latitude and longitude, one Dataset per season inside the box."""
    import re
    files = sorted(V.CLIM_DIR.glob(V.CLIM_PATTERN)) if V.CLIM_DIR.exists() else []
    out = {}
    for f in files:
        m = re.search(r'(DJF|MAM|JJA|SON)', f.name)
        key = m.group(1) if m else f.stem
        try:
            ds = xr.open_dataset(f)
            latn = next(n for n in LAT_NAMES if n in ds.variables)
            lonn = next(n for n in LON_NAMES if n in ds.variables)
            fields = {}
            for var, names, key_name, lo_, hi_ in (('T', CLIM_T, 'temperature', -5.0, 40.0),
                                                   ('S', CLIM_S, 'salinity', 20.0, 42.0)):
                name = _pick(ds, names, key_name)
                if name is None:
                    continue
                da = ds[name]
                zdim = next((n for n in DEPTH_NAMES if n in da.dims), None)
                if zdim is None:
                    continue
                for extra in [d for d in da.dims if d not in (zdim,) + ds[latn].dims + ds[lonn].dims]:
                    da = da.mean(extra)
                if ds[latn].ndim == 1:
                    ydim, xdim = ds[latn].dims[0], ds[lonn].dims[0]
                else:
                    ydim, xdim = ds[latn].dims
                da = da.transpose(zdim, ydim, xdim)
                z, zsrc = _depth_values(ds, da, zdim)
                reg = unique_coords(_on_regular_grid(da, ds[latn], ds[lonn], box, z))
                reg = reg.where((reg > lo_) & (reg < hi_))
                lon0, lon1, lat0, lat1 = box
                fields[var] = reg.sel(lon=slice(lon0 - 0.3, lon1 + 0.3),
                                      lat=slice(lat0 - 0.3, lat1 + 0.3)).load()
            ds.close()
            if 'T' not in fields:
                raise KeyError('no temperature among ' + ' '.join(map(str, ds.data_vars)))
            if 'S' in fields:
                fields['S'] = fields['S'].interp_like(fields['T'])
            out[key] = xr.Dataset(fields)
            if verbose:
                g = out[key]
                wet = np.isfinite(g['T'].values)
                deepest = np.where(wet.any(axis=0), g.z.values[np.argmax(np.cumsum(wet, axis=0), axis=0)], np.nan)
                print(f'    {f.name}: {key}, ' + ' and '.join(g.data_vars) + f' on {g.sizes["z"]} '
                      f'depths from {float(g.z.min()):.0f} to {float(g.z.max()):.0f} m (from {zsrc}) '
                      f'and {g.sizes["lat"]} by {g.sizes["lon"]} points, deepest wet level '
                      f'{np.nanmin(deepest):.0f} to {np.nanmax(deepest):.0f} m')
        except Exception as e:                        # noqa: BLE001  reported, next file
            if verbose:
                print(f'    {f.name}: left out, {str(e)[:110]}')
    return out


def era5_files(var):
    return sorted(V.ERA5_DIR.glob(f'era5_{var}_*.nc'))


def glorys_files():
    return sorted(V.GLORYS_DIR.glob('glorys_sbb_*.nc'))


# -------------------------------------------------------------- geometry
def _from_table(path):
    t = pd.read_csv(path)
    c = dict(site=pick(t.columns, 'site'), lat=pick(t.columns, 'lat'), lon=pick(t.columns, 'lon'),
             depth=pick(t.columns, 'depth'), slope=pick(t.columns, 'slope'),
             bearing=pick(t.columns, 'isobath', 'bearing') or pick(t.columns, 'bearing')
             or pick(t.columns, 'isobath'))
    if c['site'] is None or c['lat'] is None:
        return None
    out = pd.DataFrame({'site': t[c['site']].astype(str)})
    for k in ('lat', 'lon', 'depth', 'slope', 'bearing'):
        out[k] = pd.to_numeric(t[c[k]], errors='coerce') if c[k] else np.nan
    out = out[out.site.isin(V.SITES)].drop_duplicates('site')
    return out.set_index('site') if len(out) else None


def isobath_at(ds, zname, lat0, lon0, radius_km):
    """Depth, bottom slope and isobath bearing from a plane fitted to the GEBCO depth
    inside a circle of the given radius. The bearing is the direction of the isobath,
    perpendicular to the depth gradient and measured clockwise from north, taken with
    shallow water on its left so that on this margin it points to the northeast and
    matches the along-isobath axis of the records."""
    dlat = radius_km / R_EARTH_KM
    dlon = dlat / np.cos(np.deg2rad(lat0))
    sub = ds[zname].sel(lat=slice(lat0 - dlat, lat0 + dlat), lon=slice(lon0 - dlon, lon0 + dlon))
    lo, la = np.meshgrid(sub.lon.values, sub.lat.values)
    x = (lo - lon0) * R_EARTH_KM * 1e3 * np.cos(np.deg2rad(lat0))
    y = (la - lat0) * R_EARTH_KM * 1e3
    inside = np.hypot(x, y) <= radius_km * 1e3
    h = -np.asarray(sub.values, float)
    ok = np.isfinite(h) & inside
    if ok.sum() < 8:
        return np.nan, np.nan, np.nan
    A = np.c_[x[ok], y[ok], np.ones(int(ok.sum()))]
    gx, gy, h0 = np.linalg.lstsq(A, h[ok], rcond=None)[0]
    slope = float(np.hypot(gx, gy))
    downslope = float(np.rad2deg(np.arctan2(gx, gy)) % 360.0)
    return float(h0), slope, float((downslope - 90.0) % 360.0)


def _from_gebco(positions):
    ds, zname = open_gebco()
    rows = []
    for site, r in positions.iterrows():
        depth, slope, bearing = isobath_at(ds, zname, r.lat, r.lon, V.ISOBATH_RADIUS_KM)
        rows.append(dict(site=site, lat=r.lat, lon=r.lon, depth=depth, slope=slope,
                         bearing=bearing))
    ds.close()
    return pd.DataFrame(rows).set_index('site')


def geometry(verbose=True):
    """Position, depth, bottom slope and isobath bearing of every site."""
    for name in ('site_geometry.csv', 'external_site_bathymetry.csv'):
        p = find(name, TABLE_DIRS)
        if p is None:
            continue
        geo = _from_table(p)
        if geo is not None and geo[['depth', 'slope', 'bearing']].notna().all(axis=None):
            if verbose:
                print(f'  geometry from {p}')
            return geo, str(p)
    inv = find('01_inventory.csv', TABLE_DIRS) or find('01_site_campaign_positions.csv', TABLE_DIRS)
    if inv is None:
        raise FileNotFoundError('no table with the mooring positions in '
                                + ', '.join(str(d) for d in TABLE_DIRS))
    t = pd.read_csv(inv)
    t = t[t[pick(t.columns, 'site')].isin(V.SITES)]
    pos = t.groupby(pick(t.columns, 'site'))[[pick(t.columns, 'lat'), pick(t.columns, 'lon')]].median()
    pos.columns = ['lat', 'lon']
    pos.index.name = 'site'
    geo = _from_gebco(pos)
    if verbose:
        print(f'  geometry computed from the positions of {inv} and the GEBCO gradient')
    return geo, f'{inv} and GEBCO'


def lines(geo, verbose=True):
    """Distance of every site along its line and the offshore unit vector of the line."""
    p = find('lines.csv', TABLE_DIRS)
    if p is not None:
        t = pd.read_csv(p)
        if {'line', 'site', 'along_km', 'unit_east', 'unit_north', 'origin_lat',
                'origin_lon'}.issubset(t.columns):
            if verbose:
                print(f'  lines from {p}')
            return t, str(p)
    rows = []
    for line, members in V.LINES.items():
        have = [s for s in members if s in geo.index]
        if len(have) < 2:
            continue
        a, b = geo.loc[have[0]], geo.loc[have[-1]]
        c = np.cos(np.deg2rad(a.lat))
        vec = np.array([(b.lon - a.lon) * R_EARTH_KM * c, (b.lat - a.lat) * R_EARTH_KM])
        unit = vec / np.hypot(*vec)
        for s in have:
            r = geo.loc[s]
            d = np.array([(r.lon - a.lon) * R_EARTH_KM * c, (r.lat - a.lat) * R_EARTH_KM])
            rows.append(dict(line=line, site=s, along_km=float(d @ unit),
                             cross_km=float(d @ np.array([-unit[1], unit[0]])),
                             lat=r.lat, lon=r.lon, origin_lat=a.lat, origin_lon=a.lon,
                             unit_east=float(unit[0]), unit_north=float(unit[1])))
    if verbose:
        print('  lines built from the site positions, inner site to outer site')
    return pd.DataFrame(rows), 'built from the positions'


# --------------------------------------------------------------- records
def lp_files(verbose=True):
    """Low-passed record of every site that has one."""
    out = {}
    for site in V.SITES:
        p = find(f'{site}_lp.nc', DATA_DIRS)
        if p is not None:
            out[site] = p
    if verbose:
        where = {str(p.parent) for p in out.values()}
        print(f'  {len(out)} low-passed records in ' + ', '.join(sorted(where)) if out
              else '  no <site>_lp.nc found in ' + ', '.join(str(d) for d in DATA_DIRS))
    return out


def _screen():
    p = V.DIR_TAB / 'record_screen.csv'
    if p.exists() and p.stat().st_size > 5:
        t = pd.read_csv(p, dtype={'campaign': str})
        return t
    return None


def _nearest_level(levels, level):
    levels = np.asarray(levels, float)
    if levels.size == 0:
        return None
    k = int(np.argmin(np.abs(levels - level)))
    return float(levels[k]) if abs(levels[k] - level) <= V.LEVEL_MATCH_M else None


def primary_records(tab, levels):
    """The screen table reduced to the record nearest to each level of the low-passed series
    in each campaign, so that a neighbouring record (185 m beside 195 m, 85 m beside 95 m)
    never stands for a level that another record holds."""
    rows = []
    for _, r in tab.iterrows():
        lev = _nearest_level(levels, r.level)
        if lev is not None:
            rows.append(dict(r, lp_level=lev, distance=abs(r.level - lev)))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t.sort_values('distance').drop_duplicates(['site', 'campaign', 'lp_level'], keep='first')
    return t


def load_lp(path, screen=True, extra=True, verbose=False):
    """One low-passed record as plain arrays. u is along the isobath and positive toward
    the northeast against the Brazil Current, v across it and positive onshore.

    The velocity of every record that fails the ring test of screen_records.py is set to
    NaN over its campaign, the temperature being kept, and the records recovered by
    recover_records.py from data_qc are merged in, so that every figure reads the same
    accepted set."""
    path = Path(path)
    site = path.stem.split('_')[0]
    ds = xr.open_dataset(path)
    t = pd.DatetimeIndex(ds['time'].values)
    out = dict(time=t, campaign=np.asarray(ds['campaign'].values).astype(str),
               level=np.asarray(ds['level'].values, float),
               u=np.array(ds['u'].values, float), v=np.array(ds['v'].values, float),
               T=np.array(ds['T'].values, float) if 'T' in ds else
               np.full(ds['u'].shape, np.nan))
    ds.close()
    out['campaign'] = np.where(np.isin(out['campaign'], ('', 'nan', 'None')), '', out['campaign'])
    notes = []
    if extra:
        f = V.DIR_REC / f'{site}_extra.nc'
        if f.exists():
            ex = xr.open_dataset(f)
            et = pd.DatetimeIndex(ex['time'].values)
            ecamp = np.asarray(ex['campaign'].values).astype(str)
            pos = t.get_indexer(et, method='nearest', tolerance=pd.Timedelta('30min'))
            ok = pos >= 0
            for j, level in enumerate(np.asarray(ex['level'].values, float)):
                if level not in out['level']:
                    for key in ('u', 'v', 'T'):
                        out[key] = np.c_[out[key], np.full(t.size, np.nan)]
                    out['level'] = np.r_[out['level'], level]
                k = int(np.flatnonzero(out['level'] == level)[0])
                for key in ('u', 'v', 'T'):
                    col = out[key][:, k]
                    val = np.asarray(ex[key].values[:, j], float)
                    fill = ok & np.isfinite(val)
                    target = pos[fill]
                    empty = ~np.isfinite(col[target])
                    col[target[empty]] = val[fill][empty]
                    out[key][:, k] = col
                out['campaign'][pos[ok & (ecamp != '')]] = np.where(
                    out['campaign'][pos[ok & (ecamp != '')]] == '',
                    ecamp[ok & (ecamp != '')], out['campaign'][pos[ok & (ecamp != '')]])
                notes.append(f'{level:.0f} m recovered')
            ex.close()
            order = np.argsort(out['level'])
            out['level'] = out['level'][order]
            for key in ('u', 'v', 'T'):
                out[key] = out[key][:, order]
    if screen:
        tab = _screen()
        if tab is not None:
            prim = primary_records(tab[tab.site == site], out['level'])
            for _, r in (prim[prim.accepted == 0] if len(prim) else prim).iterrows():
                level = r.lp_level
                k = int(np.flatnonzero(out['level'] == level)[0])
                m = out['campaign'] == str(r.campaign)
                if m.any() and np.isfinite(out['u'][m, k]).any():
                    out['u'][m, k] = np.nan
                    out['v'][m, k] = np.nan
                    notes.append(f'{level:.0f} m campaign {r.campaign} masked')
    if verbose and notes:
        print(f'    {site}: ' + ', '.join(notes))
    return out


def main():
    print('inputs of the third version')
    print(f'  settings module   {V.__file__}')
    print(f'  old configuration {"found" if V.C is not None else "not found, defaults in use"}')
    print(f'  bathymetry        {V.BATHY_FILE}')
    for name, d, pattern in (('altimetry', V.ALT_DIR, 'duacs_l4_sbb_*.nc'),
                             ('ERA5', V.ERA5_DIR, 'era5_*.nc'),
                             ('GLORYS12V1', V.GLORYS_DIR, 'glorys_sbb_*.nc'),
                             ('OISST', V.SST_DIR, 'oisst_sbb_*.nc'),
                             ('quality controlled', V.QC_DIR, '*_qc.nc'),
                             ('climatology', V.CLIM_DIR, V.CLIM_PATTERN)):
        n = len(list(Path(d).glob(pattern))) if Path(d).exists() else 0
        print(f'  {name:<18} {n:4d} files in {d}')
    geo, _ = geometry()
    print(geo.round(4).to_string())
    ln, _ = lines(geo)
    print(ln.round(3).to_string(index=False))
    lp_files()


if __name__ == '__main__':
    main()