"""
Records accepted by the screen that the first version left out of its low-passed
series, rebuilt from data_qc with the processing of the first version.

The samples are averaged hourly, gaps up to GAP_FILL_H hours are filled
linearly, the series is low-passed with the Lanczos filter of LP_CUTOFF_H hours
and LP_HALF_WINDOW_H hours of half window (Duchon 1979), and the velocity is
rotated into the frame of the local isobath.

The rotation of the first version is measured, never assumed. The current
meters record against magnetic north, so the first version applied the magnetic
declination, about 21 degrees west off southeastern Brazil in 2002, and the
isobath bearing of its own tables. For every record the low-passed series
already holds, the complex regression

    w_old = c w_new,    w = u + iv,

of the old series on its reconstruction gives the extra rotation arg(c) and the
gain |c|. The median angle of each site is applied to the reconstruction, the
comparison is repeated, and a record is recovered only when the corrected
reconstruction reproduces the old series at that site. Only the records listed in
RECOVER of config_v3 are rebuilt, and the other candidates are listed for a
decision one by one.

    python recover_records.py     writes data/records/<site>_extra.nc,
                                  tables/recover_validation.csv and
                                  tables/recover_rotation.csv

Created by Arian Dialectaquiz.
"""
import re

import numpy as np
import pandas as pd
import xarray as xr

import config_v3 as V
import inputs_v3 as I
import physics_v3 as P

PATTERN = re.compile(r'DEPROAS_(?P<site>[A-Z0-9]+)_(?P<campaign>\d+)_(?P<level>\d{3,4})_qc\.nc$')
MIN_R_U, MIN_R_V = 0.95, 0.90
MAX_ANGLE_SPREAD = 6.0            # degrees, interquartile range of the angles of one site
TEMP_NAMES = ('STEMP', 'TEMP', 'temp', 'T', 'temperature', 'SST', 'TX')


def temperature_name(ds):
    name = next((n for n in TEMP_NAMES if n in ds.data_vars), None)
    if name is None:
        name = next((n for n in ds.data_vars if 'temp' in n.lower()
                     or 'temperature' in str(ds[n].attrs.get('long_name', '')).lower()), None)
    return name


def rebuild(f, bearing):
    """Hourly low-passed velocity in the isobath frame of the given bearing, and temperature."""
    ds = xr.open_dataset(f)
    east = next((n for n in ('VE', 'u', 'U', 'east') if n in ds.data_vars), None)
    north = next((n for n in ('VN', 'v', 'V', 'north') if n in ds.data_vars), None)
    temp = temperature_name(ds)
    t = pd.DatetimeIndex(np.asarray(ds['time'].values).ravel())

    def flat(name):
        x = np.asarray(ds[name].values, float)
        return x.ravel() if x.size == t.size else x.reshape(t.size, -1)[:, 0]

    e, n = flat(east), flat(north)
    T = flat(temp) if temp else np.full(t.size, np.nan)
    ds.close()
    if np.nanpercentile(np.hypot(e, n), 99) > 5.0:          # centimetres per second
        e, n = e / 100.0, n / 100.0
    df = pd.DataFrame({'e': e, 'n': n, 'T': T}, index=t).sort_index()
    df = df[~df.index.duplicated()]
    hourly = df.resample('1h').mean()
    hourly = hourly.interpolate(limit=V.GAP_FILL_H, limit_area='inside')
    low = hourly.apply(lambda c: pd.Series(P.lowpass(c.values), index=c.index))
    u, v = P.to_isobath(low.e.values, low.n.values, bearing)
    return pd.DataFrame({'u': u, 'v': v, 'T': low['T'].values}, index=low.index), temp


def rotate(rec, angle_deg):
    w = (rec.u.values + 1j * rec.v.values) * np.exp(1j * np.deg2rad(angle_deg))
    out = rec.copy()
    out['u'], out['v'] = w.real, w.imag
    return out


def compare(old, new):
    row = {}
    for key in ('u', 'v', 'T'):
        ok = old[key].notna() & new[key].notna()
        if ok.sum() > 48 and old[key][ok].std() > 0 and new[key][ok].std() > 0:
            row[f'r_{key}'] = float(np.corrcoef(old[key][ok], new[key][ok])[0, 1])
            row[f'rms_{key}'] = float(np.sqrt(np.mean((old[key][ok] - new[key][ok]) ** 2)))
            row[f'bias_{key}'] = float(np.mean(new[key][ok] - old[key][ok]))
        else:
            row[f'r_{key}'] = row[f'rms_{key}'] = row[f'bias_{key}'] = np.nan
    return row


def main():
    V.make_dirs()
    V.DIR_REC.mkdir(parents=True, exist_ok=True)
    geo, _ = I.geometry(verbose=False)
    lp = I.lp_files(verbose=False)
    scr = pd.read_csv(V.DIR_TAB / 'record_screen.csv', dtype={'campaign': str})
    prim = pd.concat([I.primary_records(g, V.LEVELS[s_]) for s_, g in scr.groupby('site')
                      if s_ in V.LEVELS], ignore_index=True)
    left = set(scr.file) - set(prim.file)
    for f_ in sorted(left):
        print(f'  {f_} stands beside a nearer record of the same level and campaign, left out')
    scr = prim[prim.accepted == 1]
    held, missing = [], []
    first_temp = None
    for _, r in scr.iterrows():
        site = r.site
        if site not in lp or site not in geo.index:
            continue
        d = I.load_lp(lp[site], screen=False, extra=False)
        level = r.lp_level if r.lp_level in d['level'] else I._nearest_level(d['level'], r.level)
        m = d['campaign'] == str(r.campaign)
        k = list(d['level']).index(level) if level is not None else None
        rec, tname = rebuild(V.QC_DIR / r.file, float(geo.loc[site, 'bearing']))
        first_temp = first_temp or tname
        if k is not None and m.any() and np.isfinite(d['u'][m, k]).sum() > 240:
            # the old series passed once more through the same filter, which leaves a
            # low-passed series unchanged and removes whatever it still holds above the cutoff
            old = pd.DataFrame({key: P.lowpass(np.where(m, d[key][:, k], np.nan))
                                for key in ('u', 'v', 'T')}, index=d['time'])[m]
            new = rec.reindex(old.index, method='nearest', tolerance=pd.Timedelta('30min'))
            held.append((r, old, new))
        else:
            missing.append((r, rec))
    print(f'temperature read from the variable {first_temp}')

    # ---------------------------------------------- the rotation of the first version
    rot = []
    for r, old, new in held:
        wo, wn = old.u.values + 1j * old.v.values, new.u.values + 1j * new.v.values
        ok = np.isfinite(wo) & np.isfinite(wn)
        if ok.sum() < 240:
            continue
        c = np.sum(np.conj(wn[ok]) * wo[ok]) / np.sum(np.abs(wn[ok]) ** 2)
        rot.append(dict(site=r.site, campaign=r.campaign, level=r.level,
                        angle=float(np.degrees(np.angle(c))), gain=float(abs(c))))
    rot = pd.DataFrame(rot)
    rot.to_csv(V.DIR_TAB / 'recover_rotation.csv', index=False)
    print('\nextra rotation of the first version, old = c new, by site')
    angle_of = {}
    for site, g in rot.groupby('site'):
        q = np.percentile(g.angle, [25, 50, 75])
        angle_of[site] = (q[1], q[2] - q[0], float(g.gain.median()))
        print(f'  {site}: median {q[1]:+.1f} deg, interquartile {q[0]:+.1f} to {q[2]:+.1f}, gain '
              f'{g.gain.median():.2f}, {len(g)} records')
    q = np.percentile(rot.angle, [25, 50, 75]) if len(rot) else [np.nan] * 3
    print(f'  all sites: median {q[1]:+.1f} deg, interquartile {q[0]:+.1f} to {q[2]:+.1f}')

    # ------------------------------------------------ validation after the rotation
    rows = []
    for r, old, new in held:
        a = angle_of.get(r.site, (0.0, np.nan, np.nan))[0]
        row = dict(site=r.site, campaign=r.campaign, level=r.level, angle_applied=a)
        row.update({f'{k}_raw': v for k, v in compare(old, new).items() if k.startswith('r_')})
        row.update(compare(old, rotate(new, a)))
        rows.append(row)
    chk = pd.DataFrame(rows)
    chk.to_csv(V.DIR_TAB / 'recover_validation.csv', index=False)
    print('\nreconstruction of the held records, correlation before and after the rotation')
    print(chk[['site', 'campaign', 'level', 'angle_applied', 'r_u_raw', 'r_v_raw', 'r_u', 'r_v',
               'bias_u', 'bias_v', 'r_T', 'bias_T']].round(3).to_string(index=False))

    chosen = {(s_, str(c_), int(l_)) for s_, c_, l_ in V.RECOVER}
    by_site = {}
    for r, rec in missing:
        rec = rec.dropna(how='all')
        if rec.u.notna().sum() < 240:
            continue
        if (r.site, str(r.campaign), int(r.level)) not in chosen:
            print(f'  candidate left aside, {r.site} campaign {r.campaign} at {r.level} m, '
                  f'{int(rec.u.notna().sum())} hours (add it to RECOVER in config_v3 to rebuild it)')
            continue
        g = chk[chk.site == r.site]
        a, spread, gain = angle_of.get(r.site, (np.nan, np.nan, np.nan))
        good = len(g) >= 3 and g.r_u.median() >= MIN_R_U and g.r_v.median() >= MIN_R_V \
            and spread <= MAX_ANGLE_SPREAD
        if not good:
            print(f'  {r.site} campaign {r.campaign} at {r.level} m not recovered, the corrected '
                  f'reconstruction of {r.site} gives a median correlation of {g.r_u.median():.3f} '
                  f'for u and {g.r_v.median():.3f} for v with an angle spread of {spread:.1f} deg')
            continue
        rec = rotate(rec, a)
        print(f'  recovered {r.site} campaign {r.campaign} at {r.level} m, {int(rec.u.notna().sum())} '
              f'hours, rotated by {a:+.1f} deg')
        by_site.setdefault(r.site, []).append((r, rec))
    for site, items in by_site.items():
        times = pd.DatetimeIndex(sorted(set().union(*[set(rec.index) for _, rec in items])))
        levels = sorted({float(I._nearest_level(V.LEVELS[site], r.level) or r.level) for r, _ in items})
        arr = {key: np.full((times.size, len(levels)), np.nan) for key in ('u', 'v', 'T')}
        camp = np.full(times.size, '', dtype=object)
        for r, rec in items:
            j = levels.index(float(I._nearest_level(V.LEVELS[site], r.level) or r.level))
            pos = times.get_indexer(rec.index)
            for key in ('u', 'v', 'T'):
                arr[key][pos, j] = rec[key].values
            camp[pos[rec.u.notna().values]] = str(r.campaign)
        ds = xr.Dataset({k: (('time', 'level'), a_) for k, a_ in arr.items()} |
                        {'campaign': (('time',), camp.astype(str))},
                        coords={'time': times, 'level': np.array(levels)})
        ds.attrs.update(source='data_qc rebuilt by recover_records.py',
                        bearing=float(geo.loc[site, 'bearing']), extra_rotation=angle_of[site][0])
        out = V.DIR_REC / f'{site}_extra.nc'
        if out.exists():
            out.unlink()
        ds.to_netcdf(out)
        print(f'  written {out}')


if __name__ == '__main__':
    main()