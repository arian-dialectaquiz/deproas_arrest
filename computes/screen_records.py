"""
Screening of every DEPROAS current meter record for an offset of its acoustic paths.

A current meter whose acoustic path carries an offset records a speed that never
approaches zero while the flow turns through every bearing, so the lowest speeds
of the record sit close to its median. The ring index, the first percentile of
the speed over its median, measures that. Sound records of this array sit
between 0.09 and 0.16 and offset records between 0.30 and 0.86, with nearly
constant speeds of 3 to 6 cm s-1, so RING_MAX = 0.25 separates the two groups.
The screen acts on the velocity alone, since the thermistor is independent of
the acoustic paths, and every script of the third version applies it through
inputs_v3.load_lp. A median speed above MAX_MEDIAN_SPEED marks a unit or sensor
fault and is rejected as well.

    python screen_records.py      writes tables/record_screen.csv

Created by Arian Dialectaquiz.
"""
import re

import numpy as np
import pandas as pd
import xarray as xr

import config_v3 as V

PATTERN = re.compile(r'DEPROAS_(?P<site>[A-Z0-9]+)_(?P<campaign>\d+)_(?P<level>\d{3,4})_qc\.nc$')


def components(ds):
    east = next((n for n in ('VE', 'u', 'U', 'east') if n in ds.data_vars), None)
    north = next((n for n in ('VN', 'v', 'V', 'north') if n in ds.data_vars), None)
    return east, north


def main():
    V.make_dirs()
    rows = []
    for f in sorted(V.QC_DIR.glob('DEPROAS_*_qc.nc')):
        m = PATTERN.match(f.name)
        if not m or m['site'] not in V.SITES:
            continue
        ds = xr.open_dataset(f)
        east, north = components(ds)
        if east is None or north is None:
            ds.close()
            continue
        ue = np.asarray(ds[east].values, float).ravel()
        vn = np.asarray(ds[north].values, float).ravel()
        ds.close()
        ok = np.isfinite(ue) & np.isfinite(vn)
        if ok.sum() < 100:
            continue
        ue, vn = ue[ok], vn[ok]
        if np.nanpercentile(np.hypot(ue, vn), 99) < 5.0:          # metres per second
            ue, vn = 100.0 * ue, 100.0 * vn
        spd = np.hypot(ue, vn)
        brg = np.degrees(np.arctan2(ue, vn)) % 360.0
        cover = np.histogram(brg, bins=36, range=(0, 360))[0] / spd.size
        q = np.percentile(spd, [1, 5, 50])
        ring = float(q[0] / q[2])
        rows.append(dict(file=f.name, site=m['site'], campaign=str(int(m['campaign'])),
                         level=int(m['level']), valid=ok.mean(), p01=q[0], p05=q[1], p50=q[2],
                         slow=float(np.mean(spd < 2.0)), bearings=float(np.mean(cover >= 0.005)),
                         steadiness=float(np.hypot(ue.mean(), vn.mean()) / spd.mean()),
                         ring=ring, accepted=int(ring < V.RING_MAX and q[2] < V.MAX_MEDIAN_SPEED),
                         reason='offset' if ring >= V.RING_MAX else
                         ('speed' if q[2] >= V.MAX_MEDIAN_SPEED else '')))
    tab = pd.DataFrame(rows).sort_values(['site', 'level', 'campaign'])
    tab.to_csv(V.DIR_TAB / 'record_screen.csv', index=False)
    print(tab.drop(columns='file').round(3).to_string(index=False))
    bad = tab[tab.accepted == 0]
    print(f'\n{len(bad)} of {len(tab)} records are rejected, their velocity is masked:')
    for _, r in bad.iterrows():
        print(f'  {r.site} campaign {r.campaign} at {r.level} m, {r.reason}, ring {r.ring:.2f}, '
              f'median speed {r.p50:.1f} cm s-1')


if __name__ == '__main__':
    main()