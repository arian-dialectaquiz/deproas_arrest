"""
Signature of an acoustic velocity offset in the lower records of the 200 m sites.

The first version left out records whose speed never approaches zero while the
flow turns through every bearing, the signature of an offset in one acoustic
path of the current meter. FBG3 lost its lowest meter to that test, and the
lowest meter at the steepest slope would enter the core of Figure 3. This
script sets every 155 and 194 m record of data_qc side by side, so that a
retained record can be compared with a rejected one before deciding whether
any campaign of FBG3 at 194 m can be recovered.

For each record, from the raw samples of the quality controlled file:
    valid        fraction of samples with both components
    p01 p05 p50  percentiles of the speed (cm s-1)
    slow         fraction of samples slower than 2 cm s-1
    bearings     fraction of the 36 sectors of 10 degrees that hold at least
                 0.5 % of the samples
    steadiness   speed of the mean vector over the mean speed
    ring         the first percentile of the speed over its median, which stays
                 near zero for a sound record and grows under an offset

    python check_bottom_records.py      writes tables/check_bottom_records.csv

Created by Arian Dialectaquiz.
"""
import re

import numpy as np
import pandas as pd
import xarray as xr

import config_v3 as V

PATTERN = re.compile(r'DEPROAS_(?P<site>[A-Z0-9]+)_(?P<campaign>\d+)_(?P<level>\d{3,4})_qc\.nc$')


def main():
    V.make_dirs()
    rows = []
    for f in sorted(V.QC_DIR.glob('DEPROAS_*_qc.nc')):
        m = PATTERN.match(f.name)
        if not m or m['site'] not in V.SITES_200 or int(m['level']) < 150:
            continue
        ds = xr.open_dataset(f)
        east = next((n for n in ('VE', 'u', 'U', 'east') if n in ds.data_vars), None)
        north = next((n for n in ('VN', 'v', 'V', 'north') if n in ds.data_vars), None)
        if east is None or north is None:
            print(f'  {f.name}: no velocity components among {list(ds.data_vars)}')
            continue
        ue, vn = ds[east].values.astype(float), ds[north].values.astype(float)
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
        rows.append(dict(site=m['site'], campaign=int(m['campaign']), level=int(m['level']),
                         valid=ok.mean(), p01=q[0], p05=q[1], p50=q[2],
                         slow=float(np.mean(spd < 2.0)), bearings=float(np.mean(cover >= 0.005)),
                         steadiness=float(np.hypot(ue.mean(), vn.mean()) / spd.mean()),
                         ring=float(q[0] / q[2])))
    tab = pd.DataFrame(rows).sort_values(['level', 'site', 'campaign'])
    tab.to_csv(V.DIR_TAB / 'check_bottom_records.csv', index=False)
    print(tab.round(3).to_string(index=False))


if __name__ == '__main__':
    main()