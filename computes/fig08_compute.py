"""
Quantities of Figure 7, the synthesis of the shelf-break bottom boundary layer and the
supply of central water measured by the moorings.

The schematic draws only what the records measured, so every number it quotes is read
here from the tables of Figures 2 to 6 and written to one table: the attenuation and
veering of the near-bottom flow at FUB3 in the two states (Figure 3), the stratification
of the layer and its share of unstable or mixed days (Figure 4), the restoring time of the
layer, the observed arrest time, the wedge time and the share of the Ekman transport that
builds the buoyancy (Figures 2 and 4), the phase speed of the coastal trapped wave and the
share of the near-bottom variance carried by the heave (Figure 5), and the lags across the
lines and the near-bottom flux of central water (Figure 6).

Outputs in paper_v3
    tables/fig07_numbers.csv   the quoted numbers with their intervals
    logs/fig07_compute.txt

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd

import config_v3 as V

LOG = []


def say(msg=''):
    print(msg)
    LOG.append(msg)


def table(name):
    p = V.DIR_TAB / f'{name}.csv'
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 5 else pd.DataFrame()


def main():
    V.make_dirs()
    say('figure 7, synthesis')
    num = []

    def put(key, value, lo=np.nan, hi=np.nan, source=''):
        num.append(dict(key=key, value=value, lo=lo, hi=hi, source=source))
        say(f'  {key}: {value:.3g}' + (f' ({lo:.3g} to {hi:.3g})' if np.isfinite(lo) else '') + f', {source}')

    r3 = table('fig03_response')
    for site in V.SITES_200:
        for state, tag in (('with the current', 'up'), ('against the current', 'dn')):
            q = r3[(r3.site == site) & (r3.lower == V.PAIRS['bottom'][1]) & (r3.state == state) & (r3.half == 'all')] \
                if len(r3) else pd.DataFrame()
            if len(q):
                q = q.iloc[0]
                put(f'{site}_gain_{tag}', q.gain, q.gain_lo, q.gain_hi, 'Figure 3')
                put(f'{site}_veer_{tag}', q.veer, q.veer_lo, q.veer_hi, 'Figure 3')
    mx = table('fig04_mixing')
    for site in V.SITES_200:
        for state, tag in (('with the current', 'up'), ('against the current', 'dn')):
            q = mx[(mx.site == site) & (mx.kind == state)] if len(mx) else pd.DataFrame()
            if len(q):
                q = q.iloc[0]
                put(f'{site}_N_{tag}', q.N_median, q.N_q1, q.N_q3, 'Figure 4')
                put(f'{site}_mixed_{tag}', q.unstable, q.unstable_lo, q.unstable_hi, 'Figure 4')
    md = table('fig04_model')
    for site in V.SITES_200:
        q = md[(md.site == site) & (md.forcing == 'interior')] if len(md) else pd.DataFrame()
        if len(q):
            q = q.iloc[0]
            put(f'{site}_tau_restoring_h', q.tau_h, q.tau_lo, q.tau_hi, 'Figure 4')
            put(f'{site}_share_ekman_up', q.eff_up, q.eff_up_lo, q.eff_up_hi, 'Figure 4')
            put(f'{site}_share_ekman_dn', q.eff_dn, q.eff_dn_lo, q.eff_dn_hi, 'Figure 4')
    dc = table('fig04_decay')
    for site in V.SITES_200:
        q = dc[(dc.site == site) & (dc.kind == 'with the current')] if len(dc) else pd.DataFrame()
        if len(q) and np.isfinite(q.t_e_h.iloc[0]):
            q = q.iloc[0]
            put(f'{site}_t_arrest_h', q.t_e_h, q.x_e_lo * q.tau_median_h, q.x_e_hi * q.tau_median_h, 'Figure 4')
    sm = table('fig02_summary')
    for site in V.SITES_200:
        q = sm[(sm.site == site) & (sm.half == 'all')] if len(sm) else pd.DataFrame()
        if len(q):
            q = q.iloc[0]
            put(f'{site}_tau_wedge_h', q.tau_w_median_h, q.tau_w_lo_h, q.tau_w_hi_h, 'Figure 2')
            put(f'{site}_tau_canonical_h', q.tau_c_median_h, q.tau_c_q1_h, q.tau_c_q3_h, 'Figure 2')
            put(f'{site}_s', q.s_median, q.s_q1, q.s_q3, 'Figure 2')
    ct = table('fig05_ctw')
    for _, q in (ct.iterrows() if len(ct) else []):
        put(f'{q.site}_ctw_c', q.c, q.c_min_cv, q.c_max_cv, 'Figure 5')
        put(f'{q.site}_ctw_r_cv', q.r_cv, source='Figure 5')
    pt = table('fig05_partition')
    for _, q in (pt.iterrows() if len(pt) else []):
        put(f'{q.site}_heave_share', q.transmitted, source='Figure 5')
        put(f'{q.site}_beta', q.beta, source='Figure 5')
    lg = table('fig06_lags')
    moor = lg[lg.site != 'coastal index'] if len(lg) else pd.DataFrame()
    if len(moor):
        put('lag_min_days', moor.lag.min(), source='Figure 6')
        put('lag_max_days', moor.lag.max(), source='Figure 6')
        put('lag_distance_min_km', moor.distance_km.min(), source='Figure 6')
        put('lag_distance_max_km', moor.distance_km.max(), source='Figure 6')
    fx = table('fig06_flux')
    for site in V.SITES:
        q = fx[(fx.site == site) & (fx.T_c == V.SACW_T) & (fx.rotation == 0.0)] if len(fx) else pd.DataFrame()
        if len(q):
            q = q.iloc[0]
            put(f'{site}_flux_eddy', q.flux_eddy, q.flux_eddy_lo, q.flux_eddy_hi, 'Figure 6')
            put(f'{site}_flux_mean', q.flux_mean, q.flux_mean_lo, q.flux_mean_hi, 'Figure 6')
    pd.DataFrame(num).to_csv(V.DIR_TAB / 'fig07_numbers.csv', index=False)
    (V.DIR_LOG / 'fig07_compute.txt').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()