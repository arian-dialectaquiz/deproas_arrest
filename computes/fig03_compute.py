"""
Quantities of Figure 3, the response of the near-bottom flow to the interior flow.

The records measure velocity at four heights and no stress, so the arrest is
tested through the velocity itself, with diagnostics that do not depend on the
frame of the isobaths or on a drag law.

 1  profiles     mean along-isobath and cross-isobath velocity and temperature at
                 every level of the 200 m sites, with the interval of the mean
                 from the effective number of independent samples, and the range
                 of the mean cross-isobath velocity when the frame is rotated by
                 the spread of the isobath bearing of Figure 1
 2  response     complex regression of the velocity at the lower level of a pair
                 on the velocity at its upper level,

                     w_b = a + c_up w_i 1_up + c_down w_i 1_down + e,

                 with w = u + iv, a common complex offset a and one complex gain
                 for the states in which the interior flow runs with the Brazil
                 Current (upslope Ekman transport) and one for the states against
                 it (downslope), the states being the runs of one sign of the flow
                 at 155 m. The modulus of c is the attenuation and its phase the
                 veering, clockwise negative, both invariant under any rotation of
                 the frame (Kundu 1976). The same fit over 85 to 155 m, on the
                 same hours, is the control of the baroclinic shear of the
                 current, which acts on both pairs while friction acts on the
                 lower one. Intervals come from resampling whole states.
 3  speeds       median speed at the lowest level in bins of the interior speed,
                 for each kind of state, which separates the linear response of
                 an unarrested layer from the saturation of an arrested one
 4  references   the unarrested layer from an eddy viscosity kappa u* z (1 - z/h)
                 with h the neutral thickness 0.4 u*/|f| and the stratified
                 thickness of Weatherly and Martin (1978), over the roughness
                 lengths, the interior speeds and the heights of the design and
                 of GEBCO, and the arrested layer, whose thermal wind shear
                 N^2 alpha / |f| caps the speed at the lowest meter at
                 N^2 alpha z_b / |f| (MacCready and Rhines 1993, Brink and Lentz
                 2010a)

Outputs in paper_v3
    tables/fig03_profiles.csv
    tables/fig03_response.csv     gains, veering and offset by site, pair and state
    tables/fig03_asymmetry.csv    upslope minus downslope attenuation and veering
    tables/fig03_bins.csv         binned speeds
    tables/fig03_reference.csv    unarrested and arrested references
    data/fig03_hist.nc            joint histogram of the two speeds
    logs/fig03_compute.txt

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


def say(msg=''):
    print(msg)
    LOG.append(msg)


def states_of(d, lv):
    """State label of every hour, +1 against the current, -1 with it, 0 outside a state,
    and the index of the state it belongs to."""
    u = d['u'][:, lv.index(V.CONDITION_LEVEL)]
    label = np.zeros(u.size)
    idx = np.full(u.size, -1)
    for k, (a, b, sgn) in enumerate(P.runs_of_sign(u, d['campaign'])):
        label[a:b] = sgn
        idx[a:b] = k
    return label, idx


def fit(wi, wb, up):
    X = np.c_[np.ones(wi.size), wi * up, wi * (~up)].astype(complex)
    coef = np.linalg.lstsq(X, wb, rcond=None)[0]
    res = wb - X @ coef
    return coef, 1.0 - np.sum(np.abs(res) ** 2) / np.sum(np.abs(wb - wb.mean()) ** 2)


def response(site, pair, d, lv, label, idx, rng, geo, half=None):
    ku, kl = lv.index(pair[0]), lv.index(pair[1])
    wi = d['u'][:, ku] + 1j * d['v'][:, ku]
    wb = d['u'][:, kl] + 1j * d['v'][:, kl]
    ok = np.isfinite(wi) & np.isfinite(wb) & (label != 0)
    if half is not None:
        ok &= I.halves_of(d['time']) == half
    if ok.sum() < 240:
        return None
    up = label[ok] < 0
    if up.sum() < 240 or (~up).sum() < 240:
        say(f'    {site} {pair[0]} to {pair[1]} m ({half or "all"}): {int(up.sum())} hours with the '
            f'current and {int((~up).sum())} against it, too few for the two gains')
        return None
    coef, r2 = fit(wi[ok], wb[ok], up)
    ids = idx[ok]
    uniq = np.unique(ids)
    rows_of = {k: np.flatnonzero(ids == k) for k in uniq}
    boot = []
    for _ in range(V.N_BOOT):
        pick = rng.choice(uniq, uniq.size)
        j = np.concatenate([rows_of[k] for k in pick])
        if up[j].sum() < 24 or (~up[j]).sum() < 24:
            continue
        boot.append(fit(wi[ok][j], wb[ok][j], up[j])[0])
    boot = np.array(boot)
    out = []
    for name, k, sel in (('with the current', 1, up), ('against the current', 2, ~up)):
        c, cb = coef[k], boot[:, k]
        dphi = np.degrees(np.angle(cb / c))
        ratio = wb[ok][sel] / wi[ok][sel]
        strong = np.abs(wi[ok][sel]) >= V.MIN_SPEED
        out.append(dict(site=site, pair=f'{pair[0]} to {pair[1]} m', upper=pair[0], lower=pair[1],
                        state=name, half=half or 'all', n_hours=int(sel.sum()),
                        n_states=int(np.unique(ids[sel]).size),
                        gain=float(abs(c)), gain_lo=float(np.percentile(np.abs(cb), 2.5)),
                        gain_hi=float(np.percentile(np.abs(cb), 97.5)),
                        veer=float(np.degrees(np.angle(c))),
                        veer_lo=float(np.degrees(np.angle(c)) + np.percentile(dphi, 2.5)),
                        veer_hi=float(np.degrees(np.angle(c)) + np.percentile(dphi, 97.5)),
                        offset_u=float(coef[0].real), offset_v=float(coef[0].imag),
                        offset_u_lo=float(np.percentile(boot[:, 0].real, 2.5)),
                        offset_u_hi=float(np.percentile(boot[:, 0].real, 97.5)),
                        r2=float(r2), speed_median=float(np.median(np.abs(wi[ok][sel]))),
                        reversed_percent=100.0 * float(np.mean(np.abs(np.angle(ratio[strong]))
                                                               > np.pi / 2))))
    asym = dict(site=site, pair=f'{pair[0]} to {pair[1]} m', half=half or 'all',
                gain_difference=float(abs(coef[1]) - abs(coef[2])),
                gain_difference_lo=float(np.percentile(np.abs(boot[:, 1]) - np.abs(boot[:, 2]), 2.5)),
                gain_difference_hi=float(np.percentile(np.abs(boot[:, 1]) - np.abs(boot[:, 2]), 97.5)),
                veer_difference=float(np.degrees(np.angle(coef[1] / coef[2]))),
                veer_difference_lo=float(np.percentile(np.degrees(np.angle(boot[:, 1] / boot[:, 2])), 2.5)),
                veer_difference_hi=float(np.percentile(np.degrees(np.angle(boot[:, 1] / boot[:, 2])), 97.5)))
    return out, asym


def profiles(site, d, lv, frame_dev):
    rows = []
    for k, level in enumerate(lv):
        u, v, T = d['u'][:, k], d['v'][:, k], d['T'][:, k]
        mu = P.mean_ci(u)
        mv = P.mean_ci(v)
        mT = P.mean_ci(T)
        ok = np.isfinite(u) & np.isfinite(v)
        rot = [np.mean(v[ok] * np.cos(np.deg2rad(dl)) - u[ok] * np.sin(np.deg2rad(dl)))
               for dl in (-frame_dev, frame_dev)] if ok.sum() > 48 else [np.nan, np.nan]
        rows.append(dict(site=site, level=level, n_hours=int(ok.sum()), u=mu[0], u_lo=mu[1],
                         u_hi=mu[2], v=mv[0], v_lo=mv[1], v_hi=mv[2], v_frame_lo=min(rot),
                         v_frame_hi=max(rot), frame_dev=frame_dev, T=mT[0]))
        if np.isfinite(mu[0]):
            say(f'    {site} {level:4.0f} m: u {mu[0]:+.3f} ({mu[1]:+.3f} to {mu[2]:+.3f}), v '
                f'{mv[0]:+.3f} ({mv[1]:+.3f} to {mv[2]:+.3f}), v over the frame {min(rot):+.3f} to '
                f'{max(rot):+.3f} m s-1')
    return rows


def speed_bins(site, pair, d, lv, label):
    ku, kl = lv.index(pair[0]), lv.index(pair[1])
    si = np.hypot(d['u'][:, ku], d['v'][:, ku])
    sb = np.hypot(d['u'][:, kl], d['v'][:, kl])
    edges = \
        np.arange(V.SPEED_BINS[0], V.SPEED_BINS[1] + 1e-9, V.SPEED_BINS[2])
    rows, hist = [], {}
    for name, sel in (('with the current', label < 0), ('against the current', label > 0)):
        ok = sel & np.isfinite(si) & np.isfinite(sb)
        for a, b in zip(edges[:-1], edges[1:]):
            m = ok & (si >= a) & (si < b)
            if m.sum() < V.MIN_BIN_HOURS:
                continue
            q = np.percentile(sb[m], [25, 50, 75])
            rows.append(dict(site=site, pair=f'{pair[0]} to {pair[1]} m', state=name,
                             centre=0.5 * (a + b), q1=q[0], median=q[1], q3=q[2], n=int(m.sum())))
        hist[name] = np.histogram2d(si[ok], sb[ok], bins=[edges, edges])[0]
    return rows, hist, edges


def references(site, geo, n_med):
    f = P.coriolis(geo.loc[site, 'lat'])
    alpha = geo.loc[site, 'slope']
    heights = {'design': (V.WATER_DEPTH_M[site] - V.PAIRS['bottom'][1],
                          V.WATER_DEPTH_M[site] - V.PAIRS['bottom'][0]),
               'GEBCO': (float(geo.loc[site, 'depth']) - V.PAIRS['bottom'][1],
                         float(geo.loc[site, 'depth']) - V.PAIRS['bottom'][0])}
    thick = {'neutral': lambda u: 0.4 * u / abs(f),
             'stratified': lambda u: 1.3 * u / abs(f) * (1.0 + n_med ** 2 / f ** 2) ** -0.25}
    rows = []
    for hname, (zb, zi) in heights.items():
        for tname, th in thick.items():
            for z0 in V.Z0_RANGE:
                for U in V.U_RANGE:
                    r, us = P.ekman_ratio(zb, zi, U, z0, f, th)
                    rows.append(dict(site=site, kind='unarrested', heights=hname, thickness=tname,
                                     z0=z0, U=U, zb=zb, zi=zi, gain=abs(r),
                                     veer=float(np.degrees(np.angle(r))), ustar=us,
                                     h=th(us), cap=np.nan))
        cap = n_med ** 2 * alpha * zb / abs(f)
        rows.append(dict(site=site, kind='arrested', heights=hname, zb=zb, zi=zi, cap=cap,
                         gain=np.nan, veer=0.0))
    tab = pd.DataFrame(rows)
    ua = tab[tab.kind == 'unarrested']
    ar = tab[tab.kind == 'arrested']
    say(f'    {site}: unarrested gain {ua.gain.min():.2f} to {ua.gain.max():.2f} and veering '
        f'{ua.veer.min():+.0f} to {ua.veer.max():+.0f} deg, arrested cap of the lowest meter '
        f'{ar.cap.min():.3f} to {ar.cap.max():.3f} m s-1 with N {n_med:.2e} s-1')
    return tab


def main():
    V.make_dirs()
    rng = np.random.default_rng(303)
    say('figure 3, the response of the near-bottom flow to the interior flow')
    geo, _ = I.geometry(verbose=False)
    lp = I.lp_files()
    fr = pd.read_csv(V.DIR_TAB / 'fig01_frame.csv').set_index('site')
    f2 = V.DIR_TAB / 'fig02_daily.csv'
    daily = pd.read_csv(f2) if f2.exists() else pd.DataFrame()
    prof, resp, asym, bins, refs = [], [], [], [], []
    hists = {}
    for site in V.SITES_200:
        if site not in lp:
            continue
        d = I.load_lp(lp[site])
        lv = list(d['level'])
        say(f'\n  {site}')
        prof += profiles(site, d, lv, float(fr.loc[site, 'bearing_dev_max'])
                         if site in fr.index else 0.0)
        if V.CONDITION_LEVEL not in lv:
            continue
        label, idx = states_of(d, lv)
        say(f'    {int((label < 0).sum())} hours with the current and {int((label > 0).sum())} '
            f'against it, in {idx.max() + 1} states')
        for name, pair in V.PAIRS.items():
            if not (pair[0] in lv and pair[1] in lv):
                say(f'    no {name} pair')
                continue
            both = np.isfinite(d['u'][:, lv.index(pair[0])]) & np.isfinite(d['u'][:, lv.index(pair[1])])
            if both.sum() < 240:
                say(f'    the {name} pair holds {int(both.sum())} hours together, left out')
                continue
            for half in (None,) + V.HALVES:
                out = response(site, pair, d, lv, label, idx, rng, geo, half)
                if out is None:
                    continue
                resp += out[0]
                asym.append(out[1])
                if half is None:
                    for r in out[0]:
                        say(f'    {name} pair {r["state"]}: gain {r["gain"]:.2f} ({r["gain_lo"]:.2f} to '
                            f'{r["gain_hi"]:.2f}), veering {r["veer"]:+.1f} ({r["veer_lo"]:+.1f} to '
                            f'{r["veer_hi"]:+.1f}) deg, reversed on {r["reversed_percent"]:.0f} % of '
                            f'the hours, {r["n_states"]} states')
                    a = out[1]
                    say(f'    {name} pair, with minus against the current: gain '
                        f'{a["gain_difference"]:+.2f} ({a["gain_difference_lo"]:+.2f} to '
                        f'{a["gain_difference_hi"]:+.2f}), veering {a["veer_difference"]:+.1f} '
                        f'({a["veer_difference_lo"]:+.1f} to {a["veer_difference_hi"]:+.1f}) deg, '
                        f'offset {out[0][0]["offset_u"]:+.3f} ({out[0][0]["offset_u_lo"]:+.3f} to '
                        f'{out[0][0]["offset_u_hi"]:+.3f}) m s-1 along the isobath')
            if name == 'bottom':
                b, h, edges = speed_bins(site, pair, d, lv, label)
                bins += b
                hists[site] = h
        if site in geo.index and V.PAIRS['bottom'][1] in lv:
            g = daily[(daily.site == site)] if len(daily) else pd.DataFrame()
            n_med = float(np.nanmedian(g.N)) if len(g) and g.N.notna().any() else 5.0e-3
            refs.append(references(site, geo, n_med))

    pd.DataFrame(prof).to_csv(V.DIR_TAB / 'fig03_profiles.csv', index=False)
    pd.DataFrame(resp).to_csv(V.DIR_TAB / 'fig03_response.csv', index=False)
    pd.DataFrame(asym).to_csv(V.DIR_TAB / 'fig03_asymmetry.csv', index=False)
    pd.DataFrame(bins).to_csv(V.DIR_TAB / 'fig03_bins.csv', index=False)
    if refs:
        pd.concat(refs, ignore_index=True).to_csv(V.DIR_TAB / 'fig03_reference.csv', index=False)
    if hists:
        edges = np.arange(V.SPEED_BINS[0], V.SPEED_BINS[1] + 1e-9, V.SPEED_BINS[2])
        centre = 0.5 * (edges[1:] + edges[:-1])
        names = ('with the current', 'against the current')
        arr = np.stack([np.stack([hists[s][n] for n in names]) for s in hists])
        ds = xr.Dataset({'count': (('site', 'state', 'interior', 'bottom'), arr)},
                        coords={'site': list(hists), 'state': list(names), 'interior': centre,
                                'bottom': centre})
        out = V.DIR_DATA / 'fig03_hist.nc'
        if out.exists():
            out.unlink()
        ds.to_netcdf(out)
    (V.DIR_LOG / 'fig03_compute.txt').write_text('\n'.join(LOG) + '\n')
    say(f'\ntables in {V.DIR_TAB}')


if __name__ == '__main__':
    main()