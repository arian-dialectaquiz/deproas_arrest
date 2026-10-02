"""
Quantities of the third row of Figure 6, the time and frequency of the forcing at a 200 m
mooring, FUB3 by default and any other given as the first argument (FCF3 for Cabo Frio).

The partial and multiple coherence of Figure 5 average the records over segments, so
a forcing that acts in bursts, as the passage of an eddy does, could hide in them. A
continuous wavelet transform keeps the time and resolves such bursts over the longest
continuous campaign at FUB3, the mooring nearest the axis of the Brazil Current.

 1  series       daily along-isobath velocity at 155 m, uplift of the isotherms at 155 m,
                 hindcast of the coastal trapped wave, position of the Brazil Current
                 axis and eddy kinetic energy over the slope, on the longest run of
                 days inside one campaign, gaps of up to MAX_GAP_DAYS filled linearly
 2  power        Morlet wavelet transform (omega0 = 6) of the standardised velocity and
                 uplift, with the cone of influence and the 95 % level against a red
                 noise of the same lag-one autocorrelation (Torrence and Compo 1998)
 3  coherence    wavelet coherence of the uplift with the wave, the eddy kinetic energy
                 and the position of the axis, smoothed in time by a Gaussian of the
                 width of each scale and in scale by a boxcar of 0.6 of an octave, with
                 the 95 % level of each period from N_MC pairs of red noise series of
                 the same autocorrelations, and the phase of the cross transform
                 (Torrence and Webster 1999, Grinsted et al. 2004)
 4  bands        share of the time and period plane inside the cone of influence where
                 the power or the coherence exceeds its level, and the mean coherence and
                 phase lag, in the synoptic and in the mesoscale band of Figure 5

Reads the low-passed records and paper_v3/tables/fig05_daily.csv and fig06r_eke_series.csv.
Outputs in paper_v3
    data/fig06r_wavelet.nc          power, coherence, levels, phase and the cone of influence
    tables/fig06r_wavelet_bands.csv band summaries
    logs/fig06r_wavelet_compute.txt

Created by Arian Dialectaquiz.
"""
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import chi2
from scipy.ndimage import convolve1d, uniform_filter1d

import config_v3 as V
import inputs_v3 as I

warnings.simplefilter('ignore', RuntimeWarning)
LOG = []

SITE = sys.argv[1] if len(sys.argv) > 1 else 'FUB3'   # python fig06r_wavelet_compute.py FCF3
SUFFIX = '' if SITE == 'FUB3' else f'_{SITE}'           # FUB3 keeps the names read by fig06r_plot.py
W0 = 6.0                 # nondimensional frequency of the Morlet wavelet
DJ = 1.0 / 8.0           # spacing of the scales in octaves
MAX_GAP_DAYS = 2
N_MC = 300
MIN_RUN_DAYS = 90
FOURIER = 4.0 * np.pi / (W0 + np.sqrt(2.0 + W0 ** 2))


def say(msg=''):
    print(msg)
    LOG.append(msg)


# ------------------------------------------------------------------ wavelet tools
def scales_for(n, dt=1.0):
    s0 = 2.0 * dt / FOURIER            # smallest period of two samples
    J = int(np.floor(np.log2(n * dt / s0) / DJ))
    return s0 * 2.0 ** (DJ * np.arange(J + 1))


def cwt(x, scales, dt=1.0):
    """Morlet transform of Torrence and Compo (1998) by the fast Fourier transform."""
    n = x.size
    npad = int(2 ** np.ceil(np.log2(n)) * 2)
    xp = np.zeros(npad)
    xp[:n] = x
    f = np.fft.fft(xp)
    k = 2.0 * np.pi * np.fft.fftfreq(npad, dt)
    W = np.empty((scales.size, n), complex)
    for j, s in enumerate(scales):
        psi = np.pi ** -0.25 * np.sqrt(2.0 * np.pi * s / dt) * np.exp(-0.5 * (s * k - W0) ** 2) * (k > 0)
        W[j] = np.fft.ifft(f * psi)[:n]
    return W


def coi_of(n, dt=1.0):
    """e-folding time of the edge effect, as a period."""
    edge = np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1)) * dt
    return FOURIER / np.sqrt(2.0) * edge


def lag1(x):
    x = x - x.mean()
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))


def red_level(alpha, periods, dt=1.0, p=0.95):
    """Power of red noise of unit variance at each period, times the chi-square quantile."""
    fk = dt / periods
    Pk = (1.0 - alpha ** 2) / (1.0 + alpha ** 2 - 2.0 * alpha * np.cos(2.0 * np.pi * fk))
    return Pk * chi2.ppf(p, 2) / 2.0


def smooth(W, scales, dt=1.0):
    """Smoothing of Grinsted et al. (2004), Gaussian in time and boxcar in scale."""
    out = np.empty_like(W)
    for j, s in enumerate(scales):
        sig = s / dt
        half = int(np.ceil(4.0 * sig))
        t = np.arange(-half, half + 1)
        g = np.exp(-0.5 * (t / sig) ** 2)
        g /= g.sum()
        out[j] = convolve1d(W[j].real, g, mode='constant') + 1j * convolve1d(W[j].imag, g, mode='constant')
    width = max(int(round(0.6 / DJ)), 1)
    return uniform_filter1d(out.real, width, axis=0, mode='nearest') + \
        1j * uniform_filter1d(out.imag, width, axis=0, mode='nearest')


def coherence(Wx, Wy, scales):
    s = scales[:, None]
    Sxy = smooth(Wx * np.conj(Wy) / s, scales)
    Sx = smooth(np.abs(Wx) ** 2 / s, scales).real
    Sy = smooth(np.abs(Wy) ** 2 / s, scales).real
    return np.abs(Sxy) ** 2 / (Sx * Sy), np.angle(Sxy)


def ar1(alpha, n, rng):
    e = rng.standard_normal(n + 200)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, e.size):
        x[i] = alpha * x[i - 1] + np.sqrt(1.0 - alpha ** 2) * e[i]
    return x[200:]


def coherence_level(ax, ay, n, scales, inside, rng):
    """95 % level of the wavelet coherence at each period from red noise pairs."""
    vals = [[] for _ in scales]
    for _ in range(N_MC):
        x, y = ar1(ax, n, rng), ar1(ay, n, rng)
        R2, _ = coherence(cwt((x - x.mean()) / x.std(), scales), cwt((y - y.mean()) / y.std(), scales), scales)
        for j in range(scales.size):
            if inside[j].any():
                vals[j].append(R2[j][inside[j]])
    return np.array([np.percentile(np.concatenate(v), 95) if v else np.nan for v in vals])


# ------------------------------------------------------------------------ series
def daily_series():
    lp = I.lp_files(verbose=False)
    d = I.load_lp(lp[SITE])
    lv = list(d['level'])
    u = pd.DataFrame({'u155': d['u'][:, lv.index(V.CONDITION_LEVEL)], 'campaign': d['campaign']},
                     index=d['time'])
    u = u[u.campaign != '']
    g = u.groupby([u.campaign, u.index.floor('D')])['u155']
    day = g.mean().where(g.count() >= V.MIN_DAY_HOURS)
    day.index = day.index.set_names(['campaign', 'day'])
    day = day.reset_index()
    f5 = pd.read_csv(V.DIR_TAB / 'fig05_daily.csv', parse_dates=['day'])
    f5 = f5[f5.site == SITE][['day', 'campaign', 'eta', 'wave', 'axis']]
    f5['campaign'] = f5.campaign.astype(str)
    day['campaign'] = day.campaign.astype(str)
    df = f5.merge(day, on=['day', 'campaign'], how='outer')
    ek = pd.read_csv(V.DIR_TAB / 'fig06r_eke_series.csv', parse_dates=['day'])
    ek = ek[ek.site == SITE].set_index('day').eke
    df['eke'] = ek.reindex(df.day).values
    return df.sort_values('day').reset_index(drop=True)


def longest_run(df, need):
    best = None
    for camp, g in df.groupby('campaign'):
        g = g.drop(columns='campaign').set_index('day').sort_index()
        g = g[~g.index.duplicated()].asfreq('D').astype(float)
        g = g.interpolate(limit=MAX_GAP_DAYS, limit_area='inside')
        ok = g[need].notna().all(axis=1).values
        start, length = 0, 0
        for i, v in enumerate(np.r_[ok, False]):
            if v:
                length += 1
            else:
                if best is None or length > best[2]:
                    best = (camp, g.index[i - length] if length else None, length, g.iloc[i - length:i])
                length = 0
    return best


def main():
    V.make_dirs()
    rng = np.random.default_rng(1303)
    say(f'figure 6, third row, the time and frequency of the forcing at {SITE}')
    df = daily_series()
    need = ['u155', 'eta', 'wave', 'eke']
    camp, t0, n, g = longest_run(df, need)
    if n < MIN_RUN_DAYS:
        say(f'  the longest run holds {n} days, fewer than {MIN_RUN_DAYS}, no wavelet analysis')
        return
    say(f'  campaign {camp}, {n} days from {g.index[0].date()} to {g.index[-1].date()}')
    axis_ok = g['axis'].notna().mean()
    say(f'  position of the axis defined on {100 * axis_ok:.0f} % of these days')
    scales = scales_for(n)
    periods = FOURIER * scales
    coi = coi_of(n)
    inside = periods[:, None] <= coi[None, :]

    Wz, alpha = {}, {}
    for name in ('u155', 'eta', 'wave', 'eke', 'axis'):
        x = g[name].values.astype(float)
        if name == 'axis':
            if axis_ok < 0.8:
                continue
            x = pd.Series(x).interpolate(limit_direction='both').values
        if np.nanstd(x) == 0:
            continue
        z = (x - np.nanmean(x)) / np.nanstd(x)
        Wz[name] = cwt(z, scales)
        alpha[name] = lag1(z)
        say(f'  {name}: lag-one autocorrelation {alpha[name]:.2f}')

    ds = xr.Dataset(coords={'time': g.index.values, 'period': periods})
    ds['coi'] = ('time', coi)
    rows = []
    bands = {'synoptic': V.SYNOPTIC_DAYS, 'mesoscale': V.MESOSCALE_DAYS}
    for name in ('u155', 'eta'):
        power = np.abs(Wz[name]) ** 2
        ratio = power / red_level(alpha[name], periods)[:, None]
        ds[f'power_{name}'] = (('period', 'time'), power)
        ds[f'signif_{name}'] = (('period', 'time'), ratio)
        for band, (p0, p1) in bands.items():
            b = (periods >= p0) & (periods <= p1)
            area = inside[b]
            share = float(np.mean(ratio[b][area] > 1.0)) if area.any() else np.nan
            rows.append(dict(site=SITE, quantity=f'power of {name}', band=band, share_significant=share,
                             mean_value=float(np.mean(power[b][area])) if area.any() else np.nan,
                             lag_days=np.nan))
            say(f'  power of {name}, {band} band: significant on {100 * share:.0f} % of the plane inside '
                f'the cone of influence')
    for other in ('wave', 'eke', 'axis'):
        if other not in Wz:
            say(f'  no coherence with the {other}')
            continue
        R2, phase = coherence(Wz['eta'], Wz[other], scales)
        level = coherence_level(alpha['eta'], alpha[other], n, scales, inside, rng)
        ds[f'coh_{other}'] = (('period', 'time'), R2)
        ds[f'phase_{other}'] = (('period', 'time'), phase)
        ds[f'level_{other}'] = ('period', level)
        for band, (p0, p1) in bands.items():
            b = (periods >= p0) & (periods <= p1)
            area = inside[b]
            if not area.any():
                continue
            above = R2[b] > level[b][:, None]
            share = float(np.mean(above[area]))
            sig = area & above
            if sig.any():
                ph = np.angle(np.mean(np.exp(1j * phase[b][sig])))
                lag = float(np.degrees(ph) / 360.0 * np.median(np.broadcast_to(periods[b][:, None], R2[b].shape)[sig]))
            else:
                ph, lag = np.nan, np.nan
            rows.append(dict(site=SITE, quantity=f'coherence of the uplift with the {other}', band=band,
                             share_significant=share, mean_value=float(np.mean(R2[b][area])),
                             phase_deg=float(np.degrees(ph)) if np.isfinite(ph) else np.nan, lag_days=lag))
            say(f'  coherence of the uplift with the {other}, {band} band: mean {np.mean(R2[b][area]):.2f}, '
                f'above the 95 % level on {100 * share:.0f} % of the plane inside the cone of influence'
                + (f', phase {np.degrees(ph):+.0f} deg ({lag:+.1f} days, positive when the uplift leads)'
                   if np.isfinite(ph) else ''))
    ds.attrs.update(site=SITE, campaign=str(camp), w0=W0, dj=DJ, n_mc=N_MC)
    out = V.DIR_DATA / f'fig06r_wavelet{SUFFIX}.nc'
    if out.exists():
        out.unlink()
    ds.to_netcdf(out)
    pd.DataFrame(rows).to_csv(V.DIR_TAB / f'fig06r_wavelet_bands{SUFFIX}.csv', index=False)
    (V.DIR_LOG / f'fig06r_wavelet_compute{SUFFIX}.txt').write_text('\n'.join(LOG) + '\n')
    say(f'\noutputs in {V.DIR_DATA} and {V.DIR_TAB}')


if __name__ == '__main__':
    main()