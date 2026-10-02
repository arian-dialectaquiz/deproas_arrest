"""
Physical quantities shared by the figures of the third version.

    coriolis       Coriolis parameter
    sigma0         potential density anomaly from TEOS-10 (McDougall and Barker 2011),
                   or from a linear equation of state when gsw is not installed
    n2_fit         buoyancy frequency squared fitted over a layer of a profile
    slope_field    bottom slope and depth from the GEBCO gradient on a coarse grid
    isobath_path   one isobath traced, smoothed and resampled along the shelf break
    idw            inverse distance weighting of valid grid values onto points
    runs_of_sign   intervals over which the interior flow keeps one sign
    arrest times   the canonical and the wedge scalings of the arrest time

Created by Arian Dialectaquiz.
"""
import numpy as np
import xarray as xr

import config_v3 as V

try:
    import gsw
    EOS = 'TEOS-10'
except ImportError:                                  # the linear form keeps the gradients
    gsw = None
    EOS = 'linear, thermal expansion 2.2e-4 per degree and haline contraction 7.6e-4'

R_EARTH_KM = 111.2


def coriolis(lat):
    return 2.0 * V.OMEGA * np.sin(np.deg2rad(lat))


def sigma0(T, S, p, lon, lat, potential=False):
    """Potential density anomaly referred to the surface. T is in situ temperature, or
    potential temperature when potential is True, S practical salinity and p pressure
    in dbar, taken equal to the depth in metres."""
    T, S, p = np.broadcast_arrays(np.asarray(T, float), np.asarray(S, float),
                                  np.asarray(p, float))
    if gsw is not None:
        SA = gsw.SA_from_SP(S, p, lon, lat)
        CT = gsw.CT_from_pt(SA, T) if potential else gsw.CT_from_t(SA, T, p)
        return gsw.sigma0(SA, CT)
    return 26.2 + V.RHO0 * (-2.2e-4 * (T - 15.0) + 7.6e-4 * (S - 35.5))


def n2_fit(z, sig, z_top, z_bot, dz=None, min_levels=None):
    """N squared from the least squares slope of the potential density over z_top to
    z_bot, depths positive downward, after interpolation onto a regular grid of dz."""
    dz = V.BBL_FINE_DZ if dz is None else dz
    min_levels = V.BBL_MIN_LEVELS if min_levels is None else min_levels
    z, sig = np.asarray(z, float), np.asarray(sig, float)
    ok = np.isfinite(z) & np.isfinite(sig)
    z, sig = z[ok], sig[ok]
    if z.size < 2 or ((z >= z_top - 1e-6) & (z <= z_bot + 1e-6)).sum() < min_levels:
        return np.nan
    zi = np.arange(z_top, z_bot + 1e-6, dz)
    si = np.interp(zi, z, sig, left=np.nan, right=np.nan)
    good = np.isfinite(si)
    if good.sum() < 3:
        return np.nan
    slope = np.polyfit(zi[good], si[good], 1)[0]
    return V.G / V.RHO0 * slope


def n2_two_levels(sig_top, sig_bot, z_top, z_bot):
    return V.G / V.RHO0 * (np.asarray(sig_bot) - np.asarray(sig_top)) / (z_bot - z_top)


def slope_field(gebco, zname, extent, margin=0.4):
    """Depth and bottom slope on a grid of SLOPE_RES_DEG. GEBCO is block averaged onto
    that grid, the gradient is taken in metres and both of its components are averaged
    over SLOPE_SMOOTH_KM before the magnitude, which gives the slope felt by a flow of
    that width and leaves out the roughness of the bottom."""
    from scipy.ndimage import uniform_filter
    lon0, lon1, lat0, lat1 = extent
    sub = gebco[zname].sel(lon=slice(lon0 - margin, lon1 + margin),
                           lat=slice(lat0 - margin, lat1 + margin))
    native = float(abs(sub.lon.values[1] - sub.lon.values[0]))
    k = max(1, int(round(V.SLOPE_RES_DEG / native)))
    sub = sub.coarsen(lon=k, lat=k, boundary='trim').mean()
    h = -sub.values.astype(float)
    lat, lon = sub.lat.values, sub.lon.values
    dy = R_EARTH_KM * 1e3 * np.gradient(lat)[:, None]
    dx = R_EARTH_KM * 1e3 * np.gradient(lon)[None, :] * np.cos(np.deg2rad(lat))[:, None]
    gy = np.gradient(h, axis=0) / dy
    gx = np.gradient(h, axis=1) / dx
    n = max(1, int(round(V.SLOPE_SMOOTH_KM / (R_EARTH_KM * float(np.median(np.gradient(lat)))))))
    gx_s, gy_s = uniform_filter(gx, n, mode='nearest'), uniform_filter(gy, n, mode='nearest')
    h_s = uniform_filter(h, n, mode='nearest')
    return xr.Dataset({'depth': (('lat', 'lon'), h_s), 'alpha': (('lat', 'lon'), np.hypot(gx_s, gy_s))},
                      coords={'lat': lat, 'lon': lon})


def isobath_path(field, level, extent, start_lonlat):
    """The longest segment of one isobath inside the extent, oriented from the end closest
    to start_lonlat, smoothed by a running mean of PATH_SMOOTH_KM and resampled every
    PATH_STEP_KM. Returns longitude, latitude and distance along the path in km."""
    import contourpy
    lon0, lon1, lat0, lat1 = extent
    gen = contourpy.contour_generator(field.lon.values, field.lat.values, field.depth.values)
    best = None
    for seg in gen.lines(level):
        inside = ((seg[:, 0] >= lon0) & (seg[:, 0] <= lon1) & (seg[:, 1] >= lat0)
                  & (seg[:, 1] <= lat1))
        seg = seg[inside]
        if best is None or len(seg) > len(best):
            best = seg
    if best is None or len(best) < 5:
        raise ValueError(f'no {level} m isobath inside {extent}')
    d0 = np.hypot(best[0, 0] - start_lonlat[0], best[0, 1] - start_lonlat[1])
    d1 = np.hypot(best[-1, 0] - start_lonlat[0], best[-1, 1] - start_lonlat[1])
    if d1 < d0:
        best = best[::-1]
    c = np.cos(np.deg2rad(best[:, 1]))
    step = np.hypot(np.diff(best[:, 0]) * c[1:], np.diff(best[:, 1])) * R_EARTH_KM
    s = np.r_[0.0, np.cumsum(step)]
    si = np.arange(0.0, s[-1], V.PATH_STEP_KM)
    lo, la = np.interp(si, s, best[:, 0]), np.interp(si, s, best[:, 1])
    w = max(1, int(round(V.PATH_SMOOTH_KM / V.PATH_STEP_KM)))
    kern = np.ones(w) / w
    lo_s = np.convolve(np.pad(lo, w, mode='edge'), kern, mode='same')[w:-w]
    la_s = np.convolve(np.pad(la, w, mode='edge'), kern, mode='same')[w:-w]
    c = np.cos(np.deg2rad(la_s))
    step = np.hypot(np.diff(lo_s) * c[1:], np.diff(la_s)) * R_EARTH_KM
    return lo_s, la_s, np.r_[0.0, np.cumsum(step)]


def distance_km(lat0, lon0, lat, lon):
    la = np.deg2rad(0.5 * (lat0 + np.asarray(lat)))
    return R_EARTH_KM * np.hypot(np.asarray(lat) - lat0, (np.asarray(lon) - lon0) * np.cos(la))


def idw(plat, plon, glat, glon, values, radius_km, power=2.0):
    """Inverse distance weights of the finite grid values within radius_km of each point."""
    GLo, GLa = np.meshgrid(glon, glat)
    ok = np.isfinite(values)
    gla, glo, gv = GLa[ok], GLo[ok], values[ok]
    out = np.full(np.size(plat), np.nan)
    for k, (la, lo) in enumerate(zip(np.atleast_1d(plat), np.atleast_1d(plon))):
        d = distance_km(la, lo, gla, glo)
        near = d <= radius_km
        if near.any():
            w = 1.0 / np.maximum(d[near], 0.5) ** power
            out[k] = np.sum(w * gv[near]) / np.sum(w)
    return out


def runs_of_sign(u, campaign, dt_h=None, min_speed=None, min_hours=None):
    """Start index, end index and sign of every interval over which the interior flow
    keeps one sign inside one campaign, weak hours carrying the sign of the hour before."""
    import pandas as pd
    dt_h = V.DT_H if dt_h is None else dt_h
    min_speed = V.MIN_SPEED if min_speed is None else min_speed
    min_hours = V.MIN_STATE_H if min_hours is None else min_hours
    s = np.where(np.abs(u) >= min_speed, np.sign(u), np.nan)
    s = pd.Series(s).ffill().values
    ok = np.isfinite(s) & (np.asarray(campaign) != '')
    out, i = [], 0
    while i < s.size:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < s.size and ok[j + 1] and s[j + 1] == s[i] and campaign[j + 1] == campaign[i]:
            j += 1
        if (j - i + 1) * dt_h >= min_hours:
            out.append((i, j + 1, float(s[i])))
        i = j + 1
    return out


def arrest_time_canonical(s, f):
    """(|f| s^2)^-1, the time the Ekman transport needs to arrest the layer (MacCready and
    Rhines 1991, Garrett et al. 1993), in seconds."""
    return 1.0 / (np.abs(f) * np.asarray(s, float) ** 2)


def arrest_time_wedge(s, N, cd):
    """Time the unarrested Ekman transport Cd U^2 / |f| needs to fill the wedge of the
    arrested layer, whose thickness is |f| U / (N^2 alpha) and whose fluid is displaced
    up the slope by that thickness over alpha. The volume over the transport gives
    1 / (2 Cd N s^3), in seconds, independent of the interior speed, the form of the
    upslope scaling of Trowbridge and Lentz (1991) and Brink and Lentz (2010a), whose
    order one factor is to be confirmed."""
    return 1.0 / (2.0 * cd * np.asarray(N, float) * np.asarray(s, float) ** 3)


# ------------------------------------------------------------- statistics
def integral_time_scale(x, dt_h=None, max_lag_days=30.0):
    """Integral time scale in hours, the sum of the autocorrelation up to its first zero,
    which sets the effective number of independent samples (Chelton 1983)."""
    dt_h = V.DT_H if dt_h is None else dt_h
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    n = x.size
    kmax = int(min(max_lag_days * 24.0 / dt_h, n // 4))
    if kmax < 2:
        return dt_h
    r = np.empty(kmax)
    for k in range(kmax):
        a, b = x[:n - k], x[k:]
        ok = np.isfinite(a) & np.isfinite(b)
        r[k] = np.sum(a[ok] * b[ok]) / np.sqrt(np.sum(a[ok] ** 2) * np.sum(b[ok] ** 2)) \
            if ok.sum() > 10 else 0.0
    neg = np.flatnonzero(r <= 0)
    k0 = neg[0] if neg.size else kmax
    return float(max(dt_h * (0.5 + np.sum(r[1:k0])), dt_h))


def mean_ci(x, dt_h=None):
    """Mean and its 95 % interval from the effective number of independent samples."""
    dt_h = V.DT_H if dt_h is None else dt_h
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 48:
        return np.nan, np.nan, np.nan
    tau = integral_time_scale(x[ok], dt_h)
    n_eff = max(ok.sum() * dt_h / (2.0 * tau), 2.0)
    m, sd = float(np.mean(x[ok])), float(np.std(x[ok]))
    half = 1.96 * sd / np.sqrt(n_eff)
    return m, m - half, m + half


# ------------------------------------------------- unarrested bottom Ekman layer
def ekman_layer(U, z0, f, thickness, H=300.0, n=700, cd0=2.0e-3, nu_bg=1.0e-5, iters=40):
    """Steady bottom Ekman layer under the interior velocity U (m s-1), solved for the
    complex velocity W = u + iv of

        d/dz (nu dW/dz) = i f (W - U),   W(z0) = 0,   W(H) = U,

    with the eddy viscosity nu = kappa u* z (1 - z/h) inside a layer of thickness h and
    a small background value above it (Madsen 1977, Weatherly and Martin 1978). The
    thickness is given as a function of u*, and u* is iterated until it matches the
    stress at the bottom. Returns heights, the complex profile and u*."""
    from scipy.linalg import solve_banded
    z = np.geomspace(z0, H, n)
    ustar = np.sqrt(cd0) * U
    W = np.full(n, U + 0j)
    for _ in range(iters):
        h = thickness(ustar)
        zh = 0.5 * (z[1:] + z[:-1])
        nu_h = np.where(zh < h, V.KAPPA * ustar * zh * (1.0 - zh / h), 0.0)
        nu_h = np.maximum(nu_h, nu_bg)
        dzp = np.diff(z)
        ab = np.zeros((3, n), complex)
        rhs = np.zeros(n, complex)
        ab[1, 0], rhs[0] = 1.0, 0.0
        ab[1, -1], rhs[-1] = 1.0, U
        for i in range(1, n - 1):
            dm, dp = dzp[i - 1], dzp[i]
            w = 0.5 * (dm + dp)
            a_m = nu_h[i - 1] / (dm * w)
            a_p = nu_h[i] / (dp * w)
            ab[2, i - 1] = a_m                    # below the diagonal
            ab[1, i] = -(a_m + a_p) - 1j * f
            ab[0, i + 1] = a_p                    # above the diagonal
            rhs[i] = -1j * f * U
        W = solve_banded((1, 1), ab, rhs)
        tau = nu_h[0] * abs(W[1] - W[0]) / dzp[0]
        ustar = 0.5 * ustar + 0.5 * np.sqrt(tau)
    return z, W, ustar


def ekman_ratio(zb, zi, U, z0, f, thickness):
    """Complex ratio of the velocity at zb to the velocity at zi in the unarrested layer,
    whose modulus is the attenuation and whose phase is the veering, clockwise negative."""
    z, W, ustar = ekman_layer(U, z0, f, thickness)
    wb = np.interp(zb, z, W.real) + 1j * np.interp(zb, z, W.imag)
    wi = np.interp(zi, z, W.real) + 1j * np.interp(zi, z, W.imag)
    return wb / wi, ustar


# ------------------------------------------------------ filters and frames
def lanczos_weights(cutoff_h, half_window, dt_h=1.0):
    """Weights of the Lanczos low-pass filter (Duchon 1979), normalised to unit sum."""
    fc = dt_h / cutoff_h
    k = np.arange(-half_window, half_window + 1)
    w = np.where(k == 0, 2.0 * fc, np.sin(2.0 * np.pi * fc * k) / (np.pi * np.where(k == 0, 1, k)))
    sigma = np.where(k == 0, 1.0, np.sin(np.pi * k / half_window) / (np.pi * np.where(k == 0, 1, k) / half_window))
    w = w * sigma
    return w / w.sum()


def lowpass(x, cutoff_h=None, half_window=None):
    """Lanczos low-pass of an hourly series, left undefined wherever the window is incomplete."""
    cutoff_h = V.LP_CUTOFF_H if cutoff_h is None else cutoff_h
    half_window = V.LP_HALF_WINDOW_H if half_window is None else half_window
    w = lanczos_weights(cutoff_h, half_window)
    x = np.asarray(x, float)
    good = np.isfinite(x)
    num = np.convolve(np.where(good, x, 0.0), w, mode='same')
    full = np.convolve(good.astype(float), np.ones(w.size), mode='same')
    out = num.copy()
    out[full < w.size - 0.5] = np.nan
    return out


def to_isobath(east, north, bearing_deg):
    """Along-isobath u, positive toward the bearing, and cross-isobath v, positive to its
    left, which is onshore on this margin, from eastward and northward components."""
    b = np.deg2rad(bearing_deg)
    return east * np.sin(b) + north * np.cos(b), -east * np.cos(b) + north * np.sin(b)


def complex_gains(wi, wb, up):
    """w_b = a + c_up w_i 1_up + c_down w_i 1_down, least squares on complex velocities."""
    X = np.c_[np.ones(wi.size), wi * up, wi * (~up)].astype(complex)
    return np.linalg.lstsq(X, wb, rcond=None)[0]


# ------------------------------------------------------------- wind and spectra
def drag_large_pond(speed):
    """Neutral drag coefficient of Large and Pond (1981), held at its low wind value below
    11 m s-1 and at its 25 m s-1 value above that speed."""
    sp = np.clip(speed, 0.0, 25.0)
    return np.where(sp < 11.0, 1.2e-3, (0.49 + 0.065 * sp) * 1e-3)


def surrogate(x, rng):
    """Series with the power spectrum of x and random phases (Ebisuzaki 1997)."""
    x = np.asarray(x, float)
    n = x.size
    X = np.fft.rfft(x - x.mean())
    ph = np.exp(2j * np.pi * rng.random(X.size))
    ph[0] = 1.0
    if n % 2 == 0:
        ph[-1] = 1.0
    return np.fft.irfft(X * ph, n)


def cross_spectra(series, seg, dt_days=1.0):
    """Cross-spectral matrix averaged over Hann windowed, linearly detrended segments of
    seg samples with half overlap, drawn inside each block of complete data. series is a
    list of blocks, each an array of shape (time, variables). Returns frequencies in cycles
    per day, the matrix S[f, i, j] = <X_i* X_j> and the number of segments."""
    from scipy.signal import detrend
    w = np.hanning(seg)
    acc, n = None, 0
    for blk in series:
        for a in range(0, blk.shape[0] - seg + 1, seg // 2):
            x = detrend(blk[a:a + seg], axis=0) * w[:, None]
            X = np.fft.rfft(x, axis=0)
            S = np.conj(X)[:, :, None] * X[:, None, :]
            acc = S if acc is None else acc + S
            n += 1
    if acc is None:
        return None, None, 0
    return np.fft.rfftfreq(seg, dt_days), acc / n, n


def coherences(S):
    """Ordinary, partial and multiple coherence of an output (last variable) on two inputs
    (Bendat and Piersol 2010). S[f, i, j] holds x1, x2 and y in that order."""
    S11, S22, Syy = S[:, 0, 0].real, S[:, 1, 1].real, S[:, 2, 2].real
    S12, S1y, S2y = S[:, 0, 1], S[:, 0, 2], S[:, 1, 2]
    g1 = np.abs(S1y) ** 2 / (S11 * Syy)
    g2 = np.abs(S2y) ** 2 / (S22 * Syy)
    S11_2 = S11 - np.abs(S12) ** 2 / S22
    Syy_2 = Syy - np.abs(S2y) ** 2 / S22
    S1y_2 = S1y - S12 * S2y / S22
    S22_1 = S22 - np.abs(S12) ** 2 / S11
    Syy_1 = Syy - np.abs(S1y) ** 2 / S11
    S2y_1 = S2y - S[:, 1, 0] * S1y / S11
    p1 = np.abs(S1y_2) ** 2 / (S11_2 * Syy_2)
    p2 = np.abs(S2y_1) ** 2 / (S22_1 * Syy_1)
    mult = np.empty(S.shape[0])
    for k in range(S.shape[0]):
        Sxx = S[k, :2, :2]
        sxy = S[k, :2, 2]
        if np.linalg.cond(Sxx) > 1e12:
            mult[k] = np.nan
            continue
        mult[k] = float(np.real(np.conj(sxy) @ np.linalg.solve(Sxx, sxy)) / Syy[k])
    return g1, g2, p1, p2, mult


def coherence_level(n_seg, conditioned=0, p=0.95):
    """Level that a coherence from n_seg independent segments exceeds by chance, reduced by
    one degree of freedom for every conditioning input."""
    n = max(n_seg - conditioned, 2)
    return 1.0 - (1.0 - p) ** (1.0 / (n - 1))