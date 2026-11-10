"""
Figure S2. Wavelet power of the interior velocity at FUB3.

One panel, the former panel (d) of Figure 6. Wavelet power of the along-isobath
velocity at 155 m at FUB3 over its longest campaign, normalised by the variance of the
series, with the 95 % level against red noise of the same lag-one autocorrelation
(contour) and the cone of influence (hatched). The synoptic band (2 to 8 days) and the
mesoscale band (16 to 64 days) used in the main text are marked at the right edge.

Reads paper_v3/data/fig06r_wavelet.nc. Writes paper_v3/figures/figS2_wavelet.png and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

import config_v3 as V
import style as S

FIGSIZE = (5.0, 3.6)
PERIOD_TICKS = (2, 4, 8, 16, 32, 64)
CMAP_POWER = S.truncated('cividis', 0.0, 1.0)
POWER_LIM = (-3.0, 3.0)
BANDS = (('synoptic', 2.0, 8.0), ('mesoscale', 16.0, 64.0))

f_wv = V.DIR_DATA / 'fig06r_wavelet.nc'
wv = xr.open_dataset(f_wv) if f_wv.exists() else None

fig = plt.figure(figsize=FIGSIZE)
gs = gridspec.GridSpec(1, 1, figure=fig, left=0.13, right=0.90, top=0.93, bottom=0.30)
ax = fig.add_subplot(gs[0, 0])

if wv is not None:
    t = pd.DatetimeIndex(wv.time.values)
    per = wv.period.values
    coi = wv.coi.values
    lp = np.log2(wv['power_u155'].values)
    m = ax.pcolormesh(t, per, lp, cmap=CMAP_POWER, vmin=POWER_LIM[0], vmax=POWER_LIM[1], shading='auto',
                      rasterized=True)
    ax.contour(t, per, wv['signif_u155'].values, levels=[1.0], colors=S.INK, linewidths=0.9)
    ax.fill_between(t, coi, per.max() * 1.2, facecolor='white', alpha=0.45, edgecolor='none', lw=0)
    ax.fill_between(t, coi, per.max() * 1.2, facecolor='none', edgecolor='dimgray', hatch='xx', lw=0)
    ax.plot(t, coi, color=S.INK, lw=0.8)
    ax.set_yscale('log', base=2)
    ax.set_ylim(min(per.max(), 64.0), per.min())
    ax.yaxis.set_major_locator(FixedLocator(PERIOD_TICKS))
    ax.yaxis.set_major_formatter(FixedFormatter([str(p) for p in PERIOD_TICKS]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xlabel(f'{t[0].year}' if t[0].year == t[-1].year else f'{t[0].year} and {t[-1].year}')
    ax.set_ylabel('period (days)')
    ax.set_title('along-isobath velocity at 155 m of FUB3', fontsize=S.FS)
    # the two bands of the main text, marked outside the data area
    for name, lo, hi in BANDS:
        ax.plot([1.015, 1.015], [lo, hi], transform=ax.get_yaxis_transform(), color=S.INK, lw=2.2,
                solid_capstyle='butt', clip_on=False)
        ax.text(1.03, np.sqrt(lo * hi), name, transform=ax.get_yaxis_transform(), rotation=90, ha='left',
                va='center', fontsize=S.FS_SMALL)
    p = ax.get_position()
    cax = fig.add_axes([p.x0 + 0.15 * p.width, p.y0 - 0.17, 0.7 * p.width, 0.03])
    cb = fig.colorbar(m, cax=cax, orientation='horizontal', extend=S.extend_for(lp, *POWER_LIM))
    cb.set_label('log$_2$ of the normalised power')
else:
    ax.set_axis_off()

fig.savefig(V.DIR_FIG / 'figS2_wavelet.pdf')
S.save(fig, V.DIR_FIG / 'figS2_wavelet.png')
