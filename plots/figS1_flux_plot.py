"""
Figure S1. Near-bottom flux of the deficit below 20 C across the isobaths.

One panel, the former panel (g) of the path figure. At every mooring with at least
60 days of record, the term of the mean flow (filled) and of the correlated
fluctuations (hatched) of the flux (Equation 17 of the main text), each bar spanning
its range over the rotations of the frame of Figure 1 with the value in the frame of
the isobath (black tick), positive onshore. The moorings run from the shelf break to
the coast along each line.

Reads paper_v3/tables/fig06_flux.csv. Writes paper_v3/figures/figS1_flux.png and .pdf.

Created by Arian Dialectaquiz.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config_v3 as V
import style as S

FIGSIZE = (5.0, 3.4)
ORDER = ['FUB3', 'FUB2', 'FUB1', 'FBG3', 'FCF3', 'FCF2', 'FCF1']
W = 0.34                                               # width of each bar


def read(name, **kw):
    p = V.DIR_TAB / f'{name}.csv'
    if p.exists() and p.stat().st_size > 5:
        return pd.read_csv(p, **kw)
    print(f'  {p.name} is missing, its elements are left out')
    return pd.DataFrame()


flux = read('fig06_flux')

# ----------------------------------------------------------------- layout
fig = plt.figure(figsize=FIGSIZE)
gs = gridspec.GridSpec(1, 1, figure=fig, left=0.17, right=0.985, top=0.95, bottom=0.17)
ax = fig.add_subplot(gs[0, 0])

# ------------------------------------------------------- the two terms of the flux
fq = flux[(flux.T_c == V.SACW_T) & (flux.short == 0)] if len(flux) else pd.DataFrame()
sites = [s for s in ORDER if len(fq) and s in set(fq.site)]
span = 100 * np.ptp(np.r_[fq.flux_mean.values, fq.flux_eddy.values, 0.0]) if len(fq) else 1.0
min_h = 0.02 * span                                    # a range too narrow to see is drawn at this height
for i, site in enumerate(sites):
    g = fq[fq.site == site]
    r0 = g[g.rotation == 0.0]
    col = S.SITE_COLOURS[site]
    for j, (key, filled) in enumerate((('flux_mean', True), ('flux_eddy', False))):
        x = i + (j - 0.5) * (W + 0.04)
        lo, hi = 100 * g[key].min(), 100 * g[key].max()
        if hi - lo < min_h:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - 0.5 * min_h, mid + 0.5 * min_h
        kw_ = dict(color=col, ec='black', lw=0.6) if filled else dict(color='white', ec=col, hatch=S.DOWN_HATCH, lw=0.9)
        ax.bar(x, hi - lo, W, bottom=lo, zorder=3, **kw_)
        if len(r0):
            ax.plot([x - W / 2, x + W / 2], [100 * r0[key].iloc[0]] * 2, color=S.INK, lw=1.8,
                    solid_capstyle='butt', zorder=4)
for x in [i - 0.5 for i in range(1, len(sites)) if sites[i][:3] != sites[i - 1][:3]]:
    ax.axvline(x, color=S.LIGHT_GUIDE, lw=0.7, ls=':', zorder=1)
ax.axhline(0.0, color=S.INK, lw=0.7, zorder=2)
ax.set_xticks(range(len(sites)))
ax.set_xticklabels(sites)
ax.tick_params(axis='x', length=0)
ax.set_xlim(-0.6, len(sites) - 0.4)
ax.set_ylim(-52)

S.tidy(ax, axis='y')
ax.set_ylabel('onshore flux of the deficit\nbelow 20 °C (°C cm s$^{-1}$)')
grey = S.LIGHT_GUIDE
ax.legend([S.patch_handle(grey, ec='black'), S.patch_handle('white', ec=grey, hatch=S.DOWN_HATCH),
           S.line_handle(S.INK, lw=1.8)], ['mean flow', 'correlated fluctuations', 'isobath frame'],
          loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=1.2, handletextpad=0.4,
          borderaxespad=0.0, columnspacing=1.0)

fig.savefig(V.DIR_FIG / 'figS1_flux.pdf')
S.save(fig, V.DIR_FIG / 'figS1_flux.png')
