"""
One look for every figure of the third version.

The palette is built for readers with any of the common colour vision
deficiencies. The two lines of moorings carry a blue and an orange family,
the pair that stays separable under protanopia, deuteranopia and tritanopia
(Okabe and Ito 2008, Crameri et al. 2020), each darkening offshore so that the
200 m sites are the strongest colours. The site between the lines is grey.
Every site also carries its own marker and line style, so a figure printed in
greyscale keeps every series apart. Colours are named so that any of them can
be changed in one place, fields use perceptually ordered colormaps, and text
inside the data area is kept to the minimum.

Created by Arian Dialectaquiz.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# ------------------------------------------------------------------ rc
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'mathtext.default': 'regular',
    'font.size': 7.0,
    'axes.titlesize': 7.5,
    'axes.labelsize': 7.0,
    'axes.titlepad': 3.0,
    'axes.labelpad': 2.0,
    'xtick.labelsize': 6.3,
    'ytick.labelsize': 6.3,
    'legend.fontsize': 7.0,
    'legend.frameon': False,
    'legend.handlelength': 2.2,
    'legend.columnspacing': 1.2,
    'legend.labelspacing': 0.35,
    'axes.linewidth': 0.55,
    'axes.edgecolor': 'black',
    'xtick.major.width': 0.55,
    'ytick.major.width': 0.55,
    'xtick.major.size': 2.4,
    'ytick.major.size': 2.4,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'grid.color': 'gainsboro',
    'grid.linewidth': 0.35,
    'lines.linewidth': 1.0,
    'lines.solid_capstyle': 'round',
    'hatch.linewidth': 0.5,
    'savefig.dpi': 400,
    'savefig.bbox': None,
    'figure.dpi': 110,
})

# -------------------------------------------------------------- sites
SITE_COLOURS = {'FUB1': 'lightsteelblue', 'FUB2': 'cornflowerblue', 'FUB3': 'navy',
                'FBG3': 'dimgray',
                'FCF1': 'navajowhite', 'FCF2': 'sandybrown', 'FCF3': 'darkorange'}
SITE_MARKERS = {'FUB1': '^', 'FUB2': 'D', 'FUB3': 'o',
                'FBG3': 's',
                'FCF1': 'v', 'FCF2': 'P', 'FCF3': 'h'}
SITE_LINESTYLES = {'FUB3': '-', 'FBG3': '--', 'FCF3': '-.',
                   'FUB2': (0, (5, 1.5)), 'FCF2': (0, (3, 1, 1, 1)),
                   'FUB1': ':', 'FCF1': (0, (1, 1.5))}
LINE_COLOURS = {'Ubatuba': 'navy', 'Cabo Frio': 'darkorange'}
DEEP_COLOUR = 'black'        # deep slope mooring FBC
DEEP_MARKER = 'X'
LEGEND_SIZE = 7.2
CLIM_MARK = 'mediumvioletred'  # climatology at the moorings, visible over every site colour
SEASON_RANGE = 'thistle'       # range of the four seasons of the climatology
UNARRESTED = 'khaki'           # band of an unarrested Ekman layer
ARRESTED = 'black'             # an arrested layer, drawn dotted

# ------------------------------------------------------- neutral elements
INK = 'black'
GUIDE = 'dimgray'            # reference lines of a theory or of zero
LIGHT_GUIDE = 'darkgray'
RETAINED_BAR = 'silver'      # records of the calendar
SEASON_SHADE = 'whitesmoke'  # October to March in time axes

# ----------------------------------------------------------- maps and sections
LAND = 'whitesmoke'
COAST = 'dimgray'
ISOBATH = 'darkgray'
SHELF_BREAK = 'black'
STREAM = 'black'
WIND = 'dimgray'
BOTTOM_FILL = 'lightgray'
BOTTOM_EDGE = 'dimgray'
ISOTHERM = 'black'
# the synthesis of Figure 7
SYN_TRANSPORT = 'firebrick'    # cross-isobath transport of the bottom layer
SYN_WAVE = 'rebeccapurple'     # coastal trapped wave
SYN_CURRENT = 'teal'           # Brazil Current
SYN_CAPPED = 'lightsteelblue'  # stratified layer beneath upslope transport
SYN_MIXED = 'navajowhite'      # mixed layer beneath downslope transport
CMAP_EKE = 'YlGnBu'          # sequential and ordered in lightness
CMAP_DIVERGING = 'RdBu_r'    # red and blue diverging, separable without green
CMAP_SEQUENTIAL = 'cividis'


def truncated(name, lo=0.04, hi=0.86, n=256):
    """Part of a named colormap, which keeps the darkest end for the lines drawn over it."""
    return ListedColormap(plt.get_cmap(name)(np.linspace(lo, hi, n)))


# -------------------------------------------------------------- helpers
def panel(ax, letter, dx=-0.01, dy=1.012, **kw):
    """Bold panel letter outside the upper left corner of the axes."""
    ax.text(dx, dy, f'({letter})', transform=ax.transAxes, fontsize=7.5, fontweight='bold',
            ha='right', va='bottom', **kw)


def tidy(ax, grid=True, axis='both'):
    """Grid behind the data and no upper or right spine."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)
    return ax


def line_handle(colour, ls='-', lw=1.1, marker=None, ms=3.8, mfc=None):
    return Line2D([], [], color=colour, ls=ls, lw=lw, marker=marker, ms=ms,
                  mfc=colour if mfc is None else mfc, mec='black' if marker else colour,
                  mew=0.45)


def marker_handle(colour, marker, ms=4.2, open_=False):
    return Line2D([], [], color=colour, ls='none', marker=marker, ms=ms,
                  mfc='white' if open_ else colour, mec='black', mew=0.45)


def patch_handle(colour, ec='none', hatch=None, alpha=1.0):
    return Patch(fc=colour, ec=ec, hatch=hatch, alpha=alpha, lw=0.5)


def ringed_marker(ax, x, y, marker, colour, ms=4.6, zorder=9):
    """Site marker with a white ring, which keeps it readable over a field."""
    ax.plot(x, y, marker=marker, ms=ms + 2.0, mfc='white', mec='white', ls='none', zorder=zorder)
    ax.plot(x, y, marker=marker, ms=ms, mfc=colour, mec='black', mew=0.5, ls='none',
            zorder=zorder + 1)


def degree_ticks(ax, axis='both'):
    """Longitude and latitude ticks with the hemisphere in place of a sign."""
    from matplotlib.ticker import FuncFormatter
    lon_fmt = FuncFormatter(lambda x, _: f'{abs(x):g}°{"W" if x < 0 else "E"}')
    lat_fmt = FuncFormatter(lambda y, _: f'{abs(y):g}°{"S" if y < 0 else "N"}')
    if axis in ('both', 'x'):
        ax.xaxis.set_major_formatter(lon_fmt)
    if axis in ('both', 'y'):
        ax.yaxis.set_major_formatter(lat_fmt)


def geo_aspect(ax, lat0):
    """Equal distance aspect for plain longitude and latitude axes."""
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lat0)))


def save(fig, path, tight=False):
    if tight:
        fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    else:
        fig.savefig(path)
    plt.close(fig)
    print(f'saved {path}')