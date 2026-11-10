"""
Settings of the third version of the shelf-break paper, the bottom boundary
layer beneath the Brazil Current at the 200 m isobath of the northern South
Brazil Bight and the supply of South Atlantic Central Water to the shelf.

The third version is rebuilt from the records themselves. The quality
controlled and low-passed series of the first version (paper_old/data) and the
site geometry (paper_old/tables) are read, never recomputed, and every product
of the paper is written inside paper_v3.

Run order, from /data1/currents/sbb_dynamics_paper_arian/paper_v3
    python inputs_v3.py          report where every input is found
    python fig01_compute.py      geometry, frame, fields, sections, calendar
    python fig01_plot.py         figure 1

Created by Arian Dialectaquiz.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / 'paper_old'
V2 = ROOT / 'paper_v2'
if str(OLD) not in sys.path:
    sys.path.append(str(OLD))

try:
    import config as C                   # paths of the exploratory pipeline
except Exception:                        # noqa: BLE001  the defaults below take over
    C = None


def _old(name, default):
    return getattr(C, name, default) if C is not None else default


# ----------------------------------------------------------------- inputs
BATHY_FILE = _old('BATHY_FILE', None)
CLIM_DIR = Path(_old('CLIMATOLOGY_DIR', Path('/data1/currents/climatology_sbb')))
ALT_DIR = ROOT / 'altimetry_data'
ERA5_DIR = ROOT / 'era5_data'
GLORYS_DIR = ROOT / 'glorys_data'
SST_DIR = ROOT / 'sst_data'
QC_DIR = ROOT / 'data_qc'
ISOBATH_RADIUS_KM = _old('ISOBATH_RADIUS_KM', 5.0)

# ---------------------------------------------------------------- outputs
DIR_DATA = HERE / 'data'
DIR_TAB = HERE / 'tables'
DIR_FIG = HERE / 'figures'
DIR_LOG = HERE / 'logs'


def make_dirs():
    for d in (DIR_DATA, DIR_TAB, DIR_FIG, DIR_LOG):
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ sites
SITES_200 = ['FUB3', 'FBG3', 'FCF3']
SITES_100 = ['FUB2', 'FCF2']
SITES_50 = ['FUB1', 'FCF1']
SITES = SITES_200 + SITES_100 + SITES_50
DEEP_SITES = ['FBC']                     # deep slope mooring, inventoried only for now
LINES = {'Ubatuba': ['FUB1', 'FUB2', 'FUB3'], 'Cabo Frio': ['FCF1', 'FCF2', 'FCF3']}
LEVELS = {'FUB3': [23, 85, 155, 194], 'FBG3': [23, 85, 155, 194], 'FCF3': [23, 85, 155, 194],
          'FUB2': [23, 75, 94], 'FCF2': [23, 75, 94], 'FUB1': [10, 30, 44], 'FCF1': [10, 30, 44]}
UPPER_LEVEL = {'FUB3': 85, 'FBG3': 85, 'FCF3': 85, 'FUB2': 23, 'FCF2': 23, 'FUB1': 10, 'FCF1': 10}
INTERIOR_LEVEL = {'FUB3': 155, 'FBG3': 155, 'FCF3': 155, 'FUB2': 75, 'FCF2': 75, 'FUB1': 30,
                  'FCF1': 30}
BOTTOM_LEVEL = {'FUB3': 194, 'FBG3': 194, 'FCF3': 194, 'FUB2': 94, 'FCF2': 94, 'FUB1': 44,
                'FCF1': 44}
WATER_DEPTH_M = {'FUB3': 200.0, 'FBG3': 200.0, 'FCF3': 200.0, 'FUB2': 100.0, 'FCF2': 100.0,
                 'FUB1': 50.0, 'FCF1': 50.0}

# --------------------------------------------------------- halves of the year
# Kept as a test of robustness. The second version showed that the
# stratification above the bottom at 200 m barely follows the calendar.
HALVES = ('strat', 'mixed')
HALF_OF_MONTH = {10: 'strat', 11: 'strat', 12: 'strat', 1: 'strat', 2: 'strat', 3: 'strat',
                 4: 'mixed', 5: 'mixed', 6: 'mixed', 7: 'mixed', 8: 'mixed', 9: 'mixed'}
HALF_SHORT = {'strat': 'October to March', 'mixed': 'April to September'}

# ------------------------------------------------------------- period
PERIOD = ('2001-05-01', '2004-07-31')        # records with two months of margin
DT_H = 1.0
BOOT_BLOCK_DAYS = 5
N_BOOT = 400

# ------------------------------------------------------- constants
RHO0 = 1025.0
RHO_AIR = 1.22
G = 9.81
OMEGA = 7.2921e-5
KAPPA = 0.41

# ---------------------------------------------- frame of the isobaths
# The cross-isobath velocity is the component most exposed to the choice of
# frame, so the bearing of the isobath is measured over several radii and set
# against the principal axis of the subinertial flow at every level.
FRAME_RADII_KM = (2.5, 5.0, 10.0, 20.0)

# --------------------------------------------------------------- figure 1
MAP_EXTENT = (-46.6, -40.6, -25.4, -22.4)        # lon_min, lon_max, lat_min, lat_max
INSET_BOX = (-53.5, -38.5, -33.0, -19.5)
MAP_ISOBATHS = (50, 100, 1000, 2000)
SHELF_BREAK = 200
ALT_MIN_DEPTH = 100.0      # m, altimetric fields left out over shallower water
EKE_LEVELS = (0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050)
STREAM_DENSITY = 1.15
SPEED_LW = (0.25, 1.6)     # width of the streamlines from rest to SPEED_REF
SPEED_REF = 0.45           # m s-1 given the widest streamline
WIND_STRIDE = 4            # every how many ERA5 points a wind vector is drawn
WIND_KEY = 0.05            # N m-2 of the wind stress key
REMOTE_WIND_LATS = (-27.0, -30.0)   # remote forcing of the coastal trapped waves
REMOTE_ISOBATH = 100.0
PLACE_NAMES = {'Cabo Frio': (-42.02, -22.88), 'Ubatuba': (-45.07, -23.43)}
PLACE_OFFSETS = {'Cabo Frio': ((0, 6), 'center'), 'Ubatuba': ((-5, 2), 'right')}
LABEL_OFFSETS = {'FUB1': (-22, 4), 'FUB2': (6, 3), 'FUB3': (6, -9), 'FBG3': (-8, 7),
                 'FCF1': (-22, 4), 'FCF2': (6, -2), 'FCF3': (6, -8)}
INSTRUMENT_CLEARANCE_M = 7.0   # least distance drawn between a current meter and the bottom

# sections along the lines
SECTION_KM = (-6.0, 160.0)
SECTION_STEP_KM = 1.0
SECTION_ZMAX = 262.0
ALT_AXIS_MIN = -0.10       # m s-1, weakest normal velocity accepted as the axis of the current
ISOTHERMS = (12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0)
TW_SACW = 20.0             # upper limit of the central water (Castro and Miranda 1998)
T_SECTION_SOURCE = 'climatology'     # 'climatology' of Sasaki et al. (2024), else 'glorys'
CLIM_PATTERN = 'climatology*.nc'

# calendar
GAP_MERGE_H = 24.0         # gaps shorter than this inside one segment are ignored
MIN_SEGMENT_H = 24.0

# --------------------------------------------------------------- figure 2
# bottom slope from GEBCO
SLOPE_RES_DEG = 0.02       # GEBCO block averaged to this spacing before the gradient
SLOPE_SMOOTH_KM = 5.0      # gradient averaged over this width, as the frame of the sites
# stratification of the climatology over the bottom of every grid point
BBL_N_THICKNESS = 50.0     # m above the bottom over which N is fitted
BBL_FINE_DZ = 2.0          # m of the regular grid the profile is interpolated onto
BBL_MIN_LEVELS = 2         # levels of the climatology required inside the layer
BOTTOM_MISMATCH_M = 60.0   # largest accepted difference between the two bottoms
MAP_DEPTH_RANGE = (100.0, 1000.0)
SEASONS = ('DJF', 'MAM', 'JJA', 'SON')
# the 200 m isobath followed along the shelf break
PATH_STEP_KM = 2.0
PATH_SMOOTH_KM = 20.0      # running mean that keeps the path off the roughness of the bottom
PATH_IDW_KM = 20.0         # radius of the inverse distance weights along the path
# moorings
MOORED_LAYER = (155, 194)          # levels of the moored stratification at 200 m
MOORED_LAYER_FALLBACK = (85, 155)  # used where the lowest meter is missing
TS_RADIUS_KM = 30.0        # climatology points used for the temperature and salinity relation
TS_DEPTHS = (80.0, 300.0)  # depths of that relation, the central water above the bottom
GLORYS_SEARCH_KM = 40.0    # radius searched for a reanalysed column deep enough
MIN_DAY_HOURS = 20         # hours needed for a daily mean
# arrest
CD = 2.0e-3                # quadratic drag on the interior velocity of the layer
CD_RANGE = (1.0e-3, 3.0e-3)
MIN_SPEED = 0.05           # m s-1 that defines the sign of a state of the interior flow
MIN_STATE_H = 24
S_LEVELS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0)
S_CONSTANT = 35.5          # salinity used where the climatology has none, which bounds N2 from above

# --------------------------------------------------------------- figure 3
PAIRS = {'interior': (85, 155), 'bottom': (155, 194)}   # the control and the frictional pair
CONDITION_LEVEL = 155       # level whose sign defines the state of the interior flow
Z0_RANGE = (1.0e-3, 5.0e-3, 2.0e-2)   # roughness lengths of the unarrested reference
U_RANGE = (0.10, 0.20, 0.30)          # interior speeds of the unarrested reference
SPEED_BINS = (0.0, 0.6, 0.04)         # bins of the interior speed, start, end, width
MIN_BIN_HOURS = 48

# ---------------------------------------------------- screening and recovery
RING_MAX = 0.25            # first percentile over median speed above which a record carries an
                           # offset of its acoustic paths, sound records sit between 0.09 and 0.16
MAX_MEDIAN_SPEED = 80.0    # cm s-1, a median speed above this is a unit or sensor fault
RECOVER = [('FBG3', '1', 155)]     # records rebuilt from data_qc, chosen one by one
LEVEL_MATCH_M = 12.0       # nominal level of a record matched to a level of the low-passed series
LP_CUTOFF_H = 40.0         # Lanczos low-pass of the first version (Duchon 1979)
LP_HALF_WINDOW_H = 120
GAP_FILL_H = 3
DIR_REC = HERE / 'data' / 'records'

# --------------------------------------------------------------- figure 4
BUDGET_LEVELS = (85, 155, 194)     # upper level of the background gradient, interior, lowest
TIME_EDGES = (-1.0, 2.0, 13)       # log10 of the scaled time since a reversal, start end count
TIME_SCALE = 'canonical'           # 'canonical' (|f| s^2)^-1 or 'wedge' 1/(2 Cd N s^3)
MIN_BIN_STATES = 15
BUDGET_BINS = 8
SPEED_BAND = (0.10, 0.30)          # m s-1 of the interior flow held fixed in the time composites
STATE_EDGE_H = 6                   # hours averaged at each end of a state for its change
U_REF = 0.15                       # m s-1, interior speed to which every hourly gain is corrected
TAU_GRID_H = (3.0, 1200.0, 48)     # restoring times searched by the layer model, start end count
MIN_STATE_COMPLETE = 0.9           # fraction of hours a state needs with every variable

# --------------------------------------------------------------- figure 5
CTW_ISOBATH = 100.0                # isobath along which the coastal trapped wave is forced
CTW_EXTENT = (-54.0, -38.0, -33.5, -21.5)
CTW_STEP_KM = 20.0
CTW_DT_H = 3                       # time step of the forcing, hours
CTW_C = tuple(float(c) for c in range(2, 15))            # phase speeds searched, m s-1
CTW_TF_DAYS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0) # frictional decay times searched
CTW_LIT_C = (6.0, 11.0)            # phase speeds of Dottori and Castro (2018) in the bight
ENC_SECTION_KM = (-40.0, 160.0)    # cross-isobath section through each 200 m site
ENC_AXIS_MIN = -0.10               # m s-1, weakest along-isobath velocity accepted as the axis
COH_SEG_DAYS = 40                  # segment of the cross spectra
SYNOPTIC_DAYS = (2.0, 10.0)
MESOSCALE_DAYS = (15.0, 40.0)
N_SURROGATE = 500

# --------------------------------------------------------------- figure 6
EVENT_DROP = 1.0                   # degrees of cooling in EVENT_WINDOW_H that mark a cold event
EVENT_DROP_TESTS = (0.5, 1.0, 1.5)
EVENT_WINDOW_H = 48
EVENT_SEP_H = 72
COMPOSITE_DAYS = (-8, 12)
BASELINE_DAYS = (-3, -1)
REACH_DROP = 0.5                   # degrees an inner site must cool within REACH_DAYS to count as reached
REACH_DAYS = 10
LAG_DAYS = (-5, 15)
ADVECTIVE_SPEEDS = (0.02, 0.05, 0.10)   # m s-1 drawn against the lags
SACW_T = 20.0                      # upper temperature of the central water (Castro and Miranda 1998)
SACW_T_TESTS = (18.0, 20.0)
MIN_FLUX_DAYS = 60                 # a site enters panel (f) of Figure 6 with at least this record