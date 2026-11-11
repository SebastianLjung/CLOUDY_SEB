# Tests of the cell selection of plot_SPHINXHIIs.py: models a few of the cells it does not model with Cloudy anyway,
# to see how much the halo's escaping LyC would change if they were modelled instead of given a simple f_esc.
#
# Groups (cells with stars that SPHINX has not ionised, x_HII <= XHII_MAX; cells above TEST_XHII_MAX are never tested):
#   A        nearly ionised by SPHINX, A_XHII_MIN < x_HII <= XHII_MAX: would they be fine trusted, as above XHII_MAX?
#                                                                                           -> f_esc = 1 if trusted
#   B        the stars could ionise the whole cell (N_ion_max >= N_target). Used to be assumed f_esc = 1; no longer a
#            filter, but most of these cells are faint (left out)
#   C        neutral column just above the cut: N_HI_MAX < N_HI_min < C_EDGE                -> assumed f_esc = 0
#   C_thick  neutral column C_EDGE <= N_HI_min <= C_THICK_MAX (only the cells picked before, no new ones)
#                                                                                           -> assumed f_esc = 0
#   D        would be modelled, but Q(H) <= Q_MIN (and not in B)                            -> left out (counted as 0)
#   TOL      modelled cells, rerun with N_H matched within TOL_TEST instead of TOL
# and two groups of cells SPHINX has ionised (x_HII > XHII_MAX), trusted:                   -> f_esc = 1
#   A2       A2_XHII_RANGE[0] < x_HII <= A2_XHII_RANGE[1]: does trusting SPHINX hold there?
#   A3       A3_NHI_RANGE[0] < N_HI_min < A3_NHI_RANGE[1] (any x_HII, so also above TEST_XHII_MAX): does a tiny
#            SPHINX neutral column really mean the LyC escapes?
#   N15, N16, N17   N_HI_min within NHI_BAND_DEX of 1e15, 1e16, 1e17 cm^-2 and x_HII < 1 (cells with x_HII == 1 are
#            trusted anyway): f_esc against the SPHINX neutral column, to choose N_HI_THIN
#
# Usage:  python filter_tests.py [pick] [run] [tol] [analyse]   (no arguments: all four)
#   pick:     choose the B, C and D cells: at random, with probability proportional to Q(H), so that the plain mean of
#             their f_esc estimates the Q(H)-weighted mean of the group. Cells picked before are kept if they still
#             belong to their group, and more are picked to reach N_PICK  -> HALO_DIR/filter_tests/cells.csv
#   run:      model them with plot_SPHINXHIIs.py (same deck)          -> HALO_DIR/filter_tests/models
#   tol:      pick the TOL cells from the main run's results and rerun them with TOL_TEST (needs the main run to have
#             finished)                                               -> HALO_DIR/filter_tests/tol_models
#   analyse:  per group, the Cloudy f_esc vs the assumed one and the implied change of the halo's escaping LyC
import re
import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
MAIN = HERE / 'plot_SPHINXHIIs.py'
N_PICK = {'A': 6, 'A2': 6, 'A3': 6, 'B': 8, 'C': 6, 'C_thick': 0, 'D': 6, 'N15': 4, 'N16': 4, 'N17': 4,
          'TOL': 6}                                                                    # (new) cells per group
ASSUMED = {'A': 1.0, 'A2': 1.0, 'A3': 1.0, 'B': 1.0, 'C': 0.0, 'C_thick': 0.0, 'D': 0.0,
           'N15': 1.0, 'N16': 1.0, 'N17': 1.0}                                          # f_esc when not modelled
NHI_BAND_DEX = 0.25                               # half-width of the N15, N16, N17 bands [dex]
TEST_XHII_MAX = 0.99                              # never test cells SPHINX has ionised more than this (except A3)
A_XHII_MIN = 0.9
A2_XHII_RANGE = (0.95, 0.99)
A3_NHI_RANGE = (1e14, 1e16)                       # [cm^-2]
C_EDGE = 3e20                                     # [cm^-2]
C_THICK_MAX = 5e21                                # [cm^-2]
TOL_TEST = 0.03
N_THREADS = 10                                    # Cloudy models at the same time (leaves room for the main run)
TEST_CONTINUUM_RESOLUTION = 1.0                   # for the A-D cells: Cloudy's default, faster than the main run's 0.1
                                                  # and plenty to tell f_esc ~ 0 from ~ 1 (TOL keeps the main run's)
SEED = 1


def main_source(**params):
    # Source of plot_SPHINXHIIs.py with some of its KEY PARAMETERS replaced (each line `NAME = ...` at the start)
    src = MAIN.read_text()
    for name, value in params.items():
        src, n = re.subn(rf'^{name} = .*$', f'{name} = {value!r}', src, count=1, flags=re.M)
        assert n == 1, f'parameter {name} not found in {MAIN.name}'
    return src


def run_main(**params):
    # Run plot_SPHINXHIIs.py with these KEY PARAMETERS
    exec(compile(main_source(**params), str(MAIN), 'exec'), {'__file__': str(MAIN)})


def pick(group, pool, picked):
    # Cells of `group` for the test: those in `picked` that are still in `pool`, plus N_PICK[group] new ones from
    # `pool` (random, probability proportional to Q(H)) if there are not yet that many
    keep = [c for c in picked.get(group, []) if c in set(pool)]
    rest = np.setdiff1d(pool, keep)
    n_new = min(N_PICK[group] - len(keep), len(rest))
    new = rng.choice(rest, size=n_new, replace=False, p=Q_cell[rest] / Q_cell[rest].sum()) if n_new > 0 else []
    return keep + [int(c) for c in new]


def save_cells(cells_by_group):
    rows = [{'cell': c, 'group': k, 'Q(H) [1/s]': Q_cell[c], 'x_HII SPHINX': g['gas']['xHII'][c],
             'N_ion_max / N_target': g['crit'][c], 'N_HI_min [cm^-2]': g['N_HI_min'][c]}
            for k, cells in cells_by_group.items() for c in cells]
    pd.DataFrame(rows).to_csv(CELLS_FILE, index=False)
    print(f'\nTest cells (saved to {CELLS_FILE}):')
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.3g}'))


# Everything plot_SPHINXHIIs.py knows at its cell selection (runs only its loading + selection part, no Cloudy)
g = {'__file__': str(MAIN)}
exec(compile(main_source().split('\nif CELL_LIST is not None:')[0], str(MAIN), 'exec'), g)
Q_cell, HALO_DIR = g['Q_cell'], g['HALO_DIR']
TEST_DIR = HALO_DIR / 'filter_tests'
MAIN_DIR = HALO_DIR / g['MODELS_SUBDIR']
CELLS_FILE = TEST_DIR / 'cells.csv'
stages = sys.argv[1:] or ['pick', 'run', 'tol', 'analyse']
rng = np.random.default_rng(SEED)

testable = g['has_stars'] & ~g['ionised'] & (g['gas']['xHII'] <= TEST_XHII_MAX)
B = testable & (g['crit'] >= 1)
N_HI_min = g['N_HI_min']
groups = {'A': testable & (g['gas']['xHII'] > A_XHII_MIN),
          'B': B,
          'C': testable & g['thick'] & (N_HI_min < C_EDGE),
          'C_thick': testable & (N_HI_min >= C_EDGE) & (N_HI_min <= C_THICK_MAX),
          'D': testable & ~g['thick'] & g['faint'] & ~B,
          'A2': g['has_stars'] & (g['gas']['xHII'] > A2_XHII_RANGE[0]) & (g['gas']['xHII'] <= A2_XHII_RANGE[1]),
          'A3': g['has_stars'] & (N_HI_min > A3_NHI_RANGE[0]) & (N_HI_min < A3_NHI_RANGE[1])}
for e in [15, 16, 17]:
    groups[f'N{e}'] = (g['has_stars'] & (g['gas']['xHII'] < 1)
                       & (np.abs(np.log10(np.maximum(N_HI_min, 1e-30)) - e) < NHI_BAND_DEX))
Q_share = {k: Q_cell[m].sum() / Q_cell.sum() for k, m in groups.items()}
Q_share['TOL'] = Q_cell[g['selected']].sum() / Q_cell.sum()
picked = {}
if CELLS_FILE.exists():
    old = pd.read_csv(CELLS_FILE)
    picked = {k: list(old['cell'][old['group'] == k]) for k in old['group'].unique()}


###########################################################
#               PICK AND RUN THE B, C, D CELLS            #
###########################################################
# Same deck as the main run; the cells start from the default first guess of hden

if 'pick' in stages:
    TEST_DIR.mkdir(exist_ok=True)
    tol_kept = [c for c in picked.get('TOL', []) if g['selected'][c] and g['gas']['xHII'][c] <= TEST_XHII_MAX]
    picked = {k: pick(k, np.where(m)[0], picked) for k, m in groups.items()} | {'TOL': tol_kept}
    save_cells(picked)

if 'run' in stages:
    cells = [c for k in groups for c in picked.get(k, [])]
    np.savetxt(TEST_DIR / 'filter_cells.txt', cells, fmt='%d')
    print(f'\n======== {len(cells)} B, C, D cells ========')
    run_main(CELL_LIST=str(TEST_DIR / 'filter_cells.txt'), MODELS_SUBDIR='filter_tests/models', TEST=False,
             N_THREADS=N_THREADS, CONTINUUM_RESOLUTION=TEST_CONTINUUM_RESOLUTION)


###########################################################
#                 PICK AND RUN THE TOL CELLS              #
###########################################################
# Converged cells of the main run that are not already within TOL_TEST, rerun from their converged hden
# (copied into their folder) until N_H is matched within TOL_TEST

if 'tol' in stages:
    main_res = pd.read_csv(MAIN_DIR / 'results.csv')
    pool = main_res['cell'][main_res['converged'] & (np.abs(main_res['N_H Cloudy / SPHINX'] - 1) > TOL_TEST)
                            & (main_res['x_HII SPHINX'] <= TEST_XHII_MAX)].values
    picked['TOL'] = pick('TOL', pool, picked)
    save_cells(picked)
    np.savetxt(TEST_DIR / 'tol_cells.txt', picked['TOL'], fmt='%d')

    print(f'\n======== {len(picked["TOL"])} TOL cells (TOL = {TOL_TEST}) ========')
    (TEST_DIR / 'tol_models').mkdir(parents=True, exist_ok=True)
    shutil.copy(MAIN_DIR / 'hden_converged.json', TEST_DIR / 'tol_models' / 'hden_converged.json')
    run_main(CELL_LIST=str(TEST_DIR / 'tol_cells.txt'), MODELS_SUBDIR='filter_tests/tol_models', TOL=TOL_TEST, TEST=False,
             N_THREADS=N_THREADS)


###########################################################
#                        ANALYSE                          #
###########################################################
# The change of the halo's escaping LyC (as a fraction of all stellar Q(H)) if a group were modelled instead of given
# its assumed f_esc:  Q share of the group * (mean Cloudy f_esc - assumed f_esc)

if 'analyse' in stages:
    cells = pd.read_csv(CELLS_FILE)
    res = pd.concat([pd.read_csv(p) for p in [TEST_DIR / 'models' / 'results.csv', TEST_DIR / 'tol_models' / 'results.csv']
                     if p.exists()])
    cells = cells.merge(res[['cell', 'f_esc total', 'converged', 'N_H Cloudy / SPHINX']].drop_duplicates('cell'),
                        on='cell', how='left')
    print('\nPer cell:')
    print(cells.to_string(index=False, float_format=lambda x: f'{x:.3g}'))

    print('\nPer group (change of the halo f_esc = escaping / all stellar LyC, if modelled with Cloudy):')
    for k in groups:
        f = cells['f_esc total'][(cells['group'] == k) & cells['converged'].fillna(False).astype(bool)].values
        if not len(f):
            print(f'  {k}    no converged results yet')
            continue
        err = np.std(f) / np.sqrt(len(f))
        print(f'  {k}    Q share {Q_share[k]:.4f}: Cloudy f_esc mean {np.mean(f):.3f} +- {err:.3f} ({len(f)} cells, '
              f'{f.min():.2g}-{f.max():.2g}), assumed {ASSUMED[k]:g} -> halo f_esc change '
              f'{Q_share[k] * (np.mean(f) - ASSUMED[k]):+.5f} +- {Q_share[k] * err:.5f}')

    s = cells[cells['group'] == 'TOL'].dropna(subset=['f_esc total'])
    if len(s):
        main_f = pd.read_csv(MAIN_DIR / 'results.csv').set_index('cell').loc[s['cell'], 'f_esc total'].values
        df = s['f_esc total'].values - main_f
        print(f'  TOL  Q share {Q_share["TOL"]:.4f}: f_esc change with TOL {TOL_TEST} instead of {g["TOL"]}: mean '
              f'{np.mean(df):+.4f} (cells {df.min():+.4f} to {df.max():+.4f}) -> halo f_esc change '
              f'{Q_share["TOL"] * np.mean(df):+.5f}')
