# Tests of the cell filters of plot_SPHINXHIIs.py: models a few cells of each group that the filters leave out (and
# assume f_esc = 1 or 0 for) with Cloudy, and reruns a few converged cells with a tighter TOL, to see which filter or
# assumption changes the halo's escaping LyC the most.
#
# Groups (cells with stars, as in the CELL SELECTION section of plot_SPHINXHIIs.py):
#   A     already ionised by SPHINX (x_HII > XHII_MAX), not near-critical         -> assumed f_esc = 1
#   A_nc  already ionised by SPHINX and near-critical: assumed f_esc = 1 too, but not sampled here: most were modelled
#         by the main run before ionised cells were excluded, so their saved results in MODELS_SUBDIR are used
#   C     not ionised, not near-critical, neutral column N_HI_min > N_HI_MAX      -> assumed f_esc = 0
#   D     not ionised, would be selected, but Q(H) <= Q_MIN                       -> not modelled
#   (B, density bounded, holds ~0 of the halo's Q(H), so it is not tested)
#   TOL   selected cells converged within TOL, rerun with TOL_TEST
#
# Usage:  python filter_tests.py [pick] [run] [tol] [analyse]   (no arguments: all four)
#   pick:     choose the cells (random, with probability proportional to Q(H)) -> filter_tests/*.txt, cells.csv
#   run:      model the group A, C, D cells with plot_SPHINXHIIs.py (same deck) in HALO_DIR/filter_tests/models
#   tol:      rerun the TOL cells with TOL_TEST in HALO_DIR/filter_tests/tol_models (pick and tol need the main run
#             to have finished, so that the TOL cells and their starting hden come from its final results)
#   analyse:  Cloudy f_esc vs the assumed f_esc per group, and the implied change of the halo's escaping LyC
import re
import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
MAIN = HERE / 'plot_SPHINXHIIs.py'
N_PICK = {'A': 8, 'C': 6, 'D': 6, 'TOL': 6}     # cells per group
ASSUMED = {'A': 1.0, 'C': 0.0}                    # f_esc assumed for the group when not modelled
TOL_TEST = 0.03
SEED = 1


def main_source(**params):
    # Source of plot_SPHINXHIIs.py with some of its KEY PARAMETERS replaced (each line `NAME = ...` at the start)
    src = MAIN.read_text()
    for name, value in params.items():
        src, n = re.subn(rf'^{name} = .*$', f'{name} = {value!r}', src, count=1, flags=re.M)
        assert n == 1, f'parameter {name} not found in {MAIN.name}'
    return src


def selection_state():
    # Everything plot_SPHINXHIIs.py knows at the cell selection (runs only its loading + selection part, no Cloudy)
    src = main_source().split('\nif CELL_LIST is not None:')[0]
    g = {'__file__': str(MAIN)}
    exec(compile(src, str(MAIN), 'exec'), g)
    return g


g = selection_state()
HALO_DIR, Q_cell = g['HALO_DIR'], g['Q_cell']
TEST_DIR = HALO_DIR / 'filter_tests'
MAIN_DIR = HALO_DIR / g['MODELS_SUBDIR']
stages = sys.argv[1:] or ['pick', 'run', 'tol', 'analyse']

other = g['has_stars'] & ~g['near_critical']
groups = {'A': other & g['ionised'],
          'C': other & ~g['ionised'] & ~g['density_bounded'] & g['thick'],
          'D': g['has_stars'] & ~g['ionised'] & (g['near_critical'] | g['partial_escape']) & g['faint']}
A_nc = g['has_stars'] & g['ionised'] & g['near_critical']
Q_share = {k: Q_cell[m].sum() / Q_cell.sum() for k, m in groups.items()}
Q_share['TOL'] = Q_cell[g['selected']].sum() / Q_cell.sum()


###########################################################
#                     PICK THE CELLS                      #
###########################################################

if 'pick' in stages:
    TEST_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    main_res = pd.read_csv(MAIN_DIR / 'results.csv')
    tol_pool = main_res[main_res['converged'] & (np.abs(main_res['N_H Cloudy / SPHINX'] - 1) > TOL_TEST)]['cell'].values
    pools = {k: np.where(m)[0] for k, m in groups.items()} | {'TOL': tol_pool}

    rows = []
    for k, pool in pools.items():
        p = Q_cell[pool] / Q_cell[pool].sum()
        for c in rng.choice(pool, size=min(N_PICK[k], len(pool)), replace=False, p=p):
            rows.append({'cell': int(c), 'group': k, 'Q(H) [1/s]': Q_cell[c], 'x_HII SPHINX': g['gas']['xHII'][c],
                         'N_ion_max / N_target': g['crit'][c], 'N_HI_min [cm^-2]': g['N_HI_min'][c]})
    cells = pd.DataFrame(rows)
    cells.to_csv(TEST_DIR / 'cells.csv', index=False)
    np.savetxt(TEST_DIR / 'filter_cells.txt', cells[cells['group'] != 'TOL']['cell'], fmt='%d')
    np.savetxt(TEST_DIR / 'tol_cells.txt', cells[cells['group'] == 'TOL']['cell'], fmt='%d')
    print(f'\nPicked cells (saved to {TEST_DIR / "cells.csv"}):')
    print(cells.to_string(index=False, float_format=lambda x: f'{x:.3g}'))


###########################################################
#                 MODEL THEM WITH CLOUDY                  #
###########################################################
# Same deck as the main run. Filter-test cells start from the default first guess; TOL-test cells from the main run's
# converged hden (copied into their folder), then bisect further to within TOL_TEST.

if 'run' in stages:
    print('\n======== filter-test cells ========')
    exec(compile(main_source(CELL_LIST=str(TEST_DIR / 'filter_cells.txt'), MODELS_SUBDIR='filter_tests/models',
                             TEST=False), str(MAIN), 'exec'), {'__file__': str(MAIN)})

if 'tol' in stages:
    print(f'\n======== TOL-test cells (TOL = {TOL_TEST}) ========')
    (TEST_DIR / 'tol_models').mkdir(parents=True, exist_ok=True)
    shutil.copy(MAIN_DIR / 'hden_converged.json', TEST_DIR / 'tol_models' / 'hden_converged.json')
    exec(compile(main_source(CELL_LIST=str(TEST_DIR / 'tol_cells.txt'), MODELS_SUBDIR='filter_tests/tol_models',
                             TOL=TOL_TEST, TEST=False), str(MAIN), 'exec'), {'__file__': str(MAIN)})


###########################################################
#                        ANALYSE                          #
###########################################################
# The cells were picked with probability proportional to Q(H), so the plain mean of their f_esc estimates the
# Q(H)-weighted mean f_esc of the whole group. The change of the halo's escaping LyC (as a fraction of all stellar
# Q(H)) if the group were modelled instead of assumed is then  Q share of group * (mean Cloudy f_esc - assumed f_esc).

if 'analyse' in stages:
    cells = pd.read_csv(TEST_DIR / 'cells.csv')
    res = pd.concat([pd.read_csv(p) for p in [TEST_DIR / 'models' / 'results.csv', TEST_DIR / 'tol_models' / 'results.csv']
                     if p.exists()])
    main_res = pd.read_csv(MAIN_DIR / 'results.csv').set_index('cell')
    cells = cells.merge(res[['cell', 'f_esc total', 'converged', 'N_H Cloudy / SPHINX']].drop_duplicates('cell'), on='cell', how='left')

    print('\nPer cell:')
    print(cells.to_string(index=False, float_format=lambda x: f'{x:.3g}'))

    print(f'\nPer group (change of the halo f_esc = escaping / all stellar LyC, if modelled with Cloudy):')
    for k in ['A', 'C', 'D', 'TOL']:
        s = cells[cells['group'] == k]
        if k in groups:
            s = s[groups[k][s['cell']]]          # cells picked under an older definition of the group are left out
        f = s['f_esc total'].values
        if not np.isfinite(f).any():
            print(f'  {k:4s} no results')
            continue
        if k == 'TOL':
            df = f - main_res.loc[s['cell'], 'f_esc total'].values          # TOL_TEST vs the main run's TOL
            print(f'  TOL  Q share {Q_share[k]:.3f}: f_esc change with TOL {TOL_TEST} instead of {g["TOL"]}: mean '
                  f'{np.nanmean(df):+.4f} (cells {np.nanmin(df):+.4f} to {np.nanmax(df):+.4f}) '
                  f'-> halo f_esc change {Q_share[k] * np.nanmean(df):+.4f}')
            continue
        err = np.nanstd(f) / np.sqrt(np.isfinite(f).sum())
        line = f'  {k:4s} Q share {Q_share[k]:.3f}: Cloudy f_esc mean {np.nanmean(f):.3f} +- {err:.3f} (cells {np.nanmin(f):.3g}-{np.nanmax(f):.3g})'
        if k in ASSUMED:
            line += f', assumed {ASSUMED[k]:g} -> halo f_esc change {Q_share[k] * (np.nanmean(f) - ASSUMED[k]):+.4f} +- {Q_share[k] * err:.4f}'
        else:
            line += f' -> halo f_esc {Q_share[k] * np.nanmean(f):.4f} if modelled (vs 0 if left out, {Q_share[k]:.4f} if f_esc = 1)'
        print(line)

    # A_nc: all of its cells that have a saved result from the main run (not a sample), so its Q(H)-weighted f_esc directly
    saved = [c for c in np.where(A_nc)[0] if (MAIN_DIR / f'cell_{c:07d}' / 'cell_result.npz').exists()]
    r = [np.load(MAIN_DIR / f'cell_{c:07d}' / 'cell_result.npz') for c in saved]
    f = sum(float(x['Q_total']) for x in r) / sum(float(x['Q_incident']) for x in r)
    share = Q_cell[A_nc].sum() / Q_cell.sum()
    print(f'  A_nc Q share {share:.3f}: Cloudy f_esc (Q-weighted, {len(saved)} of {A_nc.sum()} cells, '
          f'{Q_cell[saved].sum() / Q_cell[A_nc].sum():.0%} of their Q) {f:.3f}, assumed 1 -> halo f_esc change {share * (f - 1):+.4f}')
