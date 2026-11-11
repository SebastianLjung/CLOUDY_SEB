import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

plt.rcParams['figure.figsize'] = (7, 5)
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams.update({'font.size': 12})
plt.rcParams['lines.linewidth'] = 2


###########################################################
#                KEY PARAMETERS TO CHANGE                 #
###########################################################

TEST = False                # True: only run the first unresolved cell
USE_SAVED_HDEN = True       # start each cell from the hden saved by earlier runs (hden_converged.json), if any
MAX_ITER = 20               # max number of Cloudy runs per cell to match the number of H atoms
TOL = 0.2                   # converged when N_H(model)/N_H(cell) is within 1 +- TOL
hden_guess_factor = 0.005   # first guess (no saved hden): hden = factor * SPHINX nH

z_snap = 5.0                # redshift of the snapshot (for the CMB)
Z_ABN = 0.0134              # solar metal mass fraction of solar_GASS10.abn (the abundances the decks start from)
RADIUS_LOG_PC = -4          # inner radius of the Cloudy sphere [log pc]
LYC_BAND = (204.79, 911.79)   # LyC band we integrate over [Ang]

CLOUDY_ROOT = Path('/Users/sebas/OneDrive/Documents/Geneve_Documents/HPC/CLOUDY_TEST/c25.00')
PROJECT_ROOT = CLOUDY_ROOT / 'Basu_tutorial'
CLOUDY_EXE = CLOUDY_ROOT / 'source' / 'cloudy.exe'
RUNS_DIR = PROJECT_ROOT / 'runs'
STAR_DIR = RUNS_DIR / 'halo_HIIs'                                   # one sub-folder per cell model
GAS_FILE = PROJECT_ROOT / 'data' / 'galaxy_oct' / 'gas_cells.npz'   # or a whole galaxy's gas_cells.npz
STARS_FILE = PROJECT_ROOT / 'data' / 'galaxy_oct' / 'stars.npz'     # or a whole galaxy's stars.npz
BPASS_FILE = CLOUDY_ROOT / 'data' / 'bpass_sphinx.ascii'            # Cloudy's own BPASS grid
HDEN_FILE = STAR_DIR / 'hden_converged.json'

# units in cgs
pc2cm = 3.086e18
h_cgs = 6.62607e-27         # [erg s]
clight = 2.99792458e10      # [cm/s]
lambda_912 = 911.8       # 1 Ryd [Ang]

BPASS_AGE_MIN = 1.0e6       # [yr]
BPASS_LOGZ_MIN, BPASS_LOGZ_MAX = -5.00, -1.40
R_eq_cube = (3 / (4 * np.pi))**(1 / 3)   # a sphere with the volume of a cube of side dx has R = 0.620 dx
Y_He = 10**-1.103                        # He/H, as in the decks


###########################################################
#                        FUNCTIONS                        #
###########################################################

def load_bpass(path):
    # Cloudy's ascii BPASS grid -> ages [yr], Z [mass fraction], wave [Ang], L_lambda [erg/s/Ang per Msun] as [iZ, iage, iwave]
    # The grid has no age-0 model, so a zero-flux age 0 is added at the front (indices then match the RAMSES all_seds.dat)

    lines = [l for l in Path(path).read_text().split('\n') if not l.startswith('#')]
    n_models, n_wave = int(lines[5]), int(lines[6])
    v = np.array(' '.join(lines[11:]).split(), dtype=float)
    params = v[:2 * n_models].reshape(n_models, 2)          # (age, log Z), age varies fastest
    wave = v[2 * n_models:2 * n_models + n_wave]
    flux = v[2 * n_models + n_wave:2 * n_models + n_wave * (n_models + 1)]
    ages, Zs = np.unique(params[:, 0]), 10**np.unique(params[:, 1])
    sed = flux.reshape(Zs.size, ages.size, n_wave)
    sed = np.concatenate([np.zeros((Zs.size, 1, n_wave)), sed], axis=1)
    return np.concatenate([[0.0], ages]), Zs, wave, sed


def ionising_photons(L_lambda):
    # Number of H-ionising photons [1/s] of a spectrum L_lambda [erg/s/Ang] on the BPASS wavelength grid (last axis)

    ion = BP_WAVE <= lambda_912
    return np.trapezoid(L_lambda[..., ion] * BP_WAVE[ion] * 1e-8 / (h_cgs * clight), BP_WAVE[ion], axis=-1)


def cell_sed(stars_idx):
    # Summed BPASS spectrum L_lambda [erg/s/Ang] of the given star particles

    logZ = np.clip(np.log10(star_Z[stars_idx]), BPASS_LOGZ_MIN, BPASS_LOGZ_MAX)
    logt = np.clip(np.log10(np.maximum(star_age[stars_idx], BPASS_AGE_MIN)), np.log10(BP_AGES[1]), np.log10(BP_AGES[-1]))
    return np.sum(10**sed_interp(np.c_[logZ, logt]) * star_minit[stars_idx, None], axis=0)


def alpha_B(T):
    # Case-B H recombination coefficient [cm^3/s] (Hui & Gnedin 1997)

    lam = 315614.0 / T
    return 2.753e-14 * lam**1.5 / (1.0 + (lam / 2.74)**0.407)**2.242


def stromgren_radius(Q, T, ne, n_p):
    # [cm]

    with np.errstate(divide='ignore', invalid='ignore'):
        return (3.0 * Q / (4.0 * np.pi * alpha_B(T) * ne * n_p))**(1.0 / 3.0)


def write_sed_file(E, nuFnu, out_path, comment):
    # Two-column `table SED` file: energy [Ryd], nu*F_nu; the flux type is named on the first data line

    order = np.argsort(E)
    E, nuFnu = E[order], nuFnu[order]
    keep = nuFnu > 0
    E, nuFnu = E[keep], nuFnu[keep]
    with open(out_path, 'w') as f:
        f.write(f'# {comment}\n')
        f.write(f'{E[0]:.6e}  {nuFnu[0]:.6e}  nuFnu\n')
        for e, i in zip(E[1:], nuFnu[1:]):
            f.write(f'{e:.6e}  {i:.6e}\n')


def make_deck(hden_log, metals_log, logQH, rout_log):
    # Cloudy input for one cell: the cell's summed stellar SED (cell.sed) at the centre of a sphere out to R_out

    return f'''print on
abundances "solar_GASS10.abn" no grains
element abundance helium -1.103
element limit off -8.52703961178103
hden {hden_log:.6f} log
constant pressure
set pres ioniz 5000
#set temperature floor 20 linear
turbulence 1km/s
sphere
radius {RADIUS_LOG_PC:g} {rout_log:.4f} log parsecs
set nend 10000
set didz 0.05 # Encourages to make shells thinner
metals {metals_log:.2f} log
table SED "cell.sed"
Q(H) {logQH:.2f}
CMB z={z_snap:.1f}
iterate to convergence
no molecules
no H2 molecule
#no induced processes
stop temperature off
#stop efrac 0.01 linear
database H-like levels small
database He-like levels small
database stout levels 30 15
database chianti levels 30 15
#element sodium off
element aluminium off
element phosphorus off
element chlorine off
element argon off
element potassium off
element calcium off
element titanium off
element chromium off
element manganese off
element cobalt off
element nickel off
element fluorine off
element copper off
element zinc off
save overview last ".ovr"
save last ionization means file = ".ionavr"
save continuum "continuum_00000.out" last units Angstrom no hash
save continuum "continuum.out" last absolute
save incident continuum "continuum_stellar_incident.out" last absolute
'''


def cell_dir(c):
    return STAR_DIR / f'cell_{c:05d}'


def run_cloudy(model_dir, deck):
    # Write model_dir/input.in and run `cloudy.exe input.in` there; returns True if Cloudy exited OK

    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'input.in').write_text(deck)
    env = dict(os.environ, CLOUDY_DATA_PATH='.:+')   # `table SED` files are only looked for in the data path, so add the run folder
    subprocess.run([str(CLOUDY_EXE), 'input.in'], cwd=model_dir, capture_output=True, text=True, env=env)
    out_path = model_dir / 'input.out'
    ok = out_path.exists() and 'Cloudy exited OK' in out_path.read_text()
    print(f"[{'OK' if ok else 'FAILED'}] {model_dir.relative_to(RUNS_DIR)}")
    if not ok and out_path.exists():
        print(out_path.read_text()[-1500:])
    return ok


def run_cloudy_many(cells, decks):
    # Run one Cloudy model per cell in parallel (one per core); returns {cell: True/False (exited OK)}

    n = min(len(cells), os.cpu_count() or 1)
    print(f'[check] running {len(cells)} CLOUDY models on {n} threads ...')
    with ThreadPoolExecutor(n) as ex:
        oks = list(ex.map(lambda c: run_cloudy(cell_dir(c), decks[c]), cells))
    print(f'[check] CLOUDY finished: {sum(oks)} OK, {len(oks) - sum(oks)} failed')
    return dict(zip(cells, oks))


def read_overview(path):
    # {column: array} from `save overview` (depth [cm], Te [K], hden [cm^-3], HI and HII fractions, ...)

    with open(path) as f:
        cols = f.readline().lstrip('#').rstrip('\n').split('\t')
    d = np.loadtxt(path, comments='#', ndmin=2)
    return {col: d[:, i] for i, col in enumerate(cols)}


def read_continuum(path):
    # {column: array} from `save continuum units Angstrom` (nu L_nu [erg/s])

    cols = ['wave', 'incident', 'trans', 'DiffOut', 'net trans', 'reflc', 'total']
    d = np.loadtxt(path, comments='#', usecols=range(len(cols)), ndmin=2)
    return dict(zip(cols, d.T))


def model_shells(ovr, rout_cm):
    # Zone centres r, thicknesses dr and shell volumes [cm, cm, cm^3]; the shells run from R_in to R_out

    r_in = 10**RADIUS_LOG_PC * pc2cm
    r = r_in + ovr['depth']
    edges = np.concatenate(([r_in], 0.5 * (r[1:] + r[:-1]), [rout_cm]))
    return r, np.diff(edges), 4 * np.pi / 3 * np.diff(edges**3)


def model_atoms(ovr, rout_cm):
    # Total number of H atoms and of ionised H atoms in the model sphere

    r, dr, vol = model_shells(ovr, rout_cm)
    return np.sum(ovr['hden'] * vol), np.sum(ovr['hden'] * ovr['HII'] * vol)


def model_summary(ovr, rout_cm, out_path):
    # A few diagnostics of one model, for printing

    r, dr, vol = model_shells(ovr, rout_cm)
    nH, T = ovr['hden'], ovr['Te']
    ionised = ovr['HII'] > 0.5
    neutral = ~ionised
    stops = re.findall(r'Calculation stopped because (.*?)\.', Path(out_path).read_text())
    return {
        'N_HI': np.sum(nH * ovr['HI'] * dr),                                       # radial column, centre to R_out [cm^-2]
        'N_HII': np.sum(nH * ovr['HII'] * dr),
        'R_IF': r[np.argmax(neutral)] / rout_cm if neutral.any() else np.nan,     # ionisation front [R_out], nan = none
        'T_ion': np.sum(T * nH * vol * ionised) / max(np.sum(nH * vol * ionised), 1e-300),
        'T_neu': np.sum(T * nH * vol * neutral) / np.sum(nH * vol * neutral) if neutral.any() else np.nan,
        'nH_range': f'{nH.min():.3g}-{nH.max():.3g}',
        'tau_912': ovr['Tau912'][-1] if 'Tau912' in ovr else np.nan,
        'zones': len(r),
        'stop': stops[-1] if stops else '?',
    }


def band_integral(wave, nuLnu, band=LYC_BAND):
    # L [erg/s] and N [photons/s] of a nu*L_nu spectrum (wave in Ang), integrated over band [Ang]

    order = np.argsort(wave)
    wave, nuLnu = wave[order], nuLnu[order]
    inside = (wave > band[0]) & (wave < band[1])
    w = np.concatenate(([band[0]], wave[inside], [band[1]]))
    f = np.interp(w, wave, nuLnu)
    L = np.trapezoid(f / w, w)                            # L_lambda = nu L_nu / lambda
    N = np.trapezoid(f * 1e-8 / (h_cgs * clight), w)      # L_lambda / (h c / lambda)
    return L, N


###########################################################
#                   LOAD SPHINX GAS + STARS               #
###########################################################

gas, stars = dict(np.load(GAS_FILE)), dict(np.load(STARS_FILE))
n_cells = gas['nH'].size

star_age = stars['star_age'].astype(float)               # [yr]
star_Z = stars['star_met'].astype(float)                 # metal mass fraction
star_minit = stars['star_minit'].astype(float) * 1e11    # [Msun] (stored in units of 1e11 Msun)
star_dist, star_cell = cKDTree(gas['cell_pos']).query(stars['pos'])   # host cell = nearest cell centre

print(f'[check] {n_cells} cells, {star_age.size} stars')
print(f"[check] gas: nH {gas['nH'].min():.2e}-{gas['nH'].max():.2e} cm^-3, T {gas['TK'].min():.0f}-{gas['TK'].max():.0f} K, "
      f"Z {gas['metallicity'].min():.2e}-{gas['metallicity'].max():.2e}, dx {gas['cell_dx'].min()/pc2cm:.2f}-{gas['cell_dx'].max()/pc2cm:.2f} pc")
print(f'[check] stars: age {star_age.min()/1e6:.2f}-{star_age.max()/1e6:.1f} Myr, Z {star_Z.min():.2e}-{star_Z.max():.2e}, '
      f'M_init {star_minit.min():.2e}-{star_minit.max():.2e} Msun')
n_far = np.sum(star_dist > 0.5 * np.sqrt(3) * gas['cell_dx'][star_cell])   # a star inside its cell is <= sqrt(3)/2 dx from the centre
print(f'[check] {n_far} stars lie outside their nearest cell')
print(f"[check] 'metals' offsets of the gas cells: {np.log10(gas['metallicity'].min()/Z_ABN):+.2f} "
      f"to {np.log10(gas['metallicity'].max()/Z_ABN):+.2f} dex")


###########################################################
#               BPASS: Q(H) OF EVERY STAR                 #
###########################################################

BP_AGES, BP_Z, BP_WAVE, BP_SED = load_bpass(BPASS_FILE)
print(f'[check] BPASS grid: SED {BP_SED.shape} [Z, age, wave], ages {BP_AGES[1]:.1e}-{BP_AGES[-1]:.1e} yr')

q_tab = ionising_photons(BP_SED)    # H-ionising photons per second per Msun, for every (Z, age) of the grid
print(f'log q(H) per Msun at Z=0.02: 1 Myr {np.log10(q_tab[10, 1]):.2f}, 10 Myr {np.log10(q_tab[10, 11]):.2f}, '
      f'100 Myr {np.log10(q_tab[10, 21]):.2f}')

# Interpolate q and whole spectra in (log Z, log age)
q_interp = RegularGridInterpolator((np.log10(BP_Z), np.log10(BP_AGES[1:])), np.log10(q_tab[:, 1:]))
sed_interp = RegularGridInterpolator((np.log10(BP_Z), np.log10(BP_AGES[1:])), np.log10(np.maximum(BP_SED[:, 1:, :], 1e-30)))

logZ = np.clip(np.log10(star_Z), BPASS_LOGZ_MIN, BPASS_LOGZ_MAX)
logt = np.clip(np.log10(star_age), np.log10(BP_AGES[1]), np.log10(BP_AGES[-1]))
star_Q = 10**q_interp(np.c_[logZ, logt]) * star_minit    # [1/s]
print(f'[check] star log Q(H): {np.log10(star_Q.min()):.2f}-{np.log10(star_Q.max()):.2f}, total {star_Q.sum():.2e} s^-1')


###########################################################
#         SELECT CELLS WITH UNRESOLVED HII REGIONS        #
###########################################################

N_star = np.bincount(star_cell, minlength=n_cells)                     # stars per cell
Q_cell = np.bincount(star_cell, weights=star_Q, minlength=n_cells)     # summed Q(H) per cell [1/s]
R_S = stromgren_radius(Q_cell, gas['TK'], gas['ne'], gas['nH'] * gas['xHII'])
rout_cm = R_eq_cube * gas['cell_dx']                                   # Cloudy sphere with the cell's volume [cm]
N_target = gas['nH'] * gas['cell_dx']**3                               # H atoms in each SPHINX cell

collisional = (gas['TK'] > 2e4) & (gas['xHII'] > 0.9)
unresolved = (N_star > 0) & ~collisional & (R_S < rout_cm)

star_cells = np.where(N_star > 0)[0]
print('\nStromgren radius of every cell that hosts stars:')
print(pd.DataFrame({
    'cell': star_cells,
    'n_stars': N_star[star_cells],
    'log10 Q(H) [1/s]': np.log10(np.maximum(Q_cell[star_cells], 1)),
    'R_S [pc]': R_S[star_cells] / pc2cm,
    'R_S / dx': R_S[star_cells] / gas['cell_dx'][star_cells],
    'unresolved': unresolved[star_cells],
}).to_string(index=False, float_format=lambda x: f'{x:.3g}'))

todo_cells = np.where(unresolved)[0]
if TEST:
    todo_cells = todo_cells[:1]
    print(f'[TEST] only running cell {todo_cells}')
print(f'\n{unresolved.sum()} of {n_cells} cells are unresolved; each gets one Cloudy model ({N_star[todo_cells].sum()} stars)')


###########################################################
#          SUMMED STELLAR SED OF EACH CELL (cell.sed)     #
###########################################################

Q_deck = np.zeros(n_cells)       # Q(H) of each cell's summed SED [1/s]
for c in todo_cells:
    idx = np.where(star_cell == c)[0]
    L_lam = cell_sed(idx)
    Q_deck[c] = ionising_photons(L_lam)
    cell_dir(c).mkdir(parents=True, exist_ok=True)
    write_sed_file(lambda_912 / BP_WAVE, BP_WAVE * L_lam, cell_dir(c) / 'cell.sed', f'BPASS sum of {idx.size} stars in cell {c}')
    print(f'[check] cell {c}: {idx.size} stars, Q from summed SED / sum of star Q = {Q_deck[c] / Q_cell[c]:.3f}')


###########################################################
#      RUN CLOUDY, ADJUSTING hden UNTIL N_H IS CONSERVED  #
###########################################################
# We trust the number of H atoms and the volume of the SPHINX cell, not its density structure.
# So the Cloudy sphere has the cell's volume (R_out), and hden (density at the inner edge, which sets the
# constant pressure) is changed until the sphere holds the cell's number of H atoms.
# Once a too-low and a too-high hden are known, the next guess sits between them, nearer the end that
# matched N_H better (40/60 split).

hden_log = np.log10(hden_guess_factor * gas['nH'])
lo, hi = np.full(n_cells, np.nan), np.full(n_cells, np.nan)               # hden_log known to give too few / too many atoms
lo_ratio, hi_ratio = np.full(n_cells, np.nan), np.full(n_cells, np.nan)   # their N_H ratios
ratio = np.full(n_cells, np.nan)                                          # N_H(model)/N_H(cell) of the last run
run_hden = np.full(n_cells, np.nan)                                       # hden_log of the last run
failed = np.zeros(n_cells, dtype=bool)                                    # Cloudy crashed for this cell

saved_hden = {}
if USE_SAVED_HDEN and HDEN_FILE.exists():
    saved_hden = {int(k): v for k, v in json.loads(HDEN_FILE.read_text()).items()}
    for c in todo_cells:
        if c in saved_hden:
            hden_log[c] = saved_hden[c]['hden_log']
            print(f"[check] cell {c}: first guess hden {hden_log[c]:.5f} from {HDEN_FILE.name} "
                  f"(last N_H ratio {saved_hden[c]['N_H_ratio']:.3f}, converged={saved_hden[c]['converged']})")

todo = list(todo_cells)
for it in range(MAX_ITER):
    decks = {c: make_deck(hden_log[c], np.log10(gas['metallicity'][c] / Z_ABN), np.log10(Q_deck[c]),
                          np.log10(rout_cm[c] / pc2cm)) for c in todo}
    ok = run_cloudy_many(todo, decks)

    for c in todo:
        if not ok[c]:
            print(f'[check] iter {it}: cell {c}: Cloudy FAILED at hden {hden_log[c]:.5f}, dropped')
            failed[c] = True
            continue

        ovr = read_overview(cell_dir(c) / 'input.ovr')
        N_H_mod, N_HII_mod = model_atoms(ovr, rout_cm[c])
        ratio[c], run_hden[c] = N_H_mod / N_target[c], hden_log[c]
        s = model_summary(ovr, rout_cm[c], cell_dir(c) / 'input.out')
        r_end = 10**RADIUS_LOG_PC * pc2cm + ovr['depth'][-1]

        print(f'[check] iter {it}: cell {c}: hden {hden_log[c]:.5f} -> N_H(model)/N_H(cell) = {ratio[c]:.3f}, '
              f'last zone at {r_end / rout_cm[c]:.3f} R_out')
        print(f"        ionised fraction of H atoms: Cloudy {N_HII_mod / N_H_mod:.4f}, SPHINX {gas['xHII'][c]:.4f}")
        print(f"        N(HI) = {s['N_HI']:.3e} cm^-2, N(HII) = {s['N_HII']:.3e} cm^-2, "
              f"ionisation front at {s['R_IF']:.3f} R_out (nan = none), tau_912 = {s['tau_912']:.3g}")
        print(f"        T ionised = {s['T_ion']:.0f} K, T neutral = {s['T_neu']:.0f} K, n_H = {s['nH_range']} cm^-3, "
              f"{s['zones']} zones, stopped: {s['stop']}")
        if r_end / rout_cm[c] < 0.95:
            print(f'[check] cell {c}: model stopped before R_out, see the end of input.out')

        if ratio[c] < 1:
            lo[c], lo_ratio[c] = hden_log[c], ratio[c]
        else:
            hi[c], hi_ratio[c] = hden_log[c], ratio[c]

    todo = [c for c in todo if not failed[c] and abs(ratio[c] - 1) > TOL]
    for c in todo:
        if np.isnan(lo[c]) or np.isnan(hi[c]):          # not bracketed yet: step towards the target (at most 0.5 dex)
            hden_log[c] -= 0.5 * np.clip(np.log10(ratio[c]), -1, 1)
        elif hi_ratio[c] * lo_ratio[c] > 1:             # hi misses by more than lo: lean towards lo
            hden_log[c] = 0.6 * lo[c] + 0.4 * hi[c]
        else:                                           # lo misses by more than hi: lean towards hi
            hden_log[c] = 0.4 * lo[c] + 0.6 * hi[c]
    if not todo:
        break

print(f'[check] N_H converged in {it + 1} iterations; {len(todo)} cells not converged, {failed.sum()} failed')

# Save the hden of each cell's last run (cells from earlier runs that were not rerun are kept)
for c in todo_cells[~np.isnan(run_hden[todo_cells])]:
    saved_hden[int(c)] = {'hden_log': float(run_hden[c]), 'N_H_ratio': float(ratio[c]),
                          'converged': bool(abs(ratio[c] - 1) <= TOL)}
HDEN_FILE.write_text(json.dumps({str(k): v for k, v in sorted(saved_hden.items())}, indent=2))
print(f'[check] saved hden to {HDEN_FILE}')


###########################################################
#                 LyC ESCAPE OF EACH CELL                 #
###########################################################
# Cloudy continuum columns: incident (stars, unattenuated), trans (stars, after the gas),
# DiffOut (emitted by the gas, outwards), net trans = trans + DiffOut.
# Escape fractions are photon-number ratios to the incident stellar LyC:
#   stellar: N(trans)/N(incident),  nebular: N(DiffOut)/N(incident),  total: N(net trans)/N(incident)

results = []
for c in todo_cells:
    if failed[c]:
        continue
    ovr = read_overview(cell_dir(c) / 'input.ovr')
    N_H_mod, N_HII_mod = model_atoms(ovr, rout_cm[c])
    N_HII_sphinx = gas['nH'][c] * gas['xHII'][c] * gas['cell_dx'][c]**3

    con = read_continuum(cell_dir(c) / 'inputcontinuum_00000.out')
    L_inc, N_inc = band_integral(con['wave'], con['incident'])
    L_esc, N_esc = band_integral(con['wave'], con['net trans'])
    N_trans = band_integral(con['wave'], con['trans'])[1]
    N_diff = band_integral(con['wave'], con['DiffOut'])[1]
    N_ion = band_integral(con['wave'], con['incident'], (1.0, lambda_912))[1]   # should equal the deck's Q(H)
    print(f'[check] cell {c}: incident Q(H) / deck Q(H) = {N_ion / Q_deck[c]:.3f}')

    results.append({
        'cell': c,
        'n_stars': N_star[c],
        'log10 Q(H) [1/s]': np.log10(Q_deck[c]),
        'L_LyC stars [erg/s]': L_inc,
        'L_LyC escaping [erg/s]': L_esc,
        'f_esc stellar': N_trans / N_inc,
        'f_esc nebular': N_diff / N_inc,
        'f_esc total': N_esc / N_inc,
        'N_HII Cloudy / SPHINX': N_HII_mod / N_HII_sphinx,
        'x_HII Cloudy': N_HII_mod / N_H_mod,
        'x_HII SPHINX': gas['xHII'][c],
    })

results = pd.DataFrame(results)
print(f'\nLyC ({LYC_BAND[0]:g}-{LYC_BAND[1]:g} Ang) of the Cloudy cell models:')
print(results.to_string(index=False, float_format=lambda x: f'{x:.3g}'))
