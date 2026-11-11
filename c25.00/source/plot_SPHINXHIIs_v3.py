import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import scienceplots
import os
import pyCloudy as pc
import pandas as pd
from IPython.display import display
import matplotlib.colors as colors
from scipy.spatial import cKDTree
import re
import json
from concurrent.futures import ThreadPoolExecutor
import subprocess
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
#plt.rcParams['text.usetex'] = False
#plt.style.use('science')
#plt.rcParams.update({
    #"font.family": "sans-serif",      # Helvetica is sans-serif
    #"font.sans-serif": ["Helvetica"],
#    "font":'Helvetica',
#    "font.size": 18,
#    "lines.linewidth": 2,
#    "figure.figsize": (7, 5)
    #"mathtext.fontset": "stixsans"    # Makes math look good with Helvetica
#})

#plt.rc('font', serif='Helvetica'
plt.rcParams['figure.figsize'] = (7, 5)
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams.update({'font.size': 12})
plt.rcParams['lines.linewidth'] = 2


PROJECT_ROOT = Path("/Users/sebas/OneDrive/Documents/Geneve_Documents/HPC/CLOUDY_TEST/c25.00/Basu_tutorial/")  #Path.cwd().parent
CLOUDY_ROOT =  Path("/Users/sebas/OneDrive/Documents/Geneve_Documents/HPC/CLOUDY_TEST/c25.00/") #PROJECT_ROOT.parent
CLOUDY_EXE   = CLOUDY_ROOT / "source" / "cloudy.exe"
DOCS_DIR     = CLOUDY_ROOT / "docs"
RUNS_DIR     = PROJECT_ROOT / "runs"
STAR_DIR = RUNS_DIR / "halo_HIIs"  #"galaxy_stars"

BPASS_AGE_MIN = 1.0e6 
BPASS_LOGZ_MIN, BPASS_LOGZ_MAX = -5.00, -1.40
PC = 3.086e18
H_PLANCK, C_LIGHT = 6.62607e-27, 2.99792458e10
RYD_ANG = 911.2671             # 1 Ryd in Angstrom

GAS_FILE = PROJECT_ROOT / "data" / "galaxy_oct" / "gas_cells.npz"     # or a whole galaxy's gas_cells.npz
STARS_FILE = PROJECT_ROOT / "data" / "galaxy_oct" / "stars.npz"       # or a whole galaxy's stars.npz
gas, stars = dict(np.load(GAS_FILE)), dict(np.load(STARS_FILE))
print(f"[check] loaded {GAS_FILE.name}: {gas['nH'].size} cells, fields {sorted(gas)}")
print(f"[check] loaded {STARS_FILE.name}: {stars['star_age'].size} stars, fields {sorted(stars)}")


n_cells = gas["nH"].size
star_age = stars["star_age"].astype(float)                            # yr
star_Z = stars["star_met"].astype(float)                              # metal mass fraction
star_minit = stars["star_minit"].astype(float) * 1e11                # 1e11 Msun -> Msun
star_dist, star_cell = cKDTree(gas["cell_pos"]).query(stars["pos"])   # host cell = nearest cell centre
z_snap = 5.0                                                          # redshift of the snapshot (for the CMB)
print(f"[check] gas: nH {gas['nH'].min():.2e}-{gas['nH'].max():.2e} cm^-3, T {gas['TK'].min():.0f}-{gas['TK'].max():.0f} K, "
      f"Z {gas['metallicity'].min():.2e}-{gas['metallicity'].max():.2e}, dx {gas['cell_dx'].min() / PC:.2f}-{gas['cell_dx'].max() / PC:.2f} pc")
print(f"[check] stars: age {star_age.min() / 1e6:.2f}-{star_age.max() / 1e6:.1f} Myr, Z {star_Z.min():.2e}-{star_Z.max():.2e}, "
      f"M_init {star_minit.min():.2e}-{star_minit.max():.2e} Msun (total {star_minit.sum():.2e})")
# a star should sit inside its host cell: distance to the centre <= sqrt(3)/2 dx
far = star_dist > 0.5 * np.sqrt(3) * gas["cell_dx"][star_cell]
print(f"[check] star -> host cell: max distance / dx = {(star_dist / gas['cell_dx'][star_cell]).max():.2f}; "
      f"{far.sum()} stars lie outside their nearest cell")

LYC_BAND = (204.0, 911.8)      # Angstrom: the LyC band we integrate over
RADIUS_LOG_PC = -4             # inner radius of the multi-zone star models: 10^-3 pc
BPASS_TABLE = "bpass_sphinx.ascii"   # compiled grid, linked into CLOUDY_ROOT/data by the notebook


def band_integral(wave, nuLnu, band=LYC_BAND):
    '''(L [erg/s], N [photons/s]) of a nu*L_nu spectrum (wave in A), integrated over band [A].'''
    order = np.argsort(wave)
    wave, nuLnu = wave[order], nuLnu[order]
    inside = (wave > band[0]) & (wave < band[1])
    w = np.concatenate(([band[0]], wave[inside], [band[1]]))
    f = np.interp(w, wave, nuLnu)
    L = np.trapezoid(f / w, w)                                   # L_lambda = nu L_nu / lambda
    N = np.trapezoid(f * 1e-8 / (H_PLANCK * C_LIGHT), w)         # L_lambda / (h c / lambda)
    return L, N


def read_continuum(path):
    '''{column: array} from `save continuum units Angstrom` (nu L_nu [erg/s] in the luminosity case).'''
    cols = ["wave", "incident", "trans", "DiffOut", "net trans", "reflc", "total"]
    d = np.loadtxt(path, comments="#", usecols=range(len(cols)), ndmin=2)
    return dict(zip(cols, d.T))


def read_overview(path):
    '''{column: array} from `save overview` (depth [cm], Te [K], hden [cm^-3], HII fraction, ...).'''
    with open(path) as f:
        cols = f.readline().lstrip("#").rstrip("\n").split("\t")
    d = np.loadtxt(path, comments="#", ndmin=2)
    return {c: d[:, i] for i, c in enumerate(cols)}

def model_atoms(ovr, rout_cm, radius_log=RADIUS_LOG_PC):
    '''(N_H, N_HII) of a spherical model: sum of n x shell volume, shells running from R_in to R_out.'''
    r_in = 10**radius_log * PC
    r = r_in + ovr["depth"]                                           # zone centres
    edges = np.concatenate(([r_in], 0.5 * (r[1:] + r[:-1]), [rout_cm]))
    vol = 4 * np.pi / 3 * np.diff(edges**3)
    return np.sum(ovr["hden"] * vol), np.sum(ovr["hden"] * ovr["HII"] * vol)


def model_summary(ovr, rout_cm, out_path, radius_log=RADIUS_LOG_PC):
    '''A few diagnostics of one spherical model, from its .ovr and .out files (for printing).'''
    r_in = 10**radius_log * PC
    r = r_in + ovr["depth"]
    edges = np.concatenate(([r_in], 0.5 * (r[1:] + r[:-1]), [rout_cm]))
    dr, vol = np.diff(edges), 4 * np.pi / 3 * np.diff(edges**3)
    nH, xHI, xHII, T = ovr["hden"], ovr["HI"], ovr["HII"], ovr["Te"]
    ionized = xHII > 0.5
    stops = re.findall(r"Calculation stopped because (.*?)\.", Path(out_path).read_text())
    return {
        "N(HI) [cm^-2]":   np.sum(nH * xHI * dr),                  # radial columns, centre to R_out
        "N(HII) [cm^-2]":  np.sum(nH * xHII * dr),
        "R_IF / R_out":    r[np.argmax(~ionized)] / rout_cm if (~ionized).any() else np.nan,   # NaN: ionized to R_out
        "T ionized [K]":   np.sum(T * nH * vol * ionized) / max(np.sum(nH * vol * ionized), 1e-300),
        "T neutral [K]":   np.sum(T * nH * vol * ~ionized) / np.sum(nH * vol * ~ionized) if (~ionized).any() else np.nan,
        "n_H min-max":     f"{nH.min():.3g}-{nH.max():.3g}",
        "tau_912 (out)":   ovr["Tau912"][-1] if "Tau912" in ovr else np.nan,
        "zones":           len(r),
        "stop":            stops[-1] if stops else "?",
    }


def load_bpass(grid_path=CLOUDY_ROOT / "data" / BPASS_TABLE):
    '''BPASS from CLOUDY's own ascii grid (the one `table star` reads):
    (ages [yr], Z [mass fraction], wave [A], L_lambda [erg/s/A per Msun] as [iZ, iage, iwave]).
    The grid has no age-0 model; a zero-flux age 0 is prepended so indices match the RAMSES all_seds.dat.'''
    lines = Path(grid_path).read_text().split("\n")
    lines = [l for l in lines if not l.startswith("#")]
    n_models, n_wave = int(lines[5]), int(lines[6])
    v = np.array(" ".join(lines[11:]).split(), dtype=float)
    params = v[:2 * n_models].reshape(n_models, 2)                          # (age, log Z), age varies fastest
    wave = v[2 * n_models:2 * n_models + n_wave]
    flux = v[2 * n_models + n_wave:2 * n_models + n_wave * (n_models + 1)]
    ages, Zs = np.unique(params[:, 0]), 10**np.unique(params[:, 1])
    sed = flux.reshape(Zs.size, ages.size, n_wave)
    sed = np.concatenate([np.zeros((Zs.size, 1, n_wave)), sed], axis=1)
    return np.concatenate([[0.0], ages]), Zs, wave, sed


def write_sed_file(E, nuFnu, out_path, comment):
    '''Two-column `table SED` file: energy [Ryd], nu*F_nu; flux type named on the first data line.'''
    order = np.argsort(E)
    E, nuFnu = E[order], nuFnu[order]
    mask = nuFnu > 0
    E, nuFnu = E[mask], nuFnu[mask]
    with open(out_path, "w") as f:
        f.write(f"# {comment}\n")
        f.write(f"{E[0]:.6e}  {nuFnu[0]:.6e}  nuFnu\n")
        for e, i in zip(E[1:], nuFnu[1:]):
            f.write(f"{e:.6e}  {i:.6e}\n")

def run_production_style(model_dir, deck_text):
    '''Write model_dir/input.in and run `cloudy.exe input.in` there, exactly as the cluster jobs do.
    (Unlike `-r input`, this form prefixes every save file with "input": input.lines, inputcontinuum_...)'''
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "input.in").write_text(deck_text)
    # `table SED` files are only searched for in the data path, so put the run directory first ("+" = default path)
    env = dict(os.environ, CLOUDY_DATA_PATH=".:+")
    subprocess.run([str(CLOUDY_EXE), "input.in"], cwd=model_dir, capture_output=True, text=True, env=env)
    out_path = model_dir / "input.out"
    ok = out_path.exists() and "Cloudy exited OK" in out_path.read_text()
    print(f"[{'OK' if ok else 'FAILED'}] {model_dir.relative_to(RUNS_DIR)}")
    if not ok and out_path.exists():
        print(out_path.read_text()[-1500:])
    return out_path


def run_cloudy_many(model_dirs_and_decks, max_workers=None):
    '''Run many (model_dir, deck) pairs in parallel, one CLOUDY per core; returns the list of .out paths.'''
    jobs = list(model_dirs_and_decks)
    n = max_workers or min(len(jobs), os.cpu_count() or 1)
    print(f"[check] running {len(jobs)} CLOUDY models on {n} threads ...")
    with ThreadPoolExecutor(n) as ex:
        outs = list(ex.map(lambda job: run_production_style(*job), jobs))
    n_ok = sum(p.exists() and "Cloudy exited OK" in p.read_text() for p in outs)
    print(f"[check] CLOUDY finished: {n_ok} OK, {len(outs) - n_ok} failed")
    return outs


def bpass_Q(age_yr, Z, minit_msun):
    '''H-ionizing photon rate [1/s] of star particles of initial mass minit_msun.'''
    logZ = np.clip(np.log10(Z), BPASS_LOGZ_MIN, BPASS_LOGZ_MAX)
    logt = np.clip(np.log10(age_yr), np.log10(BP_AGES[1]), np.log10(BP_AGES[-1]))
    return 10**q_interp(np.c_[np.atleast_1d(logZ), np.atleast_1d(logt)]) * minit_msun


def cell_sed(stars_idx):
    '''L_lambda [erg/s/A] summed over the given star particles.'''
    logZ = np.clip(np.log10(star_Z[stars_idx]), BPASS_LOGZ_MIN, BPASS_LOGZ_MAX)
    logt = np.clip(np.log10(np.maximum(star_age[stars_idx], BPASS_AGE_MIN)),
                   np.log10(BP_AGES[1]), np.log10(BP_AGES[-1]))
    return (10**sed_interp(np.c_[logZ, logt]) * star_minit[stars_idx, None]).sum(axis=0)


def _deck(hden_log, metals_log, sed_line, logQH, z, radius_log, rout_log):
    '''Multi-zone spherical deck of one HII region, as in the production runs (minus the line list).'''
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
radius {radius_log:g} {rout_log:.4f} log parsecs
set nend 10000
metals {metals_log:.2f} log
{sed_line}
Q(H) {logQH:.2f}
CMB z={z:.1f}
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
element sodium off
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


def build_star_cloudy_input(hden_log, metals_log, age_yr, Z_real, logQH, z, bpass_table, rout_log,
                            radius_log=RADIUS_LOG_PC):
    '''Deck for one star particle. Z_real is the absolute (mass-fraction) metallicity, not Zsun-scaled.'''
    logZ = np.clip(np.log10(Z_real), BPASS_LOGZ_MIN, BPASS_LOGZ_MAX)
    age_yr = max(age_yr, BPASS_AGE_MIN)
    sed_line = f'table star "{bpass_table}" {age_yr:.5f} {logZ:.5f}'
    return _deck(hden_log, metals_log, sed_line, logQH, z, radius_log, rout_log)


def build_cell_cloudy_input(hden_log, metals_log, sed_file, logQH, z, rout_log, radius_log=RADIUS_LOG_PC):
    return _deck(hden_log, metals_log, f'table SED "{sed_file}"', logQH, z, radius_log, rout_log)


BP_AGES, BP_Z, BP_WAVE, BP_SED = load_bpass()
print(f"[check] BPASS grid: SED {BP_SED.shape} [Z, age, wave], ages {BP_AGES[1]:.1e}-{BP_AGES[-1]:.1e} yr, "
      f"Z {BP_Z.min():.0e}-{BP_Z.max():.2f}, wave {BP_WAVE.min():.0f}-{BP_WAVE.max():.0e} A")


# H-ionizing photons per second per Msun, for every (Z, age) of the grid
ion = BP_WAVE <= RYD_ANG
q_tab = np.trapezoid(BP_SED[:, :, ion] * BP_WAVE[ion] * 1e-8 / (H_PLANCK * C_LIGHT), BP_WAVE[ion], axis=-1)
print(f"log q(H) per Msun at Z=0.02: 1 Myr {np.log10(q_tab[10, 1]):.2f}, "
      f"10 Myr {np.log10(q_tab[10, 11]):.2f}, 100 Myr {np.log10(q_tab[10, 21]):.2f}")

Z_ABN = 0.0134   # solar metal mass fraction of solar_GASS10.abn (the abundances the decks start from)
print(f"[check] Z_ABN {Z_ABN} -> 'metals' offsets of the gas cells: "
      f"{np.log10(gas['metallicity'].min() / Z_ABN):+.2f} to {np.log10(gas['metallicity'].max() / Z_ABN):+.2f} dex")

# Q(H) of a star particle: BPASS q per Msun at (age, Z*), interpolated in (log Z, log age), x initial mass
q_interp = RegularGridInterpolator((np.log10(BP_Z), np.log10(BP_AGES[1:])), np.log10(q_tab[:, 1:]))

print(f"\nstar ages [Myr]: {np.round(np.sort(star_age) / 1e6, 1)}")


def alpha_B(T):
    '''Case-B H recombination coefficient [cm^3/s] (Hui & Gnedin 1997).'''
    lam = 315614.0 / T
    return 2.753e-14 * lam**1.5 / (1.0 + (lam / 2.74)**0.407)**2.242


def stromgren_radius(Q, T, ne, n_p):
    '''[cm]'''
    with np.errstate(divide="ignore", invalid="ignore"):
        return (3.0 * Q / (4.0 * np.pi * alpha_B(T) * ne * n_p))**(1.0 / 3.0)


# log L_lambda on the grid, for interpolating whole spectra in (log Z, log age)
sed_interp = RegularGridInterpolator((np.log10(BP_Z), np.log10(BP_AGES[1:])),
                                     np.log10(np.maximum(BP_SED[:, 1:, :], 1e-30)))
star_Q = bpass_Q(star_age, star_Z, star_minit)
N_star = np.bincount(star_cell, minlength=n_cells)
Q_cell = np.bincount(star_cell, weights=star_Q, minlength=n_cells)
R_S = stromgren_radius(Q_cell, gas["TK"], gas["ne"], gas["nH"] * gas["xHII"])
R_eq_cube = (3 / (4 * np.pi))**(1 / 3)     # (4/3) pi (R_eq_cube dx)^3 = dx^3  ->  R_eq_cube = 0.620
N_target = gas["nH"] * gas["cell_dx"]**3

collisional = (gas["TK"] > 2e4) & (gas["xHII"] > 0.9)
unresolved = (N_star > 0) & ~collisional & (R_S < R_eq_cube*gas["cell_dx"])
print(f"[check] star Q(H): {np.isfinite(star_Q).sum()}/{star_Q.size} finite, "
      f"log Q {np.log10(star_Q.min()):.2f}-{np.log10(star_Q.max()):.2f}, total {star_Q.sum():.2e} s^-1")
print(f"[check] cells: {(N_star > 0).sum()} host stars, {(collisional & (N_star > 0)).sum()} of those collisionally ionized, "
      f"{((N_star > 0) & ~collisional & (R_S >= gas['cell_dx'])).sum()} resolved (R_S >= dx), {unresolved.sum()} unresolved")


star_cells = np.where(N_star > 0)[0]
stromgren_table = pd.DataFrame({
    "cell":            star_cells,
    "n_stars":         N_star[star_cells],
    "log10 Q(H) [1/s]": np.log10(np.maximum(Q_cell[star_cells], 1)),
    "R_S [pc]":        R_S[star_cells] / PC,
    "R_S / dx":        R_S[star_cells] / gas["cell_dx"][star_cells],
    "unresolved":      unresolved[star_cells],
})
print("\nStromgren radius of every cell that hosts stars:")
print(stromgren_table.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
todo_cells = np.where(unresolved)[0] #todo_stars = np.where(unresolved[star_cell])[0]

TEST = False #True                                   # set False for the full run
if TEST:
    todo_cells = todo_cells[:1]               # or pick one by hand, e.g. np.array([2])
    print(f"[TEST] only running cell {todo_cells}")

print(f"\n{unresolved.sum()} of {n_cells} cells are unresolved; each gets one multi-zone model "
      f"({N_star[todo_cells].sum()} stars in total)")

jobs, Q_deck = [], {}
for c in todo_cells:
    idx = np.where(star_cell == c)[0]# the stars in this cell

    #SEB: Try to implement particle and volume conservation for CLOUDY cell. 
        #- Set radius to be the radius that would make a spherical cloudy model's volume the same as a sphinx cube. 
        #- We trust the total number of particles in SPHINX cells, but not their density distribution. So set the intial density to be the density of the same number of particles as in SPHINX cell, distributed over CLOUDY sphere with an average density. Then set "constant pressure" but not constant density, such that CLOUDY is free to evolve the density distribution (but conserves the same number of total H atoms.)

    rout = R_eq_cube * gas["cell_dx"][c] / PC
    rout_log = np.log10(rout)      # half the cell width, log pc
    L_lam = cell_sed(idx)                                # summed BPASS spectrum [erg/s/A]
    Q_deck[c] = np.trapezoid(L_lam[ion] * BP_WAVE[ion] * 1e-8 / (H_PLANCK * C_LIGHT), BP_WAVE[ion])
    model_dir = STAR_DIR / f"cell_{c:05d}"
    model_dir.mkdir(parents=True, exist_ok=True)
    write_sed_file(RYD_ANG / BP_WAVE, BP_WAVE * L_lam, model_dir / "cell.sed",
                   f"BPASS sum of {idx.size} stars in cell {c}")
    jobs.append((model_dir, build_cell_cloudy_input(np.log10(gas["nH"][c]),
                                                    np.log10(gas["metallicity"][c] / Z_ABN),
                                                    "cell.sed", np.log10(Q_deck[c]), z_snap, rout_log)))
    # Q from the summed spectrum vs the sum of the per-star Q (different interpolation, should be close)
    print(f"[check] cell {c}: {idx.size} stars, Q from summed SED / sum of star Q = {Q_deck[c] / Q_cell[c]:.3f}")

R_S_cell = stromgren_radius(Q_cell[todo_cells], 1e4, gas["nH"][todo_cells], gas["nH"][todo_cells])
print(f"R_in / R_S(1e4 K, fully ionized) of these cells: at most {10**RADIUS_LOG_PC * PC / R_S_cell.min():.1e}")
print(f"{len(jobs)} cell models in parallel ...")
if jobs:
    print(f"[check] first deck ({jobs[0][0].name}):\n{jobs[0][1]}")

MAX_ITER, TOL = 20, 0.2
hden_log = {c: np.log10( 0.005*gas["nH"][c]) for c in todo_cells}      # SPHINX density as the starting guess
lo = {c: None for c in todo_cells}       # highest hden_log known to give too FEW atoms
hi = {c: None for c in todo_cells}       # lowest hden_log known to give too MANY atoms
lo_ratio = {c: None for c in todo_cells}
hi_ratio = {c: None for c in todo_cells}

# Warm start: reuse the hden of earlier converged runs as the first guess, when there is one
HDEN_FILE = STAR_DIR / "hden_converged.json"
USE_SAVED_HDEN = True                          # set False to always start from the guess above
saved_hden = {}
if USE_SAVED_HDEN and HDEN_FILE.exists():
    saved_hden = {int(k): v for k, v in json.loads(HDEN_FILE.read_text()).items()}
    for c in todo_cells:
        if c in saved_hden:
            hden_log[c] = saved_hden[c]["hden_log"]
            print(f"[check] cell {c}: first guess hden {hden_log[c]:.5f} from {HDEN_FILE.name} "
                  f"(last N_H ratio {saved_hden[c]['N_H_ratio']:.3f}, converged={saved_hden[c]['converged']})")
run_hden, final_ratio = {}, {}                 # hden actually run last for each cell, and its N_H ratio


todo = list(todo_cells)




for it in range(MAX_ITER):
    run_cloudy_many((STAR_DIR / f"cell_{c:05d}",
                     build_cell_cloudy_input(hden_log[c], np.log10(gas["metallicity"][c] / Z_ABN), "cell.sed",
                                             np.log10(Q_deck[c]), z_snap, np.log10(R_eq_cube * gas["cell_dx"][c] / PC))) for c in todo)
    ratio = {}
    for c in todo:
        ovr = read_overview(STAR_DIR / f"cell_{c:05d}" / "input.ovr")
        rout_cm = R_eq_cube * gas["cell_dx"][c]                     # outer radius given to Cloudy [cm]
        N_H_mod, N_HII_mod = model_atoms(ovr, rout_cm)
        ratio[c] = N_H_mod / N_target[c]
        run_hden[c], final_ratio[c] = hden_log[c], ratio[c]
        r_end = 10**RADIUS_LOG_PC * PC + ovr["depth"][-1]
        print(f"[check] iter {it}: cell {c}: hden {hden_log[c]:.5f} -> N_H(model) / N_H(cell) = {ratio[c]:.3f}, "
              f"last zone at {r_end / rout_cm:.3f} R_out")
        print(f"        ionized fraction of H atoms: Cloudy {N_HII_mod / N_H_mod:.4f}, SPHINX {gas['xHII'][c]:.4f}")
        s = model_summary(ovr, rout_cm, STAR_DIR / f"cell_{c:05d}" / "input.out")
        print(f"        N(HI) = {s['N(HI) [cm^-2]']:.3e} cm^-2, N(HII) = {s['N(HII) [cm^-2]']:.3e} cm^-2, "
              f"ionization front at {s['R_IF / R_out']:.3f} R_out (nan = none), tau_912 = {s['tau_912 (out)']:.3g}")
        print(f"        T ionized = {s['T ionized [K]']:.0f} K, T neutral = {s['T neutral [K]']:.0f} K, "
              f"n_H = {s['n_H min-max']} cm^-3, {s['zones']} zones, stopped: {s['stop']}")
        if r_end / rout_cm < 0.95:
            print(f"[check] cell {c}: model stopped before R_out, see the end of input.out")
    
        if ratio[c] < 1:
            lo[c] = hden_log[c]
            lo_ratio[c] = ratio[c]
        else:
            hi[c] = hden_log[c]
            hi_ratio[c] = ratio[c]

    todo = [c for c in todo if abs(ratio[c] - 1) > TOL]
    for c in todo:
        if lo[c] is not None and hi[c] is not None:      # bracketed: bisect
            if (hi_ratio[c]*lo_ratio[c] > 1):
                hden_log[c] = (0.6*lo[c] + 0.4*hi[c])
            else:
                hden_log[c] = (0.4*lo[c] + 0.6*hi[c])

            #hden_log[c] = 0.5 * (lo[c] + hi[c])
        else:                                            # not yet bracketed: jump at most 1 dex towards the target
            hden_log[c] -= 0.5*np.clip(np.log10(ratio[c]), -1, 1)
        #hden_log[c] -= 0.1*np.log10(ratio[c])/((it+1)/4)
    if not todo:
        break
print(f"[check] N_H converged in {it + 1} iterations; {len(todo)} cells not converged")

# Save the hden of the last run of every cell (cells from earlier runs that were not rerun are kept)
for c in run_hden:
    saved_hden[int(c)] = {"hden_log": float(run_hden[c]), "N_H_ratio": float(final_ratio[c]),
                          "converged": bool(abs(final_ratio[c] - 1) <= TOL)}
HDEN_FILE.write_text(json.dumps({str(k): v for k, v in sorted(saved_hden.items())}, indent=2))
print(f"[check] saved hden of {len(run_hden)} cells to {HDEN_FILE}")

#run_cloudy_many(jobs)


# LyC leaving each cell's nebula: incident (stellar, unattenuated), trans (stellar, attenuated),
# DiffOut (nebular emission going outward), net trans = trans + DiffOut
LYC_COLS = ["incident", "trans", "DiffOut", "net trans"]
LyC_cell, N_HII_ratio, xHII_model = {}, {}, {}
for c in todo_cells:
    ovr = read_overview(STAR_DIR / f"cell_{c:05d}" / "input.ovr")
    N_H_model, N_HII_model = model_atoms(ovr, R_eq_cube * gas["cell_dx"][c])
    N_HII_sphinx = gas["nH"][c] * gas["xHII"][c] * gas["cell_dx"][c]**3
    N_HII_ratio[c] = N_HII_model / N_HII_sphinx
    xHII_model[c] = N_HII_model / N_H_model           # fraction of the model's H atoms that are ionized
    print(f"[check] cell {c}: ionized H atoms, Cloudy = {N_HII_model:.3e}, SPHINX = {N_HII_sphinx:.3e}, "
          f"ratio = {N_HII_ratio[c]:.3f}")
    print(f"[check] cell {c}: ionized fraction of H atoms, Cloudy = {xHII_model[c]:.4f}, SPHINX = {gas['xHII'][c]:.4f}")

    con_path = STAR_DIR / f"cell_{c:05d}" / "inputcontinuum_00000.out"
    if not con_path.exists():
        print(f"[check] cell {c}: no continuum file ({con_path.name}), skipped")
        continue
    con = read_continuum(con_path)
    LyC_cell[c] = {k: band_integral(con["wave"], con[k]) for k in LYC_COLS}   # (L [erg/s], N [1/s])
    # the incident spectrum should hold exactly the Q(H) put in the deck
    N_ion = band_integral(con["wave"], con["incident"], (1.0, RYD_ANG))[1]
    print(f"[check] cell {c}: incident Q(H) / deck Q(H) = {N_ion / Q_deck[c]:.3f}")
todo_cells = np.array([c for c in todo_cells if c in LyC_cell], dtype=int)

# Each LyC_cell[c][k] is (L [erg/s], N [photons/s]) in LYC_BAND; [0] picks L, [1] picks N.
# Escape fractions are photon-number ratios to the stellar LyC that enters the gas ("incident"):
#   stellar: starlight that gets through the gas           N(trans)     / N(incident)
#   nebular: LyC emitted by the gas itself (recombination) N(DiffOut)   / N(incident)
#   total:   everything that leaves = stellar + nebular    N(net trans) / N(incident)
results = pd.DataFrame([{
    "cell":                    c,
    "n_stars":                 N_star[c],
    "log10 Q(H) [1/s]":        np.log10(Q_deck[c]),
    "L_LyC stars [erg/s]":     LyC_cell[c]["incident"][0],
    "L_LyC escaping [erg/s]":  LyC_cell[c]["net trans"][0],
    "f_esc stellar":           LyC_cell[c]["trans"][1] / LyC_cell[c]["incident"][1],
    "f_esc nebular":           LyC_cell[c]["DiffOut"][1] / LyC_cell[c]["incident"][1],
    "f_esc total":             LyC_cell[c]["net trans"][1] / LyC_cell[c]["incident"][1],
    "N_HII Cloudy / SPHINX":   N_HII_ratio[c],
    "x_HII Cloudy":            xHII_model[c],
    "x_HII SPHINX":            gas["xHII"][c],
} for c in todo_cells])
print(f"\nLyC ({LYC_BAND[0]:g}-{LYC_BAND[1]:g} A) of the multi-zone cell models:")
print(results.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


