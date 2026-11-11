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
from concurrent.futures import ThreadPoolExecutor
import subprocess
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.io import FortranFile
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
ONEZONE_DIR = RUNS_DIR / "galaxy_1zone"
ABN_DIR = CLOUDY_ROOT / "data" / "abundances"
STAR_DIR = RUNS_DIR / "galaxy_stars"

BPASS_Z_RANGE = (1e-5, 0.039) 
BPASS_AGE_MIN = 1.0e6 
BPASS_LOGZ_MIN, BPASS_LOGZ_MAX = -5.00, -1.40
PC = 3.086e18
R_IN_BASIC = 1e17
BPASS_DIR = PROJECT_ROOT / "data" / "bpass_v221_135_100"
LSUN = 3.8256e33               # erg/s, the value used when the RAMSES tables were made
H_PLANCK, C_LIGHT = 6.62607e-27, 2.99792458e10
RYD_ANG = 911.2671             # 1 Ryd in Angstrom

GAS_FILE = PROJECT_ROOT / "data" / "galaxy_oct" / "gas_cells.npz"     # or a whole galaxy's gas_cells.npz
STARS_FILE = PROJECT_ROOT / "data" / "galaxy_oct" / "stars.npz"       # or a whole galaxy's stars.npz
gas, stars = dict(np.load(GAS_FILE)), dict(np.load(STARS_FILE))


n_cells = gas["nH"].size
V_cell = gas["cell_dx"]**3                                            # cm^3
star_age = stars["star_age"].astype(float)                            # yr
star_Z = stars["star_met"].astype(float)                              # metal mass fraction
star_minit = stars["star_minit"].astype(float) * 1e11                # 1e11 Msun -> Msun
_, star_cell = cKDTree(gas["cell_pos"]).query(stars["pos"])           # host cell = nearest cell centre
z_snap = 5.0                                                          # redshift of the snapshot (for the CMB)

CLOUDY_LINES_BLOCK = '''H  1 1215.67A'''

ELEMENTS = [("H", "HYDROGEN", 1.008), ("He", "HELIUM", 4.0026), ("Li", "LITHIUM", 6.94), ("Be", "BERYLLIUM", 9.012), ("B", "BORON", 10.81), ("C", "CARBON", 12.011),("N", "NITROGEN", 14.007), ("O", "OXYGEN", 15.999)]
SYMBOL = {name: sym for sym, name, _ in ELEMENTS}
WEIGHT = {sym: w for sym, _, w in ELEMENTS}
LINES_SHOWN = [("Ly$\\alpha$", ["H  1 1215.67A"])]
LINE_NAMES = [name.replace("$", "").replace("\\", "") for name, _ in LINES_SHOWN]

def parse_cloudy_lines(path):
    '''(labels, depth [cm], emissivity [erg/s/cm^3] per zone) from a `save lines, emissivity` file.'''
    cols = open(path).readline().rstrip("\n").split("\t")
    labels = [c.strip() for c in cols[1:] if c.strip()]
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    return labels, data[:, 0], data[:, 1:1 + len(labels)]


def read_abn(path):
    '''n_X/n_H by element symbol from a CLOUDY .abn file (stops at the '***' line).'''
    ab = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("*"):
            break
        if not line.strip() or line.startswith("#"):
            continue
        name, value = line.split()[:2]
        ab[SYMBOL[name.upper()]] = float(value)
    return ab


def mass_fractions(ab):
    '''(X, Y, Z) mass fractions of a mixture given number abundances n_X/n_H.'''
    m = {el: WEIGHT[el] * n for el, n in ab.items()}
    tot = sum(m.values())
    return m["H"] / tot, m.get("He", 0.0) / tot, (tot - m["H"] - m.get("He", 0.0)) / tot

def load_bpass(bpass_dir=BPASS_DIR):
    '''RAMSES-format BPASS: (ages [yr], Z [mass fraction], wave [A], L_lambda [erg/s/A per Msun] as [iZ, iage, iwave]).'''
    ages = np.loadtxt(bpass_dir / "age_bins.dat", skiprows=1)
    Zs = np.loadtxt(bpass_dir / "metallicity_bins.dat", skiprows=1)
    f = FortranFile(bpass_dir / "all_seds.dat", "r")
    nwave = f.read_ints("i4")[0]
    wave = f.read_reals("f8")
    sed = np.array([[f.read_reals("f8") for _ in ages] for _ in Zs]) * LSUN
    assert wave.size == nwave
    return ages, Zs, wave, sed



def run_production_style(model_dir, deck_text):
    '''Write model_dir/input.in and run `cloudy.exe input.in` there, exactly as the cluster jobs do.
    (Unlike `-r input`, this form prefixes every save file with "input": input.lines, inputcontinuum_...)'''
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "input.in").write_text(deck_text)
    subprocess.run([str(CLOUDY_EXE), "input.in"], cwd=model_dir, capture_output=True, text=True)
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
    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(lambda job: run_production_style(*job), jobs))

def basic_deck(sed_line, geometry=f"radius {np.log10(R_IN_BASIC):g}\nsphere", intensity="Q(H) 49"):
    '''A minimal H II region deck; the SED, geometry and intensity lines can be swapped (§7, §8).'''
    return f'''{sed_line}
    {intensity}
    {geometry}
    hden 2
    abundances "solar_GASS10.abn" no grains
    no molecules
    save overview ".ovr" last
    save lines, emissivity, emergent, last, ".lines"
    {CLOUDY_LINES_BLOCK}
    end of lines
    save incident continuum "continuum_stellar_incident.out" last absolute
    '''


def bpass_Q(age_yr, Z, minit_msun):
    '''H-ionizing photon rate [1/s] of star particles of initial mass minit_msun.'''
    logZ = np.clip(np.log10(Z), BPASS_LOGZ_MIN, BPASS_LOGZ_MAX)
    logt = np.clip(np.log10(age_yr), np.log10(BP_AGES[1]), np.log10(BP_AGES[-1]))
    return 10**q_interp(np.c_[np.atleast_1d(logZ), np.atleast_1d(logt)]) * minit_msun


def build_1zone_input(nH, T, Z_gas, U, xe):
    '''1-zone deck of one gas cell, as in the production 1-zone grid.'''
    logZ_sed = np.log10(np.clip(Z_gas, *BPASS_Z_RANGE))
    return f'''print off
    abundances "solar_GASS10.abn" no grains
    element abundance helium -1.103
    element limit off -8.52703961178103
    hden {np.log10(nH):.4f} log
    constant temperature {np.log10(T):.4f}
    metals {np.log10(Z_gas / Z_ABN):.4f} log
    ionization parameter {np.log10(U):.4f}
    set eden fraction {xe:.6g}
    CMB z={z_snap:.1f}
    stop thickness 1 cm
    save lines, emissivity, emergent, last, ".lines"
    {CLOUDY_LINES_BLOCK}
    end of lines
    '''




def cell_deck(c):
    return build_1zone_input(gas["nH"][c], gas["TK"][c], gas["metallicity"][c], gas["U"][c], gas["ne"][c] / gas["nH"][c])


BP_AGES, BP_Z, BP_WAVE, BP_SED = load_bpass()
gass10 = read_abn(ABN_DIR / "primordial.abn")#"solar_GASS10.abn")


# H-ionizing photons per second per Msun, for every (Z, age) of the grid
ion = BP_WAVE <= RYD_ANG
q_tab = np.trapezoid(BP_SED[:, :, ion] * BP_WAVE[ion] * 1e-8 / (H_PLANCK * C_LIGHT), BP_WAVE[ion], axis=-1)
print(f"log q(H) per Msun at Z=0.02: 1 Myr {np.log10(q_tab[10, 1]):.2f}, "
      f"10 Myr {np.log10(q_tab[10, 11]):.2f}, 100 Myr {np.log10(q_tab[10, 21]):.2f}")

X, Y, Z_ABN = mass_fractions(dict(gass10, He=10**-1.103))

# Q(H) of a star particle: BPASS q per Msun at (age, Z*), interpolated in (log Z, log age), x initial mass
q_interp = RegularGridInterpolator((np.log10(BP_Z), np.log10(BP_AGES[1:])), np.log10(q_tab[:, 1:]))

print(f"{'cell':>4s} {'nH':>6s} {'T [K]':>6s} {'xHII':>6s} {'ne/nH':>6s} {'log U':>6s} {'Z':>7s} {'stars':>5s}")
for c in range(n_cells):
    print(f"{c:4d} {gas['nH'][c]:6.1f} {gas['TK'][c]:6.0f} {gas['xHII'][c]:6.3f} {gas['ne'][c] / gas['nH'][c]:6.3f} "
          f"{np.log10(gas['U'][c]):6.2f} {gas['metallicity'][c]:7.4f} {(star_cell == c).sum():5d}")
print(f"\nstar ages [Myr]: {np.round(np.sort(star_age) / 1e6, 1)}")

#print(cell_deck(0)[:400], "...\n")
run_cloudy_many((ONEZONE_DIR / f"cell_{c}", cell_deck(c)) for c in range(n_cells))


def onezone_luminosities(lines_path, volume):
    '''L [erg/s] of LINES_SHOWN for a 1-zone model: emissivity 4 pi j [erg/s/cm^3] x cell volume.'''
    labels, _, emis = parse_cloudy_lines(lines_path)
    j = dict(zip(labels, emis[-1]))
    return np.array([sum(j[k] for k in keys) for _, keys in LINES_SHOWN]) * volume


L_1zone = np.array([onezone_luminosities(ONEZONE_DIR / f"cell_{c}" / "input.lines", V_cell[c]) for c in range(n_cells)])
print(f"\n1-zone line luminosities [erg/s]\n{'cell':>4s} " + " ".join(f"{n:>12s}" for n in LINE_NAMES[1:7]))
for c in range(n_cells):
    print(f"{c:4d} " + " ".join(f"{L:12.3e}" for L in L_1zone[c, 1:7]))








def alpha_B(T):
    '''Case-B H recombination coefficient [cm^3/s] (Hui & Gnedin 1997).'''
    lam = 315614.0 / T
    return 2.753e-14 * lam**1.5 / (1.0 + (lam / 2.74)**0.407)**2.242


def stromgren_radius(Q, T, ne, n_p):
    '''[cm]'''
    with np.errstate(divide="ignore", invalid="ignore"):
        return (3.0 * Q / (4.0 * np.pi * alpha_B(T) * ne * n_p))**(1.0 / 3.0)



star_Q = bpass_Q(star_age, star_Z, star_minit)
N_star = np.bincount(star_cell, minlength=n_cells)
Q_cell = np.bincount(star_cell, weights=star_Q, minlength=n_cells)
R_S = stromgren_radius(Q_cell, gas["TK"], gas["ne"], gas["nH"] * gas["xHII"])
collisional = (gas["TK"] > 2e4) & (gas["xHII"] > 0.9)
unresolved = (N_star > 0) & ~collisional & (R_S < gas["cell_dx"])


print(f"{'cell':>4s} {'stars':>5s} {'log sum Q':>9s} {'R_S [pc]':>9s} {'R_S/dx':>7s}  unresolved")
for c in range(n_cells):
    print(f"{c:4d} {N_star[c]:5d} {np.log10(max(Q_cell[c], 1)):9.2f} {R_S[c] / PC:9.3f} "
          f"{R_S[c] / gas['cell_dx'][c]:7.3f}  {'yes' if unresolved[c] else 'no'}")
todo_stars = np.where(unresolved[star_cell])[0]
print(f"\n{unresolved.sum()} of {n_cells} cells are unresolved; their {todo_stars.size} stars each get a multi-zone model")



jobs = []
for s in todo_stars:
    c = star_cell[s]
    jobs.append((STAR_DIR / f"star_{s:02d}",
                 build_star_cloudy_input(np.log10(gas["nH"][c]), np.log10(gas["metallicity"][c] / Z_ABN),
                                         star_age[s], star_Z[s], np.log10(star_Q[s]), z_snap, BPASS_TABLE)))
R_S_star = stromgren_radius(star_Q[todo_stars], 1e4, gas["nH"][star_cell[todo_stars]], gas["nH"][star_cell[todo_stars]])
print(f"R_in / R_S(1e4 K, fully ionized) of these stars: at most {10**RADIUS_LOG_PC * PC / R_S_star.min():.1e}")
print(f"{len(jobs)} star models in parallel (a few minutes) ...")

run_cloudy_many(jobs)

L_star = {s: line_table(STAR_DIR / f"star_{s:02d}" / "input.lines", RADIUS_LOG_PC) for s in todo_stars}
print(f"\n{'star':>4s} {'cell':>4s} {'age [Myr]':>9s} {'log Q':>6s} " + " ".join(f"{n:>12s}" for n in LINE_NAMES[1:7]))
for s in todo_stars:
    print(f"{s:4d} {star_cell[s]:4d} {star_age[s] / 1e6:9.1f} {np.log10(star_Q[s]):6.2f} "
          + " ".join(f"{L:12.3e}" for L in L_star[s][1:7]))


