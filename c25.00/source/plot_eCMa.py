import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import scienceplots
import os
import pyCloudy as pc
import pandas as pd
from IPython.display import display
import matplotlib.colors as colors
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

inputmodel = 'Sasha_eCMa_col5e15' #'Balmertest_age0_Tflex_col26'  #'Balmertest_age0_col18_onezone' #'Balmertest_age0_Tflex_col17_5'#'Balmertest_age0_col18_5' #'Balmertest_age9_R7e18'#'Balmertest_age0_col14'#'Balmertest_age7_R7e18'#'Balmertest_age0_col18'#'Balmertest_age7_R7e18'
full_model_name = inputmodel#'Balmertest_col16'
#full_model_name = 'grid000000000_gridT_script'
mod_name = full_model_name
N_HI_ABS = [17, 18]   # log N_HI [cm^-2] of extra neutral columns applied by hand to the net transmission (manualSED plot); [] for none
Mod = pc.CloudyModel(inputmodel)
Mod.print_stats()

if not os.path.exists(f'../cloudy_plots/{inputmodel}'):
    os.makedirs(f'../cloudy_plots/{inputmodel}')


from tqdm import tqdm
from scipy.interpolate import interp1d


h =  6.626e-34 * 10**(3+4) # [g cm^2 /s]
e = 4.8e-10 #1.602e-12 #[cgm units]
mH =1.67e-27 * 10**3
me = 9.1e-31 * 10**(3)
c = 2.997e10 # [cm/s] 
qp = 5.31e-4
A2S = 8.31
eV = 1.602e-19 * 10**(3+4)
gff = 1.3 # Based on book
gi = 0.5
kB_Ryd = 1.38065e-23*4.5874e17
kB = 1.38065e-23*10**7
Z=1


###########################################################
#                      SPECTRA PLOT                       #
###########################################################
x = Mod.get_cont_x(unit='Ang')
yinc = Mod.get_cont_y(cont ='incid',unit='esA') 
ydiffout = Mod.get_cont_y(cont ='diffout',unit='esA')
yntrans = Mod.get_cont_y(cont ='ntrans',unit='esA')
yreflec = Mod.get_cont_y(cont ='reflec',unit='esA')

idx_1100 = np.argmin(np.abs(x - 1100))
y_1100 = yntrans[idx_1100]
ynoneb = yntrans-ydiffout


plt.plot(x,yinc/y_1100 ,label=r'TLUSTY input', ls=':', c='darkgreen')
plt.plot(x,yntrans/y_1100,label=r'Total transmission', lw=3, c='darkblue')
plt.plot(x,ynoneb/y_1100,label=r'Stellar transmission', ls='--', c='orange')
plt.plot(x,ydiffout/y_1100,label=r'Nebular transmission', ls='--', c='purple')
#plt.plot(x,yreflec/y_1100,label=r'Reflected+nebular inwards')
#plt.plot(x,ydiffout/y_1100,label=r'Nebular continuum', ls='--')
#plt.plot(x, ytest/y_1100, label=r'test')
plt.scatter(0,0, label=f'')

#plt.axvline(x=600,ls=':',alpha=0.7,color='#7a2edb',lw=1.5)
plt.axvline(x=912,ls=':',alpha=0.7,color='#477af5',lw=1.5)
plt.axvline(x=1215,ls=':',alpha=1,color='#606060',lw=1.5)
plt.axvline(x=3650,ls=':',alpha=1,color='#606060',lw=1.5)
#plt.text(540,np.max(yinc/y_1100)*0.4,r'f$_{600}$',size=16,rotation=90,color='#7a2edb')
plt.text(855,np.max(yinc/y_1100)*0.7,r'Ly_C',size=16,rotation=90,color='#477af5')
plt.text(1160,np.max(yinc/y_1100)*0.7,r'Ly${\alpha}$',size=16,rotation=90,color='#606060')
plt.text(3650,np.max(yinc/y_1100)*0.7,r'H_C',size=16,rotation=90,color='#606060')
plt.xlim(500, 2000)
plt.ylim(10**(-5), 10)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'F$_{\lambda}$/F$_{1100}$',size=16)
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/SED_F110norm_'+mod_name+'.jpg')
#plt.show()
plt.clf()


###########################################################
#                    SPECTRA ZOOM IN ON LyA               #
###########################################################


plt.plot(x,yinc/y_1100 ,label=r'Incident/source')
plt.plot(x,yntrans/y_1100,label=r'Stellar+Nebular transmission')

plt.axvline(x=1215.67,ls=':',alpha=1,color='#606060',lw=1.5)
#plt.axvline(x=3650,ls=':',alpha=1,color='#606060',lw=1.5)
plt.text(1215.67,np.max(yinc/y_1100)*0.5,r'Ly${\alpha}$, 1215.67',size=14,rotation=90,color='#606060')
#plt.text(3650,np.max(yinc/y_1100)*0.7,r'H_C',size=16,rotation=90,color='#606060')
plt.xlim(1215.4, 1216)
plt.ylim(10**(-1), 10**3)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'F$_{\lambda}$/F$_{1100}$',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/SED_LyAzoom_'+mod_name+'.jpg')
#plt.show()
plt.clf()


###########################################################
#                    SPECTRA ZOOM IN ON H_A               #
###########################################################


plt.plot(x,yinc/y_1100 ,label=r'Incident/source')
plt.plot(x,ydiffout/y_1100,label=r'Nebular transmission', ls='--')

plt.plot(x,yntrans/y_1100,label=r'Stellar+Nebular transmission')
plt.scatter(3660, 0.04)

plt.axvline(x=3645,ls=':',alpha=1,color='#606060',lw=1.5)
plt.text(3645,np.max(yinc/y_1100)*0.5,r'H_C',size=16,rotation=90,color='#606060')
plt.xlim(3600, 4000)
plt.ylim(10**(-3), 10**0)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'F$_{\lambda}$/F$_{1100}$',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/SED_Balmerzoom_'+mod_name+'.jpg')
#plt.show()
plt.clf()


###########################################################
#                    Manual spectra                       #
###########################################################

df = pd.read_csv(inputmodel+'.cont', sep='\t')

r = np.array(Mod.radius)
#df_twophoton = pd.read_csv(inputmodel+'.twophoton', sep='\t')

display(df.head())
display(df)
xm = np.divide(911.8, df.iloc[:,0])


#print(df.iloc[np.where((800 < xm) & (xm < 1400))].to_string())
ym_source = df.loc[:,'incident'] * xm**0 * 1e-8/c / (4*np.pi*r[-1]**2) #np.divide( df.loc[:,'incident'] ,  xm)/ (4*np.pi*R.iloc[-1]**2) #10**(-16)
ym_net = df.loc[:,'net trans'] * xm**0 * 1e-8/c / (4*np.pi*r[-1]**2) #np.divide(df.loc[:,'net trans'], xm)/ (4*np.pi*R.iloc[-1]**2)
ym_nebular = df.loc[:,'DiffOut']  * xm**0 * 1e-8/c / (4*np.pi*r[-1]**2) #np.divide(df.loc[:,'DiffOut'], xm)/ (4*np.pi*R.iloc[-1]**2)
ym_stellar = df.loc[:,'trans'] * xm**0 * 1e-8/c / (4*np.pi*r[-1]**2) #np.divide(df.loc[:,'trans'], xm)/ (4*np.pi*R.iloc[-1]**2) #ym_net-ym_nebular

ym_nebular_nolines = df.loc[:,'DiffOut'].iloc[np.where(df.loc[:,'nLine'] == 0)]


#display(df_twophoton.head())
#ym_twophoton = np.divide(df_twophoton.iloc[:, 2], xm) #* df_twophoton.iloc[:, 1]







###########################################################
#                    IONIC STRUCTURE PLOT                 #
###########################################################
HI = Mod.get_ionic('H',0)
HII = Mod.get_ionic('H',1)
#HeI = Mod.get_ionic('He',0)
#OI = Mod.get_ionic('O',0)
#OII = Mod.get_ionic('O',1)
plt.plot(r,HI,label=r'H$_{\rm{I}}$')
plt.plot(r,HII,label=r'H$_{\rm{II}}$')
#plt.plot(r,HeI,label=r'He$_{\rm{I}}$')
#plt.plot(r,OI,label=r'O$_{\rm{I}}$')
#plt.plot(r,OII,label=r'O$_{\rm{II}}$')
plt.xscale('log')
plt.xlabel(r'r [cm]',size=16)
plt.ylabel(r'fraction',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/IONIC_STRUCTURE_'+mod_name+'.jpg')
#plt.show()
plt.clf()

plt.plot(Mod.radius, Mod.te, label = 'Te')
plt.legend(loc=3)
plt.xscale('log')
plt.xlabel(r'r [cm]',size=16)
plt.ylabel(r'Te [K]',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/Te-radius_'+mod_name+'.jpg')
#plt.show()
plt.clf()


plt.plot(Mod.radius, Mod.nH)
plt.legend(loc=3)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r'r [cm]',size=16)
plt.ylabel(r'Density [cm-3]',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/Density-radius_'+mod_name+'.jpg')
#plt.show()
plt.clf()


### Try to estimate/replicate free-bound emission analytically
print('')

def extract_gamma(lambda_grid, logT):

    columnames = [2.000, 2.100, 2.200, 2.300, 2.400, 2.500, 2.600, 2.700, 2.800, 2.900, 3.000, 3.100, 3.200, 3.300, 3.400, 3.500, 3.600, 3.700, 3.800, 3.900, 4.000, 4.100, 4.200, 4.300, 4.400, 4.500, 4.600, 4.700, 4.800, 4.900, 5.000]

    df_gamma = pd.read_csv('mnr10988_smtable3/t3_elec.ascii', sep='\s+', skiprows=5, header=None, engine='python')

    #df_gammaHe = pd.read_csv('mnras0372-1875_SMTable4/t4_elec.ascii', sep='\s+', skiprows=5, header=None, engine='python')

    full_headers = ['Node', 'Ryd'] + columnames

    df_gamma.columns = full_headers

    a2S = pd.DataFrame({
    "T": [5000, 10000, 15000, 20000],
    "a2S": [1.38e-13, 0.838e-13, 0.625e-13, 0.506e-13]
    })


    #df_gammaHe.columns = full_headers

    #display(df_gamma)

    # Extract value based on closest nu and T

    col_idx = np.abs(df_gamma.iloc[:, 2:].columns - logT).argmin()+2

    col_idx2S = np.abs(a2S['T'] - np.power(10, logT)).argmin()
    #col_idxHe = np.abs(df_gammaHe.iloc[:, 2:].columns - logT).argmin()+2
    #print('logT input ', logT, 'logT output ', full_headers[col_idx])
    # align with frequency grid
    
    Ryd_grid = np.divide(911.8, lambda_grid)

    df_gamma['lambda'] = np.divide(911.8, df_gamma.iloc[:,1])

    #df_gammaHe['lambda'] = np.divide(911.267050550915, df_gammaHe.iloc[:,1])

    #df_gnu_2S = pd.DataFrame({
    #    "lambda": [24313, 12157, 8104, 6078, 4863, 4052, 3473, 3039, 2701, 2431],
    #    "nu (Hz)": [1.23e14, 2.47e14, 3.70e14, 4.93e14, 6.17e14, 7.40e14, 8.64e14, 9.87e14, 11.10e14, 12.34e14],
    #    "g_nu": [3.03e-28, 9.79e-28, 1.83e-27, 2.77e-27, 3.77e-27, 4.80e-27, 5.79e-27, 6.77e-27, 7.74e-27, 8.59e-27]
    #})


    df_gamma_sorted = df_gamma.sort_values(by="lambda")



    print('len(df_gamma_sorted["lambda"].values) ', len(df_gamma_sorted["lambda"].values), 'len(df_gamma_sorted[col_idx].values) ',len(df_gamma_sorted.iloc[:, col_idx].values), 'shape df_gamma_sorted ', np.shape(df_gamma_sorted))
    #f_interp = interp1d(df_gnu_2S_sorted["nu"].values, df_gnu_2S_sorted["g-nu"].values, kind='cubic', bounds_error=False, fill_value=(0, df_gnu_2S_sorted["g-nu"].values[-1]))
    f_interp2 = interp1d(df_gamma_sorted["lambda"].values, df_gamma_sorted.iloc[:, col_idx].values, kind='linear', bounds_error=False, fill_value=(df_gamma_sorted.iloc[:, col_idx].values[0], 0))

    #gnu_2S_smooth = interp1d(df_gnu_2S_sorted["lambda"].values,df_gnu_2S_sorted["g_nu"].values, kind='cubic', bounds_error=False, fill_value=(0, df_gnu_2S_sorted["g_nu"].values[-1]))(lambda_grid) 
    df_gamma_smooth = f_interp2(lambda_grid) 
    gamma_smooth = df_gamma_smooth/(10**34 * np.power(10, logT*1.5)) #Ttest**1.5

    
    #h =  6.626e-34 * 10**(3+4) # [g cm^2 /s]
    #e = 1.6e-19 #[cgm units]
    #mH =1.67e-27 * 10**3
    #me = 9.1e-31 * 10**(3)
    #c = 3e10 # [cm/s] 
    #qp = 5.31e-4
    #A2S = 8.31
    #eV = 1.6e-19 * 10**(3+4)

    #nHtest = 10**2
    #nu_grid = np.divide(c, lambda_grid*10**(-8))
    #kB_Ryd = 1.38e-23*4.58e17
    #kB = 1.38e-23*10**7
    #gi=0.5
    
    lambda0 = h*c/(0.4298*eV)*10**8
    print('lambda0 ', lambda0)
    ion_cross = 5.475e-14 * np.power(lambda_grid/lambda0, 4.0185) * np.power(np.power(lambda_grid/lambda0 , -1) -1, 2) * np.power(1+ np.power(lambda_grid*32.88/(lambda0) , -0.5) , -2.963)
    #ion_cross[lambda_grid > 911] = ion_cross.iloc[test_ind2[0][0]]
    
    gamma_r = 4*np.pi * h/c**2 * np.power(lambda_grid*1e-8/c, -3) * gi * np.power(h**2/(2*np.pi*me*kB*Ttest) , 1.5) * ion_cross* c*np.power(lambda_grid*1e-8, -1)


    lambda_idx = []
    gamma_1 = []
    #lambda_idxHe = []
    gamma = []
    gamma_raw = []
    #gamma_smooth = []
    gamma2S = []
    #gammaHe = []
    deltaE = []
    deltaE_smooth = []
    gnu_2S = []
    print('Nodes ', np.where(df_gamma['Node'] > 0)[0])
    #print('First few values of lambda_grid ', lambda_grid.iloc[:5])
    for j, val in enumerate(lambda_grid):
        # Calculate absolute difference, find the index of the minimum
        #lambda_idx = (df_gamma['lambda'] - val).abs().idxmin()

        lambda_idx = df_gamma.loc[(df_gamma['lambda'] >= val) & (df_gamma['Node'] > 0), 'lambda'].idxmin()

        #print('len lambda_grid ', len(lambda_grid) , 'gamma_r indexmin ', (lambda_grid >= df_gamma['lambda'][lambda_idx]).argmin(), 'gamma_r indexmax ', (lambda_grid >= df_gamma['lambda'][lambda_idx]).argmax())
        gamma_raw.append(gamma_r.iloc[(lambda_grid <= df_gamma['lambda'][lambda_idx]).argmax()])  
        #print("lambda threshold ", df_gamma.loc[(df_gamma['lambda'] >= val) & (df_gamma['Node'] > 0), 'lambda'].iloc[-1], 'test lambda threshold ', df_gamma.iloc[lambda_idx, -1], 'lambda_idx ', lambda_idx, 'idx attempt', (df_gamma.iloc[lambda_idx, -1] < lambda_grid).idxmax())#, 'output ', df_gamma_smooth[df_gamma.iloc[lambda_idx, -1] < lambda_grid])
        #print('output ', df_gamma_smooth[(df_gamma.iloc[lambda_idx, -1] < lambda_grid).idxmax()])
        #gamma_1 = df_gamma_smooth[df_gamma.iloc[lambda_idx, -1] < lambda_grid]
        #lambda_idx = df_gamma.loc[(df_gamma['lambda'] >= val) & (df_gamma['Node'] > -1), 'lambda'].idxmin()
        #lambda_idx = (df_gamma.iloc[np.where(df_gamma['Node'] > 0)[0]] - val).abs().idxmin()
    
        #lambda_idx2S = (df_gnu_2S['lambda'] - val).abs().idxmin()
        #lambda_idxHe = (df_gammaHe['lambda'] - val).abs().idxmin()
        #nu_idx.append(idx)

        #print('nu_grid value ', val, 'nu_table value ', df_gamma['nu'][nu_idx], 'gamma value ', df_gamma.iloc[nu_idx, col_idx])
        #print('lambdazero for fit ', 911.267050550915/0.4298)
        
        
        #gamma_smooth = interp1d(df_gamma_sorted["lambda"],df_gamma_sorted.iloc[:, col_idx], kind='linear', bounds_error=False, fill_value=(0,0))(lambda_grid)
        #gamma = gamma_smooth
        #print('gamma current ', df_gamma.iloc[lambda_idx, col_idx], 'gamma test ', gamma_smooth[j])
        gamma.append(df_gamma.iloc[lambda_idx, col_idx]/(10**34 * np.power(10, logT*1.5)))
        #gamma_smooth.append(gamma_1)

        #gammaHe.append(df_gammaHe.iloc[lambda_idxHe, col_idxHe])
       
        #gnu_2S.append(df_gnu_2S['g_nu'][lambda_idx2S])

        if (val < 912):
            deltaE.append(1*(Ryd_grid.iloc[j]- df_gamma.iloc[lambda_idx,1]))
            #deltaE_smooth.append(1*(Ryd_grid.iloc[j]- df_gamma_smooth.iloc[lambda_idx,1]))
        else:
            #deltaE.append(0)
            deltaE.append(1*(Ryd_grid.iloc[j]- df_gamma.iloc[lambda_idx,1]))

        #g_nu = gnu_2S

    #print('gamma ', gamma)


    return np.array(gamma, dtype=float), deltaE, np.array(gnu_2S, dtype=float), col_idx, np.array(gamma_smooth, dtype=float), np.array(gamma_r, dtype=float)



df = pd.read_csv(inputmodel+'.cont', sep='\t')

xm = np.divide(911.8, df.iloc[:,0])

analytic_idx = np.where((1 < xm) & (xm < 20000)) # Large range required for gnu symmetry mirroring
print('First few values of lambda_grid ', xm.iloc[analytic_idx].iloc[:5])
lambda_grid =  xm.iloc[analytic_idx]
nu_grid = np.divide(c, lambda_grid*10**(-8))

test_ind = np.where((880 < lambda_grid)  & (lambda_grid < 3645))
test_ind2 = np.where((500 < lambda_grid)  & (lambda_grid < 911.8))
test_ind4 =  np.where((500 < lambda_grid)  & (lambda_grid < 3645))

FluxLH_comb = np.zeros_like(lambda_grid)
FluxLHeIII_comb = np.zeros_like(lambda_grid)
FluxLB_comb = np.zeros_like(lambda_grid)
FluxL2SHe_comb = np.zeros_like(lambda_grid) 
FluxL2S_comb = np.zeros_like(lambda_grid)
FluxLH_comb_test2 = np.zeros_like(lambda_grid)
FluxLH_arr = []
a2S = pd.DataFrame({
"T": [5000, 10000, 15000, 20000],
"a2S": [1.38e-13, 0.838e-13, 0.625e-13, 0.506e-13]
})


df_gnu_2S_alt = pd.DataFrame({
    "lambda": [350000, 24313, 12157, 8104, 6078, 4863, 4052, 3473, 3039, 2701, 2431],
    "nu": [0, 1.23e14, 2.47e14, 3.70e14, 4.93e14, 6.17e14, 7.40e14, 8.64e14, 9.87e14, 11.10e14, 12.34e14],
    "g": [0, 3.03e-28, 9.79e-28, 1.83e-27, 2.77e-27, 3.77e-27, 4.80e-27, 5.79e-27, 6.77e-27, 7.74e-27, 8.59e-27],
    "g-nu": [0, 3.73e-14, 2.42e-13, 6.77e-13, 1.36e-12, 2.33e-12, 3.55e-12, 5.01e-12, 6.69e-12, 8.59e-12, 1.06e-11]
})

df_gnu_2S_sorted = df_gnu_2S_alt.sort_values(by="nu")

#f_interp = interp1d(df_gnu_2S_sorted["nu"].values, df_gnu_2S_sorted["g-nu"].values, kind='cubic', bounds_error=False, fill_value=(0, df_gnu_2S_sorted["g-nu"].values[-1]))
f_interp = interp1d(df_gnu_2S_sorted["nu"].values, df_gnu_2S_sorted["g-nu"].values, kind='cubic', bounds_error=False, fill_value="extrapolate")

#gnu_2S_smooth = interp1d(df_gnu_2S_sorted["lambda"].values,df_gnu_2S_sorted["g_nu"].values, kind='cubic', bounds_error=False, fill_value=(0, df_gnu_2S_sorted["g_nu"].values[-1]))(lambda_grid) 
gnu_2S_smooth = f_interp(np.divide(3*10**18, lambda_grid))

y = np.divide(lambda_grid/1215, -1)

#gnu_2S_alt = h*y *202*(y*(1-y) * (1 - np.power(4*y*(1-y), 0.8))+ 0.88*np.power(y*(1-y), 1.53) * np.power(4*y*(1-y), 0.8))  

#print('lambda_grid ', lambda_grid)
#mask = ((0 < 2 * lambda_grid / 2431 - 1) & (lambda_grid < 2431))

#l_short = lambda_grid[mask]
#l_target = l_short / (2 * l_short / 2431 - 1)
#multiplier = 1 / (2 * l_short / 2431 - 1)

#peak_gnu = np.max(df_gnu_2S_sorted["g_nu"])#df_gnu_2S_sorted["g_nu"].values[0]
#gnu_2S_smooth[mask] = multiplier * f_interp(l_target)

lambda_12 = 2431 # test

for l in lambda_grid[lambda_grid < lambda_12]:
    if (l < 0.5*lambda_12):#0.500*lambda_12):
        gnu_2S_smooth[lambda_grid == l] = 0
    else:
        #print('gnusmooth A ', gnu_2S_smooth[lambda_grid == l], 'gnusmooth B ', np.divide(1 ,2*l/lambda_12 - 1)* gnu_2S_smooth[np.abs(lambda_grid - np.divide(l, 2*l/lambda_12 - 1)).argmin()])
        #gnu_2S_smooth[lambda_grid == l] = np.divide(1 ,2*l/2431 - 1)* gnu_2S_smooth[np.abs(lambda_grid - np.divide(l, 2*l/2431 - 1)).argmin()]
        #np.divide(1, 2*l/lambda_12 -1)
        gnu_2S_smooth[lambda_grid == l] = np.divide(1, 2*l/lambda_12 -1)* gnu_2S_smooth[np.abs(lambda_grid - np.divide(l, 2*l/lambda_12 - 1)).argmin()] #np.abs(np.divide(3*10**18 ,lambda_grid) - np.divide(2*lambda_grid/lambda_12 - 1, lambda_grid)).argmin()]#np.divide(l, 2*l/lambda_12 - 1)).argmin()]
        #print('peak gnu ', df_gnu_2S_sorted["g-nu"].values[0], 'next gnu ', gnu_2S_smooth[lambda_grid < lambda_12][-3], 'gnu(2nu12 - nu) ', gnu_2S_smooth[np.abs(lambda_grid - np.divide(l,np.max(2*l/lambda_12 - 1, 0))).argmin()], '2nu12 - nu l:', np.divide(l, 2*l/lambda_12 - 1), 'lambda ', l)
        


### Try to replicate gamma * nu

#h =  6.626e-34 * 10**(3+4) # [g cm^2 /s]
#e = 1.6e-19 #[cgm units]
#mH =1.67e-27 * 10**3
#me = 9.1e-31 * 10**(3)
#c = 3e10 # [cm/s] 
#qp = 5.31e-4
#A2S = 8.31
#eV = 1.6e-19 * 10**(3+4)

nHtest = 10**2
#nu_grid = np.divide(c, lambda_grid*10**(-8))
#kB_Ryd = 1.38e-23*4.58e17
#kB = 1.38e-23*10**7
Ttest = 1e4 # 1e4 #1*10**4 # Gas T, not BB T
gamma, deltaE, gnu_2S_alt, col_idx, gamma_smooth, gamma_raw = extract_gamma(lambda_grid,np.log10(Ttest))
gamma = gamma *c* np.power(lambda_grid*1e-8, -1)
gamma_smooth = gamma_smooth #*c* np.power(lambda_grid*1e-8, -1)

gamma_H = gamma_smooth # * np.exp(np.divide(deltaE,-1*kB_Ryd*Ttest)) 

y = np.power(lambda_grid/1215, -1)
y[(y < 0.01) | (0.99 < y)] =  0


i1 = np.argmax(lambda_grid < 1400)
i2 = np.argmax(lambda_grid < 2000)
i3 = np.argmax(lambda_grid < 4000)
i4 = np.argmax(lambda_grid < 850)


A2q = 8.225

gnu_2S = np.array(h/A2q *y *202*(y*(1-y) * (1 - np.power(4*y*(1-y), 0.8))+ 0.88*np.power(y*(1-y), 1.53) * np.power(4*y*(1-y), 0.8)))
yHe = np.power(lambda_grid/304, -1)
yHe[(yHe < 0.01) | (0.99 < yHe)] =  0

gnu_2SHe = np.array(h/A2q *yHe *202*(yHe*(1-yHe) * (1 - np.power(4*yHe*(1-yHe), 0.8))+ 0.88*np.power(yHe*(1-yHe), 1.53) * np.power(4*yHe*(1-yHe), 0.8)))


row_idx2S = np.abs(a2S['T'] - Ttest).argmin()
gamma_2S_alt = np.power(lambda_grid*1e-8/c, 1)*a2S.iloc[row_idx2S, 1]* np.divide(1, 1 + nHtest*qp/A2S)* gnu_2S_smooth #*3*10**(8)* np.power(lambda_grid, -1)*10**10
gamma_2S = a2S.iloc[row_idx2S, 1]* np.divide(1, 1 + nHtest*qp/A2S)* gnu_2S #*3*10**(8)* np.power(lambda_grid, -1)*10**10

expo_test = np.divide(h*c,kB*Ttest)*(1/911.8 - np.power(lambda_grid.iloc[test_ind2], -1))*10**8



gff = 1.33 #np.sqrt(3)/np.pi * (np.log(Ttest**1.5*c/(lambda_grid*1e-8)) + 17.7) 
#print('gff at 10000Ang', gff.iloc[np.argmin(np.abs(lambda_grid - 10000))])
gamma_Bremstrahlung = 2**5 * np.pi * e**6 /(3*me*c**3) *np.sqrt(2*np.pi/(3*me*kB*Ttest)) * gff * np.exp(-h*nu_grid/(kB*Ttest))#(4*np.pi)**0*5.44e-41*Ttest**(-0.5)*gff*np.exp(-h*nu_grid/(kB*Ttest))


lambda0 = h*c/(0.4298*eV)*10**8
print('lambda0 ', lambda0)

#nlevel=2
#n_boundaries = [911, 3645, 8200]
n_levels = np.zeros_like(lambda_grid)
gbf_levels = np.zeros_like(lambda_grid)
gbf = np.zeros_like(lambda_grid)
ynl= np.zeros_like(lambda_grid)
ynlHe= np.zeros_like(lambda_grid)
phi = np.zeros_like(lambda_grid)
Fnl = np.zeros_like(lambda_grid)
bf_cross = np.zeros_like(lambda_grid)
fb_cross_2 = np.zeros_like(lambda_grid)
weights = np.zeros_like(lambda_grid)
bf_cross_alt1 =np.zeros_like(lambda_grid)
bf_cross_alt2 =np.zeros_like(lambda_grid)
bf_cross_alt3 =np.zeros_like(lambda_grid)
bf_cross_alt4 =np.zeros_like(lambda_grid)
bf_cross_alt3_He =np.zeros_like(lambda_grid)
gbf2 = np.zeros_like(lambda_grid)
gbf4 = np.zeros_like(lambda_grid)
lambda_save = [[], [], [], [], []]

gbf[lambda_grid < 912] = 1

lookup = {(14579, 22781.5): [5.0, 1] , (8202, 14579): [4.0, 1] , (3645.06, 8202): [3.0, 1.22], (911.8, 3645.06): [2.0, 1.19], (0, 911.8): [1.0, 1.12]}
#lookup = {(0, 911.26): [1.0, 1.12],(911.26, 3645.06): [2.0, 1.19], (3645.06, 8202): [3.0, 1.22],(8202, 14579): [4.0, 1], (14579, 22781.5): [5.0, 1]}
u = 0


for (lower, upper), [ni, gbfi] in lookup.items():
    #if lambda_grid < boundary:
    #if (u > 0):
    #lambda_mid = np.where((lower < lambda_grid) & (lambda_grid < upper)) 
    #lambda_save[] = lambda_grid.iloc[lambda_below] 
    lambda_below = np.where((lambda_grid < upper))#np.where((lower < lambda_grid)) 
    lambda_belowHe = np.where((lambda_grid < 227.8*ni**2))
    n_levels[lambda_below] = ni 
    gbf_levels[lambda_below] = gbfi 
    gbf_levels[lambda_grid < 912] = 1

#plt.plot(lambda_grid , n_levels )
#ni = lambda_grid > 91

#for ni in n_levels:

    l=0
    Eion=np.power(911.25e-8/(h*c), -1) *np.power(ni, -2)
    ynl[lambda_below] = Eion*np.power( lambda_grid.iloc[lambda_below]*1e-8/(h*c), 1)
    tau = np.sqrt(Eion/(np.power( lambda_grid.iloc[lambda_below]*1e-8/(h*c), -1)-Eion))
    weights[lambda_below] = ni**2#2*ni**2
    #if (u>3):
    #    Fnl[lambda_below] =np.exp(-4*ni)
    #else:
    Fnl[lambda_below] = np.exp(-4*ni*tau*(np.pi/2 - np.arctan(tau)))*np.power(1- np.exp(-2*np.pi*ni*tau) ,-1)
#    phi = [ ((1+32/3*ynl[lambda_below]+208/9*ynl[lambda_below]**2+128/9*ynl[lambda_below]**3) +  (8*ynl[lambda_below] + 208/3 * ynl[lambda_below]**2 + 224/3 * ynl[lambda_below]**3) + (80/9 * ynl[lambda_below]**2 + 736/9*ynl[lambda_below]**3 + 256/3 * ynl[lambda_below]**4)) ,  (1+3*ynl[lambda_below]) + (3*ynl[lambda_below]+8*ynl[lambda_below]**2) , 1]
    phi[lambda_below] = [1 + 64*ynl[lambda_below]**1 + 1343*ynl[lambda_below]**2 + 12589*ynl[lambda_below]**3 + 55801*ynl[lambda_below]**4 + 124516*ynl[lambda_below]**5 + 138738*ynl[lambda_below]**6 + 70996*ynl[lambda_below]**7 + 13107*ynl[lambda_below]**8 , 1+38*ynl[lambda_below]**1+461.33*ynl[lambda_below]**2+2181.3*ynl[lambda_below]**3+4423.1*ynl[lambda_below]**4+3669.3*ynl[lambda_below]**5+1024*ynl[lambda_below]**6,((1+32/3*ynl[lambda_below]+208/9*ynl[lambda_below]**2+128/9*ynl[lambda_below]**3) +  (8*ynl[lambda_below] + 208/3 * ynl[lambda_below]**2 + 224/3 * ynl[lambda_below]**3) + (80/9 * ynl[lambda_below]**2 + 736/9*ynl[lambda_below]**3 + 256/3 * ynl[lambda_below]**4)) ,  (1+3*ynl[lambda_below]) + (3*ynl[lambda_below]+8*ynl[lambda_below]**2), 1][u]

    #phi = [1 + (2)*64*ynl[lambda_below]**1 + (2)*1283.4*ynl[lambda_below]**2 + (2)*6736.3*ynl[lambda_below]**3 + (2)*24180.3*ynl[lambda_below]**4 + (2)*47368.6*ynl[lambda_below]**5 + (2)*59872.7*ynl[lambda_below]**6 + (2)*30063.5*ynl[lambda_below]**7 + (2)*261.4*ynl[lambda_below]**8 , 1+(2)*38*ynl[lambda_below]**1+(2)*461.33*ynl[lambda_below]**2+(2)*2181.3*ynl[lambda_below]**3+(2)*4423.1*ynl[lambda_below]**4+(2)*3669.3*ynl[lambda_below]**5+(2)*1024*ynl[lambda_below]**6,((1+(2)*32/3*ynl[lambda_below]+(2)*208/9*ynl[lambda_below]**2+(2)*128/9*ynl[lambda_below]**3) +  ((2)*8*ynl[lambda_below] + (2)*208/3 * ynl[lambda_below]**2 + (2)*224/3 * ynl[lambda_below]**3) + ((2)*80/9 * ynl[lambda_below]**2 + (2)*736/9*ynl[lambda_below]**3 + (2)*256/3 * ynl[lambda_below]**4)) ,  (1+(2)*3*ynl[lambda_below]) + ((2)*3*ynl[lambda_below]+(2)*8*ynl[lambda_below]**2), 1]
    #1+ 38 + 461.33 + 2181.3 + 4423.1 + 3669.3 + 
    #ynl[lambda_below] = lambda_grid.iloc[lambda_below]/911

    gbf[lambda_below] = (1)*8*np.pi*np.sqrt(3)*ni**4/ni**2 *ynl[lambda_below]*phi[lambda_below]*Fnl[lambda_below]
    u1 = (np.power(lambda_grid.iloc[lambda_below]*1e-8/(h*c), -1)-Eion)*ni**2 
    gbf2[lambda_below] = 1+0.1728*(u1 - 1)*np.power((u1 + 1) , -2/3) + 0.0496 * (u1**2 + 4/3 * u1 + 1)*np.power(ni*(u1+1) , -4/3)
    gbf4[lambda_below] = 2**3*np.pi*np.sqrt(3)*ni**(-1)*lambda_grid.iloc[lambda_below]/912*np.exp(-4*ni*tau*(np.pi/2 - np.arctan((tau))))*np.power((1- np.exp(-2*np.pi*ni*tau)*(1 - (lambda_grid.iloc[lambda_below]/(ni**2*912)))) ,-1)*[5*(1 + 64*ynl[lambda_below]**1 + 1343*ynl[lambda_below]**2 + 12589*ynl[lambda_below]**3 + 55801*ynl[lambda_below]**4 + 124516*ynl[lambda_below]**5 + 138738*ynl[lambda_below]**6 + 70996*ynl[lambda_below]**7 + 13107*ynl[lambda_below]**8 ), +38*4*ynl[lambda_below]**1 + 346/3*4**2*ynl[lambda_below]**2 + 409/3*4**3*ynl[lambda_below]**3 + 622/9*4**4*ynl[lambda_below]**4 + 43/3*4**5*ynl[lambda_below]**5 + 1*4**6*ynl[lambda_below]**6, 3+ 14*4*ynl[lambda_below]**1 + 19*16* ynl[lambda_below]**2 + 8*64*ynl[lambda_below]**3 + 256*ynl[lambda_below]**4, 2+3*4*ynl[lambda_below]**1 + 16*ynl[lambda_below]**2, 1][u]#(1)*2**2 *np.pi*np.sqrt(3) *(1/137)**2 *me*c**2*(lambda_grid.iloc[lambda_below]*1e-8/(h*c*ni**2))*np.exp(-4*ni*tau*(np.pi/2 - np.arctan((tau))))*np.power((1- np.exp(-2*np.pi*ni*tau))*ni**2*(tau**2+1) ,-1)*[0, 0, 3+ 14*4*ynl[lambda_below]**1 + 19*16* ynl[lambda_below]**2 + 8*64*ynl[lambda_below]**3 + 256*ynl[lambda_below]**4, 2+3*4*ynl[lambda_below]**1 + 16*ynl[lambda_below]**2, 1][u]
#    gbf[lambda_below] = ynl[lambda_below]*phi[u]*Fnl[lambda_below]
    #bf_cross given in cm^2
    print('gbf variable', gbf[lambda_below][0], 'gbf2 approx ', gbf2[lambda_below][0])
    coeff = (ni**(1))**0 *2**4*h*e**2/(3*np.sqrt(3)*me*c)# (ni**4)
#    bf_cross[lambda_below] =  np.exp(-1/(kB * Ttest) *np.abs(h*c/(lambda_grid.iloc[lambda_below]*1e-8) -h*c/(912e-8*ni**2))) * (1/(1+ynl[lambda_below]))**0*coeff*np.power(Eion,2)*np.power(lambda_grid.iloc[lambda_below]*1e-8/(h*c), 3)*gbf[lambda_below] * np.power(ni, -1)
    bf_cross[lambda_below] +=  coeff*np.power(Eion,2)*np.power(lambda_grid.iloc[lambda_below]*1e-8/(h*c), 3)*gbf[lambda_below] * np.power(ni, -1)


    #bf_cross[lambda_grid < 912] = (2)*bf_cross[lambda_grid < 912]
    #1.7236293140868654e-28
    #bf_cross_alt1[lambda_below] = ni**(0)*2**6*0.00729735 * np.pi*5.29e-9**2 * ni/(3**1.5) * np.power(lambda_grid.iloc[lambda_below]*1e-8/(h*c) *Eion , 3)*(gbf2[lambda_below]) 
    
    bf_cross_alt2[lambda_below] += (ni**0)**0*2**6*(1/137)*np.pi*(5.29e-9)**2*ni/(3*np.sqrt(3))*np.power(ni**2*912/(lambda_grid.iloc[lambda_below]), -3) *gbf2[lambda_below] 
    
    #bf_cross_alt3[lambda_below] = (ni**1)**0*2**6*(1/137)*np.pi*(5.29e-9)**2*ni/(3*np.sqrt(3))*np.power(ni**2*912/(lambda_grid.iloc[lambda_below]), -3) *gbf3[lambda_below]#2**6*np.pi/np.sqrt(3)*ni*ni**2*ynl[lambda_below]**4*(1/137)**3*(5.29e-9)**2*gbf3[lambda_below]*me*c**2

    #mu3 = np.sqrt(ni**2*Eion*np.power( np.abs(h*c/(lambda_grid.iloc[lambda_below]*1e-8) - Eion), -1))
    #bf_cross_alt3[lambda_below] = 2**8*np.pi**2 /3 * ni**(0)*(1/137)**3 * (5.29e-9)**2 * np.power(mu3, 2)* np.exp(-4*mu3*np.arctan(ni/mu3)) *np.power((1- np.exp(-2*np.pi*mu3)) *(ni**2 + mu3**2)**2 ,-1)*[0, 0, 3+ 14*4*ynl[lambda_below]**1 + 19*16* ynl[lambda_below]**2 + 8*64*ynl[lambda_below]**3 + 256*ynl[lambda_below]**4, 2+3*4*ynl[lambda_below]**1 + 16*ynl[lambda_below]**2, 1][u]

    mu3 = np.sqrt((1/137)**2*c**2 *me/2*np.power( np.abs(h*c/(lambda_grid.iloc[lambda_below]*1e-8) - Eion), -1))
    #ynl[lambda_below] = 1/(1+ni**2/mu3**2)
#    bf_cross_alt3[lambda_below] = np.exp(-1/(kB * Ttest) *np.abs(h*c/(lambda_grid.iloc[lambda_below]*1e-8) -h*c/(912e-8*ni**2))) * me*c**2/ni**2 *np.abs(h*c/(lambda_grid.iloc[lambda_below]*1e-8) - Eion)/(h*c/(lambda_grid.iloc[lambda_below]*1e-8))**2 * 2**6 /3 *(1/137)**1 * h**2/(me**2*c**2) * np.power(mu3, 2)* np.exp(-4*mu3*np.arctan(ni/mu3)) *np.power((1- np.exp(-2*np.pi*mu3)) *(1 + ni**2/mu3**2)**2 ,-1)*[5*(1 + 64*ynl[lambda_below]**1 + 1343*ynl[lambda_below]**2 + 12589*ynl[lambda_below]**3 + 55801*ynl[lambda_below]**4 + 124516*ynl[lambda_below]**5 + 138738*ynl[lambda_below]**6 + 70996*ynl[lambda_below]**7 + 13107*ynl[lambda_below]**8 ), 4+38*4*ynl[lambda_below]**1 + 346/3*4**2*ynl[lambda_below]**2 + 409/3*4**3*ynl[lambda_below]**3 + 622/9*4**4*ynl[lambda_below]**4 + 43/3*4**5*ynl[lambda_below]**5 + 1*4**6*ynl[lambda_below]**6, 3+ 14*4*ynl[lambda_below]**1 + 19*16* ynl[lambda_below]**2 + 8*64*ynl[lambda_below]**3 + 256*ynl[lambda_below]**4, 2+3*4*ynl[lambda_below]**1 + 16*ynl[lambda_below]**2, 1][u]
    bf_cross_alt3[lambda_below] += me*c**2/ni**2 *np.abs(h*c/(lambda_grid.iloc[lambda_below]*1e-8) - Eion)/(h*c/(lambda_grid.iloc[lambda_below]*1e-8))**2 * 2**6 /3 *(1/137)**1 * h**2/(me**2*c**2) * np.power(mu3, 2)* np.exp(-4*mu3*np.arctan(ni/mu3)) *np.power((1- np.exp(-2*np.pi*mu3)) *(1 + ni**2/mu3**2)**2 ,-1)*[5*(1 + 64*ynl[lambda_below]**1 + 1343*ynl[lambda_below]**2 + 12589*ynl[lambda_below]**3 + 55801*ynl[lambda_below]**4 + 124516*ynl[lambda_below]**5 + 138738*ynl[lambda_below]**6 + 70996*ynl[lambda_below]**7 + 13107*ynl[lambda_below]**8 ), 4+38*4*ynl[lambda_below]**1 + 346/3*4**2*ynl[lambda_below]**2 + 409/3*4**3*ynl[lambda_below]**3 + 622/9*4**4*ynl[lambda_below]**4 + 43/3*4**5*ynl[lambda_below]**5 + 1*4**6*ynl[lambda_below]**6, 3+ 14*4*ynl[lambda_below]**1 + 19*16* ynl[lambda_below]**2 + 8*64*ynl[lambda_below]**3 + 256*ynl[lambda_below]**4, 2+3*4*ynl[lambda_below]**1 + 16*ynl[lambda_below]**2, 1][u]


    

    EionHe=np.power(227.8e-8/(h*c), -1) *np.power(ni, -2)
    ynlHe[lambda_belowHe] = EionHe*np.power(lambda_grid.iloc[lambda_belowHe]*1e-8/(h*c), 1)
    tau = np.sqrt(EionHe/(np.power(lambda_grid.iloc[lambda_belowHe]*1e-8/(h*c), -1)-EionHe))
    mu3He = np.sqrt((1/137)**2*c**2 *me/2*np.power( np.abs(h*c/(lambda_grid.iloc[lambda_belowHe]*1e-8) - EionHe), -1))


    bf_cross_alt3_He[lambda_belowHe] += me*c**2/ni**2 *np.abs(h*c/(lambda_grid.iloc[lambda_belowHe]*1e-8) - EionHe)/(h*c/(lambda_grid.iloc[lambda_belowHe]*1e-8))**2 * 2**6 /3 *(1/137)**1 * h**2/(me**2*c**2) * np.power(mu3He, 2)* np.exp(-4*mu3He*np.arctan(ni/mu3He)) *np.power((1- np.exp(-2*np.pi*mu3He)) *(1 + ni**2/mu3He**2)**2 ,-1)*[5*(1 + 64*ynlHe[lambda_belowHe]**1 + 1343*ynlHe[lambda_belowHe]**2 + 12589*ynlHe[lambda_belowHe]**3 + 55801*ynlHe[lambda_belowHe]**4 + 124516*ynlHe[lambda_belowHe]**5 + 138738*ynlHe[lambda_belowHe]**6 + 70996*ynlHe[lambda_belowHe]**7 + 13107*ynlHe[lambda_belowHe]**8 ), 4+38*4*ynlHe[lambda_belowHe]**1 + 346/3*4**2*ynlHe[lambda_belowHe]**2 + 409/3*4**3*ynlHe[lambda_belowHe]**3 + 622/9*4**4*ynlHe[lambda_belowHe]**4 + 43/3*4**5*ynlHe[lambda_belowHe]**5 + 1*4**6*ynlHe[lambda_belowHe]**6, 3+ 14*4*ynlHe[lambda_belowHe]**1 + 19*16* ynlHe[lambda_belowHe]**2 + 8*64*ynlHe[lambda_belowHe]**3 + 256*ynlHe[lambda_belowHe]**4, 2+3*4*ynlHe[lambda_belowHe]**1 + 16*ynlHe[lambda_belowHe]**2, 1][u]


    bf_cross_alt4[lambda_below] += 2**6*(1/137)*np.pi*(5.29e-9)**2*ni/(3*np.sqrt(3))*np.power(ni**2*912/(lambda_grid.iloc[lambda_below]), -3) *gbf4[lambda_below]#2**6*np.pi/np.sqrt(3)*ni*ni**2*ynl[lambda_below]**4*(1/137)**3*(5.29e-9)**2*gbf3[lambda_below]*me*c**2
    # Diff between kotelnikov and katkov 
        # np.pi*/(np.sqrt(3))

    #print('ratio multiple needed balmer ', ion_cross.iloc[i1]/bf_cross[i1], 'ratio multiple needed paschen ', ion_cross.iloc[i3]/bf_cross[i3], 'gbf_coeff ', 8*np.pi*np.sqrt(3)*np.power(ni, 2), '1/n**2 coeff ', ni**4, 'n= ', ni)
    
#    fb_cross_2[lambda_below] = np.power(lambda_grid.iloc[lambda_below]*1e-8/(h*c), -2)/(2*(np.power( lambda_grid.iloc[lambda_below]*1e-8/(h*c), -1)-Eion) * me* c**2)*bf_cross[lambda_below]* (weights[lambda_below])#*2*ni**2 
    #print('ratio multiple needed ', ion_cross.iloc[i4]/bf_cross_2[i4])

    u = u+1
plt.plot(lambda_grid , gbf2, label='g_n Seaton+59')
#plt.plot(lambda_grid , bf_cross_alt3 / (2**6 /(3**1.5) * h*e**2/(me*c) * Eion**2*np.power(lambda_grid/(h*c), 3)) , label='g_n Katkov')
plt.plot(lambda_grid , gbf4, label='g_n Kotelnikov+18', linewidth=4)
plt.plot(lambda_grid , gbf , label='g_n analytic', c='black')

gbf_Karzas = np.array([0.797, 0.853, 0.876, 0.975])
lambda_Karzas = np.array([911.8, 730, 3640, 1820])
plt.scatter(lambda_Karzas , gbf_Karzas ,label=r'Karzas&Latter+61 table', zorder=10)

plt.ylim(0, 1.2)
plt.xlim(600, 3700)
#bf_cross[lambda_grid < 911] = ion_cross[lambda_grid < 911]
plt.legend()
#print('ion_cross_O 911 ', ion_cross_O.iloc[test_ind2[0][0]])
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'g$_{n}$',size=16)

#plt.savefig(f'../cloudy_plots/{inputmodel}/gbf_v5_'+mod_name+'.jpg')
#plt.show()
plt.clf()

# deconstruct dependencies based on gbf components
#lambda_save_c = np.concatenate(lambda_save, axis=1)
#phis = np.concatenate(phi)

# If you want them as a single 2D array where each row is [lambda, phi]
#combined_data = np.column_stack((lambdas_save_c, phis))

#print(combined_data.shape) # Total number of points
plt.plot(lambda_grid, phi,  label='phi')
plt.plot(lambda_grid, phi[np.argmin(np.abs(lambda_grid - 350))]*np.exp(0.3*lambda_grid/350))
#plt.plot(np.concatenate([lambda_save[0], lambda_save[1], lambda_save[2], lambda_save[3]], axis=1) , np.concatenate([phi[0], phi[1], phi[2], phi[3]]) , label='phi')
plt.ylim(0, 8)
plt.xlim(300, 2000)
#bf_cross[lambda_grid < 911] = ion_cross[lambda_grid < 911]
plt.legend()
#print('ion_cross_O 911 ', ion_cross_O.iloc[test_ind2[0][0]])
#plt.savefig(f'../cloudy_plots/{inputmodel}/phi_v3_'+mod_name+'.jpg')
#plt.show()
plt.clf()

# deconstruct dependencies based on gbf components
plt.plot(lambda_grid, Fnl[np.argmin(np.abs(lambda_grid - 350))]*np.power(lambda_grid/350, -2))
plt.plot(lambda_grid , Fnl , label='Fnl')
plt.ylim(0, 0.001)
plt.xlim(3000, 4000)
#bf_cross[lambda_grid < 911] = ion_cross[lambda_grid < 911]
plt.legend()
#print('ion_cross_O 911 ', ion_cross_O.iloc[test_ind2[0][0]])
#plt.savefig(f'../cloudy_plots/{inputmodel}/Fnl_v3_'+mod_name+'.jpg')
#plt.show()
plt.clf()

# deconstruct dependencies based on gbf components
plt.plot(lambda_grid , 8*np.pi*np.sqrt(3)*n_levels**2 *ynl , label='coeff')
plt.xlim(300, 4000)
#bf_cross[lambda_grid < 911] = ion_cross[lambda_grid < 911]
plt.legend()
#print('ion_cross_O 911 ', ion_cross_O.iloc[test_ind2[0][0]])
#plt.savefig(f'../cloudy_plots/{inputmodel}/gbfcoeff_v1_'+mod_name+'.jpg')
#plt.show()
plt.clf()


ion_cross = (1)*n_levels**(-2)*5.475e-14 * np.power(lambda_grid/lambda0, 4.0185) * np.power(np.power(lambda_grid/lambda0 , -1) -1, 2) * np.power(1+ np.power(lambda_grid*32.88/(lambda0) , -0.5) , -2.963)
#ion_cross[lambda_grid > 911] = ion_cross.iloc[test_ind2[0][0]]

#ion_cross_const = 1*ion_cross
#ion_cross_const[lambda_grid > 911] = ion_cross_const.iloc[test_ind2[0][0]]

ion_cross_alt = 6.3e-18* np.power(lambda_grid/911.8, 3)
epsi = np.sqrt(np.power(lambda_grid/911.8, -1) -1)
ion_cross_O = 6.3e-18 * np.power(lambda_grid/911.8, 4)*np.exp(4-np.divide(4*np.arctan(epsi), epsi))*np.divide(1, 1-np.exp(np.divide(-2*np.pi, epsi)))

ion_cross_O2 = 6.3e-18*(1.34*np.power(lambda_grid/91.8, 2.99) + (1-1.34)*np.power(lambda_grid/911.8, 2.99+1))

#plt.plot(lambda_grid, ion_cross*10**18 , label='Verner fit 1996', c='orange', linestyle='dashed')
#plt.plot((h*c*10**8)/eV*np.power(lambda_grid, -1) , ion_cross_O*10**18 , linestyle='--', label='Osterbrock', c='red')
#plt.plot((h*c*10**8)/eV*np.power(lambda_grid, -1) , ion_cross_alt*10**18 , linestyle='-.',c='blue',  label='lambda**3 scaling')
plt.plot(lambda_grid, bf_cross_alt2*10**18 , label='Bound-free crossection Seaton+59')
#plt.plot((h*c*10**8)/eV*np.power(lambda_grid, -1) , bf_cross_alt1*10**18 , label='Bound-free crossection Seaton+59 * stat weights')#Correct for added/moved n**2
plt.plot(lambda_grid , bf_cross_alt3*10**18 , label='Bound-free Katkov', c='black', lw=2)
plt.plot(lambda_grid , bf_cross_alt4*10**18 , label='Bound-free Kotelnikov+18')
#bf_cross_alt1

plt.plot(lambda_grid, bf_cross*10**18 , label=r'Bound-free CHIANTI ($\lambda^3$ fit)', linestyle='dashed')
#plt.plot(lambda_grid[lambda_grid < 912] , ion_cross_O2[lambda_grid < 912]*10**18 , linestyle='--', label='Osterbrock ionising crossection', c='red')
plt.plot(lambda_grid, bf_cross_alt3_He*10**18 , label=r'He attempt', linestyle='dashed')

#plt.plot((h*c*10**8)/eV*np.power(lambda_grid, -1) ,  n_levels**(-4)*2**4*bf_cross_alt1[test_ind4[0][0]]*np.power(lambda_grid/3645, 3)*10**18, linestyle='--', c='r')

plt.xlim(200, 4000)
plt.ylim(0.0001, 100)
#plt.ylim(np.min(jnu_comb), np.max(ym_nebular))
#plt.xlim(100, 5000)
#plt.ylim(1*10**(-2), 10)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'ionisation cross section',size=16)
plt.legend()
plt.tight_layout()
#plt.savefig(f'../cloudy_plots/{inputmodel}/analyticcross_v9_'+mod_name+'.jpg')
#plt.show()
plt.clf()

### Recombination crossection

fig = plt.figure()
#plt.plot((h*c*10**8)/eV*np.power(lambda_grid, -1) , ion_cross_alt*10**18 , linestyle='-.',c='blue',  label='lambda**3 scaling')
#plt.plot(lambda_grid, bf_cross_alt2*10**18 , label='Bound-free crossection Seaton+59')
#plt.plot((h*c*10**8)/eV*np.power(lambda_grid, -1) , bf_cross_alt1*10**18 , label='Bound-free crossection Seaton+59 * stat weights')#Correct for added/moved n**2
plt.plot(lambda_grid , n_levels**2 * h**2 * c**2/(2*me*c**2 * lambda_grid**2)* np.power(np.abs(h*c/(lambda_grid*1e-8) - Eion), -1) *bf_cross_alt4*10**18 , label='Free-bound Katkov', c='black')
#plt.plot(lambda_grid , bf_cross_alt4*10**18 , label='Bound-free Kotelnikov+18')
#bf_cross_alt1

plt.plot(lambda_grid, n_levels**2 * h**2 * c**2/(2*me*c**2 * lambda_grid**2)* np.power(np.abs(h*c/(lambda_grid*1e-8) - Eion), -1) * bf_cross*10**18 , label=r'Freebound CHIANTI ($\lambda^3$ fit)', linestyle='dashed')
#plt.plot(lambda_grid[lambda_grid < 912] , ion_cross_O2[lambda_grid < 912]*10**18 , linestyle='--', label='Osterbrock ionising crossection', c='red')
#print('recomb cross at 850 Ang ', (n_levels**2 * h**2 * c**2/(2*me*c**2 * lambda_grid**2)* np.power(np.abs(h*c/(lambda_grid*1e-8) - Eion), -1) *bf_cross_alt4*10**18)[np.argmin(np.abs(lambda_grid - 850))])

#plt.plot((h*c*10**8)/eV*np.power(lambda_grid, -1) ,  n_levels**(-4)*2**4*bf_cross_alt1[test_ind4[0][0]]*np.power(lambda_grid/3645, 3)*10**18, linestyle='--', c='r')

plt.xlim(600, 1250)
plt.ylim(1e-21, 1e-10)
#plt.ylim(np.min(jnu_comb), np.max(ym_nebular))
#plt.xlim(100, 5000)
#plt.ylim(1*10**(-2), 10)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'\lambda [Angstrom]',size=16)
plt.ylabel(r'Recombination cross section',size=16)
plt.legend()
plt.tight_layout()
#plt.savefig(f'../cloudy_plots/{inputmodel}/Recombcross_v1_'+mod_name+'.jpg')
#plt.show()
plt.clf()

#gfftest = np.sqrt(3)/np.pi * (np.log(Ttest**1.5*c/(lambda_grid*1e-8)) + 17.7)

#gamma_B = 4*np.pi* 5.44e-41*Ttest**(-0.5)*gfftest*np.exp(-h*nu_grid/(kB*Ttest))


#weights = np.power(n_levels, 2)


#gamma_raw = np.power(lambda_grid*1e-8/c, -0.5)/np.pi * 4*np.pi * h/c**2 * np.power(lambda_grid*1e-8/c, -3) * gi * np.power(h**2/(2*np.pi*me*kB*Ttest) , 1.5) * ion_cross* c*np.power(lambda_grid*1e-8, -1) 
#gamma_raw_test = np.power(lambda_grid/911.26, 0)* np.array( 4*np.pi * h/c**2 * np.power(lambda_grid*1e-8/c, -3)  * np.power(h**2/(2*np.pi*me*kB*Ttest) , 1.5) * ion_cross)
#gamma_raw_test = np.array(gamma_raw_test)

#gamma_raw_test2 = np.power(lambda_grid/911.26, 0)* np.array( 4*np.pi * h**2* * np.power(lambda_grid*1e-8/c, -3)  * np.power(h**2/(2*np.pi*me*kB*Ttest) , 1.5) * fb_cross_2* c*np.power(lambda_grid*1e-8, -1))#*weights
gamma_raw_test =  weights* 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid*1e-8/c, -3)  * np.power(me/(2*np.pi*kB*Ttest) , 1.5) * bf_cross#*2*np.power(n_levels, 2)

gamma_raw_test = np.array(gamma_raw_test)

gamma_raw_test2 = (1)*weights* 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid*1e-8/c, -3)  * np.power(me/(2*np.pi*kB*Ttest) , 1.5) * bf_cross_alt2

gamma_raw_test2 = np.array(gamma_raw_test2)

#gamma_raw_test3 =  (ynl**(0)*n_levels**(0))*2**3*h**2*np.pi/(me**2)*c/(lambda_grid*1e-8) * np.abs(h*c/(lambda_grid*1e-8) - Eion)  *np.power(me/(2*np.pi*kB*Ttest) , 1.5)* bf_cross_alt3  #weights*4*np.pi * h**4/(me**3*c**2) *np.power(lambda_grid*1e-8/c, -3)  * np.power(me/(2*np.pi*kB*Ttest) , 1.5) * bf_cross_alt3

gamma_raw_test3 =  (1)*weights* 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid*1e-8/c, -3) *np.power(me/(2*np.pi*kB*Ttest) , 1.5)* bf_cross_alt3

gamma_raw_test3 = np.array(gamma_raw_test3)

gamma_raw_test4 =  3.0992e-46 * np.power(lambda_grid*1e-8, -5)* Ttest**(-1.5) *  weights*(bf_cross/1e-18) *lambda_grid *lambda_grid*1e-8/c

gamma_raw_test4 = np.array(gamma_raw_test4)


gamma_raw_test5 =  3.0992e-46 * np.power(lambda_grid*1e-8, -5)* Ttest**(-1.5) *  weights*(bf_cross/1e-18) *lambda_grid *lambda_grid*1e-8/c

gamma_raw_test5 = np.array(gamma_raw_test4)

Ztest=2


gamma_He5 = np.ones_like(lambda_grid)
gamma_He5[lambda_grid < 227.8*5**2] = 1/5**1*weights[lambda_grid < 227.8*5**2]* 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid[lambda_grid < 227.8*5**2]*1e-8/c, -3) *np.power(me/(2*np.pi*kB*Ttest/Ztest**2) , 1.5)* bf_cross_alt3_He[lambda_grid < 227.8*5**2] * np.exp(-h*(nu_grid[lambda_grid < 227.8*5**2]-c/(5**2*227.8e-8))/(kB*Ttest))



gamma_He4 = np.ones_like(lambda_grid)
gamma_He4[lambda_grid < 227.8*5**2] = gamma_He5[lambda_grid < 227.8*5**2]
gamma_He4[lambda_grid < 227.8*4**2] += 1/4**1*weights[lambda_grid < 227.8*4**2]* 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid[lambda_grid < 227.8*4**2]*1e-8/c, -3) *np.power(me/(2*np.pi*kB*Ttest/Ztest**2) , 1.5)* bf_cross_alt3_He[lambda_grid < 227.8*4**2] * np.exp(-h*(nu_grid[lambda_grid < 227.8*4**2]-c/(4**2*227.8e-8))/(kB*Ttest))



gamma_He3 = np.ones_like(lambda_grid)
gamma_He3[lambda_grid < 227.8*5**2] = gamma_He4[lambda_grid < 227.8*5**2]
gamma_He3[lambda_grid < 227.8*3**2] += 1/3**1*weights[lambda_grid < 227.8*3**2]* 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid[lambda_grid < 227.8*3**2]*1e-8/c, -3) *np.power(me/(2*np.pi*kB*Ttest/Ztest**2) , 1.5)* bf_cross_alt3_He[lambda_grid < 227.8*3**2] * np.exp(-h*(nu_grid[lambda_grid < 227.8*3**2]-c/(3**2*227.8e-8))/(kB*Ttest))


gamma_He2 = np.ones_like(lambda_grid)
gamma_He2[lambda_grid < 227.8*5**2] = gamma_He3[lambda_grid < 227.8*5**2]
gamma_He2[lambda_grid < 227.8*2**2] += 1/2**1*weights[lambda_grid < 227.8*2**2]* 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid[lambda_grid < 227.8*2**2]*1e-8/c, -3) *np.power(me/(2*np.pi*kB*Ttest/Ztest**2) , 1.5)* bf_cross_alt3_He[lambda_grid < 227.8*2**2] * np.exp(-h*(nu_grid[lambda_grid < 227.8*2**2]-c/(2**2*227.8e-8))/(kB*Ttest))

gamma_He = np.ones_like(lambda_grid)
gamma_He[lambda_grid < 227.8*5**2] = gamma_He2[lambda_grid < 227.8*5**2]
gamma_He[lambda_grid < 227.8] += weights[lambda_grid < 227.8]* 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid[lambda_grid < 227.8]*1e-8/c, -3) *np.power(me/(2*np.pi*kB*Ttest/Ztest**0) , 1.5)* bf_cross_alt3_He[lambda_grid < 227.8] * np.exp(-h*(nu_grid[lambda_grid < 227.8]-c/227.8e-8)/(kB*Ttest))






print('ratio of gamma_raw/gamma_raw_test3 ', gamma_raw_test3[test_ind[0][0]]/gamma_raw_test[test_ind[0][0]])
print('ratio of gamma 912 over gamma 3400 ', gamma_raw_test[np.argmin(np.abs(lambda_grid - 911.8))]/gamma_raw_test[np.argmin(np.abs(lambda_grid - 3400))])
#gamma_raw_test3 = np.power(lambda_grid/911, 0)* np.array( 4*np.pi * h/c**2 * np.power(lambda_grid*1e-8/c, -3) * gi * np.power(h**2/(2*np.pi*me*kB*Ttest) , 1.5) * ion_cross_alt
#gamma_raw_test3 = np.array(gamma_raw_test3)

#gamma_raw_test = gamma_raw#*np.exp(np.divide(deltaE,-1*kB_Ryd*Ttest)) #* gamma.iloc[test_ind[0][0]]/gamma_raw[test_ind[0][0]]

gamma_Osterbrock = np.array([ 20, 4.5, 1.5, 0.0025, 0.196, 6.17, 13.8, 20.5, 1.13, 1.45, 1.90, 2.89, 3.69, 4.18, 1.54, 1.72, 0.772, 0.262])*10**(-25) 
lambda_Osterbrock = np.array([0.35, 0.81, 1.45, 0.1, 0.15, 0.26, 0.3122, 0.3645, 0.3647, 0.4, 0.45, 0.5696, 0.7, 0.8203, 0.8205, 1, 3, 10])

fig = plt.figure()

#plt.axvline(x= (h*c*10**8)/eV*np.power(0.4298, -1)/10000,ls=':',alpha=0.7,color='#606060',lw=1.5, label='ionisation cross section fit limit')
print('exp effect near n=2 peak ', np.exp(np.divide(deltaE[np.argmin(np.abs(lambda_grid - 911))],-1*kB_Ryd*Ttest)), 'CHIANTI at n=1 ', gamma_raw_test4[np.argmin(np.abs(lambda_grid - 911))]* 911e-8 *911e-8/c, 'CHIANTI exp ', np.exp(-1/(kB * Ttest) *np.abs(h*c/911e-8-h*c/(912e-8*1**2))) , 'E= ', np.power(912*1e-8/(h*c), -1))

plt.plot(lambda_grid , nu_grid*(1+np.exp(-h*c/(lambda_grid*1e-8*kB*Ttest)))**1 * (n_levels/2)**0 *((gamma_Bremstrahlung/0.185)*0  + np.exp(np.divide(deltaE,-1*kB_Ryd*Ttest)) *gamma_raw_test3) ,label=r'Free-bound analytic (Katkov+78)', c='black')
#(1+np.exp(-h*c/(lambda_grid*1e-8*kB*Ttest)))

plt.plot(lambda_grid[lambda_grid < 227.8] ,nu_grid[lambda_grid < 227.8]*gamma_He[lambda_grid < 227.8] ,label=r'Free-bound He', c='red')
plt.plot(lambda_grid[lambda_grid < 227.8*2**2] ,nu_grid[lambda_grid < 227.8*2**2]*gamma_He2[lambda_grid < 227.8*2**2], label=r'Free-bound He 2', c='red')
plt.plot(lambda_grid[lambda_grid < 227.8*3**2] ,nu_grid[lambda_grid < 227.8*3**2]*gamma_He3[lambda_grid < 227.8*3**2], label=r'Free-bound He 3', c='red')

plt.plot(lambda_grid[lambda_grid < 227.8*4**2] ,nu_grid[lambda_grid < 227.8*4**2]*gamma_He4[lambda_grid < 227.8*4**2], label=r'Free-bound He 4', c='red')


plt.plot(lambda_grid[lambda_grid < 227.8*5**2] ,nu_grid[lambda_grid < 227.8*5**2]*gamma_He5[lambda_grid < 227.8*5**2], label=r'Free-bound He 5', c='red')
#plt.plot(lambda_grid , (1/(4*np.pi))**0* np.exp(-1/(kB * Ttest) *np.abs(h*c/(lambda_grid*1e-8) -h*c/(912e-8*n_levels**2))) *gamma_raw_test4 ,label=r'Free-bound (CHIANTI) ', linestyle='dashed', c='red')

#plt.plot(lambda_grid , np.exp(-1/(kB * Ttest) *np.abs(h*c/(lambda_grid*1e-8) -h*c/(912e-8*n_levels**2))) *gamma_raw_test2 ,label=r'Free-bound (Seaton) ')

plt.plot(lambda_grid , nu_grid * gamma_2S *np.power(lambda_grid*1e-8/c, 0),label=r'2-photon (Nussbaumer)', c='orange')
#plt.plot(lambda_grid , gamma_2S_alt *np.power(lambda_grid*1e-8/c, 0),label=r'2-photon (Osterbrock)', c='purple', linestyle='--')
plt.plot(lambda_grid , nu_grid *gamma_Bremstrahlung*np.power(lambda_grid*1e-8/c, 0) ,label=r'Free-free', c='purple')
#plt.plot(lambda_grid , np.exp(np.divide(deltaE,-1*kB_Ryd*Ttest))*gamma_raw_test2*np.power(lambda_grid*1e-8/c, 0) ,label=r'Free-bound Saton+59')
#plt.plot(lambda_grid , (1/(4*np.pi))**0*np.exp(np.divide(deltaE,-1*kB_Ryd*Ttest))*gamma_raw_test3*np.power(lambda_grid*1e-8/c, 0) ,label=r'Free-bound Katkov+78', c='darkgreen')

#plt.plot(lambda_grid/10000 , gamma_raw_test3*np.divide(1, 1) ,label=r'Free-bound (Osterbroke other fit)', linestyle='-', c='red')
#plt.plot(lambda_grid/10000 , gamma_raw_test3*np.divide(1, 1)*np.power(lambda_grid/911, -0.5) ,label=r'Free-bound gamma (power of $\nu^{2.5}$)', linestyle=':')
#plt.plot(lambda_grid/10000 , gamma*np.divide(1, 1) ,label=r'noexponential (analytic)',  linestyle='--')


#plt.scatter(lambda_Osterbrock*1e4 , gamma_Osterbrock*np.power(lambda_Osterbrock*1e4*1e-8/c, 1) ,label=r'Osterbrock table', zorder=10)

gamma_Draine = np.array([4.86e-25, 4.7e-24, 1.52e-26,1.6e-25, 8.4e-27, 2.17e-26, 4e-27, 5.26e-27, 1.72e-27])
lambda_Draine = np.array([783, 911, 2061, 3640, 4114, 8200, 9448, 14550, 22750])
#plt.scatter(lambda_Draine , (4*np.pi)**(1)*gamma_Draine*np.power(lambda_Draine*1e-8/c, 1) ,label=r'Peaks from Draine book/plot', zorder=10)

print('Draine vs our solution ratio, n=1 ', (4*np.pi)**(1)*gamma_Draine[1]*np.power(lambda_Draine[1]*1e-8/c, 1)/gamma_raw_test[np.argmin(np.abs(lambda_grid - 910))], 'n=2 ', (4*np.pi)**(1)*gamma_Draine[4]*np.power(lambda_Draine[4]*1e-8/c, 1)/gamma_raw_test[np.argmin(np.abs(lambda_grid - 3600))],  'n=3 ', (4*np.pi)**(1)*gamma_Draine[6]*np.power(lambda_Draine[6]*1e-8/c, 1)/gamma_raw_test[np.argmin(np.abs(lambda_grid - 8200))]  )


print('n=1 peak w/o sum ', gamma_raw_test3[np.argmin(np.abs(lambda_grid - 910))], 'n=1 peak w. sum ', gamma_raw_test3[np.argmin(np.abs(lambda_grid - 910))] + gamma_raw_test3[np.argmin(np.abs(lambda_grid - 3640))] + gamma_raw_test3[np.argmin(np.abs(lambda_grid - 8150))] + gamma_raw_test3[np.argmin(np.abs(lambda_grid - 14500))] , 'n=1 Draine peak ', (4*np.pi)**(1)*gamma_Draine[1]*np.power(lambda_Draine[1]*1e-8/c, 1))

gamma_Ercolano2 = 1e-40*np.array([3.704e-1, 2.376e1, 2.629e-6, 1.698e-4, 1.093e-2, 6.996e-1, 4.442e1, 1.653e0, 2.851e0, 2.418e1])
lambda_Ercolano2 = np.array([2998, 3331, 3747, 4283, 4996, 5996, 7495, 9993, 14990, 29979])

#plt.plot(lambda_grid[lambda_grid > 912] , np.exp(-1/(kB * Ttest) *np.abs(h*c/(lambda_grid[lambda_grid > 912]*1e-8) -h*c/(912e-8*n_levels[lambda_grid > 912]**2))) *gamma_H[lambda_grid > 912]*np.power(lambda_grid[lambda_grid > 912]*1e-8/c, 0) ,c='grey', linestyle=':', label=r'Free-bound (Ercolano table)')
#plt.scatter(lambda_Ercolano2 , (4*np.pi*(1e4**1.5*1e4**(-1.5)))**1*gamma_Ercolano2*np.power(lambda_Ercolano2*1e-8/c, 0) ,label=r'Free-bound (Ercolano table 2)', zorder=10)

#print('ion_cross 900 ', ion_cross.iloc[test_ind2[0][0]], 'ion_cross 3600  ', ion_cross.iloc[test_ind[0][0]], 'compared to 6Mb ', 6*10**(-18)/ion_cross.iloc[test_ind2[0][0]])
#print('gamma_raw 900 ', gamma_raw_test[test_ind2[0][0]], 'raw_smooth 3600  ', gamma_raw_test[test_ind[0][0]], 'ratio 900 ', gamma_raw_test[test_ind2[0][0]]/gamma.iloc[test_ind2[0][0]], 'ratio 3600 ', gamma_raw_test[test_ind[0][0]]/gamma.iloc[test_ind[0][0]])
#print('gamma_smooth 900 ', gamma_smooth.iloc[test_ind2[0][0]], 'gamma_smooth 3600  ', gamma_smooth.iloc[test_ind[0][0]])
#print('gamma 900 ', gamma.iloc[test_ind2[0][0]], 'gamma 3600  ', gamma.iloc[test_ind[0][0]])
print('gnu_2S_alt 1300 ', gnu_2S_smooth[i1], 'gnu_2S_alt 2000  ', gnu_2S_smooth[i2], 'gnu_2S 1300 ', gnu_2S[i1], 'gnu_2S 2000  ', gnu_2S[i2], 'ratio ', gnu_2S[i1]/gnu_2S_smooth[i1])

print('')
#print('A ', (1.5)* 1/(10**34 * Ttest**1.5) * gamma_smooth[test_ind[0][0]] *3*10**(8)* np.power(lambda_grid.iloc[test_ind[0][0]], -1)*10**10   *  lambda_grid.iloc[test_ind[0][0]]**3, 'B ', test_ind[0][0] )

#plt.plot(lambda_grid.iloc[test_ind]/10000 ,  gamma_smooth.iloc[test_ind[0][-1]] *np.ones_like(lambda_grid.iloc[test_ind]) *np.power(lambda_grid.iloc[test_ind]/910 ,-1) ,label=r'Seb test model', linestyle='--', c='r')#*3*10**(8)* np.power(lambda_grid.iloc[test_ind[0][0]], -1)*10**10   *  lambda_grid.iloc[test_ind[0][0]]**1 *np.power(lambda_grid.iloc[test_ind], -1)* np.divide(1, nu_grid.iloc[test_ind]) ,label=r'lambda scaling', linestyle='--', c='r')
expo_test = np.divide(h*c,kB*Ttest)*(1/911 - np.power(lambda_grid.iloc[test_ind2], -1))*10**8

test_line = n_levels**(-3)*2**3 *gamma_raw_test[test_ind[0][0]]*np.power(lambda_grid*1e-8/c, 0)*np.power(lambda_grid/3645, 0)
#plt.plot(lambda_grid ,  test_line, linestyle=':', c='r', lw=3)
#plt.plot(lambda_grid.iloc[test_ind2] ,  test_line.iloc[test_ind2[0][0]]*np.exp(expo_test), linestyle=':', c='r', label=r'Simple approximation', lw=3)#*3*10**(8)* np.power(lambda_grid.iloc[test_ind2[0][0]], -1)*10**10 * np.exp(expo_test)*np.divide(1, nu_grid.iloc[test_ind2])  , linestyle='--', c='r')

plt.axvline(x=600,ls=':',alpha=0.7,color='#606060',lw=1.5)
plt.axvline(x=912,ls=':',alpha=0.7,color='#606060',lw=1.5)
#plt.axvline(x=1215,ls=':',alpha=0.7,color='#606060',lw=1.5)
#plt.axvline(x=3650,ls=':',alpha=0.7,color='#606060',lw=1.5)
#plt.text(540,np.max(ym_nebular_unscaled/(4*np.pi*R.iloc[-1]**2))*0.6,r'f$_{600}$',size=16,rotation=90,color='#606060')
#plt.text(855,np.max(ym_nebular_unscaled/(4*np.pi*R.iloc[-1]**2))*0.6,r'Ly_C',size=16,rotation=90,color='#606060')
#plt.text(1160,np.max(yinc/y_1100)*0.6,r'Ly${\alpha}$',size=16,rotation=90,color='#606060')
#plt.text(3650,np.max(yinc/y_1100)*0.6,r'H_C$',size=16,rotation=90,color='#606060')

plt.xlim(1000, 6000)
plt.ylim(1e-26, 1e-22)
#plt.ylim(1e-41, 4e-38)
#plt.ylim(1e-50, 1e-30)
#plt.ylim(np.min(jnu_comb), np.max(ym_nebular))
#plt.xlim(100, 5000)
#plt.ylim(1*10**(-2), 10)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'$\nu$gamma',size=16)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
#plt.savefig(f'../cloudy_plots/{inputmodel}/analyticgamma_v28_'+mod_name+'.jpg')
#plt.show()
plt.clf()

###


# extreme temperature gamma

Thot = 1e7

#Thot = 1e4

gff = np.sqrt(3)/np.pi * (np.log((2*kB*Thot)**1.5*c/(np.pi*e**2*me**0.5 *lambda_grid*1e-8)) -1.443)

gff2 = 1 + 0.44/(1+0.058* (np.log(Thot/10**(5.4)))**2)*np.ones_like(gff)

gff3 = 4.691*(1 - 0.118*np.log(3/(lambda_grid*1e-8*(Thot/1e4)**1.5)))

gff4 = 3/np.pi**0.5 *np.log(2.25*kB*Thot*lambda_grid/(h*c)) 

fig = plt.figure()

plt.plot(lambda_grid , gff ,label=r'gff')
plt.plot(lambda_grid , gff2 ,label=r'gff2')
plt.plot(lambda_grid , gff3 ,label=r'gff3')
plt.plot(lambda_grid , gff4 ,label=r'gff4')
#plt.ylim(1e-44, 4e-34)
plt.xlim(100, 2000)
#plt.ylim(1*10**(-2), 10)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'gff',size=16)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
#plt.savefig(f'../cloudy_plots/{inputmodel}/gff_v1_'+mod_name+'.jpg')
#plt.show()
plt.clf()

gamma_hotFB = np.exp(np.divide(deltaE,-1*kB_Ryd*Thot))*weights* 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid*1e-8/c, -3) *np.power(me/(2*np.pi*kB*Thot) , 1.5)* bf_cross_alt3

gamma_hotFF = 2**2 * np.pi *5.44e-41*(Thot/1e4)**(-0.5)*gff2*np.exp(-h*nu_grid/(kB*Thot))

gamma_hotFF2 = 2**5 * np.pi * e**6 /(3*me*c**3) *np.sqrt(2*np.pi/(3*me*kB*Thot)) * gff2 * np.exp(-h*nu_grid/(kB*Thot)) 

gamma_hotFF3 = 2**2 * np.pi * 2**3 * e**6 / (3*me*c**3) * np.sqrt(2*np.pi/(3*me *kB*Thot)) * gff2 * np.exp(-h*nu_grid/(kB*Thot))

print('ratio between FFs ', gamma_hotFF2.iloc[np.argmin(np.abs(lambda_grid - 910))]/gamma_hotFF.iloc[np.argmin(np.abs(lambda_grid - 910))])

fig = plt.figure()

plt.plot(lambda_grid , gamma_hotFF ,label=r'Free-free (Draine)', c='purple')
plt.plot(lambda_grid , gamma_hotFF2 ,label=r'Free-free 2.0')
plt.plot(lambda_grid , gamma_hotFB ,label=r'FB')

plt.ylim(1e-44, 4e-34)
plt.xlim(500, 4000)
#plt.ylim(1*10**(-2), 10)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'gamma hot',size=16)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
#plt.savefig(f'../cloudy_plots/{inputmodel}/gammahot_v1_'+mod_name+'.jpg')
#plt.show()
plt.clf()


# Tenoeratyre dependence of gamma


gamma_T4 = np.exp(np.divide(deltaE,-1*kB_Ryd*1e4))*weights* 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid*1e-8/c, -3) *np.power(me/(2*np.pi*kB*1e4) , 1.5)* bf_cross_alt3
gamma_T3 = np.exp(np.divide(deltaE,-1*kB_Ryd*5e3))*weights* 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid*1e-8/c, -3) *np.power(me/(2*np.pi*kB*5e3) , 1.5)* bf_cross_alt3
gamma_T5 = np.exp(np.divide(deltaE,-1*kB_Ryd*5e4))*weights* 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid*1e-8/c, -3) *np.power(me/(2*np.pi*kB*5e4) , 1.5)* bf_cross_alt3


fig = plt.figure()

plt.plot(lambda_grid , gamma_T3 ,label=r'T = 5,000 K')
plt.plot(lambda_grid , gamma_T4 ,label=r'T = 10,000 K', c='purple')
plt.plot(lambda_grid , gamma_T5 ,label=r'T = 50,000 K')

plt.ylim(1e-44, 4e-34)
plt.xlim(500, 912)
#plt.ylim(1*10**(-2), 10)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'gamma',size=16)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
#plt.savefig(f'../cloudy_plots/{inputmodel}/gamma-T_v1_'+mod_name+'.jpg')
#plt.show()
plt.clf()



# Emissivity of free-free in hot ionised regime

fig = plt.figure()

plt.plot(lambda_grid , gamma_hotFF ,label=r'Free-free (Draine)', c='purple')
plt.plot(lambda_grid , gamma_hotFF2 ,label=r'Free-free 2.0')
plt.plot(lambda_grid , gamma_hotFB ,label=r'FB')

plt.ylim(1e-44, 4e-34)
plt.xlim(500, 4000)
#plt.ylim(1*10**(-2), 10)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'gamma hot',size=16)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
#plt.savefig(f'../cloudy_plots/{inputmodel}/gammahot_v1_'+mod_name+'.jpg')
#plt.show()
plt.clf()


df_phy = pd.read_csv(inputmodel+'.phy', sep='\t')

df_H = pd.read_csv(inputmodel+'.ele_H', sep='\t')

df_He = pd.read_csv(inputmodel+'.ele_He', sep='\t')

df_rad = pd.read_csv(inputmodel+'.rad', sep='\t')

display(df_phy)

#h =  6.6e-34 * 10**(3+4) # [g cm^2 /s]
#e = 1.6e-19
#mH =1.67e-24 
#me = 9.1e-31 * 10**(3)
#c = 3e10 # [A/s] 
#kB_Ryd = 1.38e-23*4.58e17
#kB = 1.38e-23*10**7
#gff = 1.3 # Based on book
#Z=1
#A2S = 8.31

T = df_phy['Te']
nH = df_phy['n(H)']
ne = df_phy['n(e)']
nHe = 10**(-1.0845) * nH#df_He['He+2']

Xfrac = df_H['H+'] 
XfracHe = df_He['He+2']
R = df_rad['radius']
dR = df_rad['dr']


#dR = np.diff(R)
#dR = R[0] - 1e15#np.diff(R)
print('R ', R, 'dR ', dR)
#print('R[0] ', R[0])

nu_grid = np.divide(c, lambda_grid*1e-8)


### Now try to estimate LyC area based on the simple model

ym_nebular_unscaled = df.loc[:,'DiffOut'].iloc[analytic_idx] * lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)

ym_reflect = df.loc[:,'reflc'].iloc[analytic_idx] * lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)
ym_intrinsic = df.loc[:,'incident'].iloc[analytic_idx] * lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)
ym_trans = df.loc[:,'trans'].iloc[analytic_idx] * lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)
#print('lineID', df[5700:5800]['lineID'])
print('lineID filtered ', df.loc[:,'lineID'].iloc[np.where(df.loc[:,'lineID'] == 'Inci')])
#print('lineID filtered ', df.loc[:,'lineID'][df.loc[:,'lineID'] != 'Inci'])

ym_neblines = (df.loc[:,'outlin'].iloc[analytic_idx].iloc[np.where(df.iloc[analytic_idx].loc[:,'lineID'] != 'Inci')] * lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)).fillna(0)
ym_intlines = (df.loc[:,'outlin'].iloc[analytic_idx].iloc[np.where(df.iloc[analytic_idx].loc[:,'lineID'] == 'Inci')] * lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)).fillna(0)

#print('ym_intlines ', ym_intlines, 'ym_neblines ', ym_neblines)
print('len ym_intlines ', np.sum((df.loc[:,'outlin'].iloc[analytic_idx].iloc[np.where(df.iloc[analytic_idx].loc[:,'lineID'] != 'Inci')]).fillna(0)), 'len ym_neblines ', np.sum((df.loc[:,'outlin'].iloc[analytic_idx].iloc[np.where(df.iloc[analytic_idx].loc[:,'lineID'] != 'Inci')] * lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)).fillna(0)))
#print('test6 ', ym_nebular_unscaled, 'test7 ', np.power(lambda_grid*1e-8/c, 1))
#ym_nebular_unscaled = np.multiply(ym_nebular_unscaled, lambda_grid*1e-8/c)
#print('test8 ', ym_nebular_unscaled)
Ly_bump_indices = np.where((600 < lambda_grid)  & (lambda_grid < 910))
#HA_ind = np.where((1250 < xm)  & (xm < 3644))
#print('diffout / reflect ', ym_nebular_unscaled[-1]/ym_reflect[-1])
Ly_bump_peak_idx = np.argmax(lambda_grid < 910)
H_bump_peak_idx = np.argmax(lambda_grid < 3640)

Ly_bump_peak_C = ym_nebular_unscaled.iloc[Ly_bump_peak_idx]/(R.iloc[-1]**2)*np.power(lambda_grid.iloc[Ly_bump_peak_idx]*1e-8/c, -1)
H_bump_peak_C =  ym_nebular_unscaled.iloc[H_bump_peak_idx]/(R.iloc[-1]**2)*np.power(lambda_grid.iloc[H_bump_peak_idx]*1e-8/c, -1)

print('xm.iloc[np.argmax(xm < 911)] ', lambda_grid.iloc[np.argmax(lambda_grid < 911)], 'xm.iloc[np.argmax(xm < 3645)] ', lambda_grid.iloc[np.argmax(lambda_grid < 3645)])

Ly_bump_peak_S =  FluxLH_comb[Ly_bump_peak_idx]*np.power(lambda_grid.iloc[Ly_bump_peak_idx]*1e-8/c, -1) 
H_bump_peak_S = FluxLH_comb[H_bump_peak_idx]*np.power(lambda_grid.iloc[H_bump_peak_idx]*1e-8/c, -1) 

Ly_bump_int_C = 1*np.trapezoid(ym_nebular_unscaled.iloc[Ly_bump_indices]/(R.iloc[-1]**2)*np.power(lambda_grid.iloc[Ly_bump_indices]*1e-8/c, -1), np.power(lambda_grid.iloc[Ly_bump_indices]*1e-8/c, -1))
Ly_bump_int_Seb = 1*np.trapezoid(FluxLH_comb[Ly_bump_indices], np.power(lambda_grid.iloc[Ly_bump_indices]*1e-8/c, -1))

# Factor 4 from n and factor 4 from (911/3645)**(-1) (only if dealing with F_nu * nu)
Est_Ly_bump_peak_C = 4*4*H_bump_peak_C 
Est_Ly_bump_int_C = 1*np.trapezoid(Ly_bump_peak_C * np.exp(h*c*( 1/(910e-8) - np.divide(1, lambda_grid.iloc[Ly_bump_indices]*1e-8))/(1*kB*Ttest))*np.power(lambda_grid.iloc[Ly_bump_indices]*1e-8/c, 0), np.power(lambda_grid.iloc[Ly_bump_indices]*1e-8/c, -1)) 


intrinsicTot = np.trapezoid(ym_intrinsic +ym_intlines, np.power(lambda_grid*1e-8/c, -1))
intrinsicUV = np.trapezoid(ym_intrinsic[lambda_grid<912]+ym_intlines[lambda_grid<912] , np.power(lambda_grid[lambda_grid<912]*1e-8/c, -1))

transmittedTot = np.trapezoid(ym_trans +ym_intlines, np.power(lambda_grid*1e-8/c, -1))
transmittedUV = np.trapezoid(ym_trans[lambda_grid<912] +ym_intlines[lambda_grid<912], np.power(lambda_grid[lambda_grid<912]*1e-8/c, -1))

#trans-intr_nonUV = np.trapezoid(ym_intrinsic[lambda_grid > 912] , np.power(lambda_grid[lambda_grid > 912]*1e-8/c, -1)) - np.trapezoid(ym_trans[lambda_grid > 912] , np.power(lambda_grid[lambda_grid > 912]*1e-8/c, -1))

nebularTot = np.trapezoid( ym_nebular_unscaled+ym_reflect, np.power(lambda_grid*1e-8/c, -1))
nebularUV = np.trapezoid( ym_nebular_unscaled[lambda_grid<912] + ym_reflect[lambda_grid<912]  , np.power(lambda_grid[lambda_grid<912]*1e-8/c, -1))
nebularLines = np.trapezoid(ym_neblines , np.power(lambda_grid*1e-8/c, -1))
#Tot_neb_int = np.trapezoid( ym_nebular_unscaled , np.power(lambda_grid*1e-8/c, -1))

#absUV =  -1*(transmittedUV_int - intrinsicUV_int - nebularUV_int)

absTot = -1*(transmittedTot -  intrinsicTot)
absUV = -1*(transmittedUV -  intrinsicUV)


print('nebularTot ', nebularTot+nebularLines , 'total absorbed ', absTot , 'transmittedTot - intrinsicTot ', transmittedTot -  intrinsicTot, 'subtractalt ', -1*(transmittedTot -  intrinsicTot + nebularTot ))
print('nebularTot ', nebularTot, 'nebularTot+Lines ', nebularTot+nebularLines  , 'UV absorbed ', absUV, 'UV absorbed (inc. nebUV) ', absUV+ nebularUV, 'transmittedUV - intrinsicUV ', transmittedUV -  intrinsicUV, 'nebularUV ', nebularUV, 'nebularLines ', nebularLines)
print('')

# CLOUDY total recombination coeff, output: 4.16e-13, CLOUDY: 2.15e-12, Ours: 1.5e-13, xm cloudy: 2.3e-14
# For n=1 CLOUDY: ,  Ours: 7.85e-14 
# For n!= 1, CLOUDY: , Ours: 
alpha = np.trapezoid((np.exp(np.divide(deltaE,-1*kB_Ryd*Ttest))*gamma_raw_test)[lambda_grid < 912]*np.power(lambda_grid[lambda_grid < 912]*1e-8/(c*h), 1) , np.power(lambda_grid[lambda_grid < 912]*1e-8/c, -1))

alphaC = np.trapezoid(ym_nebular_unscaled[lambda_grid < 22000]/(10*1e15*100*100)  *np.power(lambda_grid[lambda_grid < 22000]*1e-8/(c*h), 1) , np.power(lambda_grid[lambda_grid < 22000]*1e-8/c, -1))

alphaC = np.trapezoid(ym_nebular[xm < 22000] *4*np.pi* xm[xm < 22000]*1e-8/c /(4*np.pi*1e34*  10*1e15*100*100)  *np.power(xm[xm < 22000]*1e-8/(c*h), 1) , np.power(xm[xm < 22000]*1e-8/c, -1)) #2.3e-14
#lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)
print('recombination coefficient Ours ', alpha, 'alpha CLOUDY ', alphaC)

#print('Est_Ly_bump_peak_C ', Est_Ly_bump_peak_C,'Ly_bump_peak_S ', Ly_bump_peak_S, 'Ly_bump_peak_C ', Ly_bump_peak_C,'Analytic ratio peak ', Ly_bump_peak_S/Ly_bump_peak_C,'Estimate ratio peak ', Est_Ly_bump_peak_C/Ly_bump_peak_C ,'Est_Ly_bump_int_C ', Est_Ly_bump_int_C,'Ly_bump_int_C ', Ly_bump_int_C, 'ratio areas ', Est_Ly_bump_int_C/Ly_bump_int_C)
#print('test1 ', ym_nebular_unscaled.iloc[Ly_bump_peak_idx], 'test2 ', R.iloc[0], 'test3 ', df.loc[:,'DiffOut'].iloc[-1], 'test4 ', ym_nebular_unscaled[0], 'test5 ', Ly_bump_peak_idx)



#def gamma_bf_Hlike(Z, Ti, bf_cross, nu_0):
#    gamma = np.zeros_like(bf_cross)
#    gamma_base = np.zeros_like(bf_cross)
#    for n in [1, 2, 3, 4]:#,2,3,4, 5]:
#        #gamma_base = np.zeros_like(bf_cross#)
#        weight = (2**0/(2*n**0))**1 *n**2 
#        nu_cutoff = (nu_grid > nu_0/n**2) #& (nu_0/(n+1)**2 > nu_grid)
#        gamma_base[nu_cutoff] = (4/n**2)**0*weight * Z**3 * 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid[nu_cutoff]*1e-8/c, -3) *np.power(me/(2*np.pi*kB*Ti) , 1.5)* bf_cross[nu_cutoff] * np.exp(h*(nu_0/n**2 - nu_grid[nu_cutoff])/(kB*Ti)) *(1-np.exp(-h*nu_grid[nu_cutoff]/(kB*Ti)))
#        gamma += gamma_base
    
#    return gamma


def gamma_bf_Hlike(Z, Ti, bf_cross, nu_912):
    nu_0 = nu_912*Z**2
    nu_228 = c/(227.8e-8)
    gamma = np.zeros_like(bf_cross)
    gamma_base = np.zeros_like(bf_cross)
    for n in [1, 2, 3, 4]:
        gamma_base = np.zeros_like(bf_cross)
        weight = n**2
        if (n > 1):
            nu_cutoff = (nu_grid > nu_0/n**2) & (nu_0/(n-1)**2 > nu_grid)
        else:
            nu_cutoff = (nu_grid > nu_912*Z**2) & (nu_912*(Z+1)**2 > nu_grid)

        gamma_base[nu_cutoff] = weight * Z**3 * 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid[nu_cutoff]*1e-8/c, -3) *np.power(me/(2*np.pi*kB*Ti) , 1.5)* bf_cross[nu_cutoff] * np.exp(h*(nu_0/n**2 - nu_grid[nu_cutoff])/(kB*Ti)) * (1-np.exp(-h*nu_grid[nu_cutoff]/(kB*Ti)))   #np.exp(np.divide(deltaE,-1*kB_Ryd*Ttest))[nu_cutoff] #np.exp(h*(nu_0/n**2 - nu_grid[nu_cutoff])/(kB*Ti)) #*(1-np.exp(-h*nu_grid[nu_cutoff]/(kB*Ti)))
        gamma += gamma_base
    
    return gamma






print('Starting radius shell looping')

for i, Ri in tqdm(enumerate(R)): #R[:-1])):
    
    Ti = T[i]
    nHi = nH[i]* Xfrac[i]
    nei = ne[i]#* Xfrac[i]
    nHei = nHe[i]*XfracHe[i]


    #nHei = nHe[i]*XfracHe[i]
    print('i ', i, 'Ri ', Ri, 'dR[i] ', dR[i], 'Ti ', Ti, 'nproti ',nHi, 'nei ', nei)

    #gamma, deltaE, gnu_2S_alt, col_idx, gamma_smooth, gamma_raw = extract_gamma(lambda_grid,np.log10(Ti))

    #gamma_smooth = interp1d(df_gamma_sorted["lambda"].values,df_gamma_sorted.iloc[col_idx].values, kind='cubic', bounds_error=False, fill_value=(0, df_gamma_sorted["g_nu"].values[-1]))(lambda_grid)
    #print('gamma ', gamma)
    #deltaE = 
    #print('deltaE/(kB*Ti) ', deltaE[0]/(kB_Ryd*Ti), 'resulting scaling ', np.exp(-1*deltaE[0]/(kB_Ryd*Ti)), 'T scaling? ', np.power(Ttest/Ti, 1.5))
    
    #gamma_H = np.exp(deltaE/(-1*kB_Ryd*Ti)) * gamma_raw_test3 # Correct for actual temperature of layer rather than test temperature #*3*10**(8)* np.power(lambda_grid, -1)*10**10
    nu_912 = c/(911.78e-8)
    gamma_H = gamma_bf_Hlike(1, Ti, bf_cross_alt3, nu_912)  
    #gamma_H_test2 = gamma_raw_test3
    #jnu_H = (1/np.sqrt(2))*nHi*nei/(4*np.pi)*gamma_H #*1/(10**34 * Ti**1.5*np.exp(deltaE/(kB*Ti))) * gamma
    jnu_H = (dR[i]**2/dR.iloc[-1]**2)**0 * nHi*nei/(4*np.pi) *gamma_H #*1/(10**34 * Ti**1.5*np.exp(deltaE/(kB*Ti))) * gamma 
   
    nu_228 = c/(227.8e-8)
    gamma_HeIII = gamma_bf_Hlike(2, Ti, bf_cross_alt3_He, nu_912) #np.exp(h*c*(1/(195e-8) - nu_grid)/(kB*Ti/Z**2)) * gamma_raw_test3 * Z**3  
    jnu_He = nHei*nei/(4*np.pi) *gamma_HeIII

    gff = 1.3#np.sqrt(3)/np.pi * (np.log(Ti**1.5*c/(lambda_grid*1e-8)) + 17.7) 
    jnu_Bremstrahlung_alt = (1/0.7)**0* nHi*nei/(4*np.pi) *2**5 * np.pi * e**6 /(3*me*c**3) *np.sqrt(2*np.pi/(3*me*kB*Ti)) * gff * np.exp(-h*nu_grid/(kB*Ti))#5.44e-41*nei*nHi*Ti**(-0.5)*gff*np.exp(-h*nu_grid/(kB*Ti))
    #jnu_Bremstrahlung = np.divide(nei*nHi*32*Z**2*e**4*h, 4*np.pi*3*me**2*c**3) * np.sqrt(np.divide(np.pi*h*nu_grid, 3*kB*Ti))*np.exp(-h*nu_grid/(kB*Ti)) * gff*3*10**(8)* np.power(lambda_grid*1e-8, -1)
    if (Ti > 15000):
        qp = 4.71e-4
    else:
        qp = 5.31e-4
    
    row_idx2S = np.abs(a2S['T'] - Ti).argmin()
    gamma_2SHe = a2S.iloc[row_idx2S, 1]* np.divide(1, 1 + (nHei+nei)*qp/(2*A2S))* gnu_2SHe
    gamma_2S = a2S.iloc[row_idx2S, 1]* np.divide(1, 1 + (nHi+nei)*qp/(2*A2S))* gnu_2S#*3*10**(8)* np.power(lambda_grid, -1)*10**10
    jnu_2SHe = nHei*nei/(4*np.pi) *gamma_2SHe
    jnu_2S = nHi*nei/(4*np.pi) *gamma_2S#* a2S.iloc[row_idx2S, 1]* np.divide(1, 1 + nHi*qp/A2S)* gnu_2S_smooth
    
    #jnu_He = nHei*nei/(4*np.pi) * gammaHe /(10**34 * Ti**1.5)
    #FluxL = 4*np.pi*jnu_H * dR[i]*3*10**(8)* np.power(lambda_grid, -2)*10**10 #np.abs(Ri - R[i+1]) *3*10**(8+10)* np.power(nu_grid, -2)
    #artificial2 = np.ones_like(n_levels)
    #artificial2[np.where(n_levels <2)] = 2
    FluxLH = (1/1.06)**0 * 4*np.pi*jnu_H *1/3*((R[i]+0.5*dR[i])**3-(R[i]-0.5*dR[i])**3)/R.iloc[-1]**2#*(1+np.exp(-h*c/(lambda_grid*1e-8*kB*Ti) )) #dR[i]*(Ri**2/R.iloc[-1]**2)   #((R[i]+dR[i])**3-R[i]**3) #dR[i]   #*(1+np.exp(-h*c/(lambda_grid*1e-8*kB*Ti) )) #*3*10**(8)* np.power(lambda_grid, -1)*10**10
    #FluxLH_test2 = dR[i] * jnu_H#nHi*nei/1 *gamma_raw_test3*np.exp(deltaE/(-1*kB_Ryd*Ti))
    #FluxLH[lambda_grid < 912] = (2)*FluxLH[lambda_grid < 912]

    FluxLHeIII = jnu_He * 1/3*((R[i]+0.5*dR[i])**3-(R[i]-0.5*dR[i])**3)/R.iloc[-1]**2#dR[i]*(Ri**2/R.iloc[-1]**2)

    FluxLB = (1/0.9/0.5/3.5)**0* 1/(2*np.pi)**0 *4*np.pi*jnu_Bremstrahlung_alt * 1/3*((R[i]+0.5*dR[i])**3-(R[i]-0.5*dR[i])**3)/R.iloc[-1]**2#dR[i]*(Ri**2/R.iloc[-1]**2)#*3*10**(8)* np.power(lambda_grid, -1)*10**10

    widthDoppler = np.sqrt(0.5*kB*Ti/(mH*c**2))*1216


    FluxL2SHe = (1/0.047)**0 *4*np.pi*jnu_2SHe * 1/3*((R[i]+0.5*dR[i])**3-(R[i]-0.5*dR[i])**3)/R.iloc[-1]**2#dR[i]*(Ri**2/R.iloc[-1]**2)
    FluxL2S = (1/0.90/0.97)**0 * 1/(2*np.pi)**0 *4*np.pi*jnu_2S * 1/3*((R[i]+0.5*dR[i])**3-(R[i]-0.5*dR[i])**3)/R.iloc[-1]**2#dR[i]*(Ri**2/R.iloc[-1]**2)#*3*10**(8)* np.power(lambda_grid, -1)*10**10 #-FluxL2S_comb * np.exp(-1*np.power(lambda_grid-1216 ,-2)*widthDoppler**2)*(R.iloc[-1]-Ri)*np.sqrt(np.pi)*e**2/(mH*c*widthDoppler)
    FluxLHe = 4*np.pi*jnu_He * dR[i]*3*10**(8)* np.power(lambda_grid, -1)*10**10*1/3*((R[i]+0.5*dR[i])**3-(R[i]-0.5*dR[i])**3)/R.iloc[-1]**2#(Ri**2/R.iloc[-1]**2)

    #print('fraction of total flux absorbed by LyA at this layer ', (np.exp(-1*np.power(lambda_grid-1216 ,-2)*widthDoppler**2)*(R.iloc[-1]-Ri)*np.sqrt(np.pi)*e**2/(mH*c*widthDoppler))[lambda_grid > 1220])

    #FluxL = np.add(FluxLH, FluxLB)
    FluxLH_comb = (n_levels/2)**0 *np.add(FluxLH_comb, FluxLH)#*np.power(lambda_grid*1e-8, -1)#*np.power(lambda_grid*1e-8/c, -1)#  *np.power(lambda_grid*1e-8, -1)
    FluxLHeIII_comb = np.add(FluxLHeIII_comb, FluxLHeIII)
    FluxLB_comb = np.add(FluxLB_comb, FluxLB)#*np.power(lambda_grid*1e-8, -1)# *np.power(lambda_grid*1e-8/c, -1)#*np.power(lambda_grid*1e-8, -1)
    FluxL2SHe_comb = np.add(FluxL2SHe_comb, FluxL2SHe)
    FluxL2S_comb = np.add(FluxL2S_comb, FluxL2S)#*np.power(lambda_grid*1e-8, -1)#*np.power(lambda_grid*1e-8/c, -1)#*np.power(lambda_grid*1e-8, -1)
    #FluxLH_comb_test2 = np.add(FluxLH_comb_test2, FluxLH_test2)
    FluxLH_arr.append(FluxLH)


FluxLH_arr = np.array(FluxLH_arr)

FluxL_tot = 1/(2*np.pi)**0 * np.add(np.add(FluxLH_comb, FluxL2S_comb), FluxLB_comb) #, FluxLHeIII_comb)
idxm_1100 = np.argmin(np.abs(xm - 1100))


#print('test4A ', ym_nebular_unscaled[0])
A = np.array(FluxL_tot)[np.argmin(np.abs(lambda_grid - 910))]
#    Eion=ddnp.power(911.25e-8/(h*c), -1)

#B = ym_nebular_unscaled.iloc[np.argmin(np.abs(lambda_grid - 910))] /(4*np.pi*R.iloc[-1]**2) *(lambda_grid*1e-8)**(0)#*6.6*1e-27 *(912/3000)
B = ym_nebular_unscaled.iloc[np.argmin(np.abs(lambda_grid - 910))] *(lambda_grid.iloc[np.argmin(np.abs(lambda_grid - 910))]*1e-8)**(0)
print

ratio = A/B #FluxL_comb.iloc[np.argmin(np.abs(lambda_grid - 3000))] /  ym_nebular_unscaled.iloc[np.argmin(np.abs(xm - 3000))]


print('')
print('ratio of continuum', ratio, 'ratio if cumulative ',(A+ np.exp(np.divide(h*c,kB*Ttest)*(1/3640 - 1/911)*10**8)*np.array(FluxL_tot)[np.argmin(np.abs(lambda_grid - 3640))]+np.exp(np.divide(h*c,kB*Ttest)*(1/8150 - 1/3640)*10**8)* np.array(FluxL_tot)[np.argmin(np.abs(lambda_grid - 8150))])/B)
print('ratio He n=1 ', np.array(FluxL_tot)[np.argmin(np.abs(lambda_grid - 225))] / ym_nebular_unscaled.iloc[np.argmin(np.abs(lambda_grid - 225))])

print('ratio of continuum at balmer ', np.array(FluxL_tot)[np.argmin(np.abs(lambda_grid - 3640))]/(ym_nebular_unscaled.iloc[np.argmin(np.abs(lambda_grid - 3640))]), 'ratio of balmer if cumulative ', (np.array(FluxL_tot)[np.argmin(np.abs(lambda_grid - 3640))] + np.exp(np.divide(h*c,kB*Ttest)*(1/8150 - 1/3640)*10**8)*np.array(FluxL_tot)[np.argmin(np.abs(lambda_grid - 8150))])/(ym_nebular_unscaled.iloc[np.argmin(np.abs(lambda_grid - 3640))]))

print('ratio of continuum at paschen ', np.array(FluxL_tot)[np.argmin(np.abs(lambda_grid - 8190))]/(ym_nebular_unscaled.iloc[np.argmin(np.abs(lambda_grid - 8190))]))
print('ratio of continuum at n=4 ', np.array(FluxL_tot)[np.argmin(np.abs(lambda_grid - 14500))]/(ym_nebular_unscaled.iloc[np.argmin(np.abs(lambda_grid - 14500))]))
print('ratio of continuum at n=5 ', np.array(FluxL_tot)[np.argmin(np.abs(lambda_grid - 22500))]/(ym_nebular_unscaled.iloc[np.argmin(np.abs(lambda_grid - 22500))]))
print('ratio of 2S He', np.array(FluxL2SHe_comb)[np.argmin(np.abs(lambda_grid - 310))]/(ym_nebular_unscaled.iloc[np.argmin(np.abs(lambda_grid - 310))]))

print('ratio of 2-photon', np.array(FluxL_tot)[np.argmin(np.abs(lambda_grid - 1400))]/(ym_nebular_unscaled.iloc[np.argmin(np.abs(lambda_grid - 1400))]))
print(r'\nu multiple ', np.power(lambda_grid*1e-8/c, -1).iloc[i1], 'lambda multiple ', lambda_grid.iloc[i1]*1e-8, '4pi ',  4*np.pi, 'Energy ', np.power(lambda_grid.iloc[i1]*1e-8/(h*c), -1), 'dR ', dR[0])

#print('ratio of Brems', ratio, '2pi ', 2*np.pi*ratio, '1/(2*np.pi) ', ratio/(2*np.pi))

#print('nuF_nu_analytic ', A, 'nuF_nu_data ', B  , 'nuF_nu_B ', FluxLB_comb.iloc[np.argmin(np.abs(lambda_grid - 3600))], 'nuF_nu_2S ', FluxL2S_comb.iloc[np.argmin(np.abs(lambda_grid - 3600))])
#plt.plot(lambda_grid, ym_nebular_unscaled/(4*np.pi*R.iloc[-1]**2)*np.power(lambda_grid*1e-8, 0) ,label=r'Combined nebular (CLOUDY)')
plt.plot(lambda_grid, ym_nebular_unscaled*np.power(lambda_grid*1e-8, 0) ,label=r'Nebular transmitted (CLOUDY)')
#plt.plot(xm.iloc[np.where(df.loc[:,'nLine'] == 0)], ym_nebular_nolines/(4*np.pi*R.iloc[-1]**2) ,label=r'Combined nebular transmission (CLOUDY)')
#data = np.load(f'tempcorrect.npz')#'savefiles/halo_allstars.npz')
#tempcorrect = data['tempcorrect']

plt.plot(lambda_grid , FluxL_tot *np.power(lambda_grid*1e-8/c, 0),label=r'Combined nebular (analytic)', linestyle='-', c='black')
plt.plot(lambda_grid , FluxLH_comb *np.power(lambda_grid*1e-8/c, 0),label=r'Free-bound (analytic)', linestyle='dashed',c='green')#, linestyle='dashed')
#plt.plot(lambda_grid , FluxLHeIII_comb *np.power(lambda_grid*1e-8/c, 0),label=r'Free-bound HeIII scaled Z', linestyle='dashed',c='red')
plt.plot(lambda_grid , FluxLB_comb *np.power(lambda_grid*1e-8/c, 0),label=r'Bremsstrahlung (analytic)', linestyle='dashed', c='purple')

#plt.plot(lambda_grid , FluxL2SHe_comb *np.power(lambda_grid*1e-8/c, 0),label=r'2-photon He fake', linestyle='dashed', c='orange')
plt.plot(lambda_grid , FluxL2S_comb *np.power(lambda_grid*1e-8/c, 0),label=r'2-photon (analytic)', linestyle='dashed', c='orange')
#plt.plot(lambda_grid , FluxL_tot *np.power(lambda_grid*1e-8/c, 0),label=r'Combined nebular (analytic)', linestyle='-', c='black')
#plt.plot(lambda_grid , FluxLH_comb_test2 *np.power(lambda_grid*1e-8/c, 0),label=r'Free-bound Katkov+78')

#plt.plot(lambda_grid[Ly_bump_ind], np.divide(ym_stellar_intr[Ly_bump_ind], ym_stellar_trans[Ly_bump_ind]), label=r'CLOUDY trans/fesc_star', linestyle='dotted', c='darkblue')

test_Flux = (1)*n_levels**(-3)*2**0 * ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911.78)]*np.power(lambda_grid*1e-8/c, 0)
test_Flux = test_Flux * np.exp( np.divide(h*c,kB*Ttest)*(1/911.78/n_levels**2 - np.power(lambda_grid, -1))*10**8)
#test_Flux.iloc[np.where(n_levels > 1.0)] = test_Flux.iloc[np.where(n_levels > 1.0)]#*2**(-0.5-0.5+0.25)

#test_Flux2 = n_levels**(-2)*2**-1 * ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911)]
#test_Flux2[np.where(n_levels > 1.0)] = test_Flux2[np.where(n_levels > 1.0)]*2**0.5

#test_Flux =  ym_nebular_unscaled/(4*np.pi*R*np.power(lambda_grid*1e-8/c, 0)
#plt.plot(lambda_grid[lambda_grid < 911.3] , ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911.3)]*np.exp( np.divide(h*c,kB*Ttest)*(1/911.3 - np.power(lambda_grid[lambda_grid < 911.3], -1))*10**8) ,label=r'Simple approximation of exponential', linestyle='dashed', c='red')
#plt.plot(lambda_grid , test_Flux /np.exp( np.divide(h*c,kB*Ttest)*(1/911/n_levels**2 - np.power(lambda_grid, -1))*10**8),label=r'Simple approximation of amplitude', linestyle='dashed', c='orange')
#plt.plot(lambda_grid , test_Flux2 ,label=r'Simple approximation 2.0', linestyle='dashed', c='orange')

#plt.plot(lambda_grid[lambda_grid < 911] ,  FluxLH_comb[np.argmax(lambda_grid < 911)]*np.exp( np.divide(h*c,kB*Ttest)*(1/911 - np.power(lambda_grid[lambda_grid < 911], -1))*10**8), linestyle='-', c='red', label=r'Boltzman exponential', lw=3)
print('CLOUDY flux at 911 ', 4*np.pi*dR[0]*5.53634e-19 * 912*1e-8/c)
print('Ratio of bump peaks, Balmer:', 0.5*ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911)]/ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 3640)], 'Paschen: ', 0.5*ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911)]/ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 8190)], 'n=4 ', 0.5*ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911)]/ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 14500)], 'n=5 ', 0.5*ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911)]/ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 22000)])


print('logN ratio of bump peaks, Balmer:', np.emath.logn(2, ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911)]/ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 3640)]), 'Paschen: ', np.emath.logn(3, ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911)]/ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 8190)]), 'n=4 ',np.emath.logn(4,  ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911)]/ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 14500)]), 'n=5 ', np.emath.logn(5, ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911)]/ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 22000)]))

print('ratio of simple approx to testpoints, n=2 ', test_Flux.iloc[np.argmax(lambda_grid < 3640)]/ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 3640)], 'n=3 ', test_Flux.iloc[np.argmax(lambda_grid < 8190)]/ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 8190)], 'n=4 ', test_Flux.iloc[np.argmax(lambda_grid < 14500)]/ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 14500)], 'n=5 ', test_Flux.iloc[np.argmax(lambda_grid < 22000)]/ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 22000)])

#plt.scatter([lambda_grid.iloc[np.argmax(lambda_grid < 911)], lambda_grid.iloc[np.argmax(lambda_grid < 3640)], lambda_grid.iloc[np.argmax(lambda_grid < 8190)], lambda_grid.iloc[np.argmax(lambda_grid < 14500)], 22000], [ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911)] , ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 3640)], ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 8190)], ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 14500)], ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 22000)]], label='testpoints', c='red')
#plt.scatter(911, 4*np.pi*5.53634e-19 * 912/c, label='CLOUDY emissivity', c='r')
plt.axvline(x=600,ls=':',alpha=0.7,color='#606060',lw=1.5)
plt.axvline(x=912,ls=':',alpha=0.7,color='#606060',lw=1.5)
#plt.axvline(x=1215,ls=':',alpha=0.7,color='#606060',lw=1.5)
#plt.axvline(x=3650,ls=':',alpha=0.7,color='#606060',lw=1.5)
#plt.text(540,np.max(ym_nebular_unscaled/(4*np.pi*R.iloc[-1]**2))*0.6,r'f$_{600}$',size=16,rotation=90,color='#606060')
#plt.text(855,np.max(ym_nebular_unscaled/(4*np.pi*R.iloc[-1]**2))*0.6,r'Ly_C',size=16,rotation=90,color='#606060')
#plt.text(1160,np.max(yinc/y_1100)*0.6,r'Ly${\alpha}$',size=16,rotation=90,color='#606060')
#plt.text(3650,np.max(yinc/y_1100)*0.6,r'H_C$',size=16,rotation=90,color='#606060')

plt.xlim(500, 10000)
#plt.ylim(1e-28,1e-15)
#plt.ylim(np.min(jnu_comb), np.max(ym_nebular))
#plt.xlim(100, 5000)
#plt.ylim(1*10**(-2), 10)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'F$_{\nu}$',size=16)
plt.legend(loc='lower center', fontsize=10)
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/analyticSED_v20_'+mod_name+'.jpg')
#plt.show()
plt.clf()






###########################################################
#                    Nebular Escape fractions             #
###########################################################

Ly_bump_ind = np.where((503.25 < xm)  & (xm < 911.78))[0]
Flux_ind_1 = np.where((503.25 < lambda_grid)  & (lambda_grid < 911.78))[0]

print('len(Ly_bump_ind) ', len(Ly_bump_ind), 'len(Flux_ind_1) ', len(Flux_ind_1))

# Initial cloudy output is probably nuL_nu, so we divide by nu and area to get F_nu

ym_stellar_intr = df.loc[:,'incident'].iloc[analytic_idx] * lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)#np.sum(df.loc[:,'incident'].iloc[analytic_idx][:, None] * lambda_grid[:, None]*1e-8/c / (4*np.pi*R[None, :]**2), axis= 1)

ym_trans = df.loc[:,'net trans'].iloc[analytic_idx] * lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)

ym_neb_trans = df.loc[:,'DiffOut'].iloc[analytic_idx] * lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)

ym_stellar_trans = df.loc[:,'trans'].iloc[analytic_idx] * lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)

ym_reflect = df.loc[:,'reflc'].iloc[analytic_idx] * lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)


data = np.load(f'tempcorrect_v2.npz')#'savefiles/halo_allstars.npz')
tempcorrect = data['tempcorrect']

FluxL_tot = FluxL_tot#*tempcorrect
FluxLH_comb = FluxLH_comb#*tempcorrect

LyC_neb_trans = -1*np.trapezoid(ym_neb_trans[Ly_bump_ind], c/(xm.iloc[Ly_bump_ind]*1e-8))

LyC_neb_intr = -1*np.trapezoid(FluxL_tot[Ly_bump_ind]*tempcorrect[Flux_ind_1], c/(xm.iloc[Ly_bump_ind]*1e-8))

LyC_star_intr = -1*np.trapezoid(ym_stellar_intr[Ly_bump_ind], c/(xm.iloc[Ly_bump_ind]*1e-8))

LyC_star_trans = -1*np.trapezoid(ym_stellar_trans[Ly_bump_ind], c/(xm.iloc[Ly_bump_ind]*1e-8))

LyC_reflect = -1*np.trapezoid(ym_reflect[Ly_bump_ind], c/(xm.iloc[Ly_bump_ind]*1e-8))

LyC_trans = -1*np.trapezoid(ym_trans[Ly_bump_ind], c/(xm.iloc[Ly_bump_ind]*1e-8))

LyC_neb_intr_arr = np.trapezoid(FluxLH_arr[:, Flux_ind_1]*tempcorrect[None, Flux_ind_1], c/(xm.iloc[Ly_bump_ind]*1e-8), axis=1)

LyC_intr = LyC_neb_intr + LyC_star_intr

LyC_neb_fesc = LyC_neb_trans/LyC_neb_intr

LyC_star_fesc = LyC_star_trans/LyC_star_intr

LyC_fesc = LyC_trans/LyC_intr


Q_star_intr = -1*np.trapezoid(ym_stellar_intr[Ly_bump_ind] * xm.iloc[Ly_bump_ind]*1e-8/(h*c), c/(xm.iloc[Ly_bump_ind]*1e-8)) 

Q_star_trans = -1*np.trapezoid(ym_stellar_trans[Ly_bump_ind] * xm.iloc[Ly_bump_ind]*1e-8/(h*c), c/(xm.iloc[Ly_bump_ind]*1e-8)) 

Q_neb_intr = -1*np.trapezoid(FluxL_tot[Ly_bump_ind]*tempcorrect[Flux_ind_1] * xm.iloc[Ly_bump_ind]*1e-8/(h*c), c/(xm.iloc[Ly_bump_ind]*1e-8))

Q_neb_trans = -1*np.trapezoid(ym_neb_trans[Ly_bump_ind] * xm.iloc[Ly_bump_ind]*1e-8/(h*c), c/(xm.iloc[Ly_bump_ind]*1e-8))

Q_neb_fesc = Q_neb_trans/Q_neb_intr

Q_star_fesc = Q_star_trans/Q_star_intr


print('LyC_trans ', LyC_trans, 'LyC_intr ', LyC_neb_intr+LyC_star_intr , 'LyC_fesc ', LyC_fesc)
print('LyC_star_trans ', LyC_star_trans, 'LyC_star_intr ', LyC_star_intr, 'fesc star ', LyC_star_fesc, 'Q fesc star ', Q_star_fesc)

print('LyC_neb_trans ', LyC_neb_trans, 'LyC_neb_intr ', LyC_neb_intr, 'fesc nebular ', LyC_neb_fesc, 'Q fesc nebular ', Q_neb_fesc)

print('escape fraction 500Ang ', np.divide(ym_neb_trans[Ly_bump_ind].iloc[-1], FluxL_tot[Ly_bump_ind].iloc[-1]))

#print('alt LyC_neb_trans ', LyC_trans-LyC_star_trans,'alt fesc nebular ', (LyC_trans-LyC_star_trans)/LyC_neb_intr)

#print('nebular LyC reflected ', LyC_reflect, 'reflect/Diffout ', LyC_reflect/LyC_neb_trans, 'reflect/star_intrinsic ', LyC_reflect/LyC_star_intr, 'reflect/neb_intrinsic ', LyC_reflect/LyC_neb_intr,'reflect/total transmission', LyC_reflect/(LyC_neb_intr+LyC_star_intr))


#plt.plot(lambda_grid[Ly_bump_ind], np.divide(ym_trans[Ly_bump_ind], ym_stellar_intr[Ly_bump_ind]+FluxLH_comb[Flux_ind_1])  ,label=r'Combined escape fraction')
plt.plot(xm.iloc[Ly_bump_ind], np.divide(ym_neb_trans[Ly_bump_ind], FluxL_tot[Ly_bump_ind]),c='purple'  ,label=r'Nebular escape fraction')
#np.savez_compressed(f'tempcorrect_v2.npz',tempcorrect=ym_neb_trans/FluxL_tot)

plt.plot(lambda_grid[Ly_bump_ind], np.divide(ym_stellar_trans[Ly_bump_ind], ym_stellar_intr[Ly_bump_ind]),c='orange',linestyle='-'  ,label=r'Stellar escape fraction')
#plt.plot(lambda_grid[Ly_bump_ind], np.exp((lambda_grid.iloc[-5] - lambda_grid[Ly_bump_ind])/912)*np.divide(ym_stellar_trans.iloc[-5], ym_stellar_intr.iloc[-5])  ,label=r'exponential')


#print('test1 ', ym_stellar_trans.iloc[-1], 'test2 ', ym_stellar_intr.iloc[-1], 'internal exp values ', (lambda_grid.iloc[-5] - lambda_grid[Ly_bump_ind])/912, 'seed value ', np.divide(ym_stellar_trans.iloc[-5], ym_stellar_intr.iloc[-5]),'last values of exp ', (np.exp(-1*(lambda_grid.iloc[-5] - lambda_grid[Ly_bump_ind]))*np.divide(ym_stellar_trans.iloc[-5], ym_stellar_intr.iloc[-5]))[-10:])

#plt.scatter(0,0, label=f'LyC stellar f_esc: {LyC_star_fesc:.2f}', c='black')
#plt.scatter(0,0, label=f'LyC nebular f_esc: {LyC_neb_fesc:.2f}', c='black')
#plt.scatter(0,0, label=f'LyC net f_esc: {LyC_fesc:.2f}', c='black')

plt.xlim(503, 911.8)
plt.ylim(0,1.0)
#plt.ylim(np.min(jnu_comb), np.max(ym_nebular))
#plt.xlim(100, 5000)
#plt.ylim(1*10**(-2), 10)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'Escape fraction',size=16)
plt.legend(loc='lower center', fontsize=10)
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/escapefrac_v11_'+mod_name+'.jpg')
#plt.show()
plt.clf()



plt.plot(xm.iloc[Ly_bump_ind], np.divide(ym_neb_trans[Ly_bump_ind], FluxL_tot[Ly_bump_ind]*tempcorrect[Flux_ind_1]),c='purple'  ,label=r'Nebular escape fraction')

plt.plot(lambda_grid[Ly_bump_ind], np.divide(ym_stellar_trans[Ly_bump_ind], ym_stellar_intr[Ly_bump_ind]),c='orange',linestyle='--'  ,label=r'Stellar escape fraction')

plt.scatter(0,0, label=f'LyC stellar f_esc: {LyC_star_fesc:.2f}', c='black')
plt.scatter(0,0, label=f'LyC nebular f_esc: {LyC_neb_fesc:.2f}', c='black')
plt.scatter(0,0, label=f'LyC net f_esc: {LyC_fesc:.2f}', c='black')

plt.xlim(503, 911.2)
plt.ylim(0,1.1)
#plt.ylim(np.min(jnu_comb), np.max(ym_nebular))
#plt.xlim(100, 5000)
#plt.ylim(1*10**(-2), 10)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'Escape fraction',size=16)
plt.legend(loc='upper center', fontsize=10)
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/escapefrac_corrected_'+mod_name+'.jpg')
#plt.show()
plt.clf()


#Fraction of transmitted that is from nebular

#plt.plot(lambda_grid[Ly_bump_ind], np.divide(ym_trans[Ly_bump_ind], ym_stellar_intr[Ly_bump_ind]+FluxLH_comb[Flux_ind_1])  ,label=r'Combined escape fraction')
plt.plot(lambda_grid[Ly_bump_ind], np.divide(ym_neb_trans[Ly_bump_ind], ym_stellar_trans[Ly_bump_ind]+ym_neb_trans[Ly_bump_ind]),c='purple', linestyle='--'  ,label=r'Nebular transmission contribution')
plt.plot(lambda_grid[Ly_bump_ind], np.divide(ym_stellar_trans[Ly_bump_ind], ym_stellar_trans[Ly_bump_ind]+ym_neb_trans[Ly_bump_ind]),c='orange', linestyle='--'  ,label=r'Stellar transmission contribution')
#plt.plot(lambda_grid[Ly_bump_ind], np.exp((lambda_grid.iloc[-5] - lambda_grid[Ly_bump_ind])/912)*np.divide(ym_stellar_trans.iloc[-5], ym_stellar_intr.iloc[-5])  ,label=r'exponential')



print('test1 ', ym_stellar_trans.iloc[-1], 'test2 ', ym_stellar_intr.iloc[-1], 'internal exp values ', (lambda_grid.iloc[-5] - lambda_grid[Ly_bump_ind])/912, 'seed value ', np.divide(ym_stellar_trans.iloc[-5], ym_stellar_intr.iloc[-5]),'last values of exp ', (np.exp(-1*(lambda_grid.iloc[-5] - lambda_grid[Ly_bump_ind]))*np.divide(ym_stellar_trans.iloc[-5], ym_stellar_intr.iloc[-5]))[-10:])

plt.xlim(503, 911.2)
plt.ylim(0.01,1)
#plt.ylim(np.min(jnu_comb), np.max(ym_nebular))
#plt.xlim(100, 5000)
#plt.ylim(1*10**(-2), 10)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'Transmission fraction',size=16)
plt.legend(loc='upper center', fontsize=10)
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/transfrac_v2_'+mod_name+'.jpg')
#plt.show()
plt.clf()




plt.plot(lambda_grid, ym_nebular_unscaled*np.power(lambda_grid*1e-8, 0) ,label=r'Nebular transmitted (CLOUDY)')

plt.plot(lambda_grid , FluxL_tot *np.power(lambda_grid*1e-8/c, 0),label=r'Combined nebular (analytic)', linestyle='-', c='black')
plt.plot(lambda_grid , FluxLH_comb *np.power(lambda_grid*1e-8/c, 0),label=r'Free-bound (analytic)', linestyle='dashed',c='green')#, linestyle='dashed')
plt.plot(lambda_grid , FluxLB_comb *np.power(lambda_grid*1e-8/c, 0),label=r'Bremsstrahlung (analytic)', linestyle='dashed', c='purple')

plt.plot(lambda_grid , FluxL2S_comb *np.power(lambda_grid*1e-8/c, 0),label=r'2-photon (analytic)', linestyle='dashed', c='orange')
plt.plot(lambda_grid[Ly_bump_ind], ym_nebular_unscaled[Ly_bump_ind]*np.divide(ym_stellar_intr[Ly_bump_ind], ym_stellar_trans[Ly_bump_ind]), label=r'CLOUDY trans/fesc_star', linestyle='dotted', c='red', lw=3)


plt.xlim(503, 911.2)
plt.ylim(1e-21,1e-14)
#plt.ylim(np.min(jnu_comb), np.max(ym_nebular))
#plt.xlim(100, 5000)
#plt.ylim(1*10**(-2), 10)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'F$_{\nu}$',size=16)
plt.legend(loc='lower center', fontsize=10)
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/Flux_analysis_v2_'+mod_name+'.jpg')
#plt.show()
plt.clf()

# How fesc seems to vary with dR

### nHend 18

# stellar(all)  : fesc = 10%, stellar(>504)  : fesc = 5.7%

# lowres lambda : fesc = 2% 


### nHend 15

# dR = flex: fesc = 35% 

# dR = 1e16: fesc = 36% 



###########################################################
#                    Balmer values                        #
###########################################################
print('')
print('Balmer outputs')
print('')

H_b_idx = np.argmin(np.abs(xm - 4870))
df_lines = np.array(pd.read_csv(f'{inputmodel}.Hlines', sep='\t'))
print('df_lines ', df_lines)

#Xedge = float(np.array(pd.read_csv(f'{inputmodel}.ele_H', sep='\t', comment='#').T)[2,-1]) 
#print('Xedge ', Xedge, 'dtype ')

#H_b_line_alt = np.sum(df_lines[1:, 2]*np.pow(df_lines[1:, 0],2)*4*np.pi* (df_lines[1:, 0]-df_lines[:-1, 0] ))*4863

#H_b_line = df.loc[:,'outlin'][H_b_idx] #np.divide(df.loc[:,'outlin'][H_b_idx], 4863)   #[np.argmin(np.abs(df.loc[:,'nu']-486))]#Mod.get_cont_y(cont ='outlin',unit='esA')[]


H_b_line = -1*np.trapezoid(ym_net[(4850 < xm) & (xm < 4870)], xm[(4850 < xm) & (xm < 4870)]) - ( ym_net[np.argmin(np.abs(xm - 4850))] * 20 +ym_net[np.argmin(np.abs(xm - 4870))] * 20)/2

print('H_b_line_absorp ', H_b_line, 'H_b_line_absorp_A ', -1*np.trapezoid(ym_net[(4850 < xm) & (xm < 4870)], xm[(4850 < xm) & (xm < 4870)]), 'H_b_line_absorp_B  ', ym_net[np.argmin(np.abs(xm - 4850))] * 20)

if (H_b_line > 0.0):
    H_b_line = df_lines[0, 2]
    print('df_lines.iloc[0:3] ', df_lines[0:3], 'H_b_line ', H_b_line )


H_b_cont = np.divide(df.loc[:,'net trans'][H_b_idx], 4870)
H_b_EW = H_b_line/H_b_cont 


data = np.load(f'tempcorrect.npz')#'savefiles/halo_allstars.npz')
tempcorrect = data['tempcorrect']


H_jump = (ym_net[np.argmin(np.abs(xm - 3640))] - ym_net[np.argmin(np.abs(xm - 4000))])/ym_net[np.argmin(np.abs(xm - 4000))]

#LyC_star_trans = np.trapezoid(ym_stellar[xm < 912], xm[xm < 912])
#LyC_star_int = np.trapezoid(ym_source[xm < 912], xm[xm < 912])
#LyC_neb_trans = np.trapezoid(ym_nebular[xm < 912], xm[xm < 912])
#LyC_neb_int = np.trapezoid(FluxL_tot[lambda_grid < 911.2] *tempcorrect[lambda_grid < 911.2], lambda_grid[lambda_grid < 911.2])
#LyC_int = LyC_star_int + LyC_neb_int
#LyC_net_trans = np.trapezoid(ym_net[xm < 912], xm[xm < 912])
#LyC_star_fesc = LyC_star_trans/LyC_star_int
#LyC_neb_fesc = LyC_neb_trans/LyC_neb_int
#LyC_net_fesc = LyC_net_trans/LyC_int

#print('Wavelength ', xm[H_b_idx] ,'index ',H_b_idx, 'Continuum flux ', H_b_cont, 'Line ID ', df.loc[:,'lineID'][H_b_idx], 'Line flux ', H_b_line, 'EW ', H_b_EW, 'H_jump ', H_jump)
#print('')
#print('LyC_fesc ', LyC_fesc, 'LyC_star_fesc ' ,LyC_star_fesc, 'LyC_net_trans ', LyC_net_trans, 'LyC_int ', LyC_int, 'LyC_neb_trans ', LyC_neb_trans, 'LyC_neb_int ', LyC_neb_int, 'FluxL_tot[lambda_grid < 912] ',FluxL_tot[lambda_grid < 911.2], 'tempcorrect[lambda_grid < 912] ', tempcorrect[lambda_grid < 911.2], )
#print('')


plt.figure(figsize=(8,6))

# against nH
endnH = [16, 16.5, 17, 17.5, 18, 18.5, 19]

Balmerjump = [0.21, 0.26, 0.44, 0.79, 1.09, 1.18, 1.2]

Hbeta_EW = [13, -1, 123, -1, 444, -1, 506]

fesc = [0.97, 0.9, 0.73, 0.4, 0.08, 0.0, 0.0]

fesc_star = [0.97, 0.90, 0.71, 0.36, 0.07, 0.0, 0.0]

fesc_neb = [0.99, 0.92, 0.8, 0.43, 0.07, 0.0, 0.0]

Transmissionboost = [1.02, 1.05, 1.16, 1.47, 1.63, 2.25, 0.0] #[1.01, 1.13, 1.44, 0.0, 0.0]

Stellar_transmission = [3.5e48, 3.2e48, 2.6e48, 1.3e48, 2.5e47, 6.25e45, 4.8e41] #[2.6e4, 1.9e4, 2.2e3, 0, 0]

Nebular_transmission = [5.1e46, 1.6e47, 4.2e47, 6.2e47, 1.6e47, 7.85e45, 2.0e44] #[3.1e2, 2.5e3, 9.7e2, 1e0, 0]

Nebular_intrinsic = [6.2e46, 1.7e47, 6.4e47, 1.4e48, 2.3e48, 2.6e48, 3.1e48]

Xedges = [1.0, 0.0, 1.0, 0.0, 1.0, 0.3, 0.0]


# For Tflex


endnH_Tf = [16, 16.5, 17, 17.5, 18, 18.5, 19]

Transmissionboost_Tf = [1.02, 1.06, 1.2, 1.6, 1.87, 2.4, 0] #[1.01, 1.13, 1.44, 0.0, 0.0]

fesc_star_Tf = [0.97, 0.9, 0.73, 0.36, 0.07, 0.0, 0.0]

fesc_neb_Tf = [0.99, 0.94, 0.8, 0.47, 0.08, 0.0, 0.0]

Nebular_transmission_Tf = [6.1e46, 1.9e47, 5.1e47, 8e47, 2.2e47, 8.9e45, 2.0e44] #[3.1e2, 2.5e3, 9.7e2, 1e0, 0]

Stellar_transmission_Tf = [3.5e48, 3.2e48, 2.6e48, 1.3e48, 2.5e47, 6.3e45, 5e41] #[2.6e4, 1.9e4, 2.2e3, 0, 0]



from scipy.optimize import curve_fit


def model_linear(x, a, b, c):
    return a * x**1 + b

def model_linexp(x, a, b, c):
    return (a**0  * x**1 + b - c*x)

def model_test(x, a, b):
    return np.log10((x**1*10**(b)*a)/(a+x)) #(x**1 + b - np.log10())

# Quadratic: y = a*x^2 + b*x + c
def model_quadratic(x, a, b, c):
    return a * x**2 + b * x + c

# Power Law: y = a * x^b + c
def model_power_law(x, a, b, c):
    return a * (x**(1*b)) + c

# Exponential Growth: y = a * exp(b * x) + c
def model_exp(x, a, b, c):
    return a * np.exp(-1*b * x) + c

# Saturating Exponential Rise: y = a * (1 - exp(-b * x)) + c
def model_saturating_exp(x, a, b, c):
    return a * (1.0 - np.exp(-b * x)) + c


def model_hill(x, x0, n, c):
    return 1.0 / (1.0 + (x / x0)**n)+c


def model_stretched_exp(x, x0, n, c):
    return np.exp(-(x / x0)**n)+c

endnH_temp = [14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19]#, 17.5, 18, 18.5, 19]
Nebular_transmission_temp = [5.07e44, 1.6e45, 5.1e45, 1.6e46, 5.1e46, 1.6e47, 4.2e47, 6.2e47, 1.6e47, 7.85e45, 2.0e44]#np.log10([5.07e44, 1.6e45, 5.1e45, 1.6e46, 5.1e46, 1.6e47, 4.2e47])#, 6.2e47, 1.6e47, 7.85e45, 2.0e44] #[3.1e2, 2.5e3, 9.7e2, 1e0, 0]

nH_dense = np.linspace(14, 19, 100)


popt, _ = curve_fit(model_linexp, endnH_temp, np.log10(Nebular_transmission_temp), p0=[1.0, 31, 1.0])
endnH_temp2 = [14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19]#, 19.5, 20]
Nebular_intrinsic_temp2 = [5.07e44, 1.6e45, 5.09e45, 1.63e46, 6.21e46, 1.71e47, 6.4e47, 1.4e48, 2.3e48, 2.6e48, 3.1e48]#, 3.1e48, 3.1e48]
popt2, _ = curve_fit(model_test, np.power(10, endnH_temp2), np.log10(Nebular_intrinsic_temp2), p0=[1e17, 30.8],  maxfev=10000)


sc = plt.scatter(endnH, Balmerjump, label='', c=fesc, cmap='managua')
cbar = plt.colorbar(sc, pad=0.03)
cbar.set_label(r'Escape fraction', size=15)

#plt.xlim(800, 5000)
plt.ylim(-1, 1)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r'Column density of neutral gas cm^-2 ',size=16)
plt.ylabel(r'Balmer Jump',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/Balmerjump-endnH_fesc_v4.jpg')
#plt.show()
plt.clf()



sc = plt.scatter(endnH, Transmissionboost, label='T=10000 K')
sc = plt.scatter(endnH_Tf, Transmissionboost_Tf, label='non-fixed T')
cbar = plt.colorbar(sc, pad=0.03)
cbar.set_label(r'Escape fraction', size=15)

#plt.xlim(800, 5000)
plt.ylim(1, )
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r'log Column density of neutral gas cm^-2 ',size=16)
plt.ylabel(r'Net photon transmitted / stellar transmitted',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/Transmissionboost-endnH_v3.jpg')
#plt.show()
plt.clf()



#sc = plt.scatter(endnH_Tf, Stellar_transmission_Tf, c='orange', marker='x', label='Stellar LyC, flexible T')
sc = plt.scatter(endnH_temp[:-3], Nebular_transmission_temp[:-3], c='purple')
sc = plt.scatter(endnH_Tf, Nebular_transmission_Tf, c='purple', marker='x',label='Nebular LyC, flexible T')

sc = plt.scatter(endnH, Stellar_transmission, c='orange', label='Stellar LyC')
sc = plt.scatter(endnH, Nebular_transmission, c='purple', label='Nebular LyC, T=10000 K')

sc = plt.plot(nH_dense,  np.power(10, model_linexp(nH_dense, *popt)), c='green', linestyle='--', label=f'Best fit var:a={popt[0]:.1f}, b={popt[1]:.1f},c={popt[2]:.3f},')
#cbar = plt.colorbar(sc, pad=0.03)
#cbar.set_label(r'Escape fraction', size=15)

#plt.xlim(800, 5000)
#plt.ylim(3.6e45, 3.6e48)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'log Column density of neutral gas cm^-2 ',size=16)
plt.ylabel(r'Transmitted photon rate [$s^{-1}$]',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/TransmittedQ-endnH_v5.jpg')
#plt.show()
plt.clf()


# Intrinsic nebular

#sc = plt.scatter(endnH_Tf, Stellar_transmission_Tf, c='orange', marker='x', label='Stellar LyC, flexible T')
#sc = plt.scatter(endnH_temp[:-7], Nebular_transmission_temp[:-7], c='purple')
#sc = plt.scatter(endnH_Tf, Nebular_transmission_Tf, c='purple', marker='x',label='Nebular LyC, flexible T')

#sc = plt.scatter(endnH, Stellar_transmission, c='orange', label='Stellar LyC')
sc = plt.scatter(endnH_temp2, Nebular_intrinsic_temp2, c='purple', label='Nebular LyC, T=10000 K')

sc = plt.plot(nH_dense,  np.power(10, model_test(np.power(10, nH_dense), *popt2)), c='green', linestyle='--', label=f'Best fit var:a={popt2[0]:.2e}, b={popt2[1]:.2f}')
#cbar = plt.colorbar(sc, pad=0.03)
#cbar.set_label(r'Escape fraction', size=15)

#plt.xlim(800, 5000)
#plt.ylim(3.6e45, 3.6e48)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'log Column density of neutral gas cm^-2 ',size=16)
plt.ylabel(r'Transmitted photon rate [$s^{-1}$]',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/IntrinsicQ-endnH_v5.jpg')
#plt.show()
plt.clf()





# Escape fraction



#sc = plt.scatter(endnH_Tf, Nebular_transmission_Tf, c='purple', marker='x',label='Nebular LyC, flexible T')

sc = plt.scatter(endnH, fesc_star_Tf, c='orange', marker='.', label='Stellar fesc')
sc = plt.scatter(endnH, fesc_neb, c='purple', marker='.', label='Nebular fesc, T=10000 K')
sc = plt.scatter(endnH, fesc_neb_Tf, c='purple', marker='x', label='Nebular fesc, flexible T')

#cbar = plt.colorbar(sc, pad=0.03)
#cbar.set_label(r'Escape fraction', size=15)

#plt.xlim(800, 5000)
plt.ylim(0.01, 1)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'log Column density of neutral gas cm^-2 ',size=16)
plt.ylabel(r'Escape fraction',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/Fesc-endnH_v3.jpg')
#plt.show()
plt.clf()









fig, ax  = fig, ax = plt.subplots(figsize=(8, 6))#plt.subplot((8,6))

# against age
age = [1, 5, 10, 100, 1000]

Balmerjump = [0.6, 0.18,0.05, -0.38, -0.37]

Hbeta_EW = [349, 94, 33, -8, -7]

fesc = [0.42, 0.2, 0.0, 0.0, 0.0]

Xedges = [ 0.996, 0.00268, 0.00272, 0.00302, 0.0]

sc = ax.scatter(age, Balmerjump, label='', c=fesc, cmap='managua', vmin=0, vmax=1)
cbar = fig.colorbar(sc, ax=ax, pad=0.03) 
cbar.set_label(r'Escape fraction', size=15)


#ax.xlim(800, 5000)
ax.set_ylim(-1, 1)
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xlabel(r'Age [Myr]',size=16)
ax.set_ylabel(r'Balmer Jump',size=16)
ax.legend()
fig.tight_layout()
fig.savefig(f'../cloudy_plots/Balmerjump-age_fesc_v3.jpg')
#plt.show()
fig.clf()



fig, ax  = fig, ax = plt.subplots(figsize=(8, 6))#plt.subplot((8,6))

sc = ax.scatter(Hbeta_EW[0], Balmerjump[0], label='1   Myr', c=fesc[0], cmap='managua', vmin=0, vmax=1)
sc = ax.scatter(Hbeta_EW[1], Balmerjump[1], label='5   Myr', c=fesc[1], cmap='managua', vmin=0, vmax=1)
sc = ax.scatter(Hbeta_EW[2], Balmerjump[2], label='10  Myr', c=fesc[2], cmap='managua', vmin=0, vmax=1)
sc = ax.scatter(Hbeta_EW[3], Balmerjump[3], label='100 Myr', c=fesc[3], cmap='managua', vmin=0, vmax=1)
#sc = ax.scatter(Hbeta_EW[4], Balmerjump[4], label='1000 Myr', c=fesc[4], cmap='managua', vmin=0, vmax=1)

cbar = fig.colorbar(sc, ax=ax, pad=0.03) 
cbar.set_label(r'Escape fraction', size=15)


ax.set_xlim(-50, 350)
ax.set_ylim(-1, 1)
#ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xlabel(r'H_beta EW [Ang]',size=16)
ax.set_ylabel(r'Balmer Jump',size=16)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(f'../cloudy_plots/Balmerjump-Hbeta_fesc_v3.jpg')
#plt.show()
fig.clf()


###########################################################
#                      MANUAL SPEC PLOT                   #
###########################################################


print('current wavelength ratio ', xm.iloc[0]/x[0], 'energy ratio ', ym_source.iloc[0]/yinc[0])

idxm_1100 = np.argmin(np.abs(xm - 4860))
ym_1100 = ym_net[idxm_1100]**0


#print('max lambda F_lambda/F1100 ', np.max(ym_twophoton/ym_1100), 'shape twophoton ', len(ym_twophoton), 'len(ym_source) ', len(ym_source), 'len xm ', len(xm))

plt.figure(figsize=(7,5))

plt.plot(xm, ym_source/ym_1100 * (r[-1]/(10.7*6.96e11))**2/(np.pi)**0 ,label=r'TLUSTY intrinsic',ls=':', c='grey')
plt.plot(xm, ym_net/ym_1100 * (r[-1]/(10.7*6.96e11))**2/(np.pi)**0,label=r'Stellar+Nebular transmission', lw=3,c='black')
plt.plot(xm, ym_stellar/ym_1100 * (r[-1]/(10.7*6.96e11))**2/(np.pi)**0,label=r'Stellar transmission', ls='--', c='orange')
plt.plot(xm, ym_nebular/ym_1100 * (r[-1]/(10.7*6.96e11))**2/(np.pi)**0,label=r'Nebular transmission', ls='--', c='purple')
#plt.plot(xm,ym_twophoton/ym_1100,label=r'2-photon', ls='--', c='green')

# Stellar+Nebular transmission through an extra neutral column, photoelectric absorption only: x exp(-N_HI sigma_bf)
#nu_H = 911.8 / np.asarray(xm) / 0.99946     # Cloudy's nu [Ryd] (xm was made with 911.8) over the H I edge (0.99946 Ryd)
N_HI_log = 17.8
sigma_bf = np.where(911.8 >= lambda_grid, bf_cross_alt3, 0)   # [cm^2]
#for N_log, col in zip(N_HI_ABS, ['tab:blue', 'tab:cyan', 'tab:green', 'tab:olive']):
plt.plot(lambda_grid[lambda_grid < 911.8],  np.asarray(ym_net)[analytic_idx][lambda_grid < 911.8] * np.exp(-10**N_HI_log * sigma_bf)[lambda_grid < 911.8] * (r[-1]/(10.7*6.96e11))**2,   # sigma_bf is on lambda_grid = xm[analytic_idx]
             label=rf'$\times\ e^{{-N_{{\rm HI}}\sigma}}$, $N_{{\rm HI}}=10^{{{N_HI_log:g}}}$', lw=1.5, c='cyan')

#plt.scatter(0,0, label=f'H_beta EW: {H_b_EW:.1f}', c='black')

#plt.scatter(0,0, label=f'X_ion edge: {Xedge:.1f}', c='black')
#plt.scatter(0,0, label=f'Balmer jump: {H_jump:.2f}', c='black')
#plt.scatter(0,0, label=f'Q stellar f_esc: {Q_star_fesc:.2f}', c='black')
#plt.scatter(0,0, label=f'Q nebular f_esc: {Q_neb_fesc:.2f}', c='black')
#plt.scatter(0,0, label=f'LyC net f_esc: {LyC_fesc:.2f}', c='black')
#plt.scatter(0,0, label=f'Nebular intrinsic Q: {-1*Q_neb_intr*4*np.pi*R.iloc[-1]**2:.2e} /s', c='black')
#plt.scatter(0,0, label=f'Net Q transmitted/stellar: {(Q_star_trans+Q_neb_trans)/Q_star_trans:.2f}', c='black')
#plt.scatter(0,0, label=f'Nebular transmitted Q: {-1*Q_neb_trans*4*np.pi*R.iloc[-1]**2:.2e} /s', c='black')
#plt.scatter(0,0, label=f'Stellar transmitted Q: {-1*Q_star_trans*4*np.pi*R.iloc[-1]**2:.2e} /s', c='black')

#plt.axvline(x=600,ls=':',alpha=0.7,color='#606060',lw=1.5)
plt.axvline(x=912,ls=':',alpha=0.7,color='#606060',lw=1.5)
plt.axvline(x=1215,ls=':',alpha=0.7,color='#606060',lw=1.5)
#plt.axvline(x=3650,ls=':',alpha=0.7,color='#606060',lw=1.5)
#plt.text(540,np.max(yinc/y_1100)*0.6,r'f$_{600}$',size=16,rotation=90,color='#606060')
plt.text(855,np.max(yinc/y_1100)*0.6,r'Ly_C',size=16,rotation=90,color='#606060')
plt.text(1160,np.max(yinc/y_1100)*0.6,r'Ly${\alpha}$',size=16,rotation=90,color='#606060')
#plt.text(4863,np.max(yinc/y_1100)*0.6,r'H${\beta}$',size=16,rotation=90,color='#606060')
#plt.text(3650,np.max(yinc/y_1100)*0.6,r'H_C',size=16,rotation=90,color='#606060')
plt.xlim(500, 1300)#plt.xlim(100, 2000)
plt.ylim(1e-12, 1e-6)
#plt.xscale('log')
plt.yscale('log')
plt.title(f'{inputmodel}')
plt.xlabel(r'$\lambda$ [$\AA$]',size=14)
plt.ylabel(r'F$_{\lambda}\,$/$\, \pi \,$ [erg$\,s^{-1}cm^{-2}\AA^{-1}$]',size=14)
plt.legend(loc='lower right', fontsize=10)
#plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/manualSED_'+mod_name+'.png')
#plt.show()
plt.clf()





###########################################################
#                    BUMP AREAS                 #
###########################################################

Ly_bump_ind = np.where((600 < xm)  & (xm < 905))
HA_ind = np.where((1250 < xm)  & (xm < 3644))

Ly_bump_peak = ym_nebular.iloc[np.argmin(np.abs(xm - 905))]
H_bump_peak = ym_nebular.iloc[np.argmin(np.abs(xm - 3640))]

Ly_bump_int = -1*np.trapezoid(ym_nebular.iloc[Ly_bump_ind], c/(xm.iloc[Ly_bump_ind]*1e-8))
H_bump_int = -1*np.trapezoid(ym_nebular.iloc[HA_ind], c/(xm.iloc[HA_ind]*1e-8))

print('Ly_bump_int ', Ly_bump_int,'Ly_bump sum*dx/y_1100 ', -1*np.sum(ym_nebular.iloc[Ly_bump_ind]*(xm.iloc[Ly_bump_ind] - np.roll(xm.iloc[Ly_bump_ind], 1)))/ym_1100, 'LyA[0] ', ym_nebular.iloc[0]/ym_1100, 'LyA[-1] ', ym_nebular.iloc[-1]/ym_1100, 'H_bump_int ', H_bump_int)



Ly_jump = ym_nebular.iloc[np.argmin(np.abs(xm - 920))]/Ly_bump_peak
H_jump = H_bump_peak/ym_nebular.iloc[np.argmin(np.abs(xm - 3925))]

print('Ly_bump_peak ', Ly_bump_peak, 'H_bump_peak ', H_bump_peak, 'ratio ', Ly_bump_peak/H_bump_peak, 'ratio Ly_bump_Area/H_bump_peak ', Ly_bump_int/H_bump_peak, 'Ly_bump_int/H_jump ', Ly_bump_int/H_jump, 'Ly_jump ', Ly_jump, 'H_jump ', H_jump ,'Jump ratios ', Ly_jump/H_jump, 'Area ratios ', Ly_bump_int/H_bump_int, 'bump_peak_ratio/Area_ratio ', (Ly_bump_peak/H_bump_peak)/(Ly_bump_int/H_bump_int))

#LyA_strength = 


ratios = [0.00051, 0.03274, 0.2486,0.5074, 0.7160313786232616]

ratios_1 = [11040, 11371, 13432, 14838, 16012] 
ratios_2 = [23.3, 39.9, 40.4, 39.8, 39.8] 
ratios_3 = [602, 8.27, 0.93, 0.39, 0.24]
ratios_4 = [1.963e-05, 0.000756, 0.0058, 0.01215, 0.0172]

Temperatures = [1*10**4, 2.5*10**4, 5*10**4, 7.5*10**4, 10*10**4]

ratios_d = [0.249, 0.2487, 0.2473, 0.2469, 0.2486]

density = [10**1, 10**1.5, 10**2, 10**2.5, 10**3]

plt.plot(Temperatures,ratios ,label=r'density=3, cutoff=18.5')

#plt.xlim(3600, 4000)
#plt.ylim(10**(-1), 10**1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'BB T [K]',size=16)
plt.ylabel(r'Ly_bump_peak / H_bump_peak',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/BumpRatio-T'+'.jpg')
#plt.show()
plt.clf()


plt.plot(Temperatures,ratios_1 ,label=r'density=3, cutoff=18.5')

#plt.xlim(3600, 4000)
#plt.ylim(10**(-1), 10**1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'BB T [K]',size=16)
plt.ylabel(r'Ly_Area / H_bump_peak',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/AreaBumpRatio-T'+'.jpg')
#plt.show()
plt.clf()


plt.plot(Temperatures,ratios_2 ,label=r'density=3, cutoff=18.5')

#plt.xlim(3600, 4000)
#plt.ylim(10**(-1), 10**1)
plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r'BB T [K]',size=16)
plt.ylabel(r'Bump_peak_ratio / Bump_area_ratio',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/peakAreaRatio-Ratio-T'+'.jpg')
#plt.show()
plt.clf()


plt.plot(Temperatures,ratios_3 ,label=r'density=3, cutoff=18.5')

#plt.xlim(3600, 4000)
#plt.ylim(10**(-1), 10**1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'BB T [K]',size=16)
plt.ylabel(r'Ly_Jump / H_Jump',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/JumpRatio-T'+'.jpg')
#plt.show()
plt.clf()


plt.plot(Temperatures,ratios_4 ,label=r'density=3, cutoff=18.5')

#plt.xlim(3600, 4000)
#plt.ylim(0, 6)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'BB T [K]',size=16)
plt.ylabel(r'Ly_bump_Area / H_bump_Area',size=16)
plt.legend()
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/AreaRatio-T'+'.jpg')
#plt.show()
plt.clf()



# Try to approximate nebular emission with analytic solution + correction


#T = df_phy['Te']
#nH = df_phy['n(H)']
#ne = df_phy['n(e)']
#nHe = 10**(-1.0845) * nH#df_He['He+2']

#Xfrac = df_H['H+']
#XfracHe = df_He['He+2']
#R = df_rad['radius']
#dR = df_rad['dr']

# Averaged values:
Vtot_alt = 4*np.pi/3 * (R.iloc[-1]**3 - R.iloc[0]**3)
dV = 4*np.pi/3 * ((R[:]+0.5*dR[:])**3 - (R[:]-0.5*dR[:])**3) 
Vtot = np.sum(dV[:])
print('Vtot ', Vtot, 'Vtot_alt ', Vtot_alt)

nH_avg = np.sum(nH[:]*dV[:])/Vtot
Xion_avg = np.sum(Xfrac[:]*nH[:]*dV[:]/(nH_avg*Vtot))

mu = 1/(1+Xfrac[:])

mu_avg_alt = np.sum(mu[:]*dV[:]/Vtot) 

mu_avg = 1/(1+Xion_avg) 

print('mu_avg ', mu_avg, 'mu_avg_alt ', mu_avg_alt)

n_particles = nH * (1+Xfrac[:])*dV[:] #nH * (2*Xfrac + (1-Xfrac))

n_avg = np.sum(n_particles/Vtot)

Tavg = 1/np.sum(n_particles) * np.sum(T[:]*n_particles[:])


print('Tavg ', Tavg, 'n_avg ', n_avg, 'Xion_avg ', Xion_avg)

#Xion = 0.2
T1=10000
T2=100
f = 2*Xion_avg/((2*Xion_avg+(1-Xion_avg)*T2/T1)) 
#hden_ion = 21.6
Lboost = 2*Xion_avg/((2*Xion_avg+(1-Xion_avg)*1e2/1e4))*np.sqrt((1 + Xion_avg)/(2*Xion_avg+(1-Xion_avg)*1e2/1e4))
Lboost2 = 2*Xion_avg/((2*Xion_avg+(1-Xion_avg)*1e4/3e4))*np.sqrt((1 + Xion_avg)/(2*Xion_avg+(1-Xion_avg)*1e4/3e4))

print('Predicted total boost ', Lboost, 'f ', f, 'More reasonable prediction ', Lboost2)

Ly_bump_ind = np.where((503.25 < xm)  & (xm < 911.78))[0]
Flux_ind_1 = np.where((503.25 < lambda_grid)  & (lambda_grid < 911.78))[0]
F_neb_act = 1*np.trapezoid(FluxL_tot[Ly_bump_ind], c/(xm.iloc[Ly_bump_ind]*1e-8))

F_neb_avg = np.trapezoid((nH_avg*Xion_avg)**2 /(4*np.pi) *gamma_bf_Hlike(1, Tavg, bf_cross_alt3, nu_912)[Flux_ind_1] *Vtot/R.iloc[-1]**2, c/(xm.iloc[Ly_bump_ind]*1e-8)) 


print('Actual boost ', F_neb_act/F_neb_avg, 'F_neb_act ', F_neb_act, 'F_neb_avg ', F_neb_avg)




plt.plot(R[:], LyC_neb_intr_arr ,c='purple', label='Local nebular emission')
plt.plot(R[:], HII ,c='red', label='Ionisation fraction')

plt.scatter(0,0,s=0.1, color='white', label=f'Fraction from partially ionised:{np.sum(LyC_neb_intr_arr[HII<0.99])/np.sum(LyC_neb_intr_arr):.2f}')
#plt.xlim(503, 911.2)
plt.ylim(1e-7,1)
#plt.ylim(np.min(jnu_comb), np.max(ym_nebular))
#plt.xlim(100, 5000)
#plt.ylim(1*10**(-2), 10)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Radius [cm]',size=16)
plt.ylabel(r'Flux [erg/s/cm^2] ',size=16)
plt.legend(loc='upper center', fontsize=10)
plt.tight_layout()
plt.savefig(f'../cloudy_plots/{inputmodel}/LyC_intr-R_v2'+mod_name+'.png')
#plt.show()
plt.clf()



'''

### Try to estimate gamma based on calculated fluxes


#Ly_bump_indices = np.where((600 < lambda_grid)  & (lambda_grid < 911))
#HA_ind = np.where((1250 < xm)  & (xm < 3644))

#Ly_bump_peak_idx = np.argmax(lambda_grid < 911)
#H_bump_peak_idx = np.argmax(lambda_grid < 3645)
#Ly_bump_peak_S =  4*np.pi*(FluxL_tot*np.power(lambda_grid.iloc[Ly_bump_peak_idx]*1e-8/c, -1))[Ly_bump_peak_idx]

print('')
#print('Est_Ly_bump_peak_C ', Est_Ly_bump_peak_C, 'Ly_bump_peak_C ', Ly_bump_peak_C,'ratio peaks ', Est_Ly_bump_peak_C/Ly_bump_peak_C ,'Est_Ly_bump_int_C ', Est_Ly_bump_int_C,'Ly_bump_int_C ', Ly_bump_int_C, 'ratio areas ', Est_Ly_bump_int_C/Ly_bump_int_C)
#print('Est_Ly_bump_peak_C ', Est_Ly_bump_peak_C,'Ly_bump_peak_S ', Ly_bump_peak_S, 'Ly_bump_peak_C ', Ly_bump_peak_C,'ratio Analytic peak ', Ly_bump_peak_S/Ly_bump_peak_C,'ratio Estimate peak ', Est_Ly_bump_peak_C/Ly_bump_peak_C ,'Est_Ly_bump_int_C ', Est_Ly_bump_int_C,'Ly_bump_int_C ', Ly_bump_int_C, 'ratio areas ', Est_Ly_bump_int_C/Ly_bump_int_C)



### Investigate how fluctuations in T affect curve area 

Est_Ly_bump_int_C_plusT = 1*np.trapezoid(Ly_bump_peak_C * np.exp(h*c*( 1/(910e-8) - np.divide(1, lambda_grid.iloc[Ly_bump_indices]*1e-8))/(1*kB*Ttest*(1.1)))*np.power(lambda_grid.iloc[Ly_bump_indices]*1e-8/c, 0), np.power(lambda_grid.iloc[Ly_bump_indices]*1e-8/c, -1))

Est_Ly_bump_int_C_minusT = 1*np.trapezoid(Ly_bump_peak_C * np.exp(h*c*( 1/(910e-8) - np.divide(1, lambda_grid.iloc[Ly_bump_indices]*1e-8))/(1*kB*Ttest*(0.9)))*np.power(lambda_grid.iloc[Ly_bump_indices]*1e-8/c, 0), np.power(lambda_grid.iloc[Ly_bump_indices]*1e-8/c, -1))

print('10% uncertainty in T becomes upwards uncertainty of', (Est_Ly_bump_int_C_plusT-Est_Ly_bump_int_C)/Est_Ly_bump_int_C , 'or downwards uncertainty of ', (Est_Ly_bump_int_C_minusT-Est_Ly_bump_int_C)/Est_Ly_bump_int_C)

alphaS = np.trapezoid(FluxLH_comb[lambda_grid < 22000] *4*np.pi* lambda_grid[lambda_grid < 22000]*1e-8/c /(4*np.pi*1e34*  10*1e15*100*100)  *np.power(lambda_grid[lambda_grid < 22000]*1e-8/(c*h), 1) , np.power(lambda_grid[lambda_grid < 22000]*1e-8/c, -1))

print('alphaS ', alphaS)





Ly_bump_peak_S =  FluxLH_comb[Ly_bump_peak_idx]
H_bump_peak_S = FluxLH_comb[H_bump_peak_idx]
Ly_bump_peak_Stot  = FluxL_tot.iloc[Ly_bump_peak_idx]
H_bump_peak_Stot = FluxL_tot.iloc[H_bump_peak_idx]

#print('Ly_bump_indices ', Ly_bump_indices)

Ly_bump_int_S = -1*np.trapezoid(FluxLH_comb[Ly_bump_indices], lambda_grid.iloc[Ly_bump_indices])
Ly_bump_int_Stot = -1*np.trapezoid(FluxL_tot.iloc[Ly_bump_indices], lambda_grid.iloc[Ly_bump_indices])

Est_Ly_bump_peak_S = H_bump_peak_S*np.divide(lambda_grid.iloc[np.argmax(lambda_grid < 3645)], lambda_grid.iloc[np.argmax(lambda_grid < 911)]) 
Est_Ly_bump_int_S = -1*np.trapezoid(Est_Ly_bump_peak_S * np.exp(h*c*(1/911e-8 -np.divide(1, lambda_grid.iloc[Ly_bump_indices]*1e-8))/(-1*kB*Ttest)), lambda_grid.iloc[Ly_bump_indices]) 

Est_Ly_bump_peak_Stot = H_bump_peak_Stot*np.divide(lambda_grid.iloc[np.argmax(lambda_grid < 3645)], lambda_grid.iloc[np.argmax(lambda_grid < 911)]) 
Est_Ly_bump_int_Stot = -1*np.trapezoid(Est_Ly_bump_peak_Stot * np.exp(h*c*(1/911e-8 -np.divide(1, lambda_grid.iloc[Ly_bump_indices]*1e-8))/(-1*kB*Ttest)), lambda_grid.iloc[Ly_bump_indices]) 

print('Est_Ly_bump_peak_S ', Est_Ly_bump_peak_S, 'Ly_bump_peak_S ', Ly_bump_peak_S, 'Est_Ly_bump_int_S ', Est_Ly_bump_int_S,'Ly_bump_int_S ', Ly_bump_int_S)

print('Est_Ly_bump_peak_Stot ', Est_Ly_bump_peak_Stot, 'Ly_bump_peak_S ', Ly_bump_peak_Stot, 'Est_Ly_bump_int_S ', Est_Ly_bump_int_Stot,'Ly_bump_int_S ', Ly_bump_int_Stot)
print('')
'''

