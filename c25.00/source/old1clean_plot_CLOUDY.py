import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import scienceplots
#import os
import pyCloudy as pc
import pandas as pd
from IPython.display import display
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
inputmodel = 'simplemodel_script'
full_model_name = 'simplemodel_script'
#full_model_name = 'grid000000000_gridT_script'
mod_name = full_model_name
Mod = pc.CloudyModel(inputmodel)
Mod.print_stats()

###########################################################
#                      SPECTRA PLOT from pycloudy                       #
###########################################################
x = Mod.get_cont_x(unit='Ang')
yinc = Mod.get_cont_y(cont ='incid',unit='esA') 
ydiffout = Mod.get_cont_y(cont ='diffout',unit='esA')
yntrans = Mod.get_cont_y(cont ='ntrans',unit='esA')
yreflec = Mod.get_cont_y(cont ='reflec',unit='esA')

idx_1100 = np.argmin(np.abs(x - 1100))
y_1100 = yntrans[idx_1100]
ynoneb = yntrans-ydiffout


plt.plot(x,yinc/y_1100 ,label=r'Incident/source')
plt.plot(x,ynoneb/y_1100,label=r'Stellar transmission', ls=':')
plt.plot(x,ydiffout/y_1100,label=r'Nebular transmission', ls='--')
plt.plot(x,yntrans/y_1100,label=r'Stellar+Nebular transmission')
#plt.plot(x,yreflec/y_1100,label=r'Reflected+nebular inwards')
#plt.plot(x,ydiffout/y_1100,label=r'Nebular continuum', ls='--')
#plt.plot(x, ytest/y_1100, label=r'test')


#plt.axvline(x=600,ls=':',alpha=0.7,color='#7a2edb',lw=1.5)
plt.axvline(x=912,ls=':',alpha=0.7,color='#477af5',lw=1.5)
plt.axvline(x=1215,ls=':',alpha=1,color='#606060',lw=1.5)
plt.axvline(x=3650,ls=':',alpha=1,color='#606060',lw=1.5)
#plt.text(540,np.max(yinc/y_1100)*0.4,r'f$_{600}$',size=16,rotation=90,color='#7a2edb')
plt.text(855,np.max(yinc/y_1100)*0.7,r'Ly_C',size=16,rotation=90,color='#477af5')
plt.text(1160,np.max(yinc/y_1100)*0.7,r'Ly${\alpha}$',size=16,rotation=90,color='#606060')
plt.text(3650,np.max(yinc/y_1100)*0.7,r'H_C',size=16,rotation=90,color='#606060')
plt.xlim(100, 4000)
plt.ylim(1*10**(-4), 10)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'F$_{\lambda}$/F$_{1100}$',size=16)
plt.legend()
plt.tight_layout()
plt.savefig('cloudy_plots/clean/SED_F110norm_'+mod_name+'.jpg')
#plt.show()
plt.clf()



###########################################################
#                      MANUAL SPEC PLOT                   #
###########################################################
df = pd.read_csv(inputmodel+'.cont', sep='\t')

df_twophoton = pd.read_csv(inputmodel+'.twophoton', sep='\t')

display(df.head())
display(df)
xm = np.divide(911.267050550915, df.iloc[:,0])
ym_source = np.divide( df.loc[:,'incident'] ,  xm) #10**(-16)
ym_net = np.divide(df.loc[:,'net trans'], xm)
ym_nebular = np.divide(df.loc[:,'DiffOut'], xm)
ym_stellar = np.divide(df.loc[:,'trans'], xm) #ym_net-ym_nebular

ym_nebular_nolines = df.loc[:,'DiffOut'].iloc[np.where(df.loc[:,'nLine'] == 0)]


display(df_twophoton.head())
ym_twophoton = np.divide(df_twophoton.iloc[:, 2], xm) #* df_twophoton.iloc[:, 1]


print('current wavelength ratio ', xm.iloc[0]/x[0], 'energy ratio ', ym_source.iloc[0]/yinc[0])

idxm_1100 = np.argmin(np.abs(xm - 1100))
ym_1100 = ym_net[idxm_1100]


print('max lambda F_lambda/F1100 ', np.max(ym_twophoton/ym_1100), 'shape twophoton ', len(ym_twophoton), 'len(ym_source) ', len(ym_source), 'len xm ', len(xm))

plt.plot(xm, ym_source/ym_1100 ,label=r'Incident/source')
plt.plot(xm,ym_stellar/ym_1100,label=r'Stellar transmission', ls=':')
plt.plot(xm,ym_nebular/ym_1100,label=r'Nebular transmission', ls='--')
plt.plot(xm,ym_twophoton/ym_1100,label=r'2-photon', ls='--')
plt.plot(xm,ym_net/ym_1100,label=r'Stellar+Nebular transmission')

plt.axvline(x=600,ls=':',alpha=0.7,color='#606060',lw=1.5)
plt.axvline(x=912,ls=':',alpha=0.7,color='#606060',lw=1.5)
plt.axvline(x=1215,ls=':',alpha=0.7,color='#606060',lw=1.5)
plt.axvline(x=3650,ls=':',alpha=0.7,color='#606060',lw=1.5)
plt.text(540,np.max(yinc/y_1100)*0.6,r'f$_{600}$',size=16,rotation=90,color='#606060')
plt.text(855,np.max(yinc/y_1100)*0.6,r'Ly_C',size=16,rotation=90,color='#606060')
plt.text(1160,np.max(yinc/y_1100)*0.6,r'Ly${\alpha}$',size=16,rotation=90,color='#606060')
plt.text(3650,np.max(yinc/y_1100)*0.6,r'H_C$',size=16,rotation=90,color='#606060')
plt.xlim(100, 5000)
#plt.ylim(1*10**(-2), 10)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'F$_{\lambda}$/F$_{1100}$',size=16)
plt.legend()
plt.tight_layout()
plt.savefig('cloudy_plots/clean/manualSED_F110norm_'+mod_name+'.jpg')
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
    
    Ryd_grid = np.divide(911.267050550915, lambda_grid)

    df_gamma['lambda'] = np.divide(911.267050550915, df_gamma.iloc[:,1])

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

        lambda_idx = df_gamma.loc[(df_gamma['lambda'] >= val) & (df_gamma['Node'] > 0), 'lambda'].idxmin()

        gamma_raw.append(gamma_r.iloc[(lambda_grid <= df_gamma['lambda'][lambda_idx]).argmax()])  
        gamma.append(df_gamma.iloc[lambda_idx, col_idx]/(10**34 * np.power(10, logT*1.5)))

        if (val < 912):
            deltaE.append(1*(Ryd_grid.iloc[j]- df_gamma.iloc[lambda_idx,1]))
            #deltaE_smooth.append(1*(Ryd_grid.iloc[j]- df_gamma_smooth.iloc[lambda_idx,1]))
        else:
            #deltaE.append(0)
            deltaE.append(1*(Ryd_grid.iloc[j]- df_gamma.iloc[lambda_idx,1]))





    return np.array(gamma, dtype=float), deltaE, np.array(gnu_2S, dtype=float), col_idx, np.array(gamma_smooth, dtype=float), np.array(gamma_r, dtype=float)

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



df = pd.read_csv(inputmodel+'.cont', sep='\t')

xm = np.divide(911.267050550915, df.iloc[:,0])

analytic_idx = np.where((1 < xm) & (xm < 300000)) # Large range required for gnu symmetry mirroring
print('First few values of lambda_grid ', xm.iloc[analytic_idx].iloc[:5])
lambda_grid =  xm.iloc[analytic_idx]
nu_grid = np.divide(c, lambda_grid*10**(-8))

test_ind = np.where((880 < lambda_grid)  & (lambda_grid < 3645))
test_ind2 = np.where((500 < lambda_grid)  & (lambda_grid < 911))
test_ind4 =  np.where((500 < lambda_grid)  & (lambda_grid < 3645))

FluxLH_comb = np.zeros_like(lambda_grid)
FluxLHeIII_comb = np.zeros_like(lambda_grid)
FluxLB_comb = np.zeros_like(lambda_grid)
FluxL2SHe_comb = np.zeros_like(lambda_grid) 
FluxL2S_comb = np.zeros_like(lambda_grid)
FluxLH_comb_test2 = np.zeros_like(lambda_grid)
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

f_interp = interp1d(df_gnu_2S_sorted["nu"].values, df_gnu_2S_sorted["g-nu"].values, kind='cubic', bounds_error=False, fill_value="extrapolate")

gnu_2S_smooth = f_interp(np.divide(3*10**18, lambda_grid))

y = np.divide(lambda_grid/1215, -1)


lambda_12 = 2431 # test

for l in lambda_grid[lambda_grid < lambda_12]:
    if (l < 0.5*lambda_12):#0.500*lambda_12):
        gnu_2S_smooth[lambda_grid == l] = 0
    else:
        gnu_2S_smooth[lambda_grid == l] = np.divide(1, 2*l/lambda_12 -1)* gnu_2S_smooth[np.abs(lambda_grid - np.divide(l, 2*l/lambda_12 - 1)).argmin()] #np.abs(np.divide(3*10**18 ,lambda_grid) - np.divide(2*lambda_grid/lambda_12 - 1, lambda_grid)).argmin()]#np.divide(l, 2*l/lambda_12 - 1)).argmin()]
        



nHtest = 10**2
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

expo_test = np.divide(h*c,kB*Ttest)*(1/911 - np.power(lambda_grid.iloc[test_ind2], -1))*10**8



gff = 1.33 #np.sqrt(3)/np.pi * (np.log(Ttest**1.5*c/(lambda_grid*1e-8)) + 17.7) 
gamma_Bremstrahlung = 2**5 * np.pi * e**6 /(3*me*c**3) *np.sqrt(2*np.pi/(3*me*kB*Ttest)) * gff * np.exp(-h*nu_grid/(kB*Ttest))#(4*np.pi)**0*5.44e-41*Ttest**(-0.5)*gff*np.exp(-h*nu_grid/(kB*Ttest))


lambda0 = h*c/(0.4298*eV)*10**8
print('lambda0 ', lambda0)

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

lookup = {(14579, 22781.5): [5.0, 1] , (8202, 14579): [4.0, 1] , (3645.06, 8202): [3.0, 1.22], (911.26, 3645.06): [2.0, 1.19], (0, 911.26): [1.0, 1.12]}
u = 0


for (lower, upper), [ni, gbfi] in lookup.items():
    lambda_mid = np.where((lower < lambda_grid) & (lambda_grid < upper)) 
    lambda_below = np.where((lambda_grid < upper))#np.where((lower < lambda_grid)) 
    lambda_belowHe = np.where((lambda_grid < 227.8*ni**2))
    n_levels[lambda_below] = ni 
    gbf_levels[lambda_below] = gbfi 
    gbf_levels[lambda_grid < 912] = 1


    l=0
    Eion=np.power(911.25e-8/(h*c), -1) *np.power(ni, -2)
    ynl[lambda_below] = Eion*np.power( lambda_grid.iloc[lambda_below]*1e-8/(h*c), 1)
    tau = np.sqrt(Eion/(np.power( lambda_grid.iloc[lambda_below]*1e-8/(h*c), -1)-Eion))
    weights[lambda_below] = ni**2#2*ni**2
    Fnl[lambda_below] = np.exp(-4*ni*tau*(np.pi/2 - np.arctan(tau)))*np.power(1- np.exp(-2*np.pi*ni*tau) ,-1)
    
    phi[lambda_below] = [1 + 64*ynl[lambda_below]**1 + 1343*ynl[lambda_below]**2 + 12589*ynl[lambda_below]**3 + 55801*ynl[lambda_below]**4 + 124516*ynl[lambda_below]**5 + 138738*ynl[lambda_below]**6 + 70996*ynl[lambda_below]**7 + 13107*ynl[lambda_below]**8 , 1+38*ynl[lambda_below]**1+461.33*ynl[lambda_below]**2+2181.3*ynl[lambda_below]**3+4423.1*ynl[lambda_below]**4+3669.3*ynl[lambda_below]**5+1024*ynl[lambda_below]**6,((1+32/3*ynl[lambda_below]+208/9*ynl[lambda_below]**2+128/9*ynl[lambda_below]**3) +  (8*ynl[lambda_below] + 208/3 * ynl[lambda_below]**2 + 224/3 * ynl[lambda_below]**3) + (80/9 * ynl[lambda_below]**2 + 736/9*ynl[lambda_below]**3 + 256/3 * ynl[lambda_below]**4)) ,  (1+3*ynl[lambda_below]) + (3*ynl[lambda_below]+8*ynl[lambda_below]**2), 1][u]

    gbf[lambda_below] = (1)*8*np.pi*np.sqrt(3)*ni**4/ni**2 *ynl[lambda_below]*phi[lambda_below]*Fnl[lambda_below]
    u1 = (np.power(lambda_grid.iloc[lambda_below]*1e-8/(h*c), -1)-Eion)*ni**2 
    gbf2[lambda_below] = 1+0.1728*(u1 - 1)*np.power((u1 + 1) , -2/3) + 0.0496 * (u1**2 + 4/3 * u1 + 1)*np.power(ni*(u1+1) , -4/3)
    gbf4[lambda_below] = 2**3*np.pi*np.sqrt(3)*ni**(-1)*lambda_grid.iloc[lambda_below]/912*np.exp(-4*ni*tau*(np.pi/2 - np.arctan((tau))))*np.power((1- np.exp(-2*np.pi*ni*tau)*(1 - (lambda_grid.iloc[lambda_below]/(ni**2*912)))) ,-1)*[5*(1 + 64*ynl[lambda_below]**1 + 1343*ynl[lambda_below]**2 + 12589*ynl[lambda_below]**3 + 55801*ynl[lambda_below]**4 + 124516*ynl[lambda_below]**5 + 138738*ynl[lambda_below]**6 + 70996*ynl[lambda_below]**7 + 13107*ynl[lambda_below]**8 ), +38*4*ynl[lambda_below]**1 + 346/3*4**2*ynl[lambda_below]**2 + 409/3*4**3*ynl[lambda_below]**3 + 622/9*4**4*ynl[lambda_below]**4 + 43/3*4**5*ynl[lambda_below]**5 + 1*4**6*ynl[lambda_below]**6, 3+ 14*4*ynl[lambda_below]**1 + 19*16* ynl[lambda_below]**2 + 8*64*ynl[lambda_below]**3 + 256*ynl[lambda_below]**4, 2+3*4*ynl[lambda_below]**1 + 16*ynl[lambda_below]**2, 1][u]#(1)*2**2 *np.pi*np.sqrt(3) *(1/137)**2 *me*c**2*(lambda_grid.iloc[lambda_below]*1e-8/(h*c*ni**2))*np.exp(-4*ni*tau*(np.pi/2 - np.arctan((tau))))*np.power((1- np.exp(-2*np.pi*ni*tau))*ni**2*(tau**2+1) ,-1)*[0, 0, 3+ 14*4*ynl[lambda_below]**1 + 19*16* ynl[lambda_below]**2 + 8*64*ynl[lambda_below]**3 + 256*ynl[lambda_below]**4, 2+3*4*ynl[lambda_below]**1 + 16*ynl[lambda_below]**2, 1][u]
    print('gbf variable', gbf[lambda_below][0], 'gbf2 approx ', gbf2[lambda_below][0])
    coeff = (ni**(1))**0 *2**4*h*e**2/(3*np.sqrt(3)*me*c)# (ni**4)
    bf_cross[lambda_below] +=  coeff*np.power(Eion,2)*np.power(lambda_grid.iloc[lambda_below]*1e-8/(h*c), 3)*gbf[lambda_below] * np.power(ni, -1)


    
    bf_cross_alt2[lambda_below] += (ni**0)**0*2**6*(1/137)*np.pi*(5.29e-9)**2*ni/(3*np.sqrt(3))*np.power(ni**2*912/(lambda_grid.iloc[lambda_below]), -3) *gbf2[lambda_below] 
    

    mu3 = np.sqrt((1/137)**2*c**2 *me/2*np.power( np.abs(h*c/(lambda_grid.iloc[lambda_below]*1e-8) - Eion), -1))
    
    bf_cross_alt3[lambda_below] += me*c**2/ni**2 *np.abs(h*c/(lambda_grid.iloc[lambda_below]*1e-8) - Eion)/(h*c/(lambda_grid.iloc[lambda_below]*1e-8))**2 * 2**6 /3 *(1/137)**1 * h**2/(me**2*c**2) * np.power(mu3, 2)* np.exp(-4*mu3*np.arctan(ni/mu3)) *np.power((1- np.exp(-2*np.pi*mu3)) *(1 + ni**2/mu3**2)**2 ,-1)*[5*(1 + 64*ynl[lambda_below]**1 + 1343*ynl[lambda_below]**2 + 12589*ynl[lambda_below]**3 + 55801*ynl[lambda_below]**4 + 124516*ynl[lambda_below]**5 + 138738*ynl[lambda_below]**6 + 70996*ynl[lambda_below]**7 + 13107*ynl[lambda_below]**8 ), 4+38*4*ynl[lambda_below]**1 + 346/3*4**2*ynl[lambda_below]**2 + 409/3*4**3*ynl[lambda_below]**3 + 622/9*4**4*ynl[lambda_below]**4 + 43/3*4**5*ynl[lambda_below]**5 + 1*4**6*ynl[lambda_below]**6, 3+ 14*4*ynl[lambda_below]**1 + 19*16* ynl[lambda_below]**2 + 8*64*ynl[lambda_below]**3 + 256*ynl[lambda_below]**4, 2+3*4*ynl[lambda_below]**1 + 16*ynl[lambda_below]**2, 1][u]


    

    EionHe=np.power(227.8e-8/(h*c), -1) *np.power(ni, -2)
    ynlHe[lambda_belowHe] = EionHe*np.power(lambda_grid.iloc[lambda_belowHe]*1e-8/(h*c), 1)
    tau = np.sqrt(EionHe/(np.power(lambda_grid.iloc[lambda_belowHe]*1e-8/(h*c), -1)-EionHe))
    mu3He = np.sqrt((1/137)**2*c**2 *me/2*np.power( np.abs(h*c/(lambda_grid.iloc[lambda_belowHe]*1e-8) - EionHe), -1))


    bf_cross_alt3_He[lambda_belowHe] += me*c**2/ni**2 *np.abs(h*c/(lambda_grid.iloc[lambda_belowHe]*1e-8) - EionHe)/(h*c/(lambda_grid.iloc[lambda_belowHe]*1e-8))**2 * 2**6 /3 *(1/137)**1 * h**2/(me**2*c**2) * np.power(mu3He, 2)* np.exp(-4*mu3He*np.arctan(ni/mu3He)) *np.power((1- np.exp(-2*np.pi*mu3He)) *(1 + ni**2/mu3He**2)**2 ,-1)*[5*(1 + 64*ynlHe[lambda_belowHe]**1 + 1343*ynlHe[lambda_belowHe]**2 + 12589*ynlHe[lambda_belowHe]**3 + 55801*ynlHe[lambda_belowHe]**4 + 124516*ynlHe[lambda_belowHe]**5 + 138738*ynlHe[lambda_belowHe]**6 + 70996*ynlHe[lambda_belowHe]**7 + 13107*ynlHe[lambda_belowHe]**8 ), 4+38*4*ynlHe[lambda_belowHe]**1 + 346/3*4**2*ynlHe[lambda_belowHe]**2 + 409/3*4**3*ynlHe[lambda_belowHe]**3 + 622/9*4**4*ynlHe[lambda_belowHe]**4 + 43/3*4**5*ynlHe[lambda_belowHe]**5 + 1*4**6*ynlHe[lambda_belowHe]**6, 3+ 14*4*ynlHe[lambda_belowHe]**1 + 19*16* ynlHe[lambda_belowHe]**2 + 8*64*ynlHe[lambda_belowHe]**3 + 256*ynlHe[lambda_belowHe]**4, 2+3*4*ynlHe[lambda_belowHe]**1 + 16*ynlHe[lambda_belowHe]**2, 1][u]


    bf_cross_alt4[lambda_below] += 2**6*(1/137)*np.pi*(5.29e-9)**2*ni/(3*np.sqrt(3))*np.power(ni**2*912/(lambda_grid.iloc[lambda_below]), -3) *gbf4[lambda_below]#2**6*np.pi/np.sqrt(3)*ni*ni**2*ynl[lambda_below]**4*(1/137)**3*(5.29e-9)**2*gbf3[lambda_below]*me*c**2

    u = u+1
plt.plot(lambda_grid , gbf2, label='g_n Seaton+59')
#plt.plot(lambda_grid , bf_cross_alt3 / (2**6 /(3**1.5) * h*e**2/(me*c) * Eion**2*np.power(lambda_grid/(h*c), 3)) , label='g_n Katkov')
plt.plot(lambda_grid , gbf4, label='g_n Kotelnikov+18', linewidth=4)
plt.plot(lambda_grid , gbf , label='g_n analytic', c='black')

gbf_Karzas = np.array([0.797, 0.853, 0.876, 0.975])
lambda_Karzas = np.array([911, 730, 3640, 1820])
plt.scatter(lambda_Karzas , gbf_Karzas ,label=r'Karzas&Latter+61 table', zorder=10)

plt.ylim(0, 1.2)
plt.xlim(600, 3700)
#bf_cross[lambda_grid < 911] = ion_cross[lambda_grid < 911]
plt.legend()
#print('ion_cross_O 911 ', ion_cross_O.iloc[test_ind2[0][0]])
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'g$_{n}$',size=16)

plt.savefig('cloudy_plots/clean/gbf_v5_'+mod_name+'.jpg')
#plt.show()
plt.clf()

plt.plot(lambda_grid, phi,  label='phi')
#plt.plot(np.concatenate([lambda_save[0], lambda_save[1], lambda_save[2], lambda_save[3]], axis=1) , np.concatenate([phi[0], phi[1], phi[2], phi[3]]) , label='phi')
plt.ylim(0, 8)
plt.xlim(300, 2000)
plt.legend()
plt.savefig('cloudy_plots/clean/phi_v3_'+mod_name+'.jpg')
#plt.show()
plt.clf()

# deconstruct dependencies based on gbf components
plt.plot(lambda_grid , Fnl , label='Fnl')
plt.ylim(0, 0.001)
plt.xlim(3000, 4000)
plt.legend()
plt.savefig('cloudy_plots/clean/Fnl_v3_'+mod_name+'.jpg')
#plt.show()
plt.clf()

plt.clf()


ion_cross = (1)*n_levels**(-2)*5.475e-14 * np.power(lambda_grid/lambda0, 4.0185) * np.power(np.power(lambda_grid/lambda0 , -1) -1, 2) * np.power(1+ np.power(lambda_grid*32.88/(lambda0) , -0.5) , -2.963)

ion_cross_alt = 6.3e-18* np.power(lambda_grid/911, 3)
epsi = np.sqrt(np.power(lambda_grid/911, -1) -1)
ion_cross_O = 6.3e-18 * np.power(lambda_grid/911, 4)*np.exp(4-np.divide(4*np.arctan(epsi), epsi))*np.divide(1, 1-np.exp(np.divide(-2*np.pi, epsi)))

ion_cross_O2 = 6.3e-18*(1.34*np.power(lambda_grid/912, 2.99) + (1-1.34)*np.power(lambda_grid/912, 2.99+1))

plt.plot(lambda_grid, bf_cross_alt2*10**18 , label='Bound-free crossection Seaton+59')
plt.plot(lambda_grid , bf_cross_alt3*10**18 , label='Bound-free Katkov', c='black', lw=2)
plt.plot(lambda_grid , bf_cross_alt4*10**18 , label='Bound-free Kotelnikov+18')

plt.plot(lambda_grid, bf_cross*10**18 , label=r'Bound-free CHIANTI ($\lambda^3$ fit)', linestyle='dashed')
#plt.plot(lambda_grid, bf_cross_alt3_He*10**18 , label=r'He attempt', linestyle='dashed')


plt.xlim(200, 4000)
plt.ylim(0.0001, 100)
#plt.ylim(np.min(jnu_comb), np.max(ym_nebular))
#plt.xlim(100, 5000)
#plt.ylim(1*10**(-2), 10)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'ionisation cross section[Mb or 10^-18cm^2]',size=16)
plt.legend()
plt.tight_layout()
plt.savefig('cloudy_plots/clean/analyticcross_v9_'+mod_name+'.jpg')
#plt.show()
plt.clf()

### Recombination crossection

fig = plt.figure()
#plt.plot((h*c*10**8)/eV*np.power(lambda_grid, -1) , ion_cross_alt*10**18 , linestyle='-.',c='blue',  label='lambda**3 scaling')
#plt.plot(lambda_grid, bf_cross_alt2*10**18 , label='Bound-free crossection Seaton+59')
#plt.plot((h*c*10**8)/eV*np.power(lambda_grid, -1) , bf_cross_alt1*10**18 , label='Bound-free crossection Seaton+59 * stat weights')#Correct for added/moved n**2
plt.plot(lambda_grid , n_levels**2 * h**2 * c**2/(2*me*c**2 * lambda_grid**2)* np.power(np.abs(h*c/(lambda_grid*1e-8) - Eion), -1) *bf_cross_alt4*10**18 , label='Free-bound Katkov', c='black')
#plt.plot(lambda_grid , bf_cross_alt4*10**18 , label='Bound-free Kotelnikov+18')

plt.plot(lambda_grid, n_levels**2 * h**2 * c**2/(2*me*c**2 * lambda_grid**2)* np.power(np.abs(h*c/(lambda_grid*1e-8) - Eion), -1) * bf_cross*10**18 , label=r'Freebound CHIANTI ($\lambda^3$ fit)', linestyle='dashed')
#plt.plot(lambda_grid[lambda_grid < 912] , ion_cross_O2[lambda_grid < 912]*10**18 , linestyle='--', label='Osterbrock ionising crossection', c='red')

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
plt.savefig('cloudy_plots/clean/Recombcross_v1_'+mod_name+'.jpg')
#plt.show()
plt.clf()



def gamma_bf_Hlike(Z, Ti, bf_cross, nu_912):
    nu_0 = nu_912*Z**2
    nu_228 = c/(227.8e-8)
    gamma = np.zeros_like(bf_cross)
    gamma_base = np.zeros_like(bf_cross)
    for n in [1, 2, 3, 4]:#, 3, 4]:#, 4]:#,2,3,4, 5]:
        gamma_base = np.zeros_like(bf_cross)
        weight = n**2
        if (n > 1):
            nu_cutoff = (nu_grid > nu_0/n**2) & (nu_0/(n-1)**2 > nu_grid)
        else:
            nu_cutoff = (nu_grid > nu_912*Z**2) & (nu_912*(Z+1)**2 > nu_grid)

        gamma_base[nu_cutoff] = weight * Z**3 * 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid[nu_cutoff]*1e-8/c, -3) *np.power(me/(2*np.pi*kB*Ti) , 1.5)* bf_cross_alt3[nu_cutoff] * np.exp(h*(nu_0/n**2 - nu_grid[nu_cutoff])/(kB*Ti)) * (1-np.exp(-h*nu_grid[nu_cutoff]/(kB*Ti)))   #np.exp(np.divide(deltaE,-1*kB_Ryd*Ttest))[nu_cutoff] #np.exp(h*(nu_0/n**2 - nu_grid[nu_cutoff])/(kB*Ti)) #*(1-np.exp(-h*nu_grid[nu_cutoff]/(kB*Ti)))
        gamma += gamma_base
    
    return gamma

gamma_raw_test =  weights* 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid*1e-8/c, -3)  * np.power(me/(2*np.pi*kB*Ttest) , 1.5) * bf_cross#*2*np.power(n_levels, 2)

gamma_raw_test = np.array(gamma_raw_test)

gamma_raw_test2 = (1)*weights* 2**3 *np.pi * h**4/(me**3*c**2) *np.power(lambda_grid*1e-8/c, -3)  * np.power(me/(2*np.pi*kB*Ttest) , 1.5) * bf_cross_alt2

gamma_raw_test2 = np.array(gamma_raw_test2)


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
print('ratio of gamma 912 over gamma 3400 ', gamma_raw_test[np.argmin(np.abs(lambda_grid - 911))]/gamma_raw_test[np.argmin(np.abs(lambda_grid - 3400))])

gamma_Osterbrock = np.array([ 20, 4.5, 1.5, 0.0025, 0.196, 6.17, 13.8, 20.5, 1.13, 1.45, 1.90, 2.89, 3.69, 4.18, 1.54, 1.72, 0.772, 0.262])*10**(-25) 
lambda_Osterbrock = np.array([0.35, 0.81, 1.45, 0.1, 0.15, 0.26, 0.3122, 0.3645, 0.3647, 0.4, 0.45, 0.5696, 0.7, 0.8203, 0.8205, 1, 3, 10])



gamma_script = gamma_bf_Hlike(1, Ttest, bf_cross_alt3, c/(911.25e-8))


fig = plt.figure()

print('exp effect near n=2 peak ', np.exp(np.divide(deltaE[np.argmin(np.abs(lambda_grid - 911))],-1*kB_Ryd*Ttest)), 'CHIANTI at n=1 ', gamma_raw_test4[np.argmin(np.abs(lambda_grid - 911))]* 911e-8 *911e-8/c, 'CHIANTI exp ', np.exp(-1/(kB * Ttest) *np.abs(h*c/911e-8-h*c/(912e-8*1**2))) , 'E= ', np.power(912*1e-8/(h*c), -1))

plt.plot(lambda_grid , (1+np.exp(-h*c/(lambda_grid*1e-8*kB*Ttest)))**1 *(np.exp(np.divide(deltaE,-1*kB_Ryd*Ttest)) *gamma_raw_test3) ,label=r'Free-bound analytic (Katkov+78)', c='black')

plt.plot(lambda_grid , (1+np.exp(-h*c/(lambda_grid*1e-8*kB*Ttest)))**1 * gamma_script ,label=r'Script version', c='green', zorder=5)

plt.plot(lambda_grid , (1/(4*np.pi))**0* np.exp(-1/(kB * Ttest) *np.abs(h*c/(lambda_grid*1e-8) -h*c/(912e-8*n_levels**2))) *gamma_raw_test4 ,label=r'Free-bound (CHIANTI) ', linestyle='dashed', c='red')

plt.plot(lambda_grid , np.exp(-1/(kB * Ttest) *np.abs(h*c/(lambda_grid*1e-8) -h*c/(912e-8*n_levels**2))) *gamma_raw_test2 ,label=r'Free-bound (Seaton) ')

plt.plot(lambda_grid , gamma_2S *np.power(lambda_grid*1e-8/c, 0),label=r'2-photon (Nussbaumer)', c='orange')
plt.plot(lambda_grid , gamma_2S_alt *np.power(lambda_grid*1e-8/c, 0),label=r'2-photon (Osterbrock)', c='purple', linestyle='--')
plt.plot(lambda_grid , gamma_Bremstrahlung*np.power(lambda_grid*1e-8/c, 0) ,label=r'Free-free', c='purple')
#plt.plot(lambda_grid , np.exp(np.divide(deltaE,-1*kB_Ryd*Ttest))*gamma_raw_test2*np.power(lambda_grid*1e-8/c, 0) ,label=r'Free-bound Saton+59')
#plt.plot(lambda_grid , (1/(4*np.pi))**0*np.exp(np.divide(deltaE,-1*kB_Ryd*Ttest))*gamma_raw_test3*np.power(lambda_grid*1e-8/c, 0) ,label=r'Free-bound Katkov+78', c='darkgreen')

#plt.plot(lambda_grid/10000 , gamma_raw_test3*np.divide(1, 1) ,label=r'Free-bound (Osterbroke other fit)', linestyle='-', c='red')
#plt.plot(lambda_grid/10000 , gamma_raw_test3*np.divide(1, 1)*np.power(lambda_grid/911, -0.5) ,label=r'Free-bound gamma (power of $\nu^{2.5}$)', linestyle=':')
#plt.plot(lambda_grid/10000 , gamma*np.divide(1, 1) ,label=r'noexponential (analytic)',  linestyle='--')


plt.scatter(lambda_Osterbrock*1e4 , gamma_Osterbrock*np.power(lambda_Osterbrock*1e4*1e-8/c, 1) ,label=r'Osterbrock table', zorder=10)

gamma_Draine = np.array([4.86e-25, 4.7e-24, 1.52e-26,1.6e-25, 8.4e-27, 2.17e-26, 4e-27, 5.26e-27, 1.72e-27])
lambda_Draine = np.array([783, 911, 2061, 3640, 4114, 8200, 9448, 14550, 22750])
plt.scatter(lambda_Draine , (4*np.pi)**(1)*gamma_Draine*np.power(lambda_Draine*1e-8/c, 1) ,label=r'Peaks from Draine book/plot', zorder=10)

print('Draine vs our solution ratio, n=1 ', (4*np.pi)**(1)*gamma_Draine[1]*np.power(lambda_Draine[1]*1e-8/c, 1)/gamma_raw_test[np.argmin(np.abs(lambda_grid - 910))], 'n=2 ', (4*np.pi)**(1)*gamma_Draine[4]*np.power(lambda_Draine[4]*1e-8/c, 1)/gamma_raw_test[np.argmin(np.abs(lambda_grid - 3600))],  'n=3 ', (4*np.pi)**(1)*gamma_Draine[6]*np.power(lambda_Draine[6]*1e-8/c, 1)/gamma_raw_test[np.argmin(np.abs(lambda_grid - 8200))]  )


print('n=1 peak w/o sum ', gamma_raw_test3[np.argmin(np.abs(lambda_grid - 910))], 'n=1 peak w. sum ', gamma_raw_test3[np.argmin(np.abs(lambda_grid - 910))] + gamma_raw_test3[np.argmin(np.abs(lambda_grid - 3640))] + gamma_raw_test3[np.argmin(np.abs(lambda_grid - 8150))] + gamma_raw_test3[np.argmin(np.abs(lambda_grid - 14500))] , 'n=1 Draine peak ', (4*np.pi)**(1)*gamma_Draine[1]*np.power(lambda_Draine[1]*1e-8/c, 1))

gamma_Ercolano2 = 1e-40*np.array([3.704e-1, 2.376e1, 2.629e-6, 1.698e-4, 1.093e-2, 6.996e-1, 4.442e1, 1.653e0, 2.851e0, 2.418e1])
lambda_Ercolano2 = np.array([2998, 3331, 3747, 4283, 4996, 5996, 7495, 9993, 14990, 29979])

plt.plot(lambda_grid[lambda_grid > 912] , np.exp(-1/(kB * Ttest) *np.abs(h*c/(lambda_grid[lambda_grid > 912]*1e-8) -h*c/(912e-8*n_levels[lambda_grid > 912]**2))) *gamma_H[lambda_grid > 912]*np.power(lambda_grid[lambda_grid > 912]*1e-8/c, 0) ,c='grey', linestyle=':', label=r'Free-bound (Ercolano table)')
#plt.scatter(lambda_Ercolano2 , (4*np.pi*(1e4**1.5*1e4**(-1.5)))**1*gamma_Ercolano2*np.power(lambda_Ercolano2*1e-8/c, 0) ,label=r'Free-bound (Ercolano table 2)', zorder=10)

print('gnu_2S_alt 1300 ', gnu_2S_smooth[i1], 'gnu_2S_alt 2000  ', gnu_2S_smooth[i2], 'gnu_2S 1300 ', gnu_2S[i1], 'gnu_2S 2000  ', gnu_2S[i2], 'ratio ', gnu_2S[i1]/gnu_2S_smooth[i1])

print('')
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

plt.xlim(800, 6000)
plt.ylim(1e-44, 1e-36)
#plt.ylim(1e-41, 4e-38)
#plt.ylim(1e-50, 1e-30)
#plt.ylim(np.min(jnu_comb), np.max(ym_nebular))
#plt.xlim(100, 5000)
#plt.ylim(1*10**(-2), 10)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'gamma',size=16)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('cloudy_plots/clean/analyticgamma_v28_'+mod_name+'.jpg')
#plt.show()
plt.clf()

###






df_phy = pd.read_csv(inputmodel+'.phy', sep='\t')

df_H = pd.read_csv(inputmodel+'.ele_H', sep='\t')

df_He = pd.read_csv(inputmodel+'.ele_He', sep='\t')

df_rad = pd.read_csv(inputmodel+'.rad', sep='\t')

display(df_phy)


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
print('lineID filtered ', df.loc[:,'lineID'].iloc[np.where(df.loc[:,'lineID'] == 'Inci')])

ym_neblines = (df.loc[:,'outlin'].iloc[analytic_idx].iloc[np.where(df.iloc[analytic_idx].loc[:,'lineID'] != 'Inci')] * lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)).fillna(0)
ym_intlines = (df.loc[:,'outlin'].iloc[analytic_idx].iloc[np.where(df.iloc[analytic_idx].loc[:,'lineID'] == 'Inci')] * lambda_grid*1e-8/c / (4*np.pi*R.iloc[-1]**2)).fillna(0)




print('Starting radius shell looping')

for i, Ri in tqdm(enumerate(R)): #R[:-1])):
    
    Ti = T[i]
    nHi = nH[i]* Xfrac[i]
    nei = ne[i]#* Xfrac[i]
    #nHei = nHe[i]*XfracHe[i]


    #nHei = nHe[i]*XfracHe[i]
    print('i ', i, 'Ri ', Ri, 'dR[i] ', dR[i], 'Ti ', Ti, 'nproti ',nHi, 'nei ', nei)

    gamma, deltaE, gnu_2S_alt, col_idx, gamma_smooth, gamma_raw = extract_gamma(lambda_grid,np.log10(Ti))

    
    nu_912 = c/(911.25e-8)
    gamma_H = gamma_bf_Hlike(1, Ti, bf_cross_alt3, nu_912)  
    jnu_H = nHi*nei *gamma_H/(4*np.pi) #nHi*nei*gamma_raw_test3*np.exp(deltaE/(-1*kB_Ryd*Ti))  #nHi*nei *gamma_H/(4*np.pi) #*1/(10**34 * Ti**1.5*np.exp(deltaE/(kB*Ti))) * gamma 
   
    nu_228 = c/(227.8e-8)
    #gamma_HeIII = gamma_bf_Hlike(2, Ti, bf_cross_alt3_He, nu_912) #np.exp(h*c*(1/(195e-8) - nu_grid)/(kB*Ti/Z**2)) * gamma_raw_test3 * Z**3  
    #jnu_He = nHei*nei/(4*np.pi) *gamma_HeIII

    gff = 1.3#np.sqrt(3)/np.pi * (np.log(Ti**1.5*c/(lambda_grid*1e-8)) + 17.7) 
    jnu_Bremstrahlung_alt = nHi*nei/(4*np.pi) *2**5 * np.pi * e**6 /(3*me*c**3) *np.sqrt(2*np.pi/(3*me*kB*Ti)) * gff * np.exp(-h*nu_grid/(kB*Ti))#5.44e-41*nei*nHi*Ti**(-0.5)*gff*np.exp(-h*nu_grid/(kB*Ti))
    if (Ti > 15000):
        qp = 4.71e-4
    else:
        qp = 5.31e-4
    
    row_idx2S = np.abs(a2S['T'] - Ti).argmin()
    #gamma_2SHe = a2S.iloc[row_idx2S, 1]* np.divide(1, 1 + (nHei+nei)*qp/(2*A2S))* gnu_2SHe
    gamma_2S = a2S.iloc[row_idx2S, 1]* np.divide(1, 1 + (nHi+nei)*qp/(2*A2S))* gnu_2S#*3*10**(8)* np.power(lambda_grid, -1)*10**10
    #jnu_2SHe = nHei*nei/(4*np.pi) *gamma_2SHe
    jnu_2S = nHi*nei/(4*np.pi) *gamma_2S#* a2S.iloc[row_idx2S, 1]* np.divide(1, 1 + nHi*qp/A2S)* gnu_2S_smooth
    
    FluxLH =4*np.pi*jnu_H * dR[i]#*(1+np.exp(-h*c/(lambda_grid*1e-8*kB*Ti) )) #*3*10**(8)* np.power(lambda_grid, -1)*10**10
    FluxLH_test2 =dR[i] * jnu_H#nHi*nei/1 *gamma_raw_test3*np.exp(deltaE/(-1*kB_Ryd*Ti))

    #FluxLHeIII = 4*np.pi*jnu_He * dR[i]

    FluxLB = (1/1.5/0.39)**1*4*np.pi*jnu_Bremstrahlung_alt * dR[i]#*3*10**(8)* np.power(lambda_grid, -1)*10**10

    widthDoppler = np.sqrt(0.5*kB*Ti/(mH*c**2))*1216


    #FluxL2SHe = (1/0.047)**1 *4*np.pi*jnu_2SHe * dR[i]
    FluxL2S = (1/1.2)**1*4*np.pi*jnu_2S * dR[i]#*3*10**(8)* np.power(lambda_grid, -1)*10**10 #-FluxL2S_comb * np.exp(-1*np.power(lambda_grid-1216 ,-2)*widthDoppler**2)*(R.iloc[-1]-Ri)*np.sqrt(np.pi)*e**2/(mH*c*widthDoppler)
    #FluxLHe = 4*np.pi*jnu_He * dR[i]*3*10**(8)* np.power(lambda_grid, -1)*10**10


    FluxLH_comb = np.add(FluxLH_comb, FluxLH)#*np.power(lambda_grid*1e-8, -1)#*np.power(lambda_grid*1e-8/c, -1)#  *np.power(lambda_grid*1e-8, -1)
    #FluxLHeIII_comb = np.add(FluxLHeIII_comb, FluxLHeIII)
    FluxLB_comb = np.add(FluxLB_comb, FluxLB)#*np.power(lambda_grid*1e-8, -1)# *np.power(lambda_grid*1e-8/c, -1)#*np.power(lambda_grid*1e-8, -1)
    #FluxL2SHe_comb = np.add(FluxL2SHe_comb, FluxL2SHe)
    FluxL2S_comb = np.add(FluxL2S_comb, FluxL2S)#*np.power(lambda_grid*1e-8, -1)#*np.power(lambda_grid*1e-8/c, -1)#*np.power(lambda_grid*1e-8, -1)
    FluxLH_comb_test2 = np.add(FluxLH_comb_test2, FluxLH_test2)

FluxL_tot = np.add(np.add(FluxLH_comb, FluxL2S_comb), FluxLB_comb)#, FluxLHeIII_comb)
idxm_1100 = np.argmin(np.abs(xm - 1100))

#plt.plot(lambda_grid, ym_nebular_unscaled/(4*np.pi*R.iloc[-1]**2)*np.power(lambda_grid*1e-8, 0) ,label=r'Combined nebular (CLOUDY)')
plt.plot(lambda_grid, ym_nebular_unscaled*np.power(lambda_grid*1e-8, 0) ,label=r'Combined nebular (CLOUDY)')
#plt.plot(xm.iloc[np.where(df.loc[:,'nLine'] == 0)], ym_nebular_nolines/(4*np.pi*R.iloc[-1]**2) ,label=r'Combined nebular transmission (CLOUDY)')

plt.plot(lambda_grid , FluxL_tot *np.power(lambda_grid*1e-8/c, 0),label=r'Combined nebular (analytic)', linestyle='-', c='black')
plt.plot(lambda_grid , FluxLH_comb *np.power(lambda_grid*1e-8/c, 0),label=r'Free-bound (analytic)', linestyle='dashed',c='green')#, linestyle='dashed')
plt.plot(lambda_grid , FluxLB_comb *np.power(lambda_grid*1e-8/c, 0),label=r'Bremsstrahlung (analytic)', linestyle='dashed', c='purple')

plt.plot(lambda_grid , FluxL2S_comb *np.power(lambda_grid*1e-8/c, 0),label=r'2-photon (analytic)', linestyle='dashed', c='orange')
#plt.plot(lambda_grid , FluxL_tot *np.power(lambda_grid*1e-8/c, 0),label=r'Combined nebular (analytic)', linestyle='-', c='black')
#plt.plot(lambda_grid , FluxLH_comb_test2 *np.power(lambda_grid*1e-8/c, 0),label=r'Free-bound Katkov+78')

test_Flux = (1)*n_levels**(-3)*2**0 * ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911)]*np.power(lambda_grid*1e-8/c, 0)
test_Flux = test_Flux * np.exp( np.divide(h*c,kB*Ttest)*(1/911/n_levels**2 - np.power(lambda_grid, -1))*10**8)
#test_Flux.iloc[np.where(n_levels > 1.0)] = test_Flux.iloc[np.where(n_levels > 1.0)]#*2**(-0.5-0.5+0.25)

#test_Flux2 = n_levels**(-2)*2**-1 * ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911)]
#test_Flux2[np.where(n_levels > 1.0)] = test_Flux2[np.where(n_levels > 1.0)]*2**0.5

#test_Flux =  ym_nebular_unscaled/(4*np.pi*R*np.power(lambda_grid*1e-8/c, 0)
#plt.plot(lambda_grid[lambda_grid < 911.3] , ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911.3)]*np.exp( np.divide(h*c,kB*Ttest)*(1/911.3 - np.power(lambda_grid[lambda_grid < 911.3], -1))*10**8) ,label=r'Simple approximation of exponential', linestyle='dashed', c='red')
#plt.plot(lambda_grid , test_Flux /np.exp( np.divide(h*c,kB*Ttest)*(1/911/n_levels**2 - np.power(lambda_grid, -1))*10**8),label=r'Simple approximation of amplitude', linestyle='dashed', c='orange')
#plt.plot(lambda_grid , test_Flux2 ,label=r'Simple approximation 2.0', linestyle='dashed', c='orange')

#plt.plot(lambda_grid[lambda_grid < 911] ,  FluxLH_comb[np.argmax(lambda_grid < 911)]*np.exp( np.divide(h*c,kB*Ttest)*(1/911 - np.power(lambda_grid[lambda_grid < 911], -1))*10**8), linestyle='-', c='red', label=r'Boltzman exponential', lw=3)

#plt.scatter([lambda_grid.iloc[np.argmax(lambda_grid < 911)], lambda_grid.iloc[np.argmax(lambda_grid < 3640)], lambda_grid.iloc[np.argmax(lambda_grid < 8190)], lambda_grid.iloc[np.argmax(lambda_grid < 14500)], 22000], [ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 911)] , ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 3640)], ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 8190)], ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 14500)], ym_nebular_unscaled.iloc[np.argmax(lambda_grid < 22000)]], label='testpoints', c='red')
#plt.scatter(911, 4*np.pi*5.53634e-19 * 912/c, label='CLOUDY emissivity', c='r')
plt.axvline(x=600,ls=':',alpha=0.7,color='#606060',lw=1.5)
plt.axvline(x=912,ls=':',alpha=0.7,color='#606060',lw=1.5)

plt.xlim(200, 17000)
plt.ylim(1e-23,1e-17)
#plt.ylim(np.min(jnu_comb), np.max(ym_nebular))
#plt.xlim(100, 5000)
#plt.ylim(1*10**(-2), 10)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda$ [$\AA$]',size=16)
plt.ylabel(r'F$_{\nu}$',size=16)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('cloudy_plots/clean/analyticSED_v19_'+mod_name+'.jpg')
#plt.show()
plt.clf()



