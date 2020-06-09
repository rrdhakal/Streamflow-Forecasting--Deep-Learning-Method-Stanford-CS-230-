# -*- coding: utf-8 -*-
"""
Created on Sun May 24 10:51:55 2020

@author: josue
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe-Effiency
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations
    
    Returns
    -------
    float
        Nash-Sutcliffe-Efficiency
    
    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If all values in the observations are equal
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    # denominator of the fraction term
    denominator = np.sum((obs - np.mean(obs))**2)

    # this would lead to a division by zero error and nse is defined as -inf
    if denominator == 0:
        msg = [
            "The Nash-Sutcliffe-Efficiency coefficient is not defined ",
            "for the case, that all values in the observations are equal.",
            " Maybe you should use the Mean-Squared-Error instead."
        ]
        raise RuntimeError("".join(msg))

    # numerator of the fraction term
    numerator = np.sum((sim - obs)**2)

    # calculate the NSE
    nse_val = 1 - numerator / denominator

    return nse_val


filepath = 'C:\\Users\\josue\\project_CS230\\ealstm_regional_modeling\\runs\\'
filenames = {'EALSTM':    'run_0106_1903_seed21\\EALSTM_seed21.p',
             'AGRU_hps_0':'run_2305_1125_seed595609\\EALSTM_seed595609.p',             
             'AGRU_hps_1':'run_2905_1917_seed748231\\EALSTM_seed748231.p',
             'AGRU_hps_2':'run_3005_1119_seed425359\\EALSTM_seed425359.p',
             'AGRU_hps_3':'run_3005_1319_seed219836\\EALSTM_seed219836.p',
             'AGRU_hps_4':'run_3005_1906_seed98194\\EALSTM_seed98194.p',
             'AGRU_hps_5':'run_0106_1809_seed20\\EALSTM_seed20.p',
             'AGRU_hps_6':'run_3005_1536_seed984683\\EALSTM_seed984683.p',
             'AGRU_hps_7':'run_0106_1958_seed22\\EALSTM_seed22.p',
             'AGRU_hps_8':'run_0106_2051_seed23\\EALSTM_seed23.p',
             'AGRU_hps_9':'run_0206_1453_seed775825\\EALSTM_seed775825.p',
             'AGRU_hps_10':'run_0206_1812_seed24\\EALSTM_seed24.p',
             'AGRU_hps_11':'run_0206_1925_seed25\\EALSTM_seed25.p',
             'AGRU_hps_12':'run_0406_1450_seed26\\EALSTM_seed26.p',
             'AGRU_hps_13':'run_0406_1744_seed27\\EALSTM_seed27.p',
             'AGRU_hps_14':'run_0406_1857_seed28\\EALSTM_seed28.p',
             
             'EALSTMt':    'run_0106_1903_seed21\\EALSTM_train_seed21.p',
             'AGRU_hps_0t':'run_2305_1125_seed595609\\EALSTM_train_seed595609.p',
             'AGRU_hps_1t':'run_2905_1917_seed748231\\EALSTM_train_seed748231.p',
             'AGRU_hps_2t':'run_3005_1119_seed425359\\EALSTM_train_seed425359.p',
             'AGRU_hps_3t':'run_3005_1319_seed219836\\EALSTM_train_seed219836.p',
             'AGRU_hps_4t':'run_3005_1906_seed98194\\EALSTM_train_seed98194.p',
             'AGRU_hps_5t':'run_0106_1809_seed20\\EALSTM_train_seed20.p',
             'AGRU_hps_6t':'run_3005_1536_seed984683\\EALSTM_train_seed984683.p',
             'AGRU_hps_7t':'run_0106_1958_seed22\\EALSTM_train_seed22.p',
             'AGRU_hps_8t':'run_0106_2051_seed23\\EALSTM_train_seed23.p',
             'AGRU_hps_9t':'run_0206_1453_seed775825\\EALSTM_train_seed775825.p',             
             'AGRU_hps_10t':'run_0206_1812_seed24\\EALSTM_train_seed24.p',
             'AGRU_hps_11t':'run_0206_1925_seed25\\EALSTM_train_seed25.p',
             'AGRU_hps_12t':'run_0406_1450_seed26\\EALSTM_train_seed26.p',
             'AGRU_hps_13t':'run_0406_1744_seed27\\EALSTM_train_seed27.p',
             'AGRU_hps_14t':'run_0406_1857_seed28\\EALSTM_train_seed28.p',
             }

def pdf_KDE_gauss(x_samples,x_data,h):
    pdf = np.zeros(len(x_samples))
    N = len(x_data)
    dx = x_samples[1]-x_samples[0]
    for i in range(len(x_samples)):
        for j in range(N):
            pdf[i] = pdf[i]+np.exp(-(x_samples[i]-x_data[j])**2/(2*h*h))
        pdf[i] = pdf[i]*dx
    cte = 1/(N*np.sqrt(2*np.pi*h*h))
    pdf = pdf*cte
    return pdf

res_eval = {key:pd.read_pickle(filepath+filename) for key,filename in filenames.items()}
keys_eval = list(filenames.keys())
n_hps = int(len(keys_eval)/2)
keys_dev = keys_eval[:n_hps]
keys_train = keys_eval[n_hps:]

bench = 'AGRU_hps_8'
comparison = 'AGRU_hps_8t' #,'AGRU_hps_14'
res_bench = res_eval[bench]
res_comp = res_eval[comparison]

nse_basins_EALSTM = []
nse_basins_GRU_JF = []
basins = list(res_comp.keys())

nse_basins = np.empty((len(basins),len(res_eval)))

i = 0
for basin in basins:
    basin_nse_01 = calc_nse(res_bench[basin].qobs.values,res_bench[basin].qsim.values)
    basin_nse_02 = calc_nse(res_comp[basin].qobs.values,res_comp[basin].qsim.values)                        
    if i == 3:
        #fig = plt.figure('Basin qobs and qsim EALSTM')
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,9))
        #res['01073000'].plot(subplots=True, figsize=(15,6))
        res_bench[basin].plot(y=["qobs", "qsim"], 
                        title ='basin = '+basin+'; nse = '+str(basin_nse_01)+' '+bench+' 33 basins',
                        ax = axes[0])
        axes[0].set_ylabel('q')
        res_comp[basin].plot(y=["qobs", "qsim"], 
                        title ='basin = '+basin+'; nse = '+str(basin_nse_02)+' '+comparison+' 33 basins',
                        ax = axes[1])
        axes[1].set_ylabel('q')
        #res['01073000'].plot(x="R", y=["F10.7", "Dst"], style='.')    
        #plt.show()
    nse_basins_EALSTM.append(basin_nse_01)
    nse_basins_GRU_JF.append(basin_nse_02)
    for k in range(len(res_eval)):
        nse_basins[i,k] = calc_nse(res_eval[keys_eval[k]][basin].qobs.values,
                                   res_eval[keys_eval[k]][basin].qsim.values)
    i = i+1
    

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
ax = axes[0]

ax.hist(nse_basins_EALSTM, density=True, stacked=True)

ax.set_xlabel('NSE value')
ax.set_title('Histogram of '+bench+' 33 basins')
ax.set_xlim([-0.4,1])

ax = axes[1]
ax.hist(nse_basins_GRU_JF, density=True, stacked=True)
ax.set_xlabel('NSE value')
ax.set_title('Histogram of '+comparison+' 33 basins')
ax.set_xlim([-0.4,1])
#plt.show()

ax = axes[2]
ax.hist(nse_basins_EALSTM, density=True, histtype='step',
                           cumulative=True, label=bench+' CDF')
ax.hist(nse_basins_GRU_JF, density=True, histtype='step',
                           cumulative=True, label=comparison+' CDF')
ax.legend(loc='upper left')
ax.set_xlim([-0.4,1])
df = pd.DataFrame({bench:nse_basins_EALSTM,
                   comparison:nse_basins_GRU_JF})

print(df.describe())

dict_df_nse = {}
for i in range(len((res_eval))):
    dict_df_nse.update({keys_eval[i]:np.reshape(nse_basins[:,i],-1)})
df_nse = pd.DataFrame(dict_df_nse)
print(df_nse.describe())


fig = plt.figure('NSE_comparison')
fig.clf()
plt.plot(range(n_hps),df_nse[keys_train].median(),'-ob',label='Train')
plt.plot(range(n_hps),df_nse[keys_dev].median(),'-og',label='Dev')
plt.legend()
plt.ylabel('NSE median')
#plt.xlabel('set of hps')
plt.title('train/dev 33 basins and 1 year')
labelsx = ['AGRU_'+str(i) for i in range(0,n_hps-1)]
labelsx.insert(0,'EALSTM')
plt.xticks(range(0,n_hps),labelsx,rotation=60,ha='right')
plt.tight_layout()
plt.show()
gap = np.array(df_nse[keys_train].median()) - np.array(df_nse[keys_dev].median())



binwidth = 0.025
compare = np.array([0,1,2,6,12,14,9],dtype='int8')
keys_dev_c = [keys_dev[i] for i in compare]
keys_train_c = [keys_train[i] for i in compare]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
ax = axes[0][0]
for hps in keys_dev_c:
    ax.hist(df_nse[hps], density=True,label = hps,alpha=0.5,
            bins=np.arange(min(df_nse[hps]),
                           max(df_nse[hps])+binwidth, binwidth))
ax.legend(loc='upper left')
ax.set_xlabel('NSE value')
ax.set_title('NSE of 33 basins dev set (PDF)')
ax.set_xlim([-0.4,1])

ax = axes[0][1]
for hps in keys_train_c:
    ax.hist(df_nse[hps], density=True,label = hps,alpha=0.5,
            bins=np.arange(min(df_nse[hps]),
                           max(df_nse[hps])+binwidth, binwidth))
ax.legend(loc='upper left')
ax.set_xlabel('NSE value')
ax.set_title('NSE of 33 basins train set (PDF)')
ax.set_xlim([-0.4,1])


ax = axes[1][0]
for hps in keys_dev_c:
    ax.hist(df_nse[hps], density=True, label = hps, alpha=0.5,
            bins=np.arange(min(df_nse[hps]),
                           max(df_nse[hps])+binwidth, binwidth),
            cumulative = True, linestyle = '-', linewidth = 2)
ax.legend(loc='upper left')
ax.set_xlim([-0.4,1])
ax.set_xlabel('NSE value')  
ax.set_title('NSE of 33 basins dev set (CDF)')

ax = axes[1][1]   
for hps in keys_train_c:
    ax.hist(df_nse[hps], density=True, label = hps, alpha=0.5,
            bins=np.arange(min(df_nse[hps]),
                           max(df_nse[hps])+binwidth, binwidth),
            cumulative = True, linestyle = '--', linewidth = 2)
ax.legend(loc='upper left')
ax.set_xlim([-0.4,1])
ax.set_xlabel('NSE value')
ax.set_title('NSE of 33 basins train set (CDF)')
plt.tight_layout()
#plt.show()




binwidth = 0.025
compare = np.array([0,1,2,6,9,12,14],dtype='int8')
keys_dev_c = [keys_dev[i] for i in compare]
keys_train_c = [keys_train[i] for i in compare]
limx = [-0.1,1]
labels_leg = ['EALSTM', 'AGRU_0', 'AGRU_1', 'AGRU_5','AGRU_8','AGRU_11','AGRU_13']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,4))
ax = axes[1]
for i,hps in enumerate(keys_dev_c):
    Aedges = np.arange(min(df_nse[hps]),
                           max(df_nse[hps])+binwidth, binwidth)
    Acenters = (Aedges[:-1]+Aedges[1:])/2
    ax.plot(Acenters,pdf_KDE_gauss(Acenters,df_nse[hps],0.03), 
            linewidth = 2, label = labels_leg[i])
ax.legend(loc='upper left')
ax.set_xlabel('NSE value')
ax.set_title('NSE of 33 basins dev set (PDF)')
ax.set_xlim(limx)

ax = axes[0]
for i,hps in enumerate(keys_train_c):
    Aedges = np.arange(min(df_nse[hps]),
                           max(df_nse[hps])+binwidth, binwidth)
    Acenters = (Aedges[:-1]+Aedges[1:])/2
    ax.plot(Acenters,pdf_KDE_gauss(Acenters,df_nse[hps],0.03), 
            linewidth = 2 , label = labels_leg[i], linestyle = '--')
ax.legend(loc='upper left')
ax.set_xlabel('NSE value')
ax.set_title('NSE of 33 basins train set (PDF)')
ax.set_xlim(limx)


ax = axes[2]
for i,hps in enumerate(keys_dev_c):
    A, Aedges = np.histogram(df_nse[hps], bins = np.arange(min(df_nse[hps]),
                           max(df_nse[hps])+binwidth, binwidth))
    Acenters = (Aedges[:-1]+Aedges[1:])/2
    Acdf = np.cumsum(A)
    Acdf = Acdf/Acdf[-1]
    p = ax.plot(Acenters,Acdf, label = labels_leg[i], linestyle = '-', linewidth = 2)
    B, Bedges = np.histogram(df_nse[keys_train_c[i]], bins = np.arange(min(df_nse[keys_train_c[i]]),
                           max(df_nse[keys_train_c[i]])+binwidth, binwidth))
    Bcenters = (Bedges[:-1]+Bedges[1:])/2
    Bcdf = np.cumsum(B)
    Bcdf = Bcdf/Bcdf[-1]
    ax.plot(Bcenters,Bcdf, linestyle = '--', linewidth = 2,
            #label = keys_train_c[i],
            c=p[0].get_color())    
ax.legend(loc='upper left')
ax.set_xlim(limx)
ax.set_xlabel('NSE value')  
ax.set_title('NSE of 33 basins dev set (CDF), train set on dash')