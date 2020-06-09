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

filepath = 'C:\\Users\\josue\\project_CS230\\ealstm_regional_modeling\\runs\\'
filenames = {'EALSTM':    'run_EALSTM\\EALSTM_seed200.p',
             'AGRU_hps_0':'run_AWS_AGRU\\ealstm_seed200.p',             
             'AGRU_hps_1':'run_3005_AWS_AGRU\\ealstm_seed300.p',
             # #'AGRU_hps_2':'run_3005_1119_seed425359\\EALSTM_seed425359.p',
             # #'AGRU_hps_3':'run_3005_1319_seed219836\\EALSTM_seed219836.p',
             # #'AGRU_hps_4':'run_3005_1906_seed98194\\EALSTM_seed98194.p',
             'AGRU_hps_5':'run_0106_AWS_AGRU\\ealstm_seed400.p',
             # #'AGRU_hps_6':'run_3005_1536_seed984683\\EALSTM_seed984683.p',
             # #'AGRU_hps_7':'run_0106_1958_seed22\\EALSTM_seed22.p',
             'AGRU_hps_8_e15':'run_0706_AWS_AGRU\\ealstm_seed700_e15.p',
             'AGRU_hps_8_e30':'run_0706_AWS_AGRU\\ealstm_seed700_e30.p',
             # #'AGRU_hps_9':'run_0206_1453_seed775825\\EALSTM_seed775825.p',
             # #'AGRU_hps_10':'run_0206_1812_seed24\\EALSTM_seed24.p',
             'AGRU_hps_11':'run_0306_AWS_AGRU\\ealstm_seed500.p',
             # #'AGRU_hps_12':'run_0406_1450_seed26\\EALSTM_seed26.p',
             'AGRU_hps_13':'run_0506_AWS_AGRU\\ealstm_seed600.p',
             # #'AGRU_hps_14':'run_0406_1857_seed28\\EALSTM_seed28.p',
             
             'EALSTMt':    'run_EALSTM\\ealstm_train_seed100.p',
             'AGRU_hps_0t':'run_AWS_AGRU\\ealstm_train_seed200.p',
             'AGRU_hps_1t':'run_3005_AWS_AGRU\\ealstm_train_seed300.p',
             # #'AGRU_hps_2t':'run_3005_1119_seed425359\\EALSTM_train_seed425359.p',
             # #'AGRU_hps_3t':'run_3005_1319_seed219836\\EALSTM_train_seed219836.p',
             # #'AGRU_hps_4t':'run_3005_1906_seed98194\\EALSTM_train_seed98194.p',
             'AGRU_hps_5t':'run_0106_AWS_AGRU\\ealstm_train_seed400.p',
             # #'AGRU_hps_6t':'run_3005_1536_seed984683\\EALSTM_train_seed984683.p',
             # #'AGRU_hps_7t':'run_0106_1958_seed22\\EALSTM_train_seed22.p',
             'AGRU_hps_8_e15t':'run_0706_AWS_AGRU\\ealstm_train_seed700_e15.p',
             'AGRU_hps_8_e30t':'run_0706_AWS_AGRU\\ealstm_train_seed700_e30.p',
             # #'AGRU_hps_9t':'run_0206_1453_seed775825\\EALSTM_train_seed775825.p',             
             # #'AGRU_hps_10t':'run_0206_1812_seed24\\EALSTM_train_seed24.p',
             'AGRU_hps_11t':'run_0306_AWS_AGRU\\ealstm_train_seed500.p',
             # #'AGRU_hps_12t':'run_0406_1450_seed26\\EALSTM_train_seed26.p',
             'AGRU_hps_13t':'run_0506_AWS_AGRU\\ealstm_train_seed600.p'
             # #'AGRU_hps_14t':'run_0406_1857_seed28\\EALSTM_train_seed28.p',
             }



res_eval = {key:pd.read_pickle(filepath+filename) for key,filename in filenames.items()}
keys_eval = list(filenames.keys())
n_hps = int(len(keys_eval)/2)
keys_dev = keys_eval[:n_hps]
keys_train = keys_eval[n_hps:]

bench = 'AGRU_hps_8_e15'
comparison = 'AGRU_hps_8_e15t' #,'AGRU_hps_14'
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
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,9))
        res_bench[basin].plot(y=["qobs", "qsim"], 
                        title ='basin = '+basin+'; nse = '+str(basin_nse_01)+' '+bench+' 531 basins on dev set',
                        ax = axes[0])
        axes[0].set_ylabel('q')
        res_comp[basin].plot(y=["qobs", "qsim"], 
                        title ='basin = '+basin+'; nse = '+str(basin_nse_02)+' '+comparison+' 531 basins on train set',
                        ax = axes[1])
        axes[1].set_ylabel('q')
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
ax.set_title('Histogram of '+bench+' 531 basins')
ax.set_xlim([-0.4,1])

ax = axes[1]
ax.hist(nse_basins_GRU_JF, density=True, stacked=True)
ax.set_xlabel('NSE value')
ax.set_title('Histogram of '+comparison+' 531 basins')
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
compare = np.array([0,1,2,3,4,6,7],dtype='int8')
keys_dev_n = [keys_dev[i] for i in compare]
keys_train_n = [keys_train[i] for i in compare]
plt.plot(range(n_hps-1),df_nse[keys_train_n].median(),'-ob',label='Train')
plt.plot(range(n_hps-1),df_nse[keys_dev_n].median(),'-og',label='Dev')
plt.legend()
plt.ylabel('NSE median')
plt.title('train(9yrs)/dev(10yrs) 531 basins')
labelsx = keys_dev
labelsx = ['EALSTM', 'AGRU_0', 'AGRU_1', 'AGRU_5','AGRU_8','AGRU_11','AGRU_13']
plt.xticks(range(0,n_hps-1),labelsx,rotation=60,ha='right')
plt.tight_layout()
plt.show()
gap = np.array(df_nse[keys_train].median()) - np.array(df_nse[keys_dev].median())



binwidth = 0.025
compare = np.array([0,4,5],dtype='int8')
limx = [-0.1,1]
keys_dev_c = [keys_dev[i] for i in compare]
keys_train_c = [keys_train[i] for i in compare]

labels_leg = ['EALSTM', 'AGRU_8_it15', 'AGRU_8_it30']
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
ax.set_title('NSE distribution of 531 basins on dev set')
ax.set_xlim(limx)

ax = axes[0]
for i,hps in enumerate(keys_train_c):
    Aedges = np.arange(min(df_nse[hps]),
                           max(df_nse[hps])+binwidth, binwidth)
    Acenters = (Aedges[:-1]+Aedges[1:])/2
    ax.plot(Acenters,pdf_KDE_gauss(Acenters,df_nse[hps],0.03), 
            linewidth = 2 ,label = labels_leg[i], linestyle = '--')
ax.legend(loc='upper left')
ax.set_xlabel('NSE value')
ax.set_title('NSE distribution of 531 basins on train set')
ax.set_xlim(limx)


ax = axes[2]
for i,hps in enumerate(keys_dev_c):
    A, Aedges = np.histogram(df_nse[hps], bins = np.arange(min(df_nse[hps]),
                           max(df_nse[hps])+binwidth, binwidth))
    Acenters = (Aedges[:-1]+Aedges[1:])/2
    Acdf = np.cumsum(A)
    Acdf = Acdf/Acdf[-1]
    p = ax.plot(Acenters,Acdf, label = labels_leg[i] ,linestyle = '-', linewidth = 2)
    B, Bedges = np.histogram(df_nse[keys_train_c[i]], bins = np.arange(min(df_nse[keys_train_c[i]]),
                           max(df_nse[keys_train_c[i]])+binwidth, binwidth))
    Bcenters = (Bedges[:-1]+Bedges[1:])/2
    Bcdf = np.cumsum(B)
    Bcdf = Bcdf/Bcdf[-1]
    ax.plot(Bcenters,Bcdf, linestyle = '--', linewidth = 2,
            c=p[0].get_color())    
ax.legend(loc='upper left')
ax.set_xlim(limx)
ax.set_xlabel('NSE value')  
ax.set_title('Cumulative NSE distribution of 531 basins:\ndev set (full line), train set (dashed line)')


compare = np.array([0,1,2,3,4,6,7],dtype='int8')
keys_dev_c = [keys_dev[i] for i in compare]
keys_train_c = [keys_train[i] for i in compare]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,4))
ax = axes[1]
labels_leg = ['EALSTM', 'AGRU_0', 'AGRU_1', 'AGRU_5','AGRU_8','AGRU_11','AGRU_13']
for i,hps in enumerate(keys_dev_c):
    Aedges = np.arange(min(df_nse[hps]),
                           max(df_nse[hps])+binwidth, binwidth)
    Acenters = (Aedges[:-1]+Aedges[1:])/2
    ax.plot(Acenters,pdf_KDE_gauss(Acenters,df_nse[hps],0.03), 
            linewidth = 2, label = labels_leg[i])
ax.legend(loc='upper left')
ax.set_xlabel('NSE value')
ax.set_title('NSE distribution of 531 basins on dev set')
ax.set_xlim(limx)

ax = axes[0]
for hps in keys_train_c:
    Aedges = np.arange(min(df_nse[hps]),
                           max(df_nse[hps])+binwidth, binwidth)
    Acenters = (Aedges[:-1]+Aedges[1:])/2
    ax.plot(Acenters,pdf_KDE_gauss(Acenters,df_nse[hps],0.03), 
            linewidth = 2 ,label = labels_leg[i],linestyle = '--')
ax.legend(loc='upper left')
ax.set_xlabel('NSE value')
ax.set_title('NSE distribution of 531 basins on train set')
ax.set_xlim(limx)


ax = axes[2]
for i,hps in enumerate(keys_dev_c):
    A, Aedges = np.histogram(df_nse[hps], bins = np.arange(min(df_nse[hps]),
                           max(df_nse[hps])+binwidth, binwidth))
    Acenters = (Aedges[:-1]+Aedges[1:])/2
    Acdf = np.cumsum(A)
    Acdf = Acdf/Acdf[-1]
    p = ax.plot(Acenters,Acdf, label = labels_leg[i],linestyle = '-', linewidth = 2)
    B, Bedges = np.histogram(df_nse[keys_train_c[i]], bins = np.arange(min(df_nse[keys_train_c[i]]),
                           max(df_nse[keys_train_c[i]])+binwidth, binwidth))
    Bcenters = (Bedges[:-1]+Bedges[1:])/2
    Bcdf = np.cumsum(B)
    Bcdf = Bcdf/Bcdf[-1]
    ax.plot(Bcenters,Bcdf, linestyle = '--', linewidth = 2,
            c=p[0].get_color())        
ax.legend(loc='upper left')
ax.set_xlim(limx)
ax.set_xlabel('NSE value')  
ax.set_title('Cumulative NSE distribution of 531 basins:\ndev set (full line), train set (dashed line)')