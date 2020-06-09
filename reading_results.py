# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:40:40 2020

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



filename = 'ealstm_seed101.p'
res = pd.read_pickle(filename)

nse_basins = []
basins = list(res.keys())
i = 0
for basin in basins:
    basin_nse = calc_nse(res[basin].qobs.values,res[basin].qsim.values)
    if i == 2:
        fig = plt.figure('Basin qobs and qsim')
        res[basin].plot(figsize=(15,4))
        #res['01073000'].plot(subplots=True, figsize=(15,6))
        res[basin].plot(y=["qobs", "qsim"], figsize=(15,4))
        #res['01073000'].plot(x="R", y=["F10.7", "Dst"], style='.')    
        plt.title('basin = '+basin+'; nse = '+str(basin_nse)) 
        plt.show()
    nse_basins.append(basin_nse)
    i = i+1
    
fig = plt.figure('Nse dist')
plt.hist(nse_basins)
