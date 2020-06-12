""" 
Description:
    Utility functions for data preparation.
    
Created by Ehsan Haghighat on 6/10/20.
"""


import os, time
import sys
import numpy as np
import pandas as pd 
from scipy.interpolate import griddata

RADI = 50.0


def eval_mu_sig(X):
    return X.mean(), X.std()

def std(X, mu, sig): 
    return (X-mu)/sig

def invstd(X, mu, sig):
    return X*sig + mu

def get_data():
    file_path = "plate_with_hole_plastic_output_100x100_p4_mm.txt"
    data = pd.read_csv(file_path, sep='\s+', skiprows=9, dtype='float64')
    return data

def get_data_max():
    data = get_data()
    xcoord = data.x.values
    ycoord = data.y.values
    training_data_ids = np.where((xcoord**2 + ycoord**2 - RADI**2).reshape(-1) > 0)[0]
    data_max = {}
    for v in data.keys():
        data_max[v] = abs(data[v].values[training_data_ids]).max()
    return data_max

def get_training_data(ndata=None, adaptive_sampling=False):
    """ get_training_data
      Inputs:
          ndata: number of training points.
                 defaulted to all available samples. 
          adaptive_sampling: pick more points at locations with high-strains. 
                             Defaulted to False.
      Returns:
          mu_sig: normalization values (mu, sig) for each component. 
          data_s: nomalized data for training.
    """
    data = get_data()

    xcoord = data.x.values
    ycoord = data.y.values
    training_data_ids = np.where((xcoord**2 + ycoord**2 - RADI**2).reshape(-1) > 0)[0]
    
    if ndata is not None:
      if adaptive_sampling == False:
          training_data_ids = np.random.choice(
              training_data_ids, 
              ndata, 
              replace=False
          )
      else:
          prob = np.sqrt(sum([data[v].values[training_data_ids]**2 for v in ['exx', 'eyy', 'ezz', 'exy', 'exy']]))
          prob = prob / prob.sum()
          training_data_ids = np.random.choice(
                training_data_ids, 
                ndata, 
                p = prob,
                replace=False
          )
        
    xcoord = xcoord[training_data_ids]
    ycoord = ycoord[training_data_ids]

    mu_sig = {'x': [0., 1.], 'y':[0., 1.]}
    data_s = {'x': xcoord, 'y':ycoord}
    for v in ['u', 'v', 'sxx', 'syy', 'szz', 'sxy', 'exx', 'eyy', 'exy']:
        dt_val = data[v].values[training_data_ids]
        mu, sig = eval_mu_sig(dt_val)
        mu_sig[v] = [mu, sig]
        data_s[v] = std(dt_val, mu, sig)

    return mu_sig, data_s
  
def get_test_data(nx=200, ny=400):
    data = get_data()

    XMIN, XMAX = data.x.values.min(), data.x.values.max()
    YMIN, YMAX = data.y.values.min(), data.y.values.max()
    
    Xmesh_plot = np.linspace(XMIN, XMAX, nx)
    Ymesh_plot = np.linspace(YMIN, YMAX, ny)
    
    X_plot, Y_plot = np.meshgrid(Xmesh_plot, Ymesh_plot)
    
    input_plot = [X_plot.reshape(-1, 1), Y_plot.reshape(-1, 1)]
    nan_ids = np.where(input_plot[0]**2 + input_plot[1]**2 - RADI**2 < 0.0)[0].reshape(-1,1)
    
    return X_plot, Y_plot, nan_ids
  
