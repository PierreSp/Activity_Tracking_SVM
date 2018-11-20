import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

##############################################
###           Data Processing              ###
##############################################


def load_data(filename):
    data_2016 = sio.loadmat('data2016.mat')
    mdata = data_2016["data"]
    mdtype = mdata.dtype
    ndata = {n: mdata[n][0, 0] for n in mdtype.names}  # Extract col names
    return ndata


def calc_gravitation_components(data):
    """Takes measurements of one dimension and returns the mean """
    mean_val = np.mean(data)
    return mean_val


def _calc_net_acceleration(x, y, z):
    """Takes measurements of all dimension and returns the net acceleration,
    which is the norm of the measurments (per window)"""
    net_acceleration = np.sqrt(x**2 + y**2 + z**2)
    return net_acceleration


def calc_net_acceleration_dcfree(x, y, z):
    """Takes measurements and returns net acceleration without dc component"""
    net_acceleration = _calc_net_acceleration(x, y, z)
    net_acceleration_dc = net_acceleration - np.mean(net_acceleration)
    return net_acceleration_dc


def create_features(data, label):
    featuredict = {}  # Dict for all features
    ds_d_type = data[label].dtype
    ds_d_data = {n: data[label][n][0, 0] for n in ds_d_type.names}  # Extract data with colum names (x,y,z)
    N = len(ds_d_data["x"])  # Total number of measurements
    # Add features to dict
    featuredict["label"] = np.repeat(label, N)
    featuredict["x_mean"] = calc_gravitation_components(ds_d_data["x"])
    featuredict["y_mean"] = calc_gravitation_components(ds_d_data["y"])
    featuredict["z_mean"] = calc_gravitation_components(ds_d_data["z"])
    print(calc_net_acceleration_dcfree(ds_d_data["x"],ds_d_data["y"],ds_d_data["z"]))


# Load data
data_2016 = sio.loadmat('data2016.mat')
mdata = data_2016["data"]
mdtype = mdata.dtype
ndata = {n: mdata[n][0, 0] for n in mdtype.names} # Extract data with colum names
# Names are ['brushing', 'drinking', 'shoe', 'writing']
