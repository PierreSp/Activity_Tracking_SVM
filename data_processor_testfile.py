import numpy as np
import pandas as pd
from scipy.io import loadmat

TIMEWINDOW = 20 * 128  + 1
STEP = 128
FREQUENCY = 128


def _energy25(r, freq):
    N = len(r)
    R = np.abs(np.fft.fft(r.flatten()))**2
    R[0] = 0
    frequencies = [i * freq / N for i in range(N)]
    CR = R.cumsum()
    CR /= CR.max()
    index = np.where(CR < 0.25/2)[0][-1]
    f25 = frequencies[index]
    return f25


def _energy75(r, freq):
    N = len(r)
    R = np.abs(np.fft.fft(r.flatten()))**2
    R[0] = 0
    frequencies = [i * freq / N for i in range(N)]
    CR = R.cumsum()
    CR /= CR.max()
    index = np.where(CR < 0.75/2)[0][-1]
    f75 = frequencies[index]
    return f75


def _rolling_skew(df, window, step):
    rl_skew = df.loc[:, ["r"]].rolling(window=window).std()
    return rl_skew


def _rolling_std(df, window, step):
    rl_skew = df.loc[:, ["r"]].rolling(window=window).skew()
    return rl_skew


def _rolling_f75(df, window, step, freq):
    rl_f75 = df.loc[:, ["r"]].rolling(window=window).apply(
        lambda x: _energy75(x, freq), raw=True)
    return rl_f75


def _rolling_f25(df, window, step, freq):
    rl_f25 = df.loc[:, ["r"]].rolling(window=window).apply(
        lambda x: _energy25(x, freq), raw=True)
    return rl_f25


def _rolling_means(df, window, step):
    rl_means = df.loc[:, ["x", "y", "z"]].rolling(window=window).mean()
    return rl_means


def load_data(filename):
    data = loadmat(filename)
    datadict = {"x": data["data"]["x"][0][0].ravel(),
                "y": data["data"]["y"][0][0].ravel(),
                "z": data["data"]["z"][0][0].ravel()}

    r_value = np.sqrt(datadict["x"]**2 + datadict["y"]**2 + datadict["z"]**2)
    datadict["r"] = r_value - np.mean(r_value)
    df = pd.DataFrame.from_dict(datadict)
    rl_means = _rolling_means(df, TIMEWINDOW, STEP)
    df_windows = pd.DataFrame({
        'gx': rl_means.x.values,
        'gy': rl_means.y.values,
        'gz': rl_means.z.values,
        'std': _rolling_std(df, TIMEWINDOW, STEP).r.values,
        'skewness': _rolling_skew(df, TIMEWINDOW, STEP).r.values,
        'f25': _rolling_f25(df, TIMEWINDOW, STEP, FREQUENCY).r.values,
        'f75': _rolling_f75(df, TIMEWINDOW, STEP, FREQUENCY).r.values,
    })
    df_windows = df_windows.drop(range(TIMEWINDOW), axis=0)
    labels = data["data"]["Label"][0][0].ravel() == 4
    return labels, df_windows
