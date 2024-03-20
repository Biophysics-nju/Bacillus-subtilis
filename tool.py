import os
import time
import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks
from scipy.integrate import odeint
import scipy.io
from scipy.interpolate import interp1d


def mkdir(path):
    is_exits = os.path.exists(path)

    if not is_exits:
        os.makedirs(path)  # 不存在则创建该目录
        return path
    else:
        return path


def get_peaks(trace):
    trace = signal.savgol_filter(trace, 21, 3)
    peaks, _ = find_peaks(trace, height=8, distance=50, width=60, prominence=6)
    peak_width = signal.peak_widths(trace, peaks, rel_height=0.98)
    peaks, width_heights, peak_width_l, peak_width_r = peaks, peak_width[1], peak_width[2], peak_width[3]
    peak_width_l = peak_width_l.astype(np.int32)
    peak_width_r = peak_width_r.astype(np.int32)

    filtered_peaks = []
    filtered_width_heights = []
    filtered_peak_width_l = []
    filtered_peak_width_r = []

    for i in range(len(peaks)):
        is_inside_peak = False
        for j in range(len(peaks)):
            if i != j:
                if peak_width_l[i] >= peak_width_l[j] and peak_width_r[i] <= peak_width_r[j]:
                    is_inside_peak = True
                    break
        if not is_inside_peak:
            filtered_peaks.append(peaks[i])
            filtered_width_heights.append(width_heights[i])
            filtered_peak_width_l.append(peak_width_l[i])
            filtered_peak_width_r.append(peak_width_r[i])

    filtered_peaks = np.array(filtered_peaks)
    filtered_width_heights = np.array(filtered_width_heights)
    filtered_peak_width_l = np.array(filtered_peak_width_l)
    filtered_peak_width_r = np.array(filtered_peak_width_r)

    return filtered_peaks, filtered_width_heights, filtered_peak_width_l, filtered_peak_width_r


def residual_sum_of_squares(data, expected):
    residuals = np.array(data) - np.array(expected)
    rss = np.sum(residuals ** 2)
    return rss


def s_input1(ts1):
    return 0.11


mat_data3 = scipy.io.loadmat("3hour.mat")
ts3 = mat_data3['ts'][100:]-18
ts3 = ts3.flatten()
ap3 = mat_data3['Ap'][100:]
ap3 = ap3.flatten()
s_input3 = interp1d(ts3, ap3)
def ds_dt(s, t, s_input3):
    return 0.55 * (s_input3(t) - s)
s0 = 0.45
t = np.linspace(0, 40, 4000)
s_solution = odeint(ds_dt, s0, t, args=(s_input3,))
t = t.flatten()
s_solution = s_solution.flatten()
s_input3 = interp1d((t[3000:]-30.0076)/3, s_solution[3000:])

mat_data6 = scipy.io.loadmat("6hour.mat")
ts6 = mat_data6['ts'][100:]-18
ts6 = ts6.flatten()
ap6 = mat_data6['Ap'][100:]
ap6 = ap6.flatten()
s_input6 = interp1d(ts6, ap6)
def ds_dt(s, t, s_input6):
    return 0.4 * (s_input6(t) - s)
s0 = 0.45
t = np.linspace(0, 40, 4000)
s_solution = odeint(ds_dt, s0, t, args=(s_input6,))
t = t.flatten()
s_solution = s_solution.flatten()
s_input6 = interp1d((t[2000:]-20.00501)/6, s_solution[2000:])


if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
