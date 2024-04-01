import numpy as np
import os
from tool import get_peaks, s_input1, s_input3, s_input6


def trep(n):
    if n <= 1:
        return 0.95
    elif n == 3:
        return 0.5
    else:
        return 0.3


def cell_volume(t):
    V0 = 1
    volume = V0 * 2**t
    return volume


def burst_size(b, size=1):
    j_values = np.arange(1, 25)
    log_probabilities = j_values * np.log(b) - (j_values + 1) * np.log(b + 1)
    log_probabilities -= np.max(log_probabilities)
    probabilities = np.exp(log_probabilities)
    probabilities /= np.sum(probabilities)
    j = np.random.choice(j_values, size=size, p=probabilities)
    return j


def rate(sc, vc, cc, tc, ratio, fmiu):

    rc = np.zeros(18)
    ni, nr, nrr, nir, nrrr, nzm, nzp = cc
    kir, Kir, kr, krr, Krr = 20, 0.6, 5, 7, 3.5
    ki_r, krrr = 300, 50
    kzm, Kzm, kzp = 5, 3.5, 160
    dp1, dp2, dp3 = 0.5, 1, 20

    sinr_position = 218
    slrr_position = 302
    it = 0.05

    PHASE = 0
    if tc < ratio * (360 - slrr_position) / 180 + it:
        PHASE = 1
    elif ratio * (360 - slrr_position) / 180 + it <= tc <= ratio * (360 - sinr_position) / 180 + it:
        PHASE = 2
    else:
        PHASE = 3

    rc[0] = np.round(vc / fmiu * kir * sc ** 2 / (sc ** 2 + Kir ** 2), decimals=5)
    rc[1] = rc[0] + kr * vc / fmiu
    rc[2] = np.round(vc / fmiu * krr * Krr ** 4 / (Krr ** 4 + (nr / vc) ** 4), decimals=5)
    rc[3] = np.round(ki_r * ni * (nr / vc), decimals=5)
    rc[4] = np.round(krrr * (nr / vc) * nrr, decimals=5)
    rc[5] = np.round(vc / fmiu * kzm * Kzm ** 4 / (Kzm ** 4 + (nr / vc) ** 4), decimals=5)
    rc[6] = np.round(kzp * nzm, decimals=5)
    rc[7] = np.round(dp1 * ni, decimals=5)
    rc[8] = np.round(dp1 * nr, decimals=5)
    rc[9] = np.round(dp2 * nrr, decimals=5)
    rc[10] = np.round(dp1 * nir, decimals=5)
    rc[11] = np.round(dp1 * nrrr, decimals=5)
    rc[12] = np.round(dp3 * nzm, decimals=5)
    rc[13] = np.round(dp1 * nzp, decimals=5)

    if PHASE == 2:
        rc[14] = np.round(vc / fmiu * krr * Krr ** 4 / (Krr ** 4 + (nr / vc) ** 4), decimals=5)
    if PHASE == 3:
        rc[14] = np.round(vc / fmiu * krr * Krr ** 4 / (Krr ** 4 + (nr / vc) ** 4), decimals=5)
        rc[15] = np.round(vc / fmiu * kir * sc ** 2 / (sc ** 2 + Kir ** 2), decimals=5)
        rc[16] = rc[0] + kr * vc / fmiu
        rc[17] = np.round(kzm * Kzm ** 4 / (Kzm ** 4 + (nr / vc) ** 4), decimals=5)

    return np.around(rc, decimals=5)


def renew_concentration(selected_index):
    renew_dict = {0: np.array([1, 0, 0, 0, 0, 0, 0]), 1: np.array([0, 1, 0, 0, 0, 0, 0]),
                  2: np.array([0, 0, 1, 0, 0, 0, 0]), 3: np.array([-1, -1, 0, 1, 0, 0, 0]),
                  4: np.array([0, -1, -1, 0, 1, 0, 0]), 5: np.array([0, 0, 0, 0, 0, 1, 0]),
                  6: np.array([0, 0, 0, 0, 0, 0, 1]), 7: np.array([-1, 0, 0, 0, 0, 0, 0]),
                  8: np.array([0, -1, 0, 0, 0, 0, 0]), 9: np.array([0, 0, -1, 0, 0, 0, 0]),
                  10: np.array([0, 0, 0, -1, 0, 0, 0]), 11: np.array([0, 0, 0, 0, -1, 0, 0]),
                  12: np.array([0, 0, 0, 0, 0, -1, 0]), 13: np.array([0, 0, 0, 0, 0, 0, -1]),
                  14: np.array([0, 0, 1, 0, 0, 0, 0]), 15: np.array([1, 0, 0, 0, 0, 0, 0]),
                  16: np.array([0, 1, 0, 0, 0, 0, 0]), 17: np.array([0, 0, 0, 0, 0, 1, 0]),
                  }

    nc = renew_dict[int(selected_index)]
    b1, b2, b3, b4 = 10, 10, 5, 5
    if selected_index == 0:
        nc = renew_dict[int(selected_index)] * burst_size(b1)
    if selected_index == 1:
        nc = renew_dict[int(selected_index)] * burst_size(b2)
    if selected_index == 2:
        nc = renew_dict[int(selected_index)] * burst_size(b3)
    if selected_index == 5:
        nc = renew_dict[int(selected_index)] * burst_size(b4)
    if selected_index == 14:
        nc = renew_dict[int(selected_index)] * burst_size(b3)
    if selected_index == 15:
        nc = renew_dict[int(selected_index)] * burst_size(b1)
    if selected_index == 16:
        nc = renew_dict[int(selected_index)] * burst_size(b2)
    if selected_index == 17:
        nc = renew_dict[int(selected_index)] * burst_size(b4)

    return nc


def main(steps, n):

    t = np.array([0])
    concentration = np.array([[0], [40], [0], [0], [0], [0], [0]])
    ratio = trep(n)

    for i in range(steps):

        tc0 = t[-1]
        tc = tc0 % 1
        cc = concentration[:, -1]

        if n == 1:
            sc = s_input1(tc)
            fmiu = 1.8
        elif n == 3:
            sc = s_input3(tc)
            fmiu = 1.6
        else:
            sc = s_input6(tc)
            fmiu = 1.5

        vc = cell_volume(tc)

        a = rate(sc, vc, cc, tc, ratio, fmiu)
        a = np.around(a, decimals=5)

        a0 = a.sum()
        a /= a0
        t0 = (- np.log(np.random.random())) / a0
        t0 /= n

        indices = np.arange(len(a))
        try:
            selected_index = np.random.choice(indices, p=a)
        except ValueError:
            print(sc, vc, cc, tc, ratio)
            print(a)
            selected_index = np.random.choice(indices, p=a)
            break

        if tc + t0 > 1:
            concentration = np.column_stack((concentration, cc // 2))
            t = np.append(t, int(tc0) + 1)
        else:
            cn0 = renew_concentration(selected_index)
            cn = cc + cn0
            concentration = np.column_stack((concentration, cn))
            t = np.append(t, tc0 + t0)

    return t, concentration


if __name__ == '__main__':

    for n in [1, 3, 6]:

        directory = 'simulation/{}/'.format(n)

        if not os.path.exists(directory):
            os.makedirs(directory)

        peak_widths_total = []
        peak_gaps_total = []

        for i in range(1000):

            t, c = main(int(n*30000), n)

            np.savez(os.path.join(directory, 'simulation{}.npz'.format(i + 1)), t=t, c=c)

            trace = c[6]
            filtered_peaks, _, filtered_peak_width_l, filtered_peak_width_r = get_peaks(trace)
            if len(filtered_peaks) > 0:
                peak_width_r_t = t[filtered_peak_width_r]
                peak_width_l_t = t[filtered_peak_width_l]
                merged_peak_width_l_t = []
                merged_peak_width_r_t = []
                i = 0
                while i < len(filtered_peaks):
                    peak_l_t = peak_width_l_t[i]
                    peak_r_t = peak_width_r_t[i]
                    if i < len(filtered_peaks) - 1:
                        next_peak_l_t = peak_width_l_t[i + 1]
                        if next_peak_l_t - peak_r_t < 1:
                            peak_r_t = peak_width_r_t[i + 1]
                            i += 1
                    merged_peak_width_l_t.append(peak_l_t)
                    merged_peak_width_r_t.append(peak_r_t)
                    i += 1
                merged_peak_width_l_t = np.array(merged_peak_width_l_t)
                merged_peak_width_r_t = np.array(merged_peak_width_r_t)
                peak_widths = merged_peak_width_r_t - merged_peak_width_l_t
                peak_gaps = merged_peak_width_l_t[1:] - merged_peak_width_l_t[:-1]

                peak_widths_total.extend(peak_widths)
                peak_gaps_total.extend(peak_gaps)

        np.savez(os.path.join(directory, 'peak_data.npz'),
                 peak_widths=np.array(peak_widths_total), peak_gaps=np.array([peak_gaps_total]))

