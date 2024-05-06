import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def get_signal_peaks(signal):
    """
    See docs
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    Useful filters (arguments) may include:
    - distance     (int) distance between consecutive peaks in indices (>= 1)
    - prominence (float) see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html
    - height     (float) peak >= height
    """
    peaks, properties = find_peaks(signal, height=1e-7)
    return peaks


def get_signal_troughs(signal):
    peaks, properties = find_peaks(-signal)
    return peaks


def convert_peaks_abs_to_reldiff(peak_vals):
    """
    see Eq. (1) in masters thesis
    note: could replace by
        a_i+1 - a_i = np.diff(a)
    """
    reldiff = 1 - peak_vals[1:] / peak_vals[0:-1]
    return reldiff


def time_to_habituate(times, output, num_pulses_applied, tth_threshold=0.01):
    """
    Refer to docstring for preprocess_signal_habituation()

    Examples for TTH:
        if all peaks same height (reldiff = 0) --> tth_discrete = 0  (i.e. the 0th pulse already H)
        if the first peak onward are identical --> tth_discrete = 1  (i.e. the 1st pulse is now H)
    """
    does_habituate, reldiff, peaks, troughs = preprocess_signal_habituation(times, output, num_pulses_applied, tth_threshold=tth_threshold)

    tth_continuous = None
    tth_discrete = None

    if does_habituate:
        """
        Issue to address: if the peaks are initially increasing before decreasing, 
            we want to ignore that initial increase for computiong TTH
        Method: 
        - find K the index where reldiff first becomes positive
        - assert that TTH index is >= k
        """
        first_positive_reldiff = np.argwhere(reldiff > 0)[0, 0]
        peaks_below_threshold = np.argwhere(reldiff <= tth_threshold)
        #print(reldiff)
        #print(peaks_below_threshold)
        #print(first_positive_reldiff)
        if len(peaks_below_threshold) > 0:  # i.e. TTH_THRESHOLD is met: the curve has habituated, pick the first appropriate timepoint
            npulse_idx = peaks_below_threshold[first_positive_reldiff, 0]  # note: the accessing index > 0 if peaks initially increasing
            tth_discrete = npulse_idx  # see docstring for reason no "val + 1" here
            tth_time_idx = peaks[tth_discrete]
            tth_continuous = times[tth_time_idx]

    print('Warning: tth_discrete definition assumes len(peaks) ~ npulse; use tth_continuous instead')
    return does_habituate, tth_continuous, tth_discrete, reldiff, peaks, troughs


def preprocess_signal_habituation(times, output, num_pulses_applied, tth_threshold=0.01):
    """
    Definition: The signal is said to have habituated (properly) if:
      1) there are at least --------------- 5 peaks
      2) they are monotonic after the  ---- 3rd peak
      3) len(peaks) == num_pulses_applied

    Note:
        - arg "times" is currently unused, but could be used for various conditions
        - arg num_pulses_applied will be different if stimulus is
            stimulus_pulsewave               -- np.floor((t2-t1)/period_S1)
            stimulus_pulsewave_of_pulsewaves -- total npulses is less than above due to "rest" periods

    Returns:
        does_habituate (bool) does the output signal satisfy habituation criteria?
               reldiff  (arr) 1 - peak_vals[1:] / peak_vals[0:-1]
                 peaks  (arr) array indices of peaks of the output signal
               troughs  (arr) as above but for minima
    """
    x = output  # alias

    peaks = get_signal_peaks(x)
    troughs = get_signal_troughs(x)
    peak_vals = x[peaks]
    trough_vals = x[troughs]
    reldiff = convert_peaks_abs_to_reldiff(peak_vals)
    npdiff = np.diff(peak_vals)  # gives a_i+1 - a_i

    # filtering of the reldiff array: set values which are below abs(threshold) to 0.0
    reldiff_jitter_chopped = np.where(
        (-tth_threshold < reldiff) * (reldiff < tth_threshold),
        0, reldiff)

    peaks_argmax = np.argmax(peak_vals)

    cond_1 = len(peaks) > 5
    #cond_2 = np.all(npdiff[3:] <= 0)
    cond_2 = np.all(reldiff_jitter_chopped[peaks_argmax:] >= 0)  # need to 'chop' the values of reldiff which are below some numeric tolerance and after TTH
    cond_3 = (num_pulses_applied - 1 <= len(peaks) <= num_pulses_applied + 1)
    cond_4 = peaks_argmax < 5
    cond_5 = peak_vals[-1] < peak_vals[peaks_argmax] * 0.9

    positive_reldiff = np.argwhere(reldiff > 0)
    if len(positive_reldiff) > 0:
        cond_6 = True
        first_positive_reldiff = positive_reldiff[0,0]
        peaks_below_threshold = np.argwhere(reldiff <= tth_threshold)
        # now need to check that there is a positive entry relating to the peaks_below_threshold argwhere findings
        #  (e.g. if first_positive_reldiff == 2, and len(peaks_below_threshold) == 3,
        #        it implies the third entry of the peaks_below_threshold is non-negative)
        if len(peaks_below_threshold) > first_positive_reldiff:
            cond_7 = True
        else:
            cond_7 = False
    else:
        cond_6 = False
        cond_7 = False

    print('H1 conditions:')
    print('\tcond_1 (npeaks > K) |', cond_1)
    print('\tcond_2 (decr. after max peak) |', cond_2)
    print('\tcond_3 (npeaks = npulse +- 1) |', cond_3)
    print('\tcond_4 (peak vals argmax <= 5) |', cond_4)
    print('\tcond_5 (last peak < 90% of max peak |', cond_5)
    print('\tcond_6 (there is a positive reldiff, i.e. decreasing peaks somewhere) |', cond_6)
    print('\tcond_7 (there is a non-negative reldiff entry below TTH threshold) |', cond_7)
    if cond_1 and cond_2 and cond_3 and cond_4 and cond_5 and cond_6 and cond_7:
        does_habituate = True
    else:
        does_habituate = False
        if not cond_3:
            print("inspecting cond_3 failure... expect %d, see %d peaks" % (num_pulses_applied, len(peaks)))
            f = plt.figure()
            ax = f.gca()
            ax.plot(times, output)
            ax.plot(times[peaks], output[peaks], '--x', color='black')
            ax.set_title('preprocess_signal_habituation(); cond3 fails (%d < %d peaks)' %
                         (len(peaks), num_pulses_applied))
            ax.set_xlabel('times[peaks]')
            ax.set_ylabel('output[peaks]')
            plt.show()

    return does_habituate, reldiff, peaks, troughs


if __name__ == '__main__':
    t = np.linspace(np.pi, 12 * np.pi, 1000)
    x = np.sin(t) * 1/t
    x_max = get_signal_peaks(x)
    x_min = get_signal_troughs(x)

    plt.plot(t, x, '--', c='grey')
    plt.plot(t[x_max], x[x_max], 'x')
    plt.plot(t[x_min], x[x_min], 'o')
    plt.show()

    does_habituate, reldiff, peaks, troughs = preprocess_signal_habituation(t, x, 5)
    print("does_habituate ? =", does_habituate)
    plt.plot(t[peaks[1:]], reldiff); plt.title('main reldiff')
    plt.show()
