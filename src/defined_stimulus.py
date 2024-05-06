import matplotlib.pyplot as plt
import numpy as np
from numba import jit


"""
The functions below define the non-autonomous components of the ODE's being investigated (e.g. periodic forcing)

Chart of kwargs for:
    0 ~ stimulus_constant, 
    1 ~ stimulus_pulsewave, 
    2 ~ stimulus_pulsewave_of_pulsewaves

# S(t) lvl 0, 1, 2 - amp             - [default: 1.0] height of the pulse
# S(t) lvl _, 1, 2 - period          - [default: 1.0] period of one rectangle pulse on/off cycle
# S(t) lvl _, 1, 2 - duty            - in (0,1); denotes the fraction of the period where wave is ON (with value "amp")
# S(t) lvl _, _, 2 - npulses         - how many pulse cycle periods before pausing
# S(t) lvl _, _, 2 - npulses_recover - how many pulse cycle periods does it pause for?
"""


def get_npulses_from_tspan(tspan, duty, period):
    """
    Intended for use with stimulus_pulsewave context

    e.g. if duty = 0.1, period = 1.0, then
        (0.0, 2.4) returns 2
        (0.85, 2.4) returns 2
        (-0.2, 2.4) returns 3

    note that a pulse is counted if the start or endpoint of tspan lies partially within a duty cycle
    """
    t0, t1 = tspan
    assert t0 == 0       # for simplicity, can generalize later
    assert t1 >= period  # for simplicity, can generalize later
    assert 0.0 < duty < 1.0

    npulses = t1 // period
    remainder_fraction = t1 / period % 1
    if remainder_fraction > 1 - duty:  # this accounts for a "partial" pulse (somewhere in duty cycle)
        npulses += 1

    return npulses


@jit(nopython=True)
def delta_fn_amp(duty, period):
    """
    - Consider a sequence of step pulses with period T=1, with width d*T where 0<d<1 and amplitude 1.0.
    - Therefore, the unscaled pulse has an area d * T * A
    - To correspond to a discrete DiracDelta function, it should have area 1.0.
    - Therefore, choose amplitude_scaled = 1 / d * T
    """
    return 1 / (duty * period)


@jit(nopython=True)
def stimulus_constant(t, amp, t0=0):
    s = np.where(t - t0 > 0, amp, 0)
    return s


@jit(nopython=True)
def stimulus_pulsewave(t, amp, duty, period):
    # generates a pulse wave which is "1" for a fraction "duty" of each period
    # - assumes phase = 0
    tmod = t/period % 1  # float 0 <= tmod <= 1
    # use the "duty" parameter to return 0 or amp
    s = np.where(tmod < (1-duty), 0, amp)
    return s


@jit(nopython=True)
def stimulus_pulsewave_of_pulsewaves(t, amp, duty, period, npulses, npulses_recover):
    """
    Generates a pulse wave following stimulus_pulsewave() for period T, amplitude amp, duty fraction on.
        - generates npulses_on such waves, for a total duration of [T * npulses]
        - then does nothing (stimulus = 0) for a total duration of [T * npulses_recover]
        - repeats the above pattern indefinitely over the provided time interval t
    Remarks:
        - this function is itself periodic with period T_meta = T * (npulses + npulses_recover)
        - refer to stimulus_pulsewave() for simpler form
    """
    period_meta = period * (npulses + npulses_recover)
    duty_meta = npulses / (npulses + npulses_recover)
    # create arrays of float 0 <= tmod <= 1
    tmod_base = t / period % 1
    tmod_meta = t / period_meta % 1  # float 0 <= tmod <= 1
    # use these modified time arrays to compute the stimulus
    s1 = np.where(tmod_base < (1 - duty), 0, amp)
    s2 = np.where(tmod_meta < duty_meta, 1, 0)
    sval = s1 * s2
    return sval


@jit(nopython=True)
def stimulus_staircase(t, step, duration):
    """
    Every duration, the stimulus increases by step (starting at S=0)
    """
    step_counter = t // duration
    return step * step_counter


@jit(nopython=True)
def stimulus_rectangle(t, height, t0, duration):
    """
    Every duration, the stimulus increases by step (starting at S=0)
    """
    s1 = np.where(t < t0, 0, height)
    s2 = np.where(t < t0 + duration, 1, 0)
    sval = s1 * s2
    return sval


def stitch_tspans_from_stimulus(fn_stim, args_stim, t0, tmax, s_low=0.0, s_high=1.0):
    """
    Helper function for "stitching" together the outputs from a sequence of ODE trajectory calls
    TODO - optimize (original approach) by always writing it for 'one loop' from theta = 0 to theta = 2 pi (t = 0 to T)
        then post-process based on t0, tmax by truncating or duplicating and "shifting it in a circular manner"
    TODO generalize so it doesn't need to start at t0 = 0.0 (could assume t0 0, then correct at end if easier...)
    """
    #assert t0 == 0.0
    #assert tmax > 0
    assert 0 <= t0 < tmax  # TODO generalize this later to negative case
    fn_stim_name = fn_stim.__name__
    assert fn_stim_name in ['stimulus_pulsewave', 'stimulus_pulsewave_of_pulsewaves']
    tspans = []
    tsegment_start = 0  # start call cases at "t = 0" and do post-processing to correct

    if fn_stim_name == 'stimulus_pulsewave':
        amp, duty_S1, period_S1 = args_stim
        tau1 = period_S1 * (1 - duty_S1)
        tau2 = period_S1 * duty_S1
        period_total = period_S1
        s_high = amp
        # iteratively construct the list of tspan segments based on sequential pieces of the periodic function S(t)
        assert tmax - t0 > tau1
        tsegment_end = tau1
        s_current = s_low
        v = 1  # start at v=1 since we add the first segment in line 1 of the loop
        while tsegment_end <= tmax:
            tspans.append([tsegment_start, tsegment_end, s_current])
            if v == 0:
                tsegment_start = tsegment_end
                tsegment_end = tsegment_end + tau1
                s_current = s_low
                v = 1
            elif v == 1:
                tsegment_start = tsegment_end
                tsegment_end = tsegment_end + tau2
                s_current = s_high
                v = 0
    else:
        assert fn_stim_name == 'stimulus_pulsewave_of_pulsewaves'
        amp, duty_S1, period_S1, npulses, npulses_recover = args_stim
        tau1 = period_S1 * (1 - duty_S1)
        tau2 = period_S1 * duty_S1
        period_total = period_S1 * (npulses + npulses_recover)
        s_high = amp
        # iteratively construct the list of tspan segments based on sequential pieces of the periodic function S(t)
        assert tmax - t0 > tau1
        tsegment_end = tau1
        s_current = s_low
        pulse_idx = 0
        v = 1  # start at v=1 since we add the first segment in line 1 of the loop
        while tsegment_end <= tmax:
            tspans.append([tsegment_start, tsegment_end, s_current])
            if v == 0:
                tsegment_start = tsegment_end
                tsegment_end = tsegment_end + tau1
                s_current = s_low
                v = 1
            elif v == 1:
                tsegment_start = tsegment_end
                tsegment_end = tsegment_end + tau2
                s_current = s_high
                pulse_idx += 1
                if pulse_idx == npulses:
                    v = 2
                    pulse_idx = 0
                else:
                    v = 0
            elif v == 2:
                tsegment_start = tsegment_end
                tsegment_end = tsegment_end + period_S1 * npulses_recover
                s_current = s_low
                v = 0
    # (1) Post-processing step -- add tmax sliver to end
    if tmax != tspans[-1][1]:
        tspans.append([tspans[-1][1], tmax, s_current])
    # (2) Post-processing step -- remove front portion in the case that t0 > 0
    # TODO further generalize this as mentioned in docstring
    if t0 > 0.0:
        #tspans = np.array(tspans)
        #tspans_shifted = np.zeros_like(tspans)
        #t0_in_circle = t0 % period_total
        #tmax_in_circle = tmax % period_total
        for idx in range(len(tspans)):
            if tspans[idx][0] <= t0 < tspans[idx][1]:
                tspans = tspans[idx:]
                tspans[0][0] = t0
                break
    return tspans


def test_tspans_stitch_with_constant_stimulus(tspans_triplets):
    t0, tmax = tspans_triplets[0][0], tspans_triplets[-1][1]
    dt = 1e-4
    t_vals = np.arange(t0, tmax, dt)
    s_vals = np.zeros_like(t_vals)
    for triplet in tspans_triplets:
        t_left, t_right, s_lvl = triplet
        idx_left = int(t_left / dt)
        idx_right = int(t_right / dt)
        s_vals[idx_left : idx_right] = s_lvl
    return t_vals, s_vals


VALID_FN_STIMULUS = [a.__name__ for a in [stimulus_constant,
                                          stimulus_pulsewave,
                                          stimulus_pulsewave_of_pulsewaves,
                                          stimulus_staircase,
                                          stimulus_rectangle]
                     ]


if __name__ == '__main__':
    """
    Main: Example plots of the different stimulus functions
    """

    amp = 1.0                                              # S(t) lvl 1 and 2
    duty_S1 = 0.1                                          # S(t) lvl 1 and 2 - duty in (0,1)
    period_S1 = 2.0                                       # S(t) lvl 1 and 2
    npulses = 5                                           # S(t) lvl 2 - how many pulse cycle periods before pausing
    npulses_recover = 5                                   # S(t) lvl 2 - how many pulse cycle periods does it pause for?
    period_meta = period_S1 * (npulses + npulses_recover)  # S(t) lvl 2 (implicit parameter)

    p_stim_1 = [amp, duty_S1, period_S1]
    p_stim_2 = [amp, duty_S1, period_S1, npulses, npulses_recover]

    tspan_stimulus = (0.0, 2.25 * period_meta)
    npts_times = int(tspan_stimulus[1]) * 200 * int(1 / min(duty_S1, 1-duty_S1))  # duty on/off at least 10 timepoints
    tsample = np.linspace(tspan_stimulus[0], tspan_stimulus[1], npts_times)

    s1 = stimulus_pulsewave(tsample, *p_stim_1)
    s2 = stimulus_pulsewave_of_pulsewaves(tsample, *p_stim_2)

    fig, axarr = plt.subplots(2, 1, sharex=True)
    axarr[0].set_title(r"Stimulus $S(t)$: pulse wave with and without recovery portion")
    axarr[0].plot(tsample, s1, '-o', markersize=3, label='pulsewave')
    axarr[0].set_ylabel(r"$S(\tau)$")
    axarr[1].plot(tsample, s2, '-o', markersize=3, label='pulsewave with recovery')
    axarr[1].set_ylabel(r"$S(\tau)$")
    axarr[1].set_xlabel(r"time $\tau = t/T$")
    plt.show()

    # check tspan stitching behaviour, and use it to mimic pulses with constant stimulus at different levels
    tspans_triplet_S1 = stitch_tspans_from_stimulus(stimulus_pulsewave, p_stim_1, tspan_stimulus[0], tspan_stimulus[1], s_low=0.0, s_high=1.0)
    tspans_triplet_S2 = stitch_tspans_from_stimulus(stimulus_pulsewave_of_pulsewaves, p_stim_2, tspan_stimulus[0], tspan_stimulus[1], s_low=0.0, s_high=1.0)
    print(tspans_triplet_S1)
    tc, sc = test_tspans_stitch_with_constant_stimulus(tspans_triplet_S1)
    plt.plot(tsample, s1, '-x', label='orig')
    plt.plot(tc, sc, label='stitch')
    plt.show()
    tc, sc = test_tspans_stitch_with_constant_stimulus(tspans_triplet_S2)
    plt.plot(tsample, s2, '-x', label='orig')
    plt.plot(tc, sc, label='stitch')
    plt.show()

    # check staircase stimulus
    plt.plot(tsample, stimulus_staircase(tsample, 2.0, 1.0))
    plt.title('Staircase stimulus')
    plt.show()
