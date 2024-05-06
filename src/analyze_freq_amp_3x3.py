import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.integrate import solve_ivp

from analyze_freq_amp_sensitivity import analyze_freq_amp_presets
from class_ode_model import ODEModel
from defined_ode_fn import *
from defined_stimulus import delta_fn_amp, stimulus_constant, stimulus_pulsewave, stimulus_pulsewave_of_pulsewaves, \
    stimulus_staircase, stitch_tspans_from_stimulus
from utils_multitraj import plot_multi_traj_states, plot_multi_traj_output, get_multi_traj_pvary, plot_multi_traj_states, plot_multi_traj_output, \
    get_multi_traj_output_hstats, plot_multi_traj_output_hstats, plot_bode_style


if __name__ == '__main__':

    # ODE settings
    # ============================================================
    # prepared names for this main block:
    #   - 'ode_custom_1' (this is IFF simplified as quadratic vector field)
    #   - 'ode_custom_4_innerout'
    # within this main block, do some perturbations around the base parameter set for each param value
    # dict: ode_pvary_dict = {k -> list len 3}
    #                           - True/False -- solve and plot trajectory for it or not
    #                           - List of param values -- for param corresponding to the dict key
    #                           - Title for plot


    #stimulus_suffix = 'S1'
    #preset_ode_and_stimulus = 'ode_linear_filter_2d' + '_' + stimulus_suffix

    ode_name = 'ode_linear_filter_2d'

    regulate_intensity_method = 'fix_time_avg_2A'  # one of four methods, or None
    assert regulate_intensity_method in ['fix_pulse_area_1A', 'fix_pulse_area_1B',
                                         'fix_time_avg_2A', 'fix_time_avg_2B']
    #assert not regulate_amplitude      # decided ~Sept 2023 that fixing the AMP (area grows with period) is more physical
    measure_frequency_response = True  # last compartment of this main block
    measure_amplitude_response = True

    # STIMULUS settings
    # ============================================================
    stim_fn = stimulus_pulsewave
    S1_duty = 0.01
    S1_period = 1.0
    S1_amp = delta_fn_amp(S1_duty, S1_period)  # delta_fn_amp(S1_duty, S1_period) chooses amplitude for pulse area = 1
    #S1_amp = 100.0
    base_params_stim = [
        S1_amp,  # amplitude
        S1_duty,  # duty in [0,1]
        S1_period,  # period of stimulus pulses (level 1)
    ]

    x0, tspan, base_params_ode, ode_pvary_dict = analyze_freq_amp_presets(ode_name)
    if tspan is None:
        tspan = (0.0, 100.5)

    periodspace = [0.5, 1.0, 2.0]
    freqspace = [1/a for a in periodspace]
    intensitites = [0.5, 1.0, 2.0]
    tth_eps = 0.01

    ode_model_instances = [0]*3  # this will be a 3x3 list of lists

    """
    In this loop,
        - select an intensity I 
        - for each, collect a row of N trajectories each with a different frequency 
    """

    for i, intensity_i in enumerate(intensitites):

        print('working on row i=%d' % i)

        # get HSTATS for many frequencies... extract info from them

        pidx = 2  # default index of the period param for stimulus object (inverse frequency)

        # Step 1 and Step 2: for each parameter (period=X) value
        # - build ODE model object, and...
        # - get output of trajectory for some initial condition
        ode_model_instances[i] = get_multi_traj_pvary(
            ode_name, base_params_ode, stim_fn, base_params_stim, x0, tspan, pidx, periodspace,
            pvariety='stimulus', scale_tspan=True, regulate_intensity_method=regulate_intensity_method)

        # extra: plot trajectories for all frequencies
        plot_multi_traj_states(ode_model_instances[i],
                               title='Trajectories for different frequencies',
                               color_qualitative=False)

        # Step 3: analyze trajectories for all frequencies
        _, hstats_dict = get_multi_traj_output_hstats(
            ode_model_instances[i],
            tth_threshold=tth_eps)

        # Step 4: 'bode' type plots with multiple curves for different measurements (first/last peaks, troughs...)
        # TODO plotly version of this function (or argument flag plotly=True)
        title = r'Vary $T$: stimulus period (Hallmark 4)'
        plot_multi_traj_output_hstats(
            ode_model_instances[i],
            hstats_dict,
            title='H info: ' + title,
            separate=False,
            color_qualitative=False)

        title = 'Frequency response | model: %s' % ode_name
        plot_bode_style(ode_model_instances[i], hstats_dict, freqspace, title=title, freq_flag=True,
                        xlabel=r'frequency $1/T$', log_axis=False)
        plot_bode_style(ode_model_instances[i], hstats_dict, freqspace, title=title, freq_flag=True,
                        xlabel=r'frequency $1/T$', log_axis=True)


    '''

    # Main block 3/4
    # ============================================================
    if measure_frequency_response:
        # get HSTATS for many frequencies... extract info from them
        tth_eps = 0.01
        freqspace = np.logspace(-2, 2, num=nmeasure, base=10)
        pidx = 2  # default index of the period param for stimulus object (inverse frequency)
        periodspace = [1 / a for a in freqspace]

        # Step 1 and Step 2: for each parameter (period=X) value
        # - build ODE model object, and...
        # - get output of trajectory for some initial condition
        list_ode_instances = get_multi_traj_pvary(
            ode_name, base_params_ode, stim_fn, base_params_stim, x0, tspan, pidx, periodspace,
            pvariety='stimulus', scale_tspan=True, regulate_intensity_method=regulate_intensity_method)

        # extra: plot trajectories for all frequencies
        plot_multi_traj_states(list_ode_instances, title='Trajectories for different frequencies',
                               color_qualitative=False)

        # Step 3: analyze trajectories for all frequencies
        list_odemodel_instances, hstats_dict = get_multi_traj_output_hstats(list_ode_instances, tth_threshold=tth_eps)

        # Step 4: 'bode' type plots with multiple curves for different measurements (first/last peaks, troughs...)
        # TODO plotly version of this function (or argument flag plotly=True)
        title = r'Vary $T$: stimulus period (Hallmark 4)'
        plot_multi_traj_output_hstats(list_ode_instances, hstats_dict, title='H info: ' + title, separate=False,
                                      color_qualitative=False)

        title = 'Frequency response | model: %s' % ode_name
        plot_bode_style(list_ode_instances, hstats_dict, freqspace, title=title, freq_flag=True,
                        xlabel=r'frequency $1/T$', log_axis=False)
        plot_bode_style(list_ode_instances, hstats_dict, freqspace, title=title, freq_flag=True,
                        xlabel=r'frequency $1/T$', log_axis=True)

    # Main block 4/4
    # ============================================================
    if measure_amplitude_response:
        # get HSTATS for many amplitudes... extract info from them
        ampspace = np.logspace(-2, 2, num=nmeasure, base=10)
        pidx = 0  # default index of the period param for stimulus object (inverse frequency)


        # Step 1 and Step 2: for each parameter (amplitude=X) value
        # - build ODE model object, and...
        # - get output of trajectory for some initial condition
        list_ode_instances = get_multi_traj_pvary(
            ode_name, base_params_ode, stim_fn, base_params_stim, x0, tspan, pidx, ampspace,
            pvariety='stimulus', scale_tspan=True, regulate_intensity_method=regulate_intensity_method)

        # extra: plot trajectories for all frequencies
        plot_multi_traj_states(list_ode_instances, title='Trajectories for different amplitudes',
                               color_qualitative=False)

        # Step 3: analyze trajectories for all frequencies
        list_odemodel_instances, hstats_dict = get_multi_traj_output_hstats(list_ode_instances, tth_threshold=tth_eps)

        # Step 4: 'bode' type plots with multiple curves for different measurements (first/last peaks, troughs...)
        # TODO plotly version of this function (or argument flag plotly=True)
        title = r'Vary $A$: stimulus amplitude (Hallmark 5)'
        plot_multi_traj_output_hstats(list_ode_instances, hstats_dict, title='H info: ' + title, separate=False,
                                      color_qualitative=False)

        title = 'Amplitude response | model: %s' % ode_name
        plot_bode_style(list_ode_instances, hstats_dict, ampspace, title=title, freq_flag=False,
                        xlabel=r'amplitude $A$', log_axis=False)
        plot_bode_style(list_ode_instances, hstats_dict, ampspace, title=title, freq_flag=False,
                        xlabel=r'amplitude $A$', log_axis=True)
    '''