import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.integrate import solve_ivp

from defined_ode_fn import *
from defined_stimulus import delta_fn_amp, stimulus_constant, stimulus_pulsewave, stimulus_pulsewave_of_pulsewaves, \
    stimulus_staircase, stitch_tspans_from_stimulus
from utils_multitraj import get_multi_traj_pvary, plot_multi_traj_states, plot_multi_traj_output, \
    get_multi_traj_output_hstats, plot_multi_traj_output_hstats, plot_bode_style
from preset_ode_model import ode_model_preset_factory
from preset_solver import PRESET_SOLVER
from settings import DIR_OUTPUT, STYLE_DYNAMICS_VALID, STYLE_DYNAMICS


def analyze_freq_amp_presets(ode_name):
    if ode_name == 'ode_iff_1':
        x0 = [0.0, 0.0, 0.0]
        tspan = (0.0, 100.5)

        # ODE params base case -- these will be perturbed below
        x0 = np.zeros(3)
        base_params_ode = np.array([
            1,  # k_x+
            1,  # k_x-
            1,  # k_y+
            1,  # k_y-
            1,  # k_z+
            100,  # k_z-
            1,  # q_y
            0.5,  # q_z
        ])
        ode_pvary_dict = {}

    elif ode_name == 'ode_custom_1':
        x0 = [0.0, 0.0, 0.0]
        tspan = (0.0, 100.5)

        # ODE params base case -- these will be perturbed below
        base_params_ode = np.array([
                    1,  # k_x+
                    1,  # k_x-
                    1,  # k_y+
                    0.1,  # k_y-
                    1,  # k_z+
                    100,  # k_z-
                ])

        ode_pvary_dict = {
            0: [False, (0.5, 1, 2.0),
                r'Vary $k_{x+}$ aka stimulus amplitude'],
        }
    elif ode_name == 'ode_custom_2':
        x0 = [0.0, 0.0]
        tspan = (0.0, 100.5)
        # TODO for [1, 0.1, 1, 10], and S1 input with duty 0.01, (dirac amp.),
        #  the ODE seems unstable, and violates our "weak habituation" -- seems to habituate normally
        base_params_ode = np.array([
            1,    # k_x+
            0.1,  # k_x-
            1,    # k_z+
            10,   # k_z-
        ])
        '''
        base_params_ode = np.array([
            10,  # 0.1,  # k_x+
            0.01,  # 10,  # k_x-
            1,  # 1,  # k_z+
            10,  # 10,  # k_z-
        ])'''
        ode_pvary_dict = {}
    elif ode_name == 'ode_custom_4_innerout':
        x0 = [0.0, 0.0, 0.0]
        tspan = (0.0, 100.5)

        # ODE params base case -- these will be perturbed below
        base_params_ode = np.array([
            0.1,  # k_x1+ = K_1 / T_1
            0.1,  # k_x1- =   1 / T_1
            0.5,  # k_x2+ = K_2 / T_2
            5.0,  # k_x2- =   1 / T_2
            0,    # x1_high; set as a function of others param below
            1.1,  # x1_high_mult (alpha)
            1e6,  # 1/epsilon for 'singular perturbation' (last state eqn becomes output)
        ])

        # within this main block, do some perturbations around the base parameter set for each param value
        # - True/False -- solve and plot trajectory for it or not
        # - List of param values -- for param corresponding to the dict key
        # - Title for plot
        ode_pvary_dict = {
            0: [False, (0.1, 1, 10.0),
                r'Vary $k_{1+}$ aka $a_1$'],
            2: [False, (0.1, 1, 10.0),
                r'Vary $k_{2+}$ aka $a_2$'],
        }
    elif ode_name == 'ode_custom_5':
        # this is Negative Feedback
        x0 = np.zeros(3)
        tspan = (0.0, 100.5)
        base_params_ode = np.array([
            1,  # k_x+
            1,  # k_x-
            1,  # k_y+
            0.1,  # k_y-
            1,  # k_z+
            10,  # k_z-  100
        ])
        ode_pvary_dict = {}
    elif ode_name == 'ode_custom_7':
        # This is a 3d instance of AIC
        x0 = np.zeros(3)
        tspan = (0.0, 100.5)
        base_params_ode = np.array([
            1,  # a1
            1,  # a2
            1,  # k_0
            1,  # a3  10
            1,   # b3
        ])
        ode_pvary_dict = {}
    elif ode_name == 'ode_custom_7_alt':
        # This is a 4d instance of AIC
        x0 = np.zeros(4)
        tspan = (0.0, 100.5)
        base_params_ode = np.array([
            1,  # a1
            1,  # a2
            1,  # k_0
            10,  # a3
            1,  # b3
            1,  # x_+
            1,  # x_-
        ])
        ode_pvary_dict = {}
    elif ode_name == 'ode_SatLin2D_NL':
        x0 = np.zeros(2)
        tspan = (0.0, 100.5)
        base_params_ode = np.array([
            2.0,  # tau_1: timescale for x_1
            0.2,  # tau_2: timescale for x_2
            5.0,  # a_1:   production rate for dx1/dt
            0.4,  # a_2:   production rate for dx2/dt
        ])
        ode_pvary_dict = {}
    elif ode_name == 'ode_SatLin3D_NL':
        x0 = np.zeros(3)
        tspan = (0.0, 100.5)
        base_params_ode = np.array([
            2.0,  # tau_1: timescale for x_1
            0.2,  # tau_2: timescale for x_2
            50.0,  # tau_3: timescale for x_3
            5.0,  # a_1:   production rate for dx1/dt
            0.4,  # a_2:   production rate for dx2/dt
            5.0,  # a_3:   production rate for dx3/dt
        ])
        ode_pvary_dict = {}
    elif ode_name == 'ode_reverse_engineer_A':
        x0 = np.zeros(1)
        tspan = (0.0, 100.5)
        base_params_ode = np.array([
            0.05,  # accumulation of u(t); degradation of memory W(t)
        ])
        ode_pvary_dict = {}

    elif ode_name == 'ode_linear_filter_1d':
        x0 = np.zeros(1)
        tspan = (0.0, 100.5)
        base_params_ode = np.array([
            0.1,  # alpha: timescale for x decay -- prop to x(t)
            0.5,  # beta: timescale for x growth -- prop to u(t)
        ])
        ode_pvary_dict = {}

    elif ode_name == 'ode_linear_filter_3d':
        x0 = np.zeros(3)
        tspan = (0.0, 100.5)
        base_params_ode = np.array([
            0.2,  # alpha 1
            0.5,  # beta 1
            2,    # N1
            0.1,  # alpha 2
            0.5,  # beta 2
            2,     # N2
            0.05,  # alpha 3
            0.5,   # beta 3
        ])
        ''' 
        # These params motivated by Staddon mapping
        0.2231,   # alpha 1
        0.1793,   # beta 1
        2,  # N1
        0.05129,  # alpha 2
        0.1950,   # beta 2
        2,  # N2
        0.01005,  # alpha 3
        0.1990,   # beta 3'''

        ode_pvary_dict = {}

    elif ode_name == 'ode_linear_filter_1d_lifted':
        x0 = np.zeros(2)
        tspan = (0.0, 100.5)
        base_params_ode = np.array([
            0.1,   # alpha: timescale for x decay -- prop to x(t)
            1,     # beta: timescale for x growth -- prop to u(t)
            2,     # N: hill function, filter saturation
            1e-2,  # epsilon: timescale for output target synchronization (should be fastest timescale)
        ])
        ode_pvary_dict = {}

    else:
        print('main sensitivity_freq_amp.py -- ode_name %s not yet implemented' % ode_name)
        #assert 1==2
        x0 = None
        tspan = None  # TODO cannot be None
        base_params_ode = None
        ode_pvary_dict = None

    return x0, tspan, base_params_ode, ode_pvary_dict


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
    ode_name = 'ode_custom_5'

    regulate_intensity_method = 'fix_pulse_area_1A'
    assert regulate_intensity_method in ['fix_pulse_area_1A', 'fix_pulse_area_1B',
                                         'fix_time_avg_2A', 'fix_time_avg_2B']
    measure_frequency_response = True  # last compartment of this main block
    measure_amplitude_response = True

    # parameters for freq. response and amp. response plots
    nmeasure = 60  # default: 60
    tth_eps = 0.01

    x0, tspan, base_params_ode, ode_pvary_dict = analyze_freq_amp_presets(ode_name)

    # STIMULUS settings
    # ============================================================
    stim_fn = stimulus_pulsewave
    S1_duty = 0.005  # if varying duty, need to make sure period does go below base_period * duty
    S1_period = 1.0
    S1_amp = delta_fn_amp(S1_duty, S1_period)  # delta_fn_amp(S1_duty, S1_period) chooses amplitude for pulse area = 1
    #S1_amp = 100.0
    base_params_stim = [
        S1_amp,  # amplitude
        S1_duty,  # duty in [0,1]
        S1_period,  # period of stimulus pulses (level 1)
    ]

    stimulus_pvary_dict = {  # order: amp, duty, period
        0: [False, [0.5 * S1_amp, 1.0 * S1_amp, 2.0 * S1_amp],
            r'Vary $A$: stimulus amplitude (Hallmark 5)'],
        1: [False, [0.001, 0.01, 0.1], #[0.01, 0.1, 0.5, 0.9, 0.99],
            r'Vary $d$: stimulus "duty" parameter'],
        2: [False, [0.2, 0.5, 1, 2.0, 10.0],
            r'Vary $T$: stimulus period (Hallmark 4)'],
    }

    # Main block 1/4
    # ============================================================
    for pidx in ode_pvary_dict.keys():
        print('working on pvary (ode) idx #%d...' % pidx)
        dont_skip = ode_pvary_dict[pidx][0]
        pvals = ode_pvary_dict[pidx][1]
        title = ode_pvary_dict[pidx][2]
        if dont_skip:
            list_ode_instances = get_multi_traj_pvary(
                ode_name, base_params_ode, stim_fn, base_params_stim, x0, tspan, pidx, pvals, pvariety='ode', regulate_intensity_method=regulate_intensity_method)
            plot_multi_traj_states(list_ode_instances, title=title)
            plot_multi_traj_output(list_ode_instances, title='H info: ' + title, separate=False)
            plot_multi_traj_output(list_ode_instances, title='H info: ' + title, separate=True)
            list_odemodel_instances, hstats_dict = get_multi_traj_output_hstats(list_ode_instances)
            plot_multi_traj_output_hstats(list_ode_instances, hstats_dict, title='H info: ' + title, separate=False)
            plot_multi_traj_output_hstats(list_ode_instances, hstats_dict, title='H info: ' + title, separate=True)

    # Main block 2/4
    # ============================================================
    for pidx, pv_items in stimulus_pvary_dict.items():
        dont_skip = pv_items[0]
        pvals = pv_items[1]
        title = pv_items[2]
        if dont_skip:
            print('working on pvary (stim) idx #%d...' % pidx)
            list_ode_instances = get_multi_traj_pvary(
                ode_name, base_params_ode, stim_fn, base_params_stim, x0, tspan, pidx, pvals,
                pvariety='stimulus', scale_tspan=True, regulate_intensity_method=regulate_intensity_method)
            plot_multi_traj_states(list_ode_instances, title=title)
            plot_multi_traj_output(list_ode_instances, title='Output: ' + title, separate=False)
            plot_multi_traj_output(list_ode_instances, title='H info: ' + title, separate=True)
            list_odemodel_instances, hstats_dict = get_multi_traj_output_hstats(list_ode_instances)
            plot_multi_traj_output_hstats(list_ode_instances, hstats_dict, title='H info: ' + title, separate=False)
            plot_multi_traj_output_hstats(list_ode_instances, hstats_dict, title='H info: ' + title, separate=True)

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
        print('For amplitude response, note: regulate_intensity_method = None')
        list_ode_instances = get_multi_traj_pvary(
            ode_name, base_params_ode, stim_fn, base_params_stim, x0, tspan, pidx, ampspace,
            pvariety='stimulus', scale_tspan=True, regulate_intensity_method=None)

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
