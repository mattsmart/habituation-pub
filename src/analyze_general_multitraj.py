import numpy as np

from defined_ode_fn import *
from defined_stimulus import delta_fn_amp, stimulus_pulsewave
from utils_multitraj import get_multi_traj_pvary, plot_multi_traj_states, plot_multi_traj_output, plot_multi_traj_phasespace


def main_settings(ode_name):

    if ode_name == 'ode_iff_1':
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
    elif ode_name == 'ode_iff_1_6d':
        x0 = np.zeros(6)
        # TODO try params from excel table extracted from their thesis - typos in thesis?
        base_params_ode = np.array([
            1,  # k_x+
            30,  # k_x-
            5,  # k_y+
            0.5,  # k_y-
            10,  # k_z+
            1,  # k_z-
            10,  # q_y
            0.01,  # q_z

            10,  # k_xb+
            1,  # k_xb-
            0.5,  # k_yb+
            0.01,  # k_yb-
            0.5,  # k_zb+
            50,  # k_zb-
            1,  # q_yb
            1,  # q_zb
        ])

        ode_pvary_dict = {}
    elif ode_name == 'ode_custom_1':
        x0 = [0.0, 0.0, 0.0]
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
            1.0,  # x1_high_mult (alpha)
            1e6,  # 1/epsilon for 'singular perturbation' (last state eqn becomes output)
        ])

        # within this main block, do some perturbations around the base parameter set for each param value
        # - True/False -- solve and plot trajectory for it or not
        # - List of param values -- for param corresponding to the dict key
        # - Title for plot
        ode_pvary_dict = {
            0: [False, (0.1, 1, 10.0),
                r'Vary $k_{1+}$ aka $a_1$'],
            1: [False, (0.1, 1, 10.0),
                r'Vary $k_{1-}$ aka $b_1$'],
            2: [False, (0.1, 1, 10.0),
                r'Vary $k_{2+}$ aka $a_2$'],
            3: [False, (0.1, 1, 10.0),
                r'Vary $k_{2-}$ aka $b_2$'],
            5: [False, (1.0, 1.01, 1.1),
                r'Vary $\alpha$ (scales x1_high; >= 1.0)'],
            6: [False, (1e1, 1e3, 1e6),
                r'Vary $1/\epsilon$'],
        }
    elif ode_name == 'ode_custom_5':
        x0 = np.zeros(3)
        base_params_ode = np.array([
            1,  # k_x+
            1,  # k_x-
            1,  # k_y+
            0.1,  # k_y-
            1,  # k_z+
            100,  # k_z-
        ])
        ode_pvary_dict = {}
    elif ode_name == 'ode_custom_7':
        x0 = np.zeros(3)
        base_params_ode = np.array([
            1,  # a1
            1,  # a2
            1,  # k_0
            10,  # a3
            1,   # b3
        ])
        ode_pvary_dict = {}
    elif ode_name == 'ode_custom_7_alt':
        x0 = np.zeros(4)
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
    elif ode_name == 'ode_quad_2d_A':
        x0 = [0.0, 0.0]
        base_params_ode = np.array([
            1.0,  # a0 = 1 * sval
            1.0,  # b0 = 1 * sval
            -0.1,  # a1 = -0.1
            -1.0,  # b2 = -1
            -1,  # a12 = -1
            -10,  # b12 = -10
        ])
        ode_pvary_dict = {}
    elif ode_name == 'ode_martin_A':
        x0 = [0.0]
        base_params_ode = np.array([
            1.0,  # a0
            0.1,  # b0
        ])
        ode_pvary_dict = {}
    elif ode_name == 'ode_SatLin2D_NL':
        x0 = [0.0, 0.0]
        base_params_ode = np.array([
            2.0,  # tau_1: timescale for x_1
            0.2,  # tau_2: timescale for x_2
            5.0,  # a_1:   production rate for dx1/dt
            0.4,  # a_2:   production rate for dx2/dt
        ])
        ode_pvary_dict = {}
    elif ode_name == 'ode_SatLin3D_NL':
        x0 = [0.0, 0.0, 0.0]
        base_params_ode = np.array([
            2.0,  # tau_1: timescale for x_1
            0.2,  # tau_2: timescale for x_2
            50.0,  # tau_3: timescale for x_3
            5.0,  # a_1:   production rate for dx1/dt
            0.4,  # a_2:   production rate for dx2/dt
            5.0,  # a_3:   production rate for dx3/dt
        ])
        ode_pvary_dict = {}
    else:
        print('main sensitivity_freq_amp.py -- ode_name %s not yet implemented' % ode_name)
        assert 1==2
    return x0, base_params_ode, ode_pvary_dict


if __name__ == '__main__':

    # ODE Setup
    # ode_iff_1
    # ode_iff_1_6d
    # ode_custom_1           -- 3d, simplify ODE IFF 1
    # ode_custom_2           -- 2d Tyson model ('weak habituation')
    # ode_custom_4_innerout  -- LINEAR 2d + nonlinear output wrapped into x3
    # ode_custom_5           -- 3d, negative feedback
    # ode_custom_7           -- 3d, 'antithetical integral feedback' module example
    # ode_custom_7_alt       -- 4d, similar to above
    # ode_quad_2d_A          -- 2d, quadratic, more general than tyson model
    # ode_martin_A           -- 1d, linear system augmented in direction of IFF
    # ode_SatLin2D_NL, ode_SatLin3D_NL
    ode_name = 'ode_SatLin2D_NL'

    x0, base_params_ode, ode_pvary_dict = main_settings(ode_name)

    stim_fn = stimulus_pulsewave
    S1_duty = 0.01
    S1_period = 1.0
    base_params_stim = [
        delta_fn_amp(S1_duty, S1_period),  # amplitude
        S1_duty,    # duty in [0,1]
        S1_period,  # period of stimulus pulses (level 1)
    ]
    tspan = (0.0, 100.5)

    # Main block: Varying ODE parameters
    for pidx in ode_pvary_dict.keys():
        print('working on pvary (ode) idx #%d...' % pidx)
        dont_skip = ode_pvary_dict[pidx][0]
        pvals = ode_pvary_dict[pidx][1]
        title = ode_pvary_dict[pidx][2]
        if dont_skip:

            list_ode_instances = get_multi_traj_pvary(
                ode_name, base_params_ode, stim_fn, base_params_stim, x0, tspan, pidx, pvals, pvariety='ode')
            plot_multi_traj_states(list_ode_instances, title=title)
            plot_multi_traj_output(list_ode_instances, title=title)
            plot_multi_traj_output(list_ode_instances, title=title, plot_hstats=True)
            if len(x0) >= 2:
                plot_multi_traj_phasespace(list_ode_instances, indices=[0, 1], title=title)
            if len(x0) == 3:
                plot_multi_traj_phasespace(list_ode_instances, indices=[1, 2], title=title)
                plot_multi_traj_phasespace(list_ode_instances, indices=[0, 2], title=title)
                plot_multi_traj_phasespace(list_ode_instances, indices=[0, 1, 2], title=title)

    # Main block: Varying stimulus parameters
    stimulus_pvary_dict = {  # order: amp, duty, period
        0: [False, [0.5, 1, 2.0],
            r'Vary $A$: stimulus amplitude (sketchy -- need to normalize A, or not?)'],
        1: [True, [0.001, 0.01, 0.1], #, 0.5, 0.9, 1.0],
            r'Vary $d$: stimulus "duty" parameter'],
        2: [False, [0.5, 1, 2.0],
            r'Vary $T$: stimulus period'],
    }
    for pidx, pv_items in stimulus_pvary_dict.items():
        dont_skip = pv_items[0]
        pvals = pv_items[1]
        title = pv_items[2]
        if dont_skip:
            print('working on pvary (stim) idx #%d...' % pidx)
            list_ode_instances = get_multi_traj_pvary(
                ode_name, base_params_ode, stim_fn, base_params_stim, x0, tspan, pidx, pvals, pvariety='stimulus')

            plot_multi_traj_states(list_ode_instances, title=title)
            plot_multi_traj_output(list_ode_instances, title=title, plot_hstats=False)
            plot_multi_traj_output(list_ode_instances, title=title, plot_hstats=True)
            if len(x0) >= 2:
                plot_multi_traj_phasespace(list_ode_instances, indices=[0, 1], title=title)
            if len(x0) >= 3:
                plot_multi_traj_phasespace(list_ode_instances, indices=[1, 2], title=title)
                plot_multi_traj_phasespace(list_ode_instances, indices=[0, 2], title=title)
                plot_multi_traj_phasespace(list_ode_instances, indices=[0, 1, 2], title=title)

    # Main block: Varying solver param atol and tol
    print("get_multi_traj_tolerance(...) block -- TODO update/implement")
    title = 'Vary atol, rtol'
    atol_rtol_pairs = [
        (1e-4, 1e-2),
        (1e-6, 1e-3),
        (1e-12, 1e-6)]
    '''
    list_traj, list_times, list_labels = get_multi_traj_tolerance(
        x0, tspan, base_odeparams, atol_rtol_pairs)
    plot_multi_traj_states(list_traj, list_times, [base_odeparams for _ in atol_rtol_pairs], list_labels, title=title)
    '''
