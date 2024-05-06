import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.integrate import solve_ivp

from class_ode_model import ODEModel
from defined_ode_fn import *
from utils_multitraj import plot_multi_traj_states, plot_multi_traj_output, plot_multi_traj_phasespace
from preset_ode_model import PRESETS_ODE_MODEL, ode_model_preset_factory
from preset_solver import PRESET_SOLVER


def barplot_runtimes(labels, runtimes):
    nsolvers, ntrials = runtimes.shape
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    ax = plt.gca()

    if ntrials == 1:
        rects = ax.bar(labels, runtimes[:, 0],
                       color='maroon', width=0.4, zorder=11)
    else:
        assert ntrials == 3  # need to generalize width or plot std dev instead
        width = 0.25  # the width of the bars
        for k in range(ntrials):
            offset = width * k
            rects = ax.bar(np.arange(nsolvers) + offset, runtimes[:, k],
                           width=width, zorder=11)
        ax.set_xticks(np.arange(nsolvers) + width, labels)

    # ax.set_xlabel("Solvers")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel("Runtime (s)")
    plt.grid(which='both', color='#ddddee', zorder=10)
    plt.show()
    return


if __name__ == '__main__':

    # ode_quicktest_S0_one
    # ode_quicktest_S1           # no params, tinker with it locally in class_ode_base
    # ode_iff_1_S1
    # ode_iff_1_6d_S1
    # ode_custom_1_S1
    # ode_custom_2_S1            # Tyson 2d model (''weakly habituates'')
    # ode_custom_1_6d
    # ode_custom_3_integrator
    # ode_custom_3_integrator_TESTING_S1
    # ode_custom_3_simple_integrator_S1
    # ode_custom_7            # AIC - 2D + 1D
    # ode_custom_7_alt        # AIC - 2D + 2D
    # ode_quad_2d_A
    # ode_SatLin1D_NL, ode_SatLin2D_NL, ode_SatLin3D_NL  (1-dim, 2-dim or 3-dim variants -- 1dim was called ode_martin_A before)
    # ode_FHN (fitzhugh-nagumo analog)
    # ode_quicktest_notpolynomial  (modify within class_ode_base.py, not defined_ode_fn.py)
    # ode_reverse_engineer_A
    # ode_linear_filter_1d, ode_linear_filter_1d_lifted  (have 1d to 5d implemented, need to generalize)
    # ode_simplefilter_to_tyson
    # ode_circuit_diode
    #stimulus_suffix = 'S_staircase'
    stimulus_suffix = 'S1'
    preset_ode_and_stimulus = 'ode_linear_filter_1d_lifted' + '_' + stimulus_suffix

    initcond_override = None
    #initcond_override = [0.0, 1.0]
    #initcond_override = [1.0, 1.0, 1.0]

    stitch_kwargs = dict()  # trajectory computation kwargs only applied if traj_pulsewave_stitch = True
    if stimulus_suffix in ['S1', 'S2']:
        traj_pulsewave_stitch = False
        scale_by_period = True
        # Select a max step: np.Inf, arbitrary positive float, or None
        # - np.Inf --> fastest trajectory;
        # = None   --> set automatically via class init, depends on duty and period_S1 -- use with traj stitching to offset speed loss
        max_step = None
    else:
        max_step = None
        traj_pulsewave_stitch = False  # if False, use ode_model.propagate (slow)
        scale_by_period = False

    # Note: method ODEModel.trajectory_pulsewave(...) has forcetol=True, to put solvers on even footing
    solver_presets_to_try = [
        #'solve_ivp_radau_default',  # (12, 6) -- can fail due to step size becoming infinitesimal
        'solve_ivp_radau_strict',    # (8, 4)
        #'solve_ivp_radau_medium',   # (6, 4)
        'solve_ivp_radau_relaxed',   # (5, 2)
        #'solve_ivp_rk23',   # weird behaviour (can't integrate delta function sometimes)
        #'solve_ivp_rk45',    # weird behaviour (other)
        #'solve_ivp_DOP853',
        #'solve_ivp_BDF',    # weird behaviour (can't integrate delta function sometimes)
        #'solve_ivp_LSODA',  # weird behaviour (can't integrate delta function sometimes)
    ]
    nsolver = len(solver_presets_to_try)

    ode_model_instances = [0] * nsolver
    ntrials = 1
    runtimes = np.zeros((nsolver, ntrials))

    base_preset_ode_model = PRESETS_ODE_MODEL[preset_ode_and_stimulus]
    print(base_preset_ode_model)
    # override init cond if provided
    if initcond_override is not None:
        assert len(initcond_override) == len(base_preset_ode_model['args'][1])
        base_preset_ode_model['args'][1] = np.array(initcond_override)
    for idx, solv in enumerate(solver_presets_to_try):
        print('\nRunning solver %d/%d (%s) ...' % (idx, nsolver, solv))
        # 1) slot in solver
        preset_ode_model = copy.deepcopy(base_preset_ode_model)
        preset_ode_model['kwargs']['solver_settings'] = PRESET_SOLVER[solv]
        preset_ode_model['kwargs']['label'] = solv[10:]
        # 2) generate ODEModel instance
        ode_model = ODEModel(*preset_ode_model['args'], **preset_ode_model['kwargs'])

        print('Stimulus properties (for reference):')
        ode_model.ode_base.control_objects[0].printer()
        print('ODE params (for reference):', ode_model.ode_base.ode_name)
        print('\t', ode_model.ode_base.params_short)
        print('\t', ode_model.ode_base.params)
        if max_step is None:
            stitch_kwargs = dict(max_step=ode_model.ode_base.max_step_augment)
        else:
            stitch_kwargs = dict(max_step=max_step)
        print('stitch_kwargs', stitch_kwargs)

        # 3) get traj and runtime
        traj_kwargs = dict(verbose=True, traj_pulsewave_stitch=traj_pulsewave_stitch)
        for k in range(ntrials):
            if k == 0:
                _, _, runtime = ode_model.traj_and_runtime(update_history=True, **traj_kwargs, **stitch_kwargs)
            else:
                _, _, runtime = ode_model.traj_and_runtime(update_history=False, **traj_kwargs, **stitch_kwargs)
            runtimes[idx, k] = runtime
        # 4) store info from idx
        ode_model_instances[idx] = ode_model
        print('...done')

    solver_labels = [k[10:] for k in solver_presets_to_try]

    # plot A
    #barplot_runtimes(solver_labels, runtimes)

    # plot B  TODO plotly version
    plot_multi_traj_states(ode_model_instances, title=None,
                           fillstimulus=False, scale_by_period=scale_by_period)

    # plot C - output y(t) with 'hstats' decorations (e.g. time-to-habituate)
    if stimulus_suffix == 'S1':
        plot_hstats = True
    else:
        plot_hstats = False
    plot_multi_traj_output(ode_model_instances, title=None,
                           fillstimulus=False, scale_by_period=scale_by_period,
                           separate=False, plot_hstats=plot_hstats, color_qualitative=True)

    # plot D - "phase" plots
    if ode_model_instances[0].ode_base.dim_ode > 1:
        plot_multi_traj_phasespace(ode_model_instances, [0, 1], title=None, color_qualitative=True)
    if ode_model_instances[0].ode_base.dim_ode > 2:
        plot_multi_traj_phasespace(ode_model_instances, [1, 2], title=None, color_qualitative=True)
        plot_multi_traj_phasespace(ode_model_instances, [0, 2], title=None, color_qualitative=True)
        plot_multi_traj_phasespace(ode_model_instances, [0, 1, 2], title=None, color_qualitative=True)