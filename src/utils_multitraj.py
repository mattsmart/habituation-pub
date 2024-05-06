import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.integrate import solve_ivp

from class_ode_base import ODEBase
from class_ode_model import ODEModel
from class_ode_stimulus import ODEStimulus
from defined_ode_fn import *
from defined_stimulus import delta_fn_amp, stimulus_constant, stimulus_pulsewave, stimulus_pulsewave_of_pulsewaves, \
    stimulus_staircase, stitch_tspans_from_stimulus, get_npulses_from_tspan
from utils_timeseries import time_to_habituate
from preset_ode_model import ode_model_preset_factory
from preset_solver import PRESET_SOLVER
from settings import DIR_OUTPUT, STYLE_DYNAMICS_VALID, STYLE_DYNAMICS


def line_colors(n, discrete=True):
    if discrete:
        cmap = mpl.colormaps['Set2']  # 'tab20c', 'Set2'
        colors = [cmap(i) for i in range(n)]
    else:
        # selection:
        cstr = 'PuBuGn'          # choose MPL cmap ('Blues', 'PuBuGn', ...)
        low, high = 0.20, 0.70  # choose low and high vals in 0 < a < 1
        assert 0 <= low < high <= 1.0
        cmap = mpl.colormaps[cstr]
        colors = cmap(np.linspace(low, high, n))
    return colors


def change_ode_param(pbase, pidx, pval):
    pmod = pbase.copy()
    pmod[pidx] = pval
    return pmod


def change_stimulus_param(pbase_stim, pkey_stim, pval_stim, stim_fn, regulate_intensity_method=None):
    """
    if regulate_intensity_method is not None, then modifying one stimulus may cause others to co-vary

    Note: this setting assumes
        - stim_fn in [stimulus_pulsewave, stimulus_pulsewave_of_pulsewaves]; and
        - the parameter being modified is either period or duty (not amplitude)

    regulate_intensity_method == 'fix_pulse_area_1A'
    > maintain constant pulse area
        if the   duty   is halved, then fail TODO implement
        if the  period  is halved, then increase duty proportionally (don't change amplitude)

    regulate_intensity_method == 'fix_pulse_area_1B'
    > maintain constant pulse area
        if the   duty   is halved, then fail TODO implement
        if the  period  is halved, then increase amplitude proportionally (don't change duty)

    regulate_intensity_method == 'fix_time_avg_2A'
    > maintain constant input per unit time (pulse area divided by period must be constant)
        if the   duty   is halved, then fail TODO implement
        if the  period  is halved, then do nothing (amplitude stays same, duty stays same)

    regulate_intensity_method == 'fix_time_avg_2B'
    > maintain constant input per unit time (pulse area divided by period must be constant)
        if the   duty   is halved, then fail TODO implement
        if the  period  is halved, then double the duty amd halve the amplitude

    - scale pulse amplitude so that area = 1.0 if PERIOD or DUTY are being varied
    """
    pmod = pbase_stim.copy()
    pmod[pkey_stim] = pval_stim
    if regulate_intensity_method is not None:
        assert regulate_intensity_method in ['fix_pulse_area_1A', 'fix_pulse_area_1B',
                                             'fix_time_avg_2A',   'fix_time_avg_2B']
        assert stim_fn.__name__ in ['stimulus_pulsewave', 'stimulus_pulsewave_of_pulsewaves']
        # rectangle pulsewave stimulus parameters are:
        #   key 0: amplitude
        #   key 1: duty
        #   key 2: period
        base_amp, base_duty, base_period = pbase_stim[0], pbase_stim[1], pbase_stim[2]
        #base_area = base_amp * base_duty * base_period

        period_ratio = base_period / pmod[2]    # if period halved, this is 2.0
        if regulate_intensity_method == 'fix_pulse_area_1A':
            pmod[1] = base_duty * period_ratio  # if period halved, then duty needs to double (amp constant)
        elif regulate_intensity_method == 'fix_pulse_area_1B':
            pmod[0] = base_amp * period_ratio  # if period halved, then double the amplitude (duty constant)
        elif regulate_intensity_method == 'fix_time_avg_2A':
            pass
        else:
            assert regulate_intensity_method == 'fix_time_avg_2B'
            pmod[0] = base_amp / period_ratio
            pmod[1] = base_duty * period_ratio

        # walls / sanity checks after the modifications above
        assert pmod[1] < 1.0

    return pmod


def get_multi_traj_pvary(ode_name, base_params_ode, stim_fn, base_params_stim, x0, tspan, pidx, pvals, pvariety='ode',
                         regulate_intensity_method=None, scale_tspan=True):
    """
    pvariety in ['ode', 'stimulus']
        if pvariety == 'ode',      pkey is an integer corresponding to an ode parameter in a fixed order list
        if pvariety == 'stimulus', pkey is an integer corresponding to a stimulus parameter in a fixed order list
    if scale_tspan: augment tspan if the period of stimuli is being modified
        e.g. if period = 2 instead of 1, then double the tspan
    """
    assert pvariety in ['ode', 'stimulus']

    n = len(pvals)
    ode_model_instances = [0] * n

    integration_style_stitching = True
    stitch_kwargs_base = {
        'forcetol': (1e-8, 1e-4),  # ATOL, RTOL = 1e-8, 1e-4   --or--   1e-12, 1e-6
        'max_step': np.Inf,  # np.Inf default
    }
    stitch_dynamic_max_step = False  # default: True

    for i in range(n):
        pval = pvals[i]
        tspan_augment = [tspan[0], tspan[1]]

        if pvariety == 'ode':
            mod_params_ode = change_ode_param(base_params_ode, pidx, pval)
            mod_params_stim = base_params_stim
            label_ode = 'ODE (pvary %d)' % i
            label_stim = r'$u(t)$ base'
        else:
            assert pvariety == 'stimulus'
            mod_params_stim = change_stimulus_param(base_params_stim, pidx, pval, stim_fn,
                                                    regulate_intensity_method=regulate_intensity_method)
            mod_params_ode = base_params_ode
            label_ode = 'ODE base'
            label_stim = r'$u(t)$ (pvary %d)' % i

            # if we are modifying the period and scale_tspan is True...
            if pidx == 2 and scale_tspan:
                assert tspan[0] == 0.0
                tspan_augment[1] = tspan[1] * pval  # e.g., if t0, t1 = 0, 100.5 and period = 2.0: t1 = 201.0

        # (0) prep control_objects argument based on stimulus
        stim_instance = ODEStimulus(
            stim_fn,
            mod_params_stim,
            label=label_stim
        )
        mod_control_objects = [stim_instance]

        # (1) prep control_objects argument based on stimulus
        multitraj_label = '%s; %s' % (label_ode, label_stim)
        model_argprep = ode_model_preset_factory(multitraj_label, ode_name, mod_control_objects,
                                                 tspan_augment[0], tspan_augment[1],
                                                 ode_params=mod_params_ode, init_cond=x0)
        ode_model_pvary = ODEModel(*model_argprep['args'], **model_argprep['kwargs'])

        stitch_kwargs = stitch_kwargs_base.copy()
        if stitch_dynamic_max_step:
            stitch_kwargs['max_step'] = ode_model_pvary.ode_base.max_step_augment

        # update ode label (for later plotting)
        if pvariety == 'ode':
            ode_model_pvary.label = "%s: %s = %.3f" % (pvariety, ode_model_pvary.ode_base.params_short[pidx], pvals[i])
        else:
            assert pvariety == 'stimulus'
            stim_param_labels = ode_model_pvary.ode_base.control_objects[0].params_short
            ode_model_pvary.label = "%s: %s = %.3f" % (pvariety, stim_param_labels[pidx], pvals[i])

        # compute trajectory
        print()
        print(ode_model_pvary.label, '| get_multi_traj_pvary(...): full trajectory from arbitrary init cond... (%d of %d)' % (i, n))
        if stim_instance.stim_fn.__name__ == 'stimulus_pulsewave':
            stim_amp, stim_duty, stim_period = mod_params_stim
            stim_area = stim_amp * (stim_duty * stim_period)
            stim_area_over_T = stim_amp * stim_duty
            print('\tRegulate intensity: %s | d=%.2e, amp=%.2f, T=%.2e | area=%.2e; stim_area_over_T=%.2e' %
                  (regulate_intensity_method, stim_duty, stim_amp, stim_period, stim_area, stim_area_over_T))
        start = time.time()
        #r, times = ode_model_pvary.propagate(update_history=True, verbose=True, t0=tspan[0], t1=tspan[1], init_cond=x0)
        # trajectory_pulsewave(...) - splits the integration across tspan segments, with constant stimulus on each
        #r, times = ode_model_pvary.trajectory_pulsewave(update_history=True, t0=tspan[0], t1=tspan[1], init_cond=x0)
        if integration_style_stitching:
            r, times = ode_model_pvary.trajectory_pulsewave(update_history=True, init_cond=x0,
                                                            t0=tspan_augment[0], t1=tspan_augment[1], **stitch_kwargs)
        else:
            r, times = ode_model_pvary.propagate(update_history=True, init_cond=x0,
                                                 t0=tspan_augment[0], t1=tspan_augment[1],
                                                 params_solver=None, verbose=False)
        end = time.time()
        print('\ttime (propagate OR traj_stitch call) =', end - start)

        ode_model_instances[i] = ode_model_pvary

    return ode_model_instances


def get_multi_traj_tolerance(x0, tspan, pbase, atol_rtol_pairs, fillstimulus=False):
    # TODO revise this if needed or work iot into the above function
    """
    n = len(atol_rtol_pairs)
    list_traj = [0] * n
    list_times = [0] * n
    list_labels = [0] * n
    for i in range(n):
        atol, rtol = atol_rtol_pairs[i]
        r, times = trajectory(ode_iff_1, tspan, x0, pbase, atol=atol, rtol=rtol)
        list_traj[i] = r
        list_times[i] = times
        list_labels[i] = "atol=%.1e, rtol=%.1e" % (atol, rtol)
    """
    return list_traj, list_times, list_labels


def plot_multi_traj_states(list_odemodel_instances, title=None, fillstimulus=True, scale_by_period=True,
                           color_qualitative=True, prepend_stimulus=True):
    """
    Plot subplots of "x_i vs t" for a provided list of ode models (instances of custom class ODEModel)
    - plot are based on the class attributes: history_times and history_state
    """
    n = len(list_odemodel_instances)
    colors = line_colors(n, discrete=color_qualitative)

    max_ode_dimension = 0
    idx_biggest_model = 0
    for idx, odem in enumerate(list_odemodel_instances):
        dim_ode = odem.ode_base.dim_ode
        if dim_ode > max_ode_dimension:
            max_ode_dimension = dim_ode
            idx_biggest_model = idx

    if prepend_stimulus:
        nrows = max_ode_dimension + 1
    else:
        nrows = max_ode_dimension

    def row_idx_from_ode_state_idx(idx):
        if prepend_stimulus:
            row_idx = idx + 1
        else:
            row_idx = i
        return row_idx

    fig, axarr = plt.subplots(nrows, 1, sharex=True, figsize=(8, 5), squeeze=False)
    for idx in range(n):
        ode_instance = list_odemodel_instances[idx]
        r = ode_instance.history_state  # times x dim_ode
        times = ode_instance.history_times
        label = ode_instance.label
        if n > 5 and idx not in [0, n-1]:
            # only legend first and last curve if there are more than 5 curves
            label = None

        if scale_by_period:
            period_S1 = ode_instance.ode_base.control_objects[0].params[2]
            scaled_times = times / period_S1
        else:
            scaled_times = times

        # plot trajectory of the "idx" model (loop over ode dimensions i)
        assert ode_instance.ode_base.dim_control == 1
        control_obj = ode_instance.ode_base.control_objects[0]
        amp_stim = control_obj.params[0]
        stim_of_t = control_obj.fn_prepped(times)

        if prepend_stimulus:
            axarr[0, 0].plot(scaled_times, stim_of_t, c='k', label='stimulus')
            axarr[0, 0].axhline(0, linestyle='--', linewidth=1, color='grey')

        for i in range(ode_instance.ode_base.dim_ode):
            row_idx = row_idx_from_ode_state_idx(i)
            axarr[row_idx, 0].plot(scaled_times, r[:, i], c=colors[idx], label=label)
            axarr[row_idx, 0].axhline(0, linestyle='--', linewidth=1, color='grey')
            if fillstimulus:
                assert ode_instance.ode_base.dim_control == 1
                axarr[i, 0].fill_between(
                    scaled_times, 0, r[:, i] * stim_of_t / amp_stim,
                    facecolor='gainsboro',
                    alpha=0.2)

        # plot analytic prediction of trajectory, if available
        if ode_instance.ode_base.state_analytic is not None:
            for fn in ode_instance.ode_base.state_analytic:
                state_analytic = fn(times)
                for i in range(ode_instance.ode_base.dim_ode):
                    row_idx = row_idx_from_ode_state_idx(i)
                    axarr[row_idx, 0].plot(scaled_times, state_analytic[i], '--', lw=1.0, label=fn.__name__, zorder=11)

    for idx in range(max_ode_dimension):
        row_idx = row_idx_from_ode_state_idx(idx)
        x_i_label = list_odemodel_instances[idx_biggest_model].ode_base.ode_long[idx]
        axarr[row_idx, 0].set_ylabel(x_i_label)
    if prepend_stimulus:
        axarr[0, 0].set_ylabel(r'$u(t)$')

    axarr[-1, 0].set_xlabel(r"time $\tau = t/T$")
    if title is not None:
        axarr[0, 0].set_title(title)
    plt.legend()
    plt.show()
    return


def plot_multi_traj_output(list_odemodel_instances, title=None, fillstimulus=True, scale_by_period=True, separate=False,
                           plot_hstats=True, color_qualitative=True):
    """
    Plot subplots of "x_i vs t" for a provided list of ode models (instances of custom class ODEModel)
    - plot are based on the class attributes: history_times and history_state

    if plot_hstats: plot habituation info using habituation_utility.py
    """
    n = len(list_odemodel_instances)
    colors = line_colors(n, discrete=color_qualitative)

    if separate:
        fig, axarr = plt.subplots(n, 1, squeeze=False, sharex='col', figsize=(8, 5))
    else:
        fig, axarr = plt.subplots(1, 1, squeeze=False, sharex='col', figsize=(8, 3))

    for idx in range(n):
        if separate:
            ax = axarr[idx, 0]
        else:
            ax = axarr[0, 0]

        ode_instance = list_odemodel_instances[idx]

        times = ode_instance.history_times
        r = ode_instance.history_state  # times x dim_ode
        stim_object = ode_instance.ode_base.control_objects[0]
        stim_of_t = stim_object.fn_prepped(times)

        output = ode_instance.ode_base.output_fn(r, stim_of_t)
        label = ode_instance.label

        if scale_by_period:
            period_S1 = ode_instance.ode_base.control_objects[0].params[2]
            scaled_times = times / period_S1
            times_xlabel = r"time $\tau = t/T$"
        else:
            scaled_times = times
            times_xlabel = r"time $t$"

        ax.plot(scaled_times, output, c=colors[idx], label=label)

        if plot_hstats:
            assert stim_object.stim_fn.__name__ == 'stimulus_pulsewave'
            amp, duty, period = stim_object.params
            ###period = stim_object.params[2]
            num_pulses_applied = get_npulses_from_tspan([times[0], times[-1]], duty, period)
            does_habituate, tth_continuous, tth_discrete, reldiff, peaks, troughs = time_to_habituate(
                times, output, num_pulses_applied)
            print("in plot_multi_traj_output() -- does_habituate ? =", does_habituate)
            if does_habituate:
                if scale_by_period:
                    ax.axvline(tth_continuous / period_S1, linestyle='--', linewidth=1, color=colors[idx])
                else:
                    ax.axvline(tth_continuous, linestyle='--', linewidth=1, color=colors[idx])

        if fillstimulus:
            assert ode_instance.ode_base.dim_control == 1
            control_obj = ode_instance.ode_base.control_objects[0]
            amp_stim = control_obj.params[0]
            stim_of_t = control_obj.fn_prepped(times)
            ax.fill_between(
                scaled_times, 0, output * stim_of_t / amp_stim,
                facecolor='gainsboro',
                alpha=0.2)

        if separate:
            ax.legend()

    axarr[-1, 0].set_ylabel(list_odemodel_instances[0].ode_base.output_str)
    axarr[-1, 0].set_xlabel(times_xlabel)
    if not separate:
        axarr[0, 0].legend()
    if title is not None:
        axarr[0, 0].set_title(title)
    plt.show()
    return


def plot_multi_traj_phasespace(list_odemodel_instances, indices, title=None, color_qualitative=True):
    """
    Phase plots of all the trajectories defined within "list_odemodel_instances"
    - the phase...
    """
    n = len(list_odemodel_instances)
    colors = line_colors(n, discrete=color_qualitative)

    max_ode_dimension = 0
    min_ode_dimension = int(1e9)
    idx_smallest_model = 0
    idx_biggest_model = 0
    for idx, odem in enumerate(list_odemodel_instances):
        dim_ode = odem.ode_base.dim_ode
        if dim_ode > max_ode_dimension:
            max_ode_dimension = dim_ode
            idx_biggest_model = idx
        if dim_ode < min_ode_dimension:
            idx_smallest_model = idx
    assert len(indices) in [2, 3]
    assert max(indices) <= min_ode_dimension

    if len(indices) == 3:
        ax = plt.figure().add_subplot(projection='3d')
        for idx, ode_instance in enumerate(list_odemodel_instances):
            label = ode_instance.label
            r = ode_instance.history_state  # times x dim_ode
            ax.plot(r[:, indices[0]],
                    r[:, indices[1]],
                    r[:, indices[2]], lw=0.5, label=label, c=colors[idx])
        ax.set_xlabel(list_odemodel_instances[idx_biggest_model].ode_base.ode_long[indices[0]])
        ax.set_ylabel(list_odemodel_instances[idx_biggest_model].ode_base.ode_long[indices[1]])
        ax.set_zlabel(list_odemodel_instances[idx_biggest_model].ode_base.ode_long[indices[2]])
    else:
        assert len(indices) == 2
        ax = plt.figure().gca()
        for idx, ode_instance in enumerate(list_odemodel_instances):
            label = ode_instance.label
            r = ode_instance.history_state
            ax.plot(r[:, indices[0]],
                    r[:, indices[1]],
                    lw=0.5, label=label, c=colors[idx])
        ax.set_xlabel(list_odemodel_instances[idx_biggest_model].ode_base.ode_long[indices[0]])
        ax.set_ylabel(list_odemodel_instances[idx_biggest_model].ode_base.ode_long[indices[1]])

    if title is not None:
        plt.title(title)
    else:
        varnames = ''.join(['%s' % list_odemodel_instances[idx_biggest_model].ode_base.ode_short[i] for i in indices])
        plt.title('Phase portrait: ' + varnames)
    plt.legend()
    plt.show()
    return


def get_multi_traj_output_hstats(list_odemodel_instances, tth_threshold=0.01):
    """
    Prepares data for plot_multi_traj_output_hstats()
    """
    n = len(list_odemodel_instances)
    hstats_dict = dict()

    for idx in range(n):

        ode_instance = list_odemodel_instances[idx]
        label = ode_instance.label

        times = ode_instance.history_times
        r = ode_instance.history_state  # times x dim_ode
        stim_object = ode_instance.ode_base.control_objects[0]
        stim_of_t = stim_object.fn_prepped(times)

        output = ode_instance.ode_base.output_fn(r, stim_of_t)

        stim_name = stim_object.stim_fn.__name__
        assert stim_object.stim_fn.__name__ == 'stimulus_pulsewave'
        period_S1 = ode_instance.ode_base.control_objects[0].params[2]
        amp, duty, period = stim_object.params
        num_pulses_applied = get_npulses_from_tspan([times[0], times[-1]], duty, period)

        print('get hstats for ode:', label)
        does_habituate, tth_continuous, tth_discrete, reldiff, peaks, troughs = time_to_habituate(
            times, output, num_pulses_applied, tth_threshold=tth_threshold)

        hstats_dict[idx] = dict()
        hstats_dict[idx]['times'] = times
        hstats_dict[idx]['input'] = stim_of_t
        hstats_dict[idx]['states'] = r
        hstats_dict[idx]['output'] = output
        hstats_dict[idx]['label'] = label
        hstats_dict[idx]['stimulus_name'] = stim_name
        hstats_dict[idx]['period_S1'] = period_S1
        hstats_dict[idx]['does_habituate'] =  does_habituate
        hstats_dict[idx]['tth_continuous'] = tth_continuous
        hstats_dict[idx]['tth_discrete'] = tth_discrete
        hstats_dict[idx]['reldiff'] = reldiff
        hstats_dict[idx]['peaks_idx'] = peaks
        hstats_dict[idx]['peaks_vals'] = output[peaks]
        hstats_dict[idx]['troughs_idx'] = troughs
        hstats_dict[idx]['troughs_vals'] = output[troughs]

    return list_odemodel_instances, hstats_dict


def plot_multi_traj_output_hstats(list_odemodel_instances, hstats_dict,
                                  title=None, separate=False, scale_by_period=True, tth_threshold=0.01,
                                  color_qualitative=True):
    """
    Plot subplots of "y(t) vs t" for a provided list of ode models (instances of custom class ODEModel)
    - plot are based on the class attributes: history_times and history_state
    - emphasize habituation statistics (time-to-habituate; relative peak heights)

    Intended for use with the following preliminary call:
        list_odemodel_instances, hstats_dict = get_multi_traj_output_hstats(list_odemodel_instances, ...)
    """
    n = len(list_odemodel_instances)
    colors = line_colors(n, discrete=color_qualitative)

    if separate:
        fig, axarr = plt.subplots(n, 3, squeeze=False, sharex='col', figsize=(12, 5))
    else:
        fig, axarr = plt.subplots(1, 3, squeeze=False, sharex='col', figsize=(12, 3))

    for idx in range(n):
        if separate:
            ax = axarr[idx, :]
        else:
            ax = axarr[0, :]

        times = hstats_dict[idx]['times']
        output = hstats_dict[idx]['output']
        label = hstats_dict[idx]['label']
        stimulus_name = hstats_dict[idx]['stimulus_name']
        period_S1 = hstats_dict[idx].get('period_S1', None)
        does_habituate = hstats_dict[idx]['does_habituate']
        tth_continuous = hstats_dict[idx]['tth_continuous']
        tth_discrete = hstats_dict[idx]['tth_discrete']
        reldiff = hstats_dict[idx]['reldiff']
        peaks_idx = hstats_dict[idx]['peaks_idx']
        peaks_vals = hstats_dict[idx]['peaks_vals']
        troughs_idx = hstats_dict[idx]['troughs_idx']
        troughs_vals = hstats_dict[idx]['troughs_vals']
        print("in plot_multi_traj_output() -- %s -- does_habituate ? =" % label, does_habituate)

        assert stimulus_name == 'stimulus_pulsewave'

        if n > 5 and idx not in [0, n-1]:
            # only legend first and last curve if there are more than 5 curves
            label = None

        if scale_by_period:
            scaled_times = times / period_S1
            times_xlabel = r"time $\tau = t/T$"
        else:
            scaled_times = times
            times_xlabel = r"time $t$"

        # Three main curves
        ax[0].plot(scaled_times,                 output,                 c=colors[idx], label=label)
        ax[1].plot(scaled_times[peaks_idx],      output[peaks_idx], 'x', c=colors[idx])
        ax[2].plot(scaled_times[peaks_idx[:-1]], reldiff,           '^', c=colors[idx])

        # For the curves which do habituate, draw vertical lines at the time-to-habituate
        if does_habituate:
            if scale_by_period:
                plot_tth_0 = tth_continuous / period_S1
            else:
                plot_tth_0 = tth_discrete
            ax[0].axvline(plot_tth_0, linestyle='--', linewidth=1, color=colors[idx])
            ax[1].axvline(tth_discrete, linestyle='--', linewidth=1, color=colors[idx])
            ax[2].axvline(tth_discrete, linestyle='--', linewidth=1, color=colors[idx])
        ax[2].axhline(tth_threshold, linestyle='-', linewidth=1, color='grey')
        if separate:
            ax[0].legend()

    axarr[-1, 0].set_ylabel(list_odemodel_instances[0].ode_base.output_str)
    axarr[-1, 0].set_xlabel(times_xlabel)
    axarr[-1, 1].set_xlabel(r"# pulses")
    axarr[-1, 2].set_xlabel(r"# pulses")
    if not separate:
        axarr[0, 0].legend()
    if title is not None:
        axarr[0, 0].set_title(title)
        axarr[0, 1].set_title('Peak values')
        axarr[0, 2].set_title('Relative peak differences')
    plt.show()
    return


def plot_bode_style(list_odemodel_instances, hstats_dict, param_axis, title='', log_axis=False, decibelform=False,
                    freq_flag=True, xlabel=r'frequency $1/T$'):
    """
    Intended for use with the following preliminary call:
        list_odemodel_instances, hstats_dict = get_multi_traj_output_hstats(list_odemodel_instances, ...)
    # TODO decibelform (bool) -- y axis should be 'decibel' (relative?) units
    """

    def mask_where_cond(arr, arr_bool, cond=False):
        arr_masked = arr.copy()
        arr_bool = np.array(arr_bool)
        arr_masked[arr_bool == cond] = np.nan
        return arr_masked

    n = len(list_odemodel_instances)

    if freq_flag:
        fname = 'bodeplot_freq'
    else:
        fname = 'bodeplot_intensity'

    nplot = 4
    fig, axarr = plt.subplots(nplot, 1, figsize=(12, 8), squeeze=False, sharex=True)

    curve_peak_first = np.zeros(n, dtype=float)
    curve_peak_first_label = r'peak$_1$'
    curve_peak_inf = np.zeros(n, dtype=float)
    curve_peak_inf_label = r'peak$_{\infty}$'
    curve_peak_max = np.zeros(n, dtype=float)
    curve_peak_max_label = r'peak$_{max}$'
    curve_peak_argmax = np.zeros(n, dtype=float)
    curve_peak_argmax_label = r'argmax of peaks vector'
    curve_tthc_wall = np.zeros(n, dtype=float)
    curve_tthc_norm = np.zeros(n, dtype=float)
    curve_tthc_wall_label = r'TTH$_{absolute}$'
    curve_tthc_norm_label = r'TTH$_{absolute} / T$'
    curve_tthd = np.zeros(n, dtype=float)
    curve_tthd_label = r'TTH$_{discrete}$'
    indicator_h1 = [False] * n
    indicator_h1_label = "Does habituate"

    for idx, ode_model in enumerate(list_odemodel_instances):

        if freq_flag:
            # in this case, param_axis is frequency and the period is changing
            period_S1 = 1 / param_axis[idx]
        else:
            period_S1 = hstats_dict[idx].get('period_S1', None)

        times = hstats_dict[idx]['times']
        output = hstats_dict[idx]['output']
        label = hstats_dict[idx]['label']
        stimulus_name = hstats_dict[idx]['stimulus_name']

        peaks_idx = hstats_dict[idx]['peaks_idx']
        peaks_vals = hstats_dict[idx]['peaks_vals']
        troughs_idx = hstats_dict[idx]['troughs_idx']
        troughs_vals = hstats_dict[idx]['troughs_vals']

        does_habituate = hstats_dict[idx]['does_habituate']
        reldiff = hstats_dict[idx]['reldiff']

        curve_peak_first[idx] = peaks_vals[0]
        curve_peak_inf[idx] = peaks_vals[-1]
        curve_peak_max[idx] = np.max(peaks_vals)
        curve_peak_argmax[idx] = np.argmax(peaks_vals)
        if does_habituate:
            curve_tthc_wall[idx] = hstats_dict[idx]['tth_continuous']
            curve_tthc_norm[idx] = hstats_dict[idx]['tth_continuous'] / period_S1
            curve_tthd[idx] = hstats_dict[idx]['tth_discrete']
        indicator_h1[idx] = does_habituate
        print('idx', idx, ', H1? =', does_habituate, 'npeaks =', len(peaks_vals))

    curve_peak_diff = curve_peak_max - curve_peak_inf
    curve_peak_diff_label = r'peak$_{max}$ - peak$_{\infty}$'

    curve_peak_ratio = curve_peak_inf / curve_peak_max
    curve_peak_ratio_label = r'peak$_{\infty}$ / peak$_{max}$'

    # create masked curve variants (mask where H1 fails)
    curve_peak_first_masked_H1 = mask_where_cond(curve_peak_first, indicator_h1)
    curve_peak_inf_masked_H1 = mask_where_cond(curve_peak_inf, indicator_h1)
    curve_peak_max_masked_H1 = mask_where_cond(curve_peak_max, indicator_h1)
    curve_peak_diff_masked_H1 = mask_where_cond(curve_peak_diff, indicator_h1)
    curve_peak_ratio_masked_H1 = mask_where_cond(curve_peak_ratio, indicator_h1)
    curve_peak_argmax_masked_H1 = mask_where_cond(curve_peak_argmax, indicator_h1)
    curve_tthc_wall_masked_H1 = mask_where_cond(curve_tthc_wall, indicator_h1)
    curve_tthc_norm_masked_H1 = mask_where_cond(curve_tthc_norm, indicator_h1)
    curve_tthd_masked_H1 = mask_where_cond(curve_tthd, indicator_h1)

    # main curves
    axarr[0, 0].plot(param_axis, curve_peak_max, '--x', color='black', label=curve_peak_max_label)
    axarr[0, 0].plot(param_axis, curve_peak_inf, '--x', color='green', label=curve_peak_inf_label)
    axarr[0, 0].plot(param_axis, curve_peak_max_masked_H1, 'o', color='black', label=indicator_h1_label)
    axarr[0, 0].plot(param_axis, curve_peak_inf_masked_H1, 'o', color='green')

    #axarr[0, 0].plot(param_axis, curve_peak_first, '--x', color='green', label=curve_peak_first_label)
    #axarr[0, 0].plot(param_axis, curve_peak_first_masked_H1, 'o', color='green', label=indicator_h1_label)

    axarr[1, 0].plot(param_axis, curve_peak_ratio, '--x', color='brown', label=curve_peak_ratio_label)
    axarr[1, 0].plot(param_axis, curve_peak_ratio_masked_H1, 'o', color='brown', label=indicator_h1_label)
    axarr[1, 0].axhline(0.5, linestyle='--', color='brown')
    #axarr[1, 0].plot(param_axis, curve_peak_diff, '--x', color='purple', label=curve_peak_diff_label)
    #axarr[1, 0].plot(param_axis, curve_peak_diff_masked_H1, 'o', color='purple', label=indicator_h1_label)

    axarr[2, 0].plot(param_axis, curve_tthc_wall_masked_H1, '--o', color='red', label=curve_tthc_wall_label)
    axarr[2, 0].plot(param_axis, curve_tthc_norm_masked_H1, '--o', color='blue', label=curve_tthc_norm_label)

    axarr[3, 0].plot(param_axis, curve_peak_argmax, '--x', color='k', label=curve_peak_argmax_label)
    axarr[3, 0].plot(param_axis, curve_peak_argmax_masked_H1, 'o', color='k', label=indicator_h1_label)

    axarr[0, 0].set_ylabel('peaks')
    axarr[1, 0].set_ylabel('peaks (compare)')
    axarr[2, 0].set_ylabel('TTH')
    axarr[3, 0].set_ylabel('arg max peaks')
    axarr[-1, 0].set_xlabel(xlabel)

    # setup legends
    for i in [0, 1, 2, 3]:
        axarr[i, 0].legend()

    # setup axis scaling and zero hlines
    if log_axis:
        axarr[0, 0].set_yscale('log')
        axarr[1, 0].set_yscale('log')
    for idx in range(nplot):
        if log_axis:
            axarr[i, 0].set_xscale('log')
            '''
            axarr[2, 0].axhline(0)
            axarr[2, 0].set_xlim(-0.02 * freq_axis[-1], 1.02 * freq_axis[-1])
            axarr[3, 0].axhline(0)
            axarr[3, 0].set_xlim(-0.02 * freq_axis[-1], 1.02 * freq_axis[-1])
            '''
        else:
            axarr[i, 0].axhline(0)

            #axarr[2, 0].axhline(0)
            #axarr[2, 0].set_xlim(-0.02 * freq_axis[-1], 1.02 * freq_axis[-1])
            #axarr[3, 0].axhline(0)
            #axarr[3, 0].set_xlim(-0.02 * freq_axis[-1], 1.02 * freq_axis[-1])
    '''
    axarr[2, 0].axhline(0)
    axarr[2, 0].set_xlim(-0.02*freq_axis[-1], 1.02*freq_axis[-1])
    axarr[3, 0].axhline(0)
    axarr[3, 0].set_xlim(-0.02*freq_axis[-1], 1.02*freq_axis[-1])
    '''

    fpath = DIR_OUTPUT + os.sep + fname
    if log_axis:
        title += ' (log response)'
        fpath += '_log'
    axarr[0, 0].set_title(title)
    plt.savefig(fpath + '.png')
    plt.savefig(fpath + '.svg')
    plt.show()

    return
