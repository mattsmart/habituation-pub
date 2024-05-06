import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.integrate import solve_ivp

from class_ode_base import ODEBase
from class_ode_stimulus import ODEStimulus
from defined_ode_fn import *
from defined_stimulus import delta_fn_amp, stimulus_constant, stimulus_pulsewave, stimulus_pulsewave_of_pulsewaves, \
    stimulus_staircase, stitch_tspans_from_stimulus
from preset_ode_model import PRESETS_ODE_MODEL, ode_model_preset_factory
from preset_solver import PRESET_SOLVER
from settings import DIR_OUTPUT, STYLE_DYNAMICS_VALID, STYLE_DYNAMICS



class ODEModel():
    def __init__(
            self,
            ode_base,
            init_cond,
            t0,
            tmax,
            solver_settings=PRESET_SOLVER['solve_ivp_radau_default'],
            history_times=None,
            history_state=None,
            label='',
            outdir=DIR_OUTPUT):
        self.ode_base = ode_base
        self.solver_settings = solver_settings
        self.init_cond = init_cond
        self.t0 = t0
        self.tmax = tmax
        self.history_times = history_times
        self.history_state = history_state
        self.label = label
        self.outdir = outdir

        # - assertions enforcing expectations on the chosen ODE
        assert len(self.init_cond) == self.ode_base.dim_ode

        # (3) regulate the ODE solver
        # - fill in solver settings attributes
        self.style_solver = self.solver_settings['dynamics_method']
        self.params_solver = self.solver_settings['kwargs']
        assert self.style_solver in STYLE_DYNAMICS_VALID  # always
        assert self.style_solver == 'scipy_solve_ivp'     # this one in particular, for now

        # - update solver kwargs which depend on properties of ODE, time forcing
        #   - specifically S(t) 'duty' parameter
        #   - observed that short bursts of forcing are missed by the adaptive timestep of default solver scipy Radau
        if self.params_solver['method'] in ['Radau', 'BDF', 'LSODA', 'RK45', 'DOP853']:
            # use duty parameter to limit Radau solver max_step
            max_step_given = self.params_solver.get('max_step', np.Inf)
            self.params_solver['max_step'] = min(self.ode_base.max_step_augment, max_step_given)

    def extend_history_time_and_state(self, traj_times, traj_states):
        if self.history_times is None:
            assert self.history_state is None
            self.history_times = traj_times
            self.history_state = traj_states
        else:

            # edge case to prevent double counting of end/start points when stitching trajectories
            if traj_times[0] == self.history_times[-1]:
                self.history_times = self.history_times[:-1]
                self.history_state = self.history_state[:-1, :]
            total_time_pts = len(self.history_times) + len(traj_times)
            new_history_times = np.zeros(total_time_pts)
            new_history_state = np.zeros((total_time_pts, self.ode_base.dim_ode))
            # fill in times
            new_history_times[0:len(self.history_times)] = self.history_times
            new_history_times[len(self.history_times):] = traj_times
            # fill in state
            new_history_state[0:len(self.history_times), :] = self.history_state
            new_history_state[len(self.history_times):, :] = traj_states
            # assignment
            self.history_times = new_history_times
            self.history_state = new_history_state
        return

    def clear_history(self):
        self.history_times = None
        self.history_state = None

    def helper_traj_t0_t1_init_cond(self, t0, t1, init_cond, update_history=False):
        # used in methods: propagate() and trajectory_pulsewave()
        if t0 is None:
            t0 = self.t0
        if t1 is None:
            t1 = self.tmax
        if update_history and self.history_times is not None:
            #assert t0 == self.history_times[-1]
            assert np.isclose(t0, self.history_times[-1])
        # set initial condition
        if update_history and self.history_state is not None:
            assert np.allclose(init_cond, self.history_state[-1, :])
        elif init_cond is None:
            init_cond = self.init_cond
        return t0, t1, init_cond

    # TODO deprecated; new usage should be self.ode_base.fn(t, x)
    def xdot(self, t, x):
        #return self.fn_ode(t, x, self.ode_base.params, self.params_stim_fn_arg_pairs)
        return self.ode_base.fn_prepped(t, x)

    def traj_and_runtime(self, traj_pulsewave_stitch=True, update_history=True, verbose=False, **stitch_kwargs):
        """
        currently, two methods to get ODE trajectory:
        - generic: traj_pulsewave_stitch = False -- call self.propagate()
        - special: traj_pulsewave_stitch = True  -- call self.trajectory_pulsewave()
                   splits the integration across tspan segments,
                   with constant stimulus on each
        """
        if verbose:
            print('Full trajectory from arbitrary init cond...')
            print('Model:', self.label)
        if traj_pulsewave_stitch:
            start = time.time()
            r, times = self.trajectory_pulsewave(update_history=update_history, **stitch_kwargs)
            end = time.time()
        else:
            start = time.time()
            r, times = self.propagate(update_history=update_history, verbose=verbose)
            end = time.time()
        runtime = end - start
        if verbose:
            print('time (traj_and_runtime call) =', runtime)
        return r, times, runtime

    def propagate(self, update_history=False, init_cond=None, t0=None, t1=None, params_solver=None, verbose=False):
        """
        Note the max step is determined by the smallest timescale in the stimulus fn S(t)
        - this is probably overkill but without it the adaptive stepping fails for small duty values

        # can instead do dynamics_general.py
        # contains simulate_dynamics_general(init_cond, times, single_cell, dynamics_method="solve_ivp", **solver_kwargs):
        # calls ode_solve_ivp()
        """
        # specify time limits of integration and init_cond
        t0, t1, init_cond = self.helper_traj_t0_t1_init_cond(t0, t1, init_cond, update_history=update_history)
        tspan = (t0, t1)

        # other
        if params_solver is None:
            params_solver = self.params_solver

        sol = solve_ivp(
            self.ode_base.fn_prepped,
            tspan,
            init_cond,
            **params_solver)
        r = np.transpose(sol.y)
        times = sol.t

        if verbose or not sol.success:
            print('(verbose) ode_model.propagate() solution attributes:')
            print('\tsol.nfev =', sol.nfev)
            print('\tsol.njev =', sol.njev)
            print('\tsol.nlu =', sol.nlu)
            print('\tsol.status =', sol.status)
            print('\tsol.message =', sol.message)
            print('\tsol.success =', sol.success)
            print('target tspan:', tspan)
            print('end time:', times[-1])
            print('init cond:', init_cond)
            print('end state:', r[-1, :])
            print('params_solver:', params_solver)

        assert sol.success

        if update_history:
            self.extend_history_time_and_state(times, r)

        return r, times

    def trajectory_pulsewave(self, update_history=False, t0=None, t1=None, init_cond=None,
                             forcetol=(1e-8, 1e-4), max_step=np.Inf):
        """
        Alternative version of trajectory designed for step function stimulus
        - rather than 1 call to ode_solve_ivp(), it stitches many calls on short windows of time
        - these windows define the regions of time where S(t) is constant
            - TODO further speedup -- modify ode_fn itself to receive constant S_0 (i.e. S = 0 or S = amplitude)
        This can enable speedup over blunt single call to: self.propagate_between(0.0, tmax)
        HOWEVER, it needs to be accompanied by increase in atol, rtol (for radau)
            e.g.
                maxstep = period * duty / 10, self.propagate_between(0.0, tmax) -- atol, rtol become less important
                    is slower than,
                maxstep = NaN,       self.trajectory_pulsewave()       -- atol=1e-12, rtol=1e-6
        """

        # (1) asserts and setup - need to setup a grid of tspans on which to propagate the trajectory in pieces
        #assert self.ode_base.dim_control == 1
        print('warning trajectory_pulsewave() - commented out the -- assert self.ode_base.dim_control == 1 --' )
        control_obj = self.ode_base.control_objects[0]
        fn_stim = control_obj.stim_fn
        args_stim = control_obj.params
        assert fn_stim.__name__ in ['stimulus_pulsewave', 'stimulus_pulsewave_of_pulsewaves']

        # (2) specify time limits of integration and init_cond
        t0, t1, init_cond = self.helper_traj_t0_t1_init_cond(t0, t1, init_cond, update_history=update_history)

        tspans = stitch_tspans_from_stimulus(fn_stim, args_stim, t0, t1)

        # (3) update params_solver to remove max_step constraint
        if self.params_solver['method'] in ['Radau', 'BDF', 'LSODA', 'RK45', 'DOP853']:
            params_solver = self.params_solver.copy()
            params_solver['max_step'] = max_step

            # adjust atol and rtol (as max_step depends on them adaptively now)
            if forcetol is not None:
                params_solver['atol'] = forcetol[0]
                params_solver['rtol'] = forcetol[1]
            else:
                assert params_solver['atol'] <= 1e-12
                assert params_solver['rtol'] <= 1e-6

        else:
            params_solver = None  # self.trajectory() will then use self.params_solver

        # (4) propagate the trajectories over the sequence of tspan pairs and stitch them together
        if update_history:
            last_time_idx = 0 if self.history_times is None else len(self.history_times)
            for idx, triple in enumerate(tspans):
                t_left, t_right, _ = triple
                if self.history_state is not None:
                    init_cond = self.history_state[-1, :]
                self.propagate(update_history=update_history, init_cond=init_cond, t0=t_left, t1=t_right,
                               params_solver=params_solver)
            r = self.history_state[last_time_idx:, :]
            times = self.history_times[last_time_idx:]
        else:
            backup_history_state = None if self.history_state is None else np.copy(self.history_state)
            backup_history_times = None if self.history_times is None else np.copy(self.history_times)
            self.history_state = None
            self.history_times = None
            for idx, triple in enumerate(tspans):
                t_left, t_right, _ = triple
                if self.history_state is not None:
                    init_cond = self.history_state[-1, :]

                self.propagate(update_history=True, init_cond=init_cond, t0=t_left, t1=t_right,
                               params_solver=params_solver)
            r = np.copy(self.history_state)
            times = np.copy(self.history_times)
            # now reset back to old state
            self.history_state = backup_history_state
            self.history_times = backup_history_times

        return r, times

    def propagate_until(self, t):
        """
        like self.trajectory() but uses history so that it can be called in sequential steps

        e.g. self.propagate_until(10), self.propagate_until(20)

        Question: how does this impact self.tmax attribute? should we even have one?
        Question: should this have sister method self.propagate_for(t_duration) ?
        """
        # TODO implement with care for these attributes: t0, tmax, init_cond, history_times, history_state
        return

    def interpolate_trajectory(self, force_dt=None, use_min_dt=False, update_attr=False):
        """
        returns interpolated versions of t, x
        if not force_dt:
            use_min_dt:        fixed dt will be the smallest timestep
            if not use_min_dt: fixed dt will be 2 * avg timestep
        else:
            consider default of force_dt=0.001...
        """
        sol_t = self.history_times  # dim_time
        sol_state = self.history_state  # dim_time x dim_state

        dim_time, dim_state = sol_state.shape
        assert len(sol_t) == dim_time

        # 1)
        t0 = sol_t[0]
        t1 = sol_t[-1]
        if force_dt is None:
            if use_min_dt:
                # choose the minimum adaptive timestep as our timestep
                force_dt = np.min(sol_t[1:] - sol_t[:-1])  # can get very big - use with care
            else:
                # choose fixed timestep as a multiple of the average timestep
                force_dt = 2 * (t1 - t0) / dim_time
        else:
            assert force_dt < 0.5 * (t1 - t0)

        nn_samples = int((t1 - t0) / force_dt) + 1
        times_interpolated = np.linspace(t0, t1, nn_samples)
        state_interpolated = np.zeros((nn_samples, dim_state))
        print('performing interpolation (n=%d --> n=%d timepoints)...' % (dim_time, nn_samples))

        # 2)
        for idx in range(dim_state):
            # Linear interpolation
            coordinate_interpolated = np.interp(
                times_interpolated, sol_t, sol_state[:, idx])
            # Cubic Spline interpolation
            '''
            cubic_spline = CubicSpline(
                cellgraph.times_history, cellgraph.state_history[idx, :])
            coordinate_interpolated = cubic_spline(times_interpolated)
            '''
            state_interpolated[:, idx] = coordinate_interpolated

        if update_attr:
            print('updating attributes: history_state, history_times (ODEModel interpolation)...')
            self.history_state = state_interpolated
            self.history_times = times_interpolated
        print('done ODEModel interpolation')

        return state_interpolated, times_interpolated

    def poincare_map_at_x(self, x, t0=0.0):
        """
            map: x -> x' after time T of the flow
            convention: assume start at t0 = 0.0 (i.e. fix the poincare segment at some phase theta_0)
        This function returns the output of the above map, enforcing t0 = 0.0 as a convention.
        """
        assert self.ode_base.period_forcing is not None
        t0_plus_T = t0 + self.ode_base.period_forcing
        # TODO maybe should have style_integration attribute to whole class? to make this choice less hardcoded
        r, times = self.trajectory_pulsewave(update_history=False, t0=t0, t1=t0_plus_T, init_cond=x)
        assert times[-1] == t0_plus_T  # TODO care when trajectory_pulsewave
        #r, times = self.propagate(update_history=False, t0=0.0, t1=self.period_forcing, init_cond=x)
        P_of_x = r[-1, :]  # last point of the flow
        return P_of_x, r, times

    def find_point_on_limit_cycle(self, init_cond=None, newton_tol=1e-6, ddx_step=1e-5):
        # TODO consider alternate methods e.g. just keep repeating flow instead of newton raphson
        # TODO this function could probably be moved out of class methods... just needs few attributes passed to it
        assert self.ode_base.period_forcing is not None
        if init_cond is None:
            init_cond = np.zeros(self.ode_base.dim_ode)

        # reused variables
        h = ddx_step
        id_nxn = np.eye(self.ode_base.dim_ode, self.ode_base.dim_ode)
        njac = np.zeros((self.ode_base.dim_ode, self.ode_base.dim_ode))
        x = init_cond

        def q_of_x(x):
            """
            Create function q(x) whose roots will correspond to points on the LC
            Define: P(x) -- the poincare map associated with the period T flow of x
            Define: q(x) = P(x) - x
            """
            residual = x - self.poincare_map_at_x(x)[0]
            return residual

        def jac_of_q_at_x(x0, qvec_at_x0):
            """
            Numerically evaluate jacobian of q(x) at x0 using finite differences
            """
            for idx in range(self.ode_base.dim_ode):
                qvec_at_x0_plus_h = q_of_x(x0 + id_nxn[:, idx] * h)  # perturb x in direction x_i with magnitude +h
                njac[:, idx] =  (qvec_at_x0_plus_h - qvec_at_x0) / h  # TODO consider alternate schemes
            return njac

        def solve_linear_system(A, b):
            """
            Solves Ax = b for x by inverting A
            """
            # TODO consider alt methods e.g. GMRES
            LHS = np.dot( np.linalg.inv(A), b)
            return LHS

        qvec_at_x = q_of_x(init_cond)
        while np.linalg.norm(qvec_at_x) > newton_tol:
            jac_of_q_at_x(x, qvec_at_x)
            ddx_step = -1 * solve_linear_system(njac, qvec_at_x)
            x = x + ddx_step
            qvec_at_x = q_of_x(x)

        return x

    def plot_simple_trajectory(self, axarr=None, title=None, stimfill=True):
        """
        Plot subplot stack of x1(t), ... xN(t) for the N-dimensional ODE trajectory
        """
        t = self.history_times
        r = self.history_state
        if axarr is None:
            fig, axarr = plt.subplots(self.ode_base.dim_ode, 1, sharex=True, squeeze=False)
        for i in range(self.ode_base.dim_ode):
            print(t.shape)
            print(r.shape)
            axarr[i, 0].plot(t, r[:, i], label='numeric')
            axarr[i, 0].set_ylabel(self.ode_base.ode_short[i])

        # plot analytic prediction of trajectory, if available
        if self.ode_base.state_analytic is not None:
            for fn in self.ode_base.state_analytic:
                state_analytic = fn(t)
                # axarr.plot(t, output_analytic, '--', c='black', lw=1.0, label=fn.__name__, zorder=11)
                for i in range(self.ode_base.dim_ode):
                    axarr[i, 0].plot(t, state_analytic[i], '--', lw=1.0, label=fn.__name__, zorder=11)

        axarr[-1, 0].set_xlabel(r"time $\tau = t/T$")

        # decorate plot by filling below the curves where stimulus is on/off
        if stimfill:
            assert self.ode_base.dim_control == 1
            """
            fn_stim = self.params_stim_fn_arg_pairs[0][0]
            args_stim = self.params_stim_fn_arg_pairs[0][1]
            amp_stim = args_stim[0]
            stim_of_t = fn_stim(t, *args_stim)
            """
            control_obj = self.ode_base.control_objects[0]
            if control_obj.stim_fn_name == 'stimulus_staircase':
                amp_stim = 1.0 / control_obj.params[0]  # TODO fix...
                print('TODO fix amp_stim for stimfill in class_ode_model.py plot_simple_trajectory(...)')
            else:
                amp_stim = control_obj.params[0]

            stim_of_t = control_obj.fn_prepped(t)
            for idx in range(r.shape[1]):
                axarr[idx, 0].fill_between(
                    t, 0, r[:, idx] * stim_of_t / amp_stim,
                    facecolor='gainsboro',
                    alpha=0.8)

        if title is not None:
            axarr[0, 0].set_title(title)
        plt.legend()
        plt.show()  # TODO show flag and output folder for class to point to?
        return axarr

    def plot_output_timeseries(self, axarr=None, title=None, stimfill=True):
        """
        Plot subplot stack of output function for ODEbase, typically xN(t) for the N-dimensional ODE trajectory
        """
        t = self.history_times
        r = self.history_state
        control_obj = self.ode_base.control_objects[0]
        stim_of_t = control_obj.fn_prepped(t)

        output = self.ode_base.output_fn(r, stim_of_t)

        if axarr is None:
            fig, axarr = plt.subplots(1, 1, figsize=(4, 4))
        axarr.plot(t, output, lw=1.5, label='numeric', alpha=0.8, zorder=10)
        # plot the list of prepared output functions, analytically derived
        if self.ode_base.output_analytic is not None:
            for fn in self.ode_base.output_analytic:
                output_analytic = fn(t)
                #axarr.plot(t, output_analytic, '--', c='black', lw=1.0, label=fn.__name__, zorder=11)
                axarr.plot(t, output_analytic, '--', lw=1.0, label=fn.__name__, zorder=11)
        axarr.set_ylabel(self.ode_base.output_str)
        axarr.set_xlabel(r"time $\tau = t/T$")
        axarr.axhline(0, c='k', linestyle='--', lw=0.25)

        # decorate plot by filling below the curves where stimulus is on/off
        if stimfill:
            assert self.ode_base.dim_control == 1
            control_obj = self.ode_base.control_objects[0]
            amp_stim = control_obj.params[0]
            stim_of_t = control_obj.fn_prepped(t)
            axarr.fill_between(
                t, 0, output * stim_of_t / amp_stim,
                facecolor='gainsboro', alpha=0.8)

        if title is not None:
            axarr.set_title(title)
        plt.legend()
        plt.show()  # TODO show flag and output folder for class to point to?
        return axarr

    def plot_phase_portrait(self, indices, ax=None, title=None):
        """
        Plot parametric trajectory in subspace specified by integer indices from [x1, ..., xN]
        """
        assert len(indices) in [2, 3]
        r = self.history_state

        if len(indices) == 3:
            if ax is None:
                ax = plt.figure().add_subplot(projection='3d')
            ax.plot(r[:, indices[0]],
                    r[:, indices[1]],
                    r[:, indices[2]], lw=0.5)
            ax.set_xlabel(self.ode_base.ode_long[indices[0]])
            ax.set_ylabel(self.ode_base.ode_long[indices[1]])
            ax.set_zlabel(self.ode_base.ode_long[indices[2]])
        else:
            if ax is None:
                ax = plt.figure().gca()
            ax.plot(r[:, indices[0]],
                    r[:, indices[1]],
                    lw=0.5)
            ax.set_xlabel(self.ode_base.ode_long[indices[0]])
            ax.set_ylabel(self.ode_base.ode_long[indices[1]])

        if title is not None:
            plt.title(title)
        else:
            plt.title('Phase portrait: ' + ''.join(['%s' % self.ode_base.ode_short[i] for i in indices]))
        #plt.legend()
        plt.show()  # TODO show flag and output folder for class to point to?
        return ax

    def printout(self, suffix=''):
        print('shape of self.history_times', self.history_times.shape)
        np.savetxt(self.outdir + os.sep + 'history_times%s.txt' % suffix, self.history_times)
        np.savetxt(self.outdir + os.sep + 'history_state%s.txt' % suffix, self.history_state)
        return

    def write_metadata(self, fpath=None):
        """
        Generates k files:
        - io_dict['runinfo'] - stores metadata on the specific run
        """
        # TODO implement
        # First file: basedir + os.sep + run_info.txt
        if fpath is None:
            fpath = self.outdir + os.sep + 'runinfo.txt'
        with open(fpath, "w", newline='\n') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            # module choices
            """
            writer.writerow(['style_ode', self.style_ode])
            writer.writerow(['style_dynamics', self.style_dynamics])
            writer.writerow(['style_detection', self.style_detection])
            writer.writerow(['style_division', self.style_division])
            writer.writerow(['style_diffusion', self.style_diffusion])
            """
            # dimensionality
            """
            writer.writerow(['num_cells', self.num_cells])
            writer.writerow(['sc_dim_ode', self.graph_dim_ode])
            writer.writerow(['graph_dim_ode', self.sc_dim_ode])
            """
            # coupling settings
            """
            writer.writerow(['alpha_sharing', self.alpha_sharing])
            writer.writerow(['beta_sharing', self.beta_sharing])
            writer.writerow(['diffusion_arg', self.diffusion_arg])
            writer.writerow(['diffusion', self.diffusion])
            """
            # integration settings
            writer.writerow(['t0', self.t0])
            writer.writerow(['tmax', self.tmax])
            writer.writerow(['init_cond', self.init_cond])
            # ... more ...
            """
            # initialization of each cell
            X = self.state_to_rectangle(self.state_history)
            for cell in range(self.num_cells):
                writer.writerow(['cell_%d' % cell, X[:, cell, -1].flatten()])
            """
        # any single cell dynamics params
        """
        self.sc_template.write_ode_params(fpath)
        """
        return fpath


if __name__ == '__main__':

    use_preset = True
    # main settings
    timer = False
    plot_simple = True
    plot_phaseportrait = True
    assess_limitcycle = False
    assess_limitcycle_extra = False
    plot_custom_output_function = False

    if use_preset:
        #ode_model_input = PRESETS_ODE_MODEL['ode_custom_1_S1']
        #ode_model_input = PRESETS_ODE_MODEL['ode_custom_2_S1']  # model: tyson
        #ode_model_input = PRESETS_ODE_MODEL['ode_custom_3_S1']
        #ode_model_input = PRESETS_ODE_MODEL['ode_custom_3_staircase']
        #ode_model_input = PRESETS_ODE_MODEL['ode_custom_3_integrator_S1']
        #ode_model_input = PRESETS_ODE_MODEL['ode_custom_4_S1']
        #ode_model_input = PRESETS_ODE_MODEL['ode_custom_4_innerout_S1']
        #ode_model_input = PRESETS_ODE_MODEL['ode_custom_5_S1']
        #ode_model_input = PRESETS_ODE_MODEL['ode_martin_A_S1']
        #ode_model_input = PRESETS_ODE_MODEL['ode_Wang93_1d_S1']
        ode_model_input = PRESETS_ODE_MODEL['ode_linear_filter_1d_S1']

        #manual_init_cond = None  # None, np.array([0.0, 0.5])
        manual_init_cond = np.array([0.0])

        # manually adjust initial condition
        if manual_init_cond is None:
            ode_model = ODEModel(*ode_model_input['args'], **ode_model_input['kwargs'])
        else:
            ode_base, _, t0, tmax = ode_model_input['args']
            ode_model = ODEModel(ode_base, manual_init_cond, t0, tmax, **ode_model_input['kwargs'])

    else:
        # (1) specify stimulus function
        S1_duty = 0.01
        S1_period = 1.0
        S1 = ODEStimulus(
            stimulus_pulsewave,
            [delta_fn_amp(S1_duty, S1_period),  # amplitude
             S1_duty,  # duty in [0,1]
             S1_period,  # period of stimulus pulses (level 1)
             ],
            label=r'$u(t)$ = pulsewave'
        )
        control_objects = [S1]

        # (2) specify IVP and tspan for ODE
        init_cond = [0.0, 0.0, 0.0]
        t0 = 0.0
        nycles = 150.25
        tmax = nycles * S1_period

        p_ode = np.array([
            0.1,  # k_x+
            0.1,  # k_x-
            0.5,  # k_y+
            5.0,  # k_y-
            0,
            1e6,
        ])
        ode_name = 'ode_custom_4_innerout'
        params_ode = p_ode

        ode_base = ODEBase(
            ode_name,
            control_objects,
            params=p_ode,
        )

        # (3) specify custom solver settings
        solver_settings = dict(
            dynamics_method='scipy_solve_ivp',
            kwargs=dict(method='Radau', t_eval=None, atol=1e-12, rtol=1e-8, vectorized=False, jac=None),
        )

        # (4) instantiate the class
        ode_model = ODEModel(
            ode_base,
            init_cond,
            t0,
            tmax,
            solver_settings=solver_settings,
            label='example_class_ode_model_customized_in_main'
        )

    if timer:
        nn = 10000
        x0 = np.zeros(ode_model.ode_base.dim_ode)

        ode_base = ode_model.ode_base
        print(ode_base.control_objects[0].params)
        print(ode_base.fn_prepped)

        ode_base.fn_prepped(0, x0)  # initial call
        start = time.time()
        for i in range(nn):
            # u0 = stimulus_constant_jit(i, AMP)
            _ = ode_base.fn_prepped(i, x0)
        end = time.time()
        print('time odebase class =', end - start)

        ode_model.xdot(0, x0)
        start = time.time()
        for i in range(nn):
            xdot = ode_model.xdot(i, x0)
        end = time.time()
        print('time odemodel class =', end-start)

    # compute trajectory
    r, times, runtime = ode_model.traj_and_runtime(
        traj_pulsewave_stitch=True, update_history=True, verbose=True)

    # Code block to time the ODE integration
    #for idx in range(5):
    #    ode_model.trajectory_pulsewave()  # splits the integration across tspan segments, with constant stimulus on each
    #    ode_model.history_state = None
    #    ode_model.history_times = None

    # plotting
    # - simple trajectory - stacked subplots, x_i(t) vs t
    if plot_simple:
        ode_model.plot_simple_trajectory()
        ode_model.plot_output_timeseries()

    # - phase portraits of different pairs/triplets
    if plot_phaseportrait:
        if ode_model.ode_base.dim_ode >= 2:
            ode_model.plot_phase_portrait(indices=[0, 1])
        if ode_model.ode_base.dim_ode >= 3:
            ode_model.plot_phase_portrait(indices=[0, 2])
            ode_model.plot_phase_portrait(indices=[1, 2])
            ode_model.plot_phase_portrait(indices=[0, 1, 2])

    if assess_limitcycle:
        print('\nMain (2): Finding point on limit cycle...')
        ode_model.clear_history()  # refresh instance to store only LC trajectory for plotting
        xguess = ode_model.find_point_on_limit_cycle()
        print("xguess LC", xguess)
        r, times = ode_model.trajectory_pulsewave(update_history=True, t0=0.0, t1=ode_model.period_forcing, init_cond=xguess)
        #r, times = ode_model.propagate(update_history=True, t0=0.0, t1=ode_model.period_forcing, init_cond=xguess)

        # - simple trajectory - stacked subplots, x_i(t) vs t
        ode_model.plot_simple_trajectory()
        # - phase portraits of different pairs/triplets
        if ode_model.ode_base.dim_ode >= 2:
            ode_model.plot_phase_portrait(indices=[0, 1])
        if ode_model.ode_base.dim_ode >= 3:
            ode_model.plot_phase_portrait(indices=[0, 2])
            ode_model.plot_phase_portrait(indices=[1, 2])
            ode_model.plot_phase_portrait(indices=[0, 1, 2])

    if assess_limitcycle_extra:
        residual_norms = []
        times_final = []
        kk = 20
        print('Main (3): checking error of limit cycle prediction (subsampling every %d of n = %d pts)...' % (kk, len(times)) )
        for idx in range(len(times)):
            if idx % (kk) == 0:
                P_of_x, idx_P_history_state, idx_P_history_times = ode_model.poincare_map_at_x(r[idx, :], t0=times[idx])
                err = np.linalg.norm(r[idx, :] - P_of_x)
                residual_norms.append(err)
                times_final.append(idx_P_history_times[-1])
        plt.plot(np.arange(len(residual_norms)), residual_norms, label='residual_norm')
        plt.plot(np.arange(len(times_final)), times_final, label='idx_P_history_times[-1]')
        plt.legend()
        plt.title(r'$||x- P(x)||$ for each point on the presumed limit cycle')

        # TODO try plot the iterating limit cycles (starting from different points on the original claimed limit cycle) and compare them
        plt.xlabel('index')
        plt.ylabel('error')
        plt.show()

    # nonlinear transformation of ODE state variables (custom)
    if plot_custom_output_function:
        if ode_model.ode_base.ode_name == 'ode_custom_4':
            # nonlinear output used for model 4
            x1_high = np.max(r[:, 0])
            y_out = (x1_high - r[:, 0]) * r[:, 1]
            y_title = r'$y_{out} = (x_{1,high} - x_1) x_2$'
        else:
            # generic linear output -- take final variable of ODE as direct output
            y_out = r[:, -1]**2
            y_title = r'$y_{out} = x_{%d}(t)$' % ode_model.ode_base.dim_ode
        plt.plot(times, y_out)

        plt.title(y_title)
        plt.xlabel(r'$t$')
        plt.ylabel(r'output, $y(t)$')
        plt.show()
