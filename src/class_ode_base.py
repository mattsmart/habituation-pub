import copy
#import jax
#import jax.numpy as jnp
import numba
import numpy as np
import os
from scipy.optimize import check_grad, approx_fprime
from numba.core import types
from numba.typed import List, Dict

from defined_ode_fn import *
from defined_stimulus import VALID_FN_STIMULUS, stimulus_constant, stimulus_pulsewave, stimulus_pulsewave_of_pulsewaves, \
    stimulus_staircase, stitch_tspans_from_stimulus, stimulus_rectangle
from class_ode_stimulus import ODEStimulus
from settings import DIR_OUTPUT


VALID_FN_ODE = ['ode_quicktest', 'ode_quicktest_notpolynomial',
                'ode_iff_1', 'ode_custom_1',
                'ode_iff_1_6d', 'ode_custom_1_6d',
                'ode_custom_2', 'ode_custom_2_integrator',
                'ode_custom_3', 'ode_custom_3_integrator', 'ode_custom_3_integrator_TESTING',
                'ode_custom_3_simple', 'ode_custom_3_simple_integrator',
                'ode_custom_4', 'ode_custom_4_innerout', 'ode_custom_5',
                'ode_custom_6', 'ode_custom_6_simplified',
                'ode_custom_7', 'ode_custom_7_alt', 'ode_custom_7_ferrell',
                'ode_quad_2d_A',
                'ode_SatLin1D_NL',
                'ode_Wang93_1d', 'ode_Wang93_2d',
                'ode_SatLin2D_NL', 'ode_SatLin3D_NL',
                'ode_FHN',
                'ode_reverse_engineer_A',
                'ode_linear_filter_1d', 'ode_linear_filter_1d_lifted',
                'ode_linear_filter_2d',
                'ode_linear_filter_3d',
                'ode_linear_filter_4d',
                'ode_linear_filter_5d',
                'ode_simplefilter_to_tyson',
                'ode_circuit_diode',
                'ode_hallmark5', 'ode_hallmark5_lifted',
                'ode_hallmark8'
                ]

# TODO use default params and default init cond in presets / model
# TODO assert init cond and default params are as expected dimension
class ODEBase():
    """
    control_objects: list of instances of ODEStimulus class (i.e. stimulus fn and their parameters)
    """
    def __init__(
            self,
            ode_name,
            control_objects,
            params=None,
            label='ode_base_unlabelled'
            ):
        self.label = label
        self.ode_name = ode_name
        self.control_objects = control_objects
        self.params = params
        self.ode_long = None
        self.params_long = None
        self.output_fn = None  # function with signature: y(t) = g(x(t), u(t)):  def ode_output(state_traj, input_traj)
        self.output_str = None
        self.state_analytic = None   # list of [fn] which map t to x1, x2, ..., xN
        self.output_analytic = None  # list of [fn] which map t to y(t), i.e. analytic
        self.jac = None

        # init constructions -- control functions
        control_periods = []
        control_max_step_augments = []
        for cc in self.control_objects:
            if cc.period_forcing is not None:
                control_periods.append(cc.period_forcing)
            if cc.max_step_augment is not None:
                control_max_step_augments.append(cc.max_step_augment)

        unique_periods = set(control_periods)
        assert len(unique_periods) in [0, 1]
        # 0 means no periodicity e.g. constant stimulus
        # 1 means all have same periodicity
        # TODO not sure how to define forcing period if there are mixed periods... take the product probably
        # TODO for interlaced multiperiod function we might want both periods (T1 = time between delta spikes, T2 = ...)
        if len(unique_periods) == 0:
            self.period_forcing = None
        else:
            self.period_forcing = list(unique_periods)[0]
        if len(control_max_step_augments) == 0:
            self.max_step_augment = np.Inf
        else:
            self.max_step_augment = min(control_max_step_augments)

        # init constructions -- ode functions
        assert self.ode_name in VALID_FN_ODE  # TODO consider implementing ode fn here?

        if self.ode_name == 'ode_quicktest':

            @jit(nopython=True)
            def ode_quicktest(t, x, p, u):
                # compute stimulus over time t
                sval = u[0]
                a0, b0 = 1.0, 1.0
                # ======================================================
                dx0 = a0 * (1 - x[0]) * sval - b0 * x[0]
                # ======================================================
                return (dx0,)

            self.ode_fn = ode_quicktest
            self.dim_ode = 1
            self.dim_params = 0
            self.dim_control = 1
            self.ode_short = [r'$x_%d$' % (1+i) for i in range(self.dim_ode)]
            self.params_short = []
            self.params = np.array([])

            # custom output function
            local_dim = self.dim_ode
            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj):
                assert state_traj.shape[1] == local_dim
                w = state_traj[:, 0]
                u = input_traj[:]
                zout = u * (1 - np.tanh(w))
                return zout

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = g(x,u)$'

        elif self.ode_name == 'ode_quicktest_notpolynomial':

            @jit(nopython=True)
            def ode_quicktest_notpolynomial(t, x, p, u):
                # compute stimulus over time t
                sval = u[0]
                x0, y0 = x[0], x[1]

                a11, a12, a21, a22 = 0.9, 0, 0, 0.9  # need these < 1 for stability of origin, if uncoupled
                b1, b2 = 1, 1
                tau_1, tau_2 = 0.001, 0.1

                # compute vector field
                # ======================================================
                #dx0 = 1/tau_1 * (-x0 + np.tanh(a11*x0 + a12*y0 + b1*sval))
                #dx1 = 1/tau_2 * (-y0 + np.tanh(a21*x0 + a22*y0 + b2*sval))
                dx0 = 1 / tau_1 * (-x0 + np.tanh(a11 * x0 + a12 * y0 + b1 * sval))
                dx1 = 1 / tau_2 * (-y0 + np.tanh(a21 * x0 + a22 * y0 + b2 * sval))
                # ======================================================
                return dx0, dx1

            self.ode_fn = ode_quicktest_notpolynomial
            self.dim_ode = 2
            self.dim_params = 0
            self.dim_control = 1
            self.ode_short = [r'$x_%d$' % (1+i) for i in range(self.dim_ode)]
            self.params_short = []
            self.params = np.array([])

            # custom output function
            local_dim = self.dim_ode
            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj):
                assert state_traj.shape[1] == local_dim
                x1 = state_traj[:, 0]
                x2 = state_traj[:, 1]
                return (1 - x2) * x1

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = g(x,y)$'

        elif self.ode_name == 'ode_iff_1':
            self.ode_fn = ode_iff_1
            self.dim_ode = 3
            self.dim_params = 8
            self.dim_control = 1
            self.ode_short = [r"$x$", r"$y$", r"$z$"]
            self.ode_long = [r"$x=I/I_t$", r"$y=M/M_t$", r"$z=R/R_t$"]
            self.params_short = [
                r'$k_{x+}$', r'$k_{x-}$',
                r'$k_{y+}$', r'$k_{y-}$',
                r'$k_{z+}$', r'$k_{z-}$',
                r'$q_y$', r'$q_z$']
            self.params_long = [
                r'$k_x+=S_0 T k_{Ia}$', r'$k_x-=T k_{Ii}$',
                r'$k_y+=T I_t k_{Ma}$', r'$k_y-=\frac{T}{M_t} k_{Mi}$',
                r'$k_z+=T I_t k_{Ra} $', r'$k_z-=\frac{R_t T}{M_t} k_{Ri}$',
                r'$q_y=K_M/M_t$', r'$q_z=K_R/R_t$']
            if self.params is None:
                self.params = np.array([
                    1,  # k_x+
                    1,  # k_x-
                    1,  # k_y+
                    1,  # k_y-
                    1,  # k_z+
                    100,  # k_z-
                    1,  # q_y
                    0.5,  # q_z
                ])

        elif self.ode_name == 'ode_custom_1':
            self.ode_fn = ode_custom_1
            self.dim_ode = 3
            self.dim_params = 6
            self.dim_control = 1
            self.ode_short = [r"$x$", r"$y$", r"$z$"]
            self.params_short = [
                r'$k_{x+}$', r'$k_{x-}$',
                r'$k_{y+}$', r'$k_{y-}$',
                r'$k_{z+}$', r'$k_{z-}$']
            if self.params is None:
                self.params = np.array([
                    1,  # k_x+
                    1,  # k_x-
                    1,  # k_y+
                    0.1,  # k_y-
                    1,  # k_z+
                    100,  # k_z-
                ])

        elif self.ode_name == 'ode_iff_1_6d':
            self.ode_fn = ode_iff_1_6d
            self.dim_ode = 6
            self.dim_params = 16
            self.dim_control = 1
            self.ode_short = [r"$x$", r"$y$", r"$z$",
                              r"$x_b$", r"$y_b$", r"$z_b$"]
            self.params_short = [
                r'$k_{x+}$', r'$k_{x-}$',
                r'$k_{y+}$', r'$k_{y-}$',
                r'$k_{z+}$', r'$k_{z-}$',
                r'$q_y$', r'$q_z$',
                r'$k_{xb+}$', r'$k_{xb-}$',
                r'$k_{yb+}$', r'$k_{yb-}$',
                r'$k_{zb+}$', r'$k_{zb-}$',
                r'$q_{yb}$', r'$q_{zb}$']
            if self.params is None:
                self.params = np.array([
                    1,  # k_x+
                    1,  # k_x-
                    1,  # k_y+
                    1,  # k_y-
                    1,  # k_z+
                    100,  # k_z-
                    1,  # q_y
                    0.5,  # q_z

                    1,  # k_xb+
                    1,  # k_xb-
                    1,  # k_yb+
                    1,  # k_yb-
                    1,  # k_zb+
                    100,  # k_zb-
                    1,  # q_yb
                    0.5,  # q_zb
                ])

        elif self.ode_name == 'ode_custom_1_6d':
            self.ode_fn = ode_custom_1_6d
            self.dim_ode = 6
            self.dim_params = 12
            self.dim_control = 1
            self.ode_short = [r"$x$", r"$y$", r"$z$",
                              r"$x_b$", r"$y_b$", r"$z_b$"]
            self.params_short = [
                r'$k_{x+}$', r'$k_{x-}$',
                r'$k_{y+}$', r'$k_{y-}$',
                r'$k_{z+}$', r'$k_{z-}$',
                r'$k_{xb+}$', r'$k_{xb-}$',
                r'$k_{yb+}$', r'$k_{yb-}$',
                r'$k_{zb+}$', r'$k_{zb-}$',
            ]
            if self.params is None:
                self.params = np.array([
                    1,  # k_x+
                    1,  # k_x-
                    1,  # k_y+
                    0.1,  # k_y-
                    1,  # k_z+
                    100,  # k_z-

                    1,  # k_xb+
                    1,  # k_xb-
                    1,  # k_yb+
                    0.1,  # k_yb-
                    1,  # k_zb+
                    100,  # k_zb-
                ])

        elif self.ode_name == 'ode_custom_2':
            self.ode_fn = ode_custom_2
            self.dim_ode = 2
            self.dim_params = 4
            self.dim_control = 1
            self.ode_short = [r"$x$", r"$z$"]
            self.params_short = [
                r'$k_{x+}$', r'$k_{x-}$',
                r'$k_{z+}$', r'$k_{z-}$']
            if self.params is None:
                # habituate with square pulses:                        1, 0.1, 1, 100
                # habituate with staircase input (integral of pulses): 1, 1,   2,  2
                # TODO for [1, 0.1, 1, 10], and S1 input with duty 0.01, (dirac amp.),
                #  the ODE seems unstable, and violates our "weak habituation" -- seems to habituate normally
                self.params = np.array([
                    1,   #0.1,  # k_x+  # Tyson2003 Fig1d is 1, 1, 2, 2
                    0.1,   #10,  # k_x-
                    1,   #1,  # k_z+
                    10,   #10,  # k_z-
                ])

                local_dim = self.dim_ode

                @numba.jit(nopython=True)
                def ode_output(state_traj, input_traj):
                    assert state_traj.shape[1] == local_dim
                    return (10 + 1 - state_traj[:, 1]) * state_traj[:, 0]

                self.output_fn = ode_output
                self.output_str = r'$y_{out} = f(x_{1}, x_{2})$'

                if self.control_objects[0].stim_fn_name in ['stimulus_pulsewave', 'stimulus_pulsewave_of_pulsewaves']:
                    period_S1 = self.control_objects[0].params[2]
                    duty_S1 = self.control_objects[0].params[1]

                    # define the analytic output function for the system
                    # TODO also input create this for finite width rectangle pulse, see 'ode_custom_4_innerout' block
                    def state_analytic_mathematica(t):
                        a1 = self.params[0]
                        b1 = self.params[1]
                        a2 = self.params[2]
                        b2 = self.params[3]

                        x10 = 0  # TODO how to incorporate other init cond?
                        x20 = 0  # TODO how to incorporate? fn args?

                        z = np.exp(b1 * period_S1)
                        N_of_t = np.floor(t / period_S1).astype(int)  # note we are not taking the min vs. max number of pulses here... TODO
                        C = x10 * b2/b1
                        D = a1 * b2/b1 * z / (z-1)
                        mu2 = np.exp(-(D-C) * (1 - np.exp(-b1*t)) -D * N_of_t * (1/z - 1) - D * np.exp(-b1*t) * (z ** N_of_t) + D)

                        def eta(k):
                            # full expression (factor in the C-D from expression in text)
                            eta_val = (D-C) * (-1 + z ** (-k)) + D * (1-1/z) * k
                            return eta_val

                        def map_nval_to_sum_exp_eta(nval):
                            ss = 0
                            for k in range(1, nval+1):
                                eta_val = eta(k)
                                ss += np.exp(eta_val)
                            return ss

                        premapped_sum_exp_eta = {k: map_nval_to_sum_exp_eta(k) for k in range(np.max(N_of_t) + 1)}

                        sum_exp_eta_vec = np.array([premapped_sum_exp_eta[i] for i in N_of_t])

                        x1 = np.exp(-b1 * t) * (x10 + a1 * z * (1 - z ** N_of_t) / (1 - z))
                        # x2: full expression
                        x2 = 1 / mu2 * (x20 + a2 * sum_exp_eta_vec)

                        #log_sum_exp_eta_vec = np.log(sum_exp_eta_vec)  # TODO try to stabilize this calculation??
                        # TODO mathematica rep of x2 - use log for numeric stability?
                        # 1) for a given time, get N(t)
                        # 2) get sum of eta from k = 1 to N -- eta_sum(t)
                        # 3) take log: log_eta_sum
                        # 4) x2 = 1/mu2 * a2 * e^(C-D) * eta_sum
                        #    rep as:
                        #       log x2 = ln a2 + C-D - mu2_exp + log_eta_sum
                        # 5) x2 = e^(log x2)

                        return x1, x2

                    def output_analytic_mathematica(t):
                        _, x2 = state_analytic_mathematica(t)
                        return x2

                    self.state_analytic = [state_analytic_mathematica]
                    self.output_analytic = [output_analytic_mathematica]

        elif self.ode_name == 'ode_custom_2_integrator':
            self.ode_fn = ode_custom_2_integrator
            self.dim_ode = 3
            self.dim_params = 4
            self.dim_control = 1
            self.ode_short = [r"$\int u dt$", r"$x$", r"$z$"]
            self.params_short = [
                r'$k_{x+}$', r'$k_{x-}$',
                r'$k_{z+}$', r'$k_{z-}$']
            if self.params is None:
                self.params = np.array([
                    2,  # k_x+
                    2,  # k_x-
                    1,  # k_z+
                    10,  # k_z-
                ])

        elif self.ode_name == 'ode_custom_3':
            self.ode_fn = ode_custom_3
            self.dim_ode = 2
            self.dim_params = 3
            self.dim_control = 1
            self.ode_short = [r"$x$", r"$z$"]
            self.params_short = [r'$k_{z+}$', r'$k_{z-}$', r'x_{tot}']
            if self.params is None:
                self.params = np.array([
                    4, #1,  # k_z+
                    4, #0.1,  # k_z-
                    1,  # (default: 1) x_tot
                ])

            local_dim = self.dim_ode

            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj):
                assert state_traj.shape[1] == local_dim
                return state_traj[:, 1] ** 2

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = [x_{1}]^2$'

        elif self.ode_name == 'ode_custom_3_integrator':
            self.ode_fn = ode_custom_3_integrator
            self.dim_ode = 3
            self.dim_params = 3
            self.dim_control = 1
            self.ode_short = [r"$\int u dt$", r"$x$", r"$z$"]
            self.params_short = [r'$k_{z+}$', r'$k_{z-}$', 'r$x_{tot}$']
            if self.params is None:
                # H1: 10, 100, 20, 0
                self.params = np.array([
                    10,   # k_z+
                    100,  # k_z- and k_x+
                    20,   # (default: 1) x_tot
                ])

        elif self.ode_name == 'ode_custom_3_integrator_TESTING':
            self.ode_fn = ode_custom_3_integrator_TESTING
            self.dim_ode = 3
            self.dim_params = 4
            self.dim_control = 1
            self.ode_short = [r"$\int u dt$", r"$x$", r"$z$"]
            self.params_short = [r'$k_{z+}$', r'$k_{z-}$', 'r$x_{tot}$', 'r$k_{S-}$']
            if self.params is None:
                # H1: 10, 100, 20, 0.0
                self.params = np.array([
                    1,  # k_z+
                    1,  # k_z- and k_x+
                    1,  # (default: 1) x_tot
                    0.1,  # (default: 0) decay of built up stimulus
                ])

        elif self.ode_name == 'ode_custom_3_simple':
            self.ode_fn = ode_custom_3_simple
            self.dim_ode = 2
            self.dim_params = 3
            self.dim_control = 1
            self.ode_short = [r"$x$", r"$z$"]
            self.params_short = [r'$k_{z+}$', r'$k_{z-}$', 'r$x_{tot}$']
            if self.params is None:
                self.params = np.array([
                    1,
                    1,
                    1,  # (default: 1) x_tot
                ])

        elif self.ode_name == 'ode_custom_3_simple_integrator':
            self.ode_fn = ode_custom_3_simple_integrator
            self.dim_ode = 3
            self.dim_params = 3
            self.dim_control = 1
            self.ode_short = [r"$\int u dt$", r"$x$", r"$z$"]
            self.params_short = [r'$k_{z+}$', r'$k_{z-}$', 'r$x_{tot}$']
            if self.params is None:
                self.params = np.array([
                    10,
                    100,
                    20,  # (default: 1) x_tot
                ])

        elif self.ode_name == 'ode_custom_4':
            self.ode_fn = ode_custom_4
            self.dim_ode = 2
            self.dim_params = 6
            self.dim_control = 1
            self.ode_short = [r"$x_1$", r"$x_2$"]
            self.params_short = [
                r'$k_{x1+}$', r'$k_{x1-}$',
                r'$k_{x2+}$', r'$k_{x2-}$',
                r'$x_{1,high}$', r'$\alpha$',
            ]
            if self.params is None:
                self.params = np.array([
                    0.1,  # k_x1+ = K_1 / T_1
                    0.1,  # k_x1- =   1 / T_1
                    0.5,  # k_x2+ = K_2 / T_2
                    5.0,  # k_x2- =   1 / T_2
                    0.0,  # x1_high; will be filled in automatically
                    1.5,  # alpha = multiplier (>=1.0) for x1_high in output equation
                ])

            # custom output function for case of dirac comb input
            if self.control_objects[0].stim_fn_name in ['stimulus_pulsewave', 'stimulus_pulsewave_of_pulsewaves']:
                period_S1 = self.control_objects[0].params[2]

                x1_high = self.params[0] / (1 - np.exp(- self.params[1] * period_S1))  # see martin overleaf Eq. 9
                self.params[4] = x1_high

                local_dim = self.dim_ode
                local_params = self.params

                @numba.jit(nopython=True)
                def ode_output(state_traj, input_traj):
                    assert state_traj.shape[1] == local_dim
                    x1 = state_traj[:, 0]
                    x2 = state_traj[:, 1]
                    return (local_params[4] * local_params[5] - x1) * x2

                self.output_fn = ode_output
                self.output_str = r'$y_{out} = (x_{1,high} - x_1) x_2$'

        elif self.ode_name == 'ode_custom_4_innerout':
            self.ode_fn = ode_custom_4_innerout
            self.dim_ode = 3
            self.dim_params = 7
            self.dim_control = 1
            self.ode_short = [r"$x_1$", r"$x_2$", r"$x_3$"]
            self.params_short = [
                r'$k_{x1+}$',    r'$k_{x1-}$',
                r'$k_{x2+}$',    r'$k_{x2-}$',
                r'$x_{1,high}$', r'$\alpha$',
                r'$1/\epsilon$', ]
            if self.params is None:
                self.params = np.array([
                    0.1,  # k_x1+ = K_1 / T_1
                    0.1,  # k_x1- =   1 / T_1
                    0.5,  # k_x2+ = K_2 / T_2
                    5.0,  # k_x2- =   1 / T_2
                    0.0,  # x1_high; will be filled in automatically
                    1.5,  # alpha = multiplier (>=1.0) for x1_high in output equation
                    1e6,  # 1/epsilon, which is 1/T3 in Martin notation, want the RHS big unless y(t) = f(x1, x2)
                ])
            assert self.params[5] >= 1.0
            # custom output function for case of dirac comb input
            if self.control_objects[0].stim_fn_name in ['stimulus_pulsewave', 'stimulus_pulsewave_of_pulsewaves']:
                period_S1 = self.control_objects[0].params[2]
                duty_S1 = self.control_objects[0].params[1]
                x1_high = self.params[0] / (1 - np.exp(- self.params[1] * period_S1))  # see martin overleaf Eq. 9
                self.params[4] = x1_high

                # define the analytic output function for the system
                # TODO for ANY linear system with init cond, fixed params, we can just functionalize this dirac comb output module somewhere else and call it
                def output_analytic_dirac(t):
                    a1 = self.params[0]
                    b1 = self.params[1]
                    a2 = self.params[2]
                    b2 = self.params[3]
                    x1_high = self.params[4]
                    x1_high_mult = self.params[5]
                    inv_eps = self.params[6]

                    out_const = x1_high_mult * x1_high
                    print("x1_high", x1_high, a1 / (1 - np.exp(-b1 * period_S1)), "out_const", out_const)

                    c1 = np.exp(b1 * period_S1)
                    c2 = np.exp(b2 * period_S1)
                    N_of_t = np.floor(t/period_S1)  # note we are not taking the min vs. max number of pulses here... TODO

                    last_factor = x1_high_mult - np.exp(-b1 * t) * (c1 ** N_of_t - 1)
                    print('TODO fix output_analytic_dirac() after out_const change')
                    return a1 / (1 - 1/c1) * a2 / (1 - 1/c2) * np.exp(-b2 * t) * (c2 ** N_of_t - 1) * last_factor

                # define the analytic output function for the system
                def output_analytic_rectangle(t):
                    a1 = self.params[0]
                    b1 = self.params[1]
                    a2 = self.params[2]
                    b2 = self.params[3]
                    x1_high = self.params[4]
                    x1_high_mult = self.params[5]
                    inv_eps = self.params[6]

                    out_const = x1_high_mult * x1_high
                    print("x1_high", x1_high, a1 / (1-np.exp(-b1 * period_S1)), "out_const", out_const)

                    c1 = np.exp(b1 * period_S1)
                    c2 = np.exp(b2 * period_S1)
                    N_of_t = np.floor(t/period_S1)
                    d = duty_S1

                    A = 1/(d * period_S1)

                    I1a = (1 - np.exp(-d * b1 * period_S1)) * c1 * (c1 ** (N_of_t) - 1) / (c1 - 1)
                    I2a = (1 - np.exp(-d * b2 * period_S1)) * c2 * (c2 ** (N_of_t) - 1) / (c2 - 1)

                    tmod = t / period_S1 % 1                # float 0 <= tmod <= 1
                    cases = np.where(tmod < (1 - d), 0, 1)  # use the "duty" parameter to return 0 or amp
                    I1b = np.exp(b1 * t) - c1 ** (N_of_t + 1 - d)
                    I2b = np.exp(b2 * t) - c2 ** (N_of_t + 1 - d)

                    x1 = np.exp(-b1 * t) * (a1) * (A / b1) * (I1a + cases * I1b)
                    x2 = np.exp(-b2 * t) * (a2) * (A / b2) * (I2a + cases * I2b)

                    return (out_const - x1) * x2
                    #return (x1_high - x1) * x2

                self.output_analytic = [output_analytic_dirac, output_analytic_rectangle]

            else:
                print('Warning: ode_custom_4_internalizedout designed for controls: stimulus_pulsewave, '
                      'stimulus_pulsewave_of_pulsewaves - parameter #6 will be set to 0')

        elif self.ode_name == 'ode_custom_5':
            self.ode_fn = ode_custom_5
            self.dim_ode = 3
            self.dim_params = 6
            self.dim_control = 1
            self.ode_short = [r"$x$", r"$y$", r"$z$"]
            self.params_short = [
                r'$k_{x+}$', r'$k_{x-}$',
                r'$k_{y+}$', r'$k_{y-}$',
                r'$k_{z+}$', r'$k_{z-}$']
            if self.params is None:
                self.params = np.array([
                    1,  # k_x+
                    1,  # k_x-
                    1,  # k_y+
                    0.1,  # k_y-
                    1,  # k_z+
                    100,  # k_z-
                ])

        elif self.ode_name == 'ode_custom_6':
            self.ode_fn = ode_custom_6
            self.dim_ode = 2
            self.dim_params = 6
            self.dim_control = 1
            self.ode_short = [r"$A$", r"$B$"]
            self.params_short = [
                r'$k_{x1+}$', r'$k_{x1-}$',
                r'$k_{x2+}$', r'$k_{x2-}$',
                r'$Q_{+}$', r'$Q_{-}$']
            if self.params is None:
                self.params = np.array([
                    0.1,  # k_x+
                    10,  # k_x-
                    10,  # k_y+
                    1.,  # k_y-
                    10,  # q_+
                    1,  # q_-
                ])

        elif self.ode_name == 'ode_custom_6_simplified':
            self.ode_fn = ode_custom_6_simplified
            self.dim_ode = 2
            self.dim_params = 4
            self.dim_control = 1
            self.ode_short = [r"$A$", r"$B$"]
            self.params_short = [
                r'$k_{x1+}$', r'$k_{x1-}$',
                r'$k_{x2+}$', r'$k_{x2-}$']
            if self.params is None:
                self.params = np.array([
                    10,  # k_x+
                    1,  # k_x-
                    10,  # k_y+
                    0.1,  # k_y-
                ])
        elif self.ode_name == 'ode_custom_7':
            self.ode_fn = ode_custom_7
            self.dim_ode = 3
            self.dim_params = 5
            self.dim_control = 1
            self.ode_short = [r"$Z_1$", r"$Z_2$", r"$y$"]
            self.params_short = [
                r'$\mu$', r'$\eta$', r'$\theta$',
                r'$k$', r'$\gamma$']
            if self.params is None:
                """
                self.params = np.array([
                    1,  # a1
                    1,  # a2
                    1,  # k_0
                    10,  # a3
                    1,   # b3
                ])
                """
                self.params = np.array([
                    1,  # a1
                    1,  # a2
                    10,  # k_0
                    1,  # a3
                    1,  # b3
                ])
        elif self.ode_name == 'ode_custom_7_alt':
            self.ode_fn = ode_custom_7_alt
            self.dim_ode = 4
            self.dim_params = 7
            self.dim_control = 1
            self.ode_short = [r"$Z_1$", r"$Z_2$", r"$x$", r"$y$"]
            self.params_short = [
                r'$\mu$', r'$\eta$', r'$\theta$',
                r'$k$', r'$\gamma$',
                r'$x_-$', r'$x_-$']
            if self.params is None:
                self.params = np.array([
                    1,  # a1
                    1,  # a2
                    1,  # k_0
                    10,  # a3
                    1,  # b3
                    1,  # x_+
                    1,  # x_-
                ])
        elif self.ode_name == 'ode_custom_7_ferrell':
            self.ode_fn = ode_custom_7_ferrell
            self.dim_ode = 4
            self.dim_params = 7
            self.dim_control = 1
            self.ode_short = [r"$Z_1$", r"$Z_2$", r"$x$", r"$y$"]
            self.params_short = [
                r'$\mu$', r'$\eta$', r'$\theta$',
                r'$k$', r'$\gamma$',
                r'$x_-$', r'$x_-$']
            if self.params is None:
                self.params = np.array([
                    1,  # a1
                    1,  # a2
                    1,  # k_0
                    10,  # a3
                    1,  # b3
                    1,  # x_+
                    1,  # x_-
                ])
        elif self.ode_name == 'ode_quad_2d_A':
            self.ode_fn = ode_quad_2d_A
            self.dim_ode = 2
            self.dim_params = 6
            self.dim_control = 1
            self.ode_short = [r'$x_%d$' % (1 + i) for i in range(self.dim_ode)]
            self.params_short = [r'$a_0$', r'$b_0$', r'$a_1$', r'$b_2$', r'$a_{12}$', r'$b_{12}$']
            if self.params is None:
                '''
                self.params = np.array([
                    1.0,   # a0 = 1 * sval
                    1.0,   # b0 = 1 * sval
                    -0.1,  # a1 = -0.1
                    -1.0,  # b2 = -1
                    -1,    # a12 = -1
                    -10,   # b12 = -10
                ])'''
                self.params = np.array([
                    10.0,   # a0 = 1 * sval
                    1.0,   # b0 = 1 * sval
                    -0.01,  # a1 = -0.1
                    0,#-10,  # b2 = -1
                    0,#-1,    # a12 = -1
                    -10,   # b12 = -10
                ])
        elif self.ode_name == 'ode_SatLin1D_NL':
            self.ode_fn = ode_SatLin1D_NL
            self.dim_ode = 1
            self.dim_params = 2
            self.dim_control = 1
            self.ode_short = [r'$x_2$']
            self.params_short = [r'$a_0$', r'$b_0$']
            if self.params is None:
                self.params = np.array([
                    1.0,   # a0 = rate of transition to (1-x) state (only occurs in presence of u(t))
                    0.1,   # b0 = rate of transition to x from (1-x) state
                ])

            # custom output function
            local_dim = self.dim_ode

            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj):
                assert state_traj.shape[1] == local_dim
                x1 = state_traj[:, 0]
                return (1 - x1) * input_traj

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = (1 - x) u(t)$'

            # custom output function for case of dirac comb input
            if self.control_objects[0].stim_fn_name in ['stimulus_pulsewave', 'stimulus_pulsewave_of_pulsewaves']:
                period_S1 = self.control_objects[0].params[2]
                k12, k2 = self.params

                def output_analytic_dirac_eq31geo(t):
                    """
                    geoemtric series version of overleaf Eq. 31 (which assumes 0 init cond)
                    """
                    k12, k2 = self.params  # production, degradation

                    N_of_t = np.floor(t / period_S1).astype(int)
                    z = np.exp(k12 + k2 * period_S1)

                    prefactor = np.exp(- k2 * t - k12 * N_of_t) * k12
                    outvec = prefactor * z * (1 - z ** N_of_t) / (1 - z)

                    return outvec

                self.output_analytic = [output_analytic_dirac_eq31geo]
        elif self.ode_name == 'ode_Wang93_1d':
            self.ode_fn = ode_Wang93_1d
            self.dim_ode = 1
            self.dim_params = 3
            self.dim_control = 1
            self.ode_short = [r'$y$']
            self.params_short = [r'$\tau$', r'$\alpha$', r'$y_0$']
            if self.params is None:
                self.params = np.array([
                    1.0,  # tau: control timescale separation for dy/dt vs. dz/dt
                    1.0,  # alpha: production rate for dy/dt
                    1.0,  # y_0: FP for y(t) when no input
                ])
        elif self.ode_name == 'ode_Wang93_2d':
            self.ode_fn = ode_Wang93_2d
            self.dim_ode = 2
            self.dim_params = 5
            self.dim_control = 1
            self.ode_short = [r'$y$', r'$z$']
            self.params_short = [r'$\tau$', r'$\alpha$', r'$\beta$', r'$\gamma$', r'$y_0$']
            if self.params is None:
                self.params = np.array([
                    1.0,  # tau: control timescale separation for dy/dt vs. dz/dt
                    1.0,  # alpha: production rate for dy/dt
                    1.0,  # beta: degradation rate for dy/dt
                    1.0,  # gamma: production rate for z(t)
                    1.0,  # y_0: FP for y(t) when no input
                ])
        elif self.ode_name == 'ode_SatLin2D_NL':
            self.ode_fn = ode_SatLin2D_NL
            self.dim_ode = 2
            self.dim_params = 4
            self.dim_control = 1
            self.ode_short = [r'$x_1$', r'$x_2$']
            self.params_short = [r'$\tau_1$', r'$\tau_2$', r'$a_1$', r'$a_2$']
            if self.params is None:
                self.params = np.array([
                    2.0,   # tau_1: timescale for x_1
                    0.2,   # tau_2: timescale for x_2
                    5.0,  # a_1: production rate for dx1/dt
                    0.4,  # a_2:   production rate for dx2/dt
                ])

            # custom output function
            local_dim = self.dim_ode
            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj):
                assert state_traj.shape[1] == local_dim
                x1 = state_traj[:, 0]
                x2 = state_traj[:, 1]
                return (1 - x1) * x2
            self.output_fn = ode_output
            self.output_str = r'$y_{out} = (1 - x_1) x_2$'

            if self.control_objects[0].stim_fn_name in ['stimulus_pulsewave', 'stimulus_pulsewave_of_pulsewaves']:
                period_S1 = self.control_objects[0].params[2]
                duty_S1 = self.control_objects[0].params[1]
                amp_s1 = self.control_objects[0].params[0]

                # define the analytic output function for the system
                # TODO also input create this for finite width rectangle pulse, see 'ode_custom_4_innerout' block
                def state_analytic_rectangle(t):
                    # TODO make this collable; note that rectangle and dirac methods are both implemented above for custom ode 4 (linear case)

                    T1 = self.params[0]
                    T2 = self.params[1]
                    K1 = self.params[2]
                    K2 = self.params[3]

                    x2_init = 0 # how to do init cond?

                    a1 = K1/T1
                    b1 = 1/T1
                    a2 = K2/T2
                    b2 = 1/T2
                    c2 = np.exp(b2 * period_S1)
                    d = duty_S1
                    A = amp_s1
                    N_of_t = np.floor(t / period_S1)

                    I2a = (1 - c2 ** (-d)) * c2 * (c2 ** (N_of_t) - 1) / (c2 - 1)

                    tmod = t / period_S1 % 1                # float 0 <= tmod <= 1
                    cases = np.where(tmod < (1 - d), 0, 1)  # 0 if time is outside duty "1-d" window | 1 otherwise
                    I2b = np.exp(b2 * t) - c2 ** (N_of_t + 1 - d)

                    # look at supp_catalog.tex Eq. 14
                    x1 = np.zeros_like(t)  # TODO how to solve x1?
                    x2 = np.exp(-b2 * t) * (x2_init + A * (a2 / b2) * (I2a + cases * I2b))  # TODO why mult a2?


                    return x1, x2

                self.state_analytic = [state_analytic_rectangle]

        elif self.ode_name == 'ode_SatLin3D_NL':
            self.ode_fn = ode_SatLin3D_NL
            self.dim_ode = 3
            self.dim_params = 6
            self.dim_control = 1
            self.ode_short = [r'$x_1$', r'$x_2$', r'$x_3$']
            self.params_short = [r'$\tau_1$', r'$\tau_2$', r'$\tau_3$', r'$a_1$', r'$a_2$', r'$a_3$']
            if self.params is None:
                self.params = np.array([
                    2.0,   # tau_1: timescale for x_1
                    0.2,   # tau_2: timescale for x_2
                    50.0,  # tau_3: timescale for x_3
                    5.0,  # a_1:   production rate for dx1/dt
                    0.4,  # a_2:   production rate for dx2/dt
                    5.0,  # a_3:   production rate for dx3/dt
                ])

            # custom output function
            local_dim = self.dim_ode
            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj):
                assert state_traj.shape[1] == local_dim
                x1 = state_traj[:, 0]
                x2 = state_traj[:, 1]
                x3 = state_traj[:, 2]
                return (1 - x1) * (1 - x3) * x2

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = (1 - x_1) (1 - x_3) x_2$'

        elif self.ode_name == 'ode_FHN':
            self.ode_fn = ode_FHN
            self.dim_ode = 2
            self.dim_params = 3
            self.dim_control = 1
            self.ode_short = [r'$x$', r'$y$']
            self.params_short = [r'$\epsilon$', r'$\gamma$', r'$\alpha$']
            if self.params is None:
                self.params = np.array([
                    0.01,   # epsilon: timescale for x
                    0.2,   # gamma: slope of y nullcline (0 < gamma < 1/m; generally m=2)
                    1.0,  # alpha: magnitude of pulse (z)
                ])

            # custom output function
            local_dim = self.dim_ode
            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj):
                assert state_traj.shape[1] == local_dim
                x = state_traj[:, 0]
                y = state_traj[:, 1]
                return (2 - y) * x

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = g(x, y)$'

        elif self.ode_name == 'ode_reverse_engineer_A':
            self.ode_fn = ode_reverse_engineer_A
            self.dim_ode = 1
            self.dim_params = 1
            self.dim_control = 1
            self.ode_short = [r'$W$']
            self.params_short = [r'$a$']
            if self.params is None:
                self.params = np.array([
                    0.05,   # a: timescale for x dynamics)
                ])

            # custom output function
            local_dim = self.dim_ode
            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj):
                assert state_traj.shape[1] == local_dim
                x = state_traj[:, 0]
                u = input_traj[:]
                return u * (1 - np.tanh(x))

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = g(x, u)$'

        elif self.ode_name == 'ode_linear_filter_1d':
            self.ode_fn = ode_linear_filter_1d
            self.dim_ode = 1
            self.dim_params = 2
            self.dim_control = 1
            self.ode_short = [r'$x$']
            self.params_short = [r'$\alpha$', r'$\beta$']
            if self.params is None:
                self.params = np.array([
                    0.5,   # alpha: timescale for x decay -- prop to x(t)
                    0.1,   # beta: timescale for x growth -- prop to u(t)
                ])

            # custom output function
            local_dim = self.dim_ode

            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj, hill_N=2):
                assert state_traj.shape[1] == local_dim
                x = state_traj[:, 0]
                u = input_traj[:]
                return u / (1 + x ** hill_N)

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = u / (1 + x^N)$'

            if self.control_objects[0].stim_fn_name == 'stimulus_pulsewave':
                period_S1 = self.control_objects[0].params[2]
                duty_S1 = self.control_objects[0].params[1]

                # TODO also input create this for finite width rectangle pulse, see 'ode_custom_4_innerout' block
                def state_analytic_geo(t, x0=0.0):
                    # x0 = 0  # TODO how to incorporate other init cond?
                    a = self.params[0]
                    b = self.params[1]

                    t0 = t[0]
                    t_above_zero = np.where(t<0, 0, t)  # this hack assumes that we only apply pulses after t=0

                    z = np.exp(a * period_S1)
                    N_of_t = np.floor(t_above_zero / period_S1).astype(
                        int)  # note we are not taking the min vs. max number of pulses here... TODO
                    #x1 = np.exp(-a * (t - t0)) * (x0 + b * z * (1 - z ** N_of_t) / (1 - z))
                    x1 = np.exp(-a * (t - t0)) * x0 + b * np.exp(-a * t) * z * (1 - z ** N_of_t) / (1 - z)

                    return (x1,)

                self.state_analytic = [state_analytic_geo]

            elif self.control_objects[0].stim_fn_name == 'stimulus_pulsewave_of_pulsewaves':
                ### self.params_short = [r'$A$', r'$d$', r'$T$', r'$n_p$', r'$n_r$']
                period_S1 = self.control_objects[0].params[2]
                duty_S1 = self.control_objects[0].params[1]
                stim_np =   self.control_objects[0].params[3]
                stim_nr = self.control_objects[0].params[4]

                # TODO also input create this for finite width rectangle pulse, see 'ode_custom_4_innerout' block
                def state_analytic_geo_with_rest(t, x0=0.0):
                    a = self.params[0]
                    b = self.params[1]

                    t0 = t[0]
                    t_above_zero = np.where(t < 0, 0, t)  # this hack assumes that we only apply pulses after t=0

                    z = np.exp(a * period_S1)
                    Z = z ** (stim_np + stim_nr)

                    M_of_t = np.floor(t_above_zero / (period_S1 * (stim_np + stim_nr))).astype(
                        int)
                    N_of_t = np.floor((t_above_zero - period_S1 * M_of_t * (stim_np + stim_nr)) / period_S1).astype(
                        int)
                    N_of_t_cases = np.minimum(N_of_t, stim_np)

                    sum_N_factor = z * (z ** N_of_t_cases - 1) / (z - 1)
                    sum_Lfull_factor = z * (z ** stim_np - 1) / (z - 1)
                    sum_M_factor = (Z ** (M_of_t + 1) - 1) / (Z - 1)

                    full_sum_S_of_t = sum_M_factor * sum_Lfull_factor - Z ** (M_of_t) * (sum_Lfull_factor - sum_N_factor)

                    #x1 = np.exp(-a * (t - t0)) * x0 + b * np.exp(-a * t) * z * (z ** N_of_t_cases - 1) / (z - 1) * (Z ** (M_of_t + 1) - 1) / (Z - 1)
                    x1 = np.exp(-a * (t - t0)) * x0 + b * np.exp(-a * t) * full_sum_S_of_t

                    return (x1,)

                self.state_analytic = [state_analytic_geo_with_rest]

        elif self.ode_name == 'ode_linear_filter_1d_lifted':
            self.ode_fn = ode_linear_filter_1d_lifted
            self.dim_ode = 2
            self.dim_params = 4
            self.dim_control = 1
            self.ode_short = [r'$x_1$', r'$x_2$']
            self.params_short = [r'$\alpha$', r'$\beta$', r'$N$', r'$\epsilon$']
            if self.params is None:
                self.params = np.array([
                    0.1,   # alpha: timescale for x decay -- prop to x(t)
                    0.5,   # beta: timescale for x growth -- prop to u(t)
                    2,     # N: hill function, filter saturation
                    1e-2,  # epsilon: timescale for output target synchronization (should be fastest timescale)
                ])

        elif self.ode_name == 'ode_linear_filter_2d':
            self.ode_fn = ode_linear_filter_2d
            self.dim_ode = 2
            self.dim_params = 5
            self.dim_control = 1
            self.ode_short = [r'$x_1$', r'$x_2$']
            self.params_short = [r'$\alpha_0$', r'$\beta_0$', r'$N_0$',
                                 r'$\alpha_1$', r'$\beta_1$']
            if self.params is None:
                self.params = np.array([
                    0.1,  # x0: alpha: timescale for x decay -- prop to x(t)
                    0.5,  # x0: beta:  timescale for x growth -- prop to u(t)
                    2,    # N_0 - hill coeff for x0
                    0.1,  # x1: alpha
                    0.5,  # x1: beta
                ])

            # custom output function
            local_dim = self.dim_ode

            '''
            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj, hill_N=2):
                assert state_traj.shape[1] == local_dim
                x = state_traj[:, -1]
                u = input_traj[:]
                return u / (1 + x ** hill_N)'''

            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj, hill_N=2):
                assert state_traj.shape[1] == local_dim
                x_k = state_traj[:, -1]
                x_kM1 = state_traj[:, -2]
                u = input_traj[:]
                y_k = u / (1 + x_kM1 ** hill_N)
                return y_k / (1 + x_k ** hill_N)

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = u / (1 + x_{k}^N)$'

        elif self.ode_name == 'ode_linear_filter_3d':
            self.ode_fn = ode_linear_filter_3d
            self.dim_ode = 3
            self.dim_params = 8
            self.dim_control = 1
            self.ode_short = [r'$x_1$', r'$x_2$', r'$x_3$']
            self.params_short = [r'$\alpha_0$', r'$\beta_0$', r'$N_0$',
                                 r'$\alpha_1$', r'$\beta_1$', r'$N_1$',
                                 r'$\alpha_2$', r'$\beta_2$']
            if self.params is None:
                self.params = np.array([
                    0.5,  # x0: alpha: timescale for x decay -- prop to x(t)
                    0.1,  # x0: beta:  timescale for x growth -- prop to u(t)
                    2,    # N_0 - hill coeff for x0
                    0.25,  # x1: alpha
                    0.3,  # x1: beta
                    2,    # N_1 - hill coeff for x1
                    0.1,  # x2: alpha
                    0.5,  # x2: beta
                ])

            # custom output function
            local_dim = self.dim_ode

            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj, hill_N=2):
                assert state_traj.shape[1] == local_dim
                x = state_traj[:, -1]
                u = input_traj[:]
                return u / (1 + x ** hill_N)

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = u / (1 + x_{k}^N)$'

        elif self.ode_name == 'ode_linear_filter_4d':
            self.ode_fn = ode_linear_filter_4d
            self.dim_ode = 4
            self.dim_params = 11
            self.dim_control = 1
            self.ode_short = [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$']
            self.params_short = [r'$\alpha_0$', r'$\beta_0$', r'$N_0$',
                                 r'$\alpha_1$', r'$\beta_1$', r'$N_1$',
                                 r'$\alpha_2$', r'$\beta_2$', r'$N_2$',
                                 r'$\alpha_3$', r'$\beta_3$']
            if self.params is None:
                self.params = np.array([
                    0.1,  # x0: alpha: timescale for x decay -- prop to x(t)
                    0.5,  # x0: beta:  timescale for x growth -- prop to u(t)
                    2,    # N_0 - hill coeff for x0
                    0.1,  # x1: alpha
                    0.5,  # x1: beta
                    2,    # N_1 - hill coeff for x1
                    0.1,  # x2: alpha
                    0.5,  # x2: beta
                    2,    # N_2 - hill coeff for x2
                    0.1,  # x3: alpha
                    0.5,  # x3: beta
                ])

            # custom output function
            local_dim = self.dim_ode

            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj, hill_N=2):
                assert state_traj.shape[1] == local_dim
                x = state_traj[:, -1]
                u = input_traj[:]
                return u / (1 + x ** hill_N)

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = u / (1 + x_{k}^N)$'

        elif self.ode_name == 'ode_linear_filter_5d':
            self.ode_fn = ode_linear_filter_5d
            self.dim_ode = 5
            self.dim_params = 14
            self.dim_control = 1
            self.ode_short = [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$', r'$x_5$']
            self.params_short = [r'$\alpha_0$', r'$\beta_0$', r'$N_0$',
                                 r'$\alpha_1$', r'$\beta_1$', r'$N_1$',
                                 r'$\alpha_2$', r'$\beta_2$', r'$N_2$',
                                 r'$\alpha_3$', r'$\beta_3$', r'$N_3$',
                                 r'$\alpha_4$', r'$\beta_4$']
            if self.params is None:
                self.params = np.array([
                    0.1,  # x0: alpha: timescale for x decay -- prop to x(t)
                    0.5,  # x0: beta:  timescale for x growth -- prop to u(t)
                    2,    # N_0 - hill coeff for x0
                    0.1,  # x1: alpha
                    0.5,  # x1: beta
                    2,    # N_1 - hill coeff for x1
                    0.1,  # x2: alpha
                    0.5,  # x2: beta
                    2,    # N_2 - hill coeff for x2
                    0.1,  # x3: alpha
                    0.5,  # x3: beta
                    2,    # N_3 - hill coeff for x3
                    0.1,  # x4: alpha
                    0.5,  # x4: beta
                ])

            # custom output function
            local_dim = self.dim_ode

            '''
            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj, hill_N=2):
                assert state_traj.shape[1] == local_dim
                x = state_traj[:, -1]
                u = input_traj[:]
                return u / (1 + x ** hill_N)'''

            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj, hill_N=2):
                assert state_traj.shape[1] == local_dim
                x_k = state_traj[:, -1]
                x_kM1 = state_traj[:, -2]
                u = input_traj[:]
                y_k = u / (1 + x_kM1 ** hill_N)
                return y_k / (1 + x_k ** hill_N)

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = u / (1 + x_{k}^N)$'

        elif self.ode_name == 'ode_simplefilter_to_tyson':
            self.ode_fn = ode_simplefilter_to_tyson
            self.dim_ode = 2
            self.dim_params = 5
            self.dim_control = 1
            self.ode_short = [r"$\alpha$", r"$x$"]
            self.params_short = [
                r'$b_1$', r'$\lambda_1$', r'$\alpha^*$',
                r'$b_2$', r'$\lambda_2$']
            if self.params is None:
                # habituate with square pulses:                        1, 0.1, 1, 100
                # habituate with staircase input (integral of pulses): 1, 1,   2,  2
                # TODO for [1, 0.1, 1, 10], and S1 input with duty 0.01, (dirac amp.),
                #  the ODE seems unstable, and violates our "weak habituation" -- seems to habituate normally
                self.params = np.array([
                    1,     # 0.1  r'$b_1$', r'$\lambda_1$', r'$\alpha^*$',
                    0.1,       # 1,
                    0.1,     # 0,
                    0.5,     # 1,   r'$b_2$',
                    1,       # 1,   r'$\lambda_2$',  (this is a dummy parameter, fix at 1)
                ])

            local_dim = self.dim_ode
            output_hill_N = 2  # parameter of output function

            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj):
                assert state_traj.shape[1] == local_dim
                x_vec = state_traj[:, 1]
                return input_traj * 1 / (1 + x_vec ** output_hill_N)

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = u / (1 + x_2^N);  (N=%d)$' % output_hill_N


        elif self.ode_name == 'ode_circuit_diode':
            self.ode_fn = ode_circuit_diode
            self.dim_ode = 1
            self.dim_params = 1
            self.dim_control = 1
            self.ode_short = [r"$x$"]
            self.params_short = [
                r'$RC$']
            if self.params is None:
                # habituate with square pulses:                        1, 0.1, 1, 100
                # habituate with staircase input (integral of pulses): 1, 1,   2,  2
                # TODO for [1, 0.1, 1, 10], and S1 input with duty 0.01, (dirac amp.),
                #  the ODE seems unstable, and violates our "weak habituation" -- seems to habituate normally
                self.params = np.array([
                    2.209,  # C = 4700 uF ; R_3 = 470 Ohms
                ])

            # define output function
            local_dim = self.dim_ode
            output_ad = 4.381 * 10 ** (-3)
            output_bd = -2.765 * 10 ** (-3)
            output_c = 0.2115
            output_R4 = 1e5
            output_factor = output_ad * output_R4 / (1 + output_ad * output_R4)

            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj):
                assert state_traj.shape[1] == local_dim
                u_vec = input_traj
                x_vec = state_traj[:, 0]

                y_vec = output_factor * (output_c * u_vec - x_vec - output_bd * output_R4 * (1 - 1/output_factor))
                #y_vec = np.where(Q > 0, Q, 0)

                # TODO delete
                y_out = np.zeros((len(y_vec), 2))
                y_out[:, 0] = y_vec
                y_out[:, 1] = np.where(y_vec > 0, y_vec, 0)

                # Select one of the outputs stored above to return as a 1d array (either Relu(z) or z)
                return y_out[:, 1]

            self.output_fn = ode_output
            self.output_str = r'$y_{out};  (a=%.1e, b=%.1e, R_4=%.1e)$' % (output_ad, output_bd, output_R4)

        elif self.ode_name == 'ode_hallmark5':
            self.ode_fn = ode_hallmark5
            self.dim_ode = 1
            self.dim_params = 2
            self.dim_control = 1
            self.ode_short = [r'$x$']
            self.params_short = [r'$\alpha$', r'$\beta$']
            if self.params is None:
                self.params = np.array([
                    0.2,   # alpha: timescale for x decay -- prop to x(t)
                    4,   # beta: timescale for x growth -- prop to u(t)
                ])

            # custom output function
            local_dim = self.dim_ode

            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj, hill_N=2):
                assert state_traj.shape[1] == local_dim
                x = state_traj[:, 0]
                u = input_traj[:]
                return np.tanh(100 * u) / (1 + x ** hill_N)

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = \mathrm{tanh}(\gamma u) / (1 + x^N)$'

        elif self.ode_name == 'ode_hallmark5_lifted':
            self.ode_fn = ode_hallmark5_lifted
            self.dim_ode = 2
            self.dim_params = 4
            self.dim_control = 1
            self.ode_short = [r'$x_1$', r'$x_2$']
            self.params_short = [r'$\alpha$', r'$\beta$', r'$N$', r'$\epsilon$']
            if self.params is None:
                self.params = np.array([
                    0.2,   # alpha: timescale for x decay -- prop to x(t)
                    4,   # beta: timescale for x growth -- prop to u(t)
                    2,     # N: hill function, filter saturation
                    1e-2,  # epsilon: timescale for output target synchronization (should be fastest timescale)
                ])

            # custom output function
            local_dim = self.dim_ode

        elif self.ode_name == 'ode_hallmark8':
            self.ode_fn = ode_hallmark8
            self.dim_ode = 1
            self.dim_params = 3
            self.dim_control = 2
            self.ode_short = [r'$x$']
            self.params_short = [r'$\alpha$', r'$\beta$', r'$\kappa$']
            if self.params is None:
                self.params = np.array([
                    0.2,  # alpha: timescale for x decay -- prop to x(t)
                    4,    # beta: timescale for x growth -- prop to u(t)
                    100,  # kappa: timescale for re-sensitization via second stimulus
                ])

            # custom output function
            local_dim = self.dim_ode

            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj, hill_N=2):
                assert state_traj.shape[1] == local_dim
                x = state_traj[:, 0]
                u = input_traj[:, 0]  # note input is not 1d for this system
                return u / (1 + x ** hill_N)

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = u / (1 + x^N)$'

        else:
            print("TODO - implement in class_ode_base.py init()", self.ode_name)
            raise AssertionError("%s not in VALID_FN_ODE or not implemented in ODEbase class init()" % self.ode_name)
        assert self.ode_name == self.ode_fn.__name__  # a convention, and bug avoidance
        # convenience block
        if self.ode_long is None:
            self.ode_long = self.ode_short
        if self.params_long is None:
            self.params_long = self.params_short
        if self.output_fn is None:
            local_dim_ode = self.dim_ode
            @numba.jit(nopython=True)
            def ode_output(state_traj, input_traj):
                assert state_traj.shape[1] == local_dim_ode
                return state_traj[:, -1]  # the timeseries of x_n (last component) is treated as the output

            self.output_fn = ode_output
            self.output_str = r'$y_{out} = x_{%d}(t)$' % self.dim_ode

        # this builds the main function which will be called in higher level classes (e.g. ODEModel)
        #   alternatively, ODEModel can call self.ode_prepped()
        self.fn_jit = self.fn_create()  # warning -- buggy, use self.fn_prepped instead

        # init asserts
        assert np.all([self.jac is None,
                       isinstance(self.params, np.ndarray),
                       len(self.control_objects) == self.dim_control,
                       len(self.ode_short) == self.dim_ode,
                       len(self.ode_short) == len(self.ode_long),
                       len(self.params_short) == self.dim_params,
                       len(self.params_short) == len(self.params_long)
                       ])

    def fn_prepped(self, t, x):
        uvals = np.array([u.fn_prepped(t) for u in self.control_objects])
        xdot = self.ode_fn(t, x, self.params, uvals)
        return xdot

    def fn_create(self):
        jitted_ode_fn = self.ode_fn
        p = self.params
        k = self.dim_control
        assert k <= 3
        # need to manually extend the if else below to go to arbitrary k because of numba errors for general case
        """
        NOTE: this is all a workaround for the event that this more general line fails to compile:
            uvals = np.array([uj(t) for uj in uj_funcs])  # this line fails
        """

        if k == 1:
            uj_func_0 = self.control_objects[0].fn_jit

            @numba.jit(nopython=True)
            def bar(t, x):
                uvals = np.array([uj_func_0(t)])    # THIS WORKS
                xdot = jitted_ode_fn(t, x, p, uvals)
                return xdot

        elif k == 2:
            uj_func_0 = self.control_objects[0].fn_jit
            uj_func_1 = self.control_objects[1].fn_jit

            @numba.jit(nopython=True)
            def bar(t, x):
                uvals = np.array([uj_func_0(t),
                                  uj_func_1(t)])  # THIS WORKS
                xdot = jitted_ode_fn(t, x, p, uvals)
                return xdot

        else:
            assert k == 3
            uj_func_0 = self.control_objects[0].fn_jit
            uj_func_1 = self.control_objects[1].fn_jit
            uj_func_2 = self.control_objects[2].fn_jit

            @numba.jit(nopython=True)
            def bar(t, x):
                uvals = np.array([uj_func_0(t),
                                  uj_func_1(t),
                                  uj_func_2(t)])  # THIS WORKS
                xdot = jitted_ode_fn(t, x, p, uvals)
                return xdot

        return bar
