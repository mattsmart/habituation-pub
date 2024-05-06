import copy
#import jax
#import jax.numpy as jnp
import numpy as np
from scipy.optimize import check_grad, approx_fprime
from numba import jit

from defined_stimulus import stimulus_pulsewave

"""
The functions below define the "right hand side" of the ODE classes under consideration
Their inputs adhere to the following format:
- t: time (scalar or array)
- x: state at which to compute dx/dt
- p: list of ODE parameters
- u: array of control function values prepared elsewhere

The non-autonomous ODEs considered may have multiple types of time-dependent behaviour aka "Stimulus"

Below, two globals are defined
"""


@jit(nopython=True)
def ode_iff_1(t, x, p, u):
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx = sval * p[0] * (1 - x[0]) -        p[1] * x[0]
    dy = x[0] * p[2] * (1 - x[1]) -        p[3] * x[1] / (x[1] + p[6])
    dz = x[0] * p[4] * (1 - x[2]) - x[1] * p[5] * x[2] / (x[2] + p[7])
    return dx, dy, dz


@jit(nopython=True)
def jac_ode_iff_1(t, x, p, u):
    """
    Manual jacobian of the vector field: ode_iff_1
    Note: tested using scipy.optimize.approx_fprime
    """
    # compute stimulus over time t
    sval = u[0]
    # compute non-zero jacobian elements
    j11 = -sval * p[0] - p[1]
    j21 = p[2] * (1 - x[1])
    j31 = p[4] * (1 - x[2])
    j22 = - x[0] * p[2]        - p[3] * p[6] / (x[1] + p[6]) ** 2
    j33 = - x[0] * p[4] - x[1] * p[5] * p[7] / (x[2] + p[7]) ** 2
    j32 =                      - p[5] * x[2] / (x[2] + p[7])
    jac = np.array([
        [j11, 0.,  0.],
        [j21, j22, 0.],
        [j31, j32, j33]
    ])
    return jac


@jit(nopython=True)
def ode_iff_1_6d(t, x, p, u):
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx = sval * p[0] * (1 - x[0]) -        p[1] * x[0]
    dy = x[0] * p[2] * (1 - x[1]) -        p[3] * x[1] / (x[1] + p[6])
    dz = x[0] * p[4] * (1 - x[2]) - x[1] * p[5] * x[2] / (x[2] + p[7])

    # now use p8 to p15
    dx2 = x[2] * p[8] * (1 - x[3]) - p[9] * x[3]
    dy2 = x[3] * p[10] * (1 - x[4]) - p[11] * x[4] / (x[4] + p[14])
    dz2 = x[3] * p[12] * (1 - x[5]) - x[4] * p[13] * x[5] / (x[5] + p[15])

    return dx, dy, dz, dx2, dy2, dz2


@jit(nopython=True)
def ode_custom_1(t, x, p, u):
    """
    Found heuristically by simplifying ODE IFF 1 above
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx = p[0] * sval - p[1] * x[0]
    dy = p[2] * x[0] - p[3] * x[1]
    dz = p[4] * x[0] - p[5] * x[2] * x[1]
    return dx, dy, dz


@jit(nopython=True)
def ode_custom_1_6d(t, x, p, u):
    """
    Found heuristically by simplifying ODE IFF 1 above
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx = p[0] * sval - p[1] * x[0]
    dy = p[2] * x[0] - p[3] * x[1]
    dz = p[4] * x[0] - p[5] * x[2] * x[1]

    # now use p6 to p11
    dx2 = p[6]  * x[2] -  p[7] * x[3]
    dy2 = p[8]  * x[3] -  p[9] * x[4]
    dz2 = p[10] * x[3] - p[11] * x[5] * x[4]

    return dx, dy, dz, dx2, dy2, dz2


@jit(nopython=True)
def ode_custom_2(t, x, p, u):
    """
    Adapted from Fig. 1D, Box 1 of:
        Sniffers, buzzers, toggles and blinkers: dynamics of regulatory
        and signaling pathways in the cell
        John J Tyson, Katherine C Chen and Bela Novak
        2003
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx = p[0] * sval - p[1] * x[0]
    dy = p[2] * sval - p[3] * x[1] * x[0]
    return dx, dy


@jit(nopython=True)
def ode_custom_2_integrator(t, x, p, u):
    """
    Third variable is added to ode_custom_2
    - this variable serves to integrate the stimulus
    - this allows the original system to perform habituation in a different parameter regime than it otherwise would
    Note:
    - currently these integrator variants break the "limit cycle detection" because integral of S(t) grows indefinitely
    - fix the above by omitting/"masking" S(t) during limit cycle calculation... treat it as a "dummy variable" (in ode class?)
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dS = sval
    dx = p[0] * x[0] - p[1] * x[1]
    dy = p[2] * x[0] - p[3] * x[2] * x[1]
    return dS, dx, dy


@jit(nopython=True)
def ode_custom_3(t, x, p, u):
    """
    "State-dependent inactivation" model variant
    From Fig. 4D of
        "Perfect and Near-Perfect Adaptation in Cell Signaling (Ferrell, 2016)"
    Intended for use with "staircase input" rather than rectangular pulse-waves
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx0 = p[1] * x[1]

    b_off = sval - x[0] - x[1]
    x_off = p[2] - x[0] - x[1]
    dx1 = p[0] * b_off * x_off - p[1] * x[1]
    """
    print(t, x, dx0, dx1)
    """
    return dx0, dx1


@jit(nopython=True)
def ode_custom_3_integrator(t, x, p, u):
    """
    ode_custom_3 with extra dimension corresponding to integration of the input stimulus
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx0 = sval
    dx1 = p[1] * x[2]

    b_off = x[0] - x[1] - x[2]
    x_off = p[2] - x[1] - x[2]
    dx2 = p[0] * b_off * x_off - p[1] * x[2]
    """
    print(t, x, dx0, dx1, dx2)
    """
    return dx0, dx1, dx2


@jit(nopython=True)
def ode_custom_3_integrator_TESTING(t, x, p, u):
    """
    ode_custom_3 with extra dimension corresponding to integration of the input stimulus
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx0 = sval - p[3] * x[0]
    dx1 = p[1] * x[2]                        #- p[3] * x[1]

    b_off = x[0] - x[1] - x[2]
    x_off = p[2] - x[1] - x[2]
    dx2 = p[0] * b_off * x_off - p[1] * x[2] #+ p[3] * x[1]
    """
    print(t, x, dx0, dx1, dx2)
    """
    #print(dx1, p[1] * x[2], p[1], x[2])
    return dx0, dx1, dx2


@jit(nopython=True)
def ode_custom_3_simple(t, x, p, u):
    """
    "State-dependent inactivation" model
    From Fig. 4A of
        "Perfect and Near-Perfect Adaptation in Cell Signaling (Ferrell, 2016)"
    Intended for use with "staircase input" rather than rectangular pulse-waves
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx0 = p[1] * x[1]
    dx1 = p[0] * sval * (p[2] - x[0] - x[1]) - p[1] * x[1]
    return dx0, dx1


@jit(nopython=True)
def ode_custom_3_simple_integrator(t, x, p, u):
    """
    ode_custom_3_simple with extra dimension corresponding to integration of the input stimulus
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx0 = sval
    dx1 = p[1] * x[2]
    dx2 = p[0] * x[0] * (p[2] - x[1] - x[2]) - p[1] * x[2]
    return dx0, dx1, dx2


@jit(nopython=True)
def ode_custom_4(t, x, p, u):
    """
    Martin email April 19, 2023
        T_1 d/dt(x1) + x1 = K_1 u
        T_2 d/dt(x2) + x2 = K_2 u
    becomes
        d/dt(x1) = (K_1/T_1) u - (1/T_1) x1
        d/dt(x2) = (K_2/T_2) u - (1/T_2) x2
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx0 = p[0] * sval - p[1] * x[0]
    dx1 = p[2] * sval - p[3] * x[1]
    return dx0, dx1


@jit(nopython=True)
def ode_custom_4_innerout(t, x, p, u):
    """
    Extension of system in ode_custom_4
    - There the output function used is nonlinear: y0(t) = (x1_high - x1) * x2
    - Here we add extra variable to 'linearize' the new output: y(t) = x3
    - 'Singular perturbation' trick adds two new variables to parameter vector (6 total, from 4 total before)
        d/dt(x3) = (1 / eps) * y0(t) - x3
        d/dt(x3) = (1 / eps) * [(x1_high - x1) * x2 - x3]
    - call p[4] = 1/eps; default: 1e6 -- i.e. eps small, so that d/dt(x3) RHS is big -- unless it matches desired output!
    - call p[5] = x1_high; functional form depends on stimulus period, defined within ODEBase class init()
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx0 = p[0] * sval - p[1] * x[0]
    dx1 = p[2] * sval - p[3] * x[1]
    dx2 = p[6] * ((p[4] * p[5] - x[0]) * x[1] - x[2])
    #dx3 = p[6] * ((p[4] - x[0]) * x[1] - x[2])
    return dx0, dx1, dx2


@jit(nopython=True)
def ode_custom_5(t, x, p, u):
    """
    Negative feedback loop A (inspired by Fig. 2 of Eckert 2022)
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx = p[0] * sval - p[1] * x[0]
    dy = p[2] * x[2] - p[3] * x[1]
    dz = p[4] * x[0] - p[5] * x[2] * x[1]
    return dx, dy, dz


@jit(nopython=True)
def ode_custom_6(t, x, p, u):
    """
    From Fig. 2B of
        "Perfect and Near-Perfect Adaptation in Cell Signaling (Ferrell, 2016)"
    Intended for use with "staircase input" rather than rectangular pulse-waves
    Also, they use x1 as the "perfect adaptation" output, not x2
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx = p[0] * sval * (1-x[0]) - p[1] * x[0] * x[1]
    dy = p[2] * x[0] * (1-x[1]) / (p[4] + 1 - x[1]) - p[3] * x[1] / (p[5] + x[1])
    return dx, dy


@jit(nopython=True)
def ode_custom_6_simplified(t, x, p, u):
    """
    Replaced the rational factors in ode_custom_6 with linear factors (as done for the Eckert 2022 system)
    Note: seems numerically unstable near regime where it habituates...
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    #dx = p[0] * sval * (1-x[0]) - p[1] * x[0] * x[1]
    #dy = p[2] * x[0] * (1-x[1])  - p[3] * x[1]
    dx = p[0] * sval - p[1] * x[0] * x[1]
    dy = p[2] * x[0] - p[3] * x[1]
    return dx, dy


@jit(nopython=True)
def ode_custom_7(t, x, p, u):
    """
    Adapting circuit in Fig. 1 of
        A universal biomolecular integral feedback controller for robust perfect adaptation
        See SI Eq. (S1)
            x0 and x1 are the control circuit; x2 is the output
    Relevant (simpler version of the following):
        Fig. 5 (Antithetical Integral Feedback) of
        Perfect and Near-Perfect Adaptation in Cell Signaling
        James E. Ferrell, Jr
    Where to place the input? It needs to act not on x0, x1. Fig. 5 of Ferrell 2016 suggests in x2 eqn production term
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    # p = a1, a2, k_0, a3, b3
    dx0 = p[0]               - p[2] * x[0] * x[1]
    dx1 = p[1] * x[2]        - p[2] * x[0] * x[1]
    dx2 = p[3] * sval * x[0] - p[4] * x[2]              # perturbation (input) needs to be here, somewhere
    return dx0, dx1, dx2


@jit(nopython=True)
def ode_custom_7_alt(t, x, p, u):
    """
    Fig. 5 (Antithetical Integral Feedback) of
        Perfect and Near-Perfect Adaptation in Cell Signaling
        James E. Ferrell, Jr
    Where to place the input? It needs to act not on x0, x1. Fig. 5 of Ferrell 2016 suggests in x2 eqn production term
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    # p = a1, a2, k_0, a3, b3; last two for new eqn prod/degr
    dx0 = p[0]               - p[2] * x[0] * x[1]
    dx1 = p[1] * x[3]        - p[2] * x[0] * x[1]
    dx2 = p[5] * x[0]        - p[6] * x[2]           # this eqn is basically inserted into 3D variant ode_custom_7 above
    dx3 = p[3] * sval * x[0] - p[4] * x[3]           # perturbation (input) needs to be here, somewhere, or in dx2
    return dx0, dx1, dx2, dx3


@jit(nopython=True)
def ode_custom_7_ferrell(t, x, p, u):
    """
    4th eqn of ode_custom_7_alt() modified to match Fig. 5 of Ferrell2016
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    # p = a1, a2, k_0, a3, b3; last two for new eqn prod/degr
    dx0 = p[0]               - p[2] * x[0] * x[1]    # "D"
    dx1 = p[1] * x[3]        - p[2] * x[0] * x[1]    # "C"
    dx2 = p[5] * x[0]        - p[6] * x[2]           # "A"
    dx3 = p[3] * sval * x[2] - p[4] * x[3]           # "B" - x[2] on RHS replaces x[0] in ode_custom_7_alt()
    return dx0, dx1, dx2, dx3

@jit(nopython=True)
def ode_quad_2d_A(t, x, p, u):
    '''
    H1 -> Yes, H2 -> Yes
    '''
    # compute stimulus over time t
    sval = u[0]
    a0, b0, a1, b2, a12, b12 = p
    # ======================================================
    dx0 = a0 * sval  +  a1 * x[0]  +  a12 * x[0] * x[1]
    dx1 = b0 * sval  +  b2 * x[1]  +  b12 * x[0] * x[1]
    # ======================================================
    return dx0, dx1


@jit(nopython=True)
def ode_SatLin1D_NL(t, x, p, u):
    '''
    See 23/08/16 discussion
    '''
    # compute stimulus over time t
    sval = u[0]
    a0, b0 = p
    # ======================================================
    dx0 = a0 * (1 - x[0]) * sval -  b0 * x[0]
    # ======================================================
    return (dx0,)


@jit(nopython=True)
def ode_Wang93_1d(t, x, p, u):
    """
    See Eq. (1) of
        A Neural Model of Synaptic Plasticity Underlying Short-term and Long-term Habituation
        DeLiang Wang, 1994, MIT
    """
    # compute stimulus over time t
    sval = u[0]
    y = x[0]
    tau, alpha, y0 = p
    # ======================================================
    dy = 1/tau * (alpha * (y0 - y) - sval)
    # ======================================================
    return dy,


@jit(nopython=True)
def ode_Wang93_2d(t, x, p, u):
    """
    See Eq. (3) of
        A Neural Model of Synaptic Plasticity Underlying Short-term and Long-term Habituation
        DeLiang Wang, 1994, MIT
    - y(t) - the "synaptic efficacy"; when u(t)=0, it recovers to y0
    - z(t) - regulates "the rate of recovery of y(t) --> y0"
    Note dz/dt is independent of y(t):
    - for constant input and z(t0) = 0.5, get sigmoid with midpoint at t0
        z(t) = 1 / (1 + exp(\gamma * (t - t0))
    """
    # compute stimulus over time t
    sval = u[0]
    y, z = x[0], x[1]
    tau, alpha, beta, gamma, y0 = p
    # ======================================================
    dy = 1/tau * (alpha * z * (y0 - y) - beta * y * sval)
    dz =          gamma * z * (z - 1) * sval
    #dz =          gamma * z * (1 - z) * sval
    # ======================================================
    return dy, dz


@jit(nopython=True)
def ode_SatLin2D_NL(t, x, p, u):
    """
    "Saturating" version of Linear_2D_NL
    """
    # compute stimulus over time t
    sval = u[0]
    x1, x2 = x[0], x[1]
    tau_1, tau_2, a1, a2 = p
    # ======================================================
    dx1 = 1/tau_1 * (a1 * sval * (1 - x1) - x1)
    dx2 = 1/tau_2 * (a2 * sval            - x2)
    # ======================================================
    return dx1, dx2


@jit(nopython=True)
def ode_SatLin3D_NL(t, x, p, u):
    """
    3D extension of ode_SatLin2D_NL
    Idea is that additional timescales (state variables) let us construct outputs y(t) which satisfy extra hallmarks
    """
    # compute stimulus over time t
    sval = u[0]
    x1, x2, x3 = x[0], x[1], x[2]
    tau_1, tau_2, tau_3, a1, a2, a3 = p
    # ======================================================
    dx1 = 1/tau_1 * (a1 * sval * (1 - x1) - x1)
    dx2 = 1/tau_2 * (a2 * sval            - x2)
    dx3 = 1/tau_3 * (a3 * sval * (1 - x3) - x3)
    # ======================================================
    return dx1, dx2, dx3


@jit(nopython=True)
def ode_FHN_helper_g(x, a=0.25, m=2):
    """
    a, m are "shape parameters" for the piece-wise-linear function g(x) which mimics a shifted cubic function
    For positive invariance of positive quadrant (x, y), we need (m-1)/(m+1) > a; need m > 1
    """
    g1 = np.where(x < 0.5,
                  m * x,
                  0)
    g2 = np.where((0.5 <= x) & (x < ((a + 1)/2)),
                  m * (-x + 1),
                  0)
    g3 = np.where(x >= ((a + 1)/2),
                  m * (-a + x),
                  0)
    g = g1 + g2 + g3
    return g


@jit(nopython=True)
def ode_FHN(t, x, p, u):
    """
    Analog of relaxation oscillator used in growing network model
    """
    # compute stimulus over time t
    sval = u[0]
    x0, y0 = x[0], x[1]
    epsilon, gamma, alpha = p

    z = alpha * sval
    g_of_x = ode_FHN_helper_g(x0)

    # ======================================================
    dx = 1/epsilon * (y0 - g_of_x)
    dy = z - x0 - gamma * y0
    # ======================================================
    return dx, dy


@jit(nopython=True)
def ode_reverse_engineer_A(t, x, p, u):
    """
    Implement something like
        output z(t) = u(t) * (1 - tanh( W(t) )
    where
        W(t) is a memory of the previous inputs
    Simple example:
        dW/dt = a (u(t) - W(t)) for W(0)=0, this is just a linear, 1-dim ODE plus a nonlinear output z = g(x, u)
    """
    # compute stimulus over time t
    sval = u[0]
    x0 = x[0]
    a = p[0]

    dx = a * (sval - x0)

    # ======================================================
    return (dx,)


@jit(nopython=True)
def ode_linear_filter_1d(t, x, p, u):
    """
    Implement something like
        output y(t) = u(t) * g(t) where g(t) is a saturating function of x, i.e. \sigma(x)
        here we use
            \sigma(x) = 1 / (1 + x^N) where N is fixed in the model class output function
    where
        x(t) acts as a memory of the previous inputs

    Staddon, Higa 1996 Fig. 1:  Feedforward, 3 units
        a_1=0.8, a_2=0.95, a_3 =0.99; b_k=0.2 for all
        ISI=2 and ISI=8 mean in our language T=3 and T=9 ("units" of time)
        Can use a timestep of Delta T = 1.0 (at most)
    Our mapping onto those discrete time units with timestep "t + dt" | x_{t+1} = a x_t + b u_t
        a = 1 - exp(- alpha * dt)
        b = beta / alpha * (1 - a)
    inverting gives
        alpha = - ln(a) / Delta T;
        beta = b * (1 - a) / alpha  (from above)
    Therefore, the following parameters should satisfy frequency sensitivity for T=3 and T=9 stimuli for pulsewidth dT=1
        alpha_1 = 0.2231
        alpha_2 = 0.05129
        alpha_3 = 0.01005
        beta_1 = 0.1793
        beta_2 = 0.1950
        beta_3 = 0.1990
    these assume a pulsewidth of 1, so a varying duty of 1/3 for T=3 and 1/9 for T=9
    """
    # compute stimulus over time t
    sval = u[0]
    x0 = x[0]
    alpha, beta = p

    dx = beta * sval - alpha * x0

    return (dx,)


@jit(nopython=True)
def ode_linear_filter_1d_lifted(t, x, p, u):
    """
    Lifted variant of ode_reverse_engineer_B (now has dim state 2, with 4 parameters)
    """
    # compute stimulus over time t
    sval = u[0]
    x0, y0 = x
    alpha, beta, N, epsilon = p

    dx = beta * sval - alpha * x0
    dy = 1/(epsilon) * ( sval / (1 + x0 ** N) - y0 )

    return (dx, dy)


@jit(nopython=True)
def ode_linear_filter_2d(t, x, p, u):
    """
    Series (2 unit) version of ode_linear_filter_1d
    """
    # compute stimulus over time t
    sval = u[0]
    alpha_0, beta_0, N_0, alpha_1, beta_1 = p

    dx0 = beta_0 * sval - alpha_0 * x[0]
    y0 = sval / (1 + x[0] ** N_0)  # TODO mult by sval here or no? or make numerator 1.0 ?
    dx1 = beta_1 * y0 - alpha_1 * x[1]

    return dx0, dx1


@jit(nopython=True)
def ode_linear_filter_3d(t, x, p, u):
    """
    Series (3 unit) version of ode_linear_filter_1d
    """
    # compute stimulus over time t
    sval = u[0]
    alpha_0, beta_0, N_0, alpha_1, beta_1, N_1, alpha_2, beta_2 = p

    # sigma(z) - HILL
    ###y0 = sval / (1 + x[0] ** N_0)  # TODO mult by sval here or no? or make numerator 1.0 ?
    ###y1 = x[0] / (1 + x[1] ** N_1)  # TODO mult by sval or x0 here or no?

    # sigma(z) - RELU
    y0 = np.where(sval - x[0] > 0, sval - x[0], 0)
    y1 = np.where(y0 - x[1] > 0, y0 - x[1], 0)

    dx0 = beta_0 * sval - alpha_0 * x[0]
    dx1 = beta_1 * y0 - alpha_1 * x[1]
    dx2 = beta_2 * y1 - alpha_2 * x[2]

    return dx0, dx1, dx2


@jit(nopython=True)
def ode_linear_filter_4d(t, x, p, u):
    """
    Series (4 unit) version of ode_linear_filter_1d
    """
    # compute stimulus over time t
    sval = u[0]
    alpha_0, beta_0, N_0, alpha_1, beta_1, N_1, alpha_2, beta_2, N_2, alpha_3, beta_3 = p

    dx0 = beta_0 * sval - alpha_0 * x[0]
    y0 = sval / (1 + x[0] ** N_0)  # TODO mult by sval here or no? or make numerator 1.0 ?
    dx1 = beta_1 * y0 - alpha_1 * x[1]
    y1 = x[0] / (1 + x[1] ** N_1)  # TODO mult by sval or x0 here or no?
    dx2 = beta_2 * y1 - alpha_2 * x[2]
    y2 = x[1] / (1 + x[2] ** N_2)  # TODO mult by sval or x1 here or no?
    dx3 = beta_3 * y2 - alpha_3 * x[3]

    return dx0, dx1, dx2, dx3


@jit(nopython=True)
def ode_linear_filter_5d(t, x, p, u):
    """
    Series (5 unit) version of ode_linear_filter_1d
    """
    # compute stimulus over time t
    sval = u[0]
    alpha_0, beta_0, N_0, alpha_1, beta_1, N_1, alpha_2, beta_2, N_2, alpha_3, beta_3, N_3, alpha_4, beta_4 = p

    dx0 = beta_0 * sval - alpha_0 * x[0]

    #y0 = sval / (1 + x[0] ** N_0)    # TODO mult by sval here or no? or make numerator 1.0 ?
    #y0 = x[0]
    y0 = sval / (1 + x[0] ** N_0)
    dx1 = beta_1 * y0 - alpha_1 * x[1]

    #y1 = sval / (1 + x[1] ** N_1)  # TODO mult by sval or x0 here or no?
    #y1 = x[1]
    y1 = y0 / (1 + x[1] ** N_1)
    dx2 = beta_2 * y1 - alpha_2 * x[2]

    #y2 = sval / (1 + x[2] ** N_2)  # TODO mult by sval or x1 here or no?
    #y2 = x[2]
    y2 = y1 / (1 + x[2] ** N_2)
    dx3 = beta_3 * y2 - alpha_3 * x[3]

    #y3 = sval / (1 + x[3] ** N_3)  # TODO mult by sval or x2 here or no?
    #y3 = x[3]
    y3 = y2 / (1 + x[3] ** N_3)
    dx4 = beta_4 * y3 - alpha_4 * x[4]

    return dx0, dx1, dx2, dx3, dx4


@jit(nopython=True)
def ode_simplefilter_to_tyson(t, x, p, u):
    """
    Take simple filter model (linear x'=bu-ax) + (nonlinear output)
    And make the 'a' paramter itself a linear ODE (instead of constant. This resembles the "Tyson model".
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx = p[0] * sval - p[1] * (x[0] - p[2])
    dy = p[3] * sval - p[4] * x[1] * x[0]
    return dx, dy


@jit(nopython=True)
def ode_circuit_diode(t, x, p, u):
    """
    ODE version of martin circuit
    """
    # compute stimulus over time t
    sval = u[0]
    # compute vector field
    dx = (sval - x[0]) / p[0]
    return (dx,)


@jit(nopython=True)
def ode_hallmark5(t, x, p, u):
    """
    ODE version of martin circuit
    """
    # compute stimulus over time t
    sval = u[0]
    sval_prime = 2 * sval / (1 + sval ** 2)
    # compute vector field
    dx = p[1] * sval_prime - p[0] * x[0]
    return (dx,)


@jit(nopython=True)
def ode_hallmark5_lifted(t, x, p, u):
    """
    Eq. (5) of text for implementing hallmark 5 (amplitude sensitivity)
    """
    # compute stimulus over time t
    sval = u[0]
    sval_prime = 2 * sval / (1 + sval ** 2)
    # compute vector field
    dx = p[1] * sval_prime - p[0] * x[0]
    dy = 1/p[3] * (np.tanh(100 * sval) / (1 + x[0] ** p[2]) - x[1])
    return dx, dy


@jit(nopython=True)
def ode_hallmark8(t, x, p, u):
    """
    Eq. (5) of text for implementing hallmark 5 (amplitude sensitivity)
    """
    # compute stimulus over time t
    sval = u[0]
    sensitizer_val = u[1]  # note we now have two different stimuli (so two diff control objects)

    # compute vector field
    dx = p[1] * sval - p[0] * x[0] - p[2] * sensitizer_val * x[0]
    return dx,
