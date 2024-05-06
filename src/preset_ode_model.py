import numpy as np

from class_ode_base import ODEBase, VALID_FN_ODE
from class_ode_stimulus import ODEStimulus
from defined_ode_fn import *
from defined_stimulus import stimulus_constant, stimulus_pulsewave, stimulus_pulsewave_of_pulsewaves, stimulus_staircase, \
    delta_fn_amp
from preset_solver import PRESET_SOLVER


# === Stimulus S(t) ====================================================================================================
S1_duty = 0.1
S1_period = 1.0
S1_area = 1.0  # (amplitude * width)

S0_zero = ODEStimulus(
    stimulus_constant,
    [0.0],
    label=r'$u(t) = 0.0$'
)

S0_one = ODEStimulus(
    stimulus_constant,
    [1.0],
    label=r'$u(t) = 1.0$'
)

S_staircase = ODEStimulus(
    stimulus_staircase,
    [1.0, 4.0],  # step, duration
    label=r'$u(t)$ = staircase'
)

S1 = ODEStimulus(
    stimulus_pulsewave,
    [S1_area * delta_fn_amp(S1_duty, S1_period),  # amplitude
     S1_duty,    # duty in [0,1]
     S1_period,  # period of stimulus pulses (level 1)
    ],
    label=r'$u(t)$ = pulsewave'
)

S2 = ODEStimulus(
    stimulus_pulsewave_of_pulsewaves,
    [S1_area * delta_fn_amp(S1_duty, S1_period),  # amplitude
     S1_duty,  # duty in [0,1]
     S1_period,  # period of stimulus pulses (level 1)
     20,   # npulses
     10,   # npulses_recover
    ],
    label=r'$u(t)$ = pulsewave of pulsewaves'
)


# === Compose S(t) + ODE params + misc. params into ODE model preset ===================================================

def ode_model_preset_factory(preset_label, ode_name, control_objects, t0, tmax, ode_params=None, init_cond=None):
    """
    Builds a dict of args (tuple) and kwargs (dict) for the class ODEModel
    args:
        ode_base
        init_cond
        t0
        tmax
    kwargs:
        solver_settings=PRESET_SOLVER['solve_ivp_radau_default'],
        history_times=None,
        history_state=None,
        label='',
        outdir=DIR_OUTPUT):

    Relies on the predefined dictionary ODE_CATALOG in defined_ode_fn.py (keys reflect the ODE fn names)
    """
    ode_base = ODEBase(
        ode_name,
        control_objects,
        params=ode_params,
    )
    if init_cond is None:
        init_cond = np.zeros(ode_base.dim_ode)
    preset_instance = dict(
        args=[ode_base, init_cond, t0, tmax],
        kwargs=dict(
            solver_settings=PRESET_SOLVER['solve_ivp_radau_default'],  # i.e. strictest
            #solver_settings=PRESET_SOLVER['solve_ivp_radau_medium'],
            #solver_settings=PRESET_SOLVER['solve_ivp_rk45'],
            #solver_settings=PRESET_SOLVER['solve_ivp_DOP853'],
            #solver_settings=PRESET_SOLVER['solve_ivp_BDF'],
            #solver_settings=PRESET_SOLVER['solve_ivp_LSODA'],
            label=preset_label)
    )
    return preset_instance


def build_several_presets(preset_dict, ode_name):
    preset_dict['%s_S0_zero' % ode_name] = ode_model_preset_factory(
        'preset_%s_S0_zero' % ode_name,
        ode_name, [S0_zero], 0.0, 100.5)
    preset_dict['%s_S0' % ode_name] = ode_model_preset_factory(
        'preset_%s_S0' % ode_name,
        ode_name, [S0_one], 0.0, 100.5)
    preset_dict['%s_S1' % ode_name] = ode_model_preset_factory(
        'preset_%s_S1' % ode_name,
        ode_name, [S1], 0.0, 50.5)
    preset_dict['%s_S2' % ode_name] = ode_model_preset_factory(
        'preset_%s_S2' % ode_name,
        ode_name, [S2], 0.0, 200.5)
    preset_dict['%s_S_staircase' % ode_name] = ode_model_preset_factory(
        'preset_%s_S_staircase' % ode_name,
        ode_name, [S_staircase], 0.0, 100.5)
    return preset_dict


PRESETS_ODE_MODEL = dict()

print('Building ode model presets...')
for ode_name in VALID_FN_ODE:
    if ode_name not in ['ode_hallmark8']:  # these models require > 1 control objects
        PRESETS_ODE_MODEL = build_several_presets(PRESETS_ODE_MODEL, ode_name)
print('\t...done')
