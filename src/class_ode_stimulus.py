import copy
import numpy as np
from numba import jit

from defined_stimulus import *


class ODEStimulus():
    """
    control_fn: returns a scalar given a TIME and PARAMETERS
        e.g. a DiracComb -- u(t; N, T) = \sum_{k=0}^N delta(t - k T)
    """
    def __init__(
            self,
            stim_fn,
            params,
            label='ode_stimulus_unlabelled'
            ):
        self.stim_fn = stim_fn
        self.stim_fn_name = stim_fn.__name__
        self.params = params
        self.params_tuple = tuple(params)
        self.dim_params = len(params)
        self.label = label
        self.fn_jit = self.fn_create()  # returns uval given t; parameters theta of u(t, theta) are baked in

        # (2) regulate the different implemented "S(t, *args)" stimulus functions
        assert self.stim_fn_name in VALID_FN_STIMULUS
        self.period_forcing = None    # imputed below
        self.max_step_augment = None  # imputed below

        # asserts specific to particular functions
        MAX_STEP_AUGMENT_DIVISOR = 5.0  # originally: 10.0; this has strong affect on trajectory speed (wall-time)
        if self.stim_fn_name in ['stimulus_pulsewave', 'stimulus_pulsewave_of_pulsewaves']:
            assert 0 <= self.params[1] <= 1  # "duty" parameter must be in [0,1]
            duty_S1 = self.params[1]
            duty_duration = self.params[2] * duty_S1
            if duty_S1 < 0.5:
                self.max_step_augment = duty_duration / MAX_STEP_AUGMENT_DIVISOR
            else:
                self.max_step_augment = duty_duration * (1 - duty_S1) / MAX_STEP_AUGMENT_DIVISOR

        if self.stim_fn_name == 'stimulus_pulsewave':
            self.period_forcing = self.params[2]
            self.params_short = [r'$A$', r'$d$', r'$T$']
            self.params_long = ['amp', 'duty', 'period']

        elif self.stim_fn_name == 'stimulus_pulsewave_of_pulsewaves':
            self.period_forcing = self.params[2] * (self.params[3] + self.params[4])  # T_S1 * (n_on + n_off)
            self.params_short = [r'$A$', r'$d$', r'$T$', r'$n_p$', r'$n_r$']
            self.params_long = ['amp', 'duty', 'period', 'npulses', 'npulses_recover']

        elif self.stim_fn_name == 'stimulus_staircase':
            self.params_short = [r'$h$', r'$T_h$']
            self.params_long = ['step', 'step_duration']
            short_timescale = self.params[1]
            self.max_step_augment = short_timescale / MAX_STEP_AUGMENT_DIVISOR

        elif self.stim_fn_name == 'stimulus_rectangle':
            self.params_short = [r'$A$', r'$t_0$', r'$T$']
            self.params_long = ['amp', 'start', 'duration']
            short_timescale = self.params[2]
            self.max_step_augment = short_timescale / MAX_STEP_AUGMENT_DIVISOR

        else:
            assert self.stim_fn_name == 'stimulus_constant'
            self.params_short = [r'$A$']
            self.params_long = ['amplitude']

        assert len(self.params_short) == self.dim_params
        assert len(self.params_short) == len(self.params_long)

    # deprecated in favor of attribute: self.fn_jit(t, x)
    def fn_prepped(self, t):
        uval = self.stim_fn(t, *self.params)
        return uval

    def fn_create(self):
        p = self.params_tuple
        jitted_fn = self.stim_fn

        @jit(nopython=True)
        def foo(t):
            uval = jitted_fn(t, *p)  # note *args only works with tuples apparently?
            return uval

        return foo

    def printer(self):
        print('ODEStimulus.printer() method')
        print('\tstim_fn:', self.stim_fn)
        print('\tlabel:', self.label)
        print('\tparams_long:', self.params_long)
        print('\tparams:', self.params)
        print('\tparams_tuple:', self.params_tuple)
        print('\tdim_params:', self.dim_params)
        print('\tperiod_forcing:', self.period_forcing)
        print('\tmax_step_augment:', self.max_step_augment)


if __name__ == '__main__':
    print("main")

    #stimulus_choice = stimulus_constant
    #p = [1.0]

    stimulus_choice = stimulus_pulsewave
    p = [1.0, 0.1, 1.0]

    ode_stimulus_instance = ODEStimulus(
        stimulus_choice,
        p,
    )

    import time
    nn = 10000

    ode_stimulus_instance.fn_prepped(0)
    ode_stimulus_instance.fn_jit(0)

    start = time.time()
    for i in range(nn):
        uval = ode_stimulus_instance.fn_prepped(i)
    end = time.time()
    print('time class.fn_prepped =', uval, 'time', end-start)

    start = time.time()
    for i in range(nn):
        uval = ode_stimulus_instance.fn_jit(i)
    end = time.time()
    print('time class.fn_jit (fn jit factory approach) =', uval, 'time', end - start)

    if stimulus_choice == stimulus_constant:

        def stimulus_constant_loose(t, amp):
            s = np.ones_like(t) * amp
            return s


        @jit(nopython=True)
        def stimulus_constant_jit(t, amp):
            s = np.ones_like(t) * amp
            return s

        stimulus_constant_loose(0, p[0])
        stimulus_constant_jit(0, p[0])

        start = time.time()
        for i in range(nn):
            uval = stimulus_constant_loose(i, p[0])
        end = time.time()
        print('time loose fn =', end - start)

        start = time.time()
        for i in range(nn):
            uval = stimulus_constant_jit(i, p[0])
        end = time.time()
        print('time jit fn =', end - start)
