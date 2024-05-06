PRESET_SOLVER = {}

vectorized = False  # for the scipy integrators, can pass vectorization flag for faster jacobian construction

PRESET_SOLVER['solve_ivp_radau_default'] = dict(
    dynamics_method='scipy_solve_ivp',
    kwargs=dict(method='Radau', t_eval=None, atol=1e-12, rtol=1e-6, vectorized=vectorized),
)

PRESET_SOLVER['solve_ivp_radau_strictest'] = dict(
    dynamics_method='scipy_solve_ivp',
    kwargs=dict(method='Radau', t_eval=None, atol=1e-12, rtol=1e-9, vectorized=vectorized),
)

PRESET_SOLVER['solve_ivp_radau_strict'] = dict(
    dynamics_method='scipy_solve_ivp',
    kwargs=dict(method='Radau', t_eval=None, atol=1e-8, rtol=1e-4, vectorized=vectorized),
)

PRESET_SOLVER['solve_ivp_radau_medium'] = dict(
    dynamics_method='scipy_solve_ivp',
    kwargs=dict(method='Radau', t_eval=None, atol=1e-6, rtol=1e-4, vectorized=vectorized),
)

PRESET_SOLVER['solve_ivp_radau_relaxed'] = dict(
    dynamics_method='scipy_solve_ivp',
    kwargs=dict(method='Radau', t_eval=None, atol=1e-5, rtol=1e-2, vectorized=vectorized),
)

PRESET_SOLVER['solve_ivp_radau_minstep'] = dict(
    dynamics_method='scipy_solve_ivp',
    kwargs=dict(method='Radau', min_step=1e-1, vectorized=vectorized),
)

PRESET_SOLVER['solve_ivp_rk23'] = dict(
    dynamics_method='scipy_solve_ivp',
    kwargs=dict(method='RK23', vectorized=vectorized, atol=1e-12, rtol=1e-6),
)

PRESET_SOLVER['solve_ivp_rk45'] = dict(
    dynamics_method='scipy_solve_ivp',
    kwargs=dict(method='RK45', vectorized=vectorized, atol=1e-12, rtol=1e-6),
)

PRESET_SOLVER['solve_ivp_DOP853'] = dict(
    dynamics_method='scipy_solve_ivp',
    kwargs=dict(method='DOP853', vectorized=vectorized, atol=1e-12, rtol=1e-6),
)

PRESET_SOLVER['solve_ivp_BDF'] = dict(
    dynamics_method='scipy_solve_ivp',
    kwargs=dict(method='BDF', vectorized=vectorized, atol=1e-12, rtol=1e-6),
)

PRESET_SOLVER['solve_ivp_LSODA'] = dict(
    dynamics_method='scipy_solve_ivp',
    kwargs=dict(method='LSODA', vectorized=vectorized, atol=1e-12, rtol=1e-6),
)
