import numpy as np
from jax import config
config.update("jax_enable_x64", True)

import lineax as lx
from jax import vmap, numpy as jnp, jit
from functools import partial
import diffrax
import equinox as eqx

def get_analytical_2d(diff_coeff, t, v):
    return 1/(4*np.pi*diff_coeff*t)*np.exp(-(v[:, None]**2.+v[None, :]**2.)/4/diff_coeff/t)

@jit
def solve_diff_2d(dt, finp, diff_coeff):
    coeff = -0.5*dt * diff_coeff / dv**2.
    diag = 1-jnp.concatenate([jnp.ones([1]), 2*jnp.ones_like(v[1:-1]), jnp.ones([1])])*coeff
    lower_diag = np.ones_like(v[1:])*coeff
    upper_diag = np.ones_like(v[1:])*coeff
    operator = lx.TridiagonalLinearOperator(diag, lower_diag, upper_diag)
    
    interm = finp - coeff*jnp.gradient(jnp.gradient(finp, axis=1), axis=1)
    interm = linear_solve_x(operator, interm).value
    interm = interm - coeff*jnp.gradient(jnp.gradient(interm, axis=0), axis=0)
    
    out = linear_solve_y(operator, interm).value
    
    return out

class VectorField(eqx.Module):
    """
    This function returns the function that defines $d_state / dt$

    All the pushers are chosen and initialized here and a single time-step is defined here.

    We use the time-integrators provided by diffrax, and therefore, only need $d_state / dt$ here

    :param cfg:
    :return:
    """
    v: jnp.ndarray
    dt: float
    kappa: float
    
    def __init__(self, v: jnp.ndarray, dt: float, kappa: float):
        super().__init__()
        self.v = v
        self.dt = dt
        self.kappa = kappa

    def __call__(self, t: float, y: jnp.ndarray, args):
        return solve_diff_2d(self.dt, y, self.kappa)
        